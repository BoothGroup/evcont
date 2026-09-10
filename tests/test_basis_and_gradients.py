import numpy as np
import pytest

from evcont.ab_initio_gradients_loewdin import (
    fix_gauge,
    get_grad_elec_from_gradH,
    get_overlap_grad,
)
from evcont.basis.basis_utils import (
    basis_requires_reference,
    get_basis,
    get_basis_with_derivative,
    get_loewdin_trafo,
    is_abstract_basis,
    normalize_basis_type,
    run_hf,
)
from evcont.basis.localization_derivatives import orth_ao_derivative
from evcont.basis.least_change_orbitals import _solve_rank_aware_krylov
from evcont.basis.split_procrustes_derivatives import (
    ProcrustesDerivativeError,
    procrustes_rotation,
    procrustes_rotation_derivative,
    rhf_mo_coefficient_derivatives,
)


@pytest.mark.parametrize(
    ("alias", "canonical", "needs_reference"),
    [
        ("SAO", "SAO", False),
        ("oao", "SAO", False),
        ("meta-lowdin", "meta_lowdin", False),
        ("canonical_split_procrustes_none", "split_procrustes", True),
        ("least-change", "least_change_atom_coordinate", True),
        ("least_change_reference_mo", "least_change_frozen_mo", True),
        ("least_change_pointwise", "least_change_local", False),
    ],
)
def test_basis_alias_contract(alias, canonical, needs_reference):
    assert normalize_basis_type(alias) == canonical
    assert is_abstract_basis(alias)
    assert basis_requires_reference(alias) is needs_reference


def test_unknown_basis_is_rejected(h2_molecule):
    assert not is_abstract_basis("not-a-basis")
    with pytest.raises(ValueError, match="Unknown basis_type"):
        get_basis(h2_molecule(), basis_type="not-a-basis")


def test_rank_aware_krylov_bisects_singular_rhs_blocks():
    right_hand_sides = np.arange(12.0).reshape(4, 3)

    def singular_block_solver(_operator, values, **_options):
        if len(values) > 1:
            raise np.linalg.LinAlgError("singular projected system")
        return 2.0 * values

    solution, number_batches, minimum_batch_size = _solve_rank_aware_krylov(
        singular_block_solver,
        lambda values: values,
        right_hand_sides,
    )

    np.testing.assert_array_equal(solution, 2.0 * right_hand_sides)
    assert number_batches == 4
    assert minimum_batch_size == 1


def test_rank_aware_krylov_does_not_hide_single_rhs_failure():
    def singular_solver(_operator, _values, **_options):
        raise np.linalg.LinAlgError("singular operator")

    with pytest.raises(np.linalg.LinAlgError, match="singular operator"):
        _solve_rank_aware_krylov(
            singular_solver,
            lambda values: values,
            np.ones((1, 3)),
        )


@pytest.mark.parametrize("basis_name", ["SAO", "meta_lowdin", "canonical"])
def test_core_bases_are_orthonormal_for_nontrivial_basis_set(h2_molecule, basis_name):
    mol = h2_molecule(1.4, basis="6-31g")
    basis = get_basis(mol, basis_type=basis_name)
    overlap = mol.intor_symmetric("int1e_ovlp")

    assert basis.shape == (mol.nao, mol.nao)
    np.testing.assert_allclose(basis.T @ overlap @ basis, np.eye(mol.nao), atol=1e-10)


def test_loewdin_transform_zeroes_numerically_null_eigenspace():
    overlap = np.diag([4.0, 1.0, 1.0e-18])
    trafo = get_loewdin_trafo(overlap)
    np.testing.assert_allclose(trafo, np.diag([0.5, 1.0, 0.0]))


def test_run_hf_returns_converged_reusable_mean_field(h2_molecule):
    mol = h2_molecule(1.4, basis="6-31g")
    mf = run_hf(mol)

    assert mf.converged
    assert np.isfinite(mf.e_tot)
    np.testing.assert_allclose(
        mf.mo_coeff.T @ mf.get_ovlp() @ mf.mo_coeff,
        np.eye(mol.nao),
        atol=1e-10,
    )


def test_rhf_mo_coefficient_derivative_loads_pyscf_hessian(h2_molecule):
    mol = h2_molecule(1.4, basis="6-31g")
    mf = run_hf(mol)

    derivative = rhf_mo_coefficient_derivatives(mf, atmlst=[0])

    assert derivative.shape == (1, 3, mol.nao, mol.nao)
    assert np.all(np.isfinite(derivative))


def test_sao_basis_derivative_satisfies_metric_orthonormality_derivative(h2_molecule):
    mol = h2_molecule(1.4, basis="6-31g")
    basis, derivative = get_basis_with_derivative(mol, basis_type="SAO")
    overlap = mol.intor_symmetric("int1e_ovlp")
    overlap_derivative = get_overlap_grad(mol)

    residual = np.einsum("pi,pqAx,qj->ijAx", basis, overlap_derivative, basis)
    residual += np.einsum("piAx,pq,qj->ijAx", derivative, overlap, basis)
    residual += np.einsum("pi,pq,qjAx->ijAx", basis, overlap, derivative)

    np.testing.assert_allclose(residual, 0.0, atol=2e-8)


def test_lowdin_localization_derivative_matches_finite_difference_direction(h2_molecule):
    mol = h2_molecule(1.4)
    overlap = mol.intor_symmetric("int1e_ovlp")
    direction = np.array([[0.2, -0.1], [-0.1, 0.3]])
    dummy_overlap_grad = np.zeros((mol.nao, mol.nao, mol.natm, 3))
    dummy_overlap_grad[:, :, 0, 0] = direction
    basis, derivative = orth_ao_derivative(
        mol, method="lowdin", pre_orth_ao=None, s=overlap, overlap_grad=dummy_overlap_grad
    )
    step = 1.0e-6
    plus = get_loewdin_trafo(overlap + step * direction)
    minus = get_loewdin_trafo(overlap - step * direction)

    np.testing.assert_allclose(basis, get_loewdin_trafo(overlap), atol=1e-12)
    np.testing.assert_allclose(derivative[:, :, 0, 0], (plus - minus) / (2 * step), rtol=2e-6, atol=2e-8)


def test_localization_can_return_basis_without_derivatives(h2_molecule, monkeypatch):
    mol = h2_molecule(1.4, basis="6-31g")
    overlap = mol.intor_symmetric("int1e_ovlp")
    basis, _ = orth_ao_derivative(mol, method="meta_lowdin", s=overlap)

    original_intor = mol.intor

    def reject_overlap_derivative(name, *args, **kwargs):
        if name == "int1e_ipovlp":
            raise AssertionError("orbital-only construction requested AO derivatives")
        return original_intor(name, *args, **kwargs)

    monkeypatch.setattr(mol, "intor", reject_overlap_derivative)

    basis_only = orth_ao_derivative(
        mol,
        method="meta_lowdin",
        s=overlap,
        return_derivatives=False,
    )

    assert isinstance(basis_only, np.ndarray)
    np.testing.assert_allclose(basis_only, basis, atol=1e-12)
    np.testing.assert_allclose(
        basis_only.T @ overlap @ basis_only,
        np.eye(mol.nao),
        atol=1e-10,
    )


def test_meta_lowdin_basis_does_not_use_pyscf_orth_ao(h2_molecule, monkeypatch):
    mol = h2_molecule(1.4, basis="6-31g")

    def reject_pyscf_orth_ao(*args, **kwargs):
        raise AssertionError("PySCF orth_ao applies an artificial phase gauge")

    monkeypatch.setattr(
        "evcont.basis.basis_utils.lo.orth.orth_ao", reject_pyscf_orth_ao
    )
    basis = get_basis(mol, basis_type="meta_lowdin")
    overlap = mol.intor_symmetric("int1e_ovlp")

    np.testing.assert_allclose(
        basis.T @ overlap @ basis,
        np.eye(mol.nao),
        atol=1e-10,
    )


def test_procrustes_rotation_and_derivative_match_finite_difference(rng):
    overlap = rng.normal(size=(5, 5)) + 3.0 * np.eye(5)
    directions = rng.normal(size=(2, 5, 5))
    rotation, derivative = procrustes_rotation_derivative(overlap, directions)
    step = 1.0e-6

    np.testing.assert_allclose(rotation.T @ rotation, np.eye(5), atol=1e-12)
    for index, direction in enumerate(directions):
        finite_difference = (
            procrustes_rotation(overlap + step * direction)
            - procrustes_rotation(overlap - step * direction)
        ) / (2 * step)
        np.testing.assert_allclose(derivative[index], finite_difference, rtol=3e-6, atol=3e-8)


def test_procrustes_derivative_rejects_rank_deficient_matching_matrix():
    with pytest.raises(ProcrustesDerivativeError, match="not uniquely supported"):
        procrustes_rotation_derivative(np.diag([1.0, 0.0]), np.eye(2))


def test_gradient_contraction_matches_direct_einsum(rng):
    norb, natm = 4, 3
    one = rng.normal(size=(norb, norb))
    two = rng.normal(size=(norb, norb, norb, norb))
    h1_jac = rng.normal(size=(norb, norb, natm, 3))
    h2_jac = rng.normal(size=(norb, norb, norb, norb, natm, 3))

    actual = get_grad_elec_from_gradH(one, two, h1_jac, h2_jac)
    expected = np.einsum("ij,ijAx->Ax", one, h1_jac)
    expected += 0.5 * np.einsum("ijkl,ijklAx->Ax", two, h2_jac)
    np.testing.assert_allclose(actual, expected)


def test_fix_gauge_makes_largest_real_component_nonpositive():
    vectors = np.array([[1.0, -3.0, 2.0], [-4.0, 1.0, 2.0]])
    fix_gauge(vectors)
    np.testing.assert_array_equal(vectors, [[1.0, -3.0, 2.0], [-4.0, 1.0, 2.0]])
