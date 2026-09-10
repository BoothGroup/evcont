import numpy as np
import pytest
from pyscf import gto

from evcont.ccsd.CCSD_EVCont import CCSD_EVCont_obj
from evcont.low_rank_utils import reconstruct_rdm2_joint
from evcont.rdm_orthonormalization import incremental_orthonormalizer


pytestmark = pytest.mark.filterwarnings(
    r"ignore::DeprecationWarning:ebcc\..*"
)

RTOL = 1.0e-9
ATOL = 1.0e-10
OTHER_BASIS_RESULTS = {
    "meta_lowdin": (-2.1778568522863244, False),
    "split_procrustes": (-2.1779376837205913, True),
    "least_change_atom_coordinate": (-2.1779376837206126, True),
    "least_change_frozen_mo": (-2.1779376837211197, True),
    "least_change_local": (-2.1779373106984865, False),
}
DF_ABSTRACT_BASIS_RESULTS = {
    "split_procrustes": -2.177937553642934,
    "least_change_atom_coordinate": -2.177937553642871,
    "least_change_frozen_mo": -2.1779375536431838,
    "least_change_local": -2.1779371776339103,
}


def _water(oh_distance):
    half_angle = 0.5 * np.deg2rad(104.5)
    x = oh_distance * np.sin(half_angle)
    z = oh_distance * np.cos(half_angle)
    return gto.M(
        atom=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (x, 0.0, z)),
            ("H", (-x, 0.0, z)),
        ],
        basis="6-31g",
        unit="Angstrom",
        verbose=0,
    )


def _h4(spacing):
    return gto.M(
        atom=[("H", (i * spacing, 0.0, 0.0)) for i in range(4)],
        basis="sto-3g",
        unit="Bohr",
        symmetry=False,
        verbose=0,
    )


def test_ccsd_continuation_matches_pre_refactor_baseline():
    continuation = CCSD_EVCont_obj(_water(0.958), abstract_basis="SAO")
    for distance in (0.90, 1.05, 1.20):
        continuation.append_to_rdms(_water(distance))

    np.testing.assert_allclose(
        continuation.ens,
        [-76.11696114888737, -76.13275519587776, -76.17526566839865],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        continuation.train_energies,
        [-76.10703408263961, -76.11334493391917, -76.07109723294019],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        continuation.overlap,
        [
            [1.0, 0.9772429949732743, 0.9205667656992705],
            [0.9772429949732743, 1.0, 0.9821003000443925],
            [0.9205667656992705, 0.9821003000443925, 1.0],
        ],
        rtol=RTOL,
        atol=ATOL,
    )

    energy, _ = continuation.approximate(_water(1.00))
    np.testing.assert_allclose(
        np.ravel(energy)[0], -76.11962855448573, rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize(
    "abstract_basis, expected",
    OTHER_BASIS_RESULTS.items(),
    ids=OTHER_BASIS_RESULTS,
)
def test_ccsd_continuation_with_other_abstract_bases(abstract_basis, expected):
    expected_energy, requires_reference = expected
    basis_kwargs = (
        {"least_change_emit_warnings": False}
        if abstract_basis.startswith("least_change")
        else None
    )
    continuation = CCSD_EVCont_obj(
        _h4(1.5),
        abstract_basis=abstract_basis,
        abstract_basis_kwargs=basis_kwargs,
    )

    assert (continuation.abstract_basis_ref is not None) is requires_reference
    overlap_ao = continuation.comp_mol.intor_symmetric("int1e_ovlp")
    np.testing.assert_allclose(
        continuation.comp_basis_abstract.T
        @ overlap_ao
        @ continuation.comp_basis_abstract,
        np.eye(continuation.comp_mol.nao),
        rtol=1.0e-9,
        atol=1.0e-10,
    )

    for spacing in (1.2, 1.8):
        continuation.append_to_rdms(_h4(spacing))

    np.testing.assert_allclose(
        continuation.overlap,
        continuation.overlap.T.conj(),
        rtol=1.0e-9,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.diag(continuation.overlap), np.ones(2), rtol=1.0e-9, atol=1.0e-10
    )

    energy, _ = continuation.approximate(_h4(1.6))
    np.testing.assert_allclose(
        np.ravel(energy)[0], expected_energy, rtol=1.0e-7, atol=1.0e-8
    )


@pytest.mark.parametrize(
    "abstract_basis, expected_energy",
    DF_ABSTRACT_BASIS_RESULTS.items(),
    ids=DF_ABSTRACT_BASIS_RESULTS,
)
def test_ccsd_continuation_with_density_fitted_abstract_basis(
    abstract_basis, expected_energy
):
    basis_kwargs = {"density_fit": True, "df_basis": None}
    if abstract_basis.startswith("least_change"):
        basis_kwargs["least_change_emit_warnings"] = False

    with pytest.warns(
        RuntimeWarning,
        match="enabling density fitting for the CCSD RHF calculations",
    ):
        continuation = CCSD_EVCont_obj(
            _h4(1.5),
            abstract_basis=abstract_basis,
            abstract_basis_kwargs=basis_kwargs,
        )

    assert continuation.scf_density_fit is True
    assert continuation.scf_df_basis is None
    assert continuation.abstract_basis_kwargs["density_fit"] is True
    assert continuation.abstract_basis_kwargs["df_basis"] is None
    assert hasattr(continuation.comp_mf, "with_df")

    for spacing in (1.2, 1.8):
        continuation.append_to_rdms(_h4(spacing))

    energy, _ = continuation.approximate(_h4(1.6))
    np.testing.assert_allclose(
        np.ravel(energy)[0], expected_energy, rtol=1.0e-7, atol=1.0e-8
    )


@pytest.mark.parametrize(
    "abstract_basis", ["split_procrustes", "least_change_local"]
)
def test_abstract_df_basis_overrides_ccsd_df_basis_with_warnings(abstract_basis):
    basis_kwargs = {
        "density_fit": True,
        "df_basis": "weigend",
    }
    if abstract_basis.startswith("least_change"):
        basis_kwargs["least_change_emit_warnings"] = False
    with pytest.warns(RuntimeWarning) as caught:
        continuation = CCSD_EVCont_obj(
            _h4(1.5),
            abstract_basis=abstract_basis,
            abstract_basis_kwargs=basis_kwargs,
            scf_density_fit=False,
            scf_df_basis=None,
        )

    messages = [str(item.message) for item in caught]
    assert any("enabling density fitting" in message for message in messages)
    assert any("different auxiliary bases" in message for message in messages)
    assert continuation.scf_density_fit is True
    assert continuation.scf_df_basis == "weigend"
    assert continuation.abstract_basis_kwargs["df_basis"] == "weigend"


def test_ccsd_df_configuration_is_propagated_to_least_change_basis():
    continuation = CCSD_EVCont_obj(
        _h4(1.5),
        abstract_basis="least_change_local",
        abstract_basis_kwargs={"least_change_emit_warnings": False},
        scf_density_fit=True,
        scf_df_basis=None,
    )

    assert continuation.abstract_basis_kwargs["density_fit"] is True
    assert continuation.abstract_basis_kwargs["df_basis"] is None


def test_incremental_orthonormalizer_preserves_existing_columns(rng):
    raw_vectors = rng.normal(size=(6, 3))
    overlap = raw_vectors.T @ raw_vectors
    first, _, _ = incremental_orthonormalizer(overlap[:1, :1])
    second, _, _ = incremental_orthonormalizer(overlap[:2, :2], first)
    third, _, _ = incremental_orthonormalizer(overlap, second)

    np.testing.assert_allclose(third[:2, :2], second, atol=1.0e-14)
    np.testing.assert_allclose(
        third.T @ overlap @ third, np.eye(3), atol=1.0e-12
    )


@pytest.mark.parametrize("density_fit", [False, True])
def test_ccsd_checkpoint_round_trip_restores_reference_mean_fields(
    tmp_path, density_fit
):
    continuation = CCSD_EVCont_obj(
        _h4(1.5),
        abstract_basis="split_procrustes",
        abstract_basis_kwargs={"density_fit": density_fit},
    )
    expected_comp_coeff = continuation.comp_mf.mo_coeff.copy()
    expected_ref_coeff = continuation.abstract_basis_ref_mf.mo_coeff.copy()
    checkpoint = tmp_path / "ccsd.pkl"

    continuation.save(checkpoint)
    restored = CCSD_EVCont_obj.load(checkpoint)

    assert restored.comp_mf.converged
    assert restored.abstract_basis_ref_mf.converged
    assert hasattr(restored.comp_mf, "with_df") is density_fit
    np.testing.assert_array_equal(restored.comp_mf.mo_coeff, expected_comp_coeff)
    np.testing.assert_array_equal(
        restored.abstract_basis_ref_mf.mo_coeff, expected_ref_coeff
    )
    restored.get_abstract_basis(_h4(1.6))
    restored.save(tmp_path / "ccsd-restored.pkl")


def test_lowrank_ccsd_recompresses_in_orthonormal_state_basis():
    molecules = [_h4(1.35), _h4(1.50)]
    transverse = _h4(1.50)
    transverse_coordinates = transverse.atom_coords()
    transverse_coordinates[1, 1] += 0.25
    transverse.set_geom_(transverse_coordinates, unit="Bohr")
    molecules.append(transverse)
    common = {
        "abstract_basis": "least_change_atom_coordinate",
        "abstract_basis_kwargs": {"least_change_emit_warnings": False},
    }
    dense = CCSD_EVCont_obj(molecules[0], **common)
    lowrank = CCSD_EVCont_obj(
        molecules[0],
        lowrank=True,
        lowrank_kwargs={
            "truncation_style": "ham",
            "ham_thr": 1.0e-12,
            "save_diag": False,
            "Jdiag_only": False,
            "relax_amp": False,
            "density_fit": False,
            "lindep": 1.0e-14,
            "orthonormalization_lindep": 1.0e-14,
        },
        **common,
    )
    for mol in molecules:
        dense.append_to_rdms(mol)
        lowrank.append_to_rdms(mol)

    np.testing.assert_allclose(lowrank.overlap, np.eye(3), atol=2.0e-9)
    np.testing.assert_allclose(
        lowrank.state_transform.T.conj()
        @ lowrank.raw_overlap
        @ lowrank.state_transform,
        np.eye(3),
        atol=2.0e-9,
    )
    assert lowrank.two_rdm is None
    assert lowrank.raw_one_rdm.shape[:2] == (3, 3)

    dense_two_orthogonal = np.einsum(
        "ia,ijpqrs,jb->abpqrs",
        lowrank.state_transform.conj(),
        dense.two_rdm,
        lowrank.state_transform,
        optimize="optimal",
    )
    for bra in range(3):
        for ket in range(3):
            values, left, right, joint = lowrank.vecs_lowrank[bra, ket]
            assert len(values) == molecules[0].nao**2
            actual = reconstruct_rdm2_joint(
                (values, left, right), joint=joint
            )
            np.testing.assert_allclose(
                actual, dense_two_orthogonal[bra, ket], atol=2.0e-7
            )

    probe = _h4(1.43)
    probe_coordinates = probe.atom_coords()
    probe_coordinates[2, 1] -= 0.15
    probe.set_geom_(probe_coordinates, unit="Bohr")
    dense_energy, dense_gradient = dense.get_en_with_grad(probe)
    lowrank_energy, lowrank_gradient = lowrank.get_en_with_grad(probe)
    np.testing.assert_allclose(lowrank_energy, dense_energy, atol=1.0e-9)
    np.testing.assert_allclose(lowrank_gradient, dense_gradient, atol=1.0e-8)
