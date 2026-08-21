import numpy as np
import pytest

from evcont.ab_initio_eigenvector_continuation import (
    approximate_ground_state,
    approximate_multistate,
    approximate_multistate_otf,
    solve_subspace,
)
from evcont.electron_integral_utils import (
    compress_electron_exchange_symmetry,
    get_df_integrals,
    get_integrals,
    restore_electron_exchange_symmetry,
    transform_integrals,
)


def _symmetric_problem(rng, ntrain=4, norb=3):
    h1 = rng.normal(size=(norb, norb))
    h1 = h1 + h1.T
    pair_h2 = rng.normal(size=(norb**2, norb**2))
    pair_h2 = pair_h2 + pair_h2.T
    h2 = pair_h2.reshape((norb,) * 4)

    one = rng.normal(size=(ntrain, ntrain, norb, norb))
    one = 0.5 * (one + one.transpose(1, 0, 3, 2))
    two = rng.normal(size=(ntrain, ntrain, norb**2, norb**2))
    two = 0.25 * (
        two
        + two.transpose(1, 0, 2, 3)
        + two.transpose(0, 1, 3, 2)
        + two.transpose(1, 0, 3, 2)
    )
    two = two.reshape((ntrain, ntrain) + (norb,) * 4)
    overlap = np.eye(ntrain) + 0.03 * np.ones((ntrain, ntrain))
    return h1, h2, one, two, overlap


@pytest.mark.parametrize("norb", [1, 2, 5, 9])
def test_exchange_symmetry_compression_round_trip_scales_with_orbital_count(rng, norb):
    matrix = rng.normal(size=(norb**2, norb**2))
    matrix = matrix + matrix.T
    tensor = matrix.reshape((norb,) * 4)
    original = tensor.copy()

    packed = compress_electron_exchange_symmetry(tensor)
    restored = restore_electron_exchange_symmetry(packed, norb)

    assert packed.shape == (norb**2 * (norb**2 + 1) // 2,)
    np.testing.assert_allclose(restored, original)
    np.testing.assert_allclose(tensor, original)  # compression must not mutate callers


def test_exchange_symmetry_diagonal_multiplier_only_changes_packed_diagonal(rng):
    norb = 3
    matrix = rng.normal(size=(norb**2, norb**2))
    matrix = matrix + matrix.T
    tensor = matrix.reshape((norb,) * 4)

    normal = compress_electron_exchange_symmetry(tensor)
    halved = compress_electron_exchange_symmetry(tensor, diag_multiplier=0.5)
    restored_normal = restore_electron_exchange_symmetry(normal, norb).reshape(norb**2, norb**2)
    restored_halved = restore_electron_exchange_symmetry(halved, norb).reshape(norb**2, norb**2)

    np.testing.assert_allclose(np.diag(restored_halved), 0.5 * np.diag(restored_normal))
    np.testing.assert_allclose(
        restored_halved - np.diag(np.diag(restored_halved)),
        restored_normal - np.diag(np.diag(restored_normal)),
    )


def test_transform_integrals_supports_leading_batch_dimensions(rng):
    h1 = rng.normal(size=(2, 4, 4))
    h2 = rng.normal(size=(2, 4, 4, 4, 4))
    trafo = rng.normal(size=(3, 4))

    actual_h1, actual_h2 = transform_integrals(h1, h2, trafo)
    expected_h1 = np.stack([trafo @ item @ trafo.T for item in h1])
    expected_h2 = np.einsum("...ijkl,ai,bj,ck,dl->...abcd", h2, trafo, trafo, trafo, trafo)

    assert actual_h1.shape == (2, 3, 3)
    assert actual_h2.shape == (2, 3, 3, 3, 3)
    np.testing.assert_allclose(actual_h1, expected_h1)
    np.testing.assert_allclose(actual_h2, expected_h2)


def test_all_supported_two_rdm_storage_forms_give_same_subspace_solution(rng):
    h1, h2, one, two, overlap = _symmetric_problem(rng)
    ntrain, norb = one.shape[0], one.shape[-1]
    packed_data = two[np.tril_indices(ntrain)]
    packed_electrons = np.empty((ntrain, ntrain, norb**2 * (norb**2 + 1) // 2))
    for i in range(ntrain):
        for j in range(ntrain):
            packed_electrons[i, j] = compress_electron_exchange_symmetry(two[i, j])
    packed_both = packed_electrons[np.tril_indices(ntrain)]

    reference = approximate_multistate(h1, h2, one, two, overlap, nroots=3)[0]
    for representation in (packed_data, packed_electrons, packed_both):
        energies, vectors = approximate_multistate(
            h1, h2, one, representation, overlap, nroots=3
        )
        np.testing.assert_allclose(energies, reference, rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(
            np.einsum("ri,ij,sj->rs", vectors, overlap, vectors), np.eye(3), atol=1e-10
        )


def test_ground_state_and_multistate_paths_agree(rng):
    h1, h2, one, two, overlap = _symmetric_problem(rng, ntrain=3, norb=2)
    ground_energy, ground_vector = approximate_ground_state(h1, h2, one, two, overlap)
    energies, vectors = approximate_multistate(h1, h2, one, two, overlap, nroots=1)

    np.testing.assert_allclose(ground_energy, energies[0])
    np.testing.assert_allclose(abs(ground_vector), abs(vectors[0]))


def test_solve_subspace_discards_linearly_dependent_overlap_direction():
    hamiltonian = np.diag([-2.0, -1.0, 50.0])
    overlap = np.diag([1.0, 2.0, 1.0e-16])

    energies, vectors = solve_subspace(hamiltonian, overlap, nroots=2, lindep=1e-12)

    np.testing.assert_allclose(energies, [-2.0, -0.5])
    np.testing.assert_allclose(vectors @ overlap @ vectors.T, np.eye(2), atol=1e-12)


def test_on_the_fly_hamiltonian_matches_explicit_rdms(rng):
    h1, h2, one, two, overlap = _symmetric_problem(rng, ntrain=3, norb=2)
    expected = approximate_multistate_otf(h1, h2, one, two, overlap, nroots=2)

    def builder(actual_h1, actual_h2):
        hamiltonian = np.einsum("abij,ij->ab", one, actual_h1)
        hamiltonian += 0.5 * np.einsum("abijkl,ijkl->ab", two, actual_h2)
        return hamiltonian, overlap

    actual = approximate_multistate_otf(h1, h2, otf_hamiltonian=builder, nroots=2)
    np.testing.assert_allclose(actual[0], expected[0])
    np.testing.assert_allclose(abs(actual[1]), abs(expected[1]))


def test_pyscf_integrals_and_density_fitting_have_expected_symmetry(h2_molecule):
    mol = h2_molecule(1.4)
    basis = np.linalg.inv(np.linalg.cholesky(mol.intor_symmetric("int1e_ovlp"))).T
    h1, h2 = get_integrals(mol, basis)
    cderi = get_df_integrals(mol, basis=basis)
    h2_df = np.einsum("Pij,Pkl->ijkl", cderi, cderi)

    np.testing.assert_allclose(h1, h1.T, atol=1e-12)
    np.testing.assert_allclose(h2, h2.transpose(2, 3, 0, 1), atol=1e-12)
    np.testing.assert_allclose(h2_df, h2, rtol=2e-2, atol=2e-2)


def test_invalid_two_rdm_rank_is_rejected(rng):
    h1, h2, one, _, overlap = _symmetric_problem(rng, ntrain=2, norb=2)
    with pytest.raises(AssertionError):
        approximate_multistate(h1, h2, one, np.zeros((2, 2, 2, 2)), overlap)
