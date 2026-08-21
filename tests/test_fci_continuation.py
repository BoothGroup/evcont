import copy

import numpy as np

from evcont.fci.FCI_EVCont import FCI_EVCont_obj


def test_fci_continuation_end_to_end_regression(h2_molecule):
    continuation = FCI_EVCont_obj(nroots=1, abstract_basis="SAO")
    for distance in (0.8, 1.6):
        continuation.append_to_rdms(h2_molecule(distance))

    np.testing.assert_allclose(
        continuation.ens,
        [-2.207598532021782, -1.7538156436492538],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        continuation.overlap,
        [[1.0, 0.9972576510800326], [0.9972576510800326, 1.0]],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.trace(continuation.one_rdm, axis1=-2, axis2=-1).diagonal(),
        [2.0, 2.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        continuation.two_rdm,
        continuation.two_rdm.transpose(1, 0, 5, 4, 3, 2).conj(),
        atol=1e-12,
    )

    energy, vector = continuation.get_en(h2_molecule(1.2), nroots=1)
    np.testing.assert_allclose(energy, [-1.1266988216562055], rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(vector @ continuation.overlap @ vector.T, np.eye(1), atol=1e-11)


def test_fci_training_geometry_is_reproduced(h2_molecule):
    continuation = FCI_EVCont_obj(nroots=1)
    molecule = h2_molecule(1.4)
    continuation.append_to_rdms(molecule)
    energy, _ = continuation.get_en(molecule)
    np.testing.assert_allclose(energy[0], continuation.ens[0] + continuation.ens_nuc[0], atol=1e-11)


def test_fci_pruning_updates_every_training_axis_and_persists(tmp_path, h2_molecule):
    continuation = FCI_EVCont_obj(nroots=1)
    for distance in (0.8, 1.2, 1.6):
        continuation.append_to_rdms(h2_molecule(distance))
    kept_energies = [continuation.ens[index] for index in (0, 2)]

    continuation.prune_datapoints(np.array([0, 2]))
    assert continuation.overlap.shape == (2, 2)
    assert continuation.one_rdm.shape[:2] == (2, 2)
    assert continuation.two_rdm.shape[:2] == (2, 2)
    np.testing.assert_allclose(continuation.ens, kept_energies)

    path = tmp_path / "fci.pkl"
    continuation.save(path)
    loaded = FCI_EVCont_obj.load(path)
    np.testing.assert_allclose(loaded.overlap, continuation.overlap)
    np.testing.assert_allclose(loaded.get_en(h2_molecule(1.3))[0], continuation.get_en(h2_molecule(1.3))[0])


def test_lowrank_fci_vectorization_round_trip_retains_form_and_can_append(h2_molecule):
    options = {
        "lowrank": True,
        "truncation_style": "nvec",
        "nvecs": 4,
        "save_diag": True,
        "Jdiag_only": False,
        "relax_amp": False,
    }
    continuation = FCI_EVCont_obj(**options)
    for distance in (0.8, 1.6):
        continuation.append_to_rdms(h2_molecule(distance))

    control = copy.deepcopy(continuation)
    original_vectors = copy.deepcopy(continuation.vecs_lowrank)
    original_diagonals = continuation.diagonal_lr.copy()

    continuation.vectorize_lowrank(hermitian=True)
    energy_before = continuation.get_en(h2_molecule(1.2), nroots=1)[0]
    continuation.unpack_vectorized_lowrank()

    np.testing.assert_allclose(continuation.diagonal_lr, original_diagonals, atol=0.0)
    assert continuation.vecs_lowrank.keys() == original_vectors.keys()
    for pair, expected in original_vectors.items():
        actual = continuation.vecs_lowrank[pair]
        assert actual[0].shape == expected[0].shape
        assert actual[1].shape == expected[1].shape
        assert actual[2].shape == expected[2].shape
        assert actual[3] is expected[3]
        np.testing.assert_allclose(actual[0], expected[0], atol=0.0)
        np.testing.assert_allclose(actual[1], expected[1], atol=0.0)
        np.testing.assert_allclose(actual[2], expected[2], atol=0.0)

    continuation.vectorize_lowrank(hermitian=True)
    energy_after = continuation.get_en(h2_molecule(1.2), nroots=1)[0]
    np.testing.assert_allclose(energy_after, energy_before, atol=0.0)

    # Appending requires the per-pair form. Compare against a continuation that
    # was never vectorized to ensure the recovered representation is operational.
    continuation.unpack_vectorized_lowrank()
    continuation.append_to_rdms(h2_molecule(1.2))
    control.append_to_rdms(h2_molecule(1.2))
    continuation.vectorize_lowrank(hermitian=True)
    control.vectorize_lowrank(hermitian=True)

    np.testing.assert_allclose(continuation.overlap, control.overlap, atol=0.0)
    np.testing.assert_allclose(continuation.one_rdm, control.one_rdm, atol=0.0)
    np.testing.assert_allclose(
        continuation.diagonal_vectorized, control.diagonal_vectorized, atol=0.0
    )
    np.testing.assert_allclose(
        continuation.get_en(h2_molecule(1.3), nroots=1)[0],
        control.get_en(h2_molecule(1.3), nroots=1)[0],
        atol=0.0,
    )
