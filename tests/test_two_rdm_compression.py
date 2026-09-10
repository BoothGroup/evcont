import numpy as np

from evcont.ab_initio_gradients_loewdin import (
    _compress_two_el_grad,
    get_grad_elec_from_gradH,
    get_two_el_grad,
)
from evcont.dynamics import MD_utils
from evcont.electron_integral_utils import compress_electron_exchange_symmetry
from evcont.fci.FCI_EVCont import FCI_EVCont_obj


def test_packed_two_electron_gradient_contraction_matches_dense(rng):
    norb = 3
    natm = 2
    h2_ao = rng.normal(size=(norb,) * 4)
    basis = rng.normal(size=(norb, norb))
    basis_gradient = rng.normal(size=(norb, norb, natm, 3))
    h2_ao_derivative = rng.normal(size=(3,) + (norb,) * 4)
    atom_slices = ((0, 1), (1, 3))

    dense = get_two_el_grad(
        h2_ao, basis, basis_gradient, h2_ao_derivative, atom_slices
    )
    packed_derivative = _compress_two_el_grad(dense)

    matrix = rng.normal(size=(norb**2, norb**2))
    two_rdm = (matrix + matrix.T).reshape((norb,) * 4)
    packed_rdm = compress_electron_exchange_symmetry(two_rdm)
    one_rdm = np.zeros((norb, norb))
    h1_jac = np.zeros((norb, norb, natm, 3))

    dense_gradient = get_grad_elec_from_gradH(
        one_rdm, two_rdm, h1_jac, dense
    )
    packed_gradient = get_grad_elec_from_gradH(
        one_rdm, packed_rdm, h1_jac, packed_derivative
    )
    np.testing.assert_allclose(packed_gradient, dense_gradient)


def test_active_learning_matches_with_exchange_compressed_two_rdms(
    tmp_path, monkeypatch, h2_molecule
):
    trajectory = np.asarray(
        [h2_molecule(distance).atom_coords() for distance in (1.2, 1.8)]
    )

    def fixed_trajectory(continuation, init_mol, *_args, **_kwargs):
        for geometry in trajectory:
            mol = init_mol.copy().set_geom_(geometry, unit="Bohr")
            continuation.get_en_with_grad(mol)
        return trajectory

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MD_utils, "_trajectory_iteration", fixed_trajectory)
    monkeypatch.setattr(
        MD_utils,
        "_trajectory_energies",
        lambda _iteration, geometry: np.zeros(len(geometry)),
    )

    models = []
    for compressed in (False, True):
        continuation = FCI_EVCont_obj(compress_two_rdm=compressed)
        MD_utils.converge_EVCont_MD(
            continuation,
            h2_molecule(1.2),
            max_iter=2,
            nconv=2,
            restart=False,
            data_addition="farthest_point",
            model_dir=tmp_path / f"model-{compressed}",
        )
        models.append(continuation)

    dense, compressed = models
    assert dense.two_rdm.shape == (2, 2, 2, 2, 2, 2)
    assert compressed.two_rdm.shape == (2, 2, 10)
    np.testing.assert_allclose(compressed.overlap, dense.overlap, atol=1.0e-12)
    np.testing.assert_allclose(compressed.one_rdm, dense.one_rdm, atol=1.0e-12)

    test_mol = h2_molecule(1.5)
    dense_energy, dense_gradient = dense.get_en_with_grad(test_mol)
    compressed_energy, compressed_gradient = compressed.get_en_with_grad(test_mol)
    np.testing.assert_allclose(compressed_energy, dense_energy, atol=1.0e-12)
    np.testing.assert_allclose(compressed_gradient, dense_gradient, atol=1.0e-12)
