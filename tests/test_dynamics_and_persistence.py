from types import SimpleNamespace

import numpy as np
import pytest

from evcont.dynamics.MD_utils import (
    _checkpoint_path,
    _converged,
    _latest_checkpoint,
    get_scanner,
    save_rdm_trajectory,
)
from evcont.dynamics.NAMD_utils import read_dyn, read_model, read_population, write_model
from evcont.dynamics.active_learning import (
    hamiltonian_distance,
    select_active_learning_geometry,
)
from evcont.rdm_inference import RDMResult
from evcont.solver_persistence import EVContPersistenceMixin


class Persisted(EVContPersistenceMixin):
    def __init__(self):
        self.values = np.arange(5)
        self.restored = False

    def _restore_persistence_state(self):
        self.restored = True


def test_persistence_round_trip_without_calling_constructor(tmp_path):
    path = tmp_path / "object.pkl"
    original = Persisted()
    original.save(path)
    loaded = Persisted.load(path)

    np.testing.assert_array_equal(loaded.values, original.values)
    assert loaded.restored


def test_checkpoint_helpers_choose_numeric_latest_suffix(tmp_path):
    for index in (2, 10, 3):
        _checkpoint_path(index, tmp_path).touch()
    assert _latest_checkpoint(tmp_path) == _checkpoint_path(10, tmp_path)
    assert _latest_checkpoint(tmp_path / "missing") is None


def test_convergence_requires_requested_number_of_recent_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    np.savetxt("en_diff_1.txt", [1e-5, 2e-5])
    np.savetxt("en_diff_2.txt", [5e-5])
    assert not _converged(1, threshold=1e-4, nconv=2)
    assert _converged(2, threshold=1e-4, nconv=2)
    np.savetxt("en_diff_2.txt", [2e-4])
    assert not _converged(2, threshold=1e-4, nconv=2)


def test_save_rdm_trajectory_writes_metadata_and_requested_arrays(tmp_path):
    results = [
        RDMResult(((0, 0),), "AO", one=np.full((1, 2, 2), value))
        for value in (1.0, 2.0, 3.0)
    ]
    path = tmp_path / "rdms.npz"
    save_rdm_trajectory(path, results)

    with np.load(path) as data:
        assert data["basis"].item() == "AO"
        np.testing.assert_array_equal(data["state_pairs"], [[0, 0]])
        assert data["one"].shape == (3, 1, 2, 2)
        assert "two" not in data


def test_scanner_collects_coefficients_and_rdms(h2_molecule):
    mol = h2_molecule()
    rdm = RDMResult(((0, 0),), "AO", one=np.eye(mol.nao)[None])

    class Continuation:
        def get_en_with_grad(self, actual_mol, **kwargs):
            assert kwargs == {
                "nroots": 1,
                "return_coefficients": True,
                "return_rdms": ("1rdm",),
                "rdm_basis": "AO",
            }
            return np.array([[1.0]]), np.array([-1.2]), np.zeros((1, actual_mol.natm, 3)), rdm

    scanner = get_scanner(
        mol,
        Continuation(),
        return_rdms=("1rdm",),
        collect_coefficients=True,
    )
    energy, gradient = scanner(mol)

    assert energy == pytest.approx(-1.2)
    assert gradient.shape == (mol.natm, 3)
    assert len(scanner.coefficients) == len(scanner.rdm_results) == 1


def test_hamiltonian_distance_supports_many_references(rng):
    norb = 3
    h1 = rng.normal(size=(norb, norb))
    h2 = rng.normal(size=(norb,) * 4)
    refs1 = np.stack([h1, h1 + 0.1, h1 + 0.2])
    refs2 = np.stack([h2, h2, h2])
    distances = hamiltonian_distance(h1, h2, refs1, refs2)
    assert distances.shape == (3,)
    assert distances[0] == pytest.approx(0.0)
    assert np.all(np.diff(distances) > 0)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("energy", 2),
        ("farthest_point_ham", 3),
        ("first_peak_ham", 1),
        ("weighted_highest_peak_ham", 1),
    ],
)
def test_active_learning_selection_methods(method, expected):
    distances = np.array([0.0, 2.0, 0.2, 4.0, 0.1])
    result = select_active_learning_geometry(
        distances,
        method=method,
        en_diff=np.array([0.1, 0.2, 0.9, 0.3, 0.0]),
        convergence_thresh=0.1,
        exponent=2.0,
    )
    assert result == expected


def test_active_learning_farthest_point_uses_full_geometry_distance():
    trajectory = np.array([[[0.0, 0.0, z]] for z in (0.0, 1.0, 3.0)])
    training = np.array([trajectory[0], trajectory[1]])
    assert select_active_learning_geometry(
        None,
        method="farthest_point",
        trajectory=trajectory,
        trn_geometries=training,
    ) == 2


def test_unknown_active_learning_method_is_rejected():
    with pytest.raises(ValueError, match="Unknown active-learning"):
        select_active_learning_geometry(np.ones(4), method="unknown")


def test_newton_x_text_readers_parse_populations_and_geometries(tmp_path):
    populations = tmp_path / "population.out"
    populations.write_text("Population 1 0.8\nPopulation 2 0.2\nPopulation 1 0.7\nPopulation 2 0.3\n")
    np.testing.assert_allclose(read_population(populations, 2), [[0.8, 0.2], [0.7, 0.3]])

    dynamics = tmp_path / "dyn.out"
    dynamics.write_text(
        "geometry\nH 1 0.0 0.0 0.0\nH 1 0.0 0.0 1.0\n"
        "ignored\ngeometry\nH 1 0.1 0.0 0.0\nH 1 0.0 0.0 1.1\n"
    )
    geometries = read_dyn(dynamics, natm=2)
    assert len(geometries) == 2
    np.testing.assert_allclose(geometries[1], [[0.1, 0.0, 0.0], [0.0, 0.0, 1.1]])


@pytest.mark.parametrize("lowrank", [False, True])
def test_newton_x_model_io_round_trip(tmp_path, rng, lowrank):
    overlap = np.eye(2)
    one = rng.normal(size=(2, 2, 3, 3))
    two = {"pair": np.arange(3)} if lowrank else rng.normal(size=(2, 2, 3, 3, 3, 3))
    diagonal = rng.normal(size=(2, 2, 3, 3, 3)) if lowrank else None
    kwargs = {"vecs_lowrank": two, "diagonal_lr": diagonal} if lowrank else {"two_rdm": two}

    write_model(overlap, one, model_path=tmp_path, **kwargs)
    actual_overlap, actual_one, actual_two, actual_diagonal = read_model(tmp_path)
    np.testing.assert_allclose(actual_overlap, overlap)
    np.testing.assert_allclose(actual_one, one)
    if lowrank:
        np.testing.assert_array_equal(actual_two["pair"], two["pair"])
        np.testing.assert_allclose(actual_diagonal, diagonal)
    else:
        np.testing.assert_allclose(actual_two, two)
        assert actual_diagonal is None
