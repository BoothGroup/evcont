"""Shared active-learning geometry selection utilities."""

import numpy as np
from scipy.signal import find_peaks

from evcont.electron_integral_utils import get_basis, get_integrals


def hamiltonian_distance(h1, h2, h1_ref, h2_ref):
    """Return the normalized squared distance between Hamiltonians."""
    norb = h1.shape[-1]
    return 1000 * (
        np.sum(abs(h1 - h1_ref) ** 2, axis=(-1, -2)) / norb**2
        + 0.5 * np.sum(abs(h2 - h2_ref) ** 2, axis=(-1, -2, -3, -4))
        / norb**4
    )


def hamiltonian_similarity(
    init_mol, trajectory, trn_geometries, basis_getter=None
):
    """Return minimum Hamiltonian distances and closest training indices."""
    basis_getter = basis_getter or get_basis

    def integrals(geometry):
        mol = init_mol.copy().set_geom_(geometry)
        return get_integrals(mol, basis_getter(mol))

    h1_ref, h2_ref = zip(*(integrals(geometry) for geometry in trn_geometries))
    h1_ref, h2_ref = np.asarray(h1_ref), np.asarray(h2_ref)

    distances, closest = [], []
    for geometry in trajectory:
        h1, h2 = integrals(geometry)
        distance = hamiltonian_distance(h1, h2, h1_ref, h2_ref)
        closest.append(int(np.argmin(distance)))
        distances.append(np.min(distance))
    return np.asarray(distances), np.asarray(closest, dtype=int)


def select_active_learning_geometry(
    hamiltonian_distances,
    method="weighted_highest_peak_ham",
    en_diff=None,
    convergence_thresh=None,
    exponent=0.5,
    trajectory=None,
    trn_geometries=None,
    peak_threshold=0.01,
):
    """Select the trajectory geometry to add to a continuation model."""
    if method == "energy":
        return int(np.argmax(en_diff))
    if method == "farthest_point":
        distances = [
            np.sum(abs(trajectory - geometry) ** 2, axis=(-1, -2))
            for geometry in trn_geometries
        ]
        return int(np.argmax(np.min(distances, axis=0)))

    distances = np.asarray(hamiltonian_distances)
    maximum = int(np.argmax(distances))
    if method == "farthest_point_ham":
        return maximum

    peaks = find_peaks(distances)[0]
    if method == "first_peak_ham":
        valid = peaks[distances[peaks] > peak_threshold]
        return int(valid[0]) if len(valid) else maximum

    if method not in {"weighted_highest_peak_ham", "variable_weight_peak_ham"}:
        raise ValueError(f"Unknown active-learning selection method: {method}")
    if method == "variable_weight_peak_ham":
        exponent = np.clip(0.01 * np.max(en_diff) / convergence_thresh, 0, 5)

    peaks = np.unique(np.append(peaks, maximum))
    peaks = peaks[(peaks > 0) & (distances[peaks] > peak_threshold)]
    if not len(peaks):
        return maximum
    score = distances[peaks] / (peaks / len(distances)) ** exponent
    return int(peaks[np.argmax(score)])


def hamiltonian_similarity_argmin(init_mol, trajectory, trn_geometries):
    """Backward-compatible alias returning distances and closest indices."""
    return hamiltonian_similarity(init_mol, trajectory, trn_geometries)

