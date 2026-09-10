from pathlib import Path
import os

import numpy as np
from pyscf import lib, md
from threadpoolctl import threadpool_limits

from evcont.dynamics.active_learning import (
    hamiltonian_similarity,
    select_active_learning_geometry,
)

try:
    from mpi4py import MPI
except ImportError:
    class _SerialComm:
        rank = 0

        def Get_rank(self):
            return 0

        def Get_size(self):
            return 1

        def Split_type(self, *_):
            return self

        def Bcast(self, *_args, **_kwargs):
            return None

        def bcast(self, value, root=0):
            return value

        def Barrier(self):
            return None

    class _SerialMPI:
        COMM_WORLD = _SerialComm()
        COMM_TYPE_SHARED = 0

    MPI = _SerialMPI()


rank = MPI.COMM_WORLD.Get_rank()


def save_rdm_trajectory(filename, results):
    """Save a sequence of :class:`RDMResult` objects to one NPZ file."""
    if not results:
        return
    payload = {
        "basis": np.asarray(results[0].basis),
        "state_pairs": np.asarray(results[0].state_pairs),
    }
    if results[0].one is not None:
        payload["one"] = np.stack([result.one for result in results])
    if results[0].two is not None:
        payload["two"] = np.stack([result.two for result in results])
    np.savez(filename, **payload)


def get_scanner(
    mol,
    continuation,
    return_rdms=False,
    rdm_basis="AO",
    collect_coefficients=False,
):
    """Return a PySCF gradient scanner backed by a continuation object."""

    class Base:
        converged = True

    class Scanner(lib.GradScanner):
        def __init__(self):
            self.mol = mol
            self.base = Base()
            self.coefficients = []
            self.rdm_results = []

        def __call__(self, mol):
            self.mol = mol
            result = continuation.get_en_with_grad(
                mol,
                nroots=1,
                return_coefficients=collect_coefficients,
                return_rdms=return_rdms,
                rdm_basis=rdm_basis,
            )
            if collect_coefficients:
                coefficients, energies, gradients = result[:3]
                self.coefficients.append(coefficients)
            else:
                energies, gradients = result[:2]
            if return_rdms:
                self.rdm_results.append(result[-1])
            return energies[0], gradients[0]

    return Scanner()


def get_trajectory(
    init_mol,
    continuation,
    dt=10.0,
    steps=10,
    init_veloc=None,
    trajectory_output=None,
    data_output=None,
    return_rdms=False,
    rdm_basis="AO",
    rdm_output=None,
    coefficients_output=None,
):
    """Compute a ground-state MD trajectory from a continuation object."""
    trajectory = np.zeros((steps, init_mol.natm, 3))
    num_threads = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED).Get_size()
    num_threads *= int(os.getenv("OMP_NUM_THREADS", "1"))

    if rank == 0:
        with threadpool_limits(limits=num_threads):
            frames = []
            scanner = get_scanner(
                init_mol,
                continuation,
                return_rdms=return_rdms,
                rdm_basis=rdm_basis,
                collect_coefficients=coefficients_output is not None,
            )
            integrator = md.NVE(
                scanner,
                dt=dt,
                steps=steps,
                veloc=init_veloc,
                incore_anyway=True,
                frames=frames,
                trajectory_output=trajectory_output,
                data_output=data_output,
                verbose=0,
            )
            integrator.run()
            trajectory = np.asarray([frame.coord for frame in frames])
            if rdm_output is not None:
                save_rdm_trajectory(rdm_output, scanner.rdm_results)
            if coefficients_output is not None:
                np.save(coefficients_output, np.asarray(scanner.coefficients))

    trajectory = MPI.COMM_WORLD.bcast(trajectory, root=0)
    return trajectory


def _checkpoint_path(iteration, model_dir):
    return Path(model_dir) / f"continuation_object-{iteration}.pkl"


def _latest_checkpoint(model_dir):
    files = list(Path(model_dir).glob("continuation_object-*.pkl"))
    if not files:
        return None
    return max(files, key=lambda path: int(path.stem.rsplit("-", 1)[1]))


def _save_training_state(continuation, iteration, model_dir, geometries, times):
    continuation.save(_checkpoint_path(iteration, model_dir))
    np.save("trn_geometries.npy", geometries)
    np.savetxt("trn_times.txt", times, fmt="%.18e")


def _ground_energies(continuation, init_mol, trajectory):
    return np.asarray([
        continuation.get_en(
            init_mol.copy().set_geom_(geometry, unit="Bohr"), nroots=1
        )[0][0]
        for geometry in trajectory
    ])


def _trajectory_iteration(
    continuation,
    init_mol,
    iteration,
    steps,
    dt,
    return_rdms=False,
    rdm_basis="AO",
    save_coefficients=False,
):
    trajectory_file = Path(f"traj_EVCont_{iteration}.npy")
    rdm_file = Path(f"rdms_EVCont_{iteration}.npz")
    coefficients_file = Path(f"coefficients_EVCont_{iteration}.npy")
    outputs_exist = (
        (not return_rdms or rdm_file.exists())
        and (not save_coefficients or coefficients_file.exists())
    )
    if trajectory_file.exists() and outputs_exist:
        return np.load(trajectory_file)

    trajectory_handle = energy_handle = None
    if rank == 0:
        trajectory_handle = open(f"traj_EVCont_{iteration}.xyz", "w")
        energy_handle = open(f"ens_EVCont_{iteration}.xyz", "w")
    try:
        trajectory = get_trajectory(
            init_mol.copy(),
            continuation,
            steps=steps,
            dt=dt,
            trajectory_output=trajectory_handle,
            data_output=energy_handle,
            return_rdms=return_rdms,
            rdm_basis=rdm_basis,
            rdm_output=rdm_file if return_rdms else None,
            coefficients_output=coefficients_file if save_coefficients else None,
        )
    finally:
        if rank == 0:
            trajectory_handle.close()
            energy_handle.close()
    if rank == 0:
        np.save(trajectory_file, trajectory)
    return trajectory


def _trajectory_energies(iteration, trajectory):
    if rank == 0:
        data = np.atleast_2d(np.genfromtxt(f"ens_EVCont_{iteration}.xyz"))
        energies = np.ascontiguousarray(data[:, 1])
    else:
        energies = np.zeros(len(trajectory))
    return MPI.COMM_WORLD.bcast(energies, root=0)


def _converged(iteration, threshold, nconv):
    if iteration < nconv:
        return False
    errors = [
        np.max(np.atleast_1d(np.loadtxt(f"en_diff_{i}.txt")))
        for i in range(iteration - nconv + 1, iteration + 1)
    ]
    return all(error <= threshold for error in errors)


def _prune(continuation, init_mol, trajectory, energies, checkpoint, threshold):
    keep = np.ones(continuation.overlap.shape[0], dtype=bool)
    if rank == 0:
        for index in range(len(keep)):
            trial_keep = keep.copy()
            trial_keep[index] = False
            if not np.any(trial_keep):
                continue
            trial = type(continuation).load(checkpoint)
            trial.prune_datapoints(np.flatnonzero(trial_keep))
            if np.all(abs(_ground_energies(trial, init_mol, trajectory) - energies) < threshold):
                keep = trial_keep
    keep = MPI.COMM_WORLD.bcast(keep, root=0)
    continuation.prune_datapoints(np.flatnonzero(keep))
    return keep


def converge_EVCont_MD(
    EVCont_obj,
    init_mol,
    steps=100,
    dt=1,
    convergence_thresh=1.0e-3,
    nconv=2,
    max_iter=100,
    prune_irrelevant_data=False, # Might not work with current updates
    data_addition="weighted_highest_peak_ham",
    learning_exponent=0.5,
    restart=True,
    model_dir="iterative-models",
    return_rdms=False,
    rdm_basis="AO",
    save_coefficients=False,
):
    """Active-learn a ground-state MD trajectory with model checkpoints."""
    model_dir = Path(model_dir)
    if rank == 0:
        model_dir.mkdir(parents=True, exist_ok=True)
    MPI.COMM_WORLD.Barrier()

    checkpoint = _latest_checkpoint(model_dir) if restart else None
    if checkpoint is None:
        iteration = 0
        trn_times = [0]
        EVCont_obj.append_to_rdms(init_mol.copy())
        trn_geometries = np.asarray([init_mol.atom_coords()])
        if rank == 0:
            _save_training_state(
                EVCont_obj, iteration, model_dir, trn_geometries, trn_times
            )
        MPI.COMM_WORLD.Barrier()
    else:
        iteration = int(checkpoint.stem.rsplit("-", 1)[1])
        EVCont_obj = type(EVCont_obj).load(checkpoint)
        trn_geometries = np.load("trn_geometries.npy")
        trn_times = list(
            np.atleast_1d(np.loadtxt("trn_times.txt", dtype=int))
        )

    trajectory = None
    while iteration < max_iter:
        trajectory = _trajectory_iteration(
            EVCont_obj,
            init_mol,
            iteration,
            steps,
            dt,
            return_rdms=return_rdms,
            rdm_basis=rdm_basis,
            save_coefficients=save_coefficients,
        )
        updated_ens = _trajectory_energies(iteration, trajectory)

        if iteration == 0:
            en_diff = np.full_like(updated_ens, np.inf)
        else:
            previous = type(EVCont_obj).load(
                _checkpoint_path(iteration - 1, model_dir)
            )
            reference_ens = _ground_energies(previous, init_mol, trajectory)
            en_diff = abs(reference_ens - updated_ens)

        if rank == 0:
            np.savetxt(f"en_diff_{iteration}.txt", en_diff)
        MPI.COMM_WORLD.Barrier()

        if prune_irrelevant_data and len(trn_geometries) > 1:
            keep = _prune(
                EVCont_obj,
                init_mol,
                trajectory,
                updated_ens,
                _checkpoint_path(iteration, model_dir),
                convergence_thresh,
            )
            trn_geometries = trn_geometries[keep]
            if rank == 0:
                _save_training_state(
                    EVCont_obj, iteration, model_dir, trn_geometries, trn_times
                )
            MPI.COMM_WORLD.Barrier()
        if iteration and _converged(
            iteration, convergence_thresh, nconv
        ):
            break
        if iteration + 1 >= max_iter:
            break

        if data_addition.endswith("_ham"):
            distances, _ = hamiltonian_similarity(
                init_mol,
                trajectory,
                trn_geometries,
                basis_getter=getattr(EVCont_obj, "get_abstract_basis", None),
            )
        else:
            distances = None

        trn_time = select_active_learning_geometry(
            distances,
            method=data_addition,
            exponent=learning_exponent,
            en_diff=en_diff,
            convergence_thresh=convergence_thresh,
            trajectory=trajectory,
            trn_geometries=trn_geometries,
        )
        new_geometry = trajectory[trn_time]
        EVCont_obj.append_to_rdms(
            init_mol.copy().set_geom_(new_geometry, unit="Bohr")
        )
        trn_geometries = np.concatenate((trn_geometries, [new_geometry]))
        trn_times.append(int(trn_time))
        iteration += 1

        if rank == 0:
            _save_training_state(
                EVCont_obj, iteration, model_dir, trn_geometries, trn_times
            )
        MPI.COMM_WORLD.Barrier()

    return trajectory
