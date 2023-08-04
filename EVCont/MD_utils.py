from pyscf import md, scf, lib, grad

import numpy as np

from .ab_initio_gradients_loewdin import get_energy_with_grad

from .electron_integral_utils import get_integrals, get_basis


from mpi4py import MPI

import os

from threadpoolctl import threadpool_limits

rank = MPI.COMM_WORLD.Get_rank()


def get_scanner(mol, one_rdm, two_rdm, overlap, hermitian=True):
    class Base:
        converged = True

    class Scanner(lib.GradScanner):
        def __init__(self):
            self.mol = mol
            self.base = Base()
            # self.converged = True

        def __call__(self, mol):
            self.mol = mol
            if one_rdm is not None and two_rdm is not None and overlap is not None:
                return get_energy_with_grad(
                    mol, one_rdm, two_rdm, overlap, hermitian=hermitian
                )
            else:
                return mol.energy_nuc(), grad.RHF(scf.RHF(mol)).grad_nuc()

    return Scanner()


def get_trajectory(
    init_mol,
    overlap,
    one_rdm,
    two_rdm,
    dt=100.0,
    steps=10,
    init_veloc=None,
    hermitian=True,
    trajectory_output=None,
):
    trajectory = np.zeros((steps, len(init_mol.atom), 3))
    num_threads = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED).Get_size()
    if rank == 0:
        with threadpool_limits(limits=num_threads):
            scanner_fun = get_scanner(
                init_mol, one_rdm, two_rdm, overlap, hermitian=hermitian
            )

            frames = []
            myintegrator = md.NVE(
                scanner_fun,
                dt=dt,
                steps=steps,
                veloc=init_veloc,
                incore_anyway=True,
                frames=frames,
                trajectory_output=trajectory_output,
            )
            myintegrator.run()
            trajectory = np.array([frame.coord for frame in frames])

    MPI.COMM_WORLD.Bcast(trajectory, root=0)

    return trajectory
