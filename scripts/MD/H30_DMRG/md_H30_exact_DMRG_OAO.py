from pyscf import md, gto, scf, mcscf, lo, ao2mo, fci, lib, grad

import numpy as np

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad
from EVCont.MD_utils import get_trajectory

from EVCont.CustomDMRGCI import CustomDMRGCI

from EVCont.converge_dmrg import converge_dmrg


import os

from mpi4py import MPI


rank = MPI.COMM_WORLD.rank


norb = nelec = 30


def default_solver_fun(h1, h2, nelec):
    return converge_dmrg(
        h1,
        h2,
        nelec,
        "MPS",
        tolerance=1.0e-5,
    )


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", pos) for pos in geometry],
        basis="sto-6g",
        symmetry=False,
        unit="Bohr",
    )

    return mol


init_dist = 1.9

steps = 300
dt = 5

mol = get_mol(np.array([[0, 0, init_dist * i] for i in range(nelec)]))
init_mol = mol.copy()
solver = CustomDMRGCI(
    mol, 30, 30, "OAO", converge_dmrg_fun=default_solver_fun, reorder_orbitals=False
)
solver.converged = True

scanner_fun = solver.nuc_grad_method().as_scanner()


frames = []
scanner_fun.mol = init_mol.copy()

fl = "DMRG_trajectory_{}.xyz".format(rank)
fl_en = "DMRG_energies_{}.xyz".format(rank)

myintegrator = md.NVE(
    scanner_fun,
    steps=steps,
    dt=dt,
    incore_anyway=True,
    frames=frames,
    trajectory_output=fl,
    energy_output=fl_en,
)
myintegrator.run()

if rank == 0:
    np.save("traj_DMRG.npy", np.array([frame.coord for frame in frames]))
