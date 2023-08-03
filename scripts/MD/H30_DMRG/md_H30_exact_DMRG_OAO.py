from pyscf import md, gto, scf, mcscf, lo, ao2mo, fci, lib, grad

import numpy as np

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad
from EVCont.MD_utils import get_trajectory

from EVCont.CustomDMRGCI import CustomDMRGCI

import os

from mpi4py import MPI


rank = MPI.COMM_WORLD.rank


norb = nelec = 30


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", pos) for pos in geometry],
        basis="sto-6g",
        symmetry=True,
        unit="Bohr",
    )

    return mol


init_dist = 1.9

steps = 100
dt = 20

mol = get_mol(np.array([[0, 0, init_dist * i] for i in range(nelec)]))
init_mol = mol.copy()
solver = CustomDMRGCI(mol, 30, 30, "OAO")
solver.converged = True

scanner_fun = solver.nuc_grad_method().as_scanner()


frames = []
scanner_fun.mol = init_mol.copy()

if rank == 0:
    fl = "DMRG_trajectory.xyz"
else:
    fl = None

myintegrator = md.NVE(
    scanner_fun,
    dt=dt,
    steps=steps,
    incore_anyway=True,
    frames=frames,
    trajectory_output=fl,
)
myintegrator.run()

if rank == 0:
    np.save("traj_DMRG.npy", np.array([frame.coord for frame in frames]))
