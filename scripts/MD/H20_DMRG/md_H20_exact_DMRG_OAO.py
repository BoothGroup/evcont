from pyscf import md, gto, scf, mcscf, lo, ao2mo, fci, lib, grad

import numpy as np

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad
from EVCont.MD_utils import get_trajectory

from EVCont.CustomDMRGCI import CustomDMRGCI

from pyscf import dmrgscf

import os

norb = nelec = 20


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

steps = 50
dt = 25

mol = get_mol(np.array([[0, 0, init_dist * i] for i in range(nelec)]))
init_mol = mol.copy()

solver = CustomDMRGCI(mol, 20, 20, "OAO")
solver.converged = True

scanner_fun = solver.nuc_grad_method().as_scanner()


frames = []
scanner_fun.mol = init_mol.copy()
myintegrator = md.NVE(
    scanner_fun, dt=dt, steps=steps, incore_anyway=True, frames=frames
)
myintegrator.run()


np.save("traj_exact.npy", np.array([frame.coord for frame in frames]))
