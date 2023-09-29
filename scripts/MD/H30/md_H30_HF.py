from pyscf import md, gto

import numpy as np


"""
MD simulation with Hartree-Fock.
"""

norb = nelec = 30


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

mf = init_mol.RHF()

scanner_fun = mf.nuc_grad_method().as_scanner()
frames = []
scanner_fun.mol = init_mol.copy()
myintegrator = md.NVE(
    scanner_fun,
    steps=steps,
    incore_anyway=True,
    frames=frames,
    trajectory_output="HF_trajectory.xyz",
    energy_output="HF_energy.xyz",
    dt=dt,
)
myintegrator.run()


np.save("traj_HF.npy", np.array([frame.coord for frame in frames]))
