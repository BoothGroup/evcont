from pyscf import md, gto, scf, mcscf, lo, ao2mo, fci, lib, grad

import numpy as np

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad
from EVCont.MD_utils import get_trajectory

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


def get_ham(mol):
    loc_coeff = lo.orth_ao(mol, "lowdin", pre_orth_ao=None)

    norb = loc_coeff.shape[0]

    h1 = np.linalg.multi_dot((loc_coeff.T, scf.hf.get_hcore(mol), loc_coeff))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)

    return h1, h2


init_dist = 1.9

steps = 100
dt = 20

mol = get_mol(np.array([[0, 0, init_dist * i] for i in range(nelec)]))
init_mol = mol.copy()

mf = init_mol.RHF()

scanner_fun = mf.nuc_grad_method().as_scanner()
frames = []
scanner_fun.mol = init_mol.copy()
myintegrator = md.NVE(
    scanner_fun,
    dt=dt,
    steps=steps,
    incore_anyway=True,
    frames=frames,
    trajectory_output="HF_trajectory.xyz",
)
myintegrator.run()


np.save("traj_HF.npy", np.array([frame.coord for frame in frames]))
