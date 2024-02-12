from pyscf import gto

import numpy as np

from evcont.electron_integral_utils import get_basis, get_integrals

from evcont.ab_initio_eigenvector_continuation import approximate_ground_state_OAO


"""
Evaluate energetics for training points of Zundel cation MD simulation with DMRG,
eigenvector continuation, HF, DFT, and CCSD.
"""


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[
            ("O", geometry[0]),
            ("H", geometry[1]),
            ("H", geometry[2]),
            ("H", geometry[3]),
            ("O", geometry[4]),
            ("H", geometry[5]),
            ("H", geometry[6]),
        ],
        basis="6-31G",
        symmetry=False,
        unit="Bohr",
        charge=1,
    )

    return mol


one_rdm = np.load("one_rdm.npy")
overlap = np.load("overlap.npy")
two_rdm = np.load("two_rdm.npy")

# Restore training geometries
trn_times = list(np.atleast_1d(np.loadtxt("trn_times.txt").astype(int)))

trajs = [np.load("traj_EVCont_{}.npy".format(i)) for i in range(len(trn_times))]

trn_geometries = [trajs[0][0]] + [
    trajs[k][trn_times[k + 1]] for k in range(len(trajs) - 1)
]

open("energies_trn_points_DMRG.txt", "w").close()
open("energies_trn_points_full_continuation.txt", "w").close()
open("energies_trn_points_CCSD.txt", "w").close()
open("energies_trn_points_DFT.txt", "w").close()
open("energies_trn_points_HF.txt", "w").close()

for i, trn_geometry in enumerate(trn_geometries):
    np.save("trn_geometry_{}".format(i), trn_geometry)
    mol = get_mol(trn_geometry)
    h1, h2 = get_integrals(mol, get_basis(mol))

    en_DMRG = np.sum(one_rdm[i, i, :, :] * h1) + 0.5 * np.sum(
        two_rdm[i, i, :, :, :, :] * h2
    )

    with open("energies_trn_points_DMRG.txt", "a") as f:
        f.write("{}\n".format(en_DMRG + mol.energy_nuc()))

    en_continuation, _ = approximate_ground_state_OAO(mol, one_rdm, two_rdm, overlap)

    with open("energies_trn_points_full_continuation.txt", "a") as f:
        f.write("{}\n".format(en_continuation))

    try:
        ccsd_object = mol.CCSD()
        ccsd_object.kernel()
        converged = ccsd_object.converged
        if not converged:
            en_converged = np.nan
        else:
            en_converged = ccsd_object.e_tot
        en = ccsd_object.e_tot
    except:
        en = np.nan
        en_converged = np.nan

    with open("energies_trn_points_CCSD.txt", "a") as f:
        f.write("{}  {}\n".format(en_converged, en))

    try:
        dft = mol.KS()
        dft.xc = "b3lyp"
        dft.kernel()
        converged = dft.converged
        if not converged:
            en_converged = np.nan
        else:
            en_converged = dft.e_tot
        en = dft.e_tot
    except:
        en = np.nan
        en_converged = np.nan

    with open("energies_trn_points_DFT.txt", "a") as f:
        f.write("{}  {}\n".format(en_converged, en))

    try:
        mf = mol.RHF()
        mf.kernel()
        converged = dft.converged
        if not converged:
            en_converged = np.nan
        else:
            en_converged = mf.e_tot
        en = mf.e_tot
    except:
        en = np.nan
        en_converged = np.nan

    with open("energies_trn_points_HF.txt", "a") as f:
        f.write("{}  {}\n".format(en_converged, en))
