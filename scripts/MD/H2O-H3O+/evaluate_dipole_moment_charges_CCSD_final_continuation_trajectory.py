import numpy as np

from pyscf import gto

from pyscf.scf import hf

from evcont.electron_integral_utils import get_basis, get_integrals

"""
Evaluate the PES, dipole moments, and atomic Mulliken charges along the final continuation
trajectory with CCSD.
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


trajectory_final = np.load("traj_EVCont_{}.npy".format(83))


open("dipole_moment_CCSD_final_trajectory.txt", "w").close()
open("atom_charges_CCSD_final_trajectory.txt", "w").close()
open("energies_CCSD_final_trajectory.txt", "w").close()


for i, pos in enumerate(trajectory_final):
    print(i)
    mol = get_mol(pos)

    basis = get_basis(mol)
    h1, h2 = get_integrals(mol, basis)

    ccsd_object = mol.CCSD()

    ccsd_object.kernel()

    en = ccsd_object.e_tot

    converged = ccsd_object.converged

    one_rdm_mo = ccsd_object.make_rdm1()

    basis = ccsd_object.mo_coeff
    one_rdm = basis.dot(one_rdm_mo).dot(basis.T)

    dipole_moment = hf.dip_moment(mol, one_rdm)
    atomic_charges = hf.mulliken_meta(mol, one_rdm)[1]

    with open("dipole_moment_CCSD_final_trajectory.txt", "a") as fl:
        for el in dipole_moment:
            fl.write("{}  ".format(el))
        fl.write("\n")
    with open("atom_charges_CCSD_final_trajectory.txt", "a") as fl:
        for el in atomic_charges:
            fl.write("{}  ".format(el))
        fl.write("\n")
    with open("energies_CCSD_final_trajectory.txt", "a") as fl:
        if converged:
            fl.write("{}\n".format(en))
        else:
            fl.write("{}\n".format(np.nan))
