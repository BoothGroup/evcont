import numpy as np

from pyscf import gto

from pyscf.scf import hf

from EVCont.electron_integral_utils import get_basis, get_integrals

from EVCont.ab_initio_eigenvector_continuation import approximate_ground_state

import sys

"""
Evaluates the dipole moment and the electron density for the MD trajectories from a
previous eigenvector continuation with a specified number of training points
"""

# Number of training points for which the quantities are evaluated
num_training_points = int(sys.argv[1])


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


overlap = np.load("overlap.npy")
one_rdm = np.load("one_rdm.npy")
two_rdm = np.load("two_rdm.npy")

num_points = overlap.shape[0]

trajectory = np.load("traj_EVCont_{}.npy".format(num_training_points - 1))


open("dipole_moment_{}.txt".format(num_training_points - 1), "w")
open("atom_charges_{}.txt".format(num_training_points - 1), "w")
for i, pos in enumerate(trajectory):
    print(i)
    mol = get_mol(pos)

    basis = get_basis(mol)
    h1, h2 = get_integrals(mol, basis)
    red_one_rdm = one_rdm[:num_training_points, :num_training_points, :, :]
    en, vec = approximate_ground_state(
        h1,
        h2,
        (red_one_rdm),
        (two_rdm[:num_training_points, :num_training_points, :, :, :, :]),
        (overlap[:num_training_points, :num_training_points]),
        hermitian=True,
    )

    predicted_one_rdm = np.sum(
        red_one_rdm * vec.reshape((-1, 1, 1)) * vec.reshape((-1, 1, 1, 1)),
        axis=(0, 1),
    )

    predicted_one_rdm = basis.dot(predicted_one_rdm).dot(basis.T)

    dipole_moment = hf.dip_moment(mol, predicted_one_rdm)
    atomic_charges = hf.mulliken_meta(mol, predicted_one_rdm)[1]

    with open("dipole_moment_{}.txt".format(num_training_points - 1), "a") as fl:
        for el in dipole_moment:
            fl.write("{}  ".format(el))
        fl.write("\n")

    with open("atom_charges_{}.txt".format(num_training_points - 1), "a") as fl:
        for el in atomic_charges:
            fl.write("{}  ".format(el))
        fl.write("\n")
