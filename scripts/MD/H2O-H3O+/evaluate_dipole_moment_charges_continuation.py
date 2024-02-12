import numpy as np

from pyscf import gto

from pyscf.scf import hf

from evcont.electron_integral_utils import get_basis, get_integrals

from evcont.ab_initio_eigenvector_continuation import approximate_ground_state

import sys


"""
Evaluate the PES, dipole moments, and atomic Mulliken charges predicted by the eigenvector
continuation with a specified number of training points along the trajectories obtained a)
with the restricted number of training points and b) the final continuation trajectory
with all training points.
"""

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


red_one_rdm = one_rdm[:num_training_points, :num_training_points, :, :]
red_two_rdm = two_rdm[:num_training_points, :num_training_points, :, :, :, :]
red_overlap = overlap[:num_training_points, :num_training_points]

trajectory = np.load("traj_EVCont_{}.npy".format(num_training_points - 1))
trajectory_final = np.load("traj_EVCont_{}.npy".format(83))

open("dipole_moment_simulated_trajectory_{}.txt".format(num_training_points - 1), "w")
open("atom_charges_simulated_trajectory_{}.txt".format(num_training_points - 1), "w")

open("dipole_moment_final_trajectory_{}.txt".format(num_training_points - 1), "w")
open("atom_charges_final_trajectory_{}.txt".format(num_training_points - 1), "w")
open("energies_final_trajectory_{}.txt".format(num_training_points - 1), "w")

for i, pos in enumerate(trajectory):
    print(i)
    mol = get_mol(pos)

    basis = get_basis(mol)
    h1, h2 = get_integrals(mol, basis)

    en, vec = approximate_ground_state(
        h1,
        h2,
        red_one_rdm,
        red_two_rdm,
        red_overlap,
        hermitian=True,
    )

    predicted_one_rdm = np.sum(
        red_one_rdm * vec.reshape((-1, 1, 1)) * vec.reshape((-1, 1, 1, 1)),
        axis=(0, 1),
    )

    predicted_one_rdm = basis.dot(predicted_one_rdm).dot(basis.T)

    dipole_moment = hf.dip_moment(mol, predicted_one_rdm)
    atomic_charges = hf.mulliken_meta(mol, predicted_one_rdm)[1]

    with open(
        "dipole_moment_simulated_trajectory_{}.txt".format(num_training_points - 1), "a"
    ) as fl:
        for el in dipole_moment:
            fl.write("{}  ".format(el))
        fl.write("\n")

    with open(
        "atom_charges_simulated_trajectory_{}.txt".format(num_training_points - 1), "a"
    ) as fl:
        for el in atomic_charges:
            fl.write("{}  ".format(el))
        fl.write("\n")


for i, pos in enumerate(trajectory_final):
    print(i)
    mol = get_mol(pos)

    basis = get_basis(mol)
    h1, h2 = get_integrals(mol, basis)

    en, vec = approximate_ground_state(
        h1,
        h2,
        red_one_rdm,
        red_two_rdm,
        red_overlap,
        hermitian=True,
    )

    predicted_one_rdm = np.sum(
        red_one_rdm * vec.reshape((-1, 1, 1)) * vec.reshape((-1, 1, 1, 1)),
        axis=(0, 1),
    )

    predicted_one_rdm_AO = basis.dot(predicted_one_rdm).dot(basis.T)

    dipole_moment = hf.dip_moment(mol, predicted_one_rdm_AO)
    atomic_charges = hf.mulliken_meta(mol, predicted_one_rdm_AO)[1]

    with open(
        "dipole_moment_final_trajectory_{}.txt".format(num_training_points - 1), "a"
    ) as fl:
        for el in dipole_moment:
            fl.write("{}  ".format(el))
        fl.write("\n")

    with open(
        "atom_charges_final_trajectory_{}.txt".format(num_training_points - 1), "a"
    ) as fl:
        for el in atomic_charges:
            fl.write("{}  ".format(el))
        fl.write("\n")

    with open(
        "energies_final_trajectory_{}.txt".format(num_training_points - 1), "a"
    ) as fl:
        fl.write("{}\n".format(en + mol.energy_nuc()))
