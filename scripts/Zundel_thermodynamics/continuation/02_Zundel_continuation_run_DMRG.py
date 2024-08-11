from pyscf import gto


import numpy as np

import sys

from evcont.electron_integral_utils import get_basis, get_integrals


from evcont.converge_dmrg import converge_dmrg


from mpi4py import MPI

"""
Runs DMRG in the OAO basis for a single training configuration (with id specified
by argument x), this assumes that training configurations are specified in
"../trn_geometries.txt" as produced by the script "02_Zundel_continuation_trn_set_generation.py".
"""

x = int(sys.argv[1])

bohr_to_angstrom = 1 / 1.8897259886


def dmrg_converge_fun(h1, h2, nelec, tag):
    MPI.COMM_WORLD.Bcast(h1, root=0)

    h2_slice = np.empty((h2.shape[2], h2.shape[3]))

    for i in range(h2.shape[0]):
        for j in range(h2.shape[1]):
            np.copyto(h2_slice, h2[i, j, :, :])
            MPI.COMM_WORLD.Bcast(h2_slice, root=0)
            np.copyto(h2[i, j, :, :], h2_slice)

    return converge_dmrg(
        h1, h2, nelec, tag, tolerance=1.0e-3, mpi=MPI.COMM_WORLD.size > 1, mem=10
    )


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
        unit="Angstrom",
        charge=1,
    )

    return mol


with open("../trn_geometries.txt", "r") as fl:
    lines = fl.readlines()

    time, file, _ = lines[x].split()
    trajectory = np.load(file)
    if len(trajectory.shape) == 3:
        geometry = bohr_to_angstrom * trajectory[int(time)]
    else:
        geometry = trajectory

mol = get_mol(geometry)


h1, h2 = get_integrals(mol, get_basis(mol, basis_type="OAO"))

MPI.COMM_WORLD.Bcast(h1, root=0)

h2_slice = np.empty((h2.shape[2], h2.shape[3]))

for i in range(h2.shape[0]):
    for j in range(h2.shape[1]):
        np.copyto(h2_slice, h2[i, j, :, :])
        MPI.COMM_WORLD.Bcast(h2_slice, root=0)
        np.copyto(h2[i, j, :, :], h2_slice)


dmrg_converge_fun(h1, h2, mol.nelec, "MPS_{}".format(x))
