from pyscf import gto, md

import numpy as np

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad
from EVCont.electron_integral_utils import get_basis, get_integrals, transform_integrals
from EVCont.MD_utils import converge_EVCont_MD
from EVCont.converge_dmrg import converge_dmrg

from EVCont.DMRG_EVCont import (
    append_to_rdms_rerun,
    append_to_rdms_orbital_rotation,
    append_to_rdms_OAO_basis,
)

from EVCont.ab_initio_eigenvector_continuation import approximate_ground_state_OAO

from EVCont.DMRG_EVCont import (
    DMRG_EVCont_obj,
)

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad

from mpi4py import MPI

import os

rank = MPI.COMM_WORLD.rank


def dmrg_converge_fun(h1, h2, nelec, tag):
    MPI.COMM_WORLD.Bcast(h1, root=0)

    h2_slice = np.empty((h2.shape[2], h2.shape[3]))

    for i in range(h2.shape[0]):
        for j in range(h2.shape[1]):
            np.copyto(h2_slice, h2[i, j, :, :])
            MPI.COMM_WORLD.Bcast(h2_slice, root=0)
            np.copyto(h2[i, j, :, :], h2_slice)

    return converge_dmrg(
        h1, h2, nelec, tag, tolerance=1.0e-3, mpi=MPI.COMM_WORLD.size > 1
    )


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


overlap = np.load("overlap.npy")
one_rdm = np.load("one_rdm.npy")
two_rdm = np.load("two_rdm.npy")

no_data = overlap.shape[0]

init_dist = 1.9

steps = 300
dt = 5


mol = get_mol(np.array([[0, 0, init_dist * i] for i in range(nelec)]))

trajectory = np.load("traj_DMRG.npy")

open("continued_energies.txt", "w").close()

for pos in trajectory:
    inner_mol = mol.copy().set_geom_(pos)
    with open("continued_energies.txt", "a") as fl:
        for i in range(no_data):
            en = approximate_ground_state_OAO(
                inner_mol,
                one_rdm[: (i + 1), : (i + 1)],
                two_rdm[: (i + 1), : (i + 1)],
                overlap[: (i + 1), : (i + 1)],
            )[0]
            fl.write("{} ".format(en))
        fl.write("\n")
