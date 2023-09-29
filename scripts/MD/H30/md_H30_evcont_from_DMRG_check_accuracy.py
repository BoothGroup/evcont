from pyscf import gto

import numpy as np

from evcont.electron_integral_utils import get_basis, get_integrals
from evcont.converge_dmrg import converge_dmrg


from evcont.ab_initio_eigenvector_continuation import approximate_ground_state_OAO


from mpi4py import MPI


"""
Evaluate PES (DMRG reference + continuation) along the converged MD trajectory
"""

rank = MPI.COMM_WORLD.rank


def default_solver_fun(h1, h2, nelec):
    return converge_dmrg(
        h1,
        h2,
        nelec,
        "MPS",
        tolerance=1.0e-5,
        restart_tag="MPS",
        bond_dim_schedule=np.round(1.8 ** np.arange(7, 16)).astype(int),
        mem=20,
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


init_dist = 1.9

steps = 300
dt = 5


overlap = np.load("overlap.npy")
one_rdm = np.load("one_rdm.npy")
two_rdm = np.load("two_rdm.npy")

no_data = overlap.shape[0]


mol = get_mol(np.array([[0, 0, init_dist * i] for i in range(nelec)]))

trajectory = np.load("traj_EVCont_{}.npy".format(no_data - 1))

open("DMRG_energies.txt", "w").close()
open("continued_energies.txt", "w").close()

for pos in trajectory:
    inner_mol = mol.copy().set_geom_(pos)
    h1, h2 = get_integrals(inner_mol, get_basis(inner_mol))
    state, en = default_solver_fun(
        h1,
        h2,
        inner_mol.nelec,
    )
    with open("DMRG_energies.txt", "a") as fl:
        fl.write("{}\n".format(en + inner_mol.energy_nuc()))

    with open("continued_energies.txt", "a") as fl:
        for i in range(no_data):
            en = approximate_ground_state_OAO(
                inner_mol,
                one_rdm[: i + 1, : i + 1],
                two_rdm[: i + 1, : i + 1],
                overlap[: i + 1, : i + 1],
            )[0]
            fl.write("{}  ".format(en))
        fl.write("\n")
