from pyscf import gto

import numpy as np

from evcont.MD_utils import converge_EVCont_MD
from evcont.converge_dmrg import converge_dmrg


from evcont.DMRG_EVCont import (
    DMRG_EVCont_obj,
)


from mpi4py import MPI

"""
Runs the MD simulation for a water molecule in the 6-31G basis with a continuation from MPS.
"""


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


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", geometry[0]), ("H", geometry[1]), ("O", geometry[2])],
        basis="6-31G",
        symmetry=False,
        unit="Bohr",
    )

    return mol


a_to_bohr = 1.8897259886

stretch_factor = 1.2

init_geometry = (
    a_to_bohr
    * stretch_factor
    * np.array(
        [
            [0.0, 0.795, -0.454],
            [0.0, -0.795, -0.454],
            [0.0, 0.0, 0.113],
        ]
    )
)


mol = get_mol(init_geometry)


steps = 300
dt = 5


init_mol = mol.copy()


converge_EVCont_MD(
    DMRG_EVCont_obj(dmrg_converge_fun=dmrg_converge_fun),
    init_mol,
    steps=steps,
    dt=dt,
    prune_irrelevant_data=False,
    data_addition="farthest_point_ham",
)
