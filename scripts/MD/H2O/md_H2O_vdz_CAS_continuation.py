from pyscf import gto


import numpy as np

from EVCont.MD_utils import converge_EVCont_MD

from EVCont.CASCI_EVCont import CAS_EVCont_obj


from mpi4py import MPI


"""
Runs the MD simulation for a water molecule in the cc-pVDZ with a continuation from
CAS states.
"""


ncas = 8  # active space orbitals
neleca = 4  # active space electrons


rank = MPI.COMM_WORLD.rank

ncas = 8
neleca = 4


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", geometry[0]), ("H", geometry[1]), ("O", geometry[2])],
        basis="cc-pVDZ",
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
    CAS_EVCont_obj(ncas, neleca),
    init_mol,
    steps=steps,
    dt=dt,
    prune_irrelevant_data=False,
    data_addition="farthest_point_ham",
)
