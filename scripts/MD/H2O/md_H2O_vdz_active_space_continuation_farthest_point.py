from pyscf import gto, md

from pyscf.mcscf.casci import CASCI

import numpy as np

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad
from EVCont.electron_integral_utils import get_basis, get_integrals, transform_integrals
from EVCont.MD_utils import converge_EVCont_MD
from EVCont.converge_dmrg import converge_dmrg

from EVCont.CASCI_EVCont import CASCI_EVCont_obj

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad

from EVCont.ab_initio_eigenvector_continuation import approximate_ground_state_OAO


from mpi4py import MPI


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
    CASCI_EVCont_obj(ncas, neleca),
    init_mol,
    steps=steps,
    dt=dt,
    prune_irrelevant_data=False,
    data_addition="farthest_point",
)
