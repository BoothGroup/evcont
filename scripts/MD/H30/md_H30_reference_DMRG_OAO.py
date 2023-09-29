from pyscf import md, gto, scf, lib, grad

import numpy as np

from evcont.ab_initio_gradients_loewdin import get_grad_elec_OAO

from pyblock2.driver.core import DMRGDriver, SymmetryTypes


from evcont.converge_dmrg import converge_dmrg

from evcont.electron_integral_utils import get_basis, get_integrals


from mpi4py import MPI

"""
MD simulation with DMRG in the OAO basis
"""


rank = MPI.COMM_WORLD.rank


norb = nelec = 30


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

mol = get_mol(np.array([[0, 0, init_dist * i] for i in range(nelec)]))


class Base:
    converged = True


class Scanner(lib.GradScanner):
    def __init__(self):
        self.base = Base()

    def __call__(self, mol):
        self.mol = mol

        basis = get_basis(mol)

        h1, h2 = get_integrals(mol, basis)

        state, en = default_solver_fun(
            h1,
            h2,
            mol.nelec,
        )

        mps_solver = DMRGDriver(
            symm_type=SymmetryTypes.SU2,
            mpi=(MPI.COMM_WORLD.size > 1),
            stack_mem=20 << 30,
        )
        mps_solver.initialize_system(
            mol.nao, n_elec=np.sum(mol.nelec), spin=(mol.nelec[0] - mol.nelec[1])
        )

        one_rdm = np.array(mps_solver.get_1pdm(state, bra=state))
        two_rdm = np.array(
            np.transpose(mps_solver.get_2pdm(state, bra=state), (0, 3, 1, 2))
        )

        print(en)

        return mol.energy_nuc() + en, grad.RHF(
            scf.RHF(mol)
        ).grad_nuc() + get_grad_elec_OAO(mol, one_rdm, two_rdm, ao_mo_trafo=basis)


init_mol = mol.copy()

scanner_fun = Scanner()

frames = []
scanner_fun.mol = init_mol.copy()

fl = "DMRG_trajectory_{}.xyz".format(rank)
fl_en = "DMRG_energies_{}.xyz".format(rank)

myintegrator = md.NVE(
    scanner_fun,
    steps=steps,
    dt=dt,
    incore_anyway=True,
    frames=frames,
    trajectory_output=fl,
    energy_output=fl_en,
)
myintegrator.run()

if rank == 0:
    np.save("traj_DMRG.npy", np.array([frame.coord for frame in frames]))
