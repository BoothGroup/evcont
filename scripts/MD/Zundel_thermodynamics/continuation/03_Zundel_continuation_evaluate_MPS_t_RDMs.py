from pyscf import gto


import numpy as np

import sys


from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from evcont.electron_integral_utils import (
    compress_electron_exchange_symmetry,
)


from mpi4py import MPI

import shutil

import glob

rank = MPI.COMM_WORLD.rank
n_ranks = MPI.COMM_WORLD.Get_size()


"""
Evaluates 1- and 2-t-RDMS between different MPS in the OAO basis for two training
configurations (with ids specified by arguments x and y), this assumes that training
configurations are specified in "../trn_geometries.txt" as produced by the script
"01_Zundel_continuation_trn_set_generation.py", and that MPSs have been constructed by
the script "02_Zundel_continuation_run_DMRG.py", where the outputs of the i-th DMRG run
was stored in the directory "../DMRG_i"
"""

x = int(sys.argv[1])
y = int(sys.argv[2])

bohr_to_angstrom = 1 / 1.8897259886


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

    geometry = bohr_to_angstrom * trajectory[int(time)]

    mol_a = get_mol(geometry)

    time, file, _ = lines[y].split()
    trajectory = np.load(file)

    geometry = bohr_to_angstrom * trajectory[int(time)]

    mol_b = get_mol(geometry)


mps_solver = DMRGDriver(
    symm_type=SymmetryTypes.SU2,
    mpi=(MPI.COMM_WORLD.size > 1),
    stack_mem=5 << 30,
)

mps_solver.initialize_system(mol_a.nao, n_elec=np.sum(mol_a.nelec), spin=mol_a.spin)


for fl in glob.glob(f"../DMRG_{x}/nodex/*"):
    shutil.copy(fl, mps_solver.scratch)
for fl in glob.glob(f"../DMRG_{y}/nodex/*"):
    shutil.copy(fl, mps_solver.scratch)


bra = mps_solver.load_mps("MPS_{}".format(x))

ket = mps_solver.load_mps("MPS_{}".format(y))


ovlp = (
    np.array(mps_solver.expectation(bra, mps_solver.get_identity_mpo(), ket)) / n_ranks
)
o_RDM = np.array(mps_solver.get_1pdm(ket, bra=bra))
t_RDM = np.array(np.transpose(mps_solver.get_2pdm(ket, bra=bra), (0, 3, 1, 2)))


np.save("ovlp.npy", ovlp)
np.save("one_rdm.npy", o_RDM)
np.save("two_rdm.npy", compress_electron_exchange_symmetry(t_RDM))

shutil.rmtree(mps_solver.scratch)
