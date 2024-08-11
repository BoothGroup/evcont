from pyscf import gto


import numpy as np


from evcont.ab_initio_eigenvector_continuation import get_basis, get_integrals


from evcont.ab_initio_eigenvector_continuation import approximate_ground_state_OAO

import sys

from pyscf import cc, dft

from evcont.converge_dmrg import converge_dmrg

"""
Evaluates tes energies for a specified test point from a trajectory specified by an
index, for one particular time index, where the argument "trajectory_prefix" specifies
where these trajectories are located (location of the trajectory is assumed to be in
"../{}_{}/trajectory.npy".format(trajectory_prefix, trajectory_index)). Script assumes that training
configurations are specified in "../trn_geometries.txt" as produced by the script
"01_Zundel_continuation_trn_set_generation.py", that MPSs have been constructed by
the script "02_Zundel_continuation_run_DMRG.py", and that t-RDMs/overlaps have been constructed with
the script "03_Zundel_continuation_evaluate_MPS_t_RDMs.py".
"""

trajectory_index = int(sys.argv[1])
time_index = int(sys.argv[2])
trajectory_prefix = sys.argv[3]


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


def dmrg_converge_fun(h1, h2, nelec, tag):
    return converge_dmrg(
        h1,
        h2,
        nelec,
        tag,
        tolerance=1.0e-3,
        mpi=False,
        mem=10,
        bond_dim_schedule=[20, 40, 80, 160, 320, 640, 1280],
        reorder=True,
    )


init_geometry = np.load("../../init_geometry.npy")

mol = get_mol(init_geometry)

with open("../trn_geometries.txt", "r") as fl:
    lines = fl.readlines()
    no_trn_points = len(lines)

inds_trn = np.tril_indices(no_trn_points)

overlap = np.zeros((no_trn_points, no_trn_points))

for i in range(len(inds_trn[0])):
    overlap[inds_trn[0][i], inds_trn[1][i]] = np.load(
        "../MPS_cross_{}_{}/ovlp.npy".format(inds_trn[0][i], inds_trn[1][i])
    )

overlap[np.triu_indices(no_trn_points)] = (overlap.T)[np.triu_indices(no_trn_points)]

one_rdm = np.zeros((no_trn_points, no_trn_points, mol.nao, mol.nao))

for i in range(len(inds_trn[0])):
    one_rdm[inds_trn[0][i], inds_trn[1][i], :, :] = np.load(
        "../MPS_cross_{}_{}/one_rdm.npy".format(inds_trn[0][i], inds_trn[1][i])
    )

one_rdm[np.triu_indices(no_trn_points)[0], np.triu_indices(no_trn_points)[1], :, :] = (
    np.transpose(one_rdm, [1, 0, 2, 3])
)[np.triu_indices(no_trn_points)[0], np.triu_indices(no_trn_points)[1], :, :]

inds_orb = np.tril_indices(mol.nao * mol.nao)

two_rdm = np.zeros((len(inds_trn[0]), len(inds_orb[0])))

for i in range(len(inds_trn[0])):
    two_rdm[i, :] = np.load(
        "../MPS_cross_{}_{}/two_rdm.npy".format(inds_trn[0][i], inds_trn[1][i])
    )


def get_energy_continuation(mol, no_basis=-1):
    if no_basis == -1:
        approximate_ground_state_OAO(mol, one_rdm, two_rdm, overlap)[0]
    else:
        new_trn_ids = np.tril_indices(no_basis)

        targets = []
        for a in range(len(new_trn_ids[0])):
            for b in range(len(inds_trn[0])):
                if (
                    new_trn_ids[0][a] == inds_trn[0][b]
                    and new_trn_ids[1][a] == inds_trn[1][b]
                ):
                    targets.append(b)
                    break
        print(targets)
        return approximate_ground_state_OAO(
            mol,
            one_rdm[:no_basis, :no_basis, :, :],
            two_rdm[targets, :],
            overlap[:no_basis, :no_basis],
        )[0]


def get_energy_HF(mol):
    mf = mol.RHF().run()

    return mf.e_tot


def get_energy_CCSDT(mol):
    mf = mol.RHF().run()
    mcc = cc.CCSD(mf).run()

    e_tot = mcc.ccsd_t() + mcc.e_tot

    print(mcc.converged)

    return e_tot


def get_energy_CCSD(mol):
    mf = mol.RHF().run()
    mcc = cc.CCSD(mf).run()

    print(mcc.converged)

    return mcc.e_tot


def get_energy_DFT(mol, xc="CAMB3LYP"):
    mf = dft.UKS(mol, xc=xc)

    mf.kernel()

    return mf.e_tot


def get_energy_DMRG(mol):
    h1, h2 = get_integrals(mol, get_basis(mol, "split"))

    return dmrg_converge_fun(h1, h2, mol.nelec, "MPS")[1] + mol.energy_nuc()


bohr_to_angstrom = 1 / 1.8897259886


trajectory = bohr_to_angstrom * np.load(
    "../{}_{}/trajectory.npy".format(trajectory_prefix, trajectory_index)
)


with open("energies.txt", "w") as fl:
    fl.write(
        "HF  CCSD  CCSD(T)  DMRG  DFT (CAMB3LYP)  DFT (PBE)  N=20  N=40  N=60  N=80  N=100\n"
    )

mol = get_mol(trajectory[time_index])

with open("energies.txt", "a") as fl:
    fl.write("{}  ".format(get_energy_HF(mol)))
with open("energies.txt", "a") as fl:
    fl.write("{}  ".format(get_energy_CCSD(mol)))
with open("energies.txt", "a") as fl:
    fl.write("{}  ".format(get_energy_CCSDT(mol)))
with open("energies.txt", "a") as fl:
    fl.write("{}  ".format(get_energy_DMRG(mol)))
with open("energies.txt", "a") as fl:
    fl.write("{}  ".format(get_energy_DFT(mol, xc="CAMB3LYP")))
with open("energies.txt", "a") as fl:
    fl.write("{}  ".format(get_energy_DFT(mol, xc="PBE")))
with open("energies.txt", "a") as fl:
    fl.write("{}  ".format(get_energy_continuation(mol, no_basis=20)))
with open("energies.txt", "a") as fl:
    fl.write("{}  ".format(get_energy_continuation(mol, no_basis=40)))
with open("energies.txt", "a") as fl:
    fl.write("{}  ".format(get_energy_continuation(mol, no_basis=60)))
with open("energies.txt", "a") as fl:
    fl.write("{}  ".format(get_energy_continuation(mol, no_basis=80)))
with open("energies.txt", "a") as fl:
    fl.write("{}\n".format(get_energy_continuation(mol, no_basis=100)))
