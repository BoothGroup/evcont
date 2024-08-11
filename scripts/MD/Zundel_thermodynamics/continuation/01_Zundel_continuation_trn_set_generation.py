from pyscf import gto

import glob

import numpy as np

import sys

from evcont.electron_integral_utils import get_basis, get_integrals

import os

from mpi4py import MPI

rank = MPI.COMM_WORLD.rank
n_ranks = MPI.COMM_WORLD.Get_size()


"""
Adds a batch of training geometries to the file "./trn_geometries.txt", which are
taken from a set of previous trajectories from files with prefix "trajectory_prefix".
Can be run with mpi to accelerate training set generation.
"""

trajectory_prefix = sys.argv[
    1
]  # location of trajectories from which training points are taken
no_subsample = int(sys.argv[2])  # number of trajectories to subsample in each step
data_points_to_add = int(sys.argv[3])  # total batch size
seed = int(sys.argv[4])  # random seed


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


trn_h1 = []
trn_h2 = []

if os.path.isfile("trn_geometries.txt"):
    with open("trn_geometries.txt", "r") as fl:
        lines = fl.readlines()
        for line in lines:
            time, file, _ = line.split()
            trajectory = np.load(file)
            geometry = bohr_to_angstrom * trajectory[int(time)]
            trn_mol = get_mol(geometry)
            h1_trn, h2_trn = get_integrals(trn_mol, get_basis(trn_mol))
            trn_h1.append(h1_trn)
            trn_h2.append(h2_trn)

rng = np.random.default_rng(seed)
trajectory_files = glob.glob(trajectory_prefix + "*/trajectory.npy")

for i in range(data_points_to_add):
    if rank == 0:
        print(i)

    # subsample

    if len(trn_h1) > 0:
        n_sample = no_subsample
    else:
        n_sample = 1

    fls = [
        trajectory_files[file_str]
        for file_str in rng.choice(len(trajectory_files), size=n_sample, replace=True)
    ]

    geometry_candidates = []

    sampled_times = []

    file_ids = []

    for j, fl in enumerate(fls):
        trajectory = np.load(fl)
        sampled_time = rng.choice(len(trajectory))
        geometry_candidates += list(bohr_to_angstrom * trajectory)
        sampled_times += list(np.arange(len(trajectory)))
        file_ids += list(j * np.ones(len(trajectory), dtype=int))

    geometry_candidates = np.array(geometry_candidates)

    local_ids = np.array_split(np.arange(len(geometry_candidates)), n_ranks)[rank]

    distances = np.zeros((len(geometry_candidates), len(trn_h1)))

    # now compute distances
    farthest_point = None
    for j in local_ids:
        geometry = geometry_candidates[j]
        test_mol = get_mol(geometry)
        h1_test, h2_test = get_integrals(test_mol, get_basis(test_mol))

        for x in range(len(trn_h1)):
            distances[j, x] = np.sum(abs(h1_test - trn_h1[x]) ** 2) + 0.5 * np.sum(
                abs(h2_test - trn_h2[x]) ** 2
            )

    if n_ranks > 1:
        MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, distances, op=MPI.SUM)

    if distances.shape[1] > 0:
        sampled_id = np.argmax(np.min(distances, axis=1))
    else:
        sampled_id = 0

    trn_mol = get_mol(geometry_candidates[sampled_id])

    h1_trn, h2_trn = get_integrals(trn_mol, get_basis(trn_mol))
    trn_h1.append(h1_trn)
    trn_h2.append(h2_trn)

    if rank == 0:
        if distances.shape[1] > 0:
            dist = np.min(distances, axis=1)[sampled_id]
        else:
            dist = np.nan
        with open("trn_geometries.txt", "a+") as fl:
            fl.write(
                f"{sampled_times[sampled_id]}  {fls[file_ids[sampled_id]]}  {dist}\n"
            )
