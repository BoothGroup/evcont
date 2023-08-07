from pyscf import gto, md, ao2mo

import numpy as np

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad, get_loewdin_trafo
from EVCont.electron_integral_utils import get_basis, get_integrals, transform_integrals
from EVCont.MD_utils import get_trajectory
from EVCont.converge_dmrg import converge_dmrg

from EVCont.ab_initio_eigenvector_continuation import approximate_ground_state_OAO

from EVCont.DMRG_EVCont import (
    append_to_rdms_rerun,
    append_to_rdms_orbital_rotation,
    append_to_rdms_OAO_basis,
)

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad

from mpi4py import MPI


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


def append_to_rdms(mols, overlap=None, one_rdm=None, two_rdm=None):
    return append_to_rdms_OAO_basis(
        mols,
        overlap=overlap,
        one_rdm=one_rdm,
        two_rdm=two_rdm,
        converge_dmrg_fun=dmrg_converge_fun,
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
        unit="Bohr",
        charge=1,
    )

    return mol


stretch_factor = 1.5
init_geometry = stretch_factor * np.array(
    [
        [0.0000000, 0.0000000, 0.0000000],
        [-0.6237519, -0.9109667, -1.4354514],
        [-0.6237519, -0.9109667, 1.4354514],
        [5.5028821 / 2, 0.0, 0.0],
        [5.5028821, 0.0000000, 0.0000000],
        [3.6897611, 0.1745837, 0.0000000],
        [6.1311264, 1.6956360, 0.0000000],
    ]
)


mol = get_mol(init_geometry)

init_mol = mol.copy()

trn_mols = [init_mol.copy()]

steps = 200
dt = 20

overlap, one_rdm, two_rdm = append_to_rdms(trn_mols)

if rank == 0:
    np.save("overlap.npy", overlap)
    np.save("one_rdm.npy", one_rdm)
    np.save("two_rdm.npy", two_rdm)

i = 0

if rank == 0:
    fl = open("traj_EVCont_{}.xyz".format(i), "w")
else:
    fl = None

trajectory = get_trajectory(
    init_mol.copy(),
    overlap,
    one_rdm,
    two_rdm,
    steps=steps,
    dt=dt,
    trajectory_output=fl,
)
if rank == 0:
    fl.close()

updated_ens = np.array(
    [
        approximate_ground_state_OAO(get_mol(geometry), one_rdm, two_rdm, overlap)[0]
        for geometry in trajectory
    ]
)

reference_ens = updated_ens[0]


if rank == 0:
    np.save("traj_EVCont_{}.npy".format(i), trajectory)
    open("MD_convergence.txt", "w").close()

thresh = 1.0e-5

times = [0]

while True:
    i += 1
    en_diff = abs(updated_ens - reference_ens)
    trn_time = np.argmax(en_diff)
    if rank == 0:
        with open("MD_convergence.txt", "a") as fl:
            fl.write("{}\n".format(en_diff[trn_time]))
    if en_diff[trn_time] < thresh and i > 1:
        break
    trn_geometry = trajectory[trn_time]
    trn_mols.append(get_mol(trn_geometry))
    times.append(trn_time)
    if rank == 0:
        np.save("trn_geometry_{}.npy".format(i), trn_geometry)
    np.save("trn_time_{}.npy".format(i), trn_time)
    overlap, one_rdm, two_rdm = append_to_rdms(trn_mols, overlap, one_rdm, two_rdm)
    if rank == 0:
        np.save("overlap.npy", overlap)
        np.save("one_rdm.npy", one_rdm)
        np.save("two_rdm.npy", two_rdm)

    if rank == 0:
        fl = open("traj_EVCont_{}.xyz".format(i), "w")
    else:
        fl = None
    trajectory = get_trajectory(
        init_mol.copy(),
        overlap,
        one_rdm,
        two_rdm,
        steps=steps,
        dt=dt,
        trajectory_output=fl,
        hermitian=True,
    )
    if rank == 0:
        fl.close()
    if rank == 0:
        np.save("traj_EVCont_{}.npy".format(i), trajectory)

    reference_ens = updated_ens
    updated_ens = np.array(
        [
            approximate_ground_state_OAO(get_mol(geometry), one_rdm, two_rdm, overlap)[
                0
            ]
            for geometry in trajectory
        ]
    )
