from pyscf import gto, md

import numpy as np

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad
from EVCont.electron_integral_utils import get_basis, get_integrals, transform_integrals
from EVCont.MD_utils import get_trajectory
from EVCont.converge_dmrg import converge_dmrg

from EVCont.DMRG_EVCont import (
    append_to_rdms_rerun,
    append_to_rdms_orbital_rotation,
    append_to_rdms_OAO_basis,
)

from EVCont.ab_initio_eigenvector_continuation import approximate_ground_state_OAO


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
        h1, h2, nelec, tag, tolerance=1.0e-4, mpi=MPI.COMM_WORLD.size > 1
    )


def append_to_rdms(mols, overlap=None, one_rdm=None, two_rdm=None):
    return append_to_rdms_OAO_basis(
        mols,
        overlap=overlap,
        one_rdm=one_rdm,
        two_rdm=two_rdm,
        converge_dmrg_fun=dmrg_converge_fun,
    )


norb = nelec = 30


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", pos) for pos in geometry],
        basis="sto-6g",
        symmetry=True,
        unit="Bohr",
    )

    return mol


init_dist = 1.9

steps = 200

mol = get_mol(np.array([[0, 0, init_dist * i] for i in range(nelec)]))


init_mol = mol.copy()

trn_mols = [init_mol.copy()]


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
    open("en_convergence.txt", "w").close()
    open("trn_times.txt", "w").close()

thresh = 1.0e-3
inner_thresh = 1.0e-2

while True:
    i += 1
    en_diff = abs(updated_ens - reference_ens)
    if rank == 0:
        with open("en_convergence.txt", "a") as fl:
            fl.write("{}\n".format(max(en_diff)))
    if max(en_diff) > inner_thresh:
        trn_time = np.argwhere(en_diff > thresh).flatten()[0]
        converged = False
    else:
        if converged and max(en_diff) <= thresh:
            break
        trn_time = np.argmax(en_diff)
        if max(en_diff) <= thresh:
            converged = True
    if rank == 0:
        with open("trn_times.txt", "a") as fl:
            fl.write("{}\n".format(trn_time))
    trn_geometry = trajectory[trn_time]
    trn_mols.append(get_mol(trn_geometry))
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
