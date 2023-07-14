from pyscf import gto, md

import numpy as np

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad
from EVCont.electron_integral_utils import get_basis, get_integrals, transform_integrals
from EVCont.MD_utils import get_trajectory
from EVCont.converge_dmrg import converge_dmrg

from EVCont.DMRG_EVCont import append_to_rdms_rerun

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad

from mpi4py import MPI


rank = MPI.COMM_WORLD.rank


def dmrg_converge_fun(h1, h2, nelec, tag):
    MPI.COMM_WORLD.Bcast(h1, root=0)
    MPI.COMM_WORLD.Bcast(h2, root=0)
    return converge_dmrg(
        h1, h2, nelec, tag, tolerance=1.0e-3, mpi=MPI.COMM_WORLD.size > 1
    )


def append_to_rdms(mols, overlap=None, one_rdm=None, two_rdm=None):
    return append_to_rdms_rerun(
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
            ("O", geometry[3]),
            ("H", geometry[4]),
            ("H", geometry[5]),
        ],
        basis="6-31G",
        symmetry=True,
        unit="Bohr",
    )

    return mol


stretch_factor = 1.2
init_geometry = stretch_factor * np.array(
    [
        [0.0000000, 0.0000000, 0.0000000],
        [-0.6237519, -0.9109667, -1.4354514],
        [-0.6237519, -0.9109667, 1.4354514],
        [5.5028821, 0.0000000, 0.0000000],
        [3.6897611, 0.1745837, 0.0000000],
        [6.1311264, 1.6956360, 0.0000000],
    ]
)

mol = get_mol(init_geometry)


init_mol = mol.copy()

trn_mols = [init_mol.copy()]

steps = 50
dt = 10

reference_traj = np.load("traj_HF.npy")

if rank == 0:
    np.save("trn_geometry_1.npy", init_geometry)
    np.save("trn_time_1.npy", np.array(0))

overlap, one_rdm, two_rdm = append_to_rdms(trn_mols)

updated_traj = get_trajectory(
    init_mol.copy(),
    overlap,
    one_rdm,
    two_rdm,
    steps=steps,
    dt=dt,
    init_veloc=None,
    hermitian=False,
)

if rank == 0:
    np.save("traj_EVCont_1.npy", updated_traj)

thresh = 1.0e-4

times = [0]

converged_assumed = False
for i in range(2, 11):
    diff = np.mean(abs(reference_traj - updated_traj) ** 2, axis=(1, 2))
    if len(np.argwhere(diff > thresh).flatten()) > 0:
        trn_time = np.argwhere(diff > thresh).flatten()[0]
        converged_assumed = False
    else:
        if converged_assumed:
            break
        else:
            trn_time = len(diff) - 1
            converged_assumed = True
    trn_geometry = updated_traj[trn_time]
    trn_mols.append(get_mol(trn_geometry))
    times.append(trn_time)
    if rank == 0:
        np.save("trn_geometry_{}.npy".format(i), trn_geometry)
    np.save("trn_time_{}.npy".format(i), trn_time)
    overlap, one_rdm, two_rdm = append_to_rdms(trn_mols, overlap, one_rdm, two_rdm)
    reference_traj = updated_traj
    updated_traj = get_trajectory(
        init_mol.copy(), overlap, one_rdm, two_rdm, steps=steps, dt=dt, hermitian=False
    )
    if rank == 0:
        np.save("traj_EVCont_{}.npy".format(i), updated_traj)
