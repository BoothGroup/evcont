from pyscf import gto, md, CASCI

import numpy as np

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad
from EVCont.electron_integral_utils import get_basis, get_integrals, transform_integrals
from EVCont.MD_utils import get_trajectory
from EVCont.converge_dmrg import converge_dmrg

from EVCont.CASCI_EVCont import append_to_rdms

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad

from mpi4py import MPI


rank = MPI.COMM_WORLD.rank

ncas = 8
neleca = 4


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", geometry[0]), ("H", geometry[1]), ("O", geometry[2])],
        basis="cc-pVTZ",
        symmetry=True,
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


init_mol = mol.copy()

mf = init_mol.copy().RHF()
mf.kernel()
MPI.COMM_WORLD.Bcast(mf.mo_coeff)
cascis = [CASCI(mf, ncas, neleca)]

steps = 100
dt = 20

reference_traj = None


overlap, one_rdm, two_rdm = append_to_rdms(cascis)

if rank == 0:
    np.save("overlap_vtz.npy", overlap)
    np.save("one_rdm_vtz.npy", one_rdm)
    np.save("two_rdm_vtz.npy", two_rdm)

i = 0

if rank == 0:
    fl = open("traj_EVCont_vtz_{}.xyz".format(i), "w")
else:
    fl = None

updated_traj = get_trajectory(
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


if rank == 0:
    np.save("traj_EVCont_vtz_{}.npy".format(i), updated_traj)

thresh = 1.0e-6

times = [0]

converged_assumed = False
while not converged_assumed:
    i += 1
    if reference_traj is not None:
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
    else:
        diff = np.mean(abs(updated_traj - updated_traj[0]) ** 2, axis=(1, 2))
        trn_time = np.argmax(diff)
    trn_geometry = updated_traj[trn_time]
    mf = get_mol(trn_geometry).copy().RHF()
    mf.kernel()
    MPI.COMM_WORLD.Bcast(mf.mo_coeff)
    cascis.append(CASCI(mf, ncas, neleca))
    times.append(trn_time)
    if rank == 0:
        np.save("trn_geometry_{}_vtz.npy".format(i), trn_geometry)
    np.save("trn_time_{}_vtz.npy".format(i), trn_time)
    overlap, one_rdm, two_rdm = append_to_rdms(cascis, overlap, one_rdm, two_rdm)
    if rank == 0:
        np.save("overlap_vtz.npy", overlap)
        np.save("one_rdm_vtz.npy", one_rdm)
        np.save("two_rdm_vtz.npy", two_rdm)

    reference_traj = updated_traj
    if rank == 0:
        fl = open("traj_EVCont_vtz_{}.xyz".format(i), "w")
    else:
        fl = None
    updated_traj = get_trajectory(
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
        np.save("traj_EVCont_vtz_{}.npy".format(i), updated_traj)
