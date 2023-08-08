from pyscf import gto, md

from pyscf.mcscf.casci import CASCI

import numpy as np

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad
from EVCont.electron_integral_utils import get_basis, get_integrals, transform_integrals
from EVCont.MD_utils import get_trajectory
from EVCont.converge_dmrg import converge_dmrg

from EVCont.CASCI_EVCont import append_to_rdms

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


updated_ens = np.zeros(len(trajectory))

if rank == 0:
    updated_ens = np.array(
        [
            approximate_ground_state_OAO(get_mol(geometry), one_rdm, two_rdm, overlap)[
                0
            ]
            for geometry in trajectory
        ]
    )

MPI.COMM_WORLD.Bcast(updated_ens)

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
    mf = get_mol(trn_geometry).copy().RHF()
    mf.kernel()
    MPI.COMM_WORLD.Bcast(mf.mo_coeff)
    cascis.append(CASCI(mf, ncas, neleca))
    overlap, one_rdm, two_rdm = append_to_rdms(cascis, overlap, one_rdm, two_rdm)
    if rank == 0:
        np.save("overlap_vtz.npy", overlap)
        np.save("one_rdm_vtz.npy", one_rdm)
        np.save("two_rdm_vtz.npy", two_rdm)

    if rank == 0:
        fl = open("traj_EVCont_vtz_{}.xyz".format(i), "w")
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
        np.save("traj_EVCont_vtz_{}.npy".format(i), trajectory)

    reference_ens = updated_ens
    updated_ens = np.zeros(len(trajectory))

    if rank == 0:
        updated_ens = np.array(
            [
                approximate_ground_state_OAO(
                    get_mol(geometry), one_rdm, two_rdm, overlap
                )[0]
                for geometry in trajectory
            ]
        )

    MPI.COMM_WORLD.Bcast(updated_ens)
