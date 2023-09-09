from pyscf import gto, md

import numpy as np

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad
from EVCont.electron_integral_utils import get_basis, get_integrals, transform_integrals
from EVCont.MD_utils import converge_EVCont_MD
from EVCont.converge_dmrg import converge_dmrg

from EVCont.DMRG_EVCont import (
    append_to_rdms_rerun,
    append_to_rdms_orbital_rotation,
    append_to_rdms_OAO_basis,
)

from EVCont.ab_initio_eigenvector_continuation import approximate_ground_state_OAO

from EVCont.DMRG_EVCont import (
    DMRG_EVCont_obj,
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


norb = nelec = 30


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


init_mol = mol.copy()


dmrg_ev_cont_obj = DMRG_EVCont_obj(dmrg_converge_fun=dmrg_converge_fun, mem=100)

if os.path.exists("overlap.npy"):
    dmrg_ev_cont_obj.overlap = np.load("overlap.npy")
    dmrg_ev_cont_obj.one_rdm = np.load("one_rdm.npy")
    dmrg_ev_cont_obj.two_rdm = np.load("two_rdm.npy")
    if os.path.exists("trn_times.txt"):
        trn_times = list(np.atleast_1d(np.loadtxt("trn_times.txt").astype(int)))
    else:
        trn_times = [0]
    trajs = [np.load("traj_EVCont_{}.npy".format(i)) for i in range(len(trn_times) - 1)]
    dmrg_ev_cont_obj.mols = [init_mol] + [
        get_mol(trajs[i][trn_times[i + 1]]) for i in range(len(trajs))
    ]
    dmrg_ev_cont_obj.tags = [i for i in range(dmrg_ev_cont_obj.overlap.shape[0])]
    dmrg_ev_cont_obj.max_tag = dmrg_ev_cont_obj.overlap.shape[0]
else:
    trn_times = []

converge_EVCont_MD(
    dmrg_ev_cont_obj,
    init_mol,
    steps=steps,
    dt=dt,
    prune_irrelevant_data=False,
    trn_times=trn_times,
    data_addition="farthest_point",
)
