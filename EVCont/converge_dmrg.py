import numpy as np

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from mpi4py import MPI

def converge_dmrg(h1, h2, nelec, tag, init_bond_dim=25, bond_dim_incr=25, mpi=False, tolerance=1.e-5):
    norb = h1.shape[0]

    mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2, mpi=mpi)
    mps_solver.initialize_system(norb, n_elec=np.sum(nelec))

    mpo = mps_solver.get_qc_mpo(h1e=h1, g2e=h2, iprint=1, reorder=None)

    bnd_dim = init_bond_dim

    ket = mps_solver.get_random_mps(tag, bond_dim=bnd_dim, nroots=1)

    converged = False

    energies = []

    open("DMRG_result_{}.txt".format(tag), "w").close()

    rank = MPI.COMM_WORLD.Get_rank()

    while not converged:
        mps_solver.dmrg(mpo, ket, bond_dims=[bnd_dim], n_sweeps=1000, iprint=1, noises =  [1e-4] * 4 + [1e-5] * 4 + [0], tol=1.e-3)
        bnd_dms, _, ens = mps_solver.get_dmrg_results()
        energies.append(ens[-1][0])
        bnd_dim += bond_dim_incr
        if len(energies) > 1:
            if abs(energies[-1] - energies[-2]) < tolerance:
                converged = True
        if rank == 0:
            print(bnd_dms[-1], energies[-1], len(ens))
            with open("DMRG_result_{}.txt".format(tag), "a") as fl:
                fl.write("{}  {}  {}\n".format(bnd_dms[-1], energies[-1], len(ens)))

    return ket

