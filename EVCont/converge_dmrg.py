import numpy as np

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from mpi4py import MPI


def converge_dmrg(
    h1,
    h2,
    nelec,
    tag,
    bond_dim_schedule=np.round(1.8 ** np.arange(6, 16)).astype(int),
    mpi=MPI.COMM_WORLD.size > 1,
    noises=np.append(np.logspace(-2, -7, num=4), 0),
    tolerance=1.0e-4,
):
    norb = h1.shape[0]

    mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2, mpi=mpi, stack_mem=5 << 30)
    mps_solver.initialize_system(norb, n_elec=np.sum(nelec), spin=(nelec[0] - nelec[1]))

    mpo = mps_solver.get_qc_mpo(h1e=h1, g2e=h2, iprint=1, reorder=None)

    ket = mps_solver.get_random_mps(tag, bond_dim=bond_dim_schedule[0], nroots=1)

    converged = False

    final_energies = []

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        open("DMRG_result_{}.txt".format(tag), "w").close()

    for i in range(len(bond_dim_schedule) - 1):
        mps_solver.dmrg(
            mpo,
            ket,
            bond_dims=[bond_dim_schedule[i]],
            noises=noises * (4 ** (-i)),
            n_sweeps=1000,
            iprint=1,
            tol=tolerance,
        )
        bnd_dms, dws, ens = mps_solver.get_dmrg_results()
        final_energies.append(ens[-1][0])
        if rank == 0:
            print(bnd_dms, final_energies, dws)
            with open("DMRG_result_{}.txt".format(tag), "a") as fl:
                for j in range(len(bnd_dms)):
                    if j < len(noises):
                        noise = (noises * (4 ** (-i)))[j]
                    else:
                        noise = (noises * (4 ** (-i)))[-1]
                    fl.write(
                        "{}  {}  {}  {}\n".format(bnd_dms[j], ens[j][0], dws[j], noise)
                    )
        if len(final_energies) > 1:
            if abs(final_energies[-1] - final_energies[-2]) < tolerance:
                break

    return ket, final_energies[-1]
