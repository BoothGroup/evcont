import numpy as np

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from mpi4py import MPI


def converge_dmrg(
    h1,
    h2,
    nelec,
    tag,
    bond_dim_schedule=[np.round(1.4**i).astype(int) for i in range(9, 24)],
    mpi=MPI.COMM_WORLD.size > 1,
    noises=[1.0e-4, 1.0e-5, 1.0e-6, 1.0e-7, 0],
    tolerance=1.0e-4,
):
    norb = h1.shape[0]

    mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2, mpi=mpi, stack_mem=4 << 30)
    mps_solver.initialize_system(norb, n_elec=np.sum(nelec), spin=(nelec[0] - nelec[1]))

    mpo = mps_solver.get_qc_mpo(h1e=h1, g2e=h2, iprint=1, reorder=None)

    ket = mps_solver.get_random_mps(tag, bond_dim=bond_dim_schedule[0], nroots=1)

    converged = False

    final_energies = []

    rank = MPI.COMM_WORLD.Get_rank()

    if rank == 0:
        open("DMRG_result_{}.txt".format(tag), "w").close()

    for i in range(len(bond_dim_schedule) - 1):
        if i == 0:
            inner_bond_dim_schedule = list(
                np.round(
                    np.logspace(
                        np.log10(bond_dim_schedule[i] / 2),
                        np.log10(bond_dim_schedule[i]),
                        num=5,
                        endpoint=True,
                    )
                ).astype(int)
            )
        else:
            inner_bond_dim_schedule = list(
                np.round(
                    np.logspace(
                        np.log10((bond_dim_schedule[i - 1] + bond_dim_schedule[i]) / 2),
                        np.log10(bond_dim_schedule[i]),
                        num=5,
                        endpoint=True,
                    )
                ).astype(int)
            )
        mps_solver.dmrg(
            mpo,
            ket,
            bond_dims=inner_bond_dim_schedule,
            noises=np.array(noises) * (4 ** (-i)),
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
                        noise = (np.array(noises) * (4 ** (-i)))[j]
                    else:
                        noise = (np.array(noises) * (4 ** (-i)))[-1]
                    fl.write(
                        "{}  {}  {}  {}\n".format(bnd_dms[j], ens[j][0], dws[j], noise)
                    )
        if len(final_energies) > 1:
            if abs(final_energies[-1] - final_energies[-2]) < tolerance:
                break

    return ket, final_energies[-1]
