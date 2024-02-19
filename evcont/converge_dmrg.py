import numpy as np

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from mpi4py import MPI

import os


def converge_dmrg(
    h1,
    h2,
    nelec,
    tag,
    bond_dim_schedule=np.round(1.8 ** np.arange(6, 16)).astype(int),
    mpi=MPI.COMM_WORLD.size > 1,
    noises=np.append(np.logspace(-2, -7, num=4), 0),
    tolerance=1.0e-4,
    restart_tag=None,
    nroots=1,
    mem=5,
):
    """
    Converge a DMRG calculation for a given system. See block2 documentation for more
    information.

    Parameters:
    - h1: numpy.ndarray
        One-electron integrals
    - h2: numpy.ndarray
        Two-electron integrals
    - nelec: tuple
        Number of electrons in alpha and beta spin channels
    - tag: str
        Tag used for the DMRG run
    - bond_dim_schedule: numpy.ndarray, optional
        Array of bond dimensions for each DMRG iteration,
        default: np.round(1.8 ** np.arange(6, 16)).astype(int)
    - mpi: bool, optional
        Flag to enable MPI parallelization, default: MPI.COMM_WORLD.size > 1
    - noises: numpy.ndarray, optional
        Array of noise levels for each DMRG iteration,
        default: np.append(np.logspace(-2, -7, num=4), 0)
    - tolerance: float, optional
        Energy convergence tolerance, default: 1.0e-4
    - restart_tag: str, optional
        Tag used to load a previous DMRG calculation, default: None
    - mem: int, optional
        Memory allocation for the DMRG solver (in GB), default: 5

    Returns:
    - ket: numpy.ndarray
        Final converged DMRG wavefunction
    - final_energy: float
        Final energy after convergence
    """

    norb = h1.shape[0]

    mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2, mpi=mpi, stack_mem=mem << 30)
    mps_solver.initialize_system(norb, n_elec=np.sum(nelec), spin=(nelec[0] - nelec[1]))

    mpo = mps_solver.get_qc_mpo(h1e=h1, g2e=h2, iprint=1, reorder=None)

    if restart_tag is not None and os.path.exists(
        "nodex/{}-mps_info.bin".format(restart_tag)
    ):
        # Load a previous DMRG calculation if restart tag is provided
        ket = mps_solver.load_mps(restart_tag, nroots=nroots)
    else:
        ket = mps_solver.get_random_mps(tag, bond_dim=bond_dim_schedule[0], nroots=nroots)

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
        final_energies.append(ens[-1][:])
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
            if np.max(np.abs(final_energies[-1] - final_energies[-2])) < tolerance:
                break

    return ket, final_energies[-1]
