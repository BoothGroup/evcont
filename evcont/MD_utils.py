from pyscf import md, scf, lib, grad

import numpy as np

from evcont.ab_initio_gradients_loewdin import get_energy_with_grad

from evcont.ab_initio_eigenvector_continuation import approximate_ground_state_OAO

from evcont.electron_integral_utils import get_basis, get_integrals

from mpi4py import MPI

import os

from threadpoolctl import threadpool_limits

rank = MPI.COMM_WORLD.Get_rank()


def get_scanner(mol, one_rdm, two_rdm, overlap, hermitian=True):
    """
    Returns a fake scanner object to compute MD trajectories with PySCF from
    an eigenvector continuation.
    """

    class Base:
        converged = True

    class Scanner(lib.GradScanner):
        def __init__(self):
            self.mol = mol
            self.base = Base()
            # self.converged = True

        def __call__(self, mol):
            self.mol = mol
            if one_rdm is not None and two_rdm is not None and overlap is not None:
                return get_energy_with_grad(
                    mol, one_rdm, two_rdm, overlap, hermitian=hermitian
                )
            else:
                return mol.energy_nuc(), grad.RHF(scf.RHF(mol)).grad_nuc()

    return Scanner()


def get_trajectory(
    init_mol,
    overlap,
    one_rdm,
    two_rdm,
    dt=10.0,
    steps=10,
    init_veloc=None,
    hermitian=True,
    trajectory_output=None,
    data_output=None,
):
    """
    Helper function to compute an MD trajectory from eigenvector continuation with
    PySCF.

    Args:
        init_mol: The initial molecule.
        overlap: The overlap matrix.
        one_rdm: The one-particle t-RDM.
        two_rdm: The two-particle t-RDM.
        dt: Time step for the simulation. Default is 10.0.
        steps: Number of simulation steps. Default is 10.
        init_veloc: Initial velocities of the atoms. Default is None.
        hermitian (bool, optional):
            Whether problem is solved with eigh or with eig. Defaults to True.
        trajectory_output: File to write the trajectory output. Default is None.
        energy_output: File to write the energy output. Default is None.

    Returns:
        trajectory: The calculated trajectory as a numpy array.
    """
    trajectory = np.zeros((steps, len(init_mol.atom), 3))

    # Compute max number of threads we could use for the non-mpi-parallel part
    num_threads = MPI.COMM_WORLD.Split_type(MPI.COMM_TYPE_SHARED).Get_size()

    num_threads_outer = os.getenv("OMP_NUM_THREADS")

    if num_threads_outer is not None:
        num_threads *= int(num_threads_outer)

    if rank == 0:
        with threadpool_limits(limits=num_threads):
            scanner_fun = get_scanner(
                init_mol, one_rdm, two_rdm, overlap, hermitian=hermitian
            )

            frames = []
            myintegrator = md.NVE(
                scanner_fun,
                dt=dt,
                steps=steps,
                veloc=init_veloc,
                incore_anyway=True,
                frames=frames,
                trajectory_output=trajectory_output,
                data_output=data_output,
                verbose=0,
            )
            myintegrator.run()
            trajectory = np.array([frame.coord for frame in frames])

    MPI.COMM_WORLD.Bcast(trajectory, root=0)

    return trajectory


def converge_EVCont_MD(
    EVCont_obj,
    init_mol,
    steps=100,
    dt=1,
    convergence_thresh=1.0e-3,
    prune_irrelevant_data=False,
    trn_times=[],
    data_addition="farthest_point_ham",
):
    """
    Helper function to converge the prediction of MD trajectories from EV continuation.
    This includes the on-the-fly learning by iteratively adding data points from
    previously generated trajectories. The function saves the trajectories, the
    intermediate representations for the continuation, the times at which data points
    were picked, as well as PES information to disk.


    TODO: This needs to be cleaned up.

    Args:
        EVCont_obj: The data structure for the eigenvectrous continuation.
        init_mol: The initial molecule object.
        steps: Number of MD simulation steps. Default is 100.
        dt: Time step for the simulation. Default is 1.
        convergence_thresh:
            Energy convergence threshold to terminate the training. Default is 1.0e-3.
        prune_irrelevant_data (bool, optional):
            Whether to prune data points not contributing to the PES.
        trn_times (list, optional):
            List of previous training times (as indices). Required to continue a
            previous simulation.
        data_addition:
            Criterion for adding new data points. Can be "farthest_point_ham" (default),
            in which case data is added based on electron integral difference,
            "farthest_point", in which case data is added based on the farthest point
            according to Euclidean distance, or "energy", in which case data is added
            based on the energy difference.

    Returns:
        trajectory: The calculated trajectory as a numpy array.
    """
    if len(trn_times) < 1:
        i = 0
        trn_times = [0]

        EVCont_obj.append_to_rdms(init_mol.copy())

        if rank == 0:
            if prune_irrelevant_data:
                np.save("overlap_{}.npy".format(i), EVCont_obj.overlap)
                np.save("one_rdm_{}.npy".format(i), EVCont_obj.one_rdm)
                np.save("two_rdm_{}.npy".format(i), EVCont_obj.two_rdm)
            else:
                np.save("overlap.npy", EVCont_obj.overlap)
                np.save("one_rdm.npy", EVCont_obj.one_rdm)
                np.save("two_rdm.npy", EVCont_obj.two_rdm)
            trajectory_out = open("traj_EVCont_{}.xyz".format(i), "w")
            en_out = open("ens_EVCont_{}.xyz".format(i), "w")
        else:
            trajectory_out = None
            en_out = None

        trajectory = get_trajectory(
            init_mol.copy(),
            EVCont_obj.overlap,
            EVCont_obj.one_rdm,
            EVCont_obj.two_rdm,
            steps=steps,
            trajectory_output=trajectory_out,
            data_output=en_out,
            dt=dt,
        )

        if rank == 0:
            trajectory_out.close()
            en_out.close()
            np.save("traj_EVCont_{}.npy".format(i), trajectory)

            updated_ens = np.ascontiguousarray(
                np.genfromtxt("ens_EVCont_{}.xyz".format(i))[:, 1]
            )
        else:
            updated_ens = np.zeros(trajectory.shape[0])

        MPI.COMM_WORLD.Bcast(updated_ens, root=0)
        reference_ens = updated_ens[0]

        converged = False
    else:
        i = len(trn_times) - 1

        traj_computed = os.path.exists("traj_EVCont_{}.npy".format(i))

        if rank == 0:
            if prune_irrelevant_data:
                np.save("overlap_{}.npy".format(i), EVCont_obj.overlap)
                np.save("one_rdm_{}.npy".format(i), EVCont_obj.one_rdm)
                np.save("two_rdm_{}.npy".format(i), EVCont_obj.two_rdm)
                np.savetxt("trn_times_{}.txt".format(i), np.array(trn_times))
            else:
                np.save("overlap.npy", EVCont_obj.overlap)
                np.save("one_rdm.npy", EVCont_obj.one_rdm)
                np.save("two_rdm.npy", EVCont_obj.two_rdm)
                np.savetxt("trn_times.txt", np.array(trn_times))
            if not traj_computed:
                trajectory_out = open("traj_EVCont_{}.xyz".format(i), "w")
                en_out = open("ens_EVCont_{}.xyz".format(i), "w")
        else:
            trajectory_out = None
            en_out = None

        if not traj_computed:
            trajectory = get_trajectory(
                init_mol.copy(),
                EVCont_obj.overlap,
                EVCont_obj.one_rdm,
                EVCont_obj.two_rdm,
                steps=steps,
                trajectory_output=trajectory_out,
                data_output=en_out,
                dt=dt,
            )
        else:
            trajectory = np.load("traj_EVCont_{}.npy".format(i))

        if rank == 0:
            if not traj_computed:
                trajectory_out.close()
                en_out.close()
                np.save("traj_EVCont_{}.npy".format(i), trajectory)

            updated_ens = np.ascontiguousarray(
                np.genfromtxt("ens_EVCont_{}.xyz".format(i))[:, 1]
            )

            if i > 0:
                reference_ens = np.array(
                    [
                        approximate_ground_state_OAO(
                            init_mol.copy().set_geom_(geometry),
                            EVCont_obj.one_rdm[:-1, :-1],
                            EVCont_obj.two_rdm[:-1, :-1],
                            EVCont_obj.overlap[:-1, :-1],
                        )[0]
                        for geometry in trajectory
                    ]
                )
            else:
                reference_ens = updated_ens[0]

            if prune_irrelevant_data:
                print("pruning irrelevant data points")
                keep = np.ones(len(trn_times), dtype=bool)
                for j in range(len(trn_times)):
                    print(j)
                    test_keep = keep.copy()
                    test_keep[j] = False
                    if np.sum(test_keep) >= 1:
                        test_ids = np.ix_(test_keep, test_keep)

                        reference_ens_datapoint_removed = np.array(
                            [
                                approximate_ground_state_OAO(
                                    init_mol.copy().set_geom_(geometry),
                                    EVCont_obj.one_rdm[test_ids],
                                    EVCont_obj.two_rdm[test_ids],
                                    EVCont_obj.overlap[test_ids],
                                )[0]
                                for geometry in trajectory
                            ]
                        )
                        if np.all(
                            abs(reference_ens_datapoint_removed - updated_ens)
                            < convergence_thresh
                        ):
                            keep = test_keep
                            print("removing data point {}".format(j))
        else:
            reference_ens = np.zeros_like(updated_ens)
            if prune_irrelevant_data:
                keep = np.ones(len(trn_times), dtype=bool)

        MPI.COMM_WORLD.Bcast(updated_ens, root=0)
        MPI.COMM_WORLD.Bcast(reference_ens, root=0)

        if prune_irrelevant_data:
            MPI.COMM_WORLD.Bcast(keep, root=0)
            keep_ids = np.nonzero(keep)[0]
            trn_times = [trn_times[j] for j in keep_ids]
            EVCont_obj.prune_datapoints(keep_ids)

        converged = False
        if i >= 1:
            en_diff = np.loadtxt("en_diff_{}.txt".format(i - 1))
            if max(en_diff) <= convergence_thresh:
                converged = True

    while True:
        en_diff = abs(reference_ens - updated_ens)
        if rank == 0:
            np.savetxt("en_diff_{}.txt".format(i), np.array(en_diff))
        i += 1

        if converged and max(en_diff) <= convergence_thresh:
            break
        if max(en_diff) <= convergence_thresh:
            converged = True
        else:
            converged = False

        if data_addition == "energy":
            trn_time = np.argmax(en_diff)
        elif data_addition == "farthest_point":
            # Reconstruct training geometries
            trajs = [
                np.load("traj_EVCont_{}.npy".format(i)) for i in range(len(trn_times))
            ]

            trn_geometries = [trajs[0][0]] + [
                trajs[k][trn_times[k + 1]] for k in range(len(trajs) - 1)
            ]

            # Farthest point selection
            trn_time = np.argmax(
                np.min(
                    np.array(
                        [
                            np.sum(abs(trn_geom - trajectory) ** 2, axis=(-1, -2))
                            for trn_geom in trn_geometries
                        ]
                    ),
                    axis=0,
                )
            )
        elif data_addition == "farthest_point_ham":
            trn_time = 0
            if rank == 0:
                # Reconstruct training geometries
                trajs = [
                    np.load("traj_EVCont_{}.npy".format(i))
                    for i in range(len(trn_times))
                ]

                trn_geometries = [trajs[0][0]] + [
                    trajs[k][trn_times[k + 1]] for k in range(len(trajs) - 1)
                ]

                h1_trn = np.zeros((len(trn_geometries), init_mol.nao, init_mol.nao))
                h2_trn = np.zeros(
                    (
                        len(trn_geometries),
                        init_mol.nao,
                        init_mol.nao,
                        init_mol.nao,
                        init_mol.nao,
                    )
                )
                for j, trn_geom in enumerate(trn_geometries):
                    mol = init_mol.copy().set_geom_(trn_geom)
                    h1, h2 = get_integrals(mol, get_basis(mol))
                    h1_trn[j] = h1
                    h2_trn[j] = h2

                farthest_point = None

                for j, geometry in enumerate(trajectory):
                    mol = init_mol.copy().set_geom_(geometry)
                    h1, h2 = get_integrals(mol, get_basis(mol))

                    distance = np.sum(
                        abs(h1 - h1_trn) ** 2, axis=(-1, -2)
                    ) + 0.5 * np.sum(abs(h2 - h2_trn) ** 2, axis=(-1, -2, -3, -4))
                    min_dist = np.min(distance)
                    if farthest_point is None or min_dist > farthest_point:
                        farthest_point = min_dist
                        trn_time = j
            trn_time = MPI.COMM_WORLD.bcast(trn_time, root=0)
        else:
            assert False

        trn_geometry = trajectory[trn_time]
        trn_times.append(trn_time)

        EVCont_obj.append_to_rdms(init_mol.copy().set_geom_(trn_geometry))

        if rank == 0:
            if prune_irrelevant_data:
                np.save("overlap_{}.npy".format(i), EVCont_obj.overlap)
                np.save("one_rdm_{}.npy".format(i), EVCont_obj.one_rdm)
                np.save("two_rdm_{}.npy".format(i), EVCont_obj.two_rdm)
                np.savetxt("trn_times_{}.txt".format(i), np.array(trn_times))
            else:
                np.save("overlap.npy", EVCont_obj.overlap)
                np.save("one_rdm.npy", EVCont_obj.one_rdm)
                np.save("two_rdm.npy", EVCont_obj.two_rdm)
                np.savetxt("trn_times.txt", np.array(trn_times))

            trajectory_out = open("traj_EVCont_{}.xyz".format(i), "w")
            en_out = open("ens_EVCont_{}.xyz".format(i), "w")
        else:
            trajectory_out = None
            en_out = None

        trajectory = get_trajectory(
            init_mol.copy(),
            EVCont_obj.overlap,
            EVCont_obj.one_rdm,
            EVCont_obj.two_rdm,
            steps=steps,
            trajectory_output=trajectory_out,
            data_output=en_out,
            dt=dt,
        )

        if rank == 0:
            trajectory_out.close()
            en_out.close()
            np.save("traj_EVCont_{}.npy".format(i), trajectory)

            reference_ens = np.array(
                [
                    approximate_ground_state_OAO(
                        init_mol.copy().set_geom_(geometry),
                        EVCont_obj.one_rdm[:-1, :-1],
                        EVCont_obj.two_rdm[:-1, :-1],
                        EVCont_obj.overlap[:-1, :-1],
                    )[0]
                    for geometry in trajectory
                ]
            )
            updated_ens = np.ascontiguousarray(
                np.genfromtxt("ens_EVCont_{}.xyz".format(i))[:, 1]
            )

            if prune_irrelevant_data:
                print("pruning irrelevant data points")
                keep = np.ones(len(trn_times), dtype=bool)
                for j in range(len(trn_times)):
                    print(j)
                    test_keep = keep.copy()
                    test_keep[j] = False
                    if np.sum(test_keep) >= 1:
                        test_ids = np.ix_(test_keep, test_keep)

                        reference_ens_datapoint_removed = np.array(
                            [
                                approximate_ground_state_OAO(
                                    init_mol.copy().set_geom_(geometry),
                                    EVCont_obj.one_rdm[test_ids],
                                    EVCont_obj.two_rdm[test_ids],
                                    EVCont_obj.overlap[test_ids],
                                )[0]
                                for geometry in trajectory
                            ]
                        )
                        if np.all(
                            abs(reference_ens_datapoint_removed - updated_ens)
                            < convergence_thresh
                        ):
                            keep = test_keep
                            print("removing data point {}".format(j))
        else:
            reference_ens = np.zeros_like(updated_ens)
            if prune_irrelevant_data:
                keep = np.ones(len(trn_times), dtype=bool)

        MPI.COMM_WORLD.Bcast(updated_ens, root=0)
        MPI.COMM_WORLD.Bcast(reference_ens, root=0)

        if prune_irrelevant_data:
            MPI.COMM_WORLD.Bcast(keep, root=0)
            keep_ids = np.nonzero(keep)[0]
            trn_times = [trn_times[j] for j in keep_ids]
            EVCont_obj.prune_datapoints(keep_ids)
