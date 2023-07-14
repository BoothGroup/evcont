import numpy as np

import scipy.linalg

from block2 import (
    QCTypes,
    OpNamesSet,
    OpNames,
    TETypes,
    VectorUBond,
    VectorDouble,
    TruncPatternTypes,
)
from block2.su2 import (
    HamiltonianQC,
    MPOQC,
    SimplifiedMPO,
    AntiHermitianRuleQC,
    RuleQC,
    TimeEvolution,
    MovingEnvironment,
    TDDMRG,
)

from mpi4py import MPI


rank = MPI.COMM_WORLD.rank


from pyblock2.driver.core import DMRGDriver, SymmetryTypes


# Little utility function to apply the single-body orbital rotation to an MPS
def orbital_rotation_mps(
    ket,
    mpo_kappa,
    flip_sign,
    rotation_driver,
    bond_dim=1000,
    dt=0.05,
    iprint=1,
    convergence_thresh=1.0e-3,
):
    assert dt <= 1.0 and dt >= 0.0

    me_kappa = MovingEnvironment(mpo_kappa, ket, ket, "DMRG")
    me_kappa.delayed_contraction = OpNamesSet.normal_ops()
    me_kappa.cached_contraction = True
    me_kappa.init_environments(True)

    # Time Evolution (anti-Hermitian)
    # te_type can be TETypes.RK4 or TETypes.TangentSpace (TDVP)
    te_type = TETypes.RK4
    # te_type = TETypes.TangentSpace
    te = TimeEvolution(me_kappa, VectorUBond([bond_dim]), te_type)
    te.hermitian = False
    te.iprint = iprint
    te.n_sub_sweeps = 2
    te.normalize_mps = False
    te.trunc_pattern = TruncPatternTypes.TruncAfterEven

    converged = True

    for i in range(round(1.0 / abs(dt))):
        te.solve(1, dt, ket.center == 0)
        if abs(te.normsqs[-1] - 1.0) > convergence_thresh:
            converged = False
            break

    if flip_sign:
        flip_operator = rotation_driver.expr_builder()
        flip_operator.add_term("", [], 1)
        flip_operator.add_term("(C+D)0", [0, 0], -2.0 * np.sqrt(2))
        flip_operator.add_term("((C+(C+D)0)1+D)0", [0, 0, 0, 0], 4.0)
        flip_mpo = rotation_driver.get_mpo(flip_operator.finalize(), iprint=iprint)
        mps_flipped_back = ket.deep_copy("ket_flipped")
        rotation_driver.multiply(mps_flipped_back, flip_mpo, ket)
        ket = mps_flipped_back.deep_copy(ket.info.tag)

    return ket, converged


# Converges the orbital bon dimension required to match the orbital rotation
def converge_orbital_rotation_mps(
    ket,
    orbital_rotation_matrix,
    init_bond_dim=25,
    bond_dim_incr=25,
    rotation_driver=None,
    convergence_thresh=1.0e-3,
    convergence_mpos=None,
    dt=0.05,
    iprint=1,
    tag=None,
):
    bond_dim = init_bond_dim

    if rotation_driver is None:
        rotation_driver = DMRGDriver(symm_type=SymmetryTypes.SU2)
        rotation_driver.initialize_system(orbital_rotation_matrix.shape[0])

    if rank == 0:
        if convergence_mpos is None:
            convergence_mpos = (
                rotation_driver.get_identity_mpo(),
                rotation_driver.get_identity_mpo(),
            )

        if tag is None:
            tag = ket.info.tag + str(hash(orbital_rotation_matrix.tobytes()))

        reference_expectation = rotation_driver.expectation(
            ket, convergence_mpos[0], ket, iprint=iprint
        )

        open("orbital_rotation_output_{}.txt".format(tag), "w").close()

        orbital_rotation_matrix_pos = orbital_rotation_matrix.copy()
        if scipy.linalg.det(orbital_rotation_matrix_pos) < 0.0:
            flip_sign = True
            orbital_rotation_matrix_pos[:, 0] *= -1
            print("sign flip")
        else:
            flip_sign = False

        log_orb_rot = scipy.linalg.logm(orbital_rotation_matrix_pos)

        assert np.isrealobj(log_orb_rot)

        # perform orbital rotation (with positive det) from old to new basis

        # Hamiltonian for orbital transform
        hamil_kappa = HamiltonianQC(
            rotation_driver.vacuum,
            rotation_driver.n_sites,
            rotation_driver.orb_sym,
            rotation_driver.write_fcidump(log_orb_rot, None),
        )

        # MPO (anti-Hermitian)
        mpo_kappa = MPOQC(hamil_kappa, QCTypes.Conventional)
        mpo_kappa = SimplifiedMPO(
            mpo_kappa,
            AntiHermitianRuleQC(RuleQC()),
            True,
            True,
            OpNamesSet((OpNames.R, OpNames.RD)),
        )

        converged = False
        while not converged:
            rotated_ket = ket.deep_copy("rotated_ket")
            rotated_ket, converged = orbital_rotation_mps(
                rotated_ket,
                mpo_kappa,
                flip_sign,
                rotation_driver,
                bond_dim=bond_dim,
                dt=dt,
                iprint=iprint,
                convergence_thresh=convergence_thresh,
            )
            final_expectation = rotation_driver.expectation(
                rotated_ket, convergence_mpos[1], rotated_ket, iprint=iprint
            )
            print(bond_dim, reference_expectation, final_expectation)

            with open("orbital_rotation_output_{}.txt".format(tag), "a") as fl:
                fl.write(
                    "{}  {}  {}\n".format(
                        bond_dim, reference_expectation, final_expectation
                    )
                )

            if converged:
                if abs(reference_expectation - final_expectation) <= convergence_thresh:
                    converged = True
                else:
                    converged = False
            if not converged:
                bond_dim += bond_dim_incr
    MPI.COMM_WORLD.barrier()
    rotated_ket = rotation_driver.load_mps("rotated_ket")
    return rotated_ket
