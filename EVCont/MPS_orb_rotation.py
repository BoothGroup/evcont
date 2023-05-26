import scipy.linalg

from block2 import QCTypes, OpNamesSet, OpNames, TETypes, VectorUBond
from block2.su2 import HamiltonianQC, MPOQC, SimplifiedMPO, AntiHermitianRuleQC, RuleQC, TimeEvolution, MovingEnvironment


from pyblock2.driver.core import DMRGDriver, SymmetryTypes

# Little utility function to apply the single-body orbital rotation to an MPS
def orbital_rotation_mps(ket, orbital_rotation_matrix, rotation_driver = None):
    if rotation_driver is None:
        rotation_driver = DMRGDriver(symm_type=SymmetryTypes.SU2)
        rotation_driver.initialize_system(orbital_rotation_matrix.shape[0])

    log_orb_rot = scipy.linalg.logm(orbital_rotation_matrix)


    # Hamiltonain for orbital transform
    hamil_kappa = HamiltonianQC(rotation_driver.vacuum, rotation_driver.n_sites, rotation_driver.orb_sym, rotation_driver.write_fcidump(log_orb_rot, None))

    # MPO (anti-Hermitian)
    mpo_kappa = MPOQC(hamil_kappa, QCTypes.Conventional)
    mpo_kappa = SimplifiedMPO(mpo_kappa, AntiHermitianRuleQC(RuleQC()), True, True, OpNamesSet((OpNames.R, OpNames.RD)))


    # Time Step
    dt = 0.05
    # Target time
    tt = 1.0
    n_steps = int(abs(tt) / abs(dt) + 0.1)
    me_kappa = MovingEnvironment(mpo_kappa, ket, ket, "DMRG")
    me_kappa.delayed_contraction = OpNamesSet.normal_ops()
    me_kappa.cached_contraction = True
    me_kappa.init_environments(True)

    # Time Evolution (anti-Hermitian)
    # te_type can be TETypes.RK4 or TETypes.TangentSpace (TDVP)
    te_type = TETypes.RK4
    te = TimeEvolution(me_kappa, VectorUBond([1000]), TETypes.RK4)
    te.hermitian = False
    te.iprint = 2
    te.n_sub_sweeps = 2
    te.normalize_mps = False
    for i in range(n_steps):
        te.solve(1, dt, ket.center == 0)
        print("T = %10.5f <E> = %20.15f <Norm^2> = %20.15f" %
                ((i + 1) * dt, te.energies[-1], te.normsqs[-1]))

