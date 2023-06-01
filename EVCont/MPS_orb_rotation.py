import numpy as np

import scipy.linalg

from block2 import QCTypes, OpNamesSet, OpNames, TETypes, VectorUBond, VectorDouble, TruncPatternTypes
from block2.su2 import HamiltonianQC, MPOQC, SimplifiedMPO, AntiHermitianRuleQC, RuleQC, TimeEvolution, MovingEnvironment, TDDMRG


from pyblock2.driver.core import DMRGDriver, SymmetryTypes

def get_reorder_trafo(reorder_idx):
    reorder_mat = np.eye(reorder_idx.shape[0])[reorder_idx,:]
    for j in range(reorder_mat.shape[0]):
        if scipy.linalg.det(reorder_mat[:j+1, :j+1]) < 0:
            reorder_mat[:, j] *= -1
    return reorder_mat

# Little utility function to apply the single-body orbital rotation to an MPS
def orbital_rotation_mps(ket, orbital_rotation_matrix, rotation_driver = None, bond_dim=1000, convergence_thresh=1.e-6):
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
    # te_type = TETypes.TangentSpace
    te = TimeEvolution(me_kappa, VectorUBond([bond_dim]), te_type)
    te.hermitian = False
    te.iprint = 1
    te.n_sub_sweeps = 2
    te.normalize_mps = False
    te.trunc_pattern = TruncPatternTypes.TruncAfterEven
    converged = False
    for i in range(n_steps):
        te.solve(1, dt, ket.center == 0)
        dw = te.discarded_weights[0]
        if dw > convergence_thresh:
            break
    if dw <= convergence_thresh:
        converged = True
    return converged

def converge_orbital_rotation_mps(ket, orbital_rotation_matrix, init_bond_dim=25, bond_dim_incr=25, rotation_driver = None, convergence_thresh=1.e-6):
    converged = False
    bond_dim = init_bond_dim
    while not converged:
        ket_temp = ket.deep_copy("tmp_copy")
        converged = orbital_rotation_mps(ket_temp, orbital_rotation_matrix, bond_dim=bond_dim, rotation_driver=rotation_driver, convergence_thresh=convergence_thresh)
        if not converged:
            bond_dim += bond_dim_incr
    return ket_temp
