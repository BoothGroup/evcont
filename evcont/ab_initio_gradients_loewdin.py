import numpy as np

from pyscf import scf, ao2mo, grad

from evcont.ab_initio_eigenvector_continuation import (
    approximate_ground_state,
    approximate_multistate
)

from evcont.electron_integral_utils import get_loewdin_trafo


def get_overlap_grad(mol):
    """
    Calculate the gradient of the overlap matrix for a given molecule w.r.t. nuclear
    positions.

    Args:
        mol (Molecule): The molecule object.

    Returns:
        ndarray: The gradient of the overlap matrix.
    """

    inner_deriv = mol.intor("int1e_ipovlp", comp=3)

    deriv = np.zeros((3, mol.natm, mol.nao, mol.nao))

    # Loop over the nuclei in the molecule
    for i in range(mol.natm):
        _, _, x, y = mol.aoslice_by_atom()[i]

        deriv[:, i, x:y, :] -= inner_deriv[:, x:y, :]

    deriv = deriv + deriv.transpose(0, 1, 3, 2)

    # Transpose the return value to match the desired ordering of indices
    return np.transpose(deriv, (2, 3, 1, 0))


def loewdin_trafo_grad(overlap_mat):
    """
    Calculate the gradient of the Loewdin transformation. This also takes care of
    degeneracies by resorting to degenerate perturbation theory.

    Parameters:
    overlap_mat (np.ndarray): Matrix representing the overlap between atomic orbitals.

    Returns:
    np.ndarray: Gradient of the Loewdin transformation.
    """

    vals, vecs = np.linalg.eigh(overlap_mat)

    rounded_vals = np.round(vals, decimals=5)
    degenerate_vals = np.unique(rounded_vals)

    U_full = np.zeros((*overlap_mat.shape, *overlap_mat.shape))
    degenerate_subspace = np.zeros(overlap_mat.shape, dtype=bool)

    # Take care of degeneracies
    for val in degenerate_vals:
        degenerate_ids = (np.argwhere(rounded_vals == val)).flatten()
        subspace = vecs[:, degenerate_ids]

        V_projected = 0.5 * np.einsum(
            "ai,bj->abij", subspace, subspace
        ) + 0.5 * np.einsum("bi,aj->abij", subspace, subspace)

        # Get rotation to diagonalise V in degenerate subspace
        _, U = np.linalg.eigh(V_projected)
        U_full[
            np.ix_(
                np.ones(U_full.shape[0], dtype=bool),
                np.ones(U_full.shape[1], dtype=bool),
                degenerate_ids,
                degenerate_ids,
            )
        ] = U
        degenerate_subspace[np.ix_(degenerate_ids, degenerate_ids)] = True

    vecs_rotated = np.einsum("ij,abjk->abik", vecs, U_full)

    Vji = 0.5 * np.einsum(
        "abai,abbj->abij", vecs_rotated, vecs_rotated
    ) + 0.5 * np.einsum("abbi,abaj->abij", vecs_rotated, vecs_rotated)

    Zji = np.zeros((*overlap_mat.shape, *overlap_mat.shape))
    Zji[:, :, ~degenerate_subspace] = Vji[:, :, ~degenerate_subspace] / (
        (vals - np.expand_dims(vals, -1))[~degenerate_subspace]
    )

    dvecs = np.einsum("abij,abjk->abik", vecs_rotated, Zji)
    dvals = Vji[:, :, np.arange(Vji.shape[2]), np.arange(Vji.shape[3])]

    transformed_vals = np.where(vals > 1.0e-15, 1 / np.sqrt(vals), 0.0)
    d_transformed_vals = (
        np.where(vals > 1.0e-15, -(0.5 / np.sqrt(vals) ** 3), 0.0) * dvals
    )
    dS = (
        np.einsum("abij, abkj->abik", dvecs * transformed_vals, vecs_rotated)
        + np.einsum(
            "abij, abkj->abik",
            vecs_rotated * np.expand_dims(d_transformed_vals, axis=-2),
            vecs_rotated,
        )
        + np.einsum("abij, abkj->abik", vecs_rotated * transformed_vals, dvecs)
    )

    # Transpose the return value to match the desired ordering of indices
    return np.transpose(dS, (2, 3, 0, 1))


def get_derivative_ao_mo_trafo(mol):
    """
    Calculates the derivatives of the atomic orbital to molecular orbital
    transformation.

    Parameters:
        mol (Molecule): The molecule object.

    Returns:
        ndarray: The derivatives of the transformation matrix.
    """

    overlap_grad = get_overlap_grad(mol)
    trafo_grad = np.einsum(
        "ijkl, ijmn->klmn",
        loewdin_trafo_grad(mol.intor("int1e_ovlp")),
        overlap_grad,
    )

    return trafo_grad


def get_one_el_grad_ao(mol):
    """
    Calculate the one-electron integral derivatives in the AO basis.

    Parameters:
        mol (pyscf.gto.Mole): The molecular system.

    Returns:
        np.ndarray: The one-electron integral derivatives in the AO basis.
    """
    hcore_gen = grad.RHF(scf.RHF(mol)).hcore_generator()

    return_val = np.array([hcore_gen(i) for i in range(mol.natm)])

    # Transpose the return value to match the desired ordering of indices
    return np.transpose(return_val, (2, 3, 0, 1))


def get_one_el_grad(mol, ao_mo_trafo=None, ao_mo_trafo_grad=None):
    """
    Calculate the gradient of the one-electron integrals with respect to nuclear
    coordinates.

    Args:
        mol : Molecule object
            The molecule object representing the system.
        ao_mo_trafo : numpy.ndarray, optional
            The transformation matrix from atomic orbitals to molecular orbitals.
        ao_mo_trafo_grad : numpy.ndarray, optional
            The gradient of the transformation matrix from atomic orbitals to molecular
            orbitals.

    Returns:
        numpy.ndarray
            The gradient of the one-electron integrals.

    """
    if ao_mo_trafo is None:
        ao_mo_trafo = get_loewdin_trafo(mol.intor("int1e_ovlp"))

    h1_ao = scf.hf.get_hcore(mol)

    if ao_mo_trafo_grad is None:
        ao_mo_trafo_grad = get_derivative_ao_mo_trafo(mol)

    h1_grad_ao = get_one_el_grad_ao(mol)
    h1_grad = np.einsum("ijkl,im,mn->jnkl", ao_mo_trafo_grad, h1_ao, ao_mo_trafo)
    h1_grad += np.swapaxes(h1_grad, 0, 1)
    h1_grad += np.einsum("ij,iklm,kn->jnlm", ao_mo_trafo, h1_grad_ao, ao_mo_trafo)

    return h1_grad


def two_el_grad(h2_ao, two_rdm, ao_mo_trafo, ao_mo_trafo_grad, h2_ao_deriv, atm_slices):
    """
    Calculate the two-electron integral gradient.

    Args:
        h2_ao (np.ndarray): Two-electron integrals in atomic orbital basis.
        two_rdm (np.ndarray): Two-electron reduced density matrix.
        ao_mo_trafo (np.ndarray):
            Transformation matrix from atomic orbital to molecular orbital basis.
        ao_mo_trafo_grad (np.ndarray): Gradient of the transformation matrix.
        h2_ao_deriv (np.ndarray):
            Derivative of the two-electron integrals with respect to nuclear
            coordinates.
        atm_slices (list): List of atom index slices.

    Returns:
        np.ndarray: The two-electron gradient.

    """

    two_el_contraction = np.einsum(
        "ijkl,abcd,aimn,bj,ck,dl->mn",
        two_rdm
        + np.transpose(two_rdm, (1, 0, 2, 3))
        + np.transpose(two_rdm, (3, 2, 1, 0))
        + np.transpose(two_rdm, (2, 3, 0, 1)),
        h2_ao,
        ao_mo_trafo_grad,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo,
        optimize="optimal",
    )

    two_rdm_ao = np.einsum(
        "ijkl,ai,bj,ck,dl->abcd",
        two_rdm,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo,
        optimize="optimal",
    )

    two_el_contraction_from_grad = np.einsum(
        "nmbcd,abcd->nma",
        h2_ao_deriv,
        two_rdm_ao
        + np.transpose(two_rdm_ao, (1, 0, 3, 2))
        + np.transpose(two_rdm_ao, (2, 3, 0, 1))
        + np.transpose(two_rdm_ao, (3, 2, 1, 0)),
        optimize="optimal",
    )

    h2_grad_ao_b = np.zeros((3, len(atm_slices), two_rdm.shape[0], two_rdm.shape[1]))
    for i, slice in enumerate(atm_slices):
        # Subtract the gradient contribution from the contraction part
        h2_grad_ao_b[:, i, slice[0] : slice[1], :] -= two_el_contraction_from_grad[
            :, slice[0] : slice[1], :
        ]

    # Return the two-electron integral gradient
    return two_el_contraction + np.einsum("nmbb->mn", h2_grad_ao_b)


def get_grad_elec_OAO(mol, one_rdm, two_rdm, ao_mo_trafo=None, ao_mo_trafo_grad=None):
    """
    Calculates the gradient of the electronic energy based on one- and two-rdms
    in the OAO.

    Args:
        mol (object): Molecule object.
        one_rdm (ndarray): One-electron reduced density matrix.
        two_rdm (ndarray): Two-electron reduced density matrix.
        ao_mo_trafo (ndarray, optional):
            AO to MO transformation matrix. Is computed if not provided.
        ao_mo_trafo_grad (ndarray, optional):
            Gradient of AO to MO transformation matrix. Is computed if not provided.

    Returns:
        ndarray: Electronic gradient.
    """

    if ao_mo_trafo is None:
        ao_mo_trafo = get_loewdin_trafo(mol.intor("int1e_ovlp"))

    if ao_mo_trafo_grad is None:
        ao_mo_trafo_grad = get_derivative_ao_mo_trafo(mol)

    h1_jac = get_one_el_grad(
        mol, ao_mo_trafo=ao_mo_trafo, ao_mo_trafo_grad=ao_mo_trafo_grad
    )

    h2_ao = mol.intor("int2e")
    h2_ao_deriv = mol.intor("int2e_ip1", comp=3)

    two_el_gradient = two_el_grad(
        h2_ao,
        two_rdm,
        ao_mo_trafo,
        ao_mo_trafo_grad,
        h2_ao_deriv,
        tuple(
            [
                (mol.aoslice_by_atom()[i][2], mol.aoslice_by_atom()[i][3])
                for i in range(mol.natm)
            ]
        ),
    )

    grad_elec = (
        np.einsum("ij,ijkl->kl", one_rdm, h1_jac, optimize="optimal")
        + 0.5 * two_el_gradient
    )

    return grad_elec


def get_energy_with_grad(mol, one_RDM, two_RDM, S, hermitian=True):
    """
    Calculates the potential energy and its gradient w.r.t. nuclear positions of a
    molecule from the eigenvector continuation.

    Args:
        mol : pyscf.gto.Mole
            The molecule object.
        one_RDM : numpy.ndarray
            The one-electron t-RDM.
        two_RDM : numpy.ndarray
            The two-electron t-RDM.
        S : numpy.ndarray
            The overlap matrix.
        hermitian (bool, optional):
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        tuple
            A tuple containing the total potential energy and its gradient.
    """
    # Construct h1 and h2
    ao_mo_trafo = get_loewdin_trafo(mol.intor("int1e_ovlp"))

    h1 = np.linalg.multi_dot((ao_mo_trafo.T, scf.hf.get_hcore(mol), ao_mo_trafo))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, ao_mo_trafo), mol.nao)

    en, vec = approximate_ground_state(h1, h2, one_RDM, two_RDM, S, hermitian=hermitian)

    one_rdm_predicted = np.einsum("i,ijkl,j->kl", vec, one_RDM, vec, optimize="optimal")
    two_rdm_predicted = np.einsum(
        "i,ijklmn,j->klmn", vec, two_RDM, vec, optimize="optimal"
    )

    grad_elec = get_grad_elec_OAO(
        mol, one_rdm_predicted, two_rdm_predicted, ao_mo_trafo=ao_mo_trafo
    )

    return (
        en.real + mol.energy_nuc(),
        grad_elec + grad.RHF(scf.RHF(mol)).grad_nuc(),
    )


############################################################
# NEW MULTISTATE - SEPERATED FROM REST FOR TESTING
############################################################

def get_multistate_energy_with_grad(mol, one_RDM, two_RDM, S, nroots=1, hermitian=True):
    """
    Calculates the potential energy and its gradient w.r.t. nuclear positions of a
    molecule from the eigenvector continuation.

    Args:
        mol : pyscf.gto.Mole
            The molecule object.
        one_RDM : numpy.ndarray
            The one-electron t-RDM.
        two_RDM : numpy.ndarray
            The two-electron t-RDM.
        S : numpy.ndarray
            The overlap matrix.
        hermitian (bool, optional):
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        tuple
            A tuple containing the total potential energy and its gradient.
    """
    # Construct h1 and h2
    ao_mo_trafo = get_loewdin_trafo(mol.intor("int1e_ovlp"))

    h1 = np.linalg.multi_dot((ao_mo_trafo.T, scf.hf.get_hcore(mol), ao_mo_trafo))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, ao_mo_trafo), mol.nao)

    en, vec = approximate_multistate(h1, h2, one_RDM, two_RDM, S, nroots=nroots, hermitian=hermitian)

    grad_elec_all = []
    for i_state in range(nroots):
        vec_i = vec[i_state,:]

        one_rdm_predicted = np.einsum("i,ijkl,j->kl", vec_i, one_RDM, vec_i, optimize="optimal")
        two_rdm_predicted = np.einsum(
            "i,ijklmn,j->klmn", vec_i, two_RDM, vec_i, optimize="optimal"
        )
    
        grad_elec = get_grad_elec_OAO(
            mol, one_rdm_predicted, two_rdm_predicted, ao_mo_trafo=ao_mo_trafo
        )
        grad_elec_all.append(grad_elec)
        
    grad_elec_all = np.array(grad_elec_all)

    return (
        en.real + mol.energy_nuc(),
        grad_elec_all + grad.RHF(scf.RHF(mol)).grad_nuc(),
    )

if __name__ == '__main__':
    
    # Some initial checks for the code
    from pyscf import gto, fci
    from evcont.FCI_EVCont import FCI_EVCont_obj
    from evcont.ab_initio_eigenvector_continuation import approximate_multistate_OAO
    from evcont.electron_integral_utils import get_basis, get_integrals
    
    from pyscf.mcscf import CASCI

    nstate = 1 #1st excited state
    nroots_evcont = 3
    
    natom = 10
    

    def get_mol(positions):
        mol = gto.Mole()

        mol.build(
            atom=[("H", pos) for pos in positions],
            basis="sto-6g",
            symmetry=False,
            unit="Bohr",
        )

        return mol
    
    equilibrium_dist = 1.78596

    equilibrium_pos = np.array([(x * equilibrium_dist, 0.0, 0.0) for x in range(10)])

    training_stretches = np.array([0.0, 0.5, -0.5, 1.0, -1.0])

    trainig_dists = equilibrium_dist + training_stretches

    #trainig_dists = [1.0, 1.8, 2.6]

    continuation_object = FCI_EVCont_obj(nroots=nroots_evcont,
                                         cibasis='canonical')
    
        
    # Generate training data + prepare training models
    for i, dist in enumerate(trainig_dists):
        positions = [(x, 0.0, 0.0) for x in dist * np.arange(natom)]
        mol = get_mol(positions)
        continuation_object.append_to_rdms(mol)
    
    # Test the ground and excite state forces and gradients at training points
    for i, dist in enumerate(trainig_dists):
        positions = [(x, 0.0, 0.0) for x in dist * np.arange(natom)]
        mol = get_mol(positions)
        
        # Predictions from mutistate continuation
        en_continuation_ms, grad_continuation_ms = get_multistate_energy_with_grad(
            mol,
            continuation_object.one_rdm,
            continuation_object.two_rdm,
            continuation_object.overlap,
            nroots=nstate+1
        )

        # Predictions from Hartree-Fock
        hf_energy, hf_grad = mol.RHF().nuc_grad_method().as_scanner()(mol)

        # Fci reference values
        #en_exact, grad_exact = CASCI(mol.RHF(), natom, natom).nuc_grad_method().as_scanner()(mol)
        
        # Fci excited state reference values
        mc = CASCI(mol.RHF(mol), natom, natom)
        mc.fcisolver = fci.direct_spin0.FCI()
        mc.fcisolver.nroots = nroots_evcont
        ci_scan_exc = mc.nuc_grad_method().as_scanner(state=nstate)
        ci_scan_0 = mc.nuc_grad_method().as_scanner(state=0)
        
        en_exc_exact, grad_exc_exact = ci_scan_exc(mol)
        en_exact, grad_exact = ci_scan_0(mol)
        
        
        # Checks
        assert np.allclose(en_exact,en_continuation_ms[0])
        assert np.allclose(grad_exact,grad_continuation_ms[0])

        assert np.allclose(en_exc_exact,en_continuation_ms[nstate])        
        assert np.allclose(grad_exc_exact,grad_continuation_ms[nstate])
        
    
    '''
    radius = 0.4
    #radius_l = [0.1,0.4]
    
    n_data_points = 50
    seed = 1
    
    norb = nelec = 10


    
    rng = np.random.default_rng(seed)



    # Testing
    en_exact_l = []; grad_exact_l = []
    en_continuation_l = []; grad_continuation_l = []
    
    en_exc_exact_l = []; grad_exc_exact_l = []
    en_continuation_ms_l = []; grad_continuation_ms_l = []
    en_continuation_exc_l = []; grad_continuation_exc_l = []
    
    for i in range(n_data_points):
        # Sample a new test geometry
        displacement_theta = rng.random(size=(10)) * np.pi
        displacement_phi = rng.random(size=(10)) * 2 * np.pi

        sampled_displacement_x = (
            radius * np.sin(displacement_theta) * np.cos(displacement_phi)
        )
        sampled_displacement_y = (
            radius * np.sin(displacement_theta) * np.sin(displacement_phi)
        )
        sampled_displacement_z = radius * np.cos(displacement_theta)

        sampled_displacement = np.stack(
            (sampled_displacement_x, sampled_displacement_y, sampled_displacement_z),
            axis=-1,
        )

        sampled_pos = equilibrium_pos + sampled_displacement
        mol = get_mol(sampled_pos)

        # Predictions from continuation
        en_continuation, grad_continuation = get_energy_with_grad(
            mol,
            continuation_object.one_rdm,
            continuation_object.two_rdm,
            continuation_object.overlap,
        )
        
        # Predictions from mutistate continuation
        en_continuation_ms, grad_continuation_ms = get_multistate_energy_with_grad(
            mol,
            continuation_object.one_rdm,
            continuation_object.two_rdm,
            continuation_object.overlap,
            nroots=nstate+1
        )

        # Predictions from Hartree-Fock
        hf_energy, hf_grad = mol.RHF().nuc_grad_method().as_scanner()(mol)

        # Fci reference values
        en_exact, grad_exact = CASCI(mol.RHF(), 10, 10).nuc_grad_method().as_scanner()(mol)
        
        # Fci excited state reference values
        mc = CASCI(mol.RHF(mol), 10, 10)
        mc.fcisolver.nroots = nstate+2
        ci_scan_exc = mc.nuc_grad_method().as_scanner(state=nstate)
        #ci_scan_0 = mc.nuc_grad_method().as_scanner(state=0)
        
        en_exc_exact, grad_exc_exact = ci_scan_exc(mol)
        #en_exact, grad_exact = ci_scan_0(mol)
    
        # Save
        en_exact_l.append(en_exact)
        grad_exact_l.append(grad_exact)
        en_continuation_l.append(en_continuation)
        grad_continuation_l.append(grad_continuation)
        
        en_continuation_ms_l.append(np.atleast_1d(en_continuation_ms)[0])
        grad_continuation_ms_l.append(grad_continuation_ms[0,:,:])
        
        if nstate > 0:
            en_continuation_exc_l.append(np.atleast_1d(en_continuation_ms)[nstate])
            grad_continuation_exc_l.append(grad_continuation_ms[nstate,:,:])
            en_exc_exact_l.append(en_exc_exact)
            grad_exc_exact_l.append(grad_exc_exact)
        
    grad_continuation_l = np.array(grad_continuation_l)
    grad_exact_l = np.array(grad_exact_l)
    en_continuation_l = np.array(en_continuation_l)
    en_exact_l = np.array(en_exact_l)
    
    grad_continuation_ms_l = np.array(grad_continuation_ms_l)
    en_continuation_ms_l = np.array(en_continuation_ms_l)
    grad_continuation_exc_l = np.array(grad_continuation_exc_l)
    en_continuation_exc_l = np.array(en_continuation_exc_l)
    
    en_exc_exact_l = np.array(en_exc_exact_l)
    grad_exc_exact_l = np.array(grad_exc_exact_l)

    # 
    assert np.allclose(grad_continuation_ms_l,grad_continuation_l)
    
    # Compute the differences
    grad_diff = np.abs(grad_continuation_l - grad_exact_l)
    en_diff = en_continuation_l - en_exact_l
    en_diff_rel  = np.abs(en_diff/en_exact_l)
    
    grad_ms_diff = np.abs(grad_continuation_ms_l - grad_exact_l)
    en_ms_diff = en_continuation_ms_l - en_exact_l
    
    grad_exc_diff = np.abs(grad_continuation_exc_l - grad_exc_exact_l)
    en_exc_diff = en_continuation_exc_l - en_exc_exact_l
    en_exc_diff_rel = np.abs(en_exc_diff/en_exc_exact_l)

    
    print(f'Force difference - ground state: {grad_diff.mean():.4e} ({grad_diff.min():.4e} - {grad_diff.max():.4e})')
    print(f'Energy difference - ground state: {en_diff.mean():.4e} ({en_diff.min():.4e} - {en_diff.max():.4e})')
    print(f'Energy difference - ground state: {en_diff_rel.mean():.4e} ({en_diff_rel.min():.4e} - {en_diff_rel.max():.4e})')
    
    print('-- Multistate')
    print(f'Force difference - ground state: {grad_ms_diff.mean():.4e} ({grad_ms_diff.min():.4e} - {grad_ms_diff.max():.4e})')
    print(f'Energy difference - ground state: {en_ms_diff.mean():.4e} ({en_ms_diff.min():.4e} - {en_ms_diff.max():.4e})')
    
    
    print('-- Multistate - excited')
    print(f'Force difference - excited state: {grad_exc_diff.mean():.4e} ({grad_exc_diff.min():.4e} - {grad_exc_diff.max():.4e})')
    print(f'Energy difference - excited state: {en_exc_diff.mean():.4e} ({en_exc_diff.min():.4e} - {en_exc_diff.max():.4e})')
    print(f'Relative energy difference - excited state: {en_exc_diff_rel.mean():.4e} ({en_exc_diff_rel.min():.4e} - {en_exc_diff_rel.max():.4e})')
    
    
    '''
    