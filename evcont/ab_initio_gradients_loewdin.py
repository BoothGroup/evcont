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
        np.ndarray(nel,nel,nat,3): 
            The one-electron integral derivatives in the AO basis.
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
        numpy.ndarray(nel,nel,nat,3): 
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

def get_multistate_energy_with_grad_old(mol, one_RDM, two_RDM, S, nroots=1, hermitian=True):
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

def get_multistate_energy_with_grad_and_NAC_old(mol, one_RDM, two_RDM, S, nroots=1, hermitian=True):
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


    grad_nuc = grad.RHF(scf.RHF(mol)).grad_nuc()
        
    grad_elec_all = []
    nac_all = {}
    for i_state in range(nroots):
        vec_i = vec[i_state,:]

        for j_state in range(nroots):
            vec_j = vec[j_state,:]
            
            one_rdm_predicted = np.einsum("i,ijkl,j->kl", vec_i, one_RDM, vec_j, optimize="optimal")
            two_rdm_predicted = np.einsum(
                "i,ijklmn,j->klmn", vec_i, two_RDM, vec_j, optimize="optimal"
            )
        
            grad_elec = get_grad_elec_OAO(
                mol, one_rdm_predicted, two_rdm_predicted, ao_mo_trafo=ao_mo_trafo
            )
            
            # Energy gradients
            if i_state == j_state:
                grad_elec_all.append(grad_elec)
                
            # Nonadiabatic couplings
            else:
                nac_ij = grad_elec/(en[j_state]-en[i_state])
                nac_all[str(i_state)+str(j_state)] = nac_ij

    
    grad_all = np.array(grad_elec_all) + grad_nuc

    return (
        en.real + mol.energy_nuc(),
        grad_all,
        nac_all
    )

def get_two_el_grad(h2_ao, ao_mo_trafo, ao_mo_trafo_grad, h2_ao_deriv, atm_slices):
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
  
    two_el_contraction_ao = np.einsum(
        "abcd,aimn,bj,ck,dl->ijklmn",
        h2_ao,
        ao_mo_trafo_grad,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo,
        optimize="optimal",
    )
    
    two_el_contraction_ao += \
        np.transpose(two_el_contraction_ao,(1, 0, 2, 3,4,5)) +\
        np.transpose(two_el_contraction_ao,(3, 2, 1, 0,4,5)) +\
        np.transpose(two_el_contraction_ao,(2, 3, 0, 1,4,5))
    
    two_el_contraction_from_grad_ao = np.einsum(
        "nmbcd,ai,bj,ck,dl->ijklnma",
        h2_ao_deriv,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo,
        optimize="optimal",
    )
    
    two_el_contraction_from_grad_ao += \
        np.transpose(two_el_contraction_from_grad_ao,(1, 0, 2, 3, 4,5,6)) +\
        np.transpose(two_el_contraction_from_grad_ao,(3, 2, 1, 0, 4,5,6)) +\
        np.transpose(two_el_contraction_from_grad_ao,(2, 3, 0, 1, 4,5,6))

    h2_grad_ao_b = np.zeros((h2_ao.shape[0], h2_ao.shape[1], h2_ao.shape[2], h2_ao.shape[3], 3, len(atm_slices), h2_ao.shape[0], h2_ao.shape[1]))
    for i, slice in enumerate(atm_slices):
        # Subtract the gradient contribution from the contraction part
        h2_grad_ao_b[:,:,:,:, :, i, slice[0] : slice[1], :] -= two_el_contraction_from_grad_ao[
            :,:,:,:, :, slice[0] : slice[1], :
        ]

    h2_grad = two_el_contraction_ao + np.einsum(
        "ijklnmbb->ijklmn",
        h2_grad_ao_b,
        optimize="optimal",
    ) 
    
    # Return the two-electron integral gradient
    return h2_grad

def get_one_and_two_el_grad(mol,ao_mo_trafo=None, ao_mo_trafo_grad=None):
    """
    Calculates the gradient of the one- and two-electron integrals
    in the OAO.

    Args:
        mol (object): Molecule object.
        ao_mo_trafo (ndarray, optional):
            AO to MO transformation matrix. Is computed if not provided.
        ao_mo_trafo_grad (ndarray, optional):
            Gradient of AO to MO transformation matrix. Is computed if not provided.

    Returns:
        tuple of np.ndarray:
            One- and two-electron gradients.
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

    h2_jac =  get_two_el_grad(
        h2_ao,
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

    
    return (h1_jac, h2_jac)


def get_grad_elec_from_gradH(one_rdm, two_rdm, h1_jac, h2_jac):
    """
    Calculates the continuation gradient from the one- and two-electron 
    integrals derivatives.

    Args:
        mol (object): Molecule object.
        one_RDM : numpy.ndarray
            The one-electron t-RDM.
        two_RDM : numpy.ndarray
            The two-electron t-RDM.
        h1_jac : numpy.ndarray
            The gradient of the one-electron integrals.
        h2_jac : numpy.ndarray
            The gradient of the two-electron integrals.
        
    Returns:
        ndarray: Electronic gradient.
    """
    
    
    two_el_gradient = np.einsum(
            "ijkl,ijklmn->mn",
            two_rdm,
            h2_jac,
            optimize="optimal",
            )
    
    grad_elec = (
        np.einsum("ij,ijkl->kl", one_rdm, h1_jac, optimize="optimal")
        + 0.5 * two_el_gradient
    )
    
    return grad_elec

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
    
    # Note: Get the gradient of one and two-electron integrals 
    h1_jac, h2_jac = get_one_and_two_el_grad(mol,ao_mo_trafo=ao_mo_trafo)
    
    grad_elec_all = []
    for i_state in range(nroots):
        vec_i = vec[i_state,:]

        one_rdm_predicted = np.einsum("i,ijkl,j->kl", vec_i, one_RDM, vec_i, optimize="optimal")
        two_rdm_predicted = np.einsum(
            "i,ijklmn,j->klmn", vec_i, two_RDM, vec_i, optimize="optimal"
        )
        
        grad_elec = get_grad_elec_from_gradH(
            one_rdm_predicted, two_rdm_predicted, h1_jac, h2_jac
        )
        grad_elec_all.append(grad_elec)
        
    grad_elec_all = np.array(grad_elec_all)

    return (
        en.real + mol.energy_nuc(),
        grad_elec_all + grad.RHF(scf.RHF(mol)).grad_nuc(),
    )

def get_multistate_energy_with_grad_and_NAC(mol, one_RDM, two_RDM, S, nroots=1, hermitian=True):
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

    # Note: Get the gradient of one and two-electron integrals 
    h1_jac, h2_jac = get_one_and_two_el_grad(mol,ao_mo_trafo=ao_mo_trafo)
    
    grad_elec_all = []
    for i_state in range(nroots):
        vec_i = vec[i_state,:]


    grad_nuc = grad.RHF(scf.RHF(mol)).grad_nuc()
        
    grad_elec_all = []
    nac_all = {}
    for i_state in range(nroots):
        vec_i = vec[i_state,:]

        for j_state in range(nroots):
            vec_j = vec[j_state,:]
            
            one_rdm_predicted = np.einsum("i,ijkl,j->kl", vec_i, one_RDM, vec_j, optimize="optimal")
            two_rdm_predicted = np.einsum(
                "i,ijklmn,j->klmn", vec_i, two_RDM, vec_j, optimize="optimal"
            )
        
            grad_elec = get_grad_elec_from_gradH(
                one_rdm_predicted, two_rdm_predicted, h1_jac, h2_jac
            )
            
            # Energy gradients
            if i_state == j_state:
                grad_elec_all.append(grad_elec)
                
            # Nonadiabatic couplings
            else:
                nac_ij = grad_elec/(en[j_state]-en[i_state])
                nac_all[str(i_state)+str(j_state)] = nac_ij

    
    grad_all = np.array(grad_elec_all) + grad_nuc

    return (
        en.real + mol.energy_nuc(),
        grad_all,
        nac_all
    )

if __name__ == '__main__':
    
    # Some initial checks for the code
    from pyscf import gto, fci
    from evcont.FCI_EVCont import FCI_EVCont_obj
    from evcont.ab_initio_eigenvector_continuation import approximate_multistate_OAO
    from evcont.electron_integral_utils import get_basis, get_integrals
    
    from pyscf.mcscf import CASCI
    
    from pyscf.fci.addons import overlap
    
    from functools import reduce

    from time import time
    
    #CASE = 'NAC'
    CASE = 'Exc-Grad'
    
    nstate = 3 #1st excited state
    nroots_evcont = 4
    cibasis = 'canonical'
    
    natom = 6
    
    test_range = np.linspace(0.8, 3.0,20)
    

    def get_mol(positions):
        mol = gto.Mole()

        mol.build(
            atom=[("H", pos) for pos in positions],
            basis="sto-6g",
            symmetry=False,
            unit="Bohr",
            verbose=0
        )

        return mol
    
    def nac_fd_FCI(mf,mol=None,nroots=None,dx=1e-6,cibasis='canonical'):
        """ 
        Compute the nonadiabatic coupling from the central finite difference
        ( \braket{\psi_a(x)|psi_b(x+dx)} - \braket{\psi_a(x)|psi_b(x-dx)} ) / (2*dx)
        for dx along each atomic and Cartesian coordinate
        
        Args:
            mf: pyscf.mcscf.CASCI
                The CASCI solver.
            mol (optional): pyscf.gto.Mole
                The molecule object.
            nroots (optional): int
                Number of states in the solver.
            dx (optional): float
                Grid spacing in the finite difference solution
            cibasis (optional): str
                Single particle basis to use in the FCI solution
        
        Returns:
            nac_all: dictionary of np.darray(nat,3)
                Nonadiabatic coupling vectors between all states,
                e.g. nac_all['02'] is NAC along ground state and 2nd excited state
                
                Note: nac_all['ii'] is technically not well-defined but still 
                      computed for testing purposes
        
        """
        # Variables
        if mol == None:
            mol = mf.mol
        if nroots == None:
            nroots = mf.fcisolver.nroots
                
        mverbose = mol.verbose
        mol.verbose = 0
        coord = mol.atom_coords()
        
        # At given molecular coordinates
        basis_0 = get_basis(mol,cibasis)
        h1, h2 = get_integrals(mol, basis_0)
        _, fcivec_0 = mf.fcisolver.kernel(h1, h2, mol.nao, mol.nelec, tol=1.e-14, max_space=30,  nroots=6, max_cycle=250)
        
        mol0 = mol.copy()
        
        # Initialize nac
        nac_all = {}
        for i in range(nroots):
            for j in range(nroots):
                nac_all[str(i)+str(j)]=np.zeros([mol.natm,3])
                
        for ni in range(mol.natm):
            ptr = mol._atm[ni,gto.PTR_COORD]
            for i in range(3):
                # Positive
                mol._env[ptr+i] = coord[ni,i] + dx
                
                basis_pos = get_basis(mol,cibasis)
                h1, h2 = get_integrals(mol, basis_pos)
                _, fcivec_pos = mf.fcisolver.kernel(h1, h2, mol.nao, mol.nelec, tol=1.e-14, max_space=30, nroots=6, max_cycle=250)
                
                # Single particle basis overlap
                s12 = gto.intor_cross('cint1e_ovlp_sph', mol0, mol)
                s12_pos = np.einsum('ji,jk,kl->il',basis_0,s12,basis_pos)
                
                # Negative
                mol._env[ptr+i] = coord[ni,i] - dx

                basis_neg = get_basis(mol,cibasis)
                h1, h2 = get_integrals(mol, basis_neg)
                _, fcivec_neg = mf.fcisolver.kernel(h1, h2, mol.nao, mol.nelec, tol=1.e-14, max_space=30, nroots=6, max_cycle=250)
                
                # Single particle basis overlap
                s12 = gto.intor_cross('cint1e_ovlp_sph', mol0, mol)
                s12_neg = np.einsum('ji,jk,kl->il',basis_0,s12,basis_neg)

                # Iterate over different states
                for istate in range(nroots):
                    for jstate in range(nroots):
                        # Overlaps
                        ov1a = overlap(fcivec_0[istate],fcivec_pos[jstate],mol0.nao,mol0.nelec,s12_pos)
                        ov1b = overlap(fcivec_0[istate],fcivec_neg[jstate],mol0.nao,mol0.nelec,s12_neg)
                        
                        # Central difference
                        nac_all[str(istate)+str(jstate)][ni,i] = (ov1a-ov1b)/(2*dx)
                
                # Return to original coordinates for next iterations
                mol._env[ptr+i] = coord[ni,i]
            
        mol.verbose = mverbose
        
        return nac_all
        
    
    equilibrium_dist = 1.78596

    equilibrium_pos = np.array([(x * equilibrium_dist, 0.0, 0.0) for x in range(10)])

    training_stretches = np.array([0.0, 0.5, -0.5, 1.0, -1.0])

    trainig_dists = equilibrium_dist + training_stretches

    #trainig_dists = [1.0, 1.8, 2.6]

    continuation_object = FCI_EVCont_obj(nroots=nroots_evcont,
                                         cibasis=cibasis)
    
        
    # Generate training data + prepare training models
    for i, dist in enumerate(trainig_dists):
        positions = [(x, 0.0, 0.0) for x in dist * np.arange(natom)]
        mol = get_mol(positions)
        continuation_object.append_to_rdms(mol)
    
    if CASE == 'Exc-Grad':
        
        st = time()
        # Test the ground and excite state forces and gradients at training points
        for i, dist in enumerate(trainig_dists):
            positions = [(x, 0.0, 0.0) for x in dist * np.arange(natom)]
            mol = get_mol(positions)
            
            # Predictions from mutistate continuation
            """
            en_continuation_ms_old, grad_continuation_ms_old = get_multistate_energy_with_grad(
                mol,
                continuation_object.one_rdm,
                continuation_object.two_rdm,
                continuation_object.overlap,
                nroots=nstate+1
            )
            """
            en_continuation_ms, grad_continuation_ms, nac_continuation = get_multistate_energy_with_grad_and_NAC(
                mol,
                continuation_object.one_rdm,
                continuation_object.two_rdm,
                continuation_object.overlap,
                nroots=nstate+1
            )
            #"""
    
            # Predictions from Hartree-Fock
            hf_energy, hf_grad = mol.RHF().nuc_grad_method().as_scanner()(mol)
    
            # Fci reference values
            #en_exact, grad_exact = CASCI(mol.RHF(), natom, natom).nuc_grad_method().as_scanner()(mol)
            
            # Fci excited state reference values
            mc = CASCI(mol.RHF(), natom, natom)
            mc.fcisolver = fci.direct_spin0.FCI()
            mc.fcisolver.nroots = nstate+1
            ci_scan_exc = mc.nuc_grad_method().as_scanner(state=nstate)
            ci_scan_0 = mc.nuc_grad_method().as_scanner(state=0)
            
            en_exc_exact, grad_exc_exact = ci_scan_exc(mol)
            en_exact, grad_exact = ci_scan_0(mol)
            
            # Get the reference numerical FCI NACs 
            #nac_all = nac_fd_FCI(mc,nroots=nstate+1)
            #print(nac_all)
            
            # Checks
            print(i)
            
            assert np.allclose(en_exact,en_continuation_ms[0])
            assert np.allclose(grad_exact,grad_continuation_ms[0])
    
            assert np.allclose(en_exc_exact,en_continuation_ms[nstate])        
            assert np.allclose(grad_exc_exact,grad_continuation_ms[nstate],atol=1e-6)
            
        print(f'Time taken: {time()-st:.1f} sec')
            
        
    # Test NACs
    if CASE == 'NAC':
        
        fci_en = np.zeros([len(test_range),nstate+1])
        fci_nac = []
        cont_en = np.zeros([len(test_range),nstate+1])
        cont_nac = []
        
        for i, test_dist in enumerate(test_range):
            print(i)
            positions = [(x, 0.0, 0.0) for x in test_dist * np.arange(natom)]
            mol = get_mol(positions)
            h1, h2 = get_integrals(mol, get_basis(mol))
            
            mc = CASCI(mol.RHF(), natom, natom)
            mc.fcisolver = fci.direct_spin0.FCI()
            mc.fcisolver.nroots = nstate+1
            #ci_scan_exc = mc.nuc_grad_method().as_scanner(state=nstate)
            #ci_scan_0 = mc.nuc_grad_method().as_scanner(state=0)
            
            #en_exc_exact, grad_exc_exact = ci_scan_exc(mol)
            #en_exact, grad_exact = ci_scan_0(mol)
            en_exact, fcivec_pos = mc.fcisolver.kernel(h1, h2, mol.nao, mol.nelec)

            en_exact += mol.energy_nuc()
            
            # Get the reference numerical FCI NACs 
            nac_all = nac_fd_FCI(mc,nroots=nstate+1,cibasis=cibasis)
            
            fci_en[i,:] = en_exact
            fci_nac += [nac_all]
            
            # Continuation
            en_continuation_ms, _, nac_continuation = get_multistate_energy_with_grad_and_NAC(
                mol,
                continuation_object.one_rdm,
                continuation_object.two_rdm,
                continuation_object.overlap,
                nroots=nstate+1
            )
            
            cont_en[i,:] = en_continuation_ms
            cont_nac += [nac_continuation]
            
        #######################################################################
        # Plot NAC comparison
        import matplotlib.pylab as plt
        
        fci_absh = {}
        cont_absh = {}
        for istate in range(nstate+1):
            for jstate in range(nstate+1):
                if istate != jstate:
    
                    st_label = str(istate)+str(jstate)
                    fci_absh[st_label] = [np.abs(fci_nac[i][st_label]).sum() for i in range(len(test_range))]
                    cont_absh[st_label] = [np.abs(cont_nac[i][st_label]).sum() for i in range(len(test_range))]
            
        # Colors
        clr_st = {'01':'b', '10':'b',
                  '02':'r','20':'r',
                  '03':'pink','30':'pink',
                  '13':'y','31':'y',
                  '23':'violet','32':'violet',
                  '12':'g','21':'g'}
        labelsize = 15
        # Plot
        fig, axes = plt.subplots(nrows=2,ncols=2,sharex=True,sharey='row',
                                 figsize=[10,10],gridspec_kw={'hspace':0.,'wspace':0},
                                 height_ratios=[1,1])
        
        axes[0][0].plot(test_range,fci_en,'k',alpha=0.8)
        axes[0][1].plot(test_range,cont_en,'k',alpha=0.8)
        
        for key, el in fci_absh.items():
            axes[1][0].plot(test_range,fci_absh[key],label=key,c=clr_st[key])
            axes[1][1].plot(test_range,cont_absh[key],label=key,c=clr_st[key])
    
        axes[1][0].legend(loc='upper right')
        
        axes[0][0].set_title('FCI',fontsize=labelsize)
        axes[0][1].set_title('EVcont',fontsize=labelsize)
        axes[1][0].set_ylabel(r'$||\mathbf{d}_{ij}||$ (a$_0$$^{-1}$)',fontsize=labelsize)
        axes[0][0].set_ylabel(r'Energy (Hartree)',fontsize=labelsize)
        
        axes[1][0].set_ylim(ymin=0,ymax=min(10,axes[1][0].get_ylim()[1]))
        
        plt.show()
    
    '''
    from pyscf import ci
    from functools import reduce

    myhf1 = gto.M(atom='H 0 0 0; F 0 0 1.1', basis='6-31g', verbose=0).apply(scf.RHF).run()
    ci1 = ci.CISD(myhf1).run()
    print('CISD energy of mol1', ci1.e_tot)
    
    myhf2 = gto.M(atom='H 0 0 0; F 0 0 1.2', basis='6-31g', verbose=0).apply(scf.RHF).run()
    ci2 = ci.CISD(myhf2).run()
    print('CISD energy of mol2', ci2.e_tot)
    
    s12 = gto.intor_cross('cint1e_ovlp_sph', myhf1.mol, myhf2.mol)
    s12 = reduce(np.dot, (myhf1.mo_coeff.T, s12, myhf2.mo_coeff))
    nmo = myhf2.mo_energy.size
    nocc = myhf2.mol.nelectron // 2
    print('<CISD-mol1|CISD-mol2> = ', ci.cisd.overlap(ci1.ci, ci2.ci, nmo, nocc, s12))
    '''
    
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
    
    '''    
        def nac_fd_FCI_old(mf,mol=None,nroots=None,dx=1e-6):
            """ 
            Compute the nonadiabatic coupling from the central finite difference
            """
            # Variables
            if mol == None:
                mol = mf.mol
            if nroots == None:
                nroots = mf.fcisolver.nroots
                    
            mverbose = mol.verbose
            mol.verbose = 0
            coord = mol.atom_coords()
            
            
            # At given molecular coordinates
            basis_0 = get_basis(mol,'canonical')
            h1, h2 = get_integrals(mol, basis_0)
            _, fcivec_0 = mf.fcisolver.kernel(h1, h2, mol.nao, mol.nelec)
            
            mol0 = mol.copy()
            
            nac_all = {}
            for istate in range(nroots):
                for jstate in range(nroots):
                    #print(istate)
                    de = []
                    for ni in range(mol.natm):
                        de_i = []
                        ptr = mol._atm[ni,gto.PTR_COORD]
                        for i in range(3):
                            # Positive
                            mol._env[ptr+i] = coord[ni,i] + dx
                            basis_pos = get_basis(mol,'canonical')
                            h1, h2 = get_integrals(mol, basis_pos)
                            _, fcivec_pos = mf.fcisolver.kernel(h1, h2, mol.nao, mol.nelec)
                            
                            # Single particle overlap
                            s12 = gto.intor_cross('cint1e_ovlp_sph', mol0, mol)
                            s12_pos = reduce(np.dot, (basis_0.T, s12, basis_pos))
                
                            # Negative
                            mol._env[ptr+i] = coord[ni,i] - dx
                            basis_neg = get_basis(mol,'canonical')
                            h1, h2 = get_integrals(mol, basis_neg)
                            _, fcivec_neg = mf.fcisolver.kernel(h1, h2, mol.nao, mol.nelec)
                            
                            # Single particle overlap
                            s12 = gto.intor_cross('cint1e_ovlp_sph', mol0, mol)
                            s12_neg = reduce(np.dot, (basis_0.T, s12, basis_neg))
                            
                            # Overlaps
                            ov1a = overlap(fcivec_0[istate],fcivec_pos[jstate],mol.nao,mol.nelec,s12_pos)
                            ov1b = overlap(fcivec_0[istate],fcivec_neg[jstate],mol.nao,mol.nelec,s12_neg)
                            
                            # Central difference
                            de_i.append((ov1a-ov1b)/(2*dx))
                            
                            mol._env[ptr+i] = coord[ni,i]
                        de.append(de_i)
                        
                    nac_all[str(istate)+str(jstate)] = np.array(de)
                        
            mol.verbose = mverbose
            
            return nac_all
        

    '''