import numpy as np

from pyscf import scf, ao2mo, grad, lib

from evcont.ab_initio_eigenvector_continuation import (
    approximate_ground_state,
    approximate_multistate,
    solve_subspace
)

from evcont.electron_integral_utils import (
    get_loewdin_trafo,
    restore_electron_exchange_symmetry,
    get_df_integrals
)

#from evcont.low_rank_utils_mpi import (
from evcont.low_rank_utils import (
    get_jk_builds, unstack_tril
)

from evcont.logging_utils import logger, log_time, timeit

import sys


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

@timeit
def loewdin_trafo_grad(overlap_mat, degeneracy_precision=7):
    """
    Calculate the gradient of the Loewdin transformation. This also takes care of
    degeneracies by resorting to degenerate perturbation theory.

    Parameters:
    overlap_mat (np.ndarray): Matrix representing the overlap between atomic orbitals.
    degeneracy_precision (int): repurposed to set an absolute tolerance tol = 10**(-degeneracy_precision)
        for treating eigenvalues as degenerate. Lower values make the tolerance smaller.

    Returns:
    np.ndarray: Gradient of the Loewdin transformation.
    """

    vals, vecs = np.linalg.eigh(overlap_mat)

    # Determine degeneracy by small differences in eigenvalues rather than rounding
    tol = 10.0 ** (-degeneracy_precision)
    n = vals.shape[0]
    assigned = np.zeros(n, dtype=bool)
    degenerate_groups = []
    for i in range(n):
        if assigned[i]:
            continue
        same = np.where(np.abs(vals - vals[i]) <= tol)[0]
        assigned[same] = True
        degenerate_groups.append(same)

    U_full = np.zeros((*overlap_mat.shape, *overlap_mat.shape))
    degenerate_subspace = np.zeros(overlap_mat.shape, dtype=bool)

    # Take care of degeneracies
    for degenerate_ids in degenerate_groups:
        subspace = vecs[:, degenerate_ids]

        V_projected = 0.5 * lib.einsum(
            "ai,bj->abij", subspace, subspace
        ) + 0.5 * lib.einsum("bi,aj->abij", subspace, subspace, optimize='optimal')

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

    vecs_rotated = lib.einsum("ij,abjk->abik", vecs, U_full, optimize='optimal')

    Vji = 0.5 * lib.einsum(
        "abai,abbj->abij", vecs_rotated, vecs_rotated, optimize='optimal'
    ) + 0.5 * lib.einsum("abbi,abaj->abij", vecs_rotated, vecs_rotated, optimize='optimal')

    Zji = np.zeros((*overlap_mat.shape, *overlap_mat.shape))
    Zji[:, :, ~degenerate_subspace] = (
        Vji[:, :, ~degenerate_subspace]
        / ((vals - np.expand_dims(vals, -1))[~degenerate_subspace])
    )

    dvecs = lib.einsum("abij,abjk->abik", vecs_rotated, Zji, optimize='optimal')
    dvals = Vji[:, :, np.arange(Vji.shape[2]), np.arange(Vji.shape[3])]

    transformed_vals = np.where(vals > 1.0e-15, 1 / np.sqrt(vals), 0.0)
    d_transformed_vals = (
        np.where(vals > 1.0e-15, -(0.5 / np.sqrt(vals) ** 3), 0.0) * dvals
    )
    dS = (
        lib.einsum("abij, abkj->abik", dvecs * transformed_vals, vecs_rotated, optimize='optimal')
        + lib.einsum(
            "abij, abkj->abik",
            vecs_rotated * np.expand_dims(d_transformed_vals, axis=-2),
            vecs_rotated,
            optimize='optimal'
        )
        + lib.einsum("abij, abkj->abik", vecs_rotated * transformed_vals, dvecs, optimize='optimal')
    )

    # Transpose the return value to match the desired ordering of indices
    return np.transpose(dS, (2, 3, 0, 1))

@timeit
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
    trafo_grad = lib.einsum(
        "ijkl, ijmn->klmn",
        loewdin_trafo_grad(mol.intor("int1e_ovlp")),
        overlap_grad,
    )

    return trafo_grad

def fix_gauge(vec):
    """
    Make so that the first element is always positive
    """
    for vec_i in vec:
        idx = np.unravel_index(np.argmax(np.abs(vec_i.real)),vec_i.shape)
        vec_i *= -np.sign(vec_i[idx])
            
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


def get_one_el_grad(mol, h1_ao=None, ao_mo_trafo=None, ao_mo_trafo_grad=None):
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

    if h1_ao is None:
        h1_ao = scf.hf.get_hcore(mol)

    if ao_mo_trafo_grad is None:
        ao_mo_trafo_grad = get_derivative_ao_mo_trafo(mol)

    h1_grad_ao = get_one_el_grad_ao(mol)
    h1_grad = lib.einsum("ijkl,im,mn->jnkl", ao_mo_trafo_grad, h1_ao, ao_mo_trafo)
    h1_grad += np.swapaxes(h1_grad, 0, 1)
    h1_grad += lib.einsum("ij,iklm,kn->jnlm", ao_mo_trafo, h1_grad_ao, ao_mo_trafo)

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

    two_el_contraction = lib.einsum(
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

    two_rdm_ao = lib.einsum(
        "ijkl,ai,bj,ck,dl->abcd",
        two_rdm,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo,
        optimize="optimal",
    )

    two_el_contraction_from_grad_traced = lib.einsum(
        "nmbcd,mbcd->nm",
        h2_ao_deriv,
        two_rdm_ao
        + np.transpose(two_rdm_ao, (1, 0, 3, 2))
        + np.transpose(two_rdm_ao, (2, 3, 0, 1))
        + np.transpose(two_rdm_ao, (3, 2, 1, 0)),
        optimize="optimal",
    )

    h2_grad_ao_sum = np.zeros((len(atm_slices),3))
    for i, slice in enumerate(atm_slices):
        # Subtract the gradient contribution from the contraction part
        h2_grad_ao_sum[i,:] -= two_el_contraction_from_grad_traced[
            :, slice[0] : slice[1]
        ].sum(axis=1)

    # Return the two-electron integral gradient
    return h2_grad_ao_sum + two_el_contraction


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
        lib.einsum("ij,ijkl->kl", one_rdm, h1_jac, optimize="optimal")
        + 0.5 * two_el_gradient
    )

    return grad_elec


def get_grad_elec_OAO_customERI(mol, h2_ao, h2_ao_deriv, one_rdm, two_rdm, ao_mo_trafo=None, ao_mo_trafo_grad=None):
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

    #h2_ao = mol.intor("int2e")
    #h2_ao_deriv = mol.intor("int2e_ip1", comp=3)

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
        lib.einsum("ij,ijkl->kl", one_rdm, h1_jac, optimize="optimal")
        + 0.5 * two_el_gradient
    )

    return grad_elec
    #return 0.5 * two_el_gradient


def get_energy_with_grad(
    mol, one_RDM, two_RDM, S, hermitian=True, return_density_matrices=False
):
    """
    Calculates the potential energy and its gradient w.r.t. nuclear positions of a
    molecule from the eigenvector continuation.

    Args:
        mol : pyscf.gto.Mole
            The molecule object.
        one_RDM : numpy.ndarray
            The one-electron t-RDM.
        two_RDM (np.ndarray): Two-body t-RDM. Can have different shape depending on whether
            symmetry-compressed representations are used or not:
                No symmetries: shape(two_RDM) = (Ntrn, Ntrn, Norb, Norb, Norb, Norb)
                Data symmetry only: shape(two_RDM) = (Ntrn * (Ntrn + 1)/2, Norb, Norb, Norb, Norb)
                RDM electron exchange symmetry only: shape(two_RDM) = (Ntrn, Ntrn, (Norb**2 * (Norb**2 +1)/2)
                RDM electron exchange symmetry + data symmetry; shape(two_RDM) = (Ntrn * (Ntrn + 1)/2, (Norb**2 * (Norb**2 +1)/2))
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

    one_rdm_predicted = np.tensordot(np.outer(vec, vec), one_RDM, axes=2)

    if len(two_RDM.shape) == 2 or len(two_RDM.shape) == 5:
        # symmetry in data points
        eigenvec_mat = 2 * np.outer(vec, vec)

        np.fill_diagonal(eigenvec_mat, 0.5 * np.diag(eigenvec_mat))

        two_rdm_predicted = np.tensordot(
            eigenvec_mat[np.tril_indices(len(vec))], two_RDM, axes=1
        )

    else:
        two_rdm_predicted = np.tensordot(np.outer(vec, vec), two_RDM, axes=2)

    if len(two_rdm_predicted.shape) != 4:
        two_rdm_predicted = restore_electron_exchange_symmetry(
            two_rdm_predicted, mol.nao
        )

    grad_elec = get_grad_elec_OAO(
        mol, one_rdm_predicted, two_rdm_predicted, ao_mo_trafo=ao_mo_trafo
    )

    if return_density_matrices:
        return (
            en.real + mol.energy_nuc(),
            grad_elec + grad.RHF(scf.RHF(mol)).grad_nuc(),
            one_rdm_predicted,
            two_rdm_predicted,
        )

    else:
        return (
            en.real + mol.energy_nuc(),
            grad_elec + grad.RHF(scf.RHF(mol)).grad_nuc(),
        )
      
def get_energy_with_grad_cpuefficient(mol, one_RDM, two_RDM, S, hermitian=True, return_density_matrices=False):
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

    # Get the gradient of one and two-electron integrals before contracting onto
    # rdms of different states
    h1_jac, h2_jac = get_one_and_two_el_grad(mol,ao_mo_trafo=ao_mo_trafo)

    one_rdm_predicted = np.tensordot(np.outer(vec, vec), one_RDM, axes=2)

    if len(two_RDM.shape) == 2 or len(two_RDM.shape) == 5:
        # symmetry in data points
        eigenvec_mat = 2 * np.outer(vec, vec)

        np.fill_diagonal(eigenvec_mat, 0.5 * np.diag(eigenvec_mat))

        two_rdm_predicted = np.tensordot(
            eigenvec_mat[np.tril_indices(len(vec))], two_RDM, axes=1
        )

    else:
        two_rdm_predicted = np.tensordot(np.outer(vec, vec), two_RDM, axes=2)

    if len(two_rdm_predicted.shape) != 4:
        two_rdm_predicted = restore_electron_exchange_symmetry(
            two_rdm_predicted, mol.nao
        )

    grad_elec = get_grad_elec_from_gradH(
        one_rdm_predicted, two_rdm_predicted, h1_jac, h2_jac
    )

    return (
        en.real + mol.energy_nuc(),
        grad_elec + grad.RHF(scf.RHF(mol)).grad_nuc(),
    )

############################################################
# NEW MULTISTATE - SEPERATED FROM REST FOR TESTING
############################################################

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
  
    two_el_contraction_ao = lib.einsum(
        "abcd,aimn,bj,ck,dl->ijklmn",
        h2_ao + h2_ao.transpose(1,0,3,2) 
        + h2_ao.transpose(2,3,0,1) + h2_ao.transpose(3,2,0,1),
        ao_mo_trafo_grad,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo,
        optimize="optimal",
    )
    
    #    h2_ao + h2_ao.transpose(1,0,2,3) 
    #    + h2_ao.transpose(3,2,1,0) + h2_ao.transpose(2,3,0,1),

    h2_grad_ao_sum = np.zeros((h2_ao.shape[0], h2_ao.shape[1], h2_ao.shape[2], h2_ao.shape[3], len(atm_slices),3))
    for i, slice in enumerate(atm_slices):
        
        two_el_ao = lib.einsum(
            "nmbcd,mi,bj,ck,dl->ijkln",
            h2_ao_deriv[:,slice[0] : slice[1],:,:,:],
            ao_mo_trafo[slice[0] : slice[1],:],
            ao_mo_trafo,
            ao_mo_trafo,
            ao_mo_trafo,
            optimize="optimal",
        )
        
        # Subtract the gradient contribution from the contraction part
        h2_grad_ao_sum[:,:,:,:,i,:] -= two_el_ao + two_el_ao.transpose(1, 0, 2, 3, 4) \
        + two_el_ao.transpose(3, 2, 1, 0, 4) + two_el_ao.transpose(2, 3, 0, 1, 4)
    
    h2_grad = two_el_contraction_ao + h2_grad_ao_sum
    
    # Return the two-electron integral gradient
    return h2_grad

def get_two_el_grad_new(h2_ao, ao_mo_trafo, ao_mo_trafo_grad, h2_ao_deriv, atm_slices):
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
  
    two_el_contraction_ao = lib.einsum(
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
    
    h2_grad_ao_sum = np.zeros((h2_ao.shape[0], h2_ao.shape[1], h2_ao.shape[2], h2_ao.shape[3], len(atm_slices),3))
    for i, slice in enumerate(atm_slices):
        
        two_el_ao = lib.einsum(
            "nmbcd,mi,bj,ck,dl->ijkln",
            h2_ao_deriv[:,slice[0] : slice[1],:,:,:],
            ao_mo_trafo[slice[0] : slice[1],:],
            ao_mo_trafo,
            ao_mo_trafo,
            ao_mo_trafo,
            optimize="optimal",
        )
        
        # Subtract the gradient contribution from the contraction part
        h2_grad_ao_sum[:,:,:,:,i,:] -= two_el_ao + two_el_ao.transpose(1, 0, 2, 3, 4) \
        + two_el_ao.transpose(3, 2, 1, 0, 4) + two_el_ao.transpose(2, 3, 0, 1, 4)
    
    h2_grad = two_el_contraction_ao + h2_grad_ao_sum
    
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
    
    
    two_el_gradient = lib.einsum(
            "ijkl,ijklmn->mn",
            two_rdm,
            h2_jac,
            optimize="optimal",
            )
    
    grad_elec = (
        lib.einsum("ij,ijkl->kl", one_rdm, h1_jac, optimize="optimal")
        + 0.5 * two_el_gradient
    )
    
    return grad_elec


def get_orbital_derivative_coupling(mol,ao_mo_trafo=None, ao_mo_trafo_grad=None, ovlp=None):
    """ 
    For orbital contribution to nonadiabatic coupling vectors;
    < Phi_a | d/dR Phi_b> where Phi are MOs
    
    Args:
        mol (object): Molecule object.
        ao_mo_trafo (ndarray, optional):
            AO to MO transformation matrix. Is computed if not provided.
        ao_mo_trafo_grad (ndarray, optional):
            Gradient of AO to MO transformation matrix. Is computed if not provided.

    Returns:
        tuple of np.ndarray (nbasis, nbasis, nat,3):
            Orbital derivative coupling (to be contracted with 1-trdm).
    """
    if ao_mo_trafo is None:
        ao_mo_trafo = get_loewdin_trafo(mol.intor("int1e_ovlp"))

    if ao_mo_trafo_grad is None:
        ao_mo_trafo_grad = get_derivative_ao_mo_trafo(mol)
    
    if ovlp is None:
        ovlp = mol.intor("int1e_ovlp")

    # Contraction of the SAO transformation derivative
    # \sum_{ik} dC_{ij}/dR * C_{kl} * s_{ik}
    trafo_deriv_contraction = lib.einsum("ijAx,ik,kl->jlAx",ao_mo_trafo_grad, ovlp, ao_mo_trafo,optimize="optimal")

    # Orbital derivative contraction
    # \sum_{ik} C_{ij} * C_{kl} * < bas_i | d bas_j/dR >
    atm_slices = tuple(
                [
                    (mol.aoslice_by_atom()[i][2], mol.aoslice_by_atom()[i][3])
                    for i in range(mol.natm)
                ]
            )
    
    #"""
    orb_deriv_contraction = np.zeros((mol.nao, mol.nao, mol.natm, 3))
    ipovlp = mol.intor("int1e_ipovlp", comp=3)  # shape (3, nao, nao)
    
    for i, (start, stop) in enumerate(atm_slices):
        tmp = lib.einsum("ij,xik,kl->jlx", 
            ao_mo_trafo[start:stop], 
            ipovlp[:, start:stop, :], 
            ao_mo_trafo,
            optimize=True,
        )  # shape (nb, nb, 3)
        orb_deriv_contraction[:, :, i, :] -= tmp

    """
    deriv_ov = np.zeros((len(atm_slices),3,mol.nao,mol.nao))
    for i, slice in enumerate(atm_slices):
        deriv_ov[i,:, slice[0] : slice[1], :] -= mol.intor("int1e_ipovlp")[:, slice[0] : slice[1], :]
        
    orb_deriv_contraction = lib.einsum("ij,Axik,kl->jlAx",ao_mo_trafo, deriv_ov, ao_mo_trafo,optimize="optimal")
    """
    return trafo_deriv_contraction +  orb_deriv_contraction

def get_multistate_energy_with_grad(mol, one_RDM, two_RDM, S, nroots=1, hermitian=True, return_density_matrices=False):
    """
    Calculates the potential energy and its gradient w.r.t. nuclear positions of a
    molecule from the eigenvector continuation.

    Args:
        mol : pyscf.gto.Mole
            The molecule object.
        one_RDM : numpy.ndarray
            The one-electron t-RDM.
        two_RDM (np.ndarray): Two-body t-RDM. Can have different shape depending on whether
            symmetry-compressed representations are used or not:
                No symmetries: shape(two_RDM) = (Ntrn, Ntrn, Norb, Norb, Norb, Norb)
                Data symmetry only: shape(two_RDM) = (Ntrn * (Ntrn + 1)/2, Norb, Norb, Norb, Norb)
                RDM electron exchange symmetry only: shape(two_RDM) = (Ntrn, Ntrn, (Norb**2 * (Norb**2 +1)/2)
                RDM electron exchange symmetry + data symmetry; shape(two_RDM) = (Ntrn * (Ntrn + 1)/2, (Norb**2 * (Norb**2 +1)/2))
        S : numpy.ndarray
            The overlap matrix.
        nroots (optional): int
            Number of states in the solver.
        hermitian (bool, optional):
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        tuple
            A tuple containing the total potential energies and its gradients.
    """
    # Construct h1 and h2
    ao_mo_trafo = get_loewdin_trafo(mol.intor("int1e_ovlp"))

    h1 = np.linalg.multi_dot((ao_mo_trafo.T, scf.hf.get_hcore(mol), ao_mo_trafo))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, ao_mo_trafo), mol.nao)

    en, vec = approximate_multistate(h1, h2, one_RDM, two_RDM, S, nroots=nroots, hermitian=hermitian)
    
    # Get the gradient of one and two-electron integrals before contracting onto
    # rdms of different states
    h1_jac, h2_jac = get_one_and_two_el_grad(mol,ao_mo_trafo=ao_mo_trafo)
    
    grad_elec_all = []
    one_rdm_predicted_all = []
    two_rdm_predicted_all = []
    for i_state in range(nroots):
        vec_i = vec[i_state,:]

        #one_rdm_predicted = lib.einsum("i,ijkl,j->kl", vec_i, one_RDM, vec_i, optimize="optimal")
        #two_rdm_predicted = lib.einsum(
        #    "i,ijklmn,j->klmn", vec_i, two_RDM, vec_i, optimize="optimal"
        #)
        
        one_rdm_predicted = np.tensordot(np.outer(vec_i, vec_i), one_RDM, axes=2)
        if len(two_RDM.shape) == 2 or len(two_RDM.shape) == 5:
            # symmetry in data points
            eigenvec_mat = 2 * np.outer(vec_i, vec_i)

            np.fill_diagonal(eigenvec_mat, 0.5 * np.diag(eigenvec_mat))

            two_rdm_predicted = np.tensordot(
                eigenvec_mat[np.tril_indices(len(vec_i))], two_RDM, axes=1
            )

        else:
            two_rdm_predicted = np.tensordot(np.outer(vec_i, vec_i), two_RDM, axes=2)

        if len(two_rdm_predicted.shape) != 4:
            two_rdm_predicted = restore_electron_exchange_symmetry(
                two_rdm_predicted, mol.nao
            )
            

        grad_elec = get_grad_elec_from_gradH(
            one_rdm_predicted, two_rdm_predicted, h1_jac, h2_jac
        )
        
        one_rdm_predicted_all.append(one_rdm_predicted)
        two_rdm_predicted_all.append(two_rdm_predicted)
        grad_elec_all.append(grad_elec)
        
    grad_elec_all = np.array(grad_elec_all)

    if return_density_matrices:
        return (
            en.real + mol.energy_nuc(),
            grad_elec + grad.RHF(scf.RHF(mol)).grad_nuc(),
            one_rdm_predicted_all,
            two_rdm_predicted_all,
        )

    else:
        return (
            en.real + mol.energy_nuc(),
            grad_elec + grad.RHF(scf.RHF(mol)).grad_nuc(),
        )

@timeit
def get_multistate_energy_with_grad_and_NAC(mol, one_RDM, two_RDM, S, nroots=1, 
                                            savemem=True, hermitian=True):
    """
    Calculates the potential energiesm its gradient w.r.t. nuclear positions of a
    molecule and nonadiabatic couplings from eigenvector continuation for both
    ground and excited states.

    Args:
        mol : pyscf.gto.Mole
            The molecule object.
        one_RDM : numpy.ndarray
            The one-electron t-RDM.
        two_RDM : numpy.ndarray
            The two-electron t-RDM.
        S : numpy.ndarray
            The overlap matrix.
        nroots (optional): int
            Number of states in the solver.
        hermitian (optional): bool
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        tuple (vec, en, grad_all, nac_all, nac_all_hfonly)
            A tuple containing the continuation eigenvector, total potential energies, its gradients and NACs:
            
            vec: ndarray(ntrain,)
                Coefficients of the linear expansion
                
            en: ndarray(nroot,)
                Total potential energies for both ground and excited states
                
            grad_all: list of ndarray(nat,3)
                Gradients of multistate energies
                
            nac_all: dictionary of np.darray(nat,3)
                Nonadiabatic coupling vectors between all states,
                e.g. nac_all['02'] is NAC along ground state and 2nd excited state
                
            nac_all_hfonly: dictionary of ndarray(nat,3)
                Hellman-Feynmann contribution to NACs
    """
                
    # Construct h1 and h2
    ao_mo_trafo = get_loewdin_trafo(mol.intor("int1e_ovlp"))

    h1 = np.linalg.multi_dot((ao_mo_trafo.T, scf.hf.get_hcore(mol), ao_mo_trafo))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, ao_mo_trafo), mol.nao)

    # Diagonalization of the subspace Hamiltonian for the continuation of
    # energies and eigenstates
    en, vec = approximate_multistate(h1, h2, one_RDM, two_RDM, S, nroots=nroots, hermitian=hermitian)
    fix_gauge(vec)

    if not savemem:
        # Get the gradient of one and two-electron integrals before contracting onto
        # rdms and trmds of different states
        h1_jac, h2_jac = get_one_and_two_el_grad(mol,ao_mo_trafo=ao_mo_trafo)
        
    # Get the orbital derivative coupling for NACs
    orb_deriv = get_orbital_derivative_coupling(mol,ao_mo_trafo=ao_mo_trafo)
    
    # Nuclear part of the gradient
    grad_nuc = grad.RHF(scf.RHF(mol)).grad_nuc()
        
    grad_elec_all = []
    nac_all = {}
    nac_all_hfonly = {}
    # Iterate over pairs of eigenstates
    for i_state in range(nroots):
        vec_i = vec[i_state,:]

        for j_state in range(nroots):
            vec_j = vec[j_state,:]
            
            # Contracting to subspace eigenstate in hand
            #one_rdm_predicted = lib.einsum("i,ijkl,j->kl", vec_i, one_RDM, vec_j, optimize="optimal")
            #two_rdm_predicted = lib.einsum(
            #    "i,ijklmn,j->klmn", vec_i, two_RDM, vec_j, optimize="optimal"
            #)
            one_rdm_predicted = np.tensordot(np.outer(vec_i, vec_j), one_RDM, axes=2)
            if len(two_RDM.shape) == 2 or len(two_RDM.shape) == 5:
                # symmetry in data points
                eigenvec_mat = 2 * np.outer(vec_i, vec_j)

                np.fill_diagonal(eigenvec_mat, 0.5 * np.diag(eigenvec_mat))

                two_rdm_predicted = np.tensordot(
                    eigenvec_mat[np.tril_indices(len(vec_i))], two_RDM, axes=1
                )

            else:
                two_rdm_predicted = np.tensordot(np.outer(vec_i, vec_j), two_RDM, axes=2)

            if len(two_rdm_predicted.shape) != 4:
                two_rdm_predicted = restore_electron_exchange_symmetry(
                    two_rdm_predicted, mol.nao
                )
            
            # d\dR of subspace Hamiltonian
            if savemem:
                grad_elec = get_grad_elec_OAO(
                    mol, one_rdm_predicted, two_rdm_predicted
                )
            else:
                grad_elec = get_grad_elec_from_gradH(
                    one_rdm_predicted, two_rdm_predicted, h1_jac, h2_jac
                )
            
            
            # Energy gradients
            if i_state == j_state:
                grad_elec_all.append(grad_elec)
                
            # Nonadiabatic couplings
            else:
                # Hellman-Feynman contribution to NAC
                nac_hf = grad_elec/(en[j_state]-en[i_state])

                #if i_state == 0 and j_state == 1: 
                #    print(i_state, j_state, grad_elec)
                    
                # Orbital contribution to NAC
                nac_orb = lib.einsum("ij,ijkl->kl",one_rdm_predicted, orb_deriv, optimize="optimal")

                # Total NAC
                nac_ij = nac_hf + nac_orb
                
                # Save to dictionaries
                nac_all[str(i_state)+str(j_state)] = nac_ij
                nac_all_hfonly[str(i_state)+str(j_state)] = nac_hf

    # Add the nuclear contribution to gradient
    grad_all = np.array(grad_elec_all) + grad_nuc

    return (
        vec,
        en.real + mol.energy_nuc(),
        grad_all,
        nac_all,
        nac_all_hfonly
    )


##############################################################################
def two_el_grad_lowrank(mol, lowrank_vecs, ED_builds, SVD_builds, vec_i, vec_j,
                        ao_mo_trafo=None, ao_mo_trafo_grad=None):
    """
    Computing the gradient of the electronic Hamiltonian wrt atomic coordinates
    using the low-rank representation of the 2-tRDM
    
    ### OUTDATED; use state_resolved_version, or adapt it here
    
    """
    # AO to SAO basis transformation
    if ao_mo_trafo is None:
        ao_mo_trafo = get_loewdin_trafo(mol.intor("int1e_ovlp"))
    
    if ao_mo_trafo_grad is None:
        ao_mo_trafo_grad = get_derivative_ao_mo_trafo(mol)
        
    # Preliminaries
    ntrain = vec_i.shape[0]
    norb = ao_mo_trafo.shape[-1]
    
    # Unpack the preliminaries
    if lowrank_vecs['has_ed']:
        lr_vecs, lr_vecs_ao, vhf, vhf_grad, vhf_grad_t = ED_builds

    if lowrank_vecs['has_svd']:
        svd_lvecs, svd_rvecs, svd_lvecs_ao, svd_rvecs_ao, \
            vj_left, vj_right, \
            vj_l_grad, vj_r_grad = SVD_builds
            
        #print('Error in two_el_grad_lowrank: SVD not implemented yet')
        #sys.exit()
        
            
    # Basis functions indices for each atom
    atm_slices = tuple(
        [
            (mol.aoslice_by_atom()[i][2], mol.aoslice_by_atom()[i][3])
            for i in range(mol.natm)
        ]
    )
    
    # Pulay terms
    """
    pulay_term = 4*lib.einsum('xj,ABawx,ABaij,ABa,A,B->wi', ao_mo_trafo, 
                             vhf, lr_vecs, lowrank_vecs['vals'][:,:,:vhf.shape[2]],
                             vec_i, vec_j, optimize='optimal')
    
    grad_i = lib.einsum('wimn,wi->mn',ao_mo_trafo_grad,pulay_term,optimize='optimal')
    """
    pulay_term = np.zeros([ntrain,ntrain,norb,norb])
    if lowrank_vecs['has_ed']: 
        pulay_term += 2*lib.einsum('xj,ABawx,ABaij,ABa->ABwi', ao_mo_trafo, 
                                 vhf, lr_vecs, lowrank_vecs['vals'][:,:,:vhf.shape[2]],
                                 optimize='optimal')
        
        pulay_term += 2*lib.einsum('wi,ABawx,ABaij,ABa->ABxj', ao_mo_trafo, 
                                 vhf, lr_vecs, lowrank_vecs['vals'][:,:,:vhf.shape[2]],
                                 optimize='optimal')
        
    if lowrank_vecs['has_svd']: 

        pulay_term += lib.einsum('xj,ABawx,ABaij,ABa->ABwi', ao_mo_trafo, 
                                 vj_left, svd_rvecs, lowrank_vecs['vals'][:,:,:vj_left.shape[2]],
                                 optimize='optimal')
        
        pulay_term += lib.einsum('wi,ABawx,ABaij,ABa->ABxj', ao_mo_trafo, 
                                 vj_left, svd_rvecs, lowrank_vecs['vals'][:,:,:vj_left.shape[2]],
                                 optimize='optimal')
        
        pulay_term += lib.einsum('xj,ABawx,ABaij,ABa->ABwi', ao_mo_trafo, 
                                 vj_right, svd_lvecs, lowrank_vecs['vals'][:,:,:vj_left.shape[2]],
                                 optimize='optimal')
        
        pulay_term += lib.einsum('wi,ABawx,ABaij,ABa->ABxj', ao_mo_trafo, 
                                 vj_right, svd_lvecs, lowrank_vecs['vals'][:,:,:vj_left.shape[2]],
                                 optimize='optimal')
    
    if lowrank_vecs['hermitian']:
        # Set the upper triangle
        pulay_term[np.triu_indices(ntrain)] = pulay_term.transpose(1,0,2,3)[np.triu_indices(ntrain)].conj()
    
    grad_i = lib.einsum('wimn,ABwi,A,B->mn',ao_mo_trafo_grad,pulay_term,vec_i, vec_j, optimize='optimal')    

    #grad_i = np.zeros_like(grad_i) # Setting the previous part to zero for testing
    
    # JK Grad terms   
    grad_h2 = np.zeros([ntrain, ntrain, 3, norb])
    if lowrank_vecs['has_ed']: 

        grad_h2 += 2*lib.einsum('ABanij,ABaij,ABa->ABni',
                                vhf_grad_t, lr_vecs_ao,lowrank_vecs['vals'][:,:,:vhf.shape[2]],
                                optimize='optimal')
        
        grad_h2 += 2*lib.einsum('ABanij,ABaji,ABa->ABni',
                                vhf_grad, lr_vecs_ao,lowrank_vecs['vals'][:,:,:vhf.shape[2]],
                                optimize='optimal')
    
    if lowrank_vecs['has_svd']: 
                    
        #print(vj_l_grad.shape,svd_rvecs_ao.shape )
        grad_h2 += lib.einsum('ABanij,ABaij,ABa->ABni',
                                vj_l_grad, svd_rvecs_ao+svd_rvecs_ao.transpose(0,1,2,4,3),lowrank_vecs['vals'][:,:,:vj_left.shape[2]],
                                optimize='optimal')
        
        grad_h2 += lib.einsum('ABanij,ABaij,ABa->ABni',
                                vj_r_grad, svd_lvecs_ao+svd_lvecs_ao.transpose(0,1,2,4,3),lowrank_vecs['vals'][:,:,:vj_left.shape[2]],
                                optimize='optimal')
        
    if lowrank_vecs['hermitian']:
        # Set the upper triangle
        grad_h2[np.triu_indices(ntrain)] = grad_h2.transpose(1,0,2,3)[np.triu_indices(ntrain)].conj()
        
    grad_el_traced = lib.einsum('ABni,A,B->ni',grad_h2, vec_i, vec_j, optimize='optimal')
    
    # Sum contributions from each orbital on atom site, i
    for i, slice in enumerate(atm_slices):
        grad_i[i,:] += grad_el_traced[:,slice[0] : slice[1]].sum(axis=1) 

    return grad_i

@timeit
def state_resolved_two_el_grad_lowrank(mol, lowrank_vecs, ED_builds, SVD_builds,
                                       diag_builds=None, diagonals=None,
                                       ao_mo_trafo=None, ao_mo_trafo_grad=None,
                                       ints_SAO=None,
                                       df_response=False):
    """
    Computing the gradient of the electronic Hamiltonian wrt atomic coordinates
    using the low-rank representation of the 2-tRDM
    
    Note: This function does the same thing as two_el_grad_lowrank() but returns
    the 2-el gradient with the training state indices such that this can only be 
    computed once, and reused for the force and NAC calculation of each electronic
    excited state.
    
    """
    # AO to SAO basis transformation
    if ao_mo_trafo is None:
        ao_mo_trafo = get_loewdin_trafo(mol.intor("int1e_ovlp"))
    
    if ao_mo_trafo_grad is None:
        ao_mo_trafo_grad = get_derivative_ao_mo_trafo(mol)
        
    # Preliminaries
    # Get ntrain from pairloc (max index + 1) - might just make ntrain an input arg to avoid this
    if lowrank_vecs['has_ed']:
        ntrain_ed = max([i for i, j in lowrank_vecs['pairloc'].keys()]) + 1
    else:
        ntrain_ed = 0
    if lowrank_vecs['has_svd']:
        ntrain_svd = max([i for i, j in lowrank_vecs['pairloc_svd'].keys()]) + 1
    else:
        ntrain_svd = 0
    ntrain = max(ntrain_ed, ntrain_svd)
    
    norb = ao_mo_trafo.shape[-1]
    
    # Unpack the preliminaries
    if lowrank_vecs['has_ed']:
        lr_vecs, lr_vecs_ao, vhf, vhf_grad, vhf_grad_t = ED_builds

    if lowrank_vecs['has_svd']:
        svd_lvecs, svd_rvecs, svd_lvecs_ao, svd_rvecs_ao, \
            vj_left, vj_right, \
            vj_l_grad, vj_r_grad = SVD_builds

    # Unpack diagonal builds
    use_diag, Jdiag_only = False, True
    if diag_builds is not None:
        use_diag = True
        (vj, vj_grad, vk, vk_t, vk_grad) = diag_builds
        if vk is not None:
            Jdiag_only = False

    # If SAO implementation for diagonal corrections 
    if ints_SAO is not None:
        use_diag = True
        sao_diag = True
        Lpq, Lpq_sao, Lpq_grad_sao = ints_SAO
        
        if len(Lpq_grad_sao.shape) == 5:
            Jdiag_only = False
            Lpq_grad_sao_diag = np.einsum('nPiaa->nPia',Lpq_grad_sao)
            
        elif len(Lpq_grad_sao.shape) == 4:
            Jdiag_only = True
            Lpq_grad_sao_diag = Lpq_grad_sao

        else:
            print('Error in state_resolved_two_el_grad_lowrank: Lpq_grad_sao has incorrect number of indices.')
            sys.exit()
    else:
        sao_diag = False
 
    # Basis functions indices for each atom
    atm_slices = tuple(
        [
            (mol.aoslice_by_atom()[i][2], mol.aoslice_by_atom()[i][3])
            for i in range(mol.natm)
        ]
    )
    
    # Pulay terms
    pulay_term = np.zeros([ntrain,ntrain,norb,norb])
    if lowrank_vecs['has_ed']: 
        pulay_term += 2*lib.einsum('xj,ABawx,ABaij,ABa->ABwi', ao_mo_trafo, 
                                 vhf, lr_vecs, lowrank_vecs['vals'][:,:,:vhf.shape[2]],
                                 optimize='optimal')
        
        pulay_term += 2*lib.einsum('wi,ABawx,ABaij,ABa->ABxj', ao_mo_trafo, 
                                 vhf, lr_vecs, lowrank_vecs['vals'][:,:,:vhf.shape[2]],
                                 optimize='optimal')
        
    if lowrank_vecs['has_svd']: 

        pulay_term += lib.einsum('xj,ABawx,ABaij,ABa->ABwi', ao_mo_trafo, 
                                 vj_left, svd_rvecs, lowrank_vecs['vals'][:,:,:vj_left.shape[2]],
                                 optimize='optimal')
        
        pulay_term += lib.einsum('wi,ABawx,ABaij,ABa->ABxj', ao_mo_trafo, 
                                 vj_left, svd_rvecs, lowrank_vecs['vals'][:,:,:vj_left.shape[2]],
                                 optimize='optimal')
        
        pulay_term += lib.einsum('xj,ABawx,ABaij,ABa->ABwi', ao_mo_trafo, 
                                 vj_right, svd_lvecs, lowrank_vecs['vals'][:,:,:vj_left.shape[2]],
                                 optimize='optimal')
        
        pulay_term += lib.einsum('wi,ABawx,ABaij,ABa->ABxj', ao_mo_trafo, 
                                 vj_right, svd_lvecs, lowrank_vecs['vals'][:,:,:vj_left.shape[2]],
                                 optimize='optimal')
    
    if use_diag:

        if not sao_diag:
            pulay_term += 2*lib.einsum('xi,ABiwx->ABwi', 
                                       ao_mo_trafo, vj + vj.transpose(0,1,2,4,3),
                                       optimize='optimal')
            
            # Not needed! Same as above, so just times the above by 2
            #pulay_term += lib.einsum('zj,ABjyz->AByj', 
            #                           ao_mo_trafo, vj + vj.transpose(0,1,2,4,3),
            #                           optimize='optimal')
            if not Jdiag_only:

                pulay_term += 2*lib.einsum('xi,ABiwx->ABwi', 
                                           ao_mo_trafo, vk + vk_t,
                                           optimize='optimal')
                                
        else:
            diagJ_unpack = unstack_tril(diagonals[0],hermitian=False)

            pulay_term += 2*lib.einsum('xi,ABij,Pwx,Pjj->ABwi', 
                                       ao_mo_trafo, 
                                       diagJ_unpack + diagJ_unpack.transpose(0,1,3,2),
                                       Lpq,
                                       Lpq_sao,
                                       optimize='optimal')

            if not Jdiag_only:
                # TODO: diag_K contributions
                diagK_unpack = unstack_tril(diagonals[1],hermitian=False)

                pulay_term += 2*lib.einsum('xj,ABij,Pwx,Pij->ABwi', 
                                           ao_mo_trafo, 
                                           diagK_unpack + diagK_unpack.transpose(0,1,3,2),
                                           Lpq,
                                           Lpq_sao,
                                           optimize='optimal')
            
    if lowrank_vecs['hermitian']:
        # Set the upper triangle
        pulay_term[np.triu_indices(ntrain)] = pulay_term.transpose(1,0,2,3)[np.triu_indices(ntrain)].conj()
    
    intermediate_pulay = lib.einsum('wimn,ABwi->ABmn',ao_mo_trafo_grad,pulay_term, optimize='optimal')
        
    # JK Grad terms   
    grad_h2 = np.zeros([ntrain, ntrain, 3, norb])
    
    if lowrank_vecs['has_ed']: 

        grad_h2 += 2*lib.einsum('ABanij,ABaij,ABa->ABni',
                                vhf_grad_t, lr_vecs_ao,lowrank_vecs['vals'][:,:,:vhf.shape[2]],
                                optimize='optimal')
        
        grad_h2 += 2*lib.einsum('ABanij,ABaji,ABa->ABni',
                                vhf_grad, lr_vecs_ao,lowrank_vecs['vals'][:,:,:vhf.shape[2]],
                                optimize='optimal')
    
    if lowrank_vecs['has_svd']: 
                    
        #print(vj_l_grad.shape,svd_rvecs_ao.shape )
        grad_h2 += lib.einsum('ABanij,ABaij,ABa->ABni',
                                vj_l_grad, svd_rvecs_ao+svd_rvecs_ao.transpose(0,1,2,4,3),lowrank_vecs['vals'][:,:,:vj_left.shape[2]],
                                optimize='optimal')
        
        grad_h2 += lib.einsum('ABanij,ABaij,ABa->ABni',
                                vj_r_grad, svd_lvecs_ao+svd_lvecs_ao.transpose(0,1,2,4,3),lowrank_vecs['vals'][:,:,:vj_left.shape[2]],
                                optimize='optimal')
    
    if use_diag:

        ## AO Builds
        if not sao_diag:
            grad_h2 += 4*lib.einsum('ABinyz,yi,zi->ABny',
                                    vj_grad, ao_mo_trafo, ao_mo_trafo,
                                    optimize='optimal')
            
            # Same result as above
            #grad_h2 += 2*lib.einsum('ABinyz,yi,zi->ABny',
            #                        vj_grad_t, ao_mo_trafo, ao_mo_trafo,
            #                        optimize='optimal')
            
            if not Jdiag_only:

                grad_h2 += 2*lib.einsum('ABinwy,wi,yi->ABnw',
                                        vk_grad, ao_mo_trafo, ao_mo_trafo,
                                        optimize='optimal')
                
                # Included this term in the K build
                #grad_h2 += 2*lib.einsum('ABinwy,wi,yi->ABnw',
                #                        vk_grad_t, ao_mo_trafo, ao_mo_trafo,
                #                        optimize='optimal')
                
                
        else:
            # df_grad_4c_ints = np.einsum('xijP,Pkl->xijkl', deriv_cderi, cd_array)
            diagJ_unpack = unstack_tril(diagonals[0],hermitian=False)

            # NOTE: Unlike AO, the minus sign is not included in the integrals
            # so we need to subtract these contributions in SAO
            grad_h2 -= 2 * lib.einsum('ABab,nPia,Pbb->ABni',diagJ_unpack + diagJ_unpack.transpose(0,1,3,2), Lpq_grad_sao_diag, Lpq_sao)

            if not Jdiag_only:

                diagK_unpack = unstack_tril(diagonals[1],hermitian=False)
                
                grad_h2 -= 2 * lib.einsum('ABab,nPiab,Pab->ABni',diagK_unpack + diagK_unpack.transpose(0,1,3,2), Lpq_grad_sao, Lpq_sao)
                
    if lowrank_vecs['hermitian']:
        # Set the upper triangle
        grad_h2[np.triu_indices(ntrain)] = grad_h2.transpose(1,0,2,3)[np.triu_indices(ntrain)].conj()
        
    # Sum contributions from each orbital on atom site, i
    intermediate_h2 = np.zeros([ntrain, ntrain, mol.natm, 3])
    for i, slice in enumerate(atm_slices):
        intermediate_h2[:,:,i,:] += grad_h2[:,:,:,slice[0] : slice[1]].sum(axis=3) 
            
    # Response of the auxillary basis (Currently NOT WORKING)
    if df_response:
        intermediate_df = np.zeros([ntrain, ntrain, mol.natm, 3])
        if lowrank_vecs['has_ed']: 
            #vhf_aux = lib.einsum('aamn,a->mn',vj_list.aux - vk_list.aux*.5,lr_vals[ii],optimize='optimal')

            intermediate_df += vhf_grad_t.aux
            intermediate_df += vhf_grad.aux
            
            #intermediate_df += lib.einsum('ABABamn,ABa->ABmn',
            #                        vhf_grad.aux,lowrank_vecs['vals'][:,:,:vhf.shape[2]],
            #                        optimize='optimal')
            
        if lowrank_vecs['has_svd']: 
                        
            intermediate_df += vj_l_grad.aux
            intermediate_df += vj_r_grad.aux
            
        return intermediate_h2 + intermediate_pulay + intermediate_df
    else:
        return intermediate_h2 + intermediate_pulay

@timeit
def get_lowrank_en_with_grad_and_NAC(mol, one_RDM, S, lowrank_vecs, 
                                     diagonals=None, Jdiag_only=True, sao_diag=False,
                                     nroots=1, 
                                     density_fit=True, df_basis=None,
                                     ao_mo_trafo=None, ao_mo_trafo_grad=None,
                                     df_response=False,
                                     hermitian=True,):
    """
    Construct subspace Hamiltonian using the low-rank decomposition of 
    2-transition-cumulant
    
    Input:
        mol (Mole object): pySCF mole object at the test geometry
        
    """
    ### Preliminaries
    ntrain = S.shape[0]
    
    norb = one_RDM.shape[-1]
    norb_sq = norb * norb
    
    with log_time('Grad preliminaries'):
        # Initiate the mean field object to use DF integrals (no need to use kernel)
        #mol.symmetry = False
        mf = scf.RHF(mol).density_fit(auxbasis=df_basis)
        
        # AO to SAO basis transformation
        ovlp_ao = mol.intor("int1e_ovlp")
        if ao_mo_trafo is None:
            ao_mo_trafo = get_loewdin_trafo(ovlp_ao)
        
        if ao_mo_trafo_grad is None:
            ao_mo_trafo_grad = get_derivative_ao_mo_trafo(mol)
            
        # 1-electron integrals with DF
        h1_ao = mf.get_hcore()
        h1e_sao = lib.einsum('ai,ab,bj->ij', ao_mo_trafo, h1_ao, ao_mo_trafo)
        
    if df_response:
        print('Error in get_lowrank_en_with_grad_and_NAC(): \n \
        Response of the auxillary basis is not fully implemented. Set df_response=False.')
        sys.exit()
        
    ######################################################
    # Get preliminaries
    ED_builds, SVD_builds, diag_builds = get_jk_builds(
        mol, lowrank_vecs,
        diagonals=diagonals, 
        Jdiag_only=Jdiag_only, 
        sao_diag=sao_diag, 
        ao_mo_trafo=ao_mo_trafo,
        density_fit=density_fit, 
        df_basis=df_basis,
        df_response=df_response
        )

    if lowrank_vecs['has_ed']:
        lr_vecs, lr_vecs_ao, vhf, vhf_grad, vhf_grad_t = ED_builds
        
    if lowrank_vecs['has_svd']:
        svd_lvecs, svd_rvecs, svd_lvecs_ao, svd_rvecs_ao, \
            vj_left, vj_right, \
            vj_l_grad, vj_r_grad = SVD_builds
            
    use_diag = False
    if diag_builds is not None:
        use_diag = True
        (vj, vj_grad, vk, vk_t, vk_grad) = diag_builds
        
    # if sao_diag
    elif diagonals is not None:
        use_diag = True
        
    # SAO grads for gradients
    if sao_diag:
                
        with log_time('SAO transf. integrals'):
            
            # Get the integrals in DF
            Lpq, Lpq_grad = get_df_integrals(mol, auxbasis=df_basis, grad=True)
            norb = Lpq.shape[-1]
            
            # Transform to SAO basis
            Lpq = lib.pack_tril(Lpq)
            Lpq_sao = ao2mo._ao2mo.nr_e2(Lpq, ao_mo_trafo,
                (0, ao_mo_trafo.shape[1], 0, ao_mo_trafo.shape[1]),aosym="s2",mosym="s2")
            Lpq_sao = lib.unpack_tril(Lpq_sao)
            
            # Transform the derivatives to SAO basis
            """
            Lpq_grad = np.einsum('xijP->xPij',Lpq_grad)
            orig_shap = Lpq_grad.shape
            #Lpq_grad = Lpq_grad.reshape([-1, norb, norb])
            Lpq_grad = Lpq_grad.reshape([orig_shap[0]*orig_shap[1], norb, norb])
            
            Lpq_grad_sao = ao2mo._ao2mo.nr_e2(Lpq_grad, ao_mo_trafo,
            (0, norb, 0, norb), aosym='s1', mosym='s1')
            Lpq_grad_sao = Lpq_grad_sao.reshape(orig_shap)
            """
            # Explicit transformation (for testing, comment out later)
            #Lpq_grad_sao = np.einsum('xabP,ia,jb->xPij',Lpq_grad,ao_mo_trafo,ao_mo_trafo)
            
            if Jdiag_only:
                Lpq_grad_sao = np.einsum('ia,ja,nijP->nPia', ao_mo_trafo, ao_mo_trafo, Lpq_grad, optimize='optimal')
            else:
                Lpq_grad_sao = np.einsum('ia,jb,nijP->nPiab', ao_mo_trafo, ao_mo_trafo, Lpq_grad, optimize='optimal')

            ints_sao = (lib.unpack_tril(Lpq), Lpq_sao, Lpq_grad_sao)
    else:
        ints_sao = None
        
    ######################################################
    ######### CONSTRUCT SUBSPACE HAMILTONIAN AND GRADS
    ######################################################
    with log_time('Low-rank Hamiltonian'):
        ### 1-body contributions
        subspace_h = lib.einsum('...kl,kl->...', one_RDM, h1e_sao)
    
        # Contruction for subspace Hamiltonian
        if lowrank_vecs['has_ed']:
            subspace_h += 0.5*lib.einsum('xyaij,xyaij,xya->xy', vhf, lr_vecs_ao, lowrank_vecs['vals'][:,:,:vhf.shape[2]],optimize='optimal')
        
        if lowrank_vecs['has_svd']:
            subspace_h += 0.5*lib.einsum('xyaij,xyaij,xya->xy', vj_right, svd_lvecs_ao, lowrank_vecs['vals'][:,:,:vj_right.shape[2]],optimize='optimal')
        
        if use_diag:
            
            if not sao_diag:
                subspace_h += 0.5 * np.einsum('yj,zj,XYjyz->XY', ao_mo_trafo, ao_mo_trafo, vj)
                if not Jdiag_only:
                    subspace_h += 0.5 * np.einsum('xj,zj,XYjxz->XY', ao_mo_trafo, ao_mo_trafo, vk)

            else:
                diagJ_unpack = unstack_tril(diagonals[0],hermitian=False)
                subspace_h += 0.5 * np.einsum('XYij,Pii,Pjj->XY',diagJ_unpack, Lpq_sao, Lpq_sao)
                if not Jdiag_only:
                    diagK_unpack = unstack_tril(diagonals[1],hermitian=False)
                    subspace_h += 0.5 * np.einsum('XYij,Pij,Pij->XY',diagK_unpack, Lpq_sao, Lpq_sao)
                    
        if lowrank_vecs['hermitian']:
            # Set the upper triangle
            subspace_h[np.triu_indices(ntrain)] = subspace_h.T[np.triu_indices(ntrain)].conj()

        # Diagonalize
        en, vec = solve_subspace(subspace_h, S, hermitian=hermitian, nroots=nroots)
        fix_gauge(vec)

    ######################################################
    ######### GET GRADIENTS
    ######################################################
    #with log_time('Grad one-body'):
    # Get the orbital derivative coupling for NACs
    orb_deriv = get_orbital_derivative_coupling(mol,
                                                ao_mo_trafo=ao_mo_trafo,
                                                ao_mo_trafo_grad=ao_mo_trafo_grad,
                                                ovlp=ovlp_ao)

    # 1-el grad 
    h1_jac = get_one_el_grad(
        mol, 
        h1_ao=h1_ao, 
        ao_mo_trafo=ao_mo_trafo, 
        ao_mo_trafo_grad=ao_mo_trafo_grad
    )
        
    # Nuclear part of the gradient
    grad_nuc = grad.RHF(scf.RHF(mol)).grad_nuc()
        
    # 2-el grad
    grad_elec_h2_state = state_resolved_two_el_grad_lowrank(
        mol, lowrank_vecs, 
        ED_builds, SVD_builds, diag_builds,
        diagonals=diagonals,
        ao_mo_trafo=ao_mo_trafo, 
        ao_mo_trafo_grad=ao_mo_trafo_grad,
        #sao_diag=sao_diag, 
        ints_SAO=ints_sao,
        df_response=df_response
        )

    grad_elec_all = []
    nac_all = {}
    nac_all_hfonly = {}
    # Iterate over pairs of eigenstates
    for i_state in range(nroots):
        vec_i = vec[i_state,:]

        for j_state in range(nroots):
            vec_j = vec[j_state,:]
            
            # Contracting to subspace eigenstate in hand
            #one_rdm_predicted = lib.einsum("i,ijkl,j->kl", vec_i, one_RDM, vec_j, optimize="optimal")
            #two_rdm_predicted = lib.einsum(
            #    "i,ijklmn,j->klmn", vec_i, two_RDM, vec_j, optimize="optimal"
            #)
            one_rdm_predicted = np.tensordot(np.outer(vec_i, vec_j), one_RDM, axes=2)

            # 1-electron contributions to gradient
            grad_elec_h1 = lib.einsum("ij,ijkl->kl", one_rdm_predicted, h1_jac, optimize="optimal")
            
            #grad_elec_h2 = two_el_grad_lowrank(mol, lowrank_vecs, ED_builds, SVD_builds, vec_i, vec_j,
            #                        ao_mo_trafo=ao_mo_trafo, ao_mo_trafo_grad=ao_mo_trafo_grad)
            grad_elec_h2 = lib.einsum('ABmn,A,B->mn',grad_elec_h2_state,vec_i, vec_j,optimize="optimal")
            
            grad_elec = grad_elec_h1 + 0.5*grad_elec_h2
            #grad_elec = 0.5*grad_elec_h2 # For testing 2-el part only
            
            # Energy gradients
            if i_state == j_state:
                grad_elec_all.append(grad_elec)
                
            # Nonadiabatic couplings
            else:
                # Hellman-Feynman contribution to NAC
                nac_hf = grad_elec/(en[j_state]-en[i_state])

                #if i_state == 0 and j_state == 1: 
                #    print(i_state, j_state, grad_elec)
                
                # Orbital contribution to NAC
                nac_orb = lib.einsum("ij,ijkl->kl",one_rdm_predicted, orb_deriv, optimize="optimal")

                # Total NAC
                nac_ij = nac_hf + nac_orb
                
                # Save to dictionaries
                nac_all[str(i_state)+str(j_state)] = nac_ij
                nac_all_hfonly[str(i_state)+str(j_state)] = nac_hf

    # Add the nuclear contribution to gradient
    grad_all = np.array(grad_elec_all) + grad_nuc

    return (
        vec,
        en.real + mol.energy_nuc(),
        grad_all,
        nac_all,
        nac_all_hfonly
    )

if __name__ == '__main__':
    
    # Some initial checks for the code
    from pyscf import gto, fci

    from pyscf.fci.addons import fix_spin_
    from evcont.FCI_EVCont import FCI_EVCont_obj
    
    from pyscf.mcscf import CASCI
        
    from time import time
    
    CASE = 'NAC'
    #CASE = 'Exc-Grad'
    #CASE = 'Grad'
    
    nstate = 2 #1st excited state
    nroots_evcont = 3
    cibasis = 'OAO'
    
    natom = 8
    
    test_range = np.linspace(0.8, 3.0,20)

    def get_mol(positions):
        mol = gto.Mole()

        mol.build(
            atom=[("H", pos) for pos in positions],
            basis="sto-6g",
            #basis="6-31g",
            #basis='ccpvdz',
            symmetry=False,
            unit="Bohr",
            verbose=0
        )

        return mol

       
    # training geometries
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
    
    print('Finished training')
    
    if CASE == 'Grad':
        
        st = time()
        # Test the ground and excite state forces and gradients at training points
        for i, dist in enumerate(trainig_dists):
            positions = [(x, 0.0, 0.0) for x in dist * np.arange(natom)]
            mol = get_mol(positions)
            
            # Predictions from mutistate continuation
            en_continuation_ms, grad_continuation_ms = get_energy_with_grad(
                mol,
                continuation_object.one_rdm,
                continuation_object.two_rdm,
                continuation_object.overlap
            )
            
            # Predictions from Hartree-Fock
            hf_energy, hf_grad = mol.RHF().nuc_grad_method().as_scanner()(mol)
    
            # Fci reference values
            #en_exact, grad_exact = CASCI(mol.RHF(), natom, natom).nuc_grad_method().as_scanner()(mol)
            
            # Fci excited state reference values
            mc = CASCI(mol.RHF(), mol.nao, mol.nelectron)
            #mc = CASCI(mol.RHF(), 10,6) Li2
            mc.fcisolver = fci.direct_spin0.FCI()
            
            ci_scan_0 = mc.nuc_grad_method().as_scanner()
            
            en_exact, grad_exact = ci_scan_0(mol)
            
            # Checks
            print(i)
            
            assert np.allclose(en_exact,en_continuation_ms)
            assert np.allclose(grad_exact,grad_continuation_ms)
            
        print(f'Time taken: {time()-st:.1f} sec')
        

    if CASE == 'Exc-Grad':
        
        st = time()
        # Test the ground and excite state forces and gradients at training points
        for i, dist in enumerate(trainig_dists):
            positions = [(x, 0.0, 0.0) for x in dist * np.arange(natom)]
            mol = get_mol(positions)
            
            # Predictions from mutistate continuation
            _, en_continuation_ms, grad_continuation_ms, nac_continuation, _ = get_multistate_energy_with_grad_and_NAC(
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
            mc = CASCI(mol.RHF(), mol.nao, mol.nelectron)
            #mc = CASCI(mol.RHF(), 10,6) Li2
            mc.fcisolver = fci.direct_spin0.FCI()
            #mc.fcisolver = fci.direct_spin1.FCI()
            #fix_spin_(mc.fcisolver,shift=0.9,ss=0)
            mc.fcisolver.nroots = nstate+3
            #mc.fcisolver.nroots = 6
            #mc.fcisolver.conv_tol = 1.e-14
            #mc.fcisolver.max_space=30
            #mc.fcisolver.max_cycle=250
            
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
            assert np.allclose(grad_exc_exact,grad_continuation_ms[nstate],atol=1e-5)
                        
        print(f'Time taken: {time()-st:.1f} sec')
        
    # Test NACs
    if CASE == 'NAC':
        
        # TODO: Write new tests using FCI implementation
        pass
