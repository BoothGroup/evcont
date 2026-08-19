import numpy as np

from pyscf import scf, ao2mo, df

from evcont.basis.basis_utils import (
    basis_requires_reference,
    get_basis,
    get_basis_reference,
    get_basis_with_derivative,
    get_loewdin_trafo,
    normalize_basis_type,
)


def transform_integrals(h1, h2, trafo):
    """
    Transforms one- and two-electron integrals using a given transformation matrix.

    Parameters:
        h1 (ndarray): One-electron integrals.
        h2 (ndarray): Two-electron integrals.
        trafo (ndarray): Transformation matrix.

    Returns:
        tuple: Transformed h1 and h2.
    """
    h1 = np.einsum("...ij,ai,bj->...ab", h1, trafo, optimize="optimal")
    h2 = np.einsum("...ijkl,ai,bj,ck,dl->...abcd", h2, trafo, optimize="optimal")
    return h1, h2


def compress_electron_exchange_symmetry(h2, diag_multiplier=1.0):
    """
    Transforms two-electron (four-index) quantities to a compressed representation exploiting
    electron exchange symmetries.

    Parameters:
        h2 (ndarray): Two-electron quantity with four indices.
        diag_multiplier (float): Multiplicative factor all elements of the diagonal
            are multiplied with (e.g. to take into account double counting in contraction).

    Returns:
        np.ndarray: Compressed representation.
    """
    assert np.all(np.array(h2.shape) == h2.shape[0])

    norb = h2.shape[0]

    h2 = h2.reshape(norb * norb, norb * norb)

    h2_diag = np.diag(h2).copy()

    np.fill_diagonal(h2, diag_multiplier * h2_diag)

    compressed_repr = h2[np.tril_indices(norb * norb)].copy()

    # Reverse modification of diagonal to avoid confusion
    np.fill_diagonal(h2, h2_diag)

    return compressed_repr


def restore_electron_exchange_symmetry(h2, norb):
    """
    Restores two-electron quantities from a compressed representation exploiting
    electron exchange symmetries.

    Parameters:
        h2 (ndarray): Two-electron quantity compressed into one index.
        norb (int): Number of orbitals.

    Returns:
        np.ndarray: 4-index representation.
    """
    h2_restored = np.zeros((norb * norb, norb * norb))
    h2_restored[np.tril_indices(norb * norb)] = h2

    h2_restored[np.triu_indices(norb * norb)] = (h2_restored.T)[
        np.triu_indices(norb * norb)
    ]

    return h2_restored.reshape((norb, norb, norb, norb))


def get_integrals(mol, basis):
    """
    Calculate the one-electron and two-electron integrals in a specified basis.

    Parameters:
        mol (pyscf.gto.Mole): The molecule object.
        basis (numpy.ndarray): The basis set (AO->MO transformation coefficients).

    Returns:
        h1 (numpy.ndarray): The one-electron integrals.
        h2 (numpy.ndarray): The two-electron integrals.
    """

    h1 = np.linalg.multi_dot((basis.T, scf.hf.get_hcore(mol), basis))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, basis), basis.shape[1])

    return h1, h2


def get_df_integrals(mol, basis=None, auxbasis=None, grad=False):
    """
    Compute the density-fitted ERIs and ERI gradients in a specified basis.
    
    Parameters:
        mol (pyscf.gto.Mole): The molecule object.
        basis (numpy.ndarray): The basis set (AO->MO transformation coefficients).
                             If None, uses AO basis.
        auxbasis (str): The auxiliary basis for density fitting.
        grad (bool): If True, also compute gradient integrals.
    
    Returns:
        If grad=False:
            cd_array (numpy.ndarray): Cholesky decomposed ERIs in the specified basis.
        If grad=True:
            cd_array, deriv_cderi (tuple): Cholesky decomposed ERIs and their gradients.
    """
    
    # Set auxillary basis
    auxmol = df.addons.make_auxmol(mol, auxbasis=auxbasis)
    naux = auxmol.nao

    # ints_3c is the 3-center integral tensor (ij|P), where i and j are the
    # indices of AO basis and P is the auxiliary basis
    ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
    # ints_2c2e is the (P|Q) integrals
    ints_2c2e = auxmol.intor('int2c2e')
    vals, vecs = np.linalg.eigh(ints_2c2e)
    assert(len(vals[vals < 1.e-15]) == 0) # PSD
    
    metric = np.array(np.dot(vecs * (1 / np.sqrt(vals)) , vecs.conj().T))
    cd_array = np.einsum('PQ,ijP->Qij', metric, ints_3c2e)
    
    # Transform to the specified basis if provided
    if basis is not None:
        cd_array = np.einsum('Pij,ai,bj->Pab', cd_array, basis, basis, optimize='optimal')
    
    # Full 4c integrals can be reconstructed as:
    #explicit_df_eri = np.einsum('Pij,Pkl->ijkl', cd_array, cd_array)

    if grad:
        # Now consider gradient integrals. The 4c integrals we want to approximate are: 
        # (d/dx i j | k l)
        # grad_4c_ints = mol.intor("int2e_ip1", comp=3)
        
        # We can get the integrals ( d/dx i, j | P)
        ints_3c2e_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1', comp=3)
        # Use the same metric as before
        deriv_cderi = np.einsum('PQ,xijP -> xijQ', metric, ints_3c2e_ip1)
        
        # Transform to the specified basis if provided
        if basis is not None:
            deriv_cderi = np.einsum('Pxij,ai,bj->Pxab', deriv_cderi, basis, basis, optimize='optimal')
        
        # Full 4c derivative integrals can be reconstructed as:
        # df_grad_4c_ints = np.einsum('xijP,Pkl->xijkl', deriv_cderi, cd_array)

        return cd_array, deriv_cderi
    
    else:
        return cd_array
