import numpy as np

from pyscf import scf, lo, ao2mo


def get_loewdin_trafo(overlap_mat):
    """
    Computes the Loewdin transformation based on the given overlap matrix between AOs.

    Parameters:
        overlap_mat (ndarray): The overlap matrix.

    Returns:
        ndarray: The transformed matrix.
    """
    vals, vecs = np.linalg.eigh(overlap_mat)
    inverse_sqrt_vals = np.where(vals > 1.0e-15, 1 / np.sqrt(vals), 0.0)
    return np.array(np.dot(vecs * inverse_sqrt_vals, vecs.conj().T))


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


def get_basis(mol, basis_type="OAO"):
    """
    Construct a basis of orthogonal MOs for the given molecule.

    Args:
        mol: The molecule object.
        basis_type: The type of basis. Default is "OAO".

    Returns:
        basis: The basis for the molecule (as transformation coefficients from the AO
        basis).
    """
    if basis_type == "OAO":
        basis = get_loewdin_trafo(mol.intor("int1e_ovlp"))
    else:
        myhf = scf.RHF(mol)
        _ = myhf.scf()
        basis = myhf.mo_coeff
        if basis_type == "split":
            localizer = lo.Boys(mol, basis[:, : mol.nelec[0]])
            localizer.init_guess = None
            basis_occ = localizer.kernel()
            localizer = lo.Boys(mol, basis[:, mol.nelec[0] :])
            localizer.init_guess = None
            basis_vrt = localizer.kernel()
            basis = np.concatenate((basis_occ, basis_vrt), axis=1)
        else:
            assert basis_type == "canonical"
    return basis


def get_integrals(mol, basis):
    """
    Calculate the one-electron and two-electron integrals in a specified basis.

    Parameters:
        mol (pyscf.gto.Mole): The molecule object.
        basis (numpy.ndarray): The basis set.

    Returns:
        h1 (numpy.ndarray): The one-electron integrals.
        h2 (numpy.ndarray): The two-electron integrals.
    """

    h1 = np.linalg.multi_dot((basis.T, scf.hf.get_hcore(mol), basis))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, basis), basis.shape[1])

    return h1, h2
