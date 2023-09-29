import numpy as np

from pyscf import scf, ao2mo, grad

from evcont.ab_initio_eigenvector_continuation import approximate_ground_state

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
