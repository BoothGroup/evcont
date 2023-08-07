import numpy as np

from scipy.linalg import eigh, eig

from EVCont.electron_integral_utils import get_basis, get_integrals


def approximate_ground_state(h1, h2, one_RDM, two_RDM, S, hermitian=True):
    """Returns the electronic ground state approximation from solving the generalised eigenvalue problem
    defined via the one- and two-body transition RDMs.
    """
    H = np.sum(one_RDM * h1, axis=(-1, -2)) + 0.5 * np.sum(
        two_RDM * h2, axis=(-1, -2, -3, -4)
    )
    if hermitian is True:
        vals, vecs = eigh(H, S)
    else:
        vals, vecs = eig(H, S)
    valid_vals = abs(vals.imag) < 1.0e-5
    argmin = np.argmin(vals[valid_vals].real)
    en_approx = vals[valid_vals][argmin].real
    gs_approx = vecs[:, valid_vals][:, argmin].real
    return en_approx, gs_approx


def approximate_ground_state_OAO(mol, one_RDM, two_RDM, S, hermitian=True):
    # Construct h1 and h2
    h1, h2 = get_integrals(mol, get_basis(mol))
    en, vec = approximate_ground_state(h1, h2, one_RDM, two_RDM, S, hermitian=hermitian)

    return en.real + mol.energy_nuc(), vec
