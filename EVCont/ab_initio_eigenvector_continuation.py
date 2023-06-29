import numpy as np

from scipy.linalg import eigh, eig


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
    gs_approx = vecs[:, valid_vals][:, argmin]
    return en_approx, gs_approx
