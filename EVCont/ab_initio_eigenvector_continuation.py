import numpy as np

from scipy.linalg import eigh


def approximate_ground_state(h1, h2, one_RDM, two_RDM, S):
    """Returns the electronic ground state approximation from solving the generalised eigenvalue problem
    defined via the one- and two-body transition RDMs.
    """
    H = np.sum(one_RDM * h1, axis=(-1,-2)) + 0.5 * np.sum(two_RDM * h2, axis=(-1,-2,-3,-4))
    vals, vecs = eigh(H, S)
    argmin = np.argmin(vals.real)
    en_approx = vals[argmin].real
    gs_approx = vecs[:, argmin]
    return en_approx, gs_approx