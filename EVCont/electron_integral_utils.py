import numpy as np

from pyscf import scf, lo, ao2mo


def get_loewdin_trafo(overlap_mat):
    vals, vecs = np.linalg.eigh(overlap_mat)
    inverse_sqrt_vals = np.where(vals > 1.0e-15, 1 / np.sqrt(vals), 0.0)
    return np.array(np.dot(vecs * inverse_sqrt_vals, vecs.conj().T))


def transform_integrals(h1, h2, trafo):
    h1 = np.einsum("...ij,ai,bj->...ab", h1, trafo, optimize="optimal")
    h2 = np.einsum("...ijkl,ai,bj,ck,dl->...abcd", h2, trafo, optimize="optimal")
    return h1, h2


def get_basis(mol, basis_type="OAO"):
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
    h1 = np.linalg.multi_dot((basis.T, scf.hf.get_hcore(mol), basis))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, basis), basis.shape[1])

    return h1, h2
