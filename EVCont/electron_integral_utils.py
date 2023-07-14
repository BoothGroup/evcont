from jax import config

config.update("jax_enable_x64", True)


import numpy as np

import jax.numpy as jnp

from pyscf import scf, lo, ao2mo


def transform_integrals(h1, h2, trafo):
    h1 = jnp.array(h1)
    h2 = jnp.array(h2)
    trafo = jnp.array(trafo)
    h1 = jnp.einsum("...ij,ai->...aj", h1, trafo)
    h1 = jnp.einsum("...aj,bj->...ab", h1, trafo)
    h2 = jnp.einsum("...ijkl,ai->...ajkl", h2, trafo)
    h2 = jnp.einsum("...ajkl,bj->...abkl", h2, trafo)
    h2 = jnp.einsum("...abkl,ck->...abcl", h2, trafo)
    h2 = jnp.einsum("...abcl,dl->...abcd", h2, trafo)
    return np.array(h1), np.array(h2)


def get_basis(mol, basis_type="OAO"):
    if basis_type == "OAO":
        basis = lo.orth_ao(mol, "lowdin", pre_orth_ao=None)
    else:
        myhf = scf.RHF(mol)
        ehf = myhf.scf()
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
