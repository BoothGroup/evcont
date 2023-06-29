import numpy as np

from pyscf import scf, ao2mo, grad

import jax.numpy as jnp

import jax

from .ab_initio_eigenvector_continuation import approximate_ground_state


def get_overlap_grad(mol):
    inner_deriv = mol.intor("int1e_ipovlp", comp=3)

    deriv = np.zeros((3, mol.natm, mol.nao, mol.nao))
    for i in range(mol.natm):
        _, _, x, y = mol.aoslice_by_atom()[i]
        deriv[:, i, x:y, :] -= inner_deriv[:, x:y, :]

    deriv = deriv + deriv.transpose(0, 1, 3, 2)

    return np.transpose(deriv, (2, 3, 1, 0))


@jax.jit
def get_loewdin_trafo(overlap_mat):
    vals, vecs = jnp.linalg.eigh(overlap_mat)
    inverse_sqrt_vals = jnp.where(vals > 1.0e-15, 1 / jnp.sqrt(vals), 0.0)
    return jnp.dot(vecs * inverse_sqrt_vals, vecs.conj().T)


loewdin_trafo_grad = jax.jacobian(get_loewdin_trafo)


def get_derivative_ao_mo_trafo(mol):
    overlap_grad = get_overlap_grad(mol)
    trafo_grad = np.einsum(
        "ijkl, ijmn->klmn",
        np.array(loewdin_trafo_grad(mol.intor("int1e_ovlp")), dtype=float),
        overlap_grad,
    )

    return trafo_grad


def get_one_el_grad_ao(mol):
    hcore_gen = grad.RHF(scf.RHF(mol)).hcore_generator()
    return_val = np.array([hcore_gen(i) for i in range(mol.natm)])
    return np.transpose(return_val, (2, 3, 0, 1))


def get_one_el_grad(mol, ao_mo_trafo=None, ao_mo_trafo_grad=None):
    if ao_mo_trafo is None:
        ao_mo_trafo = np.array(
            get_loewdin_trafo(jnp.array(mol.intor("int1e_ovlp"))), dtype=float
        )

    h1_ao = scf.hf.get_hcore(mol)

    if ao_mo_trafo_grad is None:
        ao_mo_trafo_grad = get_derivative_ao_mo_trafo(mol)

    h1_grad_ao = get_one_el_grad_ao(mol)

    h1_grad = np.einsum("ijkl,im,mn->jnkl", ao_mo_trafo_grad, h1_ao, ao_mo_trafo)

    h1_grad += np.swapaxes(h1_grad, 0, 1)

    h1_grad += np.einsum("ij,iklm,kn->jnlm", ao_mo_trafo, h1_grad_ao, ao_mo_trafo)

    return h1_grad


def get_two_el_grad_ao(mol):
    inner_deriv = mol.intor("int2e_ip1", comp=3)

    deriv = np.zeros((3, mol.natm, mol.nao, mol.nao, mol.nao, mol.nao))
    for i in range(mol.natm):
        _, _, x, y = mol.aoslice_by_atom()[i]
        deriv[:, i, x:y, :, :, :] -= inner_deriv[:, x:y]

    deriv = (
        deriv
        + deriv.transpose(0, 1, 3, 2, 5, 4)
        + deriv.transpose(0, 1, 4, 5, 2, 3)
        + deriv.transpose(0, 1, 5, 4, 3, 2)
    )

    return np.transpose(deriv, (2, 3, 4, 5, 1, 0))


def get_two_el_grad(mol, ao_mo_trafo=None, ao_mo_trafo_grad=None):
    if ao_mo_trafo is None:
        ao_mo_trafo = np.array(
            get_loewdin_trafo(jnp.array(mol.intor("int1e_ovlp"))), dtype=float
        )

    h2_ao = mol.intor("int2e")

    if ao_mo_trafo_grad is None:
        ao_mo_trafo_grad = get_derivative_ao_mo_trafo(mol)

    h2_grad_ao = get_two_el_grad_ao(mol)

    h2_grad = np.einsum(
        "abcd,aimn,bj,ck,dl->ijklmn",
        h2_ao,
        ao_mo_trafo_grad,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo,
        optimize="optimal",
    )

    # This can certainly be done faster via appropriate transpose operations, but it shouldn't be the bottleneck atm
    h2_grad += np.einsum(
        "abcd,ai,bjmn,ck,dl->ijklmn",
        h2_ao,
        ao_mo_trafo,
        ao_mo_trafo_grad,
        ao_mo_trafo,
        ao_mo_trafo,
        optimize="optimal",
    )
    h2_grad += np.einsum(
        "abcd,ai,bj,ckmn,dl->ijklmn",
        h2_ao,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo_grad,
        ao_mo_trafo,
        optimize="optimal",
    )
    h2_grad += np.einsum(
        "abcd,ai,bj,ck,dlmn->ijklmn",
        h2_ao,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo_grad,
        optimize="optimal",
    )

    h2_grad += np.einsum(
        "abcdmn,ai,bj,ck,dl->ijklmn",
        h2_grad_ao,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo,
        ao_mo_trafo,
        optimize="optimal",
    )

    return h2_grad


def get_energy_with_grad(mol, one_RDM, two_RDM, S):
    # Construct h1 and h2
    ao_mo_trafo = np.array(
        get_loewdin_trafo(jnp.array(mol.intor("int1e_ovlp"))), dtype=float
    )
    ao_mo_trafo_grad = get_derivative_ao_mo_trafo(mol)

    h1 = np.linalg.multi_dot((ao_mo_trafo.T, scf.hf.get_hcore(mol), ao_mo_trafo))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, ao_mo_trafo), mol.nao)

    h1_jac = get_one_el_grad(
        mol, ao_mo_trafo=ao_mo_trafo, ao_mo_trafo_grad=ao_mo_trafo_grad
    )

    h2_jac = get_two_el_grad(
        mol, ao_mo_trafo=ao_mo_trafo, ao_mo_trafo_grad=ao_mo_trafo_grad
    )

    jac_H = np.sum(
        np.expand_dims(one_RDM, (-1, -2)) * h1_jac, axis=(-3, -4)
    ) + 0.5 * np.sum(np.expand_dims(two_RDM, (-1, -2)) * h2_jac, axis=(-3, -4, -5, -6))

    en, vec = approximate_ground_state(h1, h2, one_RDM, two_RDM, S)

    print("yes")

    return en + mol.energy_nuc(), np.einsum("i,ijkl,j->kl", vec, jac_H, vec)
    #  + grad.RHF(scf.RHF(mol)).grad_nuc()
