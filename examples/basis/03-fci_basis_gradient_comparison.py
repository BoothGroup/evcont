#!/usr/bin/env python3
"""Compare one-geometry FCI continuation gradients across abstract bases."""

import sys
from pathlib import Path

import numpy as np
from pyscf import gto

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evcont.ab_initio_gradients_loewdin import get_multistate_energy_with_grad
from evcont.fci.FCI_EVCont import FCI_EVCont_obj


def build_h_chain(coords, basis="sto-3g"):
    atoms = [("H", tuple(coord)) for coord in coords]
    return gto.M(atom=atoms, basis=basis, unit="Bohr", symmetry=False, verbose=0)


def chain_coords(natom, spacing):
    coords = np.zeros((natom, 3))
    coords[:, 0] = np.arange(natom) * spacing
    return coords


def finite_difference_gradient(cont, coords, basis, step):
    gradient = np.zeros_like(coords)
    for atom in range(coords.shape[0]):
        for axis in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            e_plus, _ = cont.approximate_multistate(build_h_chain(plus, basis))
            e_minus, _ = cont.approximate_multistate(build_h_chain(minus, basis))
            gradient[atom, axis] = (e_plus[0] - e_minus[0]) / (2.0 * step)
    return gradient


natom = 4
basis = "sto-3g"
train_spacings = [1.2, 1.8]
test_coords = chain_coords(natom, 1.5)
test_coords[1, 1] = 0.08
test_coords[2, 2] = -0.05
step = 1.0e-4

abstract_bases = [
    "SAO",
    "meta_lowdin",
    "split_procrustes",
    "least_change",
    "least_change_frozen_mo",
    "least_change_local",
]

print("basis                            energy (Ha)       ||gradient||       ||analytic-FD||")
for abstract_basis in abstract_bases:
    abstract_basis_kwargs = (
        {"least_change_emit_warnings": False}
        if abstract_basis.startswith("least_change")
        else None
    )
    cont = FCI_EVCont_obj(
        nroots=1,
        abstract_basis=abstract_basis,
        abstract_basis_kwargs=abstract_basis_kwargs,
    )
    for spacing in train_spacings:
        cont.append_to_rdms(build_h_chain(chain_coords(natom, spacing), basis))

    mol = build_h_chain(test_coords, basis)
    basis_kwargs = dict(cont.abstract_basis_kwargs)
    if cont.abstract_basis_ref is not None:
        basis_kwargs["basis_ref"] = cont.abstract_basis_ref
    if cont.abstract_basis_ref_mol is not None:
        basis_kwargs["basis_ref_mol"] = cont.abstract_basis_ref_mol

    energies, gradients = get_multistate_energy_with_grad(
        mol,
        cont.one_rdm,
        cont.two_rdm,
        cont.overlap,
        nroots=1,
        abstract_basis=abstract_basis,
        basis_kwargs=basis_kwargs,
    )
    finite_difference = finite_difference_gradient(cont, test_coords, basis, step)
    analytic = gradients[0]
    print(
        f"{abstract_basis:<32s} {energies[0]:>14.8f}   "
        f"{np.linalg.norm(analytic):>14.6e}   "
        f"{np.linalg.norm(analytic - finite_difference):>14.6e}"
    )
