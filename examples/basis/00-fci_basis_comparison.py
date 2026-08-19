#!/usr/bin/env python3
"""
Compare FCI eigenvector continuation in different abstract bases.

This mirrors the minimal FCI continuation example, but repeats the workflow for
SAO, meta-Lowdin, and split-Procrustes abstract bases.
"""

import sys
from pathlib import Path

import numpy as np
from pyscf import fci, gto

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evcont.FCI_EVCont import FCI_EVCont_obj
from evcont.ab_initio_eigenvector_continuation import approximate_multistate_OAO
from evcont.electron_integral_utils import get_basis, get_integrals


def build_h_chain(natom, spacing_bohr, basis="sto-3g"):
    """Create a linear hydrogen chain with fixed spacing in Bohr."""
    atom = [("H", (i * spacing_bohr, 0.0, 0.0)) for i in range(natom)]
    return gto.M(atom=atom, basis=basis, unit="Bohr", symmetry=False, verbose=0)


def fci_reference_energies(mol, nroots, abstract_basis="SAO", basis_ref=None):
    """Direct FCI reference energies (total energies, including E_nuc)."""
    h1, h2 = get_integrals(
        mol,
        get_basis(mol, basis_type=abstract_basis, basis_ref=basis_ref),
    )
    cisolver = fci.direct_spin0.FCI()
    e_ref, _ = cisolver.kernel(h1, h2, mol.nao, mol.nelec, nroots=nroots)

    if nroots == 1:
        e_ref = np.array([e_ref], dtype=float)
    else:
        e_ref = np.array(e_ref, dtype=float)

    return e_ref + mol.energy_nuc()


natom = 4
nroots = 2
basis = "sto-3g"

train_spacings = [1.2, 1.8]
test_spacing = 1.5

test_mol = build_h_chain(natom=natom, spacing_bohr=test_spacing, basis=basis)
abstract_bases = ["SAO", "meta_lowdin", "split_procrustes"]
results = {}

for abstract_basis in abstract_bases:
    cont = FCI_EVCont_obj(nroots=nroots, abstract_basis=abstract_basis)

    for spacing in train_spacings:
        mol = build_h_chain(natom=natom, spacing_bohr=spacing, basis=basis)
        cont.append_to_rdms(mol)

    e_cont, _ = cont.approximate_multistate(test_mol)

    if abstract_basis == "SAO":
        e_legacy, _ = approximate_multistate_OAO(
            test_mol,
            cont.one_rdm,
            cont.two_rdm,
            cont.overlap,
            nroots=nroots,
        )
        assert np.allclose(e_cont, e_legacy)

    e_ref = fci_reference_energies(
        test_mol,
        nroots=nroots,
        abstract_basis=abstract_basis,
        basis_ref=cont.abstract_basis_ref,
    )
    results[abstract_basis] = (e_ref, e_cont)

print("=" * 72)
print("FCI Eigenvector Continuation Abstract-Basis Comparison")
print("=" * 72)
print(f"System: H{natom}, basis={basis}, nroots={nroots}")
print(f"Training spacings (Bohr): {train_spacings}")
print(f"Test spacing (Bohr):      {test_spacing}")
print("-" * 72)
print("basis                state     FCI ref (Ha)     EVCont (Ha)    Error (mHa)")
print("-" * 72)

for abstract_basis in abstract_bases:
    e_ref, e_cont = results[abstract_basis]
    for i in range(nroots):
        err_mha = 1000.0 * abs(e_cont[i] - e_ref[i])
        print(
            f"{abstract_basis:<18s} {i:>3d}   "
            f"{e_ref[i]:>16.8f}   {e_cont[i]:>14.8f}   {err_mha:>10.3f}"
        )

print("=" * 72)
