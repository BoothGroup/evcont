#!/usr/bin/env python3
"""
Minimal working example: FCI eigenvector continuation.

Workflow:
1) Build FCI training data at a few H-chain geometries.
2) Predict state energies at a test geometry via EVCont.
3) Compare against a direct FCI reference.

Author: Kemal Atalar
"""

import numpy as np
from pyscf import gto, fci

from evcont.electron_integral_utils import get_basis, get_integrals
from evcont.fci.FCI_EVCont import FCI_EVCont_obj
from evcont.ab_initio_eigenvector_continuation import approximate_multistate_OAO


def build_h_chain(natom, spacing_bohr, basis="sto-3g"):
    """Create a linear hydrogen chain with fixed spacing in Bohr."""
    atom = [("H", (i * spacing_bohr, 0.0, 0.0)) for i in range(natom)]
    return gto.M(atom=atom, basis=basis, unit="Bohr", symmetry=False, verbose=0)


def fci_reference_energies(mol, nroots):
    """Direct FCI reference energies (total energies, including E_nuc)."""
    h1, h2 = get_integrals(mol, get_basis(mol))
    cisolver = fci.direct_spin0.FCI()
    e_ref, _ = cisolver.kernel(h1, h2, mol.nao, mol.nelec, nroots=nroots)

    if nroots == 1:
        e_ref = np.array([e_ref], dtype=float)
    else:
        e_ref = np.array(e_ref, dtype=float)

    return e_ref + mol.energy_nuc()


# Problem setup kept intentionally small so this runs quickly.
natom = 4
nroots = 2
basis = "sto-3g"

train_spacings = [1.2, 1.8]
test_spacing = 1.5

cont = FCI_EVCont_obj(nroots=nroots)

# Build training set.
for spacing in train_spacings:
    mol = build_h_chain(natom=natom, spacing_bohr=spacing, basis=basis)
    cont.append_to_rdms(mol)

# Predict at test geometry.
test_mol = build_h_chain(natom=natom, spacing_bohr=test_spacing, basis=basis)

e_cont, _ = approximate_multistate_OAO(
    test_mol,
    cont.one_rdm,
    cont.two_rdm,
    cont.overlap,
    nroots=nroots,
)

e_ref = fci_reference_energies(test_mol, nroots=nroots)

print("=" * 60)
print("Minimal FCI Eigenvector Continuation Example")
print("=" * 60)
print(f"System: H{natom}, basis={basis}, nroots={nroots}")
print(f"Training spacings (Bohr): {train_spacings}")
print(f"Test spacing (Bohr):      {test_spacing}")
print("-" * 60)
print("state     FCI ref (Ha)     EVCont (Ha)    Error (mHa)")
print("-" * 60)

for i in range(nroots):
    err_mha = 1000.0 * abs(e_cont[i] - e_ref[i])
    print(f"{i:>3d}   {e_ref[i]:>16.8f}   {e_cont[i]:>14.8f}   {err_mha:>10.3f}")

print("=" * 60)
