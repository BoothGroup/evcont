#!/usr/bin/env python3
"""
Minimal working example: CAS eigenvector continuation.

Workflow:
1) Build CASCI training data at a few H-chain geometries.
2) Predict state energies at a test geometry via EVCont.
3) Compare against a direct CASCI reference.

Author: Kemal Atalar
"""

import numpy as np
from pyscf import gto, scf, mcscf

from evcont.CASCI_EVCont import CAS_EVCont_obj
from evcont.ab_initio_eigenvector_continuation import approximate_multistate_OAO


def build_h_chain(natom, spacing_bohr, basis="sto-3g"):
    """Create a linear hydrogen chain with fixed spacing in Bohr."""
    atom = [("H", (i * spacing_bohr, 0.0, 0.0)) for i in range(natom)]
    return gto.M(atom=atom, basis=basis, unit="Bohr", symmetry=False, verbose=0)


def casci_reference_energies(mol, ncas, neleca, nroots):
    """Direct CASCI reference energies (total energies, including E_nuc)."""
    mf = scf.RHF(mol)
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge for reference calculation.")

    mc = mcscf.CASCI(mf, ncas, neleca)
    mc.fcisolver.nroots = nroots
    mc.kernel()

    return np.array(mc.e_tot, dtype=float)


# Problem setup kept intentionally small so this runs quickly.
natom = 4
nroots = 2
ncas = 4
neleca = 2
basis = "6-31g"

train_spacings = [1.2, 1.8]
test_spacing = 1.5

cont = CAS_EVCont_obj(ncas, neleca, nroots=nroots, solver="SA-CASSCF")

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

e_ref = casci_reference_energies(test_mol, ncas=ncas, neleca=neleca, nroots=nroots)

print("=" * 60)
print("Minimal CASCI Eigenvector Continuation Example")
print("=" * 60)
print(f"System: H{natom}, basis={basis}, CAS({ncas}, {neleca}), nroots={nroots}")
print(f"Training spacings (Bohr): {train_spacings}")
print(f"Test spacing (Bohr):      {test_spacing}")
print("-" * 60)
print("state    CASCI ref (Ha)    EVCont (Ha)    Error (mHa)")
print("-" * 60)

for i in range(nroots):
    err_mha = 1000.0 * abs(e_cont[i] - e_ref[i])
    print(f"{i:>3d}   {e_ref[i]:>16.8f}   {e_cont[i]:>14.8f}   {err_mha:>10.3f}")

print("=" * 60)

