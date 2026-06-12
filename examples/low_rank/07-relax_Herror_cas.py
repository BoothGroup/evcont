#!/usr/bin/env python3
"""
Minimal working example: 
comparing low-rank continuation with and without amplitude 
relaxation after Hamiltonian error truncation for CAS states.

Truncation is based on relaxed amplitude Hamiltonian error (relax_after=False).

Workflow:
1) Build CASCI training data at a few H-chain geometries.
2) Build both full and low-rank continuation models.
3) Predict state energies at a test geometry.
4) Compare against a direct CASCI reference.

Author: Kemal Atalar
"""

import numpy as np
from pyscf import gto, scf, mcscf

from evcont.CASCI_EVCont import CAS_EVCont_obj
from evcont.ab_initio_eigenvector_continuation import (
    approximate_multistate_OAO,
    approximate_multistate_lowrank_OAO,
)


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
natom = 8
nroots = 2
basis = "6-31g"
ncas = 4
neleca = 2

train_spacings = [1.2, 1.8]
test_spacing = 1.5

# Hamiltonian-error threshold used during low-rank truncation.
lowrank_kwargs = {
    "truncation_style": "ham",
    "ham_thr": 1e-3,
    "save_diag": False,
    "relax_amp": False,
}

lowrank_relax_kwargs = {
    "truncation_style": "ham",
    "ham_thr": 1e-3,
    "save_diag": False,
    "relax_amp": True,
    "relax_after": False, # Relax after selecting low-rank vectors based on Hamiltonian error (vs relaxing during selection)
}

# Low-rank and full models for side-by-side comparison.
cont_lr = CAS_EVCont_obj(ncas, neleca, nroots=nroots, solver="CASCI", lowrank=True, **lowrank_kwargs)
cont_lr_relax = CAS_EVCont_obj(ncas, neleca, nroots=nroots, solver="CASCI", lowrank=True, **lowrank_relax_kwargs)
cont_full = CAS_EVCont_obj(ncas, neleca, nroots=nroots, solver="CASCI", lowrank=False)

# Build training set.
for spacing in train_spacings:
    mol = build_h_chain(natom=natom, spacing_bohr=spacing, basis=basis)
    cont_lr.append_to_rdms(mol)
    cont_lr_relax.append_to_rdms(mol)
    cont_full.append_to_rdms(mol)

# Vectorize low-rank representation for fast inference.
cont_lr.vectorize_lowrank(hermitian=True)
cont_lr_relax.vectorize_lowrank(hermitian=True)
nvecs = cont_lr.lowrank_vectorized["nvecs"]
nvecs_relax = cont_lr_relax.lowrank_vectorized["nvecs"]

# Predict at test geometry.
test_mol = build_h_chain(natom=natom, spacing_bohr=test_spacing, basis=basis)

e_lr, _ = approximate_multistate_lowrank_OAO(
    test_mol,
    cont_lr.one_rdm,
    cont_lr.lowrank_vectorized,
    cont_lr.diagonal_vectorized,
    cont_lr.overlap,
    nroots=nroots,
    density_fit=False,
    Jdiag_only=True,
    sao_diag=False,
)

e_lr_relax, _ = approximate_multistate_lowrank_OAO(
    test_mol,
    cont_lr_relax.one_rdm,
    cont_lr_relax.lowrank_vectorized,
    cont_lr_relax.diagonal_vectorized,
    cont_lr_relax.overlap,
    nroots=nroots,
    density_fit=False,
    Jdiag_only=True,
    sao_diag=False,
)

e_full, _ = approximate_multistate_OAO(
    test_mol,
    cont_full.one_rdm,
    cont_full.two_rdm,
    cont_full.overlap,
    nroots=nroots,
)

e_ref = casci_reference_energies(test_mol, ncas=ncas, neleca=neleca, nroots=nroots)

print("=" * 72)
print("Minimal CASCI Low-Rank Continuation Example (Hamiltonian Threshold)")
print("=" * 72)
print(f"System: H{natom}, basis={basis}, CAS({ncas}, {neleca}), nroots={nroots}")
print(f"Training spacings (Bohr): {train_spacings}")
print(f"Test spacing (Bohr):      {test_spacing}")
print("-" * 72)
print("Low-rank representation:")
print(f"   truncation style: {lowrank_kwargs['truncation_style']}")
print(f"   Hamiltonian threshold: {lowrank_kwargs['ham_thr']:.1e} Ha")
print(f"   {nvecs.mean():.1f} vectors per 2RDM (out of max {test_mol.nao**2})")
print("Low-rank representation with amplitude relaxation:")
print(f"   truncation style: {lowrank_relax_kwargs['truncation_style']}")
print(f"   Hamiltonian threshold: {lowrank_relax_kwargs['ham_thr']:.1e} Ha")
print(f"   {nvecs_relax.mean():.1f} vectors per 2RDM (out of max {test_mol.nao**2})")
print("-" * 72)
print("state    CASCI ref (Ha)    Full EVCont (Ha)    Low-rank EVCont (Ha)   Low-rank EVCont with Relax (Ha)")
print("-" * 72)

for i in range(nroots):
    print(f"{i:>3d}   {e_ref[i]:>16.8f}   {e_full[i]:>16.8f}   {e_lr[i]:>19.8f}   {e_lr_relax[i]:>19.8f}")

print("-" * 72)
print("Absolute errors vs CASCI (mHa):")
for i in range(nroots):
    err_full_mha = 1000.0 * abs(e_full[i] - e_ref[i])
    err_lr_mha = 1000.0 * abs(e_lr[i] - e_ref[i])
    err_lr_relax_mha = 1000.0 * abs(e_lr_relax[i] - e_ref[i])
    print(f"state {i}: full={err_full_mha:8.3f} mHa, low-rank={err_lr_mha:8.3f} mHa, low-rank with relax={err_lr_relax_mha:8.3f} mHa")
print("=" * 72)
