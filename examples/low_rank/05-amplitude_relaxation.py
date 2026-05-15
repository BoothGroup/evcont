#!/usr/bin/env python3
"""
Minimal working example: 
comparing low-rank continuation with and without amplitude 
relaxation after eigenvalue truncation.

Workflow:
1) Build FCI training data at a few H-chain geometries.
2) Build both full, low-rank and low-rank with amplitude relaxation continuation models.
3) Predict state energies at a test geometry.
4) Compare against a direct FCI reference.

Author: Kemal Atalar
"""

import numpy as np
from pyscf import gto, fci

from evcont.electron_integral_utils import get_basis, get_integrals
from evcont.FCI_EVCont import FCI_EVCont_obj
from evcont.ab_initio_eigenvector_continuation import (
    approximate_multistate_OAO,
    approximate_multistate_lowrank_OAO,
)


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
natom = 8
nroots = 2
basis = "sto-3g"

train_spacings = [1.2, 1.8]
test_spacing = 1.5

# Hamiltonian-error threshold used during low-rank truncation.
lowrank_kwargs = {
    "truncation_style": "eigval",
    "eval_thr": 1e-1,
    "save_diag": True,
    "Jdiag_only": True,
    "relax_amp": False,
    "opt_no_diag": False,
}

# With amplitude relaxation after eigenvalue truncation.
lowrank_relax_kwargs = {
    "truncation_style": "eigval",
    "eval_thr": 1e-1,
    "save_diag": True,
    "Jdiag_only": True,
    "relax_amp": True,
    "opt_no_diag": True,
}

# Low-rank and full models for side-by-side comparison.
cont_lr = FCI_EVCont_obj(nroots=nroots, lowrank=True, **lowrank_kwargs)
cont_lr_relax = FCI_EVCont_obj(nroots=nroots, lowrank=True, **lowrank_relax_kwargs)
cont_full = FCI_EVCont_obj(nroots=nroots, lowrank=False)

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

e_ref = fci_reference_energies(test_mol, nroots=nroots)

print("=" * 72)
print("Minimal FCI Low-Rank Continuation Example (Eigenvalue Truncation)")
print("=" * 72)
print(f"System: H{natom}, basis={basis}, nroots={nroots}")
print(f"Training spacings (Bohr): {train_spacings}")
print(f"Test spacing (Bohr):      {test_spacing}")
print("-" * 72)
print("Low-rank representation:")
print(f"   truncation style: {lowrank_kwargs['truncation_style']}")
print(f"   Eval threshold: {lowrank_kwargs['eval_thr']:.1e}")
print(f"   {nvecs.mean():.1f} vectors per 2RDM (out of max {test_mol.nao**2})")
print("Low-rank representation with amplitude relaxation:")
print(f"   truncation style: {lowrank_relax_kwargs['truncation_style']}")
print(f"   Eval threshold: {lowrank_relax_kwargs['eval_thr']:.1e}")
print(f"   {nvecs_relax.mean():.1f} vectors per 2RDM (out of max {test_mol.nao**2})")
print("-" * 72)
print("state     FCI ref (Ha)     Full EVCont (Ha)    Low-rank EVCont (Ha)   Low-rank EVCont with Relax (Ha)")
print("-" * 72)

for i in range(nroots):
    print(f"{i:>3d}   {e_ref[i]:>16.8f}   {e_full[i]:>16.8f}   {e_lr[i]:>19.8f}   {e_lr_relax[i]:>19.8f}")

print("-" * 72)
print("Absolute errors vs FCI (mHa):")
for i in range(nroots):
    err_full_mha = 1000.0 * abs(e_full[i] - e_ref[i])
    err_lr_mha = 1000.0 * abs(e_lr[i] - e_ref[i])
    err_lr_relax_mha = 1000.0 * abs(e_lr_relax[i] - e_ref[i])
    print(f"state {i}: full={err_full_mha:8.3f} mHa, low-rank={err_lr_mha:8.3f} mHa, low-rank with relax={err_lr_relax_mha:8.3f} mHa")
print("=" * 72)
