#!/usr/bin/env python3
"""
Minimal working example: low-rank continuation inference for NACs 
with diagonal corrections.

This script demonstrates an end-to-end workflow:
1) Build CAS training data at a few H-chain geometries.
2) Build both full and low-rank continuation models.
3) Predict energies/gradients/NACs at a test geometry.
4) Compare low-rank vs full continuation results.

Author: Kemal Atalar
"""

import numpy as np
from pyscf import gto

from evcont.cas.CASCI_EVCont import CAS_EVCont_obj
from evcont.ab_initio_gradients_loewdin import (
    get_lowrank_en_with_grad_and_NAC,
    get_multistate_energy_with_grad_and_NAC,
)
# Keep example output focused on the comparison lines below.
from evcont.logging_utils import logger as evcont_logger
evcont_logger.disabled = True

def build_h_chain(natom, spacing_bohr, basis="sto-3g"):
    """Create a linear hydrogen chain with fixed spacing in Bohr."""
    atom = [("H", (i * spacing_bohr, 0.0, 0.0)) for i in range(natom)]
    return gto.M(atom=atom, basis=basis, unit="Bohr", symmetry=False, verbose=0)


def pair_labels(nroots):
    """All unique state-pair labels i<j as strings (e.g. '01')."""
    labels = []
    for i in range(nroots):
        for j in range(i + 1, nroots):
            labels.append(f"{i}{j}")
    return labels


# Problem setup kept intentionally small so this runs quickly.
natom = 8
nroots = 3
ncas = 4
neleca = 2
basis = "6-31g"
df_basis = "cc-pvdz-ri" # Default is None - which reverts back to pyscf default

train_spacings = [1.2, 1.8]
test_spacing = 1.5

# Simple low-rank setting for demonstration.
lowrank_kwargs = {
    "truncation_style": "eigval",
    "eval_thr": 1e-3,
    "save_diag": True,
    "Jdiag_only": True,
    "use_svd": False, # Allow selection of SVD if more compact
    "svd_weight": 2 # Weighting factor towards SVD vs joint decomposition
}

# Low-rank and full models for side-by-side comparison.
cont_lr = CAS_EVCont_obj(
    ncas,
    neleca,
    nroots=nroots,
    solver="CASCI",
    lowrank=True,
    **lowrank_kwargs,
)
cont_full = CAS_EVCont_obj(
    ncas,
    neleca,
    nroots=nroots,
    solver="CASCI",
    lowrank=False,
)

# Build training set.
for spacing in train_spacings:
    mol = build_h_chain(natom=natom, spacing_bohr=spacing, basis=basis)
    cont_lr.append_to_rdms(mol)
    cont_full.append_to_rdms(mol)

# Vectorize low-rank representation for fast inference.
cont_lr.vectorize_lowrank(hermitian=True)

# Low-rank representation details.
nvecs = cont_lr.lowrank_vectorized["nvecs"]
nvecs_used = nvecs[np.tril_indices_from(nvecs)]
mean_nvec = nvecs_used.mean() if nvecs_used.size else 0.0

# Predict at test geometry.
test_mol = build_h_chain(natom=natom, spacing_bohr=test_spacing, basis=basis)

_, e_full, g_full, nac_full, _ = get_multistate_energy_with_grad_and_NAC(
    test_mol,
    cont_full.one_rdm,
    cont_full.two_rdm,
    cont_full.overlap,
    nroots=nroots,
)


_, e_lr, g_lr, nac_lr, _ = get_lowrank_en_with_grad_and_NAC(
    test_mol,
    cont_lr.one_rdm,
    cont_lr.overlap,
    cont_lr.lowrank_vectorized,
    diagonals=cont_lr.diagonal_vectorized,
    Jdiag_only=lowrank_kwargs["Jdiag_only"],
    sao_diag=False,
    df_basis=df_basis,
    nroots=nroots,
)

labels = pair_labels(nroots)

print("=" * 80)
print("Minimal CAS Low-Rank NAC Example")
print("=" * 80)
print(f"System: H{natom}, basis={basis}, CAS({ncas}, {neleca}), nroots={nroots}")
print(f"Training spacings (Bohr): {train_spacings}")
print(f"Test spacing (Bohr):      {test_spacing}")
print("-" * 80)
print("Low-rank representation:")
print(f"   truncation style: {lowrank_kwargs['truncation_style']}")
print(f"   threshold:         {lowrank_kwargs['eval_thr']:.1e}")
print(f"   {mean_nvec:.1f} vectors per 2RDM pair (lower triangle)")
print("-" * 80)

print("State energies (Ha)")
print("state    Full EVCont        Low-rank EVCont        |Delta| (mHa)")
print("-" * 80)
for i in range(nroots):
    de_mha = 1000.0 * abs(e_lr[i] - e_full[i])
    print(f"{i:>3d}   {e_full[i]:>14.8f}      {e_lr[i]:>14.8f}      {de_mha:>11.4f}")
print("-" * 80)

print("Gradient comparison (Ha/a0)")
for i in range(nroots):
    dg = np.linalg.norm(g_lr[i] - g_full[i])
    print(f"state {i}: |Delta grad| = {dg:.6e}")
print("-" * 80)

print("NAC comparison (a0^-1)")
for label in labels:
    nac_full_norm = np.linalg.norm(nac_full[label])
    nac_lr_norm = np.linalg.norm(nac_lr[label])

    # NAC vectors can differ by a global sign due to phase/gauge choices.
    nac_err = min(
        np.linalg.norm(nac_lr[label] - nac_full[label]),
        np.linalg.norm(nac_lr[label] + nac_full[label]),
    )

    print(
        f"pair {label}: |Full|={nac_full_norm:.6e}, "
        f"|Low-rank|={nac_lr_norm:.6e}, "
        f"|Delta|={nac_err:.6e}"
    )

print("=" * 80)
