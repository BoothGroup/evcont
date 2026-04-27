#!/usr/bin/env python3
"""
Minimal example: compute FCI nonadiabatic couplings (NACs) with PySCF + evcont.

This script:
1) builds a small H-chain geometry,
2) computes direct FCI energies, gradients, and NACs,
3) computes continuation energies, gradients, and NACs,
4) compares continuation results against direct FCI.
"""

import numpy as np
from pyscf import fci, gto

from evcont.FCI_EVCont import FCI_EVCont_obj
from evcont.FCI_NAC import get_FCI_energy_with_grad_and_NAC
from evcont.ab_initio_gradients_loewdin import get_multistate_energy_with_grad_and_NAC


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


# Setup molecule and FCI solver
natom = 4
basis = "sto-3g"
nroots = 3
test_distance = 1.5
train_distances = [1.2, 1.8]

# Spin-restricted singlet FCI solver.
fcisolver = fci.direct_spin0.FCI()
fci.addons.fix_spin_(fcisolver, ss=0)
fcisolver.nroots = nroots

# Reuse the same spin-adapted solver for continuation training.
continuation_object = FCI_EVCont_obj(
    nroots=nroots,
    cibasis="OAO",
    cisolver=fcisolver,
)

for dist in train_distances:
    train_mol = build_h_chain(natom=natom, spacing_bohr=float(dist), basis=basis)
    continuation_object.append_to_rdms(train_mol)

labels = pair_labels(nroots)

mol = build_h_chain(natom=natom, spacing_bohr=float(test_distance), basis=basis)

en_fci, grad_fci, nac_fci, _ = get_FCI_energy_with_grad_and_NAC(
    mol,
    fcisolver,
    cibasis="OAO",
    nroots=nroots,
)

_, en_cont, grad_cont, nac_cont, _ = get_multistate_energy_with_grad_and_NAC(
    mol,
    continuation_object.one_rdm,
    continuation_object.two_rdm,
    continuation_object.overlap,
    nroots=nroots,
)

print("=" * 72)
print("FCI NAC Example: Direct FCI vs Continuation")
print("=" * 72)
print(f"System: H{natom}, basis={basis}, nroots={nroots}")
print(f"Training distances (Bohr): {train_distances}")
print(f"Test distance (Bohr): {test_distance}")
print(f"State pairs: {labels}")
print("-" * 72)

print(f"distance = {test_distance:.3f} Bohr")
print("  FCI energies (Ha): ", " ".join(f"{e:.8f}" for e in en_fci))
print("  Cont energies (Ha):", " ".join(f"{e:.8f}" for e in en_cont))
print("  |Delta E| (mHa):   ", " ".join(f"{1000.0 * abs(de):.3f}" for de in (en_cont - en_fci)))

for label in labels:
    nac_fci_norm = np.linalg.norm(nac_fci[label])
    nac_cont_norm = np.linalg.norm(nac_cont[label])
    # Account for possible sign ambiguity in NACs by taking the minimum of ||Cont - FCI|| and ||Cont + FCI||.
    nac_err_norm = min(np.linalg.norm(nac_cont[label] - nac_fci[label]),
                       np.linalg.norm(nac_cont[label] + nac_fci[label]))
    print(
        f"  NAC {label}: |FCI|={nac_fci_norm:.6e} a0^-1, "
        f"|Cont|={nac_cont_norm:.6e} a0^-1, "
        f"|Delta norm|={nac_err_norm:.6e} a0^-1"
    )

grad_err_norm = np.linalg.norm(grad_cont - grad_fci,axis=(1,2))
for i in range(nroots):
    print(f"  Gradient {i}: |Delta grad|={grad_err_norm[i]:.6e} Ha/a0")
print("-" * 72)

