#!/usr/bin/env python3
"""
Minimal working example: CCSD eigenvector continuation.

Workflow:
1) Choose a reference water geometry for the computational abstract basis.
2) Build CCSD training data at a few stretched water geometries.
3) Predict the ground-state energy at a test geometry via EVCont.
4) Compare against a direct CCSD reference.

Author: Kemal Atalar
"""

import sys
from pathlib import Path

import numpy as np
from pyscf import cc, gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evcont.ccsd.CCSD_EVCont import CCSD_EVCont_obj


def build_water(oh_distance_angstrom, angle_degrees=104.5, basis="6-31g"):
    """Create H2O with a symmetric O-H stretch in Angstrom."""
    theta = np.deg2rad(angle_degrees)
    half_angle = 0.5 * theta
    x = oh_distance_angstrom * np.sin(half_angle)
    z = oh_distance_angstrom * np.cos(half_angle)
    atom = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (x, 0.0, z)),
        ("H", (-x, 0.0, z)),
    ]
    return gto.M(atom=atom, basis=basis, unit="Angstrom", symmetry=False, verbose=0)


def ccsd_reference_energy(mol):
    """Direct PySCF RCCSD total energy."""
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.conv_tol_grad = 1.0e-10
    mf.conv_tol_cpscf = 1.0e-10
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge for reference calculation.")

    mycc = cc.CCSD(mf)
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD did not converge for reference calculation.")

    return float(mycc.e_tot)


# Water/6-31g is deliberately larger than the H4/STO-3G FCI example, while
# still small enough for a quick all-electron CCSD demonstration.
nroots = 1
basis = "6-31g"
angle_degrees = 104.5
abstract_basis = "SAO"

reference_oh = 0.958
train_oh_distances = [0.90, 1.05, 1.20]
test_oh_distance = 1.00

comp_mol = build_water(
    reference_oh,
    angle_degrees=angle_degrees,
    basis=basis,
)
cont = CCSD_EVCont_obj(
    comp_mol,
    nroots=nroots,
    abstract_basis=abstract_basis,
)

# Build training set.
for oh_distance in train_oh_distances:
    mol = build_water(
        oh_distance,
        angle_degrees=angle_degrees,
        basis=basis,
    )
    cont.append_to_rdms(mol)

# Predict at test geometry.
test_mol = build_water(
    test_oh_distance,
    angle_degrees=angle_degrees,
    basis=basis,
)
e_cont, _ = cont.approximate(test_mol, nroots=nroots)
e_ref = ccsd_reference_energy(test_mol)

e_cont = float(np.ravel(e_cont)[0])
err_mha = 1000.0 * abs(e_cont - e_ref)

print("=" * 64)
print("Minimal CCSD Eigenvector Continuation Example")
print("=" * 64)
print(f"System: H2O, basis={basis}, abstract_basis={abstract_basis}, nroots={nroots}")
print(f"Reference O-H distance (Angstrom): {reference_oh}")
print(f"Training O-H distances (Angstrom): {train_oh_distances}")
print(f"Test O-H distance (Angstrom):      {test_oh_distance}")
print("-" * 64)
print("state      CCSD ref (Ha)      EVCont (Ha)      Error (mHa)")
print("-" * 64)
print(f"{0:>3d}   {e_ref:>16.8f}   {e_cont:>14.8f}   {err_mha:>12.3f}")
print("=" * 64)
