#!/usr/bin/env python3
"""Add the zero-amplitude RHF determinant to a CCSD continuation space.

The synthetic state is useful with an RHF-based abstract basis because it is
the target-geometry Hartree--Fock determinant at every geometry.  It therefore
provides an HF variational candidate even when evaluating outside the range of
the CCSD training geometries.
"""

import sys
from pathlib import Path

import numpy as np
from pyscf import cc, gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evcont.ccsd.CCSD_EVCont import CCSD_EVCont_obj


def build_water(oh_distance_angstrom, angle_degrees=104.5):
    """Return water with a symmetric O--H stretch in cc-pVDZ."""

    half_angle = 0.5 * np.deg2rad(angle_degrees)
    x = oh_distance_angstrom * np.sin(half_angle)
    z = oh_distance_angstrom * np.cos(half_angle)
    return gto.M(
        atom=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (x, 0.0, z)),
            ("H", (-x, 0.0, z)),
        ],
        basis="cc-pvdz",
        unit="Angstrom",
        symmetry=False,
        verbose=0,
    )


def reference_energies(mol):
    """Return direct RHF and RCCSD total energies."""

    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-10
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge")

    mycc = cc.CCSD(mf)
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD did not converge")
    return float(mf.e_tot), float(mycc.e_tot)


reference_oh_distance = 0.958
train_oh_distances = [0.40, 0.85]
# This stretched geometry lies beyond both CCSD training points.
extrapolation_oh_distance = 1.80


def build_continuation(include_zero_amplitude):
    """Build a continuation space with the requested RHF-state setting."""

    continuation = CCSD_EVCont_obj(
        build_water(reference_oh_distance),
        abstract_basis="split_procrustes",
        include_zero_amplitude=include_zero_amplitude,
    )
    for oh_distance in train_oh_distances:
        continuation.append_to_rdms(build_water(oh_distance))
    return continuation


continuation_without_zero = build_continuation(include_zero_amplitude=False)
continuation_with_zero = build_continuation(include_zero_amplitude=True)

test_mol = build_water(extrapolation_oh_distance)
e_cont_without_zero = float(
    np.ravel(continuation_without_zero.approximate(test_mol)[0])[0]
)
e_cont_with_zero = float(
    np.ravel(continuation_with_zero.approximate(test_mol)[0])[0]
)
e_hf, e_ccsd = reference_energies(test_mol)

print("=" * 80)
print("CCSD Continuation With and Without a Zero-Amplitude RHF State")
print("=" * 80)
print("System:                                    H2O/cc-pVDZ")
print(f"Training O-H distances (Angstrom):         {train_oh_distances}")
print(f"Extrapolation O-H distance (Angstrom):     {extrapolation_oh_distance}")
print(
    "Continuation states without zero:       "
    f"{len(continuation_without_zero.states)}"
)
print(
    "Continuation states with zero:          "
    f"{len(continuation_with_zero.states)}"
)
print("-" * 80)
print(f"RHF energy (Ha):                          {e_hf: .10f}")
print(f"CCSD energy (Ha):                         {e_ccsd: .10f}")
print(f"Continuation without zero (Ha):           {e_cont_without_zero: .10f}")
print(f"Continuation with zero (Ha):              {e_cont_with_zero: .10f}")
print("=" * 80)

# The zero-amplitude vector is exactly the RHF determinant in an RHF-based
# abstract basis, so the lowest continuation root cannot lie above RHF.
assert e_cont_with_zero <= e_hf + 1.0e-9
