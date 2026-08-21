#!/usr/bin/env python3
"""Compare conventional and density-fitted CCSD continuation.

The density-fitted run applies DF to both the current-geometry and fixed
reference RHF calculations used by a split-Procrustes abstract basis, as well
as to the RHF calculations underlying the CCSD solver.  The conventional run
uses the same geometries and settings with density fitting disabled.
"""

import sys
import time
from pathlib import Path

import numpy as np
from pyscf import gto

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evcont.ccsd.CCSD_EVCont import CCSD_EVCont_obj


def build_h_chain(natom, spacing_bohr, basis="cc-pvdz"):
    """Create an equally spaced linear hydrogen chain in Bohr."""

    atoms = [("H", (i * spacing_bohr, 0.0, 0.0)) for i in range(natom)]
    return gto.M(
        atom=atoms,
        basis=basis,
        unit="Bohr",
        symmetry=False,
        verbose=0,
    )


natom = 4
orbital_basis = "cc-pvdz"
auxiliary_basis = None  # Use PySCF's default auxiliary basis.
train_spacings = [1.2, 1.8]
test_spacing = 1.5


def run_continuation(density_fit):
    """Build and evaluate one conventional or density-fitted continuation."""

    # Generic DF keywords apply to the current and reference split-Procrustes
    # basis calculations and to every least-change basis.  The more specific
    # procrustes_ref_* options can override the reference calculation alone.
    abstract_basis_kwargs = {
        "density_fit": density_fit,
        "df_basis": auxiliary_basis,
    }

    start = time.perf_counter()
    reference_mol = build_h_chain(natom, test_spacing, basis=orbital_basis)
    continuation = CCSD_EVCont_obj(
        reference_mol,
        abstract_basis="split_procrustes",
        abstract_basis_kwargs=abstract_basis_kwargs,
        scf_density_fit=density_fit,
        scf_df_basis=auxiliary_basis,
    )
    for spacing in train_spacings:
        continuation.append_to_rdms(
            build_h_chain(natom, spacing, basis=orbital_basis)
        )

    test_mol = build_h_chain(natom, test_spacing, basis=orbital_basis)
    energy, _ = continuation.approximate(test_mol)
    test_basis = continuation.get_abstract_basis(test_mol)
    overlap = test_mol.intor_symmetric("int1e_ovlp")
    orthogonality_error = np.max(
        np.abs(test_basis.T @ overlap @ test_basis - np.eye(test_mol.nao))
    )
    return {
        "energy": float(np.ravel(energy)[0]),
        "orthogonality_error": float(orthogonality_error),
        "elapsed": time.perf_counter() - start,
    }


conventional = run_continuation(density_fit=False)
density_fitted = run_continuation(density_fit=True)
energy_difference_mha = 1000.0 * (
    density_fitted["energy"] - conventional["energy"]
)

print("Conventional vs Density-Fitted Split-Procrustes CCSD Continuation")
print(f"Orbital basis:                    {orbital_basis}")
print(f"DF auxiliary basis:               {auxiliary_basis}")
print(f"Training spacings (Bohr):         {train_spacings}")
print(f"Test spacing (Bohr):              {test_spacing}")
print()
print("method                  energy (Ha)       orthogonality max    time (s)")
for label, result in (
    ("conventional", conventional),
    ("density fitted", density_fitted),
):
    print(
        f"{label:<22s} {result['energy']:>14.10f}   "
        f"{result['orthogonality_error']:>17.3e}   {result['elapsed']:>8.3f}"
    )
print(f"DF - conventional energy (mHa): {energy_difference_mha:+.6f}")
