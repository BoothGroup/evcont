#!/usr/bin/env python3
"""Compare FCI continuation with the available least-change orbital gauges."""

import sys
from pathlib import Path

import numpy as np
from pyscf import fci, gto

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evcont.electron_integral_utils import get_basis, get_integrals
from evcont.fci.FCI_EVCont import FCI_EVCont_obj


def build_h_chain(natom, spacing_bohr, basis="sto-3g"):
    atoms = [("H", (i * spacing_bohr, 0.0, 0.0)) for i in range(natom)]
    return gto.M(atom=atoms, basis=basis, unit="Bohr", symmetry=False, verbose=0)


def fci_reference_energies(mol, nroots):
    h1, h2 = get_integrals(mol, get_basis(mol, basis_type="canonical"))
    energies, _ = fci.direct_spin0.FCI().kernel(
        h1, h2, mol.nao, mol.nelec, nroots=nroots
    )
    return np.atleast_1d(energies) + mol.energy_nuc()


natom = 4
nroots = 2
basis = "sto-3g"
train_spacings = [1.2, 1.8]
test_spacing = 1.5

# These options apply to every least-change gauge.  They are written out here
# so users can adjust the gauge-conditioning and verification thresholds.
common_least_change_kwargs = {
    "least_change_rank_tolerance": 1.0e-10,
    "least_change_verification_tolerance": 1.0e-8,
    "least_change_require_converged": True,
    "least_change_emit_warnings": False,
}

# Each entry exposes the keywords specific to that least-change definition.
# "least_change" remains an alias for "least_change_atom_coordinate".
least_change_bases = {
    "least_change_atom_coordinate": {
        **common_least_change_kwargs,
        # Moving atom-labelled frame used to freeze the reference transform.
        "least_change_anchor": "meta_lowdin",  # or "lowdin"
        "least_change_pre_orth_ao": "ANO",
    },
    "least_change_frozen_mo": {
        **common_least_change_kwargs,
        # No anchor keyword: the complete reference RHF MO frame is the anchor.
    },
    "least_change_local": {
        **common_least_change_kwargs,
        # Pointwise atom-labelled frame, rebuilt independently at each geometry.
        "least_change_anchor": "meta_lowdin",  # "lowdin" and "nao" also work
        "least_change_pre_orth_ao": "ANO",
        # True enables the two-stage atom -> SAP -> RHF projector alignment.
        "least_change_use_sap": False,
    },
}
test_mol = build_h_chain(natom, test_spacing, basis)
reference = fci_reference_energies(test_mol, nroots)

print("basis                         state     FCI ref (Ha)     EVCont (Ha)    error (mHa)")
for abstract_basis, abstract_basis_kwargs in least_change_bases.items():
    cont = FCI_EVCont_obj(
        nroots=nroots,
        abstract_basis=abstract_basis,
        abstract_basis_kwargs=abstract_basis_kwargs,
    )
    for spacing in train_spacings:
        cont.append_to_rdms(build_h_chain(natom, spacing, basis))

    energies, _ = cont.approximate_multistate(test_mol)
    for state in range(nroots):
        error = 1000.0 * abs(energies[state] - reference[state])
        print(
            f"{abstract_basis:<29s} {state:>3d}   "
            f"{reference[state]:>16.8f}   {energies[state]:>14.8f}   {error:>10.3f}"
        )
