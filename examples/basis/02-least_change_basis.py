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

# Bare "least_change" intentionally exercises the default atom-coordinate gauge.
abstract_bases = [
    "least_change",
    "least_change_frozen_mo",
    "least_change_local",
]
test_mol = build_h_chain(natom, test_spacing, basis)
reference = fci_reference_energies(test_mol, nroots)

print("basis                         state     FCI ref (Ha)     EVCont (Ha)    error (mHa)")
for abstract_basis in abstract_bases:
    cont = FCI_EVCont_obj(
        nroots=nroots,
        abstract_basis=abstract_basis,
        abstract_basis_kwargs={"least_change_emit_warnings": False},
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
