#!/usr/bin/env python3
"""Evaluate FCI continuation energies and gradients from the solver object."""

import numpy as np
from pyscf import gto

from evcont.fci.FCI_EVCont import FCI_EVCont_obj


def build_h_chain(natom, spacing_bohr, basis="sto-3g"):
    atom = [("H", (i * spacing_bohr, 0.0, 0.0)) for i in range(natom)]
    return gto.M(atom=atom, basis=basis, unit="Bohr", symmetry=False, verbose=0)


def finite_difference_gradients(cont, mol, step=1.0e-4):
    coords = mol.atom_coords(unit="Bohr")
    gradients = np.zeros((cont.nroots, mol.natm, 3))

    for atom in range(mol.natm):
        for axis in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step

            mol_plus = mol.set_geom_(plus, unit="Bohr", inplace=False)
            mol_minus = mol.set_geom_(minus, unit="Bohr", inplace=False)
            energies_plus, _ = cont.get_en(mol_plus)
            energies_minus, _ = cont.get_en(mol_minus)
            gradients[:, atom, axis] = (energies_plus - energies_minus) / (2.0 * step)

    return gradients


natom = 4
nroots = 2
basis = "sto-3g"
train_spacings = [1.2, 1.8]
test_spacing = 1.5

cont = FCI_EVCont_obj(nroots=nroots)
for spacing in train_spacings:
    cont.append_to_rdms(build_h_chain(natom, spacing, basis))

test_mol = build_h_chain(natom, test_spacing, basis)
energies, gradients = cont.get_en_with_grad(test_mol)
finite_difference = finite_difference_gradients(cont, test_mol)

print("state       energy (Ha)       ||gradient||       ||analytic-FD||")
for state in range(nroots):
    print(
        f"{state:>3d}   {energies[state]:>16.8f}   "
        f"{np.linalg.norm(gradients[state]):>14.6e}   "
        f"{np.linalg.norm(gradients[state] - finite_difference[state]):>14.6e}"
    )
