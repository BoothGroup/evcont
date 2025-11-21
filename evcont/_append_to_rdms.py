#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Helper script to append new geometries to an existing continuation object
- Can be submitted as an HPC job within an active learning workflow (as used in NAMD_utils.py)

@author: Kemal Atalar
"""

import numpy as np

from pyscf import gto

import sys

CONT_OBJ_FILENAME = sys.argv[1]
NEW_OBJ_FILENAME = sys.argv[2]  # Overwrite the same file
CONT_OBJ_TYPE = sys.argv[3]  # 'CAS', 'FCI', etc. 
GEOM_FILENAME = sys.argv[4]
BASIS = sys.argv[5]  # Basis set information
TAG = sys.argv[6]  # Optional tag for reading in previous solutions
    
# TODO: Add support for different types of continuation objects
if CONT_OBJ_TYPE == 'CAS':
    from evcont.CASCI_EVCont import CAS_EVCont_obj as CONT_OBJ_CLASS
else:
    raise ValueError(f"Unsupported CONT_OBJ_TYPE: {CONT_OBJ_TYPE}")

# Load continuation object
try:
    cont_obj = CONT_OBJ_CLASS.load(CONT_OBJ_FILENAME)
except:
    raise RuntimeError(f"Failed to load continuation object from {CONT_OBJ_FILENAME}")

# Load geometry
#mol = cont_obj.mols[0]
mol = gto.Mole()
mol.build(
    atom=GEOM_FILENAME,
    basis=BASIS,
    unit='Angstrom',
    verbose=0
)

# Append the new geometry to the continuation object
print(f"Appending new geometry from {GEOM_FILENAME} to continuation object...")
if cont_obj.software == 'quantel':
    cont_obj.append_to_rdms(mol, quantel_tag=TAG)
else:
    cont_obj.append_to_rdms(mol)

# Save updated continuation object
cont_obj.save(NEW_OBJ_FILENAME)

print(f"Appended new geometry from {GEOM_FILENAME} to continuation object and saved to {NEW_OBJ_FILENAME}")
print('DONE')

