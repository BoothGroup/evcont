#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_append_to_rdms.py

Helper script to append new geometries to an existing continuation object.
Can be submitted as an HPC job within an active learning workflow (as used in NAMD_utils.py).

Inputs (via CLI args):
  CONT_OBJ_FILENAME    Path to existing continuation object pickle file
  NEW_OBJ_FILENAME     Path to save updated continuation object
  CONT_OBJ_TYPE        Type of continuation object ('CAS', 'FCI', etc.)
  GEOM_FILENAME        XYZ geometry file for the new geometry
  BASIS                Basis set string (e.g., '6-31g')
  TAG                  Optional tag for reading in previous solutions (e.g., 'ref', 'geom0')

Operation:
  1. Load existing continuation object from pickle file.
  2. Build molecule from geometry file + basis.
  3. Append new geometry to continuation object (with quantum chemistry calculation).
  4. Save updated continuation object to new file.

Exit codes:
  0 success
  1 failure (exception)

Author: Kemal Atalar
Date: November 2025
"""

import os
import sys
import numpy as np
from pyscf import gto


def main():
    # Parse positional arguments (compatible with HPC job script like append_states.sh)
    if len(sys.argv) < 7:
        print("Usage: python _append_to_rdms.py <cont_obj_file> <new_obj_file> <obj_type> <geom_file> <basis> <tag>")
        return 1

    cont_obj_filename = sys.argv[1]
    new_obj_filename = sys.argv[2]
    cont_obj_type = sys.argv[3]
    geom_filename = sys.argv[4]
    basis = sys.argv[5]
    tag = sys.argv[6]

    try:
        # Import continuation object class based on type
        if cont_obj_type == 'CAS':
            from evcont.CASCI_EVCont import CAS_EVCont_obj as CONT_OBJ_CLASS
        else:
            print(f"Error: Unsupported continuation object type: {cont_obj_type}")
            return 1

        # Load continuation object
        print(f"Loading continuation object from {cont_obj_filename}")
        try:
            cont_obj = CONT_OBJ_CLASS.load(cont_obj_filename)
        except Exception as e:
            print(f"Error: Failed to load continuation object from {cont_obj_filename}")
            print(f"  {e}")
            return 1

        # Build molecule from geometry file
        print(f"Building molecule from {geom_filename} with basis {basis}")
        mol = gto.Mole()
        mol.build(
            atom=geom_filename,
            basis=basis,
            unit='Angstrom',
            verbose=0
        )

        # Append the new geometry to the continuation object
        print(f"Appending new geometry to continuation object (tag: {tag})...")
        if hasattr(cont_obj, 'software') and cont_obj.software == 'quantel':
            cont_obj.append_to_rdms(mol, quantel_tag=tag)
        else:
            cont_obj.append_to_rdms(mol)

        # Save updated continuation object
        print(f"Saving updated continuation object to {new_obj_filename}")
        cont_obj.save(new_obj_filename)

        print(f"Successfully appended geometry from {geom_filename}")
        print("DONE")
        return 0

    except Exception as e:
        print(f"Error appending geometry: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

