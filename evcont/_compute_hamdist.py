#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
_compute_hamdist.py

HPC/offline helper to compute Hamiltonian distances between a trajectory and
existing training geometries for EVCont active learning.

Inputs (via CLI args):
  GEOM_FILENAME               XYZ geometry file (initial geometry to build molecule)
  BASIS                       Basis set string (e.g., '6-31g')
  TRAJECTORY_NPY              Numpy file containing trajectory geometries (nsteps, natm, 3)
  TRN_GEOMETRIES_NPY          Numpy file containing training geometries (ntrn, natm, 3)
  OUTPUT_FILE                 Output text filename for ham distances
  [CACHE_PREFIX]              Optional prefix for cached training integrals (default: training_integrals)
  [--force-recompute]         Optional flag to ignore existing cached integrals

Operation:
  1. Build molecule from geometry file + basis (like _append_to_rdms.py).
  2. Load training geometries and trajectory geometries from numpy files.
  3. (Re)compute one- and two-electron integrals for training geometries unless
     cached arrays exist (<prefix>_h1.npy / <prefix>_h2.npy).
  4. Loop over trajectory geometries, compute integrals, evaluate minimum
     Hamiltonian distance to training set (using evcont.NAMD_utils.hamiltonian_distance).
  5. Write distances to plain text file (np.savetxt).

The script deliberately avoids modifying any continuation object; it is purely
computational. Caching prevents repeated recomputation of training integrals
across iterations.

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
from evcont.electron_integral_utils import get_integrals, get_basis
from evcont.NAMD_utils import hamiltonian_distance


def build_molecule(geom_filename, basis):
    """Build PySCF molecule from geometry file and basis string."""
    mol = gto.Mole()
    mol.build(
        atom=geom_filename,
        basis=basis,
        unit='Angstrom',
        verbose=0
    )
    return mol


def compute_training_integrals(init_mol, trn_geometries, cache_prefix, force):
    """Return (h1_trn, h2_trn) arrays, appending to cache when training set grows.

    Behavior:
    - If cache exists and --force-recompute is not given, reuse cached entries and
      only compute integrals for newly added training geometries, then extend the cache.
    - If basis/nao changes (shape mismatch), ignore cache and recompute fully.
    """
    h1_cache = f"{cache_prefix}_h1.npy"
    h2_cache = f"{cache_prefix}_h2.npy"

    ntrn = len(trn_geometries)
    nao = init_mol.nao

    cached_len = 0
    h1_cached = None
    h2_cached = None

    if os.path.isfile(h1_cache) and os.path.isfile(h2_cache) and not force:
        try:
            h1_cached = np.load(h1_cache)
            h2_cached = np.load(h2_cache)
            # Validate cached shapes (other than length dim)
            if h1_cached.shape[1:] != (nao, nao) or h2_cached.shape[1:] != (nao, nao, nao, nao):
                print("Warning: cached integrals have incompatible shape (basis/nao changed); ignoring cache and recomputing.")
                h1_cached = None
                h2_cached = None
            else:
                cached_len = h1_cached.shape[0]
                if cached_len == ntrn:
                    print(f"Loaded cached training integrals from {h1_cache}, {h2_cache}")
                    return h1_cached, h2_cached
                elif cached_len > ntrn:
                    # Truncate to match current trn_geometries length
                    print(f"Warning: cache has more entries ({cached_len}) than trn_geometries ({ntrn}); truncating cache in-memory.")
                    h1_cached = h1_cached[:ntrn]
                    h2_cached = h2_cached[:ntrn]
                    cached_len = ntrn
                else:
                    print(f"Extending cache from {cached_len} -> {ntrn} training geometries...")
        except Exception as e:
            print(f"Warning: failed loading cached integrals; recomputing. Reason: {e}")
            h1_cached = None
            h2_cached = None

    # Allocate full arrays
    h1_trn = np.zeros((ntrn, nao, nao))
    h2_trn = np.zeros((ntrn, nao, nao, nao, nao))

    # If we have valid cached data, copy it into the front
    if h1_cached is not None and h2_cached is not None and cached_len > 0:
        h1_trn[:cached_len] = h1_cached
        h2_trn[:cached_len] = h2_cached

    # Compute missing integrals (either full or only the tail)
    start_i = cached_len if (h1_cached is not None and h2_cached is not None) else 0
    todo = ntrn - start_i
    if todo > 0:
        print(f"Computing integrals for {todo} training geometries (from index {start_i})...")
        for i in range(start_i, ntrn):
            geom = trn_geometries[i]
            mol = init_mol.copy().set_geom_(geom)
            h1, h2 = get_integrals(mol, get_basis(mol))
            h1_trn[i] = h1
            h2_trn[i] = h2
    else:
        print("No new training geometries to compute; using cached integrals.")

    # Persist/extend cache for future iterations
    try:
        np.save(h1_cache, h1_trn)
        np.save(h2_cache, h2_trn)
        if todo > 0:
            print(f"Updated cached training integrals to {h1_cache}, {h2_cache}")
        else:
            print(f"Cache verified: {h1_cache}, {h2_cache}")
    except Exception as e:
        print(f"Warning: failed to save training integrals cache: {e}")

    return h1_trn, h2_trn


def compute_distances(init_mol, trajectory, h1_trn, h2_trn):
    """Compute minimum Hamiltonian distance and argmin training index for each trajectory geometry.

    Returns:
        distances (ndarray): min distance per traj geometry
        argmins (ndarray[int]): index of closest training geometry per traj geometry
    """
    distances = []
    argmins = []
    print(f"Computing Hamiltonian distances for {len(trajectory)} trajectory geometries...")
    for geom in trajectory:
        mol = init_mol.copy().set_geom_(geom)
        h1, h2 = get_integrals(mol, get_basis(mol))
        d_all = hamiltonian_distance(h1, h2, h1_trn, h2_trn)
        distances.append(np.min(d_all))
        argmins.append(int(np.argmin(d_all)))
    return np.array(distances), np.array(argmins, dtype=int)


def main():
    # Parse positional arguments (compatible with HPC job script like append_states.sh)
    if len(sys.argv) < 6:
        print("Usage: python _compute_hamdist.py <geom_file> <basis> <trajectory_npy> <trn_geometries_npy> <output_file> [cache_prefix] [--force-recompute]")
        return 1

    geom_filename = sys.argv[1]
    basis = sys.argv[2]
    trajectory_npy = sys.argv[3]
    trn_geometries_npy = sys.argv[4]
    output_file = sys.argv[5]
    
    # Optional cache prefix (default: training_integrals)
    cache_prefix = sys.argv[6] if len(sys.argv) > 6 and not sys.argv[6].startswith('--') else 'training_integrals'
    
    # Optional force recompute flag
    force_recompute = '--force-recompute' in sys.argv

    try:
        print(f"Building molecule from {geom_filename} with basis {basis}")
        init_mol = build_molecule(geom_filename, basis)
        
        print(f"Loading trajectory from {trajectory_npy}")
        trajectory = np.load(trajectory_npy)
        
        print(f"Loading training geometries from {trn_geometries_npy}")
        trn_geometries = np.load(trn_geometries_npy)

        h1_trn, h2_trn = compute_training_integrals(init_mol, trn_geometries, cache_prefix, force_recompute)
        distances, argmins = compute_distances(init_mol, trajectory, h1_trn, h2_trn)

        # Write distances
        np.savetxt(output_file, distances)
        print(f"Hamiltonian distances written to {output_file}")

        # Also write argmin indices to a sibling file for downstream use
        argmin_file = output_file.replace('ham_dist', 'ham_argmin')
        try:
            np.savetxt(argmin_file, argmins, fmt='%d')
            print(f"Closest training indices written to {argmin_file}")
        except Exception as e:
            print(f"Warning: could not write argmin indices to {argmin_file}: {e}")
        print("DONE")
        return 0
    except Exception as e:
        print(f"Error computing Hamiltonian distances: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
