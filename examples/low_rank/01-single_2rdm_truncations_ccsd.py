#!/usr/bin/env python3
"""
Example: single CCSD 2RDM compression with Hamiltonian-error threshold truncation.

This script compares three diagonal-correction choices:
1) no diagonal correction
2) diagonal J correction
3) diagonal J+K correction

Author: Kemal Atalar
"""

import numpy as np
from pyscf import cc, gto, scf

from evcont.electron_integral_utils import get_basis, get_integrals
from evcont.logging_utils import logger as evcont_logger
from evcont.low_rank_utils import reduce_2rdm, reconstruct_rdm2_joint
# Keep example output focused on the comparison lines below.
evcont_logger.disabled = True

# -----------------------------------------------------------------------
# 1) Build molecule and run RHF + CCSD
# -----------------------------------------------------------------------

mol = gto.M(
    atom="""
    C  0.000000  0.000000  0.000000
    H  0.000000  0.000000  1.089000
    H  1.026719  0.000000 -0.363000
    H -0.513360 -0.889165 -0.363000
    H -0.513360  0.889165 -0.363000
    """,
    basis="cc-pvdz",
    unit="Angstrom",
    symmetry=False,
    verbose=0,
)

mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

e_ccsd_elec = mycc.e_tot - mol.energy_nuc()

# -----------------------------------------------------------------------
# 2) Build AO RDMs, then transform to OAO basis
#
#    evcont works in the OAO basis (get_basis / get_integrals use it),
#    so we transform the CCSD AO density matrices accordingly.
# -----------------------------------------------------------------------

rdm1_ao = mycc.make_rdm1(ao_repr=True)
rdm2_ao = mycc.make_rdm2(ao_repr=True)

# OAO transformation matrix S^{-1/2}; its inverse maps AO -> OAO indices
basis_oao = get_basis(mol)
basis_inv = np.linalg.inv(basis_oao)

rdm1 = np.einsum(
    "ap,bq,pq->ab", basis_inv, basis_inv, rdm1_ao, optimize="optimal"
)
rdm2 = np.einsum(
    "ap,bq,cr,ds,pqrs->abcd",
    basis_inv, basis_inv, basis_inv, basis_inv,
    rdm2_ao,
    optimize="optimal",
)

# -----------------------------------------------------------------------
# 3) Reference energy from OAO integrals and OAO RDMs
# -----------------------------------------------------------------------

h1, h2 = get_integrals(mol, basis_oao)

# E = sum_pq h1_pq rdm1_pq + 1/2 sum_pqrs h2_pqrs rdm2_pqrs
e_ref = (
    np.einsum("pq,pq->", h1, rdm1)
    + 0.5 * np.einsum("pqrs,pqrs->", h2, rdm2)
)

print(f"CCSD electronic energy:          {e_ref:+.10f} Ha")
print()

# For a single 2RDM the self-overlap is 1.
ovlp = 1.0

# -----------------------------------------------------------------------
# 4) HAM threshold truncation with three diagonal-correction modes
# -----------------------------------------------------------------------

print("Truncation style: ham (ham_thr=1e-3), no diagonal correction")
lowrank_vecs, diagonals, joint = reduce_2rdm(
    rdm1, rdm2, ovlp,
    truncation_style='ham',
    ham_thr=1e-3,
    save_diag=False, Jdiag_only=True,
    mol=mol, train_en=e_ref,
)
rdm2_rec = reconstruct_rdm2_joint(
    lowrank_vecs, diagonals=diagonals, joint=joint
)
e_rec = (
    np.einsum('pq,pq->', h1, rdm1)
    + 0.5 * np.einsum('pqrs,pqrs->', h2, rdm2_rec)
)
print(
    f"  rank = {len(lowrank_vecs[0])} / {rdm2.shape[0]**2},"
    f"  ||dRDM2|| = {np.linalg.norm(rdm2_rec - rdm2):.4e},"
    f"  |dE| = {abs(e_rec - e_ref):.4e} Ha"
)
print()

print("Truncation style: ham (ham_thr=1e-3), diagonal J correction")
lowrank_vecs, diagonals, joint = reduce_2rdm(
    rdm1, rdm2, ovlp,
    truncation_style='ham',
    ham_thr=1e-3,
    save_diag=True, Jdiag_only=True,
    mol=mol, train_en=e_ref,
)
rdm2_rec = reconstruct_rdm2_joint(
    lowrank_vecs, diagonals=diagonals, joint=joint
)
e_rec = (
    np.einsum('pq,pq->', h1, rdm1)
    + 0.5 * np.einsum('pqrs,pqrs->', h2, rdm2_rec)
)
print(
    f"  rank = {len(lowrank_vecs[0])} / {rdm2.shape[0]**2},"
    f"  ||dRDM2|| = {np.linalg.norm(rdm2_rec - rdm2):.4e},"
    f"  |dE| = {abs(e_rec - e_ref):.4e} Ha"
)
print()

print("Truncation style: ham (ham_thr=1e-3), diagonal J+K correction")
lowrank_vecs, diagonals, joint = reduce_2rdm(
    rdm1, rdm2, ovlp,
    truncation_style='ham',
    ham_thr=1e-3,
    save_diag=True, Jdiag_only=False,
    mol=mol, train_en=e_ref,
)
rdm2_rec = reconstruct_rdm2_joint(
    lowrank_vecs, diagonals=diagonals, joint=joint
)
e_rec = (
    np.einsum('pq,pq->', h1, rdm1)
    + 0.5 * np.einsum('pqrs,pqrs->', h2, rdm2_rec)
)
print(
    f"  rank = {len(lowrank_vecs[0])} / {rdm2.shape[0]**2},"
    f"  ||dRDM2|| = {np.linalg.norm(rdm2_rec - rdm2):.4e},"
    f"  |dE| = {abs(e_rec - e_ref):.4e} Ha"
)
print()


