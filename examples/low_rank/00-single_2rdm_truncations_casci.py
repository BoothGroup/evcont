#!/usr/bin/env python3
"""
Example: single CASCI 2RDM with two truncation styles.

This script compares:
1) eigval truncation (eval_thr=1e-4)
2) ham truncation (ham_thr=1e-5 Ha)

Both are run without diagonal correction.
"""

import numpy as np
from pyscf import gto, mcscf, scf

from evcont.electron_integral_utils import get_basis, get_integrals
from evcont.logging_utils import logger as evcont_logger
from evcont.low_rank_utils import reduce_2rdm, reconstruct_rdm2_joint


# Keep example output focused on comparison lines.
evcont_logger.disabled = True

# -----------------------------------------------------------------------
# 1) Build molecule and run RHF + CASCI
# -----------------------------------------------------------------------

mol = gto.M(
    atom="""
    C  0.000000  0.000000  0.000000
    H  0.000000  0.000000  1.089000
    H  1.026719  0.000000 -0.363000
    H -0.513360 -0.889165 -0.363000
    H -0.513360  0.889165 -0.363000
    """,
    basis="6-31g",
    unit="Angstrom",
    symmetry=False,
    verbose=0,
)

mf = scf.RHF(mol)
mf.kernel()

# Use full-valence active space for a simple, robust single-state CASCI example.
ncas = 4
nelecas = 2
mc = mcscf.CASCI(mf, ncas, nelecas)
mc.kernel()

e_casci_elec = mc.e_tot - mol.energy_nuc()

# -----------------------------------------------------------------------
# 2) Build AO RDMs from CASCI active-space RDMs, then transform to OAO
# -----------------------------------------------------------------------

rdm1_cas, rdm2_cas = mc.fcisolver.make_rdm12(mc.ci, mc.ncas, mc.nelecas)
mo_cas = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]

rdm1_ao = np.einsum("pi,ij,qj->pq", mo_cas, rdm1_cas, mo_cas, optimize="optimal")
rdm2_ao = np.einsum(
    "pi,qj,rk,sl,ijkl->pqrs",
    mo_cas,
    mo_cas,
    mo_cas,
    mo_cas,
    rdm2_cas,
    optimize="optimal",
)

basis_oao = get_basis(mol)
basis_inv = np.linalg.inv(basis_oao)

rdm1 = np.einsum("ap,bq,pq->ab", basis_inv, basis_inv, rdm1_ao, optimize="optimal")
rdm2 = np.einsum(
    "ap,bq,cr,ds,pqrs->abcd",
    basis_inv,
    basis_inv,
    basis_inv,
    basis_inv,
    rdm2_ao,
    optimize="optimal",
)

# -----------------------------------------------------------------------
# 3) Reference energy from OAO integrals and OAO RDMs
# -----------------------------------------------------------------------

h1, h2 = get_integrals(mol, basis_oao)
e_ref = np.einsum("pq,pq->", h1, rdm1) + 0.5 * np.einsum("pqrs,pqrs->", h2, rdm2)

print(f"CASCI electronic energy:         {e_casci_elec:+.10f} Ha")
#print(f"Electronic energy from OAO RDMs: {e_ref:+.10f} Ha")
#print(f"Consistency check |dE|:          {abs(e_ref - e_casci_elec):.2e} Ha  (should be ~0)")
print()

ovlp = 1.0

# -----------------------------------------------------------------------
# 4) Compare eigval vs ham, both without diagonal correction
# -----------------------------------------------------------------------

print("Truncation style: eigval (eval_thr=1e-4), no diagonal correction")
lowrank_vecs, diagonals, joint = reduce_2rdm(
    rdm1,
    rdm2,
    ovlp,
    truncation_style="eigval",
    eval_thr=1e-4,
    save_diag=False,
    Jdiag_only=True,
    mol=mol,
    train_en=e_ref,
)
rdm2_rec = reconstruct_rdm2_joint(lowrank_vecs, diagonals=diagonals, joint=joint)
e_rec = np.einsum("pq,pq->", h1, rdm1) + 0.5 * np.einsum("pqrs,pqrs->", h2, rdm2_rec)
print(
    f"  rank = {len(lowrank_vecs[0])},"
    f"  ||dRDM2|| = {np.linalg.norm(rdm2_rec - rdm2):.4e},"
    f"  |dE| = {abs(e_rec - e_ref):.4e} Ha"
)
print()

print("Truncation style: ham (ham_thr=1e-5), no diagonal correction")
lowrank_vecs, diagonals, joint = reduce_2rdm(
    rdm1,
    rdm2,
    ovlp,
    truncation_style="ham",
    ham_thr=1e-5,
    save_diag=False,
    Jdiag_only=True,
    mol=mol,
    train_en=e_ref,
)
rdm2_rec = reconstruct_rdm2_joint(lowrank_vecs, diagonals=diagonals, joint=joint)
e_rec = np.einsum("pq,pq->", h1, rdm1) + 0.5 * np.einsum("pqrs,pqrs->", h2, rdm2_rec)
print(
    f"  rank = {len(lowrank_vecs[0])},"
    f"  ||dRDM2|| = {np.linalg.norm(rdm2_rec - rdm2):.4e},"
    f"  |dE| = {abs(e_rec - e_ref):.4e} Ha"
)
print()
