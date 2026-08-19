#!/usr/bin/env python3
"""Compare low-rank CCSD continuation across abstract orbital bases.

For each abstract basis, this example builds a full 2RDM continuation model and
three Hamiltonian-thresholded low-rank models: without diagonal corrections,
with Coulomb (J) diagonals, and with Coulomb-plus-exchange (J+K) diagonals.  It
reports the selected ranks, a scalar-storage proxy, and energy errors on a test
grid relative to both full EVCont and direct CCSD.

The storage proxy counts the independent lower-triangular training pairs.  A
joint-decomposition vector costs ``norb**2 + 1`` scalars (vector plus amplitude),
and each retained diagonal channel costs ``norb**2`` scalars.  It intentionally
does not include representation-independent 1RDM and overlap storage.
"""

import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
from pyscf import cc, gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evcont.ccsd.CCSD_EVCont import CCSD_EVCont_obj
from evcont.logging_utils import logger as evcont_logger


def build_h_chain(natom, spacing_bohr, basis="sto-3g"):
    """Create an equally spaced linear hydrogen chain in Bohr."""
    atoms = [("H", (i * spacing_bohr, 0.0, 0.0)) for i in range(natom)]
    return gto.M(
        atom=atoms,
        basis=basis,
        unit="Bohr",
        symmetry=False,
        verbose=0,
    )


def ccsd_reference_energy(mol):
    """Return a direct PySCF RCCSD total energy."""
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.conv_tol_grad = 1.0e-10
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge for the CCSD reference")

    mycc = cc.CCSD(mf)
    mycc.conv_tol = 1.0e-10
    mycc.kernel()
    if not mycc.converged:
        raise RuntimeError("CCSD did not converge for the reference")
    return float(mycc.e_tot)


def build_continuation(
    comp_mol,
    abstract_basis,
    train_spacings,
    natom,
    orbital_basis,
    lowrank_kwargs=None,
):
    """Build either a full or low-rank CCSD continuation model."""
    lowrank = lowrank_kwargs is not None
    cont = CCSD_EVCont_obj(
        comp_mol,
        abstract_basis=abstract_basis,
        lowrank=lowrank,
        lowrank_kwargs=lowrank_kwargs,
    )

    # reduce_2rdm prints diagnostics for every transition pair.  Suppress those
    # here so that the final comparison table remains the focus of the example.
    with redirect_stdout(StringIO()):
        for spacing in train_spacings:
            cont.append_to_rdms(
                build_h_chain(natom, spacing, basis=orbital_basis)
            )
    return cont


def evaluate_grid(cont, molecules):
    """Evaluate the single CCSD continuation root on a molecular grid."""
    return np.array(
        [float(np.ravel(cont.approximate(mol, nroots=1)[0])[0]) for mol in molecules]
    )


def compactness(cont, diagonal_channels):
    """Return rank statistics and low-rank/full 2RDM scalar-storage ratio."""
    cont.vectorize_lowrank(hermitian=True)
    nvecs = cont.lowrank_vectorized["nvecs"]
    pair_ranks = nvecs[np.tril_indices_from(nvecs)]

    norb = cont.one_rdm.shape[-1]
    npair = pair_ranks.size
    full_scalars = npair * norb**4
    lowrank_scalars = pair_ranks.sum() * (norb**2 + 1)
    lowrank_scalars += npair * diagonal_channels * norb**2

    return (
        float(pair_ranks.mean()),
        int(pair_ranks.max()),
        100.0 * lowrank_scalars / full_scalars,
    )


# H4/cc-pvdz remains inexpensive while giving SAO and meta-Lowdin distinct
# representations and nontrivial CCSD transition 2RDM compression.
natom = 4
orbital_basis = "cc-pvdz"
reference_spacing = 1.5
train_spacings = [1.2, 1.8]
test_spacings = [1.35, 1.50, 1.65]
ham_threshold = 1.0e-3

abstract_bases = ["SAO", "meta_lowdin", "split_procrustes", "least_change"]
diagonal_modes = [
    ("none", False, True, 0),
    ("J", True, True, 1),
    ("J+K", True, False, 2),
]

evcont_logger.disabled = True
comp_mol = build_h_chain(natom, reference_spacing, basis=orbital_basis)
test_molecules = [
    build_h_chain(natom, spacing, basis=orbital_basis)
    for spacing in test_spacings
]
reference_energies = np.array(
    [ccsd_reference_energy(mol) for mol in test_molecules]
)

results = []
for abstract_basis in abstract_bases:
    full = build_continuation(
        comp_mol,
        abstract_basis,
        train_spacings,
        natom,
        orbital_basis,
    )
    full_energies = evaluate_grid(full, test_molecules)
    full_ccsd_errors = 1000.0 * np.abs(full_energies - reference_energies)
    results.append(
        {
            "basis": abstract_basis,
            "diagonal": "full",
            "mean_rank": None,
            "max_rank": None,
            "storage_percent": 100.0,
            "mae_full_mha": 0.0,
            "max_full_mha": 0.0,
            "mae_ccsd_mha": float(full_ccsd_errors.mean()),
        }
    )

    for label, save_diag, jdiag_only, diagonal_channels in diagonal_modes:
        lowrank_kwargs = {
            "truncation_style": "ham",
            "ham_thr": ham_threshold,
            "save_diag": save_diag,
            "Jdiag_only": jdiag_only,
            "relax_amp": False,
            "opt_no_diag": True,
            "use_svd": False,
            # Use exact JK builds for a clean comparison among abstract bases.
            "density_fit": False,
            "sao_diag": False,
        }
        lowrank = build_continuation(
            comp_mol,
            abstract_basis,
            train_spacings,
            natom,
            orbital_basis,
            lowrank_kwargs=lowrank_kwargs,
        )
        mean_rank, max_rank, storage_percent = compactness(
            lowrank, diagonal_channels
        )
        lowrank_energies = evaluate_grid(lowrank, test_molecules)
        full_errors = 1000.0 * np.abs(lowrank_energies - full_energies)
        ccsd_errors = 1000.0 * np.abs(lowrank_energies - reference_energies)

        results.append(
            {
                "basis": abstract_basis,
                "diagonal": label,
                "mean_rank": mean_rank,
                "max_rank": max_rank,
                "storage_percent": storage_percent,
                "mae_full_mha": float(full_errors.mean()),
                "max_full_mha": float(full_errors.max()),
                "mae_ccsd_mha": float(ccsd_errors.mean()),
            }
        )

print("=" * 112)
print("Low-Rank CCSD Continuation: Abstract Bases and Diagonal Corrections")
print("=" * 112)
print(f"System: H{natom}, orbital basis={orbital_basis}, nroots=1")
print(f"Reference spacing (Bohr): {reference_spacing}")
print(f"Training spacings (Bohr): {train_spacings}")
print(f"Test spacings (Bohr):     {test_spacings}")
print(f"Hamiltonian truncation threshold: {ham_threshold:.1e} Ha")
print("-" * 112)
print(
    "abstract basis       diag    mean rank   max rank   storage (%)   "
    "MAE LR-full   max LR-full   MAE LR-CCSD"
)
print(
    "                                                        "
    "          (mHa)          (mHa)          (mHa)"
)
print("-" * 112)

for result in results:
    if result["mean_rank"] is None:
        rank_text = f"{'--':>9s}   {'--':>8s}"
    else:
        rank_text = (
            f"{result['mean_rank']:>9.2f}   {result['max_rank']:>8d}"
        )
    print(
        f"{result['basis']:<20s} {result['diagonal']:<6s} {rank_text}   "
        f"{result['storage_percent']:>11.2f}   "
        f"{result['mae_full_mha']:>11.4f}   "
        f"{result['max_full_mha']:>11.4f}   "
        f"{result['mae_ccsd_mha']:>11.4f}"
    )

print("=" * 112)
print(
    "Rank is per unique training-state pair; the full pair rank is "
    f"{comp_mol.nao**2}. Storage excludes the shared 1RDM and overlap."
)
