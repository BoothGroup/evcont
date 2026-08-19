#!/usr/bin/env python3
"""Predict AO-basis 1RDMs for an ethene EVCont trajectory."""

from __future__ import annotations

import argparse
import sys
import ast
import re
from pathlib import Path

import numpy as np
from pyscf import cc, gto


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = next(
    parent for parent in (SCRIPT_DIR, *SCRIPT_DIR.parents) if (parent / "evcont").is_dir()
)
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from evcont.ab_initio_eigenvector_continuation import approximate_multistate
from evcont.basis.basis_utils import get_basis
from evcont.electron_integral_utils import get_integrals
from evcont.excited_utils import make_rdm1
from evcont.basis.split_procrustes_derivatives import run_rhf


TRAJECTORY_FILE = SCRIPT_DIR / "traj_EVCont_13.npy"
ONE_RDM_FILE = SCRIPT_DIR / "one_rdm.npy"
TWO_RDM_FILE = SCRIPT_DIR / "two_rdm.npy"
OVERLAP_FILE = SCRIPT_DIR / "overlap.npy"

ABSTRACT_BASIS = "split_procrustes"
BASIS_KWARGS = {"procrustes_overlap": "none"}
TRAJECTORY_RE = re.compile(r"^traj_EVCont_(\d+)$")


def load_get_mol():
    """Load get_mol from the MD setup file without importing its MD runner."""

    setup_file = SCRIPT_DIR / "md_ethene_ccpvdz_CCSD_continuation_split_procrustes.py"
    tree = ast.parse(setup_file.read_text())
    get_mol_node = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "get_mol"
    )
    module = ast.Module(body=[get_mol_node], type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {"gto": gto}
    exec(compile(module, str(setup_file), "exec"), namespace)
    return namespace["get_mol"]


def abstract_rdm1_to_ao(rdm1_abstract: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Rotate a 1RDM from the abstract orbital basis back to the AO basis."""

    return np.einsum(
        "pa,ab,qb->pq",
        basis,
        rdm1_abstract,
        basis,
        optimize="optimal",
    )


def infer_ntrain_from_trajectory(trajectory_file: Path) -> int:
    match = TRAJECTORY_RE.match(trajectory_file.stem)
    if match is None:
        raise ValueError(
            f"Could not infer continuation subset from trajectory name {trajectory_file.name}. "
            "Expected a name like traj_EVCont_9.npy."
        )
    return int(match.group(1)) + 1


def explicit_ccsd_rdm1_ao(mol):
    """Compute an explicit CCSD 1RDM in the AO basis for one molecule."""

    mf = run_rhf(mol)
    ccsd = cc.CCSD(mf)
    ccsd.verbose = 0
    ccsd.kernel()
    if not ccsd.converged:
        print("Warning: explicit first-frame CCSD did not converge")
    return ccsd.make_rdm1(ao_repr=True)


def compare_first_frame_rdm(
    mol,
    predicted_one_rdm_abstract,
    predicted_one_rdm_ao,
    first_ccsd_one_rdm_ao_out,
    first_rdm_diff_out,
):
    explicit_one_rdm_ao = explicit_ccsd_rdm1_ao(mol)
    diff = predicted_one_rdm_ao - explicit_one_rdm_ao
    overlap_ao = mol.intor_symmetric("int1e_ovlp")

    explicit_norm = np.linalg.norm(explicit_one_rdm_ao)
    rel_frob = np.linalg.norm(diff) / explicit_norm

    print("First-frame AO 1RDM check:")
    print(f"  Predicted abstract-basis trace: {np.trace(predicted_one_rdm_abstract):.12e}")
    print(f"  Frobenius norm difference: {np.linalg.norm(diff):.12e}")
    print(f"  Relative Frobenius difference: {rel_frob:.12e}")
    print(f"  Max absolute difference: {np.max(np.abs(diff)):.12e}")
    print(
        "  Predicted electron count Tr(SD): "
        f"{np.einsum('pq,qp->', overlap_ao, predicted_one_rdm_ao):.12e}"
    )
    print(
        "  Explicit CCSD electron count Tr(SD): "
        f"{np.einsum('pq,qp->', overlap_ao, explicit_one_rdm_ao):.12e}"
    )

    np.save(first_ccsd_one_rdm_ao_out, explicit_one_rdm_ao)
    np.save(first_rdm_diff_out, diff)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict AO-basis 1RDMs for an ethene EVCont trajectory."
    )
    parser.add_argument(
        "trajectory",
        nargs="?",
        type=Path,
        default=TRAJECTORY_FILE,
        help="Trajectory .npy file. Defaults to traj_EVCont_13.npy.",
    )
    parser.add_argument(
        "--check-first",
        action="store_true",
        help="Compute explicit CCSD AO 1RDM for frame 0 and compare it to prediction.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    get_mol = load_get_mol()

    trajectory_file = args.trajectory
    if not trajectory_file.is_absolute():
        trajectory_file = SCRIPT_DIR / trajectory_file
    output_prefix = SCRIPT_DIR / trajectory_file.stem
    mols_out = output_prefix.with_name(f"{output_prefix.name}_mols.npy")
    one_rdm_ao_out = output_prefix.with_name(
        f"{output_prefix.name}_predicted_one_rdm_ao.npy"
    )
    first_ccsd_one_rdm_ao_out = output_prefix.with_name(
        f"{output_prefix.name}_first_ccsd_one_rdm_ao.npy"
    )
    first_rdm_diff_out = output_prefix.with_name(
        f"{output_prefix.name}_first_one_rdm_ao_diff.npy"
    )

    ntrain = infer_ntrain_from_trajectory(trajectory_file)
    trajectory = np.load(trajectory_file)
    one_rdm_all = np.load(ONE_RDM_FILE, mmap_mode="r")
    two_rdm_all = np.load(TWO_RDM_FILE, mmap_mode="r")
    overlap_all = np.load(OVERLAP_FILE, mmap_mode="r")
    if ntrain > one_rdm_all.shape[0] or ntrain > two_rdm_all.shape[0]:
        raise ValueError(
            f"Requested {ntrain} prelim states from {trajectory_file.name}, "
            f"but only {one_rdm_all.shape[0]} are available."
        )
    one_rdm = one_rdm_all[:ntrain, :ntrain]
    two_rdm = two_rdm_all[:ntrain, :ntrain]
    overlap = overlap_all[:ntrain, :ntrain]

    print(f"Using first {ntrain} continuation prelim states for {trajectory_file.name}")

    reference_mol = get_mol(trajectory[0])
    reference_mf = run_rhf(reference_mol)

    mols = []
    predicted_one_rdms_ao = []

    for iframe, geometry in enumerate(trajectory):
        mol = get_mol(geometry)

        basis = get_basis(
            mol,
            basis_type=ABSTRACT_BASIS,
            basis_ref_mol=reference_mol,
            ref_mf=reference_mf,
            **BASIS_KWARGS,
        )
        h1, h2 = get_integrals(mol, basis)

        _, eigenvectors = approximate_multistate(
            h1,
            h2,
            one_rdm,
            two_rdm,
            overlap,
            nroots=1,
            hermitian=True,
        )

        predicted_one_rdm_abstract = make_rdm1(mol, one_rdm, eigenvectors[0])
        predicted_one_rdm_ao = abstract_rdm1_to_ao(predicted_one_rdm_abstract, basis)

        mols.append(mol)
        predicted_one_rdms_ao.append(predicted_one_rdm_ao)

        if iframe == 0 and args.check_first:
            compare_first_frame_rdm(
                mol,
                predicted_one_rdm_abstract,
                predicted_one_rdm_ao,
                first_ccsd_one_rdm_ao_out,
                first_rdm_diff_out,
            )

        print(f"Finished frame {iframe + 1}/{len(trajectory)}")

    np.save(mols_out, np.asarray(mols, dtype=object), allow_pickle=True)
    np.save(one_rdm_ao_out, np.asarray(predicted_one_rdms_ao), allow_pickle=True)

    print(f"Saved mol objects to {mols_out}")
    print(f"Saved AO-basis predicted 1RDMs to {one_rdm_ao_out}")


if __name__ == "__main__":
    main()
