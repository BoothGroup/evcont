#!/usr/bin/env python3
"""
Water CCSD continuation along a reaction coordinate in multiple abstract bases.

This follows the SCI continuation pattern: CCSD amplitudes for AO-like bases are
computed in a reference computational gauge, then mixed CCSD RDMs are transformed
back to the selected abstract representation for transferable ground-state
energy and gradient evaluation.  Direct PySCF CCSD energies and analytic
gradients are used as references.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

mpl_cache = Path(tempfile.gettempdir()) / "evcont-matplotlib"
xdg_cache = Path(tempfile.gettempdir()) / "evcont-xdg-cache"
mpl_cache.mkdir(parents=True, exist_ok=True)
xdg_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pyscf import cc, gto, scf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evcont.CCSD_EVCont import CCSD_EVCont_obj
from evcont.basis_utils import normalize_basis_type


DIFFERENTIABLE_BASES = {"SAO", "meta_lowdin", "split_procrustes"}
ENERGY_ONLY_BASES = set()


def water_mol(d_oh=0.96, angle=104.5, asymmetry=0.0, basis="sto-3g"):
    theta = np.deg2rad(angle)
    half = 0.5 * theta
    d1 = d_oh * (1.0 + asymmetry)
    d2 = d_oh * (1.0 - asymmetry)
    atom = [
        ("O", (0.0, 0.0, 0.0)),
        ("H", (0.0, d1 * np.sin(half), -d1 * np.cos(half))),
        ("H", (0.0, -d2 * np.sin(half), -d2 * np.cos(half))),
    ]
    return gto.M(atom=atom, basis=basis, unit="Angstrom", verbose=0)


def basis_label(basis_name):
    normalized = normalize_basis_type(basis_name)
    if normalized == "meta_lowdin":
        return "meta-Lowdin"
    if normalized == "split_procrustes":
        return "split-Procrustes"
    return basis_name


def run_rhf(mol):
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.conv_tol_grad = 1.0e-10
    mf.conv_tol_cpscf = 1.0e-10
    mf.max_cycle = 100
    mf.kernel()
    return mf


def coordinate_values(args):
    if args.coord == "angle":
        train_values = np.asarray(args.train_angles, dtype=float)
        test_values = np.linspace(args.angle_min, args.angle_max, args.test_points)
        x_label = r"H-O-H angle ($^\circ$)"
    else:
        train_values = np.asarray(args.train_bonds, dtype=float)
        test_values = np.linspace(args.bond_min, args.bond_max, args.test_points)
        x_label = r"O-H distance ($\AA$)"
    return train_values, test_values, x_label


def mol_from_coordinate(value, args):
    if args.coord == "angle":
        return water_mol(args.bond, value, asymmetry=args.asymmetry, basis=args.basis)
    return water_mol(value, args.angle, asymmetry=args.asymmetry, basis=args.basis)


def comp_mol_from_args(args):
    return water_mol(
        args.comp_bond,
        args.comp_angle,
        asymmetry=args.asymmetry,
        basis=args.basis,
    )


def direct_ccsd_ground_reference(test_values, args):
    energies = np.zeros(len(test_values))
    gradients = []
    abs_grad = np.zeros(len(test_values))
    for i, value in enumerate(test_values):
        mol = mol_from_coordinate(value, args)
        mf = run_rhf(mol)
        mycc = cc.CCSD(mf)
        mycc.conv_tol = 1.0e-8
        mycc.kernel()
        grad_i = mycc.nuc_grad_method().kernel()
        energies[i] = mycc.e_tot
        gradients.append(grad_i)
        abs_grad[i] = np.linalg.norm(grad_i)
    return {
        "energies": energies,
        "grad": np.asarray(gradients),
        "abs_grad": abs_grad,
    }


def evaluate_continuation(cont, test_values, args, include_properties):
    energies = np.zeros(len(test_values))
    abs_grad = np.full(len(test_values), np.nan)
    grad_all = []

    for i, value in enumerate(test_values):
        mol = mol_from_coordinate(value, args)
        if include_properties:
            _, en_i, grad_i, _, _ = cont.get_energy_with_grad(mol, nroots=1)
            energies[i] = en_i[0]
            grad_all.append(grad_i[0])
            abs_grad[i] = np.linalg.norm(grad_i[0])
        else:
            en_i, _ = cont.approximate(mol, nroots=1)
            energies[i] = en_i[0]

    return {
        "energies": energies,
        "abs_grad": abs_grad,
        "grad": np.asarray(grad_all),
        "train_energies": np.asarray(cont.train_energies),
    }


def plot_results(test_values, train_values, x_label, reference, data_by_basis, out_path):
    basis_styles = {
        "SAO": {"color": "#006EAF", "ls": "--"},
        "meta_lowdin": {"color": "#D24000", "ls": ":"},
        "split_procrustes": {"color": "#426A5A", "ls": "-."},
    }
    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        sharex=True,
        figsize=(8.4, 6.2),
        gridspec_kw={"hspace": 0.08, "wspace": 0.30},
        height_ratios=[1.5, 1.0],
    )

    axes[0][0].plot(
        test_values,
        reference["energies"],
        color="#222222",
        alpha=0.35,
        lw=2,
        label="direct CCSD",
    )
    axes[0][1].plot(
        test_values,
        reference["abs_grad"],
        color="#222222",
        alpha=0.35,
        lw=2,
        label="direct CCSD",
    )

    for basis_name, data in data_by_basis.items():
        normalized = normalize_basis_type(basis_name)
        style = basis_styles.get(normalized, {"color": "#4B5563", "ls": "-."})
        label_basis = basis_label(basis_name)
        axes[0][0].plot(
            test_values,
            data["energies"],
            color=style["color"],
            ls=style["ls"],
            lw=1.6,
            alpha=0.8,
            label=label_basis,
        )
        axes[1][0].plot(
            test_values,
            np.abs(data["energies"] - reference["energies"]),
            color=style["color"],
            ls=style["ls"],
            lw=1.4,
            alpha=0.8,
        )

        if normalized in ENERGY_ONLY_BASES:
            continue

        axes[0][1].plot(
            test_values,
            data["abs_grad"],
            color=style["color"],
            ls=style["ls"],
            lw=1.6,
            alpha=0.78,
            label=label_basis,
        )
        force_error = np.linalg.norm(data["grad"] - reference["grad"], axis=(1, 2))
        axes[1][1].plot(
            test_values,
            force_error,
            color=style["color"],
            ls=style["ls"],
            lw=1.4,
            alpha=0.78,
            label=label_basis,
        )

    for ax in axes[1]:
        ax.set_yscale("log")

    for row in axes:
        for ax in row:
            ylims = ax.get_ylim()
            ax.vlines(train_values, ymin=ylims[0], ymax=ylims[1], ls="--", color="gray", alpha=0.45)
            ax.set_ylim(ylims)
            ax.grid(ls=":", alpha=0.6)

    axes[0][0].set_title("Energies")
    axes[0][1].set_title("Force Norms")

    axes[0][0].set_ylabel("Energy (Ha)")
    axes[0][1].set_ylabel(r"$||F||$ (Ha $a_0^{-1}$)")
    axes[1][0].set_ylabel(r"$|E_\mathrm{CCSD} - E_\mathrm{cont}|$ (Ha)")
    axes[1][1].set_ylabel(r"$||F_\mathrm{CCSD} - F_\mathrm{cont}||$ (Ha $a_0^{-1}$)")

    fig.text(0.5, 0.02, x_label, ha="center")
    for ax, fontsize in [
        (axes[0][0], 8),
        (axes[0][1], 7),
        (axes[1][1], 7),
    ]:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", fontsize=fontsize)

    fig.savefig(out_path, bbox_inches="tight", dpi=300)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="cc-pvdz", help="PySCF AO basis")
    parser.add_argument("--coord", choices=["angle", "bond"], default="angle")
    parser.add_argument("--bond", type=float, default=0.96, help="Fixed O-H bond in Angstrom")
    parser.add_argument("--angle", type=float, default=104.5, help="Fixed H-O-H angle in degrees")
    parser.add_argument("--asymmetry", type=float, default=0.0)
    parser.add_argument("--comp-bond", type=float, default=1.05)
    parser.add_argument("--comp-angle", type=float, default=120.0)
    parser.add_argument("--train-angles", type=float, nargs="+", default=[70.0, 104.5, 140.0])
    parser.add_argument("--train-bonds", type=float, nargs="+", default=[0.85, 0.96, 1.12])
    parser.add_argument("--angle-min", type=float, default=55.0)
    parser.add_argument("--angle-max", type=float, default=160.0)
    parser.add_argument("--bond-min", type=float, default=0.78)
    parser.add_argument("--bond-max", type=float, default=1.25)
    parser.add_argument("--test-points", type=int, default=21)
    parser.add_argument("--nroots", type=int, default=1)
    parser.add_argument(
        "--abstract-bases",
        nargs="+",
        default=["SAO", "meta_lowdin", "split_procrustes"],
    )
    parser.add_argument(
        "--procrustes-density-fit",
        action="store_true",
        help="Use density-fitted RHF objects when constructing split-Procrustes bases.",
    )
    parser.add_argument(
        "--procrustes-ref-density-fit",
        action="store_true",
        help="Use a density-fitted RHF object for only the split-Procrustes reference.",
    )
    parser.add_argument("--procrustes-df-basis", default=None)
    parser.add_argument("--procrustes-ref-df-basis", default=None)
    parser.add_argument(
        "--out",
        default=str(Path(__file__).with_name("water_ccsd_energy_gradient_basis_comparison_object.png")),
    )
    parser.add_argument("--quick", action="store_true", help="Use a tiny scan for smoke testing")
    parser.add_argument("--verbose-ccsd", action="store_true", help="Show EBCC iteration output")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.basis = "sto-3g"
        args.test_points = 5
        args.train_angles = [80.0, 110.0, 140.0]
        args.train_bonds = [0.86, 0.96, 1.10]
    if args.nroots != 1:
        raise ValueError("CCSD continuation is currently implemented only for nroots=1")

    train_values, test_values, x_label = coordinate_values(args)
    train_mols = [mol_from_coordinate(value, args) for value in train_values]
    comp_mol = comp_mol_from_args(args)

    print("Computing direct ground-state CCSD energy/gradient reference", flush=True)
    reference = direct_ccsd_ground_reference(test_values, args)

    data_by_basis = {}
    for abstract_basis in args.abstract_bases:
        print(f"Training CCSD continuation in {abstract_basis}", flush=True)
        abstract_basis_kwargs = {"procrustes_overlap": "none"}
        if normalize_basis_type(abstract_basis) == "split_procrustes":
            abstract_basis_kwargs.update(
                {
                    "procrustes_density_fit": args.procrustes_density_fit,
                    "procrustes_ref_density_fit": args.procrustes_ref_density_fit
                    or args.procrustes_density_fit,
                    "procrustes_df_basis": args.procrustes_df_basis,
                    "procrustes_ref_df_basis": args.procrustes_ref_df_basis
                    or args.procrustes_df_basis,
                }
            )
        cont = CCSD_EVCont_obj(
            comp_mol,
            nroots=1,
            abstract_basis=abstract_basis,
            abstract_basis_kwargs=abstract_basis_kwargs,
            verbose_ccsd=args.verbose_ccsd,
        )
        for mol in train_mols:
            cont.append_to_rdms(mol)
        include_properties = normalize_basis_type(abstract_basis) in DIFFERENTIABLE_BASES
        print(f"Evaluating {abstract_basis}", flush=True)
        data_by_basis[abstract_basis] = evaluate_continuation(
            cont,
            test_values,
            args,
            include_properties=include_properties,
        )

    out_path = Path(args.out).resolve()
    plot_results(test_values, train_values, x_label, reference, data_by_basis, out_path)
    print(f"Saved plot to {out_path}", flush=True)


if __name__ == "__main__":
    main()
