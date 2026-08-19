#!/usr/bin/env python3
"""
Replot the H4 FCI EVCont NAC figure with multiple abstract bases.

This is a script version of the Faraday-paper plotting notebook workflow.  It
computes direct FCI references once, then overlays EVCont results generated in
SAO and meta-Lowdin abstract bases on the energy, force, and NAC panels.
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
from pyscf import fci, gto

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evcont.FCI_EVCont import FCI_EVCont_obj
from evcont.FCI_NAC import get_FCI_energy_with_grad_and_NAC
from evcont.ab_initio_gradients_loewdin import get_multistate_energy_with_grad_and_NAC
from evcont.basis_utils import normalize_basis_type


def build_h_chain(natom, spacing_bohr, basis, symmetry=False):
    """Create a linear hydrogen chain with fixed spacing in Bohr."""
    atom = [("H", (i * spacing_bohr, 0.0, 0.0)) for i in range(natom)]
    return gto.M(atom=atom, basis=basis, unit="Bohr", symmetry=symmetry, verbose=0)


def pair_labels(nroots):
    """All unique state-pair labels i<j as strings."""
    return [f"{i}{j}" for i in range(nroots) for j in range(i + 1, nroots)]


def gauge_fixed_norm_diff(nac_ref, nac_cont):
    """NACs have a sign gauge, so compare the better of +/- continuation."""
    return min(np.linalg.norm(nac_cont - nac_ref), np.linalg.norm(nac_cont + nac_ref))


def make_solver(fix_singlet=True):
    """Build the spin-adapted FCI solver used by the notebook."""
    solver = fci.direct_spin0.FCI()
    if fix_singlet:
        fci.addons.fix_spin_(solver, ss=0)
    return solver


def compute_reference(test_range, natom, basis, nroots):
    """Compute direct FCI energy, gradient, and NAC reference data."""
    solver = make_solver()
    solver.nroots = nroots

    energies = np.zeros((len(test_range), nroots))
    abs_grad = np.zeros((len(test_range), nroots))
    nac_abs = {label: np.zeros(len(test_range)) for label in pair_labels(nroots)}
    grad_all = []
    nac_all = []

    for i, spacing in enumerate(test_range):
        mol = build_h_chain(natom, spacing, basis=basis)
        en_i, grad_i, nac_i, _ = get_FCI_energy_with_grad_and_NAC(
            mol,
            solver,
            cibasis="OAO",
            nroots=nroots,
        )

        energies[i] = en_i[:nroots]
        grad_all.append(grad_i)
        nac_all.append(nac_i)
        abs_grad[i] = np.linalg.norm(grad_i.reshape(nroots, -1), axis=1)

        for label in nac_abs:
            nac_abs[label][i] = np.linalg.norm(nac_i[label])

    return {
        "energies": energies,
        "grad": grad_all,
        "nac": nac_all,
        "abs_grad": abs_grad,
        "nac_abs": nac_abs,
    }


def compute_continuation(test_range, train_dists, natom, basis, nroots, abstract_basis):
    """Train and evaluate one EVCont model in an abstract basis."""
    solver = make_solver()
    solver.nroots = nroots
    cont = FCI_EVCont_obj(
        nroots=nroots,
        cibasis="OAO",
        cisolver=solver,
        abstract_basis=abstract_basis,
    )

    for spacing in train_dists:
        cont.append_to_rdms(build_h_chain(natom, spacing, basis=basis))

    energies = np.zeros((len(test_range), nroots))
    abs_grad = np.zeros((len(test_range), nroots))
    nac_abs = {label: np.zeros(len(test_range)) for label in pair_labels(nroots)}
    grad_all = []
    nac_all = []
    train_energies = []

    for spacing in train_dists:
        mol = build_h_chain(natom, spacing, basis=basis)
        _, en_i, grad_i, nac_i, _ = get_multistate_energy_with_grad_and_NAC(
            mol,
            cont.one_rdm,
            cont.two_rdm,
            cont.overlap,
            nroots=nroots,
            abstract_basis=abstract_basis,
        )
        train_energies.append(en_i)

    for i, spacing in enumerate(test_range):
        mol = build_h_chain(natom, spacing, basis=basis)
        _, en_i, grad_i, nac_i, _ = get_multistate_energy_with_grad_and_NAC(
            mol,
            cont.one_rdm,
            cont.two_rdm,
            cont.overlap,
            nroots=nroots,
            abstract_basis=abstract_basis,
        )

        energies[i] = en_i
        grad_all.append(grad_i)
        nac_all.append(nac_i)
        abs_grad[i] = np.linalg.norm(grad_i.reshape(nroots, -1), axis=1)

        for label in nac_abs:
            nac_abs[label][i] = np.linalg.norm(nac_i[label])

    return {
        "energies": energies,
        "grad": grad_all,
        "nac": nac_all,
        "abs_grad": abs_grad,
        "nac_abs": nac_abs,
        "train_energies": np.asarray(train_energies),
    }


def summarize_errors(reference, continuation, labels):
    """Build energy, gradient, and NAC error arrays for one continuation result."""
    energy_err = np.abs(continuation["energies"] - reference["energies"])
    grad_err = np.array(
        [
            np.linalg.norm((g_cont - g_ref).reshape(g_ref.shape[0], -1), axis=1)
            for g_cont, g_ref in zip(continuation["grad"], reference["grad"])
        ]
    )
    nac_err = {
        label: np.array(
            [
                gauge_fixed_norm_diff(n_ref[label], n_cont[label])
                for n_ref, n_cont in zip(reference["nac"], continuation["nac"])
            ]
        )
        for label in labels
    }
    return energy_err, grad_err, nac_err


def plot_figure(test_range, train_dists, reference, cont_by_basis, out_path):
    """Recreate the notebook 2x3 figure with basis overlays."""
    nplot_states = max(1, reference["energies"].shape[1] - 1)
    labels = pair_labels(nplot_states)
    basis_styles = {
        "SAO": {"color": "#006EAF", "ls": "--"},
        "meta_lowdin": {"color": "#D24000", "ls": ":"},
    }
    state_colors = ["#EB7300", "#DD2501", "#006EAF", "#751E66"]
    nac_colors = {"01": "#E44C01", "02": "#4E7075", "12": "#933D3B"}

    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        sharex=True,
        figsize=(12.0, 6.2),
        gridspec_kw={"hspace": 0.08, "wspace": 0.32},
        height_ratios=[1.5, 1.0],
    )

    for state_id in range(nplot_states):
        color = state_colors[state_id % len(state_colors)]
        axes[0][0].plot(
            test_range,
            reference["energies"][:, state_id],
            color=color,
            alpha=0.45,
            lw=2,
            label="FCI" if state_id == 0 else None,
        )
        axes[0][1].plot(
            test_range,
            reference["abs_grad"][:, state_id],
            color=color,
            alpha=0.45,
            lw=2,
        )

    for basis_name, cont_data in cont_by_basis.items():
        style = basis_styles.get(
            normalize_basis_type(basis_name),
            {"color": "#4B5563", "ls": "-."},
        )
        energy_err, grad_err, nac_err = summarize_errors(reference, cont_data, labels)
        for state_id in range(nplot_states):
            axes[0][0].plot(
                test_range,
                cont_data["energies"][:, state_id],
                color=style["color"],
                ls=style["ls"],
                lw=1.6,
                alpha=0.75,
                label=basis_name if state_id == 0 else None,
            )
            axes[1][0].plot(
                test_range,
                energy_err[:, state_id],
                color=style["color"],
                ls=style["ls"],
                lw=1.5,
                alpha=0.75,
            )
            axes[0][1].plot(
                test_range,
                cont_data["abs_grad"][:, state_id],
                color=style["color"],
                ls=style["ls"],
                lw=1.6,
                alpha=0.75,
                label=fr"{basis_name} S{state_id}",
            )
            axes[1][1].plot(
                test_range,
                grad_err[:, state_id],
                color=style["color"],
                ls=style["ls"],
                lw=1.5,
                alpha=0.75,
            )

        for label in labels:
            if label not in nac_colors:
                continue
            axes[0][2].plot(
                test_range,
                cont_data["nac_abs"][label],
                color=style["color"],
                ls=style["ls"],
                lw=1.6,
                alpha=0.75,
                label=f"{basis_name} d{label}",
            )
            axes[1][2].plot(
                test_range,
                nac_err[label],
                color=style["color"],
                ls=style["ls"],
                lw=1.5,
                alpha=0.75,
            )

    for label in labels:
        if label not in nac_colors:
            continue
        axes[0][2].plot(
            test_range,
            reference["nac_abs"][label],
            color=nac_colors[label],
            alpha=0.45,
            lw=2,
            label=fr"FCI d{label}",
        )

    for basis_name, cont_data in cont_by_basis.items():
        style = basis_styles.get(
            normalize_basis_type(basis_name),
            {"color": "#4B5563", "ls": "-."},
        )
        axes[0][0].plot(
            train_dists,
            cont_data["train_energies"][:, 0],
            marker="x",
            ls="",
            color=style["color"],
        )

    for ax in axes[1]:
        ax.set_yscale("log")

    for row in axes:
        for ax in row:
            ylims = ax.get_ylim()
            ax.vlines(train_dists, ymin=ylims[0], ymax=ylims[1], ls="--", color="gray", alpha=0.45)
            ax.set_ylim(ylims)
            ax.grid(ls=":", alpha=0.6)

    axes[0][0].set_title("Energies")
    axes[0][1].set_title("Forces")
    axes[0][2].set_title("Nonadiabatic Couplings")

    axes[0][0].set_ylabel("Energy (Ha)")
    axes[0][1].set_ylabel(r"$||\nabla E_i||$ (Ha $a_0^{-1}$)")
    axes[0][2].set_ylabel(r"$||\mathbf{d}_{AB}||$ ($a_0^{-1}$)")

    axes[1][0].set_ylabel(r"$|E_\mathrm{FCI} - E_\mathrm{cont}|$ (Ha)")
    axes[1][1].set_ylabel(r"$||\nabla E_\mathrm{FCI} - \nabla E_\mathrm{cont}||$")
    axes[1][2].set_ylabel(r"$||\mathbf{d}_\mathrm{FCI} - \mathbf{d}_\mathrm{cont}||$")

    fig.text(0.5, 0.02, r"H-H separation ($a_0$)", ha="center")
    axes[0][0].legend(loc="best", fontsize=8)
    axes[0][1].legend(loc="best", fontsize=7, ncol=1)
    axes[0][2].legend(loc="best", fontsize=7, ncol=1)

    fig.savefig(out_path, bbox_inches="tight", dpi=300)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="sto-6g", help="PySCF AO basis")
    parser.add_argument("--natom", type=int, default=4)
    parser.add_argument("--nroots", type=int, default=4)
    parser.add_argument(
        "--test-points",
        type=int,
        default=40,
        help="Number of H-H spacings in the scan",
    )
    parser.add_argument("--test-min", type=float, default=0.8)
    parser.add_argument("--test-max", type=float, default=3.0)
    parser.add_argument(
        "--train-dists",
        type=float,
        nargs="+",
        default=[0.97, 1.76, 2.60],
    )
    parser.add_argument(
        "--abstract-bases",
        nargs="+",
        default=["SAO", "meta_lowdin"],
        help="Abstract bases to overlay for EVCont",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).with_name("H4_nac_basis_comparison.png")),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a tiny scan for a fast smoke test",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.basis = "sto-3g"
        args.nroots = 3
        args.test_points = 5
        args.train_dists = [1.0, 1.8, 2.6]

    test_range = np.linspace(args.test_min, args.test_max, args.test_points)

    print("Computing direct FCI reference", flush=True)
    reference = compute_reference(test_range, args.natom, args.basis, args.nroots)

    cont_by_basis = {}
    for abstract_basis in args.abstract_bases:
        print(f"Computing EVCont in {abstract_basis}", flush=True)
        cont_by_basis[abstract_basis] = compute_continuation(
            test_range,
            args.train_dists,
            args.natom,
            args.basis,
            args.nroots,
            abstract_basis,
        )

    out_path = Path(args.out).resolve()
    plot_figure(test_range, args.train_dists, reference, cont_by_basis, out_path)
    print(f"Saved plot to {out_path}", flush=True)


if __name__ == "__main__":
    main()
