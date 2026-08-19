#!/usr/bin/env python3
"""Diagnostics for predicted AO-basis 1RDMs along traj_EVCont_13."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(SCRIPT_DIR / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(SCRIPT_DIR / ".cache"))

import matplotlib.pyplot as plt
import numpy as np
from pyscf import dft
from pyscf.scf import hf


DEFAULT_MOLS = SCRIPT_DIR / "traj_EVCont_13_mols.npy"
DEFAULT_RDMS = SCRIPT_DIR / "traj_EVCont_13_predicted_one_rdm_ao.npy"
DEFAULT_OUTDIR = SCRIPT_DIR / "rdm_diagnostics_traj13"


def parse_frame_list(frame_spec: str, nframe: int) -> list[int]:
    if frame_spec.lower() == "all":
        return list(range(nframe))

    frames = []
    for item in frame_spec.split(","):
        frame = int(item.strip())
        if frame < 0:
            frame += nframe
        if frame < 0 or frame >= nframe:
            raise ValueError(f"Frame {frame} is outside [0, {nframe})")
        frames.append(frame)
    return sorted(set(frames))


def symmetric_matrix_power(matrix: np.ndarray, power: float) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(matrix)
    if np.any(eigvals <= 0.0):
        raise ValueError("AO overlap matrix is not positive definite")
    return (eigvecs * eigvals**power) @ eigvecs.T


def rdm_metrics(mol, rdm: np.ndarray) -> dict[str, float]:
    overlap = mol.intor_symmetric("int1e_ovlp")
    rdm_sym = 0.5 * (rdm + rdm.T)
    s_half = symmetric_matrix_power(overlap, 0.5)
    rdm_orth = s_half @ rdm_sym @ s_half
    natural_occupations = np.linalg.eigvalsh(rdm_orth)

    electron_count = np.einsum("pq,qp->", overlap, rdm)
    hermiticity_error = np.linalg.norm(rdm - rdm.T) / max(np.linalg.norm(rdm), 1.0e-15)
    idem_error = np.linalg.norm(rdm @ overlap @ rdm - 2.0 * rdm)
    idem_error /= max(np.linalg.norm(2.0 * rdm), 1.0e-15)

    return {
        "nelectron": float(mol.nelectron),
        "electron_count": float(electron_count),
        "electron_count_error": float(electron_count - mol.nelectron),
        "hermiticity_rel_error": float(hermiticity_error),
        "min_natural_occ": float(natural_occupations[0]),
        "max_natural_occ": float(natural_occupations[-1]),
        "natural_occ_below_0": float(np.count_nonzero(natural_occupations < -1.0e-6)),
        "natural_occ_above_2": float(np.count_nonzero(natural_occupations > 2.0 + 1.0e-6)),
        "idempotency_rel_error": float(idem_error),
    }


def density_plane(mol, rdm: np.ndarray, grid_size: int, margin: float):
    coords = mol.atom_coords()
    center = coords.mean(axis=0)
    centered = coords - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axis_u = vh[0]
    axis_v = vh[1]

    projected_u = centered @ axis_u
    projected_v = centered @ axis_v
    u_values = np.linspace(projected_u.min() - margin, projected_u.max() + margin, grid_size)
    v_values = np.linspace(projected_v.min() - margin, projected_v.max() + margin, grid_size)
    uu, vv = np.meshgrid(u_values, v_values, indexing="xy")
    grid_coords = center + uu.reshape(-1, 1) * axis_u + vv.reshape(-1, 1) * axis_v

    ao = dft.numint.eval_ao(mol, grid_coords)
    rho = dft.numint.eval_rho(mol, ao, rdm).reshape(grid_size, grid_size)
    atom_uv = np.column_stack((projected_u, projected_v))
    return u_values, v_values, rho, atom_uv, mol.atom_charges()


def save_density_plot(
    mol,
    rdm: np.ndarray,
    frame: int,
    outdir: Path,
    grid_size: int,
    margin: float,
    plot_scale: str,
):
    u_values, v_values, rho, atom_uv, charges = density_plane(mol, rdm, grid_size, margin)
    plot_rho = rho
    colorbar_label = "electron density"
    if plot_scale == "log":
        positive = rho[rho > 0.0]
        floor = positive.min() if positive.size else 1.0e-16
        plot_rho = np.log10(np.clip(rho, floor, None))
        colorbar_label = "log10 electron density"

    fig, ax = plt.subplots(figsize=(6.0, 5.0), constrained_layout=True)
    mesh = ax.pcolormesh(u_values, v_values, plot_rho, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label=colorbar_label)
    ax.scatter(atom_uv[:, 0], atom_uv[:, 1], c="white", edgecolors="black", s=60, zorder=3)
    for atom_index, (xy, charge) in enumerate(zip(atom_uv, charges)):
        symbol = mol.atom_symbol(atom_index)
        ax.text(xy[0], xy[1], symbol, ha="center", va="center", fontsize=8, zorder=4)
    ax.set_title(f"Frame {frame} density slice")
    ax.set_xlabel("principal axis 1 / Bohr")
    ax.set_ylabel("principal axis 2 / Bohr")
    fig.savefig(outdir / f"density_frame_{frame:04d}_{plot_scale}.png", dpi=200)
    plt.close(fig)


def write_summary(summary_path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = ["frame", *[key for key in rows[0] if key != "frame"]]
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mols", type=Path, default=DEFAULT_MOLS)
    parser.add_argument("--rdms", type=Path, default=DEFAULT_RDMS)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--frames",
        default="0,150,-1",
        help="Comma-separated frames to plot/check in detail, or 'all'. Negative indices work.",
    )
    parser.add_argument("--grid-size", type=int, default=160)
    parser.add_argument("--margin", type=float, default=3.0, help="Plot margin in Bohr.")
    parser.add_argument("--plot-scale", choices=("log", "linear"), default="log")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    mols = np.load(args.mols, allow_pickle=True)
    rdms = np.load(args.rdms, allow_pickle=True)
    if len(mols) != len(rdms):
        raise ValueError(f"mols and rdms have different lengths: {len(mols)} vs {len(rdms)}")

    frames = parse_frame_list(args.frames, len(mols))
    rows = []
    for frame in frames:
        mol = mols[frame]
        rdm = np.asarray(rdms[frame])
        if rdm.shape != (mol.nao, mol.nao):
            raise ValueError(f"Frame {frame} RDM shape {rdm.shape} does not match mol.nao={mol.nao}")

        metrics = rdm_metrics(mol, rdm)
        charges = hf.mulliken_meta(mol, rdm, verbose=0)[1]
        metrics.update(
            {
                "min_mulliken_charge": float(np.min(charges)),
                "max_mulliken_charge": float(np.max(charges)),
            }
        )
        rows.append({"frame": frame, **metrics})

        if not args.no_plots:
            save_density_plot(
                mol,
                rdm,
                frame,
                args.outdir,
                args.grid_size,
                args.margin,
                args.plot_scale,
            )

    summary_path = args.outdir / "summary.csv"
    write_summary(summary_path, rows)

    print(f"Read {len(mols)} mols and {len(rdms)} AO 1RDMs")
    print(f"Wrote summary to {summary_path}")
    if not args.no_plots:
        print(f"Wrote density plots to {args.outdir}")
    for row in rows:
        print(
            "Frame {frame}: Ne={electron_count:.8f}, "
            "herm={hermiticity_rel_error:.2e}, "
            "occ=[{min_natural_occ:.4f}, {max_natural_occ:.4f}], "
            "Mulliken charge=[{min_mulliken_charge:.4f}, {max_mulliken_charge:.4f}]".format(
                **row
            )
        )


if __name__ == "__main__":
    main()
