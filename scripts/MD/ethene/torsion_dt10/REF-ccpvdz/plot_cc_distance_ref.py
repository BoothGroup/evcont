#!/usr/bin/env python3
"""Plot C=C distance and HCCH dihedral vs time for REF and EVCont trajectories.

Loads trajectory_x.npy files from REF and the largest traj_EVCont_N.npy from the
parent folder.

Assumed atom order:
    0:C, 1:C, 2:H, 3:H, 4:H, 5:H

Representative torsion tracked here:
    |H(2)-C(0)-C(1)-H(5)|
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os

AU_TO_FS = 0.024188843265857
DT_AU = 10.0
TIME_PER_STEP_FS = DT_AU * AU_TO_FS

REF_DIR = Path(__file__).resolve().parent
PARENT_DIR = REF_DIR.parent
# Modify parent directory to another path if needed, e.g., for a different run or solver.
drc = 'CCSD_ccpvdz_split_procrustes'
PARENT_DIR = Path('../' + drc)

def cc_distance(traj):
    """Return C=C distance in Bohr for every frame."""
    return np.linalg.norm(traj[:, 0, :] - traj[:, 1, :], axis=1)


def dihedral_angle(p0, p1, p2, p3):
    """Return signed dihedral angle (degrees) for four points."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2

    b1_norm = np.linalg.norm(b1)
    if b1_norm == 0.0:
        return 0.0
    b1_unit = b1 / b1_norm

    v = b0 - np.dot(b0, b1_unit) * b1_unit
    w = b2 - np.dot(b2, b1_unit) * b1_unit

    x = np.dot(v, w)
    y = np.dot(np.cross(b1_unit, v), w)
    return np.degrees(np.arctan2(y, x))


def hcch_dihedral(traj, h_left=2, c_left=0, c_right=1, h_right=5):
    vals = np.zeros(traj.shape[0], dtype=float)
    for i in range(traj.shape[0]):
        p0 = traj[i, h_left]
        p1 = traj[i, c_left]
        p2 = traj[i, c_right]
        p3 = traj[i, h_right]
        vals[i] = dihedral_angle(p0, p1, p2, p3)
    return vals


def load_ref_trajectories(folder: Path):
    """Load all trajectory_x.npy files; return list of (label, array)."""
    pattern = re.compile(r"trajectory_(.+)\.npy$")
    results = []
    for path in sorted(folder.glob("trajectory_*.npy")):
        m = pattern.match(path.name)
        if m:
            results.append((m.group(1), np.load(path)))
    return results


def load_largest_evcont(folder: Path):
    """Return (N, array) for the largest traj_EVCont_N.npy in folder."""
    pattern = re.compile(r"traj_EVCont_(\d+)\.npy$")
    candidates = []
    for path in folder.glob("traj_EVCont_*.npy"):
        m = pattern.match(path.name)
        if m:
            candidates.append((int(m.group(1)), path))
    if not candidates:
        return None
    n, path = max(candidates, key=lambda x: x[0])
    return n, np.load(path)


def main():
    ref_trajs = load_ref_trajectories(REF_DIR)
    evcont = load_largest_evcont(PARENT_DIR)

    if not ref_trajs and evcont is None:
        raise SystemExit("No trajectory files found.")

    all_series = []
    for label, traj in ref_trajs:
        all_series.append((label, traj))
    if evcont is not None:
        n, traj = evcont
        all_series.append((f"N={n}", traj))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    cmap = cm.plasma
    n_series = len(all_series)

    for idx, (label, traj) in enumerate(all_series):
        d_cc = cc_distance(traj)
        phi = -hcch_dihedral(traj, h_left=2, c_left=0, c_right=1, h_right=5)
        time_fs = np.arange(len(d_cc)) * TIME_PER_STEP_FS
        color = cmap(idx / max(1, n_series - 1))
        lw = 2.5 if label.startswith("N=") else 1.8
        ls = "-" if label.startswith("N=") else "--"
        ax1.plot(time_fs, d_cc, color=color, lw=lw, ls=ls, label=label)
        ax2.plot(time_fs, phi, color=color, lw=lw, ls=ls, label=label)

    ax1.set_ylabel("C=C distance (Bohr)")
    ax1.set_title("Ethene torsion run: REF vs EVCont")
    ax1.grid(True, alpha=0.25)

    ax2.set_xlabel("Time (fs)")
    ax2.set_ylabel("HCCH dihedral (deg)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=10, frameon=False)
    fig.tight_layout()

    out_png = REF_DIR / "cc_and_dihedral_ref.png"
    out_pdf = REF_DIR / "cc_and_dihedral_ref.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
