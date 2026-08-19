#!/usr/bin/env python3
"""Plot C=C distance and HCCH dihedral vs time from traj_EVCont_*.npy files.

Assumed atom order:
    0:C, 1:C, 2:H, 3:H, 4:H, 5:H

Representative torsion tracked here:
    H(2)-C(0)-C(1)-H(5)
"""

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


TRAJ_PATTERN = re.compile(r"traj_EVCont_(\d+)\.npy$")
AU_TO_FS = 0.024188843265857
DT_AU = 10.0
TIME_PER_STEP_FS = DT_AU * AU_TO_FS


def load_trajectories(folder: Path):
    trajectories = []
    for path in sorted(folder.glob("traj_EVCont_*.npy")):
        match = TRAJ_PATTERN.match(path.name)
        if match is None:
            continue
        ntrain = int(match.group(1))
        traj = np.load(path)
        trajectories.append((ntrain, traj))
    return sorted(trajectories, key=lambda x: x[0])


def cc_distance(traj):
    c1 = traj[:, 0, :]
    c2 = traj[:, 1, :]
    return np.linalg.norm(c1 - c2, axis=1)


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


def hcch_dihedral(traj, h_left=2, c_left=0, c_right=1, h_right=4):
    vals = np.zeros(traj.shape[0], dtype=float)
    for i in range(traj.shape[0]):
        p0 = traj[i, h_left]
        p1 = traj[i, c_left]
        p2 = traj[i, c_right]
        p3 = traj[i, h_right]
        vals[i] = dihedral_angle(p0, p1, p2, p3)
    return vals


def main():
    folder = Path(__file__).resolve().parent
    trajectories = load_trajectories(folder)
    if not trajectories:
        raise SystemExit(f"No traj_EVCont_N.npy files found in {folder}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    cmap = cm.plasma

    for idx, (ntrain, traj) in enumerate(trajectories):
        color = cmap(idx / max(1, len(trajectories) - 1))
        time_fs = np.arange(traj.shape[0]) * TIME_PER_STEP_FS

        d_cc = cc_distance(traj)
        phi = -hcch_dihedral(traj, h_left=2, c_left=0, c_right=1, h_right=5)

        ax1.plot(time_fs, d_cc, lw=2.0, color=color, label=f"N={ntrain}")
        ax2.plot(time_fs, phi, lw=2.0, color=color, label=f"N={ntrain}")

    ax1.set_ylabel("C=C distance (Bohr)")
    ax1.set_title("Ethene torsion run: C=C distance and HCCH dihedral")
    ax1.grid(True, alpha=0.25)

    ax2.set_xlabel("Time (fs)")
    ax2.set_ylabel("HCCH dihedral (deg)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(frameon=False, title="Training N")

    fig.tight_layout()

    out_png = folder / "cc_and_dihedral_vs_time.png"
    out_pdf = folder / "cc_and_dihedral_vs_time.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)

    print(f"Wrote {out_png}")
    print(f"Wrote {out_pdf}")


if __name__ == "__main__":
    main()
