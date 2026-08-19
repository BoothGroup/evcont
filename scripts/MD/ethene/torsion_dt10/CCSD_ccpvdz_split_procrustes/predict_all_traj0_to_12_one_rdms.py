#!/usr/bin/env python3
"""Run AO-basis 1RDM prediction for traj_EVCont_0.npy through traj_EVCont_12.npy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PREDICT_SCRIPT = SCRIPT_DIR / "predict_traj13_one_rdms.py"


def main() -> None:
    for traj_index in range(13):
        traj_file = SCRIPT_DIR / f"traj_EVCont_{traj_index}.npy"
        if not traj_file.exists():
            raise FileNotFoundError(traj_file)

        print(f"Computing predicted AO 1RDMs for {traj_file.name}")
        subprocess.run(
            [sys.executable, str(PREDICT_SCRIPT), str(traj_file)],
            check=True,
        )


if __name__ == "__main__":
    main()
