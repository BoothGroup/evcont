#!/usr/bin/env python3
"""Write AO-basis CCSD 1RDMs and subspace coefficients during water MD."""

import math

import numpy as np
from pyscf import gto

from evcont.ccsd.CCSD_EVCont import CCSD_EVCont_obj
from evcont.dynamics.MD_utils import converge_EVCont_MD


def build_water(oh_distance_angstrom=0.958, angle_degrees=104.5):
    half_angle = math.radians(angle_degrees / 2)
    x = oh_distance_angstrom * math.sin(half_angle)
    z = oh_distance_angstrom * math.cos(half_angle)
    return gto.M(
        atom=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (x, 0.0, z)),
            ("H", (-x, 0.0, z)),
        ],
        basis="sto-3g",
        unit="Angstrom",
        symmetry=False,
        verbose=0,
    )


# Start about 20% beyond equilibrium so the restoring force launches a
# symmetric O-H vibration.
initial_mol = build_water(oh_distance_angstrom=1.15)
cont = CCSD_EVCont_obj(comp_mol=initial_mol, nroots=1)

converge_EVCont_MD(
    cont,
    initial_mol,
    steps=20,
    dt=2.0,
    convergence_thresh=1.0e-2,
    prune_irrelevant_data=False,
    data_addition="weighted_highest_peak_ham",
    learning_exponent=0.5,
    return_rdms=("1rdm",),
    save_coefficients=True,
)

with np.load("rdms_EVCont_0.npz") as rdms:
    one_rdms_ao = rdms["one"]
coefficients = np.load("coefficients_EVCont_0.npy")

print(f"Active-learning MD converged with {len(cont.states)} training states.")
print("AO 1RDM trajectory:", one_rdms_ao.shape)
print("Subspace coefficient trajectory:", coefficients.shape)
