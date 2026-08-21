#!/usr/bin/env python3
"""Quick active-learning MD example using FCI eigenvector continuation."""

from pyscf import fci, gto

from evcont.dynamics.MD_utils import converge_EVCont_MD
from evcont.fci.FCI_EVCont import FCI_EVCont_obj


def build_h2(distance_bohr):
    return gto.M(
        atom=[("H", (0.0, 0.0, -distance_bohr / 2)),
              ("H", (0.0, 0.0, distance_bohr / 2))],
        basis="sto-3g",
        unit="Bohr",
        symmetry=False,
        verbose=0,
    )


# A loose FCI tolerance and active-learning threshold keep this example quick.
cisolver = fci.direct_spin0.FCI()
cisolver.conv_tol = 1.0e-6
cont = FCI_EVCont_obj(cisolver=cisolver, nroots=1)

converge_EVCont_MD(
    cont,
    build_h2(1.6),
    steps=20,
    dt=2.0,
    convergence_thresh=1.0e-2,
    prune_irrelevant_data=False,
    data_addition="farthest_point",
)

print(f"Active-learning MD converged with {len(cont.fcivecs)} training states.")
