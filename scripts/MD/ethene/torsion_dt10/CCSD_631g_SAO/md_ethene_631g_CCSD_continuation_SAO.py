from pyscf import gto

import os
import numpy as np

from evcont.MD_utils import converge_EVCont_MD
from evcont.CCSD_EVCont import CCSD_EVCont_obj


"""
Runs an MD simulation for an ethene molecule in the 6-31g basis with
CCSD continuation in the SAO representation.
"""


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[
            ("C", geometry[0]),
            ("C", geometry[1]),
            ("H", geometry[2]),
            ("H", geometry[3]),
            ("H", geometry[4]),
            ("H", geometry[5]),
        ],
        basis="6-31g",
        symmetry=False,
        unit="Bohr",
    )

    return mol


def rotate_points_about_axis(points, axis_point1, axis_point2, angle_rad):
    """Rotate points around the axis defined by two points using Rodrigues' formula."""
    axis = axis_point2 - axis_point1
    axis = axis / np.linalg.norm(axis)

    shifted = points - axis_point1
    cos_t = np.cos(angle_rad)
    sin_t = np.sin(angle_rad)

    rotated = (
        shifted * cos_t
        + np.cross(axis, shifted) * sin_t
        + np.outer(np.dot(shifted, axis), axis) * (1.0 - cos_t)
    )
    return rotated + axis_point1


a_to_bohr = 1.8897259886
stretch_factor = 1.5
torsion_angle_deg = 45.0

# Planar ethene (approximate): C=C along x, molecule in xy plane.
# The initial condition stretches only the C=C bond.
c_c_eq_ang = 1.339
c_h_ang = 1.086
hch_half_angle_deg = 58.0
hch_half_angle = np.deg2rad(hch_half_angle_deg)

c_c = stretch_factor * c_c_eq_ang
half_cc = 0.5 * c_c

hx = c_h_ang * np.cos(hch_half_angle)
hy = c_h_ang * np.sin(hch_half_angle)

init_geometry = np.array(
    [
        [-half_cc, 0.0, 0.0],
        [half_cc, 0.0, 0.0],
        [-half_cc - hx, hy, 0.0],
        [-half_cc - hx, -hy, 0.0],
        [half_cc + hx, hy, 0.0],
        [half_cc + hx, -hy, 0.0],
    ]
)

# Apply torsion around C=C (x-axis here). Rotate left/right CH2 groups oppositely.
torsion_angle_rad = np.deg2rad(torsion_angle_deg)
axis_p1 = init_geometry[0]
axis_p2 = init_geometry[1]

init_geometry[[2, 3]] = rotate_points_about_axis(
    init_geometry[[2, 3]], axis_p1, axis_p2, 0.5 * torsion_angle_rad
)
init_geometry[[4, 5]] = rotate_points_about_axis(
    init_geometry[[4, 5]], axis_p1, axis_p2, -0.5 * torsion_angle_rad
)

init_geometry = a_to_bohr * init_geometry

mol = get_mol(init_geometry)
init_mol = mol.copy()

# Keep MD setup identical to the existing water continuation scripts.
steps = 300
dt = 10

# Reference/computational geometry for transferable basis construction.
comp_mol = init_mol.copy()

# Build CCSD continuation object for ground-state MD.
ccsd_cont = CCSD_EVCont_obj(
    comp_mol=comp_mol,
    nroots=1,
    abstract_basis="SAO",
)

if os.environ.get("EVCONT_MD_DRY_RUN") == "1":
    print(f"Dry run OK: basis=6-31g, abstract_basis=SAO, nao={mol.nao}")
else:
    converge_EVCont_MD(
        ccsd_cont,
        init_mol,
        steps=steps,
        dt=dt,
        prune_irrelevant_data=False,
        data_addition="farthest_point_ham",
    )
