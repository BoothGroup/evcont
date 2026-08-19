from pyscf import gto, md, dft

import numpy as np


"""
Single MD trajectory simulation with DFT (CAMB3LYP exchange correlation function)
for stretched ethene.
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
        basis="cc-pvdz",
        symmetry=False,
        unit="Angstrom",
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


stretch_factor = 1.5
torsion_angle_deg = 45.0

# Planar ethene (approximate): C=C along x, molecule in xy plane.
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

mol = get_mol(init_geometry)


init_mol = mol.copy()

mf = dft.RKS(init_mol, xc="CAMB3LYP")

mf.max_cycle = 100

steps = 300
dt = 10


scanner_fun = mf.nuc_grad_method().as_scanner()


frames = []
scanner_fun.mol = init_mol.copy()
myintegrator = md.NVE(
    scanner_fun,
    dt=dt,
    steps=steps,
    incore_anyway=True,
    frames=frames,
    trajectory_output="trajectory_CAMB3LYP.xyz",
    data_output="energy_CAMB3LYP.xyz",
    verbose=0,
)
myintegrator.run()

trajectory = np.array([frame.coord for frame in frames])
print(trajectory)
np.save("trajectory_CAMB3LYP.npy", trajectory)
