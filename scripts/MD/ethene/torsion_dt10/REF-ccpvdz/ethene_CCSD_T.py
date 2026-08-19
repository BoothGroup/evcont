from pyscf import gto, scf, md, cc

import numpy as np

from pyscf.grad import ccsd_t as ccsd_t_grad
from pyscf.geomopt.addons import as_pyscf_method

"""
Single MD trajectory simulation with CCSD for stretched ethene.
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
        basis="6-31G",
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

def ccsd_t_scan(mol_target):
    """Custom function that computes CCSD(T) energy and analytic gradients."""
    # 1. Run mean field and CCSD
    myhf = mol_target.RHF().run()
    mycc = myhf.CCSD().run()
    
    # 2. Get the CCSD(T) energy correction
    e_t = mycc.ccsd_t()
    e_tot = mycc.e_tot + e_t
    
    # 3. Solve standard CCSD lambda equations (needed by the gradient kernel)
    mycc.solve_lambda()
    
    # 4. Compute analytic nuclear gradients (handles the (T) response internally)
    g_cc = ccsd_t_grad.Gradients(mycc)
    de = g_cc.kernel()
    
    return e_tot, de

# Generate a geometry scanner object
scanner_fun = as_pyscf_method(init_mol, ccsd_t_scan)

steps = 300
dt = 10

frames = []
scanner_fun.mol = init_mol.copy()
myintegrator = md.NVE(
    scanner_fun,
    dt=dt,
    steps=steps,
    incore_anyway=True,
    frames=frames,
    trajectory_output="trajectory_CCSD(T).xyz",
    data_output="energy_CCSD(T).xyz",
    verbose=0,
)
myintegrator.run()

trajectory = np.array([frame.coord for frame in frames])
print(trajectory)
np.save("trajectory_CCSD(T).npy", trajectory)
