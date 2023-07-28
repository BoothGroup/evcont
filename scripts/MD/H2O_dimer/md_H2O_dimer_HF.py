from pyscf import gto, md

import numpy as np

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[
            ("O", geometry[0]),
            ("H", geometry[1]),
            ("H", geometry[2]),
            ("O", geometry[3]),
            ("H", geometry[4]),
            ("H", geometry[5]),
        ],
        basis="6-31G",
        symmetry=False,
        unit="Bohr",
    )

    return mol


stretch_factor = 1.2
init_geometry = stretch_factor * np.array(
    [
        [0.0000000, 0.0000000, 0.0000000],
        [-0.6237519, -0.9109667, -1.4354514],
        [-0.6237519, -0.9109667, 1.4354514],
        [5.5028821, 0.0000000, 0.0000000],
        [3.6897611, 0.1745837, 0.0000000],
        [6.1311264, 1.6956360, 0.0000000],
    ]
)

mol = get_mol(init_geometry)


init_mol = mol.copy()

mf = init_mol.RHF()

steps = 100
dt = 20

scanner_fun = mf.nuc_grad_method().as_scanner()
frames = []
scanner_fun.mol = init_mol.copy()
myintegrator = md.NVE(
    scanner_fun,
    dt=dt,
    steps=steps,
    incore_anyway=True,
    frames=frames,
    veloc=None,
    trajectory_output="HF_trajectory.xyz",
)
myintegrator.run()


np.save("traj_HF.npy", np.array([frame.coord for frame in frames]))
