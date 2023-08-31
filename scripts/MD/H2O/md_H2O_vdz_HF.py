from pyscf import gto, md

import numpy as np

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", geometry[0]), ("H", geometry[1]), ("O", geometry[2])],
        basis="cc-pVDZ",
        symmetry=False,
        unit="Bohr",
    )

    return mol


a_to_bohr = 1.8897259886

stretch_factor = 1.2

init_geometry = (
    a_to_bohr
    * stretch_factor
    * np.array(
        [
            [0.0, 0.795, -0.454],
            [0.0, -0.795, -0.454],
            [0.0, 0.0, 0.113],
        ]
    )
)

mol = get_mol(init_geometry)


init_mol = mol.copy()

mf = init_mol.RHF()


steps = 300
dt = 5


scanner_fun = mf.nuc_grad_method().as_scanner()
frames = []
scanner_fun.mol = init_mol.copy()
myintegrator = md.NVE(
    scanner_fun,
    steps=steps,
    dt=dt,
    incore_anyway=True,
    frames=frames,
    veloc=None,
    trajectory_output="HF_trajectory_vtz.xyz",
    energy_output="HF_energy_vtz.xyz",
)
myintegrator.run()


np.save("traj_HF_vtz.npy", np.array([frame.coord for frame in frames]))
