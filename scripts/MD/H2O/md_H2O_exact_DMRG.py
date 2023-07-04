from pyscf import gto, md

import numpy as np

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad

from EVCont.CustomDMRGCI import CustomDMRGCI


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", geometry[0]), ("H", geometry[1]), ("O", geometry[2])],
        basis="cc-pVDZ",
        symmetry=True,
        unit="Bohr",
    )

    return mol


# init_geometry = np.array(
#     [[0.0, 0.795, -0.454], [0.0, -0.795, -0.454], [0.0, 0.0, 0.113]]
# )
init_geometry = np.array(
    [
        [0.0, 0.795 * 1.1, -0.454 * 1.1],
        [0.0, -0.795 * 1.1, -0.454 * 1.1],
        [0.0, 0.0, 0.113],
    ]
)


mol = get_mol(init_geometry)


init_mol = mol.copy()


steps = 50
dt = 2

solver = CustomDMRGCI(mol, mol.nao, mol.nelec, "split")
solver.converged = True

scanner_fun = solver.nuc_grad_method().as_scanner()

frames = []
scanner_fun.mol = init_mol.copy()
myintegrator = md.NVE(
    scanner_fun,
    dt=dt,
    steps=steps,
    incore_anyway=True,
    frames=frames,
    trajectory_output="exact_trajectory.xyz",
)
myintegrator.run()


np.save("traj_exact.npy", np.array([frame.coord for frame in frames]))
