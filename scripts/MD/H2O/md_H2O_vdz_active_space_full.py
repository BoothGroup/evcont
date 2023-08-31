from pyscf import md, gto, scf, mcscf, lo, ao2mo, fci, lib, grad

import numpy as np


ncas = 8
neleca = 4


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


steps = 300
dt = 5

mf = scf.RHF(init_mol.copy())
scanner_fun = mcscf.CASCI(mf, ncas, neleca).nuc_grad_method().as_scanner()
frames = []
scanner_fun.mol = init_mol.copy()
myintegrator = md.NVE(
    scanner_fun,
    steps=steps,
    dt=dt,
    incore_anyway=True,
    frames=frames,
    trajectory_output="active_space_trajectory_vdz.xyz",
    energy_output="active_space_energy_vdz.xyz",
)
myintegrator.run()


np.save("active_space_trajectory_vdz.npy", np.array([frame.coord for frame in frames]))
