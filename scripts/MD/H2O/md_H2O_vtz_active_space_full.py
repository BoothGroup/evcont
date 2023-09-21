from pyscf import md, gto, scf, mcscf, lo, ao2mo, fci, lib, grad

import numpy as np


ncas = 8
neleca = 4


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", geometry[0]), ("H", geometry[1]), ("O", geometry[2])],
        basis="cc-pVTZ",
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

mf = mol.RHF()

mf.max_cycle = 1000
mf.conv_tol = 1.0e-12


casci_object = mcscf.CASCI(mf, ncas, neleca)
casci_object.fcisolver = fci.direct_spin0.FCISolver()

casci_object.fcisolver.max_cycle = 1000
casci_object.fcisolver.conv_tol = 1.0e-12

scanner_fun = casci_object.nuc_grad_method().as_scanner()


frames = []

open("converged.txt", "w").close()


def write_converged(dict):
    with open("converged.txt", "a") as fl:
        fl.write(
            "{} {}\n".format(
                dict["scanner"].base._scf.converged,
                dict["scanner"].base.fcisolver.converged,
            )
        )


myintegrator = md.NVE(
    casci_object,
    steps=steps,
    dt=dt,
    incore_anyway=True,
    frames=frames,
    trajectory_output="active_space_trajectory_vtz.xyz",
    energy_output="active_space_energy_vtz.xyz",
    callback=write_converged,
)
myintegrator.run()


np.save("active_space_trajectory_vtz.npy", np.array([frame.coord for frame in frames]))
