from pyscf import md, gto, mcscf, fci

import numpy as np

"""
Runs a reference MD simulation for a water molecule in the 6-31G basis with FCI.
"""


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", geometry[0]), ("H", geometry[1]), ("O", geometry[2])],
        basis="6-31G",
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
casci_object = mcscf.CASCI(mf, mol.nao, np.sum(mol.nelec))
casci_object.fcisolver = fci.direct_spin0.FCISolver()

casci_object.fcisolver.max_cycle = 1000

scanner_fun = casci_object.nuc_grad_method().as_scanner()
frames = []
scanner_fun.mol = init_mol.copy()
myintegrator = md.NVE(
    scanner_fun,
    steps=steps,
    incore_anyway=True,
    frames=frames,
    trajectory_output="FCI_trajectory.xyz",
    energy_output="FCI_energy.xyz",
    dt=dt,
)
myintegrator.run()


np.save("traj_FCI.npy", np.array([frame.coord for frame in frames]))
