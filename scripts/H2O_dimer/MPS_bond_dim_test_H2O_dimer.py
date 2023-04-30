import numpy as np

from pyscf import scf, gto, ao2mo, fci, lo

from scipy.linalg import eig

import os

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

equilibrium_config = [("O", (0.0000000, 0.0000000, 0.0000000))]
equilibrium_config += [("H", (-0.6237519, -0.9109667, -1.4354514))]
equilibrium_config += [("H", (-0.6237519, -0.9109667, 1.4354514))]
equilibrium_config += [("O", (5.5028821, 0.0000000, 0.0000000))]
equilibrium_config += [("H", (3.6897611, 0.1745837, 0.0000000))]
equilibrium_config += [("H", (6.1311264, 1.6956360, 0.0000000))]

mol = gto.Mole()
mol.build(
    atom = equilibrium_config,
    basis = 'cc-pVDZ',
    symmetry = True,
    unit="Bohr"
)


loc_coeff = lo.orth_ao(mol, 'lowdin', pre_orth_ao=None)

norb = loc_coeff.shape[0]

h1 = np.linalg.multi_dot((loc_coeff.T, scf.hf.get_hcore(mol), loc_coeff))
h2 = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)

for i, M_max in enumerate([50, 100, 150, 200, 250, 300, 350, 400, 450, 500]):
    mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2)
    mps_solver.initialize_system(norb, n_elec=np.sum(mol.nelec))

    mpo = mps_solver.get_qc_mpo(h1e=h1, g2e=h2, ecore=mol.energy_nuc(), iprint=1)

    ket = mps_solver.get_random_mps(tag="GS_MPS_{}".format(i), bond_dim=M_max, nroots=1)


    en = mps_solver.dmrg(mpo, ket, n_sweeps=50, iprint=1)

    with open("DMRG_energy.txt", "a+") as fl:
        fl.write("{}  {}\n".format(M_max, en))
