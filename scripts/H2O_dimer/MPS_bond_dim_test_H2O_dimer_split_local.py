import numpy as np

from pyscf import scf, gto, ao2mo, fci, lo

from scipy.linalg import eig

import os

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from mpi4py import MPI

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

myhf = scf.RHF(mol)
ehf = myhf.scf()

mol.nelec


norb = myhf.mo_coeff.shape[1]
print('Number of molecular orbitals: ', norb)

loc_coeff = myhf.mo_coeff

localizer = lo.Boys(mol, myhf.mo_coeff[:,:mol.nelec[0]])
loc_coeff_occ = localizer.kernel()
localizer = lo.Boys(mol, myhf.mo_coeff[:, mol.nelec[0]:])
loc_coeff_vrt = localizer.kernel()
loc_coeff = np.concatenate((loc_coeff_occ, loc_coeff_vrt), axis=1)

h1 = np.zeros((norb, norb))
h2 = np.zeros((norb, norb, norb, norb))


h1 = np.linalg.multi_dot((loc_coeff.T, scf.hf.get_hcore(mol), loc_coeff))
h2 = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)


MPI.COMM_WORLD.Bcast(h1, root=0)

h2_slice = np.empty((h2.shape[2],h2.shape[3]))

for i in range(h2.shape[0]):
    for j in range(h2.shape[1]):
        np.copyto(h2_slice, h2[i,j,:,:])
        MPI.COMM_WORLD.Bcast(h2_slice, root = 0)
        np.copyto(h2[i,j,:,:], h2_slice)



mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2, mpi=True)
mps_solver.initialize_system(norb, n_elec=np.sum(mol.nelec))

mpo = mps_solver.get_qc_mpo(h1e=h1, g2e=h2, ecore=mol.energy_nuc(), iprint=1)

bnd_dim = 25

ket = mps_solver.get_random_mps("GS_approx", bond_dim=bnd_dim, nroots=1)

converged = False

energies = []

open("DMRG_result.txt", "w").close()

rank = MPI.COMM_WORLD.Get_rank()

while not converged:
    mps_solver.dmrg(mpo, ket, bond_dims=[bnd_dim], n_sweeps=1000, iprint=1, noises =  [1e-4] * 4 + [1e-5] * 4 + [0], tol=1.e-3)
    ket.info.save_data("./MPS_{}.bin".format(bnd_dim))
    bnd_dms, _, ens = mps_solver.get_dmrg_results()
    energies.append(ens[-1][0])
    bnd_dim += 25
    if len(energies) > 1:
        if np.isclose(energies[-1], energies[-2]):
            converged = True
    if rank == 0:
        print(bnd_dms[-1], energies[-1], len(ens))
        with open("DMRG_result.txt", "a") as fl:
            fl.write("{}  {}  {}\n".format(bnd_dms[-1], energies[-1], len(ens)))

