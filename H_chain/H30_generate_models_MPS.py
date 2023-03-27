import numpy as np

from pyscf import scf, gto, ao2mo, fci, lo

from scipy.linalg import eig

import os

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

dist_list = [1.0, 1.4, 1.8, 2.0, 2.4, 2.8, 3.2]

norb = 30
nelec = 30

M_max = 200


fci_vecs = []

def construct_ham(dist):
    mol = gto.Mole()

    mol.build(
        atom = [('H', (x, 0., 0.)) for x in dist*np.arange(nelec)],
        basis = 'sto-6g',
        symmetry = True,
        unit="Bohr"
    )

    loc_coeff = lo.orth_ao(mol, 'meta_lowdin')

    h1 = np.linalg.multi_dot((loc_coeff.T, scf.hf.get_hcore(mol), loc_coeff))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)

    return h1, h2


energies = []
states = []

# Generate MPS states/run DMRG
for i, dist in enumerate(dist_list):
    mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2)
    mps_solver.initialize_system(norb, n_elec=nelec)
    h1, h2 = construct_ham(dist)
    mpo = mps_solver.get_qc_mpo(h1e=h1, g2e=h2, iprint=1)
    ket = mps_solver.get_random_mps(tag="GS_MPS_{}".format(i), bond_dim=M_max, nroots=1)
    energies.append(mps_solver.dmrg(mpo, ket, n_sweeps=50, iprint=1))
    states.append(ket)


"""
Create overlap matrix, 1RDM, and 2RDM matrices
"""

S = np.zeros((len(dist_list), len(dist_list)))
one_RDM = np.zeros((len(dist_list), len(dist_list), norb, norb))
two_RDM = np.zeros((len(dist_list), len(dist_list), norb, norb, norb, norb))

for (i, dist_a) in enumerate(dist_list):
    for (j, dist_b) in enumerate(dist_list):
        S[i,j] = mps_solver.expectation(states[i], mps_solver.get_identity_mpo(), states[j])
        one_RDM[i, j, :, :] = mps_solver.get_1pdm(states[j], bra=states[i])
        two_RDM[i, j, :, :, :, :] = np.transpose(mps_solver.get_2pdm(states[j], bra=states[i]), (0,3,1,2))

np.save("S.npy", S)
np.save("one_RDM.npy", one_RDM)
np.save("two_RDM.npy", two_RDM)




