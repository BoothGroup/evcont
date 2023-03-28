import numpy as np

from pyscf import scf, gto, ao2mo, fci, lo

from scipy.linalg import eig

import os

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

import sys


box_edge = float(sys.argv[1])
n_data_points = int(sys.argv[2])
seed = int(sys.argv[3])


load_path = "/users/k1802890/data/AC/H30/H30_DMRG_training_data/"

rng = np.random.default_rng(seed)

norb = 30
nelec = 30

M_max = 200


def construct_ham(positions):
    mol = gto.Mole()

    mol.build(
        atom = [('H', pos) for pos in positions],
        basis = 'sto-6g',
        symmetry = True,
        unit="Bohr"
    )

    loc_coeff = lo.orth_ao(mol, 'meta_lowdin')

    h1 = np.linalg.multi_dot((loc_coeff.T, scf.hf.get_hcore(mol), loc_coeff))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb)

    return h1, h2, mol.nuc_en()


reference_pos = np.array([(x*2., 0., 0.) for x in range(nelec)])

S = np.load(load_path + "S.npy")
one_RDM = np.load(load_path + "one_RDM.npy")
two_RDM = np.load(load_path + "two_RDM.npy")

open("ev_cont_data_{}.txt".format(box_edge), "w").close()

for i in range(n_data_points):
    # Sample position
    shifts = (rng.random(size=(nelec, 3)) - 0.5) * 2 * box_edge
    sampled_pos = reference_pos + shifts
    mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2)
    mps_solver.initialize_system(norb, n_elec=nelec)
    h1, h2, nuc_en = construct_ham(sampled_pos)
    mpo = mps_solver.get_qc_mpo(h1e=h1, g2e=h2, iprint=1)
    ket = mps_solver.get_random_mps(tag="GS_MPS", bond_dim=M_max, nroots=1)
    en_MPS = mps_solver.dmrg(mpo, ket, n_sweeps=50)
    H = np.sum(one_RDM * h1, axis=(-1,-2)) + 0.5 * np.sum(two_RDM * h2, axis=(-1,-2,-3,-4))
    vals, vecs = eigh(H, S)
    argmin = np.argmin(vals.real)
    en_approx = vals[argmin].real
    with open("ev_cont_data_{}.txt".format(box_edge), "a") as fl:
        fl.write("{}  {}  {}\n".format(en_MPS+nuc_en, en_approx+nuc_en, ehf))









