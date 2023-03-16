import numpy as np

from scipy.linalg import eigh

from pyscf import scf, gto, ao2mo, fci, lo, cc

import sys

box_edge = float(sys.argv[1])
basis = int(sys.argv[2]) # 0: local, 1: canonical, 2: split
number_atoms = int(sys.argv[3])
n_data_points = int(sys.argv[4])
seed = int(sys.argv[5])
load_path = sys.argv[6]

norb = nelec = number_atoms

rng = np.random.default_rng(seed)

if basis == 0:
    basis_str = "local"
elif basis == 1:
    basis_str = "canonical"
elif basis == 2:
    basis_str = "split"
else:
    assert(False)


def get_ham(positions):
    mol = gto.Mole()

    mol.build(
        atom = [('H', pos) for pos in positions],
        basis = 'sto-6g',
        symmetry = True,
        unit="Bohr"
    )

    myhf = scf.RHF(mol)
    ehf = myhf.scf()

    # Get hamiltonian elements
    # 1-electron 'core' hamiltonian terms, transformed into MO basis
    h1 = np.linalg.multi_dot((myhf.mo_coeff.T, myhf.get_hcore(), myhf.mo_coeff))

    # Get 2-electron electron repulsion integrals, transformed into MO basis
    eri = ao2mo.incore.general(myhf._eri, (myhf.mo_coeff,)*4, compact=False)

    # Previous representation exploited permutational symmetry in storage. Change this to a 4D array.
    # Integrals now stored as h2[p,q,r,s] = (pq|rs) = <pr|qs>. Note 8-fold permutational symmetry.
    h2 = ao2mo.restore(1, eri, myhf.mo_coeff.shape[1])

    if basis == 0:
        loc_coeff = lo.orth_ao(mol, 'meta_lowdin')
    elif basis == 1:
        loc_coeff = myhf.mo_coeff
    elif basis == 2:
        localizer = lo.Boys(mol, myhf.mo_coeff[:,:nelec//2])
        loc_coeff_occ = localizer.kernel()
        localizer = lo.Boys(mol, myhf.mo_coeff[:, nelec//2:])
        loc_coeff_vrt = localizer.kernel()
        loc_coeff = np.concatenate((loc_coeff_occ, loc_coeff_vrt), axis=1)
    else:
        assert(False)

    ovlp = myhf.get_ovlp()
    # Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
    assert(np.allclose(np.linalg.multi_dot((loc_coeff.T, ovlp, loc_coeff)),np.eye(norb)))
    # Find the hamiltonian in the local basis
    h1 = np.linalg.multi_dot((loc_coeff.T, myhf.get_hcore(), loc_coeff)).copy()
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb).copy()
    return h1, h2, mol.energy_nuc(), ehf

if load_path == "":
    if number_atoms == 10:
        equilibrium_dist = 1.78596
    elif number_atoms == 8:
        equilibrium_dist = 1.76960
    elif number_atoms == 12:
        equilibrium_dist = 1.79612
    else:
        # Find equilibrium dist
        left = 1.4
        right = 1.9

        for j in range(30):
            dist_range = np.linspace(left, right, num=5, endpoint=True)
            ens = []
            for dist in dist_range:
                cisolver = fci.direct_spin0.FCI()
                h1, h2, nuc_en, ehf = get_ham([(x*dist, 0., 0.) for x in range(number_atoms)])
                e, fcivec = cisolver.kernel(h1, h2, norb, (nelec//2, nelec//2))
                ens.append(e+nuc_en)
            arg_min = np.argmin(np.array(ens))
            left = dist_range[arg_min-1]
            right = dist_range[arg_min+1]
            equilibrium_dist = dist_range[arg_min]
            print(equilibrium_dist)

    equilibrium_pos = np.array([(x*equilibrium_dist, 0., 0.) for x in range(number_atoms)])

    training_stretches = np.array([0., -0.25, 0.25, 0.5, -0.5, 1.0, -1.0])

    trainig_dists = equilibrium_dist + training_stretches

    # Generate fci states
    for i, dist in enumerate(trainig_dists):
        positions = [(x, 0., 0.) for x in dist*np.arange(number_atoms)]
        h1, h2, _, _ = get_ham(positions)
        cisolver = fci.direct_spin0.FCI()
        e, fcivec = cisolver.kernel(h1, h2, norb, (nelec//2, nelec//2))
        np.save("fci_vec_{}.npy".format(i), fcivec)


    """
    Create overlap matrix, 1RDM, and 2RDM matrices
    """

    S = np.zeros((len(trainig_dists), len(trainig_dists)))
    one_RDM = np.zeros((len(trainig_dists), len(trainig_dists), norb, norb))
    two_RDM = np.zeros((len(trainig_dists), len(trainig_dists), norb, norb, norb, norb))

    for (i, dist_a) in enumerate(trainig_dists):
        vec_a = np.load("fci_vec_{}.npy".format(i))
        for (j, dist_b) in enumerate(trainig_dists):
            vec_b = np.load("fci_vec_{}.npy".format(j))
            S[i,j] = vec_a.flatten().dot(vec_b.flatten())
            cisolver = fci.direct_spin0.FCI()
            rdm1, rdm2 = cisolver.trans_rdm12(vec_a, vec_b, norb, (nelec//2, nelec//2))
            one_RDM[i, j, :, :] = rdm1
            two_RDM[i, j, :, :, :, :] = rdm2

    np.save("S.npy", S)
    np.save("one_RDM.npy", one_RDM)
    np.save("two_RDM.npy", two_RDM)
else:
    S = np.load(load_path + "S.npy")
    one_RDM = np.load(load_path + "one_RDM.npy")
    two_RDM = np.load(load_path + "two_RDM.npy")

open("ev_cont_data_{}_{}.txt".format(box_edge, basis_str), "w").close()

for i in range(n_data_points):
    # Sample position
    shifts = (rng.random(size=(number_atoms, 3)) - 0.5) * 2 * box_edge
    sampled_pos = equilibrium_pos + shifts
    h1, h2, nuc_en, ehf = get_ham(sampled_pos)
    H = np.sum(one_RDM * h1, axis=(-1,-2)) + 0.5 * np.sum(two_RDM * h2, axis=(-1,-2,-3,-4))
    vals, vecs = eigh(H, S)
    argmin = np.argmin(vals.real)
    en_approx = vals[argmin].real
    cisolver = fci.direct_spin0.FCI()
    en_exact, _ = cisolver.kernel(h1, h2, norb, (nelec//2, nelec//2))
    with open("ev_cont_data_{}_{}.txt".format(box_edge, basis_str), "a") as fl:
        fl.write("{}  {}  {}\n".format(en_exact+nuc_en, en_approx+nuc_en, ehf))




