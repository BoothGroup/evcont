import numpy as np

from scipy.linalg import pinvh
from pyscf import scf, gto, ao2mo, fci, lo, cc

from dscribe.descriptors import SOAP
from ase import Atoms

from dscribe.kernels import AverageKernel

from sklearn import linear_model

import sys

box_edge = float(sys.argv[1])

n_data_points = 1000
seed = 1
number_atoms = 10

norb = nelec = number_atoms

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

    loc_coeff = lo.orth_ao(mol, 'meta_lowdin')

    ovlp = myhf.get_ovlp()
    # Check that we still have an orthonormal basis, i.e. C^T S C should be the identity
    assert(np.allclose(np.linalg.multi_dot((loc_coeff.T, ovlp, loc_coeff)),np.eye(norb)))
    # Find the hamiltonian in the local basis
    h1 = np.linalg.multi_dot((loc_coeff.T, myhf.get_hcore(), loc_coeff)).copy()
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, loc_coeff), norb).copy()
    return h1, h2, mol.energy_nuc(), ehf

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

energies = []
features = []

soap = SOAP(species=["H"], periodic=False, r_cut=3.0, n_max=3, l_max=3)

# Generate fci states
for i, dist in enumerate(trainig_dists):
    positions = [(x, 0., 0.) for x in dist*np.arange(number_atoms)]
    h1, h2, nuc_en, _ = get_ham(positions)
    cisolver = fci.direct_spin0.FCI()
    e, _ = cisolver.kernel(h1, h2, norb, (nelec//2, nelec//2))
    energies.append(e+nuc_en)
    h_chain = Atoms("H"*number_atoms, [(x, 0., 0.) for x in dist*np.arange(number_atoms)])
    features.append(soap.create(h_chain))

energies = np.array(energies)


"""
Create global SOAP average kernel
"""

avg_kernel = AverageKernel(metric="linear")

kernel_mat = avg_kernel.create(features)

np.save("SOAP_kernel.npy", kernel_mat)

# Fitting + hyperparameter optimization
fitted_model = linear_model.BayesianRidge()
fitted_model.fit(kernel_mat, energies)

rng = np.random.default_rng(seed)

open("ev_cont_data_soap_{}.txt".format(box_edge), "w").close()

for i in range(n_data_points):
    # Sample position
    shifts = (rng.random(size=(10, 3)) - 0.5) * 2 * box_edge
    sampled_pos = equilibrium_pos + shifts
    h1, h2, nuc_en, ehf = get_ham(sampled_pos)
    cisolver = fci.direct_spin0.FCI()
    en_exact, _ = cisolver.kernel(h1, h2, norb, (nelec//2, nelec//2))
    h_chain = Atoms("H"*number_atoms, [pos for pos in sampled_pos])
    new_feat = avg_kernel.create([soap.create(h_chain)], features)
    with open("ev_cont_data_soap_{}.txt".format(box_edge), "a") as fl:
        fl.write("{}  {}  {}\n".format(en_exact+nuc_en, fitted_model.predict(new_feat)[0], ehf))




