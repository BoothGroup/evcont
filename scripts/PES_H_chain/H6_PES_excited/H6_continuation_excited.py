from pyscf import gto, fci

import sys

from evcont.electron_integral_utils import get_basis, get_integrals

from evcont.FCI_EVCont import FCI_EVCont_obj

from evcont.ab_initio_eigenvector_continuation import approximate_multistate_OAO

import numpy as np

"""
Prediction of the ground and excited state PES for a 6-atom H chain from 
different GS and excited eigenstates
"""

n_atoms = 6

nroots_fci = 20 # For the exact surface

# States to train and predict
nroots_l = [3,3,6]
roots_train_l = [[0],[1],[0,1]]

# Tets and training distances
test_range = np.linspace(0.8, 3.0)
train_dists = [1.0, 1.8, 2.6]

roots_train_str_l = [''.join([str(i) for i in ri]) for ri in roots_train_l]
ntrain = len(train_dists)

def get_mol(dist):
    mol = gto.Mole()

    mol.build(
        atom=[
            ("H", (x, 0.0, 0.0))
            for x in (np.arange(n_atoms) - np.median(np.arange(n_atoms))) * dist
        ],
        basis="sto-6g",
        symmetry=False,
        unit="Bohr",
    )

    return mol

# Iterate over different EV sets
for ri, roots_train in enumerate(roots_train_l):
    
    nroots = nroots_l[ri]
    roots_train_str = roots_train_str_l[ri]
    nroots_t = len(roots_train)
    
    # Initialize the continuation object
    continuation_object = FCI_EVCont_obj(roots_train=roots_train,
                                         cibasis='canonical')

    for i, trn_dist in enumerate(train_dists):
        mol = get_mol(trn_dist)
    
        continuation_object.append_to_rdms(mol)
    
        np.savetxt(
            "GS_dist_{}_rootstrain_{}.txt".format(trn_dist,roots_train_str), continuation_object.fcivecs[-1].flatten()
        )
    
        np.savetxt(
            "en_dist_{}_rootstrain_{}.txt".format(trn_dist,roots_train_str), np.atleast_1d(continuation_object.ens[-nroots_t:])
        )


    if (i+1)*nroots_t < nroots:
        print(f'Incompatible nroots_t {nroots_t} and nroots {nroots} - PES not predicted')
        #sys.exit()
    else:
        open("predicted_surface_rootstrain_{}_datapoints.txt".format(roots_train_str), "w").close()
        for test_dist in test_range:
            mol = get_mol(test_dist)
            en, _ = approximate_multistate_OAO(
                mol,
                continuation_object.one_rdm,
                continuation_object.two_rdm,
                continuation_object.overlap,
                nroots=nroots,
            )
            with open("predicted_surface_rootstrain_{}_datapoints.txt".format(roots_train_str), "a") as fl:
                fl.write(f"{test_dist:.8f}  {'  '.join(['%.8f'%i for i in en])}\n")
    
    
        test_dist = 2.2
        mol = get_mol(test_dist)
        _, c = approximate_multistate_OAO(
            mol,
            continuation_object.one_rdm,
            continuation_object.two_rdm,
            continuation_object.overlap,
            nroots=nroots
        )
        np.savetxt("continuation_gs_rootstrain_{}_datapoints.txt".format(roots_train_str), c)

open("exact_surface.txt", "w").close()
for i, test_dist in enumerate(test_range):
    mol = get_mol(test_dist)
    h1, h2 = get_integrals(mol, get_basis(mol))
    e_all, fcivec_all = fci.direct_spin0.FCI().kernel(h1, h2, mol.nao, mol.nelec,nroots=nroots_fci)
    
    e_all += mol.energy_nuc()
    
    with open("exact_surface.txt", "a") as fl:
        fl.write(f"{test_dist:.8f}  {'  '.join(['%.8f'%i for i in e_all])}\n")



