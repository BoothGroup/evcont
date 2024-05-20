#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:02:20 2024

Script to test low rank construction of eigenvector continuation

@author: Kemal Atalar
"""

import numpy as np

from pyscf import gto, fci, scf, lib, ao2mo

from evcont.FCI_EVCont import FCI_EVCont_obj
from evcont.CASCI_EVCont import CAS_EVCont_obj

from evcont.electron_integral_utils import get_basis, get_integrals, get_loewdin_trafo

#from evcont.ab_initio_gradients_loewdin import get_multistate_energy_with_grad_and_NAC

from evcont.ab_initio_eigenvector_continuation import (
    approximate_multistate_lowrank_OAO,
    approximate_multistate_OAO,
    approximate_multistate
)

#from evcont.FCI_NAC import get_FCI_energy_with_grad_and_NAC_withsym
#from pyscf.mcscf import CASCI
#import pickle

import matplotlib.pylab as plt
#import matplotlib as mpl
plt.style.use('default')


nroots_evcont = 1
cibasis = 'canonical'
#cibasis = 'OAO'

df_basis = 'weigend'

natom = 6

solver = 'FCI'
cassolver='SS-CASSCF'
ncas, neleca = 2,2

#plot_extensive = False
fix_singlet = True
#withMolcas = False

fix_sym = 'A1g'
#fix_sym = None

if fix_sym == None:
    mol_sym = False
else:
    mol_sym = True
    
lowrank_kwargs = {'truncation_style':'nvec', 'nvecs':10}
lowrank_kwargs = {'truncation_style':'eigval', 'eval_thr':1e-8}
#lowrank_kwargs = {'truncation_style':'ham', 'ham_thr':0.002}
#lowrank_kwargs = {'truncation_style':'ham_en', 'ham_thr':0.0002}

test_range = np.linspace(0.8, 3.0,40)
#test_range = np.linspace(0.4, 1.5,20)

def get_mol(positions):
    mol = gto.Mole()

    mol.build(
        atom=[("H", pos) for pos in positions],
        #basis="sto-3g",
        basis="sto-6g",
        #basis="6-31g",
        symmetry=True,
        unit="Bohr",
        verbose=0
    )

    return mol

mol_dummy = get_mol([(x, 0.0, 0.0) for x in test_range[0] * np.arange(natom)])

# Set fci solver to be used
if fix_sym == None:
    myci = fci.direct_spin0.FCI()
else:
    myci = fci.direct_spin0_symm.FCI(mol_dummy)
    myci.wfnsym = fix_sym
    
#myci = fci.direct_spin0.FCI()
#myci = fci.direct_spin1.FCISolver()
    

if fix_singlet:
    fci.addons.fix_spin_(myci,ss=0) # Fix spin

equilibrium_dist = 1.78596

equilibrium_pos = np.array([(x * equilibrium_dist, 0.0, 0.0) for x in range(10)])

trainig_dists = [0.97, 1.76, 2.60]
trainig_dists = np.linspace(0.97,2.60,12)

if solver == 'FCI':
    continuation_object = FCI_EVCont_obj(nroots=nroots_evcont,
                                         cibasis=cibasis,cisolver=myci,
                                         irrep_name=fix_sym,
                                         lowrank=True,
                                         **lowrank_kwargs)
    
    continuation_object_full = FCI_EVCont_obj(nroots=nroots_evcont,
                                         cibasis=cibasis,cisolver=myci,
                                         irrep_name=fix_sym)

else:
    continuation_object = CAS_EVCont_obj(ncas, neleca,nroots=nroots_evcont,
                                         solver=cassolver,
                                         lowrank=True,
                                         **lowrank_kwargs)
    
    continuation_object_full = CAS_EVCont_obj(ncas, neleca,nroots=nroots_evcont,
                                         solver=cassolver)
 
trn_geometries = []
# Generate training data + prepare training models
for i, dist in enumerate(trainig_dists):
    positions = [(x, 0.0, 0.0) for x in dist * np.arange(natom)]
    # Add the geometry to set of training points
    trn_geometries.append(positions)
    
    # Build molecule    
    mol = get_mol(positions)
    continuation_object.append_to_rdms(mol)
    continuation_object_full.append_to_rdms(mol)

# Save
i = 'final'
np.save("overlap_{}.npy".format(i), continuation_object.overlap)
np.save("one_rdm_{}.npy".format(i), continuation_object.one_rdm)

np.save("cum_diagonal_{}.npy".format(i), continuation_object.cum_diagonal)
np.save("lowrank_vecs_{}.npy".format(i), continuation_object.vecs_lowrank)

np.save('trn_geometries_{}.npy'.format(i), trn_geometries)

train_lowrank_en = []
train_en = []
for i, test_dist in enumerate(trainig_dists):
    print(i)
    positions = [(x, 0.0, 0.0) for x in test_dist * np.arange(natom)]
    
    mol = get_mol(positions)
    h1, h2 = get_integrals(mol, get_basis(mol))
    
    # Continuation
    en_continuation_ms, vec = approximate_multistate_lowrank_OAO(
        mol, 
        continuation_object.one_rdm, 
        continuation_object.cum_diagonal, 
        continuation_object.vecs_lowrank, 
        continuation_object.overlap,
        nroots=nroots_evcont
    )
    
    train_lowrank_en += [en_continuation_ms]

    en_continuation_ms, vec = approximate_multistate_OAO(
        mol,
        continuation_object_full.one_rdm,
        continuation_object_full.two_rdm,
        continuation_object_full.overlap,
        nroots=nroots_evcont
    )
    
    train_en += [en_continuation_ms]
    
#1/0

# Prediction on test dataset and comparison against FCI results
fci_en = np.zeros([len(test_range),nroots_evcont])
hf_en = np.zeros([len(test_range)])
cont_en = np.zeros([len(test_range),nroots_evcont])
cont_lowrank_en = np.zeros([len(test_range),nroots_evcont])
for i, test_dist in enumerate(test_range):
    print(i)
    positions = [(x, 0.0, 0.0) for x in test_dist * np.arange(natom)]
    
    mol = get_mol(positions)
    h1, h2 = get_integrals(mol, get_basis(mol,'canonical'))
    
    print('   low rank')
    # Continuation
    en_continuation_ms, vec = approximate_multistate_lowrank_OAO(
        mol, 
        continuation_object.one_rdm, 
        continuation_object.cum_diagonal, 
        continuation_object.vecs_lowrank, 
        continuation_object.overlap,
        nroots=nroots_evcont,
        df_basis=df_basis
    )
    
    #cont_lowrank_en += [en_continuation_ms]
    cont_lowrank_en[i,:] = en_continuation_ms
    
    ## HF and FCI

    mf = scf.RHF(mol).density_fit(auxbasis=df_basis)
    # Note that in performant code, we don't actually need to run HF at the training points,
    # just have access to the get_jk function.
    ehf = mf.scf()
    hf_en[i] = ehf
    assert(mf.converged)

    # Do FCI for the exact energy
    h1_ao = mf.get_hcore()
    Lpq_ao = lib.unpack_tril(mf.with_df._cderi)
    Lpq_mo = lib.einsum('pi,qj,Lpq->Lij', mf.mo_coeff, mf.mo_coeff, Lpq_ao)
    df_eri = lib.einsum('Pij,Pkl->ijkl', Lpq_mo, Lpq_mo)
    h1e_mo = np.einsum('ai,ab,bj->ij', mf.mo_coeff, h1_ao, mf.mo_coeff)

    e_fci, c_fci = myci.kernel(h1e_mo, df_eri, mol.nao, mol.nelec, nroots=nroots_evcont,)
    e_fci += mol.energy_nuc()
    fci_en[i,:] = e_fci
    
    
    # Full continuation
    print('   full')
    # Find h1 and eris in SAO basis
    sao_basis = get_loewdin_trafo(mol.intor("int1e_ovlp"))
    h1e_sao = np.einsum('ai,ab,bj->ij', sao_basis, h1_ao, sao_basis)
    #Lpq_sao = lib.einsum('pi,qj,Lpq->Lij', sao_basis, sao_basis, Lpq_ao)
    Lpq_sao = ao2mo._ao2mo.nr_e2(mf.with_df._cderi, sao_basis,
        (0, sao_basis.shape[1], 0, sao_basis.shape[1]),aosym="s2",mosym="s2")
    Lpq_sao = lib.unpack_tril(Lpq_sao)
    df_eri_sao = lib.einsum('Pij,Pkl->ijkl', Lpq_sao, Lpq_sao)
    
    en_continuation_ms, vec = approximate_multistate(
        h1e_sao,
        df_eri_sao,
        continuation_object_full.one_rdm,
        continuation_object_full.two_rdm,
        continuation_object_full.overlap,
        nroots=nroots_evcont
    )
    
    cont_en[i,:] = en_continuation_ms + mol.energy_nuc()


# PLOT
fig, [ax1,ax2] = plt.subplots(nrows=2,sharex=True,figsize=[4,5],height_ratios=[3,2])

ax1.plot(test_range, hf_en,'orange',label='HF')
if nroots_evcont > 1:
    ax1.plot(test_range,fci_en,'k',label=['FCI']+[None]*(nroots_evcont-1))
    ax1.plot(test_range,cont_en,'b',label=['full evcont']+[None]*(nroots_evcont-1))
    ax1.plot(test_range,cont_lowrank_en,'--r',label=['low rank evcont']+[None]*(nroots_evcont-1))
else:
    ax1.plot(test_range,fci_en,'k',label='FCI')
    ax1.plot(test_range,cont_en,'b',label='full evcont')
    ax1.plot(test_range,cont_lowrank_en,'--r',label='low rank evcont')
   

ax1.plot(trainig_dists,train_en,'xb')
ax1.plot(trainig_dists,train_lowrank_en,'xr')
ax1.legend()

ax2.plot(test_range,cont_en - fci_en,'b')
ax2.plot(test_range,cont_lowrank_en - fci_en,'--r')

ax1.set_ylabel('Energy (Ha)')
ax2.set_ylabel(r'$E_{cont}$ - $E_{FCI}$ (Ha)')
ax2.set_xlabel('Atomic separation ($a_0$)')
plt.show()

# Expansion limit
no_vec_dic = {}
for key, item in continuation_object.vecs_lowrank.items():
    #print(key, item[0].shape)
    if key[1] >= key[0]:
        no_vec_dic[','.join([str(i) for i in key])] = item[0].shape[0]

fig, ax = plt.subplots(figsize=[4,7])
ax.grid(alpha=0.5)

#D = {u'Label1':26, u'Label2': 17, u'Label3':30}
ax.barh(*zip(*no_vec_dic.items()))
ax.set_xlabel('Number of vectors (max %i)'%(continuation_object.one_rdm.shape[-1]**2))
ax.set_ylabel('(bra, ket) index')
plt.show()
