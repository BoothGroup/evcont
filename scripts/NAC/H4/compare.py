#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 15:12:36 2023

Benchmark FCI NACs against OpenMOLCAS CASSCF NACs with full active space

@author: katalar
"""

import numpy as np

from pyscf import gto, fci

from evcont.FCI_EVCont import FCI_EVCont_obj

from evcont.electron_integral_utils import get_basis, get_integrals

from evcont.ab_initio_gradients_loewdin import get_multistate_energy_with_grad_and_NAC

from pyscf.mcscf import CASCI

import pickle

from time import time


nstate = 2 #1st excited state
nroots_evcont = 3
cibasis = 'OAO'

natom = 4

test_range = np.linspace(0.8, 3.0,20)
#test_range = np.linspace(0.4, 1.5,20)

def get_mol(positions):
    mol = gto.Mole()

    mol.build(
        atom=[("H", pos) for pos in positions],
        basis="sto-3g",
        symmetry=False,
        unit="Bohr",
        verbose=0
    )

    return mol


equilibrium_dist = 1.78596
#equilibrium_dist = 

equilibrium_pos = np.array([(x * equilibrium_dist, 0.0, 0.0) for x in range(10)])

#training_stretches = np.array([0.0, 0.5, -0.5, 1.0, -1.0])
#trainig_dists = equilibrium_dist + training_stretches

trainig_dists = [1.0, 1.8, 2.6]

# New angstrom ones
#trainig_dists = [0.5,0.9,1.3]

continuation_object = FCI_EVCont_obj(nroots=nroots_evcont,
                                     cibasis=cibasis)

# Generate training data + prepare training models
for i, dist in enumerate(trainig_dists):
    positions = [(x, 0.0, 0.0) for x in dist * np.arange(natom)]
    mol = get_mol(positions)
    continuation_object.append_to_rdms(mol)

print('Finished training')

train_nac = []
train_grad = []
for i, test_dist in enumerate(trainig_dists):
    print(i)
    positions = [(x, 0.0, 0.0) for x in test_dist * np.arange(natom)]
    
    mol = get_mol(positions)
    h1, h2 = get_integrals(mol, get_basis(mol))
    
    # Continuation
    en_continuation_ms, grad_continuation, nac_continuation = get_multistate_energy_with_grad_and_NAC(
        mol,
        continuation_object.one_rdm,
        continuation_object.two_rdm,
        continuation_object.overlap,
        nroots=nstate+1
    )
    
    #cont_en[i,:] = en_continuation_ms
    train_nac += [nac_continuation]
    train_grad += [grad_continuation]

fci_en = np.zeros([len(test_range),nstate+1])
fci_nac = []
cont_en = np.zeros([len(test_range),nstate+1])
cont_nac = []

for i, test_dist in enumerate(test_range):
    print(i)
    positions = [(x, 0.0, 0.0) for x in test_dist * np.arange(natom)]
    
    mol = get_mol(positions)
    h1, h2 = get_integrals(mol, get_basis(mol))
    
    # Continuation
    en_continuation_ms, _, nac_continuation = get_multistate_energy_with_grad_and_NAC(
        mol,
        continuation_object.one_rdm,
        continuation_object.two_rdm,
        continuation_object.overlap,
        nroots=nstate+1
    )
    
    cont_en[i,:] = en_continuation_ms
    cont_nac += [nac_continuation]
    
    # FCI ones
    mc = CASCI(mol.RHF(), natom, natom)
    mc.fcisolver = fci.direct_spin0.FCI()
    mc.fcisolver.nroots = nstate+1
    #ci_scan_exc = mc.nuc_grad_method().as_scanner(state=nstate)
    #ci_scan_0 = mc.nuc_grad_method().as_scanner(state=0)
    
    #en_exc_exact, grad_exc_exact = ci_scan_exc(mol)
    #en_exact, grad_exact = ci_scan_0(mol)
    en_exact, fcivec_pos = mc.fcisolver.kernel(h1, h2, mol.nao, mol.nelec)

    en_exact += mol.energy_nuc()
    
    # Get the reference numerical FCI NACs (update this part)
    #nac_all = nac_continuation
    
    fci_en[i,:] = en_exact
    #fci_nac += [nac_all]
    
# Read NACs computed from openMOLCAS
fname = 'test_NACs.pkl'
with open(fname,'rb') as f:
    test_NACs = pickle.load(f)

# Separate different contributions
fci_nac = []
for key in test_NACs.keys():
    all_NACs = test_NACs[key]
    nac_i = {}
    for keyj in all_NACs.keys():
        ci_NAC = all_NACs[keyj][0]
        nac_i[keyj] = ci_NAC
    fci_nac.append(nac_i)
    
#######################################################################
# Plot NAC comparison
import matplotlib.pylab as plt

fci_absh = {}
cont_absh = {}
for istate in range(nstate+1):
    for jstate in range(nstate+1):
        if istate != jstate:

            st_label = str(istate)+str(jstate)
            fci_absh[st_label] = [np.abs(fci_nac[i][st_label]).sum() for i in range(len(test_range))]
            cont_absh[st_label] = [np.abs(cont_nac[i][st_label]).sum() for i in range(len(test_range))]
    
# Colors
clr_st = {'01':'b', '10':'b',
          '02':'r','20':'r',
          '03':'pink','30':'pink',
          '13':'y','31':'y',
          '23':'violet','32':'violet',
          '12':'g','21':'g'}

labelsize = 15
# Plot
fig, axes = plt.subplots(nrows=2,ncols=2,sharex=True,sharey='row',
                         figsize=[10,10],gridspec_kw={'hspace':0.,'wspace':0},
                         height_ratios=[1,1])

axes[0][0].plot(test_range,fci_en,'k',alpha=0.8)
axes[0][1].plot(test_range,cont_en,'k',alpha=0.8)

for key, el in fci_absh.items():
    axes[1][0].plot(test_range,fci_absh[key],label=key,c=clr_st[key])
    axes[1][1].plot(test_range,cont_absh[key],label=key,c=clr_st[key])

axes[1][0].legend(loc='upper right')

axes[0][0].set_title('FCI',fontsize=labelsize)
axes[0][1].set_title('EVcont',fontsize=labelsize)
axes[1][0].set_ylabel(r'$||\mathbf{d}_{ij}||$ (a$_0$$^{-1}$)',fontsize=labelsize)
axes[0][0].set_ylabel(r'Energy (Hartree)',fontsize=labelsize)

axes[1][0].set_ylim(ymin=-0.2, ymax=min(10,axes[1][0].get_ylim()[1]))

plt.show()




