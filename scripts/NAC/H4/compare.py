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

import matplotlib.pylab as plt


nstate = 2 #1st excited state
nroots_evcont = 3
cibasis = 'OAO'

natom = 4

check_spin = False
plot_extensive = True

test_range = np.linspace(0.8, 3.0,40)
#test_range = np.linspace(0.4, 1.5,20)

# Set fcisolver - fix spin 
myci =  fci.direct_spin0.FCI()
#myci.nroots = nstate+4
fci.addons.fix_spin_(myci,ss=0)

def get_mol(positions):
    mol = gto.Mole()

    mol.build(
        atom=[("H", pos) for pos in positions],
        basis="sto-3g",
        #basis="6-31g",
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

#trainig_dists = [1.0, 1.8, 2.6]
#trainig_dists = [1.0, 1.8]

trainig_dists = np.linspace(0.8,3.0,6)
# New angstrom ones
#trainig_dists = [0.5,0.9,1.3]

continuation_object = FCI_EVCont_obj(nroots=nroots_evcont,
                                     cibasis=cibasis,cisolver=myci)

# Generate training data + prepare training models
for i, dist in enumerate(trainig_dists):
    positions = [(x, 0.0, 0.0) for x in dist * np.arange(natom)]
    mol = get_mol(positions)
    continuation_object.append_to_rdms(mol)

print('Finished training')

train_nac = []
train_grad = []
train_en = []
for i, test_dist in enumerate(trainig_dists):
    print(i)
    positions = [(x, 0.0, 0.0) for x in test_dist * np.arange(natom)]
    
    mol = get_mol(positions)
    h1, h2 = get_integrals(mol, get_basis(mol))
    
    # Continuation
    en_continuation_ms, grad_continuation, nac_continuation, _ = get_multistate_energy_with_grad_and_NAC(
        mol,
        continuation_object.one_rdm,
        continuation_object.two_rdm,
        continuation_object.overlap,
        nroots=nstate+1
    )
    
    #cont_en[i,:] = en_continuation_ms
    train_nac += [nac_continuation]
    train_grad += [grad_continuation]
    train_en += [en_continuation_ms]

# Prediction on test dataset and comparison against FCI results
fci_en = np.zeros([len(test_range),nstate+4])
fci_nac = []
cont_en = np.zeros([len(test_range),nstate+1])
cont_nac = []
cont_nac_hf = []
fci_spin = np.zeros([len(test_range),nstate+4])

for i, test_dist in enumerate(test_range):
    print(i)
    positions = [(x, 0.0, 0.0) for x in test_dist * np.arange(natom)]
    
    mol = get_mol(positions)
    h1, h2 = get_integrals(mol, get_basis(mol))
    
    # Continuation
    en_continuation_ms, _, nac_continuation, nac_cont_hfonly = get_multistate_energy_with_grad_and_NAC(
        mol,
        continuation_object.one_rdm,
        continuation_object.two_rdm,
        continuation_object.overlap,
        nroots=nstate+1
    )
    
    cont_en[i,:] = en_continuation_ms
    cont_nac += [nac_continuation]
    cont_nac_hf += [nac_cont_hfonly]
    
    # FCI ones
    mc = CASCI(mol.RHF(), natom, natom)
    mc.fcisolver = fci.direct_spin0.FCI()
    mc.fcisolver.nroots = nstate+4
    fci.addons.fix_spin_(mc.fcisolver,ss=0) # Fix spin
    #ci_scan_exc = mc.nuc_grad_method().as_scanner(state=nstate)
    #ci_scan_0 = mc.nuc_grad_method().as_scanner(state=0)
    
    #en_exc_exact, grad_exc_exact = ci_scan_exc(mol)
    #en_exact, grad_exact = ci_scan_0(mol)
    en_exact, fcivec_pos = mc.fcisolver.kernel(h1, h2, mol.nao, mol.nelec)
    en_exact += mol.energy_nuc()

    # [spin, 2S+1]
    spin_exact = [mc.fcisolver.spin_square(fcivec_i,  mol.nao, mol.nelec)[0] for fcivec_i in fcivec_pos]
    
    # Get the reference numerical FCI NACs (update this part)
    #nac_all = nac_continuation
    
    fci_en[i,:] = en_exact
    fci_spin[i,:] = spin_exact
    #fci_nac += [nac_all]
    
#######################################################################
if check_spin:
    
    fig, axes = plt.subplots(nrows=2,ncols=1,sharex=True,sharey='row',
                             figsize=[6,10],gridspec_kw={'hspace':0.,'wspace':0},
                             height_ratios=[3,1])
    
    axes[0].plot(test_range,fci_en,alpha=0.8,label=['FCI-pyscf']+[None]*(fci_en.shape[1]-1))
    
    axes[1].plot(test_range,fci_spin)
    
    axes[0].set_ylabel('E (Ha)')
    axes[1].set_ylabel(r'$S^2$')
    
    plt.show()

#######################################################################
# Read NACs computed from openMOLCAS
fname = 'test_NACs.pkl'
with open(fname,'rb') as f:
    test_NACs = pickle.load(f)

# MOLCAS energies for comparison
fname = 'test_en.npy'
with open(fname,'rb') as f:
    molcas_en = np.load(f)

# Separate for geometry for plotting
fci_nac = []
for key in test_NACs.keys():
    all_NACs = test_NACs[key]
    nac_i = {}
    
    for keyj in all_NACs.keys():
        # 0 - CI contribution (not divided by energy difference)
        # 1 - CSF contribution
        # 2 - Full NACs
        ci_NAC = all_NACs[keyj][2]
        nac_i[keyj] = ci_NAC
    fci_nac.append(nac_i)
    

# CI only part of the molcas NACs
fci_cionly_nac = []
for key in test_NACs.keys():
    all_NACs = test_NACs[key]
    nac_i = {}
    
    for keyj in all_NACs.keys():
        ci_NAC = all_NACs[keyj][2] -  all_NACs[keyj][1]
        nac_i[keyj] = ci_NAC
    fci_cionly_nac.append(nac_i)
    
#######################################################################
# Plot NAC comparison
import matplotlib.pylab as plt

fci_absh = {}
fci_cionly_absh = {}
cont_absh = {}
cont_hf_absh = {}
for istate in range(nstate+1):
    for jstate in range(nstate+1):
        if istate != jstate:

            st_label = str(istate)+str(jstate)
            fci_absh[st_label] = np.array([np.abs(fci_nac[i][st_label]).sum() for i in range(len(test_range))])
            fci_cionly_absh[st_label] = np.array([np.abs(fci_cionly_nac[i][st_label]).sum() for i in range(len(test_range))])
            cont_absh[st_label] = np.array([np.abs(cont_nac[i][st_label]).sum() for i in range(len(test_range))])
            cont_hf_absh[st_label] = np.array([np.abs(cont_nac_hf[i][st_label]).sum() for i in range(len(test_range))])
    
# Colors
clr_st = {'01':'b', '10':'b',
          '02':'r','20':'r',
          '03':'pink','30':'pink',
          '13':'y','31':'y',
          '23':'violet','32':'violet',
          '12':'g','21':'g'}

labelsize = 15
interfont=12

if not plot_extensive:
    # Plot
    fig, axes = plt.subplots(nrows=2,ncols=2,sharex=True,sharey='row',
                             figsize=[10,10],gridspec_kw={'hspace':0.,'wspace':0},
                             height_ratios=[1,1])
    
    axes[0][0].plot(test_range,molcas_en,'k',alpha=0.8,label=['CASSCF-molcas']+[None]*(molcas_en.shape[1]-1))
    axes[0][0].plot(test_range,fci_en,'r--',alpha=0.8,label=['FCI-pyscf']+[None]*(fci_en.shape[1]-1))
    axes[0][1].plot(test_range,cont_en,'k',alpha=0.8)
    axes[0][1].plot(trainig_dists, np.array(train_en),'xr')
    
    for key, el in fci_absh.items():
        axes[1][0].plot(test_range,fci_absh[key],label=key,c=clr_st[key])
        axes[1][1].plot(test_range,cont_absh[key],label=key,c=clr_st[key])
    
    axes[1][0].legend(loc='upper right',fontsize=interfont)
    axes[0][0].legend(loc='upper right',fontsize=interfont)
    
    axes[0][0].set_title('CASSCF',fontsize=labelsize)
    axes[0][1].set_title('EVcont',fontsize=labelsize)
    axes[1][0].set_ylabel(r'$||\mathbf{d}_{ij}||$ (a$_0$$^{-1}$)',fontsize=labelsize)
    axes[0][0].set_ylabel(r'Energy (Hartree)',fontsize=labelsize)
    
    
    axes[1][0].set_ylim(ymin=-0.2, ymax=min(10,axes[1][0].get_ylim()[1]))
    
    for axcol in axes[1:]:
        ylims =  axcol[1].get_ylim()
        axcol[1].vlines(x=trainig_dists,ymin=ylims[0],ymax=ylims[1],ls='--',color='gray')
        
    plt.show()

#######################################################################
else:
    # Plot
    fig, axes = plt.subplots(nrows=4,ncols=3,sharex=True,
                             figsize=[15,20],gridspec_kw={'hspace':0.,'wspace':0.2},
                             height_ratios=[1,1,1,1])
    
    #axes[0][1].plot(test_range,molcas_en,'k',alpha=0.8,label=['CASSCF-molcas']+[None]*(molcas_en.shape[1]-1))
    axes[0][0].plot(test_range,molcas_en,'k',alpha=0.8,label=['CASSCF-molcas']+[None]*(molcas_en.shape[1]-1))
    axes[0][0].plot(test_range,fci_en,'r--',alpha=0.8,label=['FCI-pyscf']+[None]*(fci_en.shape[1]-1))
    
    axes[0][1].plot(test_range,cont_en,'k',alpha=0.8)
    axes[0][1].plot(trainig_dists, np.array(train_en),'xr')
    
    axes[0][2].plot(test_range, np.abs(cont_en-molcas_en[:,:nstate+1]),'k')
    
    for key, el in fci_absh.items():
        # FCI
        axes[1][0].plot(test_range,fci_absh[key],label=key,c=clr_st[key])
        axes[2][0].plot(test_range,fci_cionly_absh[key],label=key,c=clr_st[key])
        fci_corr = np.array(fci_absh[key])-np.array(fci_cionly_absh[key])
        axes[3][0].plot(test_range,fci_corr,label=key,c=clr_st[key])
        # EVcont
        axes[1][1].plot(test_range,cont_absh[key],label=key,c=clr_st[key])
        axes[2][1].plot(test_range,cont_hf_absh[key],label=key,c=clr_st[key])
        cont_corr = np.array(cont_absh[key])-np.array(cont_hf_absh[key])
        axes[3][1].plot(test_range,cont_corr,label=key,c=clr_st[key])
        # Diff
        axes[1][2].plot(test_range,np.abs(cont_absh[key]-fci_absh[key]),label=key,c=clr_st[key])
        axes[2][2].plot(test_range,np.abs(cont_hf_absh[key]-fci_cionly_absh[key]),label=key,c=clr_st[key])
        #cont_corr = np.array(cont_absh[key])-np.array(cont_hf_absh[key])
        axes[3][2].plot(test_range,np.abs(cont_corr-fci_corr),label=key,c=clr_st[key])
    
    axes[1][0].legend(loc='upper right',fontsize=interfont)
    axes[0][0].legend(loc='upper right',fontsize=interfont)
    
    axes[0][0].set_title('CASSCF',fontsize=labelsize)
    #axes[0][1].set_title('CASSCF - CI only',fontsize=labelsize)
    #axes[0][2].set_title('CASSCF - CSF correction',fontsize=labelsize)
    axes[0][1].set_title('EVcont',fontsize=labelsize)
    #axes[0][4].set_title('EVcont - HF only',fontsize=labelsize)
    #axes[0][5].set_title('EVcont - basis correction',fontsize=labelsize)
    axes[0][2].set_title('|EVcont - FCI|',fontsize=labelsize)
    
    axes[0][0].set_ylabel(r'Energy (Hartree)',fontsize=labelsize)
    axes[1][0].set_ylabel(r'$||\mathbf{d}_{ij}||$ (a$_0$$^{-1}$)',fontsize=labelsize)
    axes[2][0].set_ylabel(r'$||\mathbf{d}_{ij}||$ (a$_0$$^{-1}$) - CI only',fontsize=labelsize)
    axes[3][0].set_ylabel(r'$||\mathbf{d}_{ij}||$ (a$_0$$^{-1}$) - correction',fontsize=labelsize)
    
    axes[1][0].set_ylim(ymin=-0.2, ymax=min(10,axes[1][0].get_ylim()[1]))
    # Equate ylims
    for i in range(0,4):
        axes[i][0].set_ylim(axes[i][1].get_ylim())
        axes[i][2].set_yscale('log')
    
    for axcol in axes[1:]:
        for ax in axcol:
            ylims =  ax.get_ylim()
            ax.set_ylim(ylims)
            ax.vlines(x=trainig_dists,ymin=ylims[0],ymax=ylims[1],ls='--',color='k')
    
    for axcol in axes:
        for ax in axcol:
            ax.yaxis.grid(color='gray', linestyle='dashed')
            ax.xaxis.grid(color='gray', linestyle='dashed')
            
    #plt.savefig('compare-nac.png',dpi=1000,bbox_inches='tight')
    plt.show()
    
    
    
    
