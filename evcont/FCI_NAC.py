#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 13:59:17 2023

Full CI nonadiabatic coupling vectors in SAO basis

@author: katalar
"""

import numpy as np

from pyscf import scf, ao2mo, grad, fci

from evcont.ab_initio_gradients_loewdin import (
    get_one_and_two_el_grad,
    get_loewdin_trafo,
    get_orbital_derivative_coupling,
    get_grad_elec_from_gradH
)

def get_FCI_energy_with_grad_and_NAC(mol, fcisolver, nroots=1, hermitian=True):
    """
    Calculates the potential energiesm its gradient w.r.t. nuclear positions of a
    molecule and nonadiabatic couplings from full CI in SAO basis
    
    Args:
        mol : pyscf.gto.Mole
            The molecule object.
        fcisolver : pyscf.fci solver
            The solver object for full CI
        nroots (optional): int
            Number of states in the solver.
        hermitian (optional): bool
            Whether problem is solved with eigh or with eig. Defaults to True.

    Returns:
        tuple (en, grad_all, nac_all, nac_all_hfonly)
            A tuple containing the total potential energies, its gradients and NACs:
            
            en: ndarray(nroot,)
                Total potential energies for both ground and excited states
                
            grad_all: list of ndarray(nat,3)
                Gradients of multistate energies
                
            nac_all: dictionary of np.darray(nat,3)
                Nonadiabatic coupling vectors between all states,
                e.g. nac_all['02'] is NAC along ground state and 2nd excited state
                
            nac_all_hfonly: dictionary of ndarray(nat,3)
                Hellman-Feynmann contribution to NACs
    """
                
    # Construct h1 and h2
    ao_mo_trafo = get_loewdin_trafo(mol.intor("int1e_ovlp"))
    
    h1 = np.linalg.multi_dot((ao_mo_trafo.T, scf.hf.get_hcore(mol), ao_mo_trafo))
    h2 = ao2mo.restore(1, ao2mo.kernel(mol, ao_mo_trafo), mol.nao)
    
    # Get FCI energies and wavefunctions in SAO basis
    en, fcivec = fcisolver.kernel(h1, h2, mol.nao, mol.nelec)
    
    # Get the gradient of one and two-electron integrals before contracting onto
    # rdms and trmds of different states
    h1_jac, h2_jac = get_one_and_two_el_grad(mol,ao_mo_trafo=ao_mo_trafo)
    
    # Get the orbital derivative coupling for NACs
    orb_deriv = get_orbital_derivative_coupling(mol,ao_mo_trafo=ao_mo_trafo)
    
    # Nuclear part of the gradient
    grad_nuc = grad.RHF(scf.RHF(mol)).grad_nuc()
        
    grad_elec_all = []
    nac_all = {}
    nac_all_hfonly = {}
    # Iterate over pairs of eigenstates
    for i_state in range(nroots):
        ci_i = fcivec[i_state]

        for j_state in range(nroots):
            ci_2 = fcivec[j_state]
            
            # FCI 1- and 2-particle tRDMs
            one_rdm, two_rdm = fcisolver.trans_rdm12(ci_i, ci_2, mol.nao, mol.nelec)
            
            # d\dR of subspace Hamiltonian
            grad_elec = get_grad_elec_from_gradH(
                one_rdm, two_rdm, h1_jac, h2_jac
            )
            
            # Energy gradients
            if i_state == j_state:
                grad_elec_all.append(grad_elec)
                
            # Nonadiabatic couplings
            else:
                nac_orb = np.einsum("ij,ijkl->kl",one_rdm, orb_deriv, optimize="optimal")

                nac_hf = grad_elec/(en[j_state]-en[i_state])
                nac_ij = nac_hf + nac_orb
                nac_all[str(i_state)+str(j_state)] = nac_ij
                nac_all_hfonly[str(i_state)+str(j_state)] = nac_hf

    # Add the nuclear contribution to gradient
    grad_all = np.array(grad_elec_all) + grad_nuc

    return (
        en.real + mol.energy_nuc(),
        grad_all,
        nac_all,
        nac_all_hfonly
    )

if __name__ == '__main__':

    from pyscf import gto
    import pickle
    
    nstate = 2 # Max no of state to compute NACs up to
    
    natom = 4
    
    check_spin = True
    withMolcas = False
    fix_singlet = True
    
    test_range = np.linspace(0.8, 3.0,40)
    
    # Set fci solver to be used
    fcisolver = fci.direct_spin0.FCI()
    fcisolver.nroots = nstate+4
    
    if fix_singlet:
        fci.addons.fix_spin_(fcisolver,ss=0) # Fix spin
        
    
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

    # Prediction on test dataset and comparison against FCI results
    fci_en = np.zeros([len(test_range),nstate+4])
    fci_nac = []
    fci_cionly_nac = []
    
    for i, test_dist in enumerate(test_range):
        print(i)
        positions = [(x, 0.0, 0.0) for x in test_dist * np.arange(natom)]
        
        mol = get_mol(positions)
        #h1, h2 = get_integrals(mol, get_basis(mol))
        
        # Continuation
        en_f, grad_f, nac_f, nac_f_hfonly = get_FCI_energy_with_grad_and_NAC(
            mol,
            fcisolver,
            nroots=nstate+1
        )
        
        fci_en[i,:] = en_f
        fci_nac += [nac_f]
        fci_cionly_nac += [nac_f_hfonly]
        
    
    #######################################################################
    if withMolcas:
        # Read NACs computed from openMOLCAS
        fname = 'test_NACs.pkl'
        with open(fname,'rb') as f:
            test_NACs = pickle.load(f)
    
        # MOLCAS energies for comparison
        fname = 'test_en.npy'
        with open(fname,'rb') as f:
            molcas_en = np.load(f)
    
        # Separate for geometry for plotting
        molcas_nac = []
        for key in test_NACs.keys():
            all_NACs = test_NACs[key]
            nac_i = {}
            
            for keyj in all_NACs.keys():
                # 0 - CI contribution (not divided by energy difference)
                # 1 - CSF contribution
                # 2 - Full NACs
                ci_NAC = all_NACs[keyj][2]
                nac_i[keyj] = ci_NAC
            molcas_nac.append(nac_i)
            
    
        # CI only part of the molcas NACs
        molcas_cionly_nac = []
        for key in test_NACs.keys():
            all_NACs = test_NACs[key]
            nac_i = {}
            
            for keyj in all_NACs.keys():
                ci_NAC = all_NACs[keyj][2] -  all_NACs[keyj][1]
                nac_i[keyj] = ci_NAC
            molcas_cionly_nac.append(nac_i)
            
    #######################################################################
    # Plot NAC comparison
    import matplotlib.pylab as plt

    fci_absh = {}
    fci_cionly_absh = {}
    molcas_absh = {}
    molcas_hf_absh = {}
    for istate in range(nstate+1):
        for jstate in range(nstate+1):
            if istate != jstate:

                st_label = str(istate)+str(jstate)
                fci_absh[st_label] = np.array([np.abs(fci_nac[i][st_label]).sum() for i in range(len(test_range))])
                fci_cionly_absh[st_label] = np.array([np.abs(fci_cionly_nac[i][st_label]).sum() for i in range(len(test_range))])
                
                if withMolcas:
                    molcas_absh[st_label] = np.array([np.abs(molcas_nac[i][st_label]).sum() for i in range(len(test_range))])
                    molcas_hf_absh[st_label] = np.array([np.abs(molcas_cionly_nac[i][st_label]).sum() for i in range(len(test_range))])
            
    # Colors
    clr_st = {'01':'b', '10':'b',
              '02':'r','20':'r',
              '03':'pink','30':'pink',
              '13':'y','31':'y',
              '23':'violet','32':'violet',
              '12':'g','21':'g'}

    labelsize = 15
    interfont=12

    if withMolcas:
        # Plot
        fig, axes = plt.subplots(nrows=2,ncols=3,sharex=True,
                                 figsize=[15,10],gridspec_kw={'hspace':0.,'wspace':0.1},
                                 height_ratios=[1,1])
        
        axes[0][0].plot(test_range,molcas_en,'k',alpha=0.8,label=['CASSCF-molcas']+[None]*(molcas_en.shape[1]-1))
        axes[0][0].plot(test_range,fci_en,'r--',alpha=0.8,label=['FCI-pyscf']+[None]*(fci_en.shape[1]-1))
        
        axes[0][1].plot(test_range,fci_en,alpha=0.8,label=['FCI']+[None]*(fci_en.shape[1]-1))
        #axes[0][1].plot(test_range,cont_en,'k',alpha=0.8)
        #axes[0][1].plot(trainig_dists, np.array(train_en),'xr')
        axes[0][2].plot(test_range, np.abs(fci_en[:,:nstate+1]-molcas_en[:,:nstate+1]))

        
        for key, el in fci_absh.items():
            axes[1][0].plot(test_range,molcas_absh[key],label=key,c=clr_st[key])
            axes[1][1].plot(test_range,fci_absh[key],label=key,c=clr_st[key])
            axes[1][2].plot(test_range,np.abs(molcas_absh[key]-fci_absh[key]),label=key,c=clr_st[key])

        axes[1][0].legend(loc='upper right',fontsize=interfont)
        axes[0][0].legend(loc='upper right',fontsize=interfont)
        
        axes[0][0].set_title('CASSCF - MOLCAS',fontsize=labelsize)
        axes[0][1].set_title('FCI - new implementation',fontsize=labelsize)
        axes[0][2].set_title('Diff',fontsize=labelsize)
        axes[1][0].set_ylabel(r'$||\mathbf{d}_{ij}||$ (a$_0$$^{-1}$)',fontsize=labelsize)
        axes[0][0].set_ylabel(r'Energy (Hartree)',fontsize=labelsize)
        
        
        axes[1][0].set_ylim(ymin=-0.2, ymax=min(10,axes[1][0].get_ylim()[1]))
        
        for axcol in axes:
            for ax in axcol:
                ax.yaxis.grid(color='gray', linestyle='dashed')
                ax.xaxis.grid(color='gray', linestyle='dashed')
                
        plt.show()
        
    else:
        # Plot
        fig, axes = plt.subplots(nrows=1,ncols=2,sharex=True,#sharey='row',
                                 figsize=[10,10],gridspec_kw={'hspace':0.,'wspace':0},
                                 height_ratios=[1])
        
        axes[0].plot(test_range,fci_en,'r--',alpha=0.8,label=['FCI-pyscf']+[None]*(fci_en.shape[1]-1))
        #axes[0][1].plot(test_range,cont_en,'k',alpha=0.8)
        #axes[0][1].plot(trainig_dists, np.array(train_en),'xr')
        
        for key, el in fci_absh.items():
            #axes[1][0].plot(test_range,molcas_absh[key],label=key,c=clr_st[key])
            axes[1].plot(test_range,fci_absh[key],label=key,c=clr_st[key])
        
        axes[1].legend(loc='upper right',fontsize=interfont)
        
        #axes[1][0].set_ylabel(r'$||\mathbf{d}_{ij}||$ (a$_0$$^{-1}$)',fontsize=labelsize)
        #axes[0][0].set_ylabel(r'Energy (Hartree)',fontsize=labelsize)
        
        
        #axes[1][0].set_ylim(ymin=-0.2, ymax=min(10,axes[1][0].get_ylim()[1]))

        plt.show()

    #######################################################################

