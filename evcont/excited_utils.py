#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:01:50 2023

@author: katalar
"""

import numpy as np

import numpy.linalg as LA

from pyscf import gto

from pyscf.scf import hf

from evcont.FCI_EVCont import FCI_EVCont_obj

from evcont.ab_initio_eigenvector_continuation import approximate_multistate_OAO

from evcont.electron_integral_utils import get_basis, get_integrals

import sys


def make_trdm1(mol, one_rdm, vec_i, vec_j):
    """ 
    Predicted 1-body transition reduced density matrices from eigenvector continuation
    in the AO basis (converted from OAO basis)
    
    Input:
        mol: pyscf.gto.Mole()
            Molecule object
        one_rdm: Training one-rdm matrix ndarray(ntrain, ntrain, nao, nao)
            One or a list of 1-body reduced density matrices
        
    Returns:
        predicted_one_trdm_ao: ndarray(nao, nao)
            Predicted 1-body transition reduced density matrix
    """
    
    basis = get_basis(mol)

    predicted_one_trdm = np.einsum("i,ijkl,j->kl", vec_i, one_rdm, vec_j, optimize="optimal")
    # Convert to AO basis
    #predicted_one_trdm_ao = np.einsum("ji,jk,kl->il",basis, predicted_one_trdm, basis, optimize="optimal")
    predicted_one_trdm_ao = basis.dot(predicted_one_trdm).dot(basis.T)
    
    return predicted_one_trdm_ao

def make_rdm1(mol, one_rdm, vec):
    """ 
    Predicted 1-body reduced density matrices from eigenvector continuation
    in the AO basis (converted from OAO basis)
    """
    
    #basis = get_basis(mol)
    #predicted_one_rdm = np.einsum("i,ijkl,j->kl", vec, one_rdm, vec, optimize="optimal")
    #predicted_one_rdm = np.einsum("ji,jk,kl->il",basis, predicted_one_rdm, basis, optimize="optimal")
    
    predicted_one_rdm = make_trdm1(mol, one_rdm, vec, vec)
    
    return predicted_one_rdm


def trans_dip_moment(mol, one_trdm_inp):
    
    # Set gauge for dipole integrals
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
    mol.set_common_orig_(nuc_charge_center)
    dip_ints = mol.intor('cint1e_r_sph', comp=3)
    
    single = True
    if not isinstance(one_trdm_inp,list):
        one_trdm_l = [one_trdm_inp]
    else:
        one_trdm_l = one_trdm_inp
        single = False
    
    # Iterate over one_rdm s
    mol_tdip_l = []
    for one_trdm in one_trdm_l:
        el_tdip = np.einsum('xij,ji->x', dip_ints, one_trdm).real
        mol_tdip_l += [el_tdip]
    
    if not single:
        return mol_tdip_l
    else:
        return mol_tdip_l[0]
    
    #return dip_moment(mol, one_trdm_inp)
    #return [hf.dip_moment(mol, nn, unit='au') for nn in one_trdm_inp]


def dip_moment(mol, one_rdm_inp):
    """
    Compute dipole moment from 1-body reduced density matrices
    (Based on pyscf implementation)
    
    Input:
        mol: pyscf.gto.Mole()
            Molecule object
        one_rdm_inp: list of ndarray(mol.nao, mol.nao) or ndarray(mol.nao, mol.nao)
            One or a list of 1-body reduced density matrices
        
    Returns:
        mol_dip or mol_dip_l: ndarray(3,) or list of ndarray(3,)
            Dipole moments of the molecule (units A.U.)
    """
    
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    nucl_dip = np.einsum('i,ix->x', charges, coords)

    with mol.with_common_orig((0,0,0)):
        ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        
    single = True
    if not isinstance(one_rdm_inp,list):
        one_rdm_l = [one_rdm_inp]
    else:
        one_rdm_l = one_rdm_inp
        single = False
        
    # Iterate over one_rdm s
    mol_dip_l = []
    for one_rdm in one_rdm_l:
        el_dip = np.einsum('xij,ji->x', ao_dip, one_rdm).real
        mol_dip = nucl_dip - el_dip
        mol_dip_l += [mol_dip]
    
    if not single:
        return mol_dip_l
    else:
        return mol_dip_l[0]
        
    #dipole_mom = hf.dip_moment(mol, one_rone_rdm, unit='au')
    #return dipole_mom

def oscillator_strength(en_i, en_j, tran_dipole):
    """
    Compute the oscillator strength from the energies and transition dipole moment
    f_ij = 2/3 (E_i - E_j) * | t_ij |^2

    """
    
    return 2/3. * (en_i-en_j)* LA.norm(tran_dipole)**2.
    

def print_excited(en, dip_mom, trans_dip, osc_str=None):
    """
    Print excited state information from eigenvector continuations
    """

    nstate = len(en)
    # Add dummy first element for ground state index
    trans_dip_n = [np.array([0.,0.,0.])] + trans_dip
    if osc_str is not None:
        osc_str_n = [0.] + osc_str
    
    if osc_str is None:
        pr_format = '{:5}   {:8.5f}   {:6.3f} {:6.3f} {:6.3f} |{:8.5f}    {:6.3f} {:6.3f} {:6.3f} |{:8.5f}'
        print('{:6} {:15} {:25}    {:30}'.format('state','DeltaE (H)','dipole moment (au)','transition dipole moment (au)'))
        for i in range(nstate):
            print(pr_format.format(i, en[i]-en[0],*dip_mom[i],LA.norm(dip_mom[i]),*trans_dip_n[i],LA.norm(trans_dip_n[i])))
        print()
        
    else:
        pr_format = '{:5}   {:8.5f}   {:6.3f} {:6.3f} {:6.3f} |{:8.5f}    {:6.3f} {:6.3f} {:6.3f} |{:8.5f}    {:8.5f} '
        print('{:6} {:15} {:25}    {:30}'.format('state','DeltaE (H)','dipole moment (au)','transition dipole moment (au)'))
        for i in range(nstate):
            print(pr_format.format(i, en[i]-en[0],*dip_mom[i],LA.norm(dip_mom[i]),*trans_dip_n[i],LA.norm(trans_dip_n[i]),osc_str_n[i]))
        print()
        
        
if __name__ == '__main__':
    
    from pyscf import fci
    
    cibasis='canonical'
    nroots_evcont = 6
    cisolver = fci.direct_spin0.FCI()
    
    natom = 2 # Fixed, don't change
    
    def get_mol(geometry):
        mol = gto.Mole()

        mol.build(
            atom=[
            ("Li", geometry[0]),
            ("H", geometry[1]),
            ],
            #basis="aug-cc-pVDZ",
            basis="cc-pVDZ",
            #basis="631-G*",
            #basis="sto-6g",
            symmetry=True,
            unit="Bohr",
            verbose=0
        )

        return mol

    ang2bohr = 1.88973
    equilibrium_dist = 1.5957*ang2bohr

    #equilibrium_dist = 1.78596

    equilibrium_pos = np.array([(x * equilibrium_dist, 0.0, 0.0) for x in range(10)])

    training_stretches = np.array([0.0, 0.5, -0.5, 1.0, -1.0])

    trainig_dists = equilibrium_dist + training_stretches

    #trainig_dists = [1.0, 1.8, 2.6]
    
    continuation_object = FCI_EVCont_obj(nroots=nroots_evcont,
                                         cibasis=cibasis,
                                         cisolver=cisolver)
    
        
    # Generate training data + prepare training models
    for i, dist in enumerate(trainig_dists):
        positions = [(x, 0.0, 0.0) for x in dist * np.arange(natom)]
        mol = get_mol(positions)
        continuation_object.append_to_rdms(mol)
    
    for di, dist in enumerate(trainig_dists):
        print()
        print('Training dist: %.3f'%dist)
        print()
        
        positions = [(x, 0.0, 0.0) for x in dist * np.arange(natom)]
        mol = get_mol(positions)
    
        basis = get_basis(mol,basis_type=cibasis)
        h1, h2 = get_integrals(mol, basis)
    
        # Continuation
        en, vec = approximate_multistate_OAO(mol,
                                         continuation_object.one_rdm, 
                                         continuation_object.two_rdm,
                                         continuation_object.overlap,
                                         nroots=nroots_evcont)
        
        # Continuation dipole moment from one-rdms of individual states
        one_rdm_predicted_l = [make_rdm1(mol, continuation_object.one_rdm, vec[ii]) for ii in range(len(vec))]
        dipole_moment_predicted_l =  dip_moment(mol,one_rdm_predicted_l)
        
        # Transition rdms and dipole moments from ground to excited states
        one_trdm_predicted_l = [make_trdm1(mol, continuation_object.one_rdm, vec[0], vec[ii]) for ii in range(1,len(vec))]
        transition_dipmoment_predicted_l = trans_dip_moment(mol, one_trdm_predicted_l)
        
        # Oscillator strength
        osc_strength_predicted_l = [oscillator_strength(en[ii], en[0], transition_dipmoment_predicted_l[ii-1]) for ii in range(1,len(vec))]
        
        print_excited(en,dipole_moment_predicted_l, transition_dipmoment_predicted_l,osc_strength_predicted_l)
        #print()
        
        # FCI
        e_all, fcivec_all = cisolver.kernel(h1, h2, mol.nao, mol.nelec,
                                         nroots=nroots_evcont,)
                                         #tol = 1.e-14,max_space=30,
                                         #max_cycle=250)
        e_all += mol.energy_nuc()
        assert np.allclose(e_all, en)
        
        one_rdm_l = [cisolver.make_rdm1(fcivec_all[ii],mol.nao,mol.nelec) for ii in range(len(fcivec_all))]
        one_rdm_l = [basis.dot(one_rdm_l[ii]).dot(basis.T) for ii in range(len(one_rdm_l))]
        
        #dipole_moment_l = [dip_moment(mol,one_rdm_l[i]) for i in range(len(one_rdm_l))]
        dipole_moment_l = dip_moment(mol,one_rdm_l)
        
        # Transition dipole moment
        one_trdm_l = [cisolver.trans_rdm1(fcivec_all[0],fcivec_all[ii],mol.nao,mol.nelec) for ii in range(1,len(fcivec_all))]
        one_trdm_l = [basis.dot(one_trdm_l[ii]).dot(basis.T) for ii in range(len(one_trdm_l))]
        #one_trdm_l = [np.einsum("ji,jk,kl->il",basis, one_trdm_l[ii], basis, optimize="optimal") for ii in range(len(one_trdm_l))]
        
        transition_dipmoment_l = trans_dip_moment(mol, one_trdm_l)
        
        osc_strength_l = [oscillator_strength(e_all[ii], e_all[0], transition_dipmoment_l[ii-1]) for ii in range(1,len(fcivec_all))]
        
        print_excited(e_all,dipole_moment_l, transition_dipmoment_l,osc_strength_l)

        assert np.allclose(dipole_moment_l,dipole_moment_predicted_l)
        
        #assert np.allclose(transition_dipmoment_l,transition_dipmoment_predicted_l)

        #1/0
