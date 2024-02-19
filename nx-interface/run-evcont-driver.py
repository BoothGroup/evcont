#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 16:04:25 2023

Driver to interface Newton-X CS 2.4 for non-adiabatic molecular dynamics
simulations with eigenvector continuation

For the interface, we need:
    - Reading geometry
    - Computing and writing necessary quantities:
        + Dynamics: Multistate energies, gradients and nonadiabatic coupling vectors (NAC)
        + InitCond: Multistate energies and oscillator strengths

@author: Kemal Atalar
"""

import numpy as np

import os
import sys

############################
# INPUTS (migth be converted to an input file later on)
BASIS = "sto-6g"
use_pyscf = False

# FCI related if use_pyscf
fix_singlet = True
fix_sym = 'A1g' #None
#fix_sym = None
############################
# Checks for evcont and pyscf
try:
   from evcont.ab_initio_gradients_loewdin import get_multistate_energy_with_grad_and_NAC
   from evcont.FCI_NAC import get_FCI_energy_with_grad_and_NAC, get_FCI_energy_with_grad_and_NAC_withsym
except:
   print('evcont is not installed!')
   sys.exit()
   
try:
    from pyscf import gto
except:
   print('pyscf is not installed!')
   sys.exit()
            
   
# Get parameters from nx-interface
NSTAT  	  = int(sys.argv[1])
NSTATDYN  = int(sys.argv[2])

############################

def read_mol(basis):
    """
    Read the current geometry from the trajectory and build the molecule object
    """
    # Assumes geom file is in the current directory
    geom_f = 'geom'
    
    atom_f = []
    with open(geom_f,'r') as f:
        for line in f.readlines():
            splt = line.split()
            #sym, atomic no, xc, yc, zc, mass
            atom_f.append((splt[0], np.array(splt[2:5],dtype=np.float64)))            
            #atom_f.append((splt[0], [float(i) for i in splt[2:5]]))      
            
    #print(atom_f)
    
    # Create the molecule
    mol = gto.Mole()

    mol.build(
        atom=atom_f,
        basis=basis,
        symmetry=mol_sym,
        unit="Bohr",
        verbose=0
    )
    
    return mol

############################
# Symmetry
if fix_sym == None or not use_pyscf:
    mol_sym = False
else:
    mol_sym = True
    
MOL = read_mol(BASIS)

# Set FCI solver if use_pyscf
if use_pyscf:
    from pyscf import fci
    # Set fci solver to be used
    
    if fix_sym == None:
        FCISOLVER = fci.direct_spin0.FCI()
    else:
        FCISOLVER = fci.direct_spin0_symm.FCI(MOL)
        FCISOLVER.wfnsym = fix_sym
        
    FCISOLVER.nroots = NSTAT+1

    if fix_singlet:
        fci.addons.fix_spin_(FCISOLVER,ss=0) # Fix spin
        
############################

def run_training(path):
    """
    TODO: Automate training, it needs pretraining for now
    """
    pass


def read_model(path):
    """
    Read the intermediate data that will be used for predictions, namely:
        - Overlap of training wavefunctions, S
        - 1-el reduced transition density matrices of training wavefunctions
        - 2-el reduced transition density matrices of training wavefunctions

    Args:
        path (str): 
            path to the model files - for now it is in the JOB_NAD drc
            (assumes a single directory containing files: 
                 overlap_final.npy, one_rdm_final.npy, two_rdm_final.npy)
        
    Returns:
        overlap (ndarray)
        one_rdm (ndarray)
        two_rdm (ndarray)
    """
    
    overlap = np.load(os.path.join(path,'overlap_final.npy'))
    one_rdm = np.load(os.path.join(path,'one_rdm_final.npy'))
    two_rdm = np.load(os.path.join(path,'two_rdm_final.npy'))
    
    return overlap, one_rdm, two_rdm

def get_phase(old,new):
    
    #norm_old = np.linalg.norm(old)
    #norm_new = np.linalg.norm(new)

    cosq = np.einsum("ij,ij",old, new)
        
    if cosq >= 0:
        return 1.
    else:
        return -1.
    
def adjust_phase(natm):
    
    # Read old and current NACs
    currentnac = np.loadtxt('nad_vectors')
    try:
        oldnac = np.loadtxt('oldh')
    except:
        oldnac = currentnac
        
    # Compute the overlap and adjust the phase
    n_nac = int(oldnac.shape[0]/natm)
    
    adjusted_nacs = []
    for i in range(n_nac):
        oldi= oldnac[i*natm : (i+1)*natm, :]
        curri = currentnac[i*natm : (i+1)*natm, :]
        
        phase = get_phase(oldi,curri)
        adjusted_nacs.append(phase * curri)
        
    # Write the adjusted NACs
    np.savetxt('nad_vectors',np.vstack(adjusted_nacs))
    
def write_traj(mol):
    '''
    Write positions along the trajectory to a separate file, 'traj_geom.npy' 
    (to retain more precision than 'dyn.out')

    '''
    fnam = 'traj_geom.npy'
    
    if not os.path.isfile(fnam):
        # Create the first instance
        np.save(fnam, [mol.atom_coords()])
        
    else:
        # Load
        coord = np.load(fnam)
        
        # Add the new geometry
        new_traj = np.concatenate((coord,[mol.atom_coords()]))
        
        # Write to file
        np.save(fnam, new_traj)

    
def evcont_feed_nx(mode, adjustphase=True):
    '''
    Call evcont at the geometry to extract energies, gradients and nonadiabatic 
    coupling vectors (can be extended to other properties)
    
    Modified from run-mlatom-driver.py in Newton-X MLAtom interface
    
    Args:
        mode (int):
            0 - initcond
                    Only modifies oscillator strengths (Not implemented yet)
            1 - dynamics
                    Updates energies, gradients and NACs
    '''
    
    # Get the mol object for continuation
    mol = read_mol(BASIS)
    
    # Add the current geometry to list of geometries along the trajectory
    write_traj(mol)
    
    # Get energies, gradients, NAC
    if not use_pyscf:
        print('Implementation: evcont')

        # Read the intermediate state from continuation training
        cwd = os.getcwd()
        cont_ovlp, cont_1rdm, cont_2rdm = read_model(cwd)
        
        # From eigenvector continuation
        en_cont, grad_cont, nac_cont, _ = get_multistate_energy_with_grad_and_NAC(
            mol,
            cont_1rdm, cont_2rdm, cont_ovlp,
            nroots=NSTAT+1
            )

    else:
        print('Implementation: pyscf FCI - sym_%s'%fix_sym)
        # FCI results in SAO basis
        en_cont, grad_cont, nac_cont, _ = get_FCI_energy_with_grad_and_NAC_withsym(
            mol,
            FCISOLVER,
            nroots=NSTAT+1,
            irrep_name=fix_sym
            )
    
    # Checks - write to output (going into EVCont.out)
    print('geom',mol.atom_coords())
    print()
    print('en',en_cont)
    print()
    print('grad', grad_cont)
    print()
    print('nac',nac_cont)
    print()
    
    # Write energies and gradients
    with open('epot', 'w') as fepot, open('grad.all', 'w') as fgradall, open('grad', 'w') as fgrad:
        for istate in range(1,NSTAT+1):
            fepot.writelines(' %.13f\n' % en_cont[istate-1])
            
            for iatom in range(mol.natm):
                current = grad_cont[istate-1,iatom,:]
                
                fgradall.writelines(' %.13f %.13f %.13f\n' % (current[0],current[1],current[2]))
                if (istate == NSTATDYN):
                    fgrad.writelines(' %.13f %.13f %.13f\n' % (current[0],current[1],current[2]))

    # Write nonadiabatic coupling vectors
    with open('nad_vectors', 'w') as fnad:
        for ii in range(NSTAT):
            for jj in range(ii):
                nac_str = str(ii)+str(jj)
                
                for iatom in range(mol.natm):
                    current = nac_cont[nac_str][iatom,:]

                    fnad.writelines(' %.13f  %.13f  %.13f\n' % (current[0],current[1],current[2]))

    if adjustphase:
        adjust_phase(mol.natm)
        
	# TODO: Transition moments, Oscillator strengths, energy gaps, etc.

    return 1

if __name__ == '__main__':
    
    check = False
    #mol = read_mol(basis=BASIS)

    # Dynamics only for now
    evcont_feed_nx(1)

    # Try for specific cases
    if check:
        import os
        
        drc = '/Users/katalar/Code/newtonx/Analysis/H8/S2-dt01/evcont-ntrain11/TEMP'
        
        cwd = os.getcwd()
        
        os.chdir(drc)
        #evcont_feed_nx(1)
    
        mol = read_mol(BASIS)
        
        tmpd = os.getcwd()
        cont_ovlp, cont_1rdm, cont_2rdm = read_model(tmpd)
        
        # From eigenvector continuation
        en_cont, grad_cont, nac_cont, _ = get_multistate_energy_with_grad_and_NAC(
            mol,
            cont_1rdm, cont_2rdm, cont_ovlp,
            nroots=NSTAT+1
            )
        
        os.chdir(cwd)

