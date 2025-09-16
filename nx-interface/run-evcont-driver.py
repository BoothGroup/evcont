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
import pickle

############################
# INPUTS (might be converted to an input file later on)
#BASIS = "sto-6g"
#use_pyscf = False
#trdm_path = None

# FCI related if use_pyscf
#fix_singlet = True
#fix_sym = 'A1g' #None
#fix_sym = None

############################
# Checks for evcont and pyscf
try:
   from evcont.ab_initio_gradients_loewdin import get_multistate_energy_with_grad_and_NAC, get_lowrank_en_with_grad_and_NAC
   from evcont.FCI_NAC import get_FCI_energy_with_grad_and_NAC, get_FCI_energy_with_grad_and_NAC_withsym
except:
   print('Error in run-evcont-driver: evcont is not installed!')
   sys.exit()
   
try:
    from pyscf import gto
except:
   print('Error in run-evcont-driver: pyscf is not installed!')
   sys.exit()
            

# Get parameters from nx-interface
NSTAT  	  = int(sys.argv[1])
NSTATDYN  = int(sys.argv[2])

############################

def read_mol(basis, mol_sym):
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

def read_input_file(filename='evcont.in'):

    # Default input parameters
    defaults = {
        'basis': 'sto-6g',
        'use_pyscf': False,
        'trdm_path': None,
        'fix_singlet' : False,
        'fix_sym' : None,
        'lowrank' : False,
        'density_fit' : False,
        'df_basis' : None
    }

    variables = defaults.copy()
    input_path = os.path.join(os.getcwd(), filename)

    if not os.path.exists(input_path):
        print(f"Warning: '{filename}' not found in the current directory. Using all default values.")
        return variables

    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' in line:
                key, value = line.split('=', 1)
                key = key.strip()
                value = value.strip()

                # Attempt type conversion based on defaults
                if key in defaults:
                    expected_type = type(defaults[key])
                    try:
                        if expected_type == bool:
                            value = value.lower() == 'true'
                        else:
                            value = expected_type(value)
                    except ValueError:
                        print(f"Warning: Could not convert '{key}' to {expected_type.__name__}, using default.")
                        continue
                variables[key] = value

    return variables


def read_input_file(filename='evcont.in'):
    """
    Reads key=value pairs from an input file and fills in defaults.
    
    Args:
        filename (str): Path to input file. Defaults to 'evcont.in'.
        defaults (dict): Dictionary of default values.
        required_keys (list): Keys that must be present in input or defaults.
    
    Returns:
        dict: Dictionary of input parameters.
    
    Raises:
        FileNotFoundError: If input file is not found.
        ValueError: If required keys are missing.
    """
    # Default input parameters
    defaults = {
        'basis': 'sto-6g',
        'use_pyscf': False,
        'trdm_path': None,
        'fix_singlet' : False,
        'fix_sym' : None,
        'lowrank' : False,
        'density_fit' : False,
        'df_basis' : None
    }
    
    required_keys=[]

    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Input file '{filename}' not found.")

    user_inputs = {}
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if '=' not in line:
                continue
            key, value = map(str.strip, line.split('=', 1))
            value_lower = value.lower()

            # If value is "none", interpret as Python None
            if value_lower == 'none':
                converted_value = None
            elif key in defaults:
                default_value = defaults[key]
                if default_value is None:
                    converted_value = value
                else:
                    expected_type = type(default_value)
                    try:
                        if expected_type == bool:
                            converted_value = value_lower == 'true'
                        else:
                            converted_value = expected_type(value)
                    except ValueError:
                        print(f"Warning: Could not convert '{key}' to {expected_type.__name__}, using raw value.")
                        converted_value = value
            else:
                # No default given — store as string or None
                converted_value = None if value_lower == 'none' else value

            user_inputs[key] = converted_value

    # Fill in defaults for missing keys
    for key, val in defaults.items():
        if key not in user_inputs:
            user_inputs[key] = val

    # Enforce required keys
    missing = [k for k in required_keys if k not in user_inputs or user_inputs[k] is None]
    if missing:
        raise ValueError(f"Missing required input(s): {', '.join(missing)}")

    return user_inputs

def run_training(path):
    """
    TODO: Automate training, it needs pretraining for now
    """
    pass

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def read_model(path):
    """
    Read the intermediate data that will be used for predictions, namely:
        - Overlap of training wavefunctions, S
        - 1-el reduced transition density matrices of training wavefunctions
        - 2-el reduced transition density matrices of training wavefunctions,
          either as a full tensor (two_rdm_final.npy) or low-rank vectors (lowrank_vecs.pkl)

    Args:
        path (str): 
            Path to the model files directory. Must contain:
                - overlap_final.npy
                - one_rdm_final.npy
                - two_rdm_final.npy OR lowrank_vecs.pkl

    Returns:
        overlap (ndarray)
        one_rdm (ndarray)
        two_rdm (ndarray or any object from lowrank_vecs.pkl)
    """
    overlap = np.load(os.path.join(path, 'overlap_final.npy'))
    one_rdm = np.load(os.path.join(path, 'one_rdm_final.npy'))

    two_rdm_npy = os.path.join(path, 'two_rdm_final.npy')
    two_rdm_pkl = os.path.join(path, 'lowrank_vecs.pkl')

    if os.path.exists(two_rdm_npy):
        two_rdm = np.load(two_rdm_npy)
    elif os.path.exists(two_rdm_pkl):
        two_rdm = load_pickle(two_rdm_pkl)
    else:
        raise FileNotFoundError("Neither 'two_rdm_final.npy' nor 'lowrank_vecs.pkl' was found in the specified path.")

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
        
def write_cont(vec):
    '''
    Write positions along the trajectory to a separate file, 'traj_geom.npy' 
    (to retain more precision than 'dyn.out')

    '''
    fnam = 'traj_vec.npy'
    
    if not os.path.isfile(fnam):
        # Create the first instance
        np.save(fnam, [vec])
        
    else:        
        # Write to file
        np.save(fnam, np.concatenate((np.load(fnam),[vec])))

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
    
    # Read the input parameters
    inputs = read_input_file()

    trdm_path = inputs['trdm_path']
    use_pyscf = inputs['use_pyscf']
    fix_sym = inputs['fix_sym']
    
    # Symmetry
    if fix_sym == None or not use_pyscf:
        mol_sym = False
    else:
        mol_sym = True
        
    # Get the mol object for continuation
    mol = read_mol(inputs['basis'], mol_sym)

    # Set FCI solver if use_pyscf
    if use_pyscf:
        from pyscf import fci
        # Set fci solver to be used
        
        if fix_sym == None:
            FCISOLVER = fci.direct_spin0.FCI()
        else:
            FCISOLVER = fci.direct_spin0_symm.FCI(mol)
            FCISOLVER.wfnsym = fix_sym
            
        FCISOLVER.nroots = NSTAT+1

        if fix_singlet:
            fci.addons.fix_spin_(FCISOLVER,ss=0) # Fix spin

    # Add the current geometry to list of geometries along the trajectory
    write_traj(mol)
    
    # Get energies, gradients, NAC
    if not use_pyscf:
        print('Implementation: evcont')

        # Read the intermediate state from continuation training
        cwd = os.getcwd()
        if trdm_path is None:
            cont_ovlp, cont_1rdm, cont_2rdm = read_model(cwd)
        else:
            cont_ovlp, cont_1rdm, cont_2rdm = read_model(trdm_path)
        
        # From eigenvector continuation
        if inputs['lowrank']:
            vec_cont, en_cont, grad_cont, nac_cont, _ = get_lowrank_en_with_grad_and_NAC(
                mol,
                cont_1rdm,
                cont_ovlp,
                cont_2rdm, 
                None,
                nroots=NSTAT+1,
                density_fit=inputs['density_fit'],
                df_basis=inputs['df_basis']
                )
        else:
            vec_cont, en_cont, grad_cont, nac_cont, _ = get_multistate_energy_with_grad_and_NAC(
                mol,
                cont_1rdm, cont_2rdm, cont_ovlp,
                nroots=NSTAT+1
                )
        
        write_cont(vec_cont)

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
    print('vec', vec_cont, vec_cont.shape)
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
    
        mol = read_mol('sto-3g',False)
        
        tmpd = os.getcwd()
        cont_ovlp, cont_1rdm, cont_2rdm = read_model(tmpd)
        
        # From eigenvector continuation
        en_cont, grad_cont, nac_cont, _ = get_multistate_energy_with_grad_and_NAC(
            mol,
            cont_1rdm, cont_2rdm, cont_ovlp,
            nroots=NSTAT+1
            )
        
        os.chdir(cwd)

