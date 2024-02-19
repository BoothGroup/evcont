#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 09:56:23 2023

Functions to read Newton-X trajectories and their properties as well as 
active learning the training geometries for continuation

@author: katalar
"""

import numpy as np
import os
import sys
import subprocess
import glob

from scipy.signal import find_peaks

from evcont.electron_integral_utils import get_integrals, get_basis

##############################################################################
# NX I/O FUNCTIONS
##############################################################################

def read_population(fname,nstat):
    population = [[] for i in range(nstat)]
    with open(fname) as f:
        for line in f:
            if 'Population' in line:
                splt = line.split()
                population[int(splt[1])-1].append(float(splt[2]))

    return np.array(population).T

def read_dyn(dynout_f, natm):

    pos_all = []
    read_pos = False
    with open(dynout_f) as f:
        pos_i = []; pos_line = 0
        for line in f:

            # Read positions
            if read_pos:
                pos_line += 1

                pos_i.append([float(i) for i in line.split()[2:5]])
                
                # Stop reading and reset if all atoms are read
                if pos_line == natm:
                    # Save
                    pos_all.append(np.array(pos_i))
                    # Reset
                    read_pos = False
                    pos_line = 0
                    pos_i = [] #

            # Initiate reading
            if 'geometry' in line:
                read_pos = True

    return pos_all

def read_traj():
    
    geom_all = np.load('TEMP/traj_geom.npy')
    
    # Clean the duplicates after hopping
    v, c = np.unique(geom_all, return_counts=True, axis=0)
    
    dupl = v[c > 1]
    
    for dupl_i in dupl:
        indices = np.argwhere((dupl_i ==  geom_all).all(axis=2).all(axis=1))
        # If duplicates are consecutive, remove the duplicate
        if abs(indices[1]-indices[0]) == 1:
            geom_all = np.delete(geom_all, indices[1], axis=0)
    
    return geom_all

def read_NX(path,pos='TEMP'):
    
    # Record current directory and change directory into path
    cwd = os.getcwd()
    os.chdir(path)
    
    # Read energies
    en = np.loadtxt('RESULTS/en.dat')
    
    tprob_all = np.loadtxt('RESULTS/tprob',skiprows=1)
    tprob = tprob_all[:,3:]
    randi, substep, step = tprob_all[:,0], tprob_all[:,1], tprob_all[:,2]
    
    dynall = np.loadtxt('RESULTS/typeofdyn.log',usecols=[2,7])
    tim, pes = dynall[:,0], dynall[:,1]
    
    # Number of states
    nstat = en[:,1:-2].shape[1]
    
    # Read populations
    populations = read_population('RESULTS/sh.out',nstat)
    
    geom_f = 'geom'
    
    atom_f = []
    with open(geom_f,'r') as f:
        for line in f.readlines():
            splt = line.split()
            #sym, atomic no, xc, yc, zc, mass
            atom_f.append((splt[0], np.array(splt[2:5],dtype=np.float64)))            
            #atom_f.append((splt[0], [float(i) for i in splt[2:5]]))      
    
    natm = len(atom_f)
        
    if pos != 'TEMP':
        pos_all = read_dyn('RESULTS/dyn.out',natm)
    else:
        pos_all = read_traj()

    # Return to original directory
    os.chdir(cwd)
    
    return natm, nstat, en, [randi, substep, step, tprob], [tim, pes], populations, pos_all

def write_model(overlap, one_rdm, two_rdm):
    """
    Write the intermediate data that will be used for predictions, namely:
        - Overlap of training wavefunctions, S
        - 1-el reduced transition density matrices of training wavefunctions
        - 2-el reduced transition density matrices of training wavefunctions

    Args:
        overlap (ndarray)
        one_rdm (ndarray)
        two_rdm (ndarray)
    """
    
    path = 'sample/JOB_NAD'
    
    np.save(os.path.join(path,'overlap_final.npy'),overlap)
    np.save(os.path.join(path,'one_rdm_final.npy'),one_rdm)
    np.save(os.path.join(path,'two_rdm_final.npy'),two_rdm)
    
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

def remove_model(path):
    """
    Remove intermediate representation files from the path given IF they exist
    """
    ov_path = os.path.join(path,'overlap_final.npy')
    
    if os.path.isfile(ov_path):
        os.remove(ov_path)
        os.remove(os.path.join(path,'one_rdm_final.npy'))
        os.remove(os.path.join(path,'two_rdm_final.npy'))
    
    
def clean_traj():
    """
    Clean the memory intensive files after each trajectory finishes
    """
    existing_ind = [int(i.split('_')[-1].split('.')[0]) for i in glob.glob('ham_dist*')]
    
    if len(existing_ind) > 0:
        max_it = max(existing_ind)
        
        for nit in range(max_it-1):
            path_i = 'TRAJ_%i'%nit
            
            # Remove from TRAJ/DEBUG/TEMP
            remove_model(os.path.join(path_i,'DEBUG','TEMP'))
            
            # Remove from TRAJ/JOB_NAD
            remove_model(os.path.join(path_i,'JOB_NAD'))
            
            # Remove from TRAJ/TEMP/JOB*
            remove_model(os.path.join(path_i,'TEMP','JOB_NAD'))
            remove_model(os.path.join(path_i,'TEMP','JOB_AD'))
            remove_model(os.path.join(path_i,'TEMP'))
            
            # Remove from TRAJ/INFO_RESTART/JOB_NAD
            remove_model(os.path.join(path_i,'INFO_RESTART','JOB_NAD'))
    
##############################################################################
# ACTIVE LEARNING
##############################################################################

# Pseudocode

#1 Set initial tRDMs and overlaps for the first run (e.g. starting geometry)
#2 Run NX dynamics trajectory for n steps with timestep dt
### Save the trajectory info; geometries and energies and anything else?
#3 Find geometry with farthrest ham distance and add it to the representation
#4 Repeat 2-3 until convergence

# Convergence is achieved when the energies between consecutive iterations do 
# not change for more than a threshold (mHa) for at 2(?) iterations

# Details of step 2
#i 

# Function input
#a 

def converge_NAMD_traj(
        EVCont_obj,
        init_mol,
        steps=100,
        dt=0.1,
        nstat=3,
        nstatdyn=2,
        iseed=8,
        convergence_thresh=1.0e-3,
        nconv=2,
        max_iter=100,
        data_addition='weighted_highest_peak_ham',
        nx_path=None,
        run_command='sbatch $NX/moldyn.pl'
        ):
    """
    Converging eigenvector continuation training set for Newton-X nonadiabatic 
    dynamics trajectories. On-the-fly learning of which geometries to add by 
    converging the trajectory.
    
    Call and run from a separate directory that contains a directory 
    called 'sample' with sample NX input files. Otherwise, the calculation will be stuck
    
    Args:
        EVCont_obj: 
            The data structure for the eigenvector continuation.
        init_mol: 
            The initial molecule object.
        steps (int): 
            Number of MD simulation steps. Default is 100.
        dt (float): 
            Time step for the simulation. Default is 0.1 ns.
        nstat (int):
            Total number of states to be used in the NAMD simulation.
        nstatdyn (int):
            The state at which dynamics start from.
        iseed (int):
            Random seed of the NX trajectory. This achieves direct comparison.
            Choose a value > 1 as NX has different meaning for iseed = 0,1.
        convergence_thresh (float):
            Energy convergence threshold to terminate the training. Default is 1.0e-3.
        nconv (int):
            Number of consecutive iterations required for convergence
        max_iter (int):
            Calculation is stopped after max_iter iterations if convergence 
            is not achieved.
        data_addition (str):
            Criterion for adding new data points. Can be "farthest_point_ham" (default),
            in which case data is added based on electron integral difference,
            "farthest_point", in which case data is added based on the farthest point
            according to Euclidean distance, or "energy", in which case data is added
            based on the energy difference.
        nx_path (str):
            Path for the 'bin' folder of the installed Newton-X code. If not defined,
            the $NX environment variable will be expected to be predefined in the terminal.
            Otherwise, the calculation will crash.
        run_command (str):
            Terminal command to use for running NX. Based on the HPC, different
            commands may be required.
    
    Returns:
        1

    """
    
    # Set Newton-X path
    if nx_path is not None:
        os.system('export NX={}'.format(nx_path))
    
    # Check if it is a restart calculation or a new calculation
    #existing_ind = [int(i.split('_')[-1]) for i in glob.glob('TRAJ*')]
    existing_ind = [int(i.split('_')[-1].split('.')[0]) for i in glob.glob('ham_dist*')]
    
    # Current iteration of the convergence
    if len(existing_ind) > 0:
        nit = max(existing_ind) + 1
    else:
        nit = 0
        
    print('NAMD convergence - Starting iteration {}'.format(nit))
    
    # Update the model from the last iteration or initialize one if it doesn't exist
    if EVCont_obj.overlap is None:
        EVCont_obj.append_to_rdms(init_mol.copy())
        
        trn_geometries = [init_mol.atom_coords()]
        np.save('trn_geometries.npy', trn_geometries)
    else:
        # Read initial training geometries
        trn_geometries = np.load('trn_geometries.npy')
        
    # Save to sample/JOB_NAD directory
    write_model(EVCont_obj.overlap,
                EVCont_obj.one_rdm,
                EVCont_obj.two_rdm)
    
    ###########################################################################
    # Setup and run NAMD trajectory
    inp_par = [steps, dt, nstat, nstatdyn, iseed]
    run_trajectory(nit, inp_par, run_command)
    
    # Read output of current and previous trajectory
    out_n = read_NX('TRAJ_%i'%nit)
    trajectory = out_n[-1]
    
    # Check convergence
    # Write en_diff (or other convergence) to file
    # TODO 
    converged = False
    
    # Only check convergence after 0th iteration
    if nit > 0:
        out_prev = read_NX('TRAJ_%i'%(nit-1))
        
        en = out_n[2][:,1:nstat+1] # Energies of all states
        en_prev = out_prev[2][:,1:nstat+1]
        
        # If lens are different, e.g. when different commensurate timesteps are used
        no_en = en.shape[0]
        no_en_prev = en_prev.shape[0]
        if no_en > no_en_prev:
            en = en[::round(no_en/no_en_prev),:]
        
        elif no_en < no_en_prev:
            en_prev = en_prev[::round(no_en_prev/no_en),:]
        
        # Mean energy difference across all states
        en_diff = np.abs(en-en_prev).mean(axis=1)
        np.savetxt('en_diff_{}.txt'.format(nit), en_diff)
        
        print('Current max(en_diff) is {:.4f} Ha'.format(max(en_diff)))
        
        if max(en_diff) < convergence_thresh:
            converged = True
            
            # Check for previous iterations
            for i in range(max(1,nit-nconv),nit):
                # Check for the previous iteration as well
                en_diff_prev = np.loadtxt('en_diff_{}.txt'.format(i))
                
                if max(en_diff_prev) < convergence_thresh and converged:
                    converged = True
                else:
                    converged = False
                    
            if converged:
                print('NAMD trajectory is converged within specified threshold of {:.4f} Ha for {} consecutive iterations'.format(convergence_thresh,nconv))

        if nit >= max_iter:
            print('Convergence was NOT achieved within the specified limit of {} iterations'.format(max_iter))
            # Stop the calculation
            converged = True
            
    else:
        # Set a very high energy difference, implying very far away from
        # convergence, for later heuristics
        en_diff = np.array([100.])
    
    ###########################################################################
    if converged:
        return 1
    
    else:
    
        ######################################################################
        ##### SELECTION OF NEW TRAINING GEOMETRY
        ######################################################################
        
        # Compute hamiltonian distance to training set
        hamiltonian_distance_all = hamiltonian_similarity(init_mol, trajectory, trn_geometries)
        np.savetxt('ham_dist_{}.txt'.format(nit),hamiltonian_distance_all)
        
        # Find which new geometry to add to the training set
        if data_addition == "farthest_point_ham":
            
            # Find the index of the geometry with maximum H_dist
            addgeom_ind = np.argmax(hamiltonian_distance_all)
            
        elif data_addition == "first_peak_ham":
            # Disregards peaks with hamdist less than this threshold
            # (make this a parameter later on, also depends on norb)
            threshold = 0.01 
            
            # If no peak is found, choose the index with largest ham distance
            addgeom_ind = np.argmax(hamiltonian_distance_all)
                
            # Find all peaks
            peaks = find_peaks(hamiltonian_distance_all)[0]
            
            # If peaks above a threshold exist, choose that over farthest
            if len(peaks) > 0:
                ind_above_thr = np.where(hamiltonian_distance_all[peaks] > threshold)
                if len(ind_above_thr[0]) > 0:
                    addgeom_ind = peaks[ind_above_thr][0]
               
        elif data_addition == "weighted_highest_peak_ham":
            # Disregards peaks with hamdist less than this threshold
            # (make this a parameter later on, also depends on norb)
            threshold = 0.01
            
            # Exponent of the time penalty function
            exponent = 1.

            # If no peak is found, choose the index with largest ham distance
            addgeom_ind = np.argmax(hamiltonian_distance_all)
                
            # Find all peaks
            peaks = find_peaks(hamiltonian_distance_all)[0]
            
            # Add the max point to the peaks as a possible selection geometry
            if addgeom_ind not in peaks:
                peaks = np.append(peaks, addgeom_ind)
                
            # If peaks above a threshold exist, choose that over farthest
            ind_above_thr = np.where((hamiltonian_distance_all[peaks] > threshold) & (peaks > 0)) 
            
            if len(ind_above_thr[0]) > 0:
                # Peaks above threshold
                peaks_above_thr = peaks[ind_above_thr]
                
                # penalty function ranging from 0 (favourable) to 1 (unfavourable)
                penalty = (peaks_above_thr/len(hamiltonian_distance_all))**exponent
                
                step_weighted_hamdist = hamiltonian_distance_all[peaks_above_thr]/penalty
                
                addgeom_ind = peaks_above_thr[np.argmax(step_weighted_hamdist)]
                
        elif data_addition == "variable_weight_peak_ham":
            # Disregards peaks with hamdist less than this threshold
            # (make this a parameter later on, also depends on norb)
            threshold = 0.01
            
            # Exponent of the time penalty function 
            # (0 - chooses max, -->inf chooses 1st peak)
            #exponent = 2.
            scaling = 0.01
            scaled_exp = scaling * en_diff.max()/convergence_thresh
            exponent = min(max(scaled_exp, 0.), 5.) # Limit between 0 and 5
                
            # If no peak is found, choose the index with largest ham distance
            addgeom_ind = np.argmax(hamiltonian_distance_all)
                
            # Find all peaks
            peaks = find_peaks(hamiltonian_distance_all)[0]
            
            # Add the max point to the peaks as a possible selection geometry
            if addgeom_ind not in peaks:
                peaks = np.append(peaks, addgeom_ind)
                
            # If peaks above a threshold exist, choose that over farthest
            ind_above_thr = np.where((hamiltonian_distance_all[peaks] > threshold) & (peaks > 0)) 
            
            if len(ind_above_thr[0]) > 0:
                # Peaks above threshold
                peaks_above_thr = peaks[ind_above_thr]
                
                # penalty function ranging from 0 (favourable) to 1 (unfavourable)
                penalty = (peaks_above_thr/len(hamiltonian_distance_all))**exponent
                
                step_weighted_hamdist = hamiltonian_distance_all[peaks_above_thr]/penalty
                
                addgeom_ind = peaks_above_thr[np.argmax(step_weighted_hamdist)]
                
        else:
            print('The data_addition method {} is not implemented.'.format(data_addition))
            sys.exit()
            
            
        ######################################################################
        ##### AFTER SELECTION
        ######################################################################
        # Add the new geometry to the training set
        new_geom = trajectory[addgeom_ind]
        new_trn_geometries = np.concatenate((trn_geometries,[new_geom]))
        
        # Write
        np.save('trn_geometries.npy',new_trn_geometries)
        
        # Add to continuation
        EVCont_obj.append_to_rdms(init_mol.copy().set_geom_(new_geom))
        
        # Go to next iteration
        converge_NAMD_traj(
                EVCont_obj,
                init_mol,
                steps=steps,
                dt=dt,
                nstat=nstat,
                nstatdyn=nstatdyn,
                iseed=iseed,
                convergence_thresh=convergence_thresh,
                nconv=nconv,
                max_iter=max_iter,
                data_addition=data_addition,
                nx_path=nx_path,
                run_command=run_command
                )
        

def run_trajectory(traj_ind, inp_par, run_command):
    """
    Run a Newton-X calculation with sample input files from the 'sample' directory
    """
    # Copy sample files into a separate directory - named TRAJ_ind
    new_path = 'TRAJ_%i'%traj_ind
    os.system('cp -r sample %s'%new_path)
    
    # Clean scratch data from other trajectories
    os.system('sleep 10')
    clean_traj()
    
    # Record current working directory and change it to TRAJ_ind
    cwd = os.getcwd()
    os.chdir(new_path)
    
    # Setup input files
    # TODO - for now, they remain the same as sample
    
    # Run
    os.system(run_command)
    
    # Wait until calculation finishes or crashes
    while True:
        os.system('sleep 10')

        status = check_status()
        
        if status == 'Error':
            print('NX has crushed - check the end of output at DEBUG/runnx.error')
            sys.exit()
        elif status == 'Success':
            break
        
        # TODO - check ham distance, and if a peak is found; kill the job,
        # add the geometry, and force it to the next iteration
        # Need to check if a shorter initial trajectories would cause consistency
        # issues
            
    # Return to original directory for next iteration
    os.chdir(cwd)
    
def check_status():
    """
    Check the status Newton-X calculation
    
    Returns:
        
    """
    # Read last line
    try:
        line = str(subprocess.check_output(['tail', '-1', 'RESULTS/nx.log']))
        line2 = str(subprocess.check_output(['tail', '-2', 'RESULTS/nx.log']))
    except:
        # If file hasn't been created yet
        return 'Starting'

    # Return the status
    if 'DEBUG/runnx.error' in line2:
        return 'Error'
    elif 'NEWTON-X ends here' in line:
        return 'Success'
    else:
        return 'Running'
    
def hamiltonian_similarity(init_mol, trajectory, trn_geometries):
    """
    Compute the minimum Hamiltonian distance of a trajectory to a set
    of training geometries
    """
    # Initialize hamiltonians of the training set
    h1_trn = np.zeros((len(trn_geometries), init_mol.nao, init_mol.nao))
    h2_trn = np.zeros(
        (
            len(trn_geometries),
            init_mol.nao,
            init_mol.nao,
            init_mol.nao,
            init_mol.nao,
        )
    )
    
    # Compute 1- and 2-electron integrals for all training geometries
    for j, trn_geom in enumerate(trn_geometries):
        mol = init_mol.copy().set_geom_(trn_geom)
        h1, h2 = get_integrals(mol, get_basis(mol))
        h1_trn[j] = h1
        h2_trn[j] = h2
    
    # Compute min Hamiltonian distance to the training geometries for the new traj
    min_dist_l = []
    for j, geometry in enumerate(trajectory):
        mol = init_mol.copy().set_geom_(geometry)
        h1, h2 = get_integrals(mol, get_basis(mol))

        distance = np.sum(
            abs(h1 - h1_trn) ** 2, axis=(-1, -2)
        ) + 0.5 * np.sum(abs(h2 - h2_trn) ** 2, axis=(-1, -2, -3, -4))
        min_dist = np.min(distance)
        min_dist_l += [min_dist]
        
    return np.array(min_dist_l)


if __name__ == '__main__':
    print('yes')





