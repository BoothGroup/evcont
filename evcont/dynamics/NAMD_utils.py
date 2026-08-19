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

from evcont.electron_integral_utils import get_integrals, get_basis
from evcont.dynamics.active_learning import (
    hamiltonian_distance,
    hamiltonian_similarity,
    hamiltonian_similarity_argmin,
    select_active_learning_geometry,
)

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

def write_model(overlap, one_rdm, two_rdm=None, vecs_lowrank=None, diagonal_lr=None, model_path=None):
    """
    Write the intermediate data that will be used for predictions, namely:
        - Overlap of training wavefunctions, S
        - 1-el reduced transition density matrices of training wavefunctions
        - 2-el reduced transition density matrices of training wavefunctions
          OR low-rank vectors if using low-rank approximation

    Args:
        overlap (ndarray)
        one_rdm (ndarray)
        two_rdm (ndarray, optional): Full 2-RDM if not using low-rank
        vecs_lowrank (dict, optional): Low-rank vectors if using low-rank
        diagonal_lr (ndarray, optional): Diagonal components if using low-rank
    """
    import pickle

    # Allow caller to specify a custom model path; fall back to original location if not provided
    if model_path is None:
        model_path = 'sample/JOB_NAD'

    os.makedirs(model_path, exist_ok=True)

    np.save(os.path.join(model_path,'overlap_final.npy'),overlap)
    np.save(os.path.join(model_path,'one_rdm_final.npy'),one_rdm)

    if two_rdm is not None:
        # Full 2-RDM case
        np.save(os.path.join(model_path,'two_rdm_final.npy'),two_rdm)

    if vecs_lowrank is not None:
        # Low-rank case - save as pickle
        with open(os.path.join(model_path,'lowrank_vecs.pkl'), 'wb') as f:
            pickle.dump(vecs_lowrank, f, protocol=pickle.HIGHEST_PROTOCOL)

    if diagonal_lr is not None:
        np.save(os.path.join(model_path,'diagonal_lr.npy'), diagonal_lr)
    
def read_model(path):
    """
    Read the intermediate data that will be used for predictions, namely:
        - Overlap of training wavefunctions, S
        - 1-el reduced transition density matrices of training wavefunctions
        - 2-el reduced transition density matrices of training wavefunctions,
          either as a full tensor (two_rdm_final.npy) or low-rank vectors (lowrank_vecs.pkl)

    Args:
        path (str): path to the model files
        
    Returns:
        overlap (ndarray)
        one_rdm (ndarray)
        two_rdm (ndarray or dict): Full 2-RDM or low-rank vectors
        diagonals (ndarray or None): Diagonals if they exist, None otherwise
    """
    import pickle
    
    overlap = np.load(os.path.join(path,'overlap_final.npy'))
    one_rdm = np.load(os.path.join(path,'one_rdm_final.npy'))
    
    two_rdm_npy = os.path.join(path, 'two_rdm_final.npy')
    two_rdm_pkl = os.path.join(path, 'lowrank_vecs.pkl')

    if os.path.exists(two_rdm_npy):
        two_rdm = np.load(two_rdm_npy)
    elif os.path.exists(two_rdm_pkl):
        with open(two_rdm_pkl, 'rb') as f:
            two_rdm = pickle.load(f)
    else:
        raise FileNotFoundError("Neither 'two_rdm_final.npy' nor 'lowrank_vecs.pkl' was found in the specified path.")
    
    # Try to load diagonals (optional, mainly for low-rank models)
    diag_file = os.path.join(path, 'diagonal_lr.npy')
    diagonals = None
    if os.path.exists(diag_file):
        diagonals = np.load(diag_file, allow_pickle=True)
    
    return overlap, one_rdm, two_rdm, diagonals

def remove_model(path):
    """
    Remove intermediate representation files from the path given IF they exist
    """
    ov_path = os.path.join(path,'overlap_final.npy')
    
    if os.path.isfile(ov_path):
        os.remove(ov_path)
        os.remove(os.path.join(path,'one_rdm_final.npy'))
        
        # Remove two_rdm if exists
        two_rdm_path = os.path.join(path,'two_rdm_final.npy')
        if os.path.isfile(two_rdm_path):
            os.remove(two_rdm_path)
        
        # Remove low-rank files if they exist
        lowrank_path = os.path.join(path,'lowrank_vecs.pkl')
        if os.path.isfile(lowrank_path):
            os.remove(lowrank_path)
        
        diagonal_path = os.path.join(path,'diagonal_lr.npy')
        if os.path.isfile(diagonal_path):
            os.remove(diagonal_path)
    
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
        reconverge_from_closest_hdist=False,
        append_as_HPC_job=False,
        run_command='sbatch $NX/moldyn.pl',
        run_append_command='qsub append_states.sh',
        compute_hamdist_during_traj=False,
        compute_hamdist_as_HPC_job=False,
        run_hamdist_command='qsub compute_hamdist.sh',
        solver='CAS'
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
        reconverge_from_closest_hdist (bool):
            Whether to reconverge from the closest geometry based on Hamiltonian distance. Default is False.
        run_command (str):
            Terminal command to use for running NX. Based on the HPC, different
            commands may be required.
        run_append_command (str):
            Terminal command to use for appending new states to the continuation object.
            Based on the HPC, different commands may be required.
        compute_hamdist_during_traj (bool):
            Whether to compute Hamiltonian distances locally while the trajectory is running.
        compute_hamdist_as_HPC_job (bool):
            Whether to offload Hamiltonian distance computation to a separate HPC job.
        run_hamdist_command (str):
            Command used to submit/run the Hamiltonian distance job.
        solver (str):
            The type of solver used for the continuation object. Used to determine continuation object type
            when appending new states as a HPC job. Default is 'CAS'.
    
    Returns:
        1

    """
    
    # Set Newton-X path
    if nx_path is not None:
        os.environ['NX'] = nx_path

    # Check if it is a restart calculation or a new calculation
    existing_ind = [int(i.split('_')[-1].split('.')[0]) for i in glob.glob('ham_dist*')]

    # Current iteration of the convergence
    if len(existing_ind) > 0:
        nit = max(existing_ind) + 1
    else:
        nit = 0

    # Setup models directory - if it doesn't exist
    OBJECT_CAN_BE_SAVED = False
    if hasattr(EVCont_obj, 'load'):
        OBJECT_CAN_BE_SAVED = True

        if not os.path.exists('iterative-models'):
            os.mkdir('iterative-models')

        CONT_OBJ_FILENAME = f'iterative-models/continuation_object-{nit}.pkl'

    print('NAMD convergence - Starting iteration {}'.format(nit))

    # Update the model from the last iteration or initialize one if it doesn't exist
    if EVCont_obj.overlap is None:

        if append_as_HPC_job and OBJECT_CAN_BE_SAVED:
            # Create an empty object and append the first geometry as an HPC job
            EMPTY_OBJ_FILENAME = f'iterative-models/continuation_object-empty.pkl'
            EVCont_obj.save(EMPTY_OBJ_FILENAME)

            EVCont_obj = run_append_states(init_mol.copy(), EMPTY_OBJ_FILENAME, CONT_OBJ_FILENAME, solver, run_append_command, quantel_tag='ref')
            trn_geometries = [init_mol.atom_coords()]
            np.save('trn_geometries.npy', trn_geometries)

        else:
            EVCont_obj.append_to_rdms(init_mol.copy())
            trn_geometries = [init_mol.atom_coords()]
            np.save('trn_geometries.npy', trn_geometries)

            # Optionally save the continuation object if possible
            if OBJECT_CAN_BE_SAVED:
                try:
                    print(f"Saving continuation object to {CONT_OBJ_FILENAME}")
                    EVCont_obj.save(CONT_OBJ_FILENAME)
                except Exception as e:
                    print(f"Warning: Could not save continuation object: {e}")

    else:
        # Read initial training geometries
        trn_geometries = np.load('trn_geometries.npy')

    # Write model to iteration-specific directory and point evcont.in to it
    model_dir = os.path.join('iterative-models', f'model_{nit}')  # user-requested path pattern
    abs_model_dir = os.path.abspath(model_dir)
    os.makedirs(model_dir, exist_ok=True)

    # Update sample/JOB_NAD/evcont.in to reference this model directory via trdm_path
    evcont_in_path = os.path.join('sample', 'JOB_NAD', 'evcont.in')
    try:
        if os.path.isfile(evcont_in_path):
            with open(evcont_in_path, 'r') as f:
                lines = f.readlines()
            found = False
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith('trdm_path') or stripped.startswith('#trdm_path'):
                    lines[i] = f'trdm_path = {abs_model_dir}\n'
                    found = True
            if not found:
                # Prepend if no existing trdm_path directive
                lines.insert(0, f'trdm_path = {abs_model_dir}\n')
            with open(evcont_in_path, 'w') as f:
                f.writelines(lines)
        else:
            print(f"Warning: evcont.in not found at {evcont_in_path}; cannot set trdm_path.")
    except Exception as e:
        print(f"Warning: could not modify evcont.in to set trdm_path: {e}")

    if EVCont_obj.lowrank:
        EVCont_obj.vectorize_lowrank()
        write_model(EVCont_obj.overlap,
                    EVCont_obj.one_rdm,
                    two_rdm=None,
                    vecs_lowrank=EVCont_obj.lowrank_vectorized,
                    diagonal_lr=EVCont_obj.diagonal_vectorized,
                    model_path=model_dir)
    else:
        write_model(EVCont_obj.overlap,
                    EVCont_obj.one_rdm,
                    two_rdm=EVCont_obj.two_rdm,
                    model_path=model_dir)

    ###########################################################################
    # Setup and run NAMD trajectory
    inp_par = [steps, dt, nstat, nstatdyn, iseed]
    run_trajectory(nit, inp_par, run_command, 
                   init_mol=init_mol, 
                   trn_geometries=trn_geometries,
                   compute_hamdist_during_traj=compute_hamdist_during_traj)
    
    # Read output of current and previous trajectory
    out_n = read_NX('TRAJ_%i'%nit,pos='dyn')
    trajectory = out_n[-1]
    ########################################################################### 
    # Check convergence
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
        hamdist_file = f'ham_dist_{nit}.txt'
        
        # Check if distances were already computed during trajectory
        if os.path.isfile(hamdist_file):
            print(f"Loading Hamiltonian distances computed during trajectory from {hamdist_file}")
            hamiltonian_distance_all = np.loadtxt(hamdist_file)
        
        # Compute via HPC job submission
        elif compute_hamdist_as_HPC_job and append_as_HPC_job:
            # Use in-memory trajectory instead of relying on TEMP/traj_geom.npy
            hamiltonian_distance_all = run_hamdist_job(
                nit,
                init_mol,
                run_hamdist_command,
                trajectory
            )

        # Compute locally after trajectory completes
        else:
            if compute_hamdist_as_HPC_job and not append_as_HPC_job:
                print("Warning: When compute_hamdist_as_HPC_job is True, append_as_HPC_job must also be True. Continuing to local hamdist computation.")
            # Compute locally (also produce argmin indices for reuse)
            hamiltonian_distance_all, argmins_all = hamiltonian_similarity_argmin(init_mol, trajectory, trn_geometries)
            np.savetxt(hamdist_file, hamiltonian_distance_all)
            try:
                np.savetxt(f"ham_argmin_{nit}.txt", argmins_all, fmt='%d')
            except Exception as e:
                print(f"Warning: could not write argmin indices ham_argmin_{nit}.txt: {e}")

        ######################################################################
        ##### SELECTION OF NEW TRAINING GEOMETRY
        ######################################################################
        addgeom_ind = select_active_learning_geometry(hamiltonian_distance_all, data_addition, en_diff, convergence_thresh)    
            
        ######################################################################
        ##### AFTER SELECTION
        ######################################################################
        # Add the new geometry to the training set
        new_geom = trajectory[addgeom_ind]
        new_trn_geometries = np.concatenate((trn_geometries,[new_geom]))
        mol_new = init_mol.copy().set_geom_(new_geom)

        # Write
        np.save('trn_geometries.npy',new_trn_geometries)
        
        ### Add to continuation
        if reconverge_from_closest_hdist:
            if hasattr(EVCont_obj, 'software') and EVCont_obj.software == 'quantel':
                # Prefer using precomputed closest training indices to avoid recomputation
                argmin_file = f"ham_argmin_{nit}.txt"
                if os.path.isfile(argmin_file):
                    argmins_all = np.loadtxt(argmin_file, dtype=int)
                    closest_ind = int(argmins_all[addgeom_ind])+1
                    closest_geom_tag = f'geom{closest_ind}'
                else:
                    # No argmin file; fallback to original integral-based computation
                    oei_new, tei_new = get_integrals(mol_new, get_basis(mol_new))
                    files = glob.glob(os.path.join(EVCont_obj.quantel_path, 'geom*'))
                    oei_trn = []
                    tei_trn = []
                    for f in files:
                        ind = int(f.split('geom')[-1])
                        oei_trn.append(np.loadtxt(os.path.join(f,'oei.dat')))
                        tei_trn.append(np.load(os.path.join(f,'tei.npy')))
                    hamiltonian_distance_to_new = hamiltonian_distance(
                        oei_new,
                        tei_new,
                        np.array(oei_trn),
                        np.array(tei_trn)
                    )
                    closest_ind = np.argmin(hamiltonian_distance_to_new)
                    closest_geom_tag = f'geom{files[closest_ind].split("geom")[-1]}'

                print('Re-converging from geometry closest in ham distance to the new geometry ({})'.format(closest_geom_tag))
                if append_as_HPC_job and OBJECT_CAN_BE_SAVED:
                    NEW_OBJ_FILENAME = f'iterative-models/continuation_object-{nit+1}.pkl'
                    EVCont_obj = run_append_states(mol_new, CONT_OBJ_FILENAME, NEW_OBJ_FILENAME, solver, run_append_command, quantel_tag=closest_geom_tag)
                else:
                    EVCont_obj.append_to_rdms(mol_new, quantel_tag=closest_geom_tag)

            else:
                print('Re-converging from closest ham distance is only implemented for Quantel software.')
                sys.exit()
        else:
            if append_as_HPC_job and OBJECT_CAN_BE_SAVED:
                NEW_OBJ_FILENAME = f'iterative-models/continuation_object-{nit+1}.pkl'
                EVCont_obj = run_append_states(mol_new, CONT_OBJ_FILENAME, NEW_OBJ_FILENAME, solver, run_append_command, quantel_tag='ref')
            else:
                EVCont_obj.append_to_rdms(mol_new)
        
        # Save the continuation object if possible
        if OBJECT_CAN_BE_SAVED and not append_as_HPC_job:
            CONT_OBJ_FILENAME = f'iterative-models/continuation_object-{nit+1}.pkl'
            try:
                print(f"Saving continuation object to {CONT_OBJ_FILENAME}")
                EVCont_obj.save(CONT_OBJ_FILENAME)
            except Exception as e:
                print(f"Warning: Could not save continuation object: {e}")

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
            reconverge_from_closest_hdist=reconverge_from_closest_hdist,
            append_as_HPC_job=append_as_HPC_job,
            run_command=run_command,
            run_append_command=run_append_command,
            compute_hamdist_during_traj=compute_hamdist_during_traj,
            compute_hamdist_as_HPC_job=compute_hamdist_as_HPC_job,
            run_hamdist_command=run_hamdist_command,
            solver=solver
        )
        
def run_append_states(mol, cont_obj_path, newcont_obj_path, solver, run_append_command, quantel_tag='None'):
    """
    Run a job to append new geometries to the continuation object
    """

    # Save geometry
    # Check if file exists and change name if necessary
    geomfname = 'iterative-models/geom_0.xyz'    
    count = 0
    while os.path.isfile(geomfname):
        count += 1
        geomfname = f'iterative-models/geom_{count}.xyz'
    mol.tofile(geomfname)

    # Submit job with the correct arguments
    os.system(f'{run_append_command} {cont_obj_path} {newcont_obj_path} {solver} {geomfname} {mol.basis} {quantel_tag}')

    # Check until the job finishes
    poll_seconds = 20  # minimal polling interval
    print(f"Waiting for appended continuation object file: {newcont_obj_path}")
    while not os.path.isfile(newcont_obj_path):
        os.system(f'sleep {poll_seconds}')
    print(f"Detected file {newcont_obj_path}. Loading updated continuation object.")

    # Read in the appended continuation object
    if solver == 'CAS':
        try:
            from evcont.cas.CASCI_EVCont import CAS_EVCont_obj
            cont_obj = CAS_EVCont_obj.load(newcont_obj_path)
        except Exception as e:
            print(f"Error loading appended continuation object: {e}")
            sys.exit(1)

    return cont_obj

def run_hamdist_job(nit, init_mol, run_hamdist_command, trajectory):
    """Submit a job to compute Hamiltonian distances for a trajectory.

    Instead of relying on an existing TEMP/traj_geom.npy file, this function
    receives the in-memory `trajectory` array, writes it to
    TRAJ_<nit>/traj_geom.npy, and passes that path to the job script.

    Arguments passed to the external script (compute_hamdist.sh):
        geom_file basis trajectory_npy trn_geometries_npy output_file [cache_prefix]

    Parameters:
        nit (int): iteration / trajectory index
        init_mol: Mole object (used for basis retrieval)
        run_hamdist_command (str): submission command (e.g. 'qsub compute_hamdist.sh')
        trajectory (ndarray): geometries from the just-completed NX trajectory
    Returns:
        distances (ndarray): Hamiltonian distances (loaded from ham_dist_<nit>.txt)
    """

    # Any geometry file to initialize the molecule - should already exist
    geom_file = 'iterative-models/geom_0.xyz'

    # Write provided trajectory to a new file at top-level of TRAJ_<nit>
    traj_dir = f'TRAJ_{nit}'
    if not os.path.isdir(traj_dir):
        print(f"Error: trajectory directory {traj_dir} not found")
        sys.exit(1)
    traj_fname = os.path.join(traj_dir, 'traj_geom.npy')
    try:
        np.save(traj_fname, trajectory)
    except Exception as e:
        print(f"Error writing trajectory to {traj_fname}: {e}")
        sys.exit(1)

    # Training geometries and hamdist output file
    trn_fname = 'trn_geometries.npy'
    hamdist_outfile = f'ham_dist_{nit}.txt'

    basis = init_mol.basis
    cache_prefix = 'training_integrals'

    # Build command with positional arguments (compatible with HPC script)
    cmd = f"{run_hamdist_command} {geom_file} {basis} {traj_fname} {trn_fname} {hamdist_outfile} {cache_prefix}"

    print(f"Submitting Hamiltonian distance job: {cmd}")
    os.system(cmd)

    # Poll for output files
    poll_seconds = 20
    print(f"Waiting for Hamiltonian distance output file: {hamdist_outfile}")
    while not os.path.isfile(hamdist_outfile):
        os.system(f'sleep {poll_seconds}')

    # Also wait (with a timeout) for the argmin file written by the job script
    argmin_outfile = f'ham_argmin_{nit}.txt'
    print(f"Waiting for closest training indices file: {argmin_outfile}")
    max_polls = 30  # ~10 minutes
    polls = 0
    while not os.path.isfile(argmin_outfile) and polls < max_polls:
        os.system(f'sleep {poll_seconds}')
        polls += 1
    if not os.path.isfile(argmin_outfile):
        print(f"Warning: {argmin_outfile} not detected after waiting. Will proceed without it and fall back later if needed.")

    try:
        distances = np.loadtxt(hamdist_outfile)
        print(f"Loaded Hamiltonian distances from {hamdist_outfile}")
    except Exception as e:
        print(f"Error reading Hamiltonian distance output {hamdist_outfile}: {e}")
        sys.exit(1)
    return distances

def run_trajectory(traj_ind, inp_par, run_command, init_mol=None, trn_geometries=None, compute_hamdist_during_traj=False):
    """
    Run a Newton-X calculation with sample input files from the 'sample' directory.
    
    Optionally compute Hamiltonian distances on-the-fly during the trajectory.
    
    Args:
        traj_ind (int): Trajectory index
        inp_par (list): Input parameters [steps, dt, nstat, nstatdyn, iseed]
        run_command (str): Command to run NX
        init_mol (Mole, optional): Initial molecule object for hamdist computation
        trn_geometries (ndarray, optional): Training geometries for hamdist computation
        compute_hamdist_during_traj (bool): Whether to compute hamdist during trajectory
    """
    # Copy sample files into a separate directory - named TRAJ_ind
    new_path = 'TRAJ_%i'%traj_ind
    os.system('cp -r sample %s'%new_path)
    
    # Clean scratch data from other trajectories
    os.system('sleep 10')
    clean_traj()

    print(run_command)
    
    # Record current working directory and change it to TRAJ_ind
    cwd = os.getcwd()
    os.chdir(new_path)
    
    # Check if the calculation has already finished
    status = check_status()
    if status == 'Success':
        print(f"Trajectory {traj_ind} has already finished. Skipping run.")
        os.chdir(cwd)
        return
    
    # Setup input files
    # TODO - for now, they remain the same as sample
    
    # Run
    os.system(run_command)
    os.system('sleep 100')
    
    # Wait until calculation finishes or crashes
    # Optionally compute Hamiltonian distances during the trajectory
    hamdist_outfile = None
    prev_traj_length = 0
    hamiltonian_distances = []
    
    if compute_hamdist_during_traj and init_mol is not None and trn_geometries is not None:
        hamdist_outfile = os.path.join(cwd, f'ham_dist_{traj_ind}.txt')
        print(f"Will compute Hamiltonian distances on-the-fly and save to {hamdist_outfile}")
    
    while True:
        os.system('sleep 100')

        status = check_status()
        
        if status == 'Error':
            print('NX has crushed - check the end of output at DEBUG/runnx.error')
            sys.exit()
        elif status == 'Success':
            break
        
        # Compute Hamiltonian distances for new trajectory points
        if compute_hamdist_during_traj and status == 'Running':
            traj_file = 'TEMP/traj_geom.npy'
            if os.path.isfile(traj_file):
                try:
                    current_traj = np.load(traj_file)
                    current_length = len(current_traj)
                    
                    # Only compute if new geometries have been added
                    if current_length > prev_traj_length:
                        print(f"Computing Hamiltonian distances for new geometries (total: {current_length})...")
                        
                        # Compute distances for new geometries only
                        new_geoms = current_traj[prev_traj_length:current_length]
                        new_distances, _ = hamiltonian_similarity(
                            init_mol, new_geoms, trn_geometries
                        )
                        hamiltonian_distances.extend(new_distances.tolist())
                        
                        # Save updated distances to file
                        np.savetxt(hamdist_outfile, np.array(hamiltonian_distances))
                        print(f"  Computed distances for geometries {prev_traj_length} to {current_length-1}")
                        
                        prev_traj_length = current_length
                        
                except Exception as e:
                    print(f"Warning: Could not compute Hamiltonian distances during trajectory: {e}")

        # TODO: Add an early exit condition for an early peak detection and killing the trajectory 
        # e.g. if a peak in hamdist is detected, and that peak is higher than previous hamdist peaks
        # Need to add a function to kill the NX job; copy bits of addgeom_ind from convergence loop; etc.
        # Need to iteratively check for convergence as well since if max(en_diff) < convergence_thresh early on,
        # the trajectory needs to continue.
            
    # Final computation if any geometries were missed
    if compute_hamdist_during_traj and hamdist_outfile is not None:
        traj_file = 'TEMP/traj_geom.npy'
        if os.path.isfile(traj_file):
            try:
                final_traj = np.load(traj_file)
                final_length = len(final_traj)
                
                if final_length > prev_traj_length:
                    print(f"Computing final Hamiltonian distances (total: {final_length})...")
                    new_geoms = final_traj[prev_traj_length:final_length]
                    new_distances, _ = hamiltonian_similarity(
                        init_mol, new_geoms, trn_geometries
                    )
                    hamiltonian_distances.extend(new_distances.tolist())
                    np.savetxt(hamdist_outfile, np.array(hamiltonian_distances))
                    print(f"  Final distances computed for geometries {prev_traj_length} to {final_length-1}")
            except Exception as e:
                print(f"Warning: Could not compute final Hamiltonian distances: {e}")
            
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
    
if __name__ == '__main__':
    print('yes')



