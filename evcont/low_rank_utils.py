#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 17:13:27 2025

Low-rank decomposition of 2-body (transition) reduced density matrices

Mixed decomposition:
    Joint ED: Joint eigenvalue decomposition where each low-rank vector 
              contribute to both Coulomb and exchange channels. (Hermitian)
    Coulomb SVD: SVD in the Coulomb grouping of the state indices (non-Hermitian)

Function here include:
    - Static and dynamic truncation of the decomposition based on 
      eval magnitude/Hamiltonian error
    - Subspace Hamiltonian construction from low-rank vectors
    - Vectorizing of the low-rank vectors for fast inference
    - Computation of JK and JK grad builds from low-rank vectors
    - Gradient inference from the low-rank representation
    
Note: Not compatible with complex RDMs as it stands; e.g. vectors from SVD are
assumed to be real-valued. Can be changed in the future if necessary.

@author: Kemal Atalar
"""

import multiprocessing as mp
mp.set_start_method("fork", force=True)

from multiprocessing import Process, Pipe, cpu_count

import numpy as np
import sys
import itertools
import time
import math
import os

import scipy
from scipy.linalg import eigh, svd
from scipy.sparse.linalg import eigsh, svds

import pyscf
from pyscf import scf, gto, ao2mo, fci, lib, df

from pyscf.df.grad.rhf import get_jk as get_jk_grad_df
from pyscf.df.grad.rhf import Gradients as mf_grad_df

from pyscf.grad.rhf import get_jk as get_jk_grad_nodf
from pyscf.grad.rhf import Gradients as mf_grad_nodf

from evcont.electron_integral_utils import get_loewdin_trafo, get_integrals, get_df_integrals, get_basis
from evcont.logging_utils import logger, log_time, timeit

########################################################################
def _eigsh_worker(mat, k, conn, which='LM',use_svd=False):
    try:
        if not use_svd:
            conn.send(eigsh(mat, k=k, which=which))
        else:
            conn.send(svds(mat, k=k, which=which))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()

def try_iterative_diag(mat, k, which='LM', use_svd=False, max_time=100000):
    parent_conn, child_conn = Pipe()
    p = Process(target=_eigsh_worker, args=(mat, k, child_conn, which, use_svd))
    p.start()
    p.join(timeout=max_time)

    if p.is_alive():
        p.terminate()
        p.join()
        print("[timeout] Falling back to full diagonalization...")

    if parent_conn.poll():
        result = parent_conn.recv()

        if isinstance(result, tuple) and isinstance(result[0], str) and result[0] == "error":
            print(f"[error] {result[1]}")
        else:
            print("[iterative] Success")
            return result

    # Full fallback
    print("[full] Running full eigh/svds...")
    if not use_svd:
        return eigh(mat)
    else:
        return svd(mat)
    
#########
from joblib import Parallel, delayed
from pyscf import lib

@timeit
def parallel_get_jk(dm_array, jk_func, n_jobs=None, chunk_size=None, parallel=False, **jk_kwargs):
    """
    Parallelized wrapper for PySCF get_jk-like functions with automatic core detection.

    Parameters
    ----------
    dm_array : np.ndarray
        Stacked density matrices (or low-rank vectors in our case), shape (n_dm, n_orb, n_orb)
    jk_func : callable
        Function that computes get_jk(dm, **kwargs) and returns (vj, vk)
    n_jobs : int or None
        Number of parallel workers. If None, auto-detect CPU cores
    chunk_size : int or None
        Number of DMs per chunk. If None, splits evenly across n_jobs
    parallel: bool
        Whether to parallelise over low-rank vectors (for testing)
    **jk_kwargs : dict
        Additional kwargs passed to jk_func (hermi, with_j, with_k, etc.)

    Returns
    -------
    vj_combined, vk_combined : np.ndarray or None
        Concatenated results across all chunks. If with_j or with_k is False, 
        the corresponding output may be None.
    """
    dm_array = np.ascontiguousarray(dm_array)
    n_dm = dm_array.shape[0]

    # Auto-detect cores if not provided
    if n_jobs is None and parallel:
        n_jobs = cpu_count()
        #n_jobs = int(os.environ.get('OMP_NUM_THREADS'))
        #print(n_jobs, 'processes')
            
    if not parallel or n_jobs < 2:
        return jk_func(dm=dm_array, **jk_kwargs)
    
    else:
        # Determine chunking
        if chunk_size is None:
            chunk_size = max(n_dm // n_jobs, 1)
    
        dm_chunks = [dm_array[i:i+chunk_size] for i in range(0, n_dm, chunk_size)]
    
        # Avoid OpenMP oversubscription in PySCF
        original_threads = lib.num_threads()
        #print(original_threads)
        lib.num_threads(1)
    
        try:
            # Worker function
            def _compute_chunk(dm_chunk):
                return jk_func(dm=dm_chunk, **jk_kwargs)
        
            # Run chunks in parallel
            results = Parallel(n_jobs=n_jobs)(
                delayed(_compute_chunk)(chunk) for chunk in dm_chunks
            )

            # Combine results
            vj_list = [r[0] for r in results if r[0] is not None]
            vk_list = [r[1] for r in results if r[1] is not None]
        
            vj_combined = np.concatenate(vj_list, axis=0) if vj_list else None
            vk_combined = np.concatenate(vk_list, axis=0) if vk_list else None
        
        finally:
            # Restore original thread count - even if the code crashes
            lib.num_threads(original_threads)
    
        return vj_combined, vk_combined

########################################################################
@timeit
def reduce_2rdm(rdm1, rdm2, ovlp, 
                truncation_style='eigval',nvecs=10, eval_thr=0.1, ham_thr=0.001,
                save_diag=False,
                use_svd=False, svd_weight = 1.0,
                iterative=False, nit=None, max_iter_time=10000,
                mol=None,train_en=None,Jdiag_only=True):
    """
    Function to lower the rank of 2-transition-RDM between a pair of 
    training states into the diagonals of 2-transition-cumulant and
    low rank decomposition vectors for the remainder

    Input:
        rdm1 (np.array([n,n])): 1-body reduced density matrix between two training states
        rdm2 (np.array([n,n,n,n])): 2-body reduced density matrix between two training states
        ovlp (float): Overlap between the training state pair
        truncation_style (str): 
            Criteria of low rank truncation.
            Available options: {'eigval' (default) : Choose vectors whose eigenvalue**2 is more than eval_thr,
                                'nvec'             : Choose 'nvec' highest eval**2 number of vectors,
                                'ham' : Choose the minimum number of vectors such that error in the subspace 
                                        Hamiltonian matrix elements is less than 'ham_thr'
                                'ham_en' : Choose the minimum number of vectors such that error in the subspace 
                                        Hamiltonian*overlap matrix elements is less than 'ham_thr'}
        nvecs (int): Number of low rank vectors to include
        eval_thr (float): Threshold to choose vectors based on their eval**2
        ham_thr (float): Threshold to choose vectors based on their H matrix elements (Hartree units)

        diag_mas (np.array([n,n,n,n])): Mask choosing diagonal matrices of (n,n,n,n) tensor. Computed 
                                        OTF if not given
                           
        # Parameters relevant for Hamiltonian error truncation
        mol (pyscf Mole object): Molecule object that is used for computing the Hamiltonian error
        train_en (float):  Energy of the training geometry used for the truncation
        Jdiag_only (bool): Whether only Coulomb diagonals are used in the inference when determining the 
                           Hamiltonian error truncation

    Output:
        lowrank_vecs (vals_trunc, vecs_trunc): Low rank eigenvalues and eigenvectors of off-diagonal 2-cumulant
        diagonals (np.array([3,n,n])): Diagonal matrices of 2-transition-cumulants
    """

    # Matrix to decompose
    mat_decomp = rdm2
    
    norb = rdm1.shape[0]
    norb_sq = norb * norb

    # Matrix to decompose
    mat_decomp = rdm2.copy()
    
    # Refactor the 2(t)RDM such that its eigenvectors solely correponds to
    # Coulomb grouping
    mat_decomp = 4/3*mat_decomp + 2/3*np.einsum('ijkl->ilkj',mat_decomp)
    
    # Check the nit is given is iterative is True
    if iterative and nit is None:
        print('Error in reduce_rdm: nit is not given for iterative diagonalization')
        sys.exit()
        
    # Check that it is hermitian and diagonalize the matrix
    #assert(np.allclose(mat_decomp.reshape((norb_sq, norb_sq)), mat_decomp.reshape((norb_sq,norb_sq)).T))
    if np.allclose(mat_decomp.reshape((norb_sq, norb_sq)), mat_decomp.reshape((norb_sq,norb_sq)).T):
        if not iterative:
            evals, evecs = scipy.linalg.eigh(mat_decomp.reshape((norb_sq, norb_sq)))
        else:
            evals, evecs = try_iterative_diag(mat_decomp.reshape((norb_sq, norb_sq)),
                                              k=nit, 
                                              which='LM', 
                                              max_time=max_iter_time)

        rightvecs = None
        joint = True # Joint decomp
    else:
        print('**Using SVD')
        if not iterative:
            evecs, evals, rightvecs = scipy.linalg.svd(rdm2.reshape((norb_sq, norb_sq)))
        else:
            evecs, evals, rightvecs = try_iterative_diag(rdm2.reshape((norb_sq, norb_sq)),
                                              k=nit, 
                                              which='LM', 
                                              use_svd=True,
                                              max_time=max_iter_time)

        joint = False

    ########################################################################
    #### Select low rank vectors
    
    # Choose at least one vector (lower bound for dynamic truncation)
    min_nvecs = 1
    
    # Make sure the mol and training energy is given for this truncation
    if truncation_style in ['ham','ham_en']:
        if mol is None or train_en is None:
            print('Error in reduce_2rdm: Insufficient input for decomposition based on Hamiltonian error.')
            sys.exit()
            
    elif truncation_style not in ['eigval','nvec']:
        print('Unknown truncation_style in reduce_2rdm: %s'%truncation_style)
        sys.exit()    
        
    # Fixed truncation based on 'nvecs' parameter
    # Or dynamic truncation based on the eigenvalue magnitude
        
    # Choose the most compact decomposition between joint eigendecomp and Coulomb SVD
    if not joint:
        
        if truncation_style in ['eigval','nvec']:

            # If not Hermitian, just use SVD
            lowrank_vecs = select_lowrank(evals, evecs, norb, rightvecs=rightvecs, truncation_style=truncation_style, 
                                          nvecs=nvecs, eval_thr=eval_thr, min_nvec=min_nvecs)
        elif truncation_style in ['ham','ham_en']:
                
            lowrank_vecs = select_lowrank_ham(evals, evecs, joint, norb,
                                   rdm2, rdm1, ovlp, save_diag,
                                   mol, train_en, Jdiag_only,
                                   rightvecs=rightvecs,
                                   truncation_style=truncation_style,
                                   ham_thr=ham_thr, min_nvec=min_nvecs)
    else:
        if truncation_style in ['eigval','nvec']:

            lowrank_vecs_joint = select_lowrank(evals, evecs, norb, rightvecs=rightvecs, 
                                                truncation_style=truncation_style, 
                                                nvecs=nvecs, eval_thr=eval_thr, min_nvec=min_nvecs)
            
        elif truncation_style in ['ham','ham_en']:
                
            lowrank_vecs_joint = select_lowrank_ham(evals, evecs, joint, norb,
                                   rdm2, rdm1, ovlp,save_diag,
                                   mol, train_en, Jdiag_only,
                                   rightvecs=rightvecs,
                                   truncation_style=truncation_style,
                                   ham_thr=ham_thr, min_nvec=min_nvecs)
            
        if not use_svd:
            lowrank_vecs = lowrank_vecs_joint

        else:

            if not iterative:
                evecs2, evals2, rightvecs2 = scipy.linalg.svd(rdm2.reshape((norb_sq, norb_sq)))
            else:
                evecs2, evals2, rightvecs2 = svds(rdm2.reshape((norb_sq, norb_sq)), k=nit, which='LM')

            if truncation_style in ['eigval','nvec']:

                lowrank_vecs_svd = select_lowrank(evals2, evecs2, norb, rightvecs=rightvecs2, truncation_style=truncation_style, 
                                              nvecs=nvecs, eval_thr=eval_thr, min_nvec=min_nvecs)
            
            elif truncation_style in ['ham','ham_en']:
                    
                lowrank_vecs_svd = select_lowrank_ham(evals2, evecs2, False, norb,
                                       rdm2, rdm1, ovlp,save_diag,
                                       mol, train_en, Jdiag_only,
                                       rightvecs=rightvecs2,
                                       truncation_style=truncation_style,
                                       ham_thr=ham_thr, min_nvec=min_nvecs)
            
            # Check which one is more compact
            if truncation_style in ['ham','ham_en']:
                if len(lowrank_vecs_joint[0])*svd_weight < len(lowrank_vecs_svd[0]):
                    lowrank_vecs = lowrank_vecs_joint
                else:
                    print('**Using SVD')
                    lowrank_vecs = lowrank_vecs_svd
                    joint = False
                    
            # TODO: Add considerations for norm error; not just compactness
            # SVD as well
            elif truncation_style in ['eigval']:
                if len(lowrank_vecs_joint[0]) < len(lowrank_vecs_svd[0]):
                    lowrank_vecs = lowrank_vecs_joint
                else:
                    print('**Using SVD')
                    lowrank_vecs = lowrank_vecs_svd
                    joint = False
                    
            else:
                if np.abs(lowrank_vecs_joint[0]).max() < np.abs(lowrank_vecs_svd[0]).max():
                    lowrank_vecs = lowrank_vecs_joint
                else:
                    print('**Using SVD')
                    lowrank_vecs = lowrank_vecs_svd
                    joint = False

    ########################################################################
    if not save_diag:
        diagonals = None
        
    else:
        #print('Error in reduce_2rdm: Saving diagonals not implemented yet.')
        #sys.exit()
        
        remainder = rdm2 - reconstruct_rdm2_joint(lowrank_vecs,joint=joint)
        
        # Save diagonals of the remainder
        diagonals = np.zeros([3,norb,norb])
        for (i,j) in itertools.product(range(norb), range(norb)):
            diagonals[0, i, j] = remainder[ i, i, j, j]
            if i != j:
                diagonals[1, i, j] = remainder[ i, j, i, j]
                diagonals[2, i, j] = remainder[ i, j, j, i]

    print('-- Norm error:', np.linalg.norm(reconstruct_rdm2_joint(lowrank_vecs, diagonals, joint=joint) - rdm2))

    return lowrank_vecs, diagonals, joint

def reconstruct_rdm2_joint(lowrank_vecs, diagonals=None, joint=True):
    """
    Reconstructing the 2RDM
    """
    lr_vals, lr_vecs, lr_rightvecs = lowrank_vecs
    rdm2_i = np.einsum('ija,a,akl->ijkl',lr_vecs, lr_vals, lr_rightvecs.conj(),optimize='optimal')
    # Add exchange part as well
    if joint:
        rdm2_i -= 0.5*np.einsum('kja,a,ail->ijkl',lr_vecs, lr_vals, lr_rightvecs.conj(),optimize='optimal')
    
    if diagonals is not None:
        norb = diagonals.shape[-1]
        for (i,j) in itertools.product(range(norb), range(norb)):
            rdm2_i[ i, i, j, j] += diagonals[0, i, j] 
            if i != j:
                rdm2_i[ i, j, i, j] += diagonals[1, i, j] 
                rdm2_i[ i, j, i, j] += diagonals[2, i, j] 
    
    return rdm2_i

"""
def reconstruct_subspace(lowrank_vecs, h2, ntrain=3):
    subspace_h = np.zeros([ntrain, ntrain])
    for vecs in lowrank_vecs.values():
        rdm2_i = reconstruct_rdm2_joint(vecs)
        subspace_h += 
"""   

########################################################################
@timeit         
def lowrank_hamiltonian(mol, one_RDM, S, lowrank_vecs, diagonals=None,
                        sao_basis=None, density_fit=True, df_basis=None, 
                        Jdiag_only=True, sao_diag=False,
                        hermitian=True,
                        debug=False):
    """
    Construct subspace Hamiltonian using the low-rank decomposition of 
    2-transition-cumulant
    
    Input:
        mol (Mole object): pySCF mole object at the test geometry
        
    """

    ### Preliminaries
    ntrain = S.shape[0]
    
    norb = one_RDM.shape[-1]
    norb_sq = norb * norb
    
    if diagonals is not None:
        use_diag = True
    else:
        use_diag = False
        
    # Initiate the mean field object to use DF integrals (no need to use kernel)
    #mol.symmetry = False
    if density_fit:
        mf = scf.RHF(mol).density_fit(auxbasis=df_basis)
        mf_grad = mf_grad_df
        get_jk = mf.with_df.get_jk
        
    else:
        mf = scf.RHF(mol)
        mf_grad = mf_grad_nodf
        get_jk = mf.get_jk
    
    # AO to SAO basis transformation
    if sao_basis is None:
        sao_basis = get_loewdin_trafo(mol.intor("int1e_ovlp"))
    
    # 1-electron integrals with DF
    h1_ao = mf.get_hcore()
    h1e_sao = np.einsum('ai,ab,bj->ij', sao_basis, h1_ao, sao_basis)
    
    # Get ERIs in SAO basis (using density fitting) for debugging
    if debug or sao_diag:
        # Run calculation to fill the MF object
        #mf.scf()
        #Lpq = mf.with_df._cderi
        #print(Lpq.shape)
        
        # Alternative - without using MF object
        Lpq = get_df_integrals(mol,auxbasis=df_basis)
        Lpq = lib.pack_tril(Lpq)
        #print(Lpq.shape)

        Lpq_sao = ao2mo._ao2mo.nr_e2(Lpq, sao_basis,
            (0, sao_basis.shape[1], 0, sao_basis.shape[1]),aosym="s2",mosym="s2")
        Lpq_sao = lib.unpack_tril(Lpq_sao)
        df_eri_sao = lib.einsum('Pij,Pkl->ijkl', Lpq_sao, Lpq_sao,optimize='optimal')

    ### 1-body contributions
    subspace_h = np.einsum('...kl,kl->...', one_RDM, h1e_sao,optimize='optimal')

    # Check if low-rank vectors have been vectorized
    if ('vals' in lowrank_vecs):
        vectorized = True
        # Whether to use training point symmetry; keep original 'hermitian' arg as default
        hermitian = lowrank_vecs.get('hermitian', hermitian)

    else:
        vectorized = False
        
    # Construct the subspace Hamiltonian
    if not vectorized:
        
        # Vectorized version is more efficient but keeping this here for testing purposes
        for bra in range(ntrain):
            if hermitian:
                ket_max = bra+1
            else:
                ket_max = ntrain
            for ket in range(ntrain):
                
                nvec = lowrank_vecs[(bra,ket)][0].shape[0]
                use_joint = lowrank_vecs[(bra, ket)][3]

                # Transform the low-rank vecs into
                lr_vecs_group = np.ascontiguousarray(lowrank_vecs[(bra, ket)][1].transpose((2,0,1)))
                sao_basis_T = np.ascontiguousarray(sao_basis.T)
                lr_vecs_ao = ao2mo._ao2mo.nr_e2(lr_vecs_group, sao_basis_T,
                (0, norb, 0, norb), aosym='s1', mosym='s1')
                lr_vecs_ao = lr_vecs_ao.reshape((nvec,norb,norb))
    
                # JK build
                if use_joint:
                    vj_list, vk_list = get_jk(dm=lr_vecs_ao.transpose(0,2,1), hermi=0)  # Specify hermiticity per case
                    subspace_h[bra,ket] += 0.5*np.einsum('aij,aij,a->', vj_list - 0.5 * vk_list, lr_vecs_ao.conj(), lowrank_vecs[(bra, ket)][0])
        
                # J build from SVD
                else:
                    lr_rightvecs_group = np.ascontiguousarray(lowrank_vecs[(bra, ket)][2])
                    sao_basis_arr = np.ascontiguousarray(sao_basis)
                    lr_rightvecs_ao = ao2mo._ao2mo.nr_e2(lr_rightvecs_group, sao_basis_arr,
                    (0, norb, 0, norb), aosym='s1', mosym='s1')
                    lr_rightvecs_ao = lr_rightvecs_ao.reshape((nvec,norb,norb))
                    #lr_rightvecs_ao = np.einsum('ai,...ij,bj->...ab', sao_basis, lowrank_vecs[(bra, ket)][2], sao_basis)
                    #lr_vecs_ao = np.einsum('ai,ij...,bj->...ab', sao_basis, lowrank_vecs[(bra, ket)][1], sao_basis)

                    # For reference; direct contraction:
                    #rdm2_i = np.einsum('ija,a,akl->ijkl',lr_vecs, lr_vals, lr_rightvecs.conj(),optimize='optimal')

                    # For test purposes, explicitly reconstruct RDM and contract with ERIs
                    if debug:
                        lowrank_vecs_i = lowrank_vecs[(bra, ket)][0],lowrank_vecs[(bra, ket)][1],lowrank_vecs[(bra, ket)][2]
                        rdm2_i = reconstruct_rdm2_joint(lowrank_vecs_i, None, joint=use_joint)
                        subspace_h[bra,ket] += 0.5*np.einsum('ijkl,ijkl->', rdm2_i, df_eri_sao)
                        
                    else:
                        vj_list, vk_list = get_jk(dm=lr_rightvecs_ao, hermi=0, with_k=False)  # Specify hermiticity per case
                        subspace_h[bra,ket] += 0.5*np.einsum('aij,aij,a->', vj_list, lr_vecs_ao, lowrank_vecs[(bra, ket)][0])

                if use_diag:
                    
                    # Avoid transforming DF array, and do everything via Coulomb and exchange builds
                    diag_ao_1 = np.einsum('ij,wi,xi->jwx',diagonals[bra, ket, 0, :, :], sao_basis, sao_basis)
                    vj = get_jk(dm = diag_ao_1, hermi=0, with_k=False)[0]
                    subspace_h[bra, ket] += 0.5 * np.einsum('yj,zj,jyz->', sao_basis, sao_basis, vj)
            
                    if not Jdiag_only:
                        diag_ao_23 = np.einsum('ij,wi,yi->jwy',diagonals[bra, ket, 1, :, :] + diagonals[bra, ket, 2, :, :], sao_basis, sao_basis)
                        vk = get_jk(dm = diag_ao_23, hermi=0, with_j=False)[1]
                        subspace_h[bra, ket] += 0.5 * np.einsum('xj,zj,jxz->', sao_basis, sao_basis, vk)
            
                    #diag_ao_3 = np.einsum('ij,wi,yi->jwy', diagonals[bra, ket, 2, :, :], sao_basis, sao_basis)
                    #vk = get_jk(dm = diag_ao_3, hermi=0, with_j=False)[1]
                    #subspace_h[bra, ket] += 0.5 * np.einsum('xj,zj,jxz->', sao_basis, sao_basis, vk)
                    
                    """
                    # Contract diagonal J using the diagonal of Lpq_sao to avoid ambiguous
                    # repeated-index broadcasting ('Pii') in einsum which can lead to shape errors.
                    Ldiag = np.diagonal(Lpq_sao, axis1=1, axis2=2)  # shape (P, norb)
                    subspace_h[bra, ket] += 0.5 * np.einsum('ij,pi,pj->', diagonals[bra, ket, 0, :, :], Ldiag, Ldiag, optimize='optimal')
                    subspace_h[bra, ket] += 0.5 * np.einsum('ij,Pij,Pij->',diagonals[bra, ket, 1, :, :], Lpq_sao, Lpq_sao)
                    subspace_h[bra, ket] += 0.5 * np.einsum('ij,Pij,Pji->',diagonals[bra, ket, 2, :, :], Lpq_sao, Lpq_sao)
                    """
        
    else:
        nvec = lowrank_vecs['vals'].shape[2]
        
        ### Joint ED inference
        if lowrank_vecs['has_ed']:
            # Grouped JK builds 
            lr_vecs_grouped = lowrank_vecs['vecs_stacked']
            
            # Transform the low-rank vecs into
            lr_vecs_grouped_c = np.ascontiguousarray(lr_vecs_grouped)
            sao_basis_T = np.ascontiguousarray(sao_basis.T)
            lr_vecs_ao = ao2mo._ao2mo.nr_e2(lr_vecs_grouped_c, sao_basis_T,
            (0, norb, 0, norb), aosym='s1', mosym='s1')
            lr_vecs_ao = lr_vecs_ao.reshape((lr_vecs_grouped.shape[0],norb,norb))
            
            # JK build
            vj_list, vk_list = get_jk(dm=lr_vecs_ao.transpose(0,2,1), hermi=0)  # Specify hermiticity per case
            vhf = vj_list - 0.5*vk_list
            
            # Reindex to separate bra, ket, nvec indices
            vhf = unpack_vec(vhf, lowrank_vecs['pairloc'],hermitian=hermitian, nbra=ntrain)
            lr_vecs_ao = unpack_vec(lr_vecs_ao, lowrank_vecs['pairloc'],hermitian=hermitian, nbra=ntrain)
            
            # Contruction for subspace Hamiltonian
            #print(shap, lr_vecs_ao.shape, lowrank_vecs['vals'].shape, lowrank_vecs['vals'][:shap[0],:shap[1],:shap[2]].shape)
            subspace_h += 0.5*np.einsum('xyaij,xyaij,xya->xy', vhf, lr_vecs_ao, lowrank_vecs['vals'][:,:,:vhf.shape[2]],optimize='optimal')
            
        ### Coulomb SVD inference
        if lowrank_vecs['has_svd']:
            
            # Grouped J Builds
            svd_vecs_grouped = lowrank_vecs['vecs_svd_stacked']
            svd_rightvecs_grouped = lowrank_vecs['rightvecs_stacked']
    
            # Transform the low-rank vecs into AO basis
            svd_rightvecs_grouped_c = np.ascontiguousarray(svd_rightvecs_grouped)
            sao_basis_T = np.ascontiguousarray(sao_basis.T)
            svd_rightvecs_ao = ao2mo._ao2mo.nr_e2(svd_rightvecs_grouped_c, sao_basis_T,
            (0, norb, 0, norb), aosym='s1', mosym='s1')
            svd_rightvecs_ao = svd_rightvecs_ao.reshape((svd_rightvecs_grouped.shape[0],norb,norb))
            
            svd_vecs_grouped_c = np.ascontiguousarray(svd_vecs_grouped)
            sao_basis_T = np.ascontiguousarray(sao_basis.T)
            svd_vecs_ao = ao2mo._ao2mo.nr_e2(svd_vecs_grouped_c, sao_basis_T,
            (0, norb, 0, norb), aosym='s1', mosym='s1')
            svd_vecs_ao = svd_vecs_ao.reshape((svd_vecs_grouped.shape[0],norb,norb))
            
            # J build
            vj_list, _ = get_jk(dm=svd_rightvecs_ao, hermi=0, with_k=False)  # Specify hermiticity per case
    
            # Reindex to separate bra, ket, nvec indices
            vj = unpack_vec(vj_list, lowrank_vecs['pairloc_svd'],hermitian=hermitian, nbra=ntrain)
            svd_vecs_ao = unpack_vec(svd_vecs_ao, lowrank_vecs['pairloc_svd'],hermitian=hermitian, nbra=ntrain)
            
            subspace_h += 0.5*np.einsum('xyaij,xyaij,xya->xy', vj, svd_vecs_ao, lowrank_vecs['vals'][:,:,:vj.shape[2]],optimize='optimal')
            
        #if use_diag:
        #    print('Error in lowrank_hamiltonian: Diagonal contraction not implemented')
        #    sys.exit()

        if use_diag:

            if not sao_diag:
                # Transform the low-rank vecs into AO basis for J builds
                diagJ_ao = np.einsum('Nij,wi,xi->Njwx', diagonals[0], sao_basis, sao_basis)

                # Flatten
                orig_shape = diagJ_ao.shape[:2]
                flat_diagJ_ao = diagJ_ao.reshape(orig_shape[0] * orig_shape[1], *diagJ_ao.shape[2:])

                # JK Builds
                vj_list = get_jk(dm=flat_diagJ_ao, hermi=0, with_k=False)[0]

                # Unflatten
                vj_unflat = vj_list.reshape(*orig_shape, *flat_diagJ_ao.shape[1:])
                vj = unstack_tril(vj_unflat, hermitian=hermitian)

                subspace_h += 0.5 * np.einsum('yj,zj,XYjyz->XY', sao_basis, sao_basis, vj)

            else:
                diagJ_unpack = unstack_tril(diagonals[0], hermitian=hermitian)
                subspace_h += 0.5 * np.einsum('XYij,Pii,Pjj->XY', diagJ_unpack, Lpq_sao, Lpq_sao)

            if not Jdiag_only:

                if not sao_diag:
                    # Transform the low-rank vecs into AO basis for K builds
                    diagK_ao = np.einsum('Nij,wi,yi->Njwy', diagonals[1], sao_basis, sao_basis)

                    # Flatten
                    orig_shape = diagK_ao.shape[:2]
                    flat_diagK_ao = diagK_ao.reshape(orig_shape[0] * orig_shape[1], *diagK_ao.shape[2:])

                    # JK Builds
                    vk_list = get_jk(dm=flat_diagK_ao, hermi=0, with_j=False)[1]

                    # Unflatten
                    vk_unflat = vk_list.reshape(*orig_shape, *flat_diagK_ao.shape[1:])
                    vk = unstack_tril(vk_unflat, hermitian=hermitian)

                    subspace_h += 0.5 * np.einsum('xj,zj,XYjxz->XY', sao_basis, sao_basis, vk)

                else:
                    diagK_unpack = unstack_tril(diagonals[1], hermitian=hermitian)
                    subspace_h += 0.5 * np.einsum('XYij,Pij,Pij->XY', diagK_unpack, Lpq_sao, Lpq_sao)
                    
    if hermitian:
        # Set the upper triangle
        subspace_h[np.triu_indices(ntrain)] = subspace_h.T[np.triu_indices(ntrain)].conj()
    
    # Check that hermitian
    #assert np.allclose(subspace_h, subspace_h.T.conj())

    return subspace_h

###############################################################################
@timeit
def get_jk_builds(mol, lowrank_vecs,
                  diagonals=None, Jdiag_only=True, sao_diag=False,
                  ao_mo_trafo=None,
                  density_fit=False, df_basis=None,
                  df_response=False):
    """
    Precompute the J(K) builds for the low-rank vectors for fast inference
    """
    # AO to SAO basis transformation
    if ao_mo_trafo is None:
        ao_mo_trafo = get_loewdin_trafo(mol.intor("int1e_ovlp"))
    
    # Initiate the mean field object to use DF integrals (no need to use kernel)
    #mol.symmetry = False
    if density_fit:
        mf = scf.RHF(mol).density_fit(auxbasis=df_basis)
        mf_grad = mf_grad_df
        get_jk = mf.with_df.get_jk
        #get_jk_grad = get_jk_grad_df
        
    else:
        mf = scf.RHF(mol)
        mf_grad = mf_grad_nodf
        get_jk = mf.get_jk
        #get_jk_grad = get_jk_grad_nodf

    # Check if diagonals are given
    if diagonals is None:
        use_diag = False
    else:
        use_diag = True
        
    ######################################################
    # Check if low-rank vectors have been vectorized
    if ('vals' in lowrank_vecs):
        vectorized = True
        # Whether to use training point symmetry
        hermitian = lowrank_vecs['hermitian']
        
    else:
        print('Error in get_jk_builds: Lowrank vectors are not in the vectorized format. Run "continuation_object.vectorize_lowrank()".')
        sys.exit()
        
    ######################################################
    norb = lowrank_vecs['vecs'].shape[-1]
    nvec = lowrank_vecs['vals'].shape[2]
    ntrain = lowrank_vecs['vals'].shape[0]

    ######################################################
    ######### COMPUTE PRELIMINARIES & FOCK BUILDS
    ######################################################
    # Initiate grad object
    grad_obj = mf_grad(mf)
    # TODO: Add auxbasis_response in the future, for now ignore it
    grad_obj.auxbasis_response = df_response
    
    ### Joint ED inference
    if lowrank_vecs['has_ed']:
        # Grouped JK builds 
        lr_vecs_grouped = lowrank_vecs['vecs_stacked']
        
        with log_time("AO transform (1)"):
            # Transform the low-rank vecs into
            lr_vecs_grouped_c = np.ascontiguousarray(lr_vecs_grouped)
            ao_mo_trafo_T = np.ascontiguousarray(ao_mo_trafo.T)
            lr_vecs_ao = ao2mo._ao2mo.nr_e2(lr_vecs_grouped_c, ao_mo_trafo_T,
            (0, norb, 0, norb), aosym='s1', mosym='s1')
            lr_vecs_ao = lr_vecs_ao.reshape((lr_vecs_grouped.shape[0],norb,norb))
            
        # JK build
        with log_time("JK Builds (1)"):
            vj_list, vk_list = get_jk(dm=lr_vecs_ao.transpose(0,2,1), hermi=0)  # Specify hermiticity per case
            #vj_list, vk_list = parallel_get_jk(lr_vecs_ao.transpose(0,2,1), get_jk, hermi=0)

        vhf = vj_list - 0.5*vk_list
        
        # Grad JK builds
        # TODO: Add auxbasis_response in the future, for now ignore it
        with log_time("JK Grad Builds (2)"):
            vj_grad_list_t, vk_grad_list_t = grad_obj.get_jk(dm=lr_vecs_ao.transpose(0,2,1), hermi=0) 
            vj_grad_list, vk_grad_list = grad_obj.get_jk(dm=lr_vecs_ao, hermi=0) 
            #vj_grad_list, vk_grad_list = parallel_get_jk(lr_vecs_ao, grad_obj.get_jk, hermi=0)
            #vj_grad_list_t, vk_grad_list_t = parallel_get_jk(lr_vecs_ao.transpose(0,2,1), grad_obj.get_jk, hermi=0)

        vhf_grad = vj_grad_list - 0.5*vk_grad_list
        vhf_grad_t = vj_grad_list_t - 0.5*vk_grad_list_t
    
        #vhf_aux = np.einsum('aamn,a->mn',vj_list.aux - vk_list.aux*.5,lr_vals[ii],optimize='optimal')
        #vhf_aux = (vj_list.aux - vk_list.aux*.5).sum((0,1))#,lr_vals[ii])
        #grad_i += vhf_aux
        # Reindex to separate bra, ket, nvec indices
        vhf = unpack_vec(vhf, lowrank_vecs['pairloc'],hermitian=hermitian, nbra=ntrain)
        lr_vecs_ao = unpack_vec(lr_vecs_ao, lowrank_vecs['pairloc'],hermitian=hermitian, nbra=ntrain)
        lr_vecs = unpack_vec(lr_vecs_grouped, lowrank_vecs['pairloc'],hermitian=hermitian, nbra=ntrain)
        vhf_grad = unpack_grad_vec(vhf_grad, lowrank_vecs['pairloc'],hermitian=hermitian, nbra=ntrain)
        vhf_grad_t = unpack_grad_vec(vhf_grad_t, lowrank_vecs['pairloc'],hermitian=hermitian, nbra=ntrain)
        
        if df_response:
            vhf_aux = unpack_grad_aux(vj_grad_list.aux - 0.5*vk_grad_list.aux,
                                        lowrank_vecs['pairloc'],
                                        lowrank_vecs['vals'],hermitian=hermitian)
            
            vhf_aux_t = unpack_grad_aux(vj_grad_list_t.aux - 0.5*vk_grad_list_t.aux,
                                        lowrank_vecs['pairloc'],
                                        lowrank_vecs['vals'],hermitian=hermitian)

            vhf_grad = lib.tag_array(vhf_grad, aux=np.array(vhf_aux))
            vhf_grad_t = lib.tag_array(vhf_grad_t, aux=np.array(vhf_aux_t))


    ### Coulomb SVD inference
    if lowrank_vecs['has_svd']:
        
        # Grouped J Builds
        svd_vecs_grouped = lowrank_vecs['vecs_svd_stacked']
        svd_rightvecs_grouped = lowrank_vecs['rightvecs_stacked']

        with log_time("AO transform (2)"):
            # Transform the low-rank vecs into AO basis
            svd_rightvecs_grouped_c = np.ascontiguousarray(svd_rightvecs_grouped)
            ao_mo_trafo_T = np.ascontiguousarray(ao_mo_trafo.T)
            svd_rightvecs_ao = ao2mo._ao2mo.nr_e2(svd_rightvecs_grouped_c, ao_mo_trafo_T,
            (0, norb, 0, norb), aosym='s1', mosym='s1')
            svd_rightvecs_ao = svd_rightvecs_ao.reshape((svd_rightvecs_grouped.shape[0],norb,norb))
            
            svd_vecs_grouped_c = np.ascontiguousarray(svd_vecs_grouped)
            ao_mo_trafo_T = np.ascontiguousarray(ao_mo_trafo.T)
            svd_vecs_ao = ao2mo._ao2mo.nr_e2(svd_vecs_grouped_c, ao_mo_trafo_T,
            (0, norb, 0, norb), aosym='s1', mosym='s1')
            svd_vecs_ao = svd_vecs_ao.reshape((svd_vecs_grouped.shape[0],norb,norb))
            
        # J builds
        with log_time("J Builds (2)"):
            vj_r_list, _ = get_jk(dm=svd_rightvecs_ao, hermi=0, with_k=False)
            vj_l_list, _ = get_jk(dm=svd_vecs_ao, hermi=0, with_k=False)
            #vj_r_list, _ = parallel_get_jk(svd_rightvecs_ao, get_jk, hermi=0,with_k=False)
            #vj_l_list, _ = parallel_get_jk(svd_vecs_ao, get_jk, hermi=0,with_k=False)

        # Grad JK builds
        # TODO: Add auxbasis_response in the future, for now ignore it
        with log_time("J Grad Builds (2)"):
            vj_lgrad_list, _ = grad_obj.get_jk(dm=svd_vecs_ao, hermi=0, with_k=False) 
            vj_rgrad_list, _ = grad_obj.get_jk(dm=svd_rightvecs_ao, hermi=0, with_k=False) 
            # gradient JK builds 
            #vj_lgrad_list = parallel_get_jk(svd_vecs_ao, grad_obj.get_j, hermi=0)
            #vj_rgrad_list = parallel_get_jk(svd_rightvecs_ao, grad_obj.get_j, hermi=0)

        # Reindex to separate bra, ket, nvec indices
        vj_right = unpack_vec(vj_r_list, lowrank_vecs['pairloc_svd'],hermitian=hermitian, nbra=ntrain)
        vj_left = unpack_vec(vj_l_list, lowrank_vecs['pairloc_svd'],hermitian=hermitian, nbra=ntrain)
        svd_lvecs_ao = unpack_vec(svd_vecs_ao, lowrank_vecs['pairloc_svd'],hermitian=hermitian, nbra=ntrain)
        svd_rvecs_ao = unpack_vec(svd_rightvecs_ao, lowrank_vecs['pairloc_svd'],hermitian=hermitian, nbra=ntrain)
        svd_lvecs = unpack_vec(svd_vecs_grouped, lowrank_vecs['pairloc_svd'],hermitian=hermitian, nbra=ntrain)
        svd_rvecs = unpack_vec(svd_rightvecs_grouped, lowrank_vecs['pairloc_svd'],hermitian=hermitian, nbra=ntrain)
        
        vj_l_grad = unpack_grad_vec(vj_lgrad_list, lowrank_vecs['pairloc_svd'],hermitian=hermitian, nbra=ntrain)
        vj_r_grad = unpack_grad_vec(vj_rgrad_list, lowrank_vecs['pairloc_svd'],hermitian=hermitian, nbra=ntrain)
                
        if df_response:
            vj_l_aux = unpack_grad_aux(vj_l_grad.aux,
                                        lowrank_vecs['pairloc_svd'],
                                        lowrank_vecs['vals'],hermitian=hermitian)
            
            vj_r_aux = unpack_grad_aux(vj_r_grad.aux,
                                        lowrank_vecs['pairloc_svd'],
                                        lowrank_vecs['vals'],hermitian=hermitian)

            vj_l_grad = lib.tag_array(vj_l_grad, aux=np.array(vj_l_aux))
            vj_r_grad = lib.tag_array(vhf_grad_t, aux=np.array(vj_r_aux))

    # Diagonal JK builds
    if use_diag and not sao_diag:
                
        ### First the Coulomb build
        
        # Expand out the 'i' indices for J builds
        diagJ_ao = np.einsum('Nij,wj,xj->Niwx',diagonals[0], ao_mo_trafo, ao_mo_trafo,optimize='optimal')
        #diagJ_ao_T = np.einsum('Nji,wj,xj->Niwx',diagonals[0], ao_mo_trafo, ao_mo_trafo,optimize='optimal')

        # Flatten
        orig_shape = diagJ_ao.shape[:2]
        flat_diagJ_ao = diagJ_ao.reshape(orig_shape[0] * orig_shape[1], *diagJ_ao.shape[2:])
        #flat_diagJ_ao_T = diagJ_ao_T.reshape(orig_shape[0] * orig_shape[1], *diagJ_ao.shape[2:])
        
        # JK Builds
        with log_time("Diag J Builds (1)"):
            #vj_list = get_jk(dm = flat_diagJ_ao, hermi=0, with_k=False)[0]
            vj_list = parallel_get_jk(flat_diagJ_ao, get_jk, hermi=0,with_k=False)[0]

        # JK grad builds
        with log_time("Diag Grad J Builds (1)"):
            #vj_grad_list = grad_obj.get_jk(dm=flat_diagJ_ao.transpose(0,2,1), hermi=0, with_k=False) [0]
            #vj_grad_t_list = grad_obj.get_jk(dm=flat_diagJ_ao_T.transpose(0,2,1), hermi=0, with_k=False) [0]

            # gradient JK builds for diagonals
            #vj_grad_list = parallel_get_jk(flat_diagJ_ao.transpose(0,2,1), grad_obj.get_jk, hermi=0, with_k=False)[0]
            vj_grad_list = parallel_get_jk(flat_diagJ_ao.transpose(0,2,1), grad_obj.get_j, hermi=0)

        # Unflatten
        vj_unflat = vj_list.reshape(*orig_shape, *vj_list.shape[1:])
        vj = unstack_tril(vj_unflat,hermitian=hermitian)
        
        vj_grad_unflat = vj_grad_list.reshape(*orig_shape, *vj_grad_list.shape[1:])  
        vj_grad = unstack_tril(vj_grad_unflat,hermitian=hermitian)
        
        #vj_grad_t_unflat = vj_grad_t_list.reshape(*orig_shape, *vj_grad_t_list.shape[1:])  
        #vj_grad_t = unstack_tril(vj_grad_t_unflat,hermitian=hermitian)
    
        if not Jdiag_only:
            # Transform the low-rank vecs into
            diagK_ao = np.einsum('Nij,wi,yi->Njwy',diagonals[1], ao_mo_trafo, ao_mo_trafo,optimize='optimal')
            diagK_ao_T = np.einsum('Nji,wi,yi->Njwy',diagonals[1], ao_mo_trafo, ao_mo_trafo,optimize='optimal')

            # Flatten
            orig_shape = diagK_ao.shape[:2]
            flat_diagK_ao = diagK_ao.reshape(orig_shape[0] * orig_shape[1], *diagK_ao.shape[2:])
            flat_diagK_ao_T = diagK_ao_T.reshape(orig_shape[0] * orig_shape[1], *diagK_ao.shape[2:])
            
            # JK Builds
            with log_time("Diag K Builds (1)"):
                #vk_list = get_jk(dm = flat_diagK_ao, hermi=0, with_j=False)[1]
                #vk_t_list = get_jk(dm = flat_diagK_ao_T, hermi=0, with_j=False)[1]
                vk_list = parallel_get_jk(flat_diagK_ao, get_jk, hermi=0,with_j=False)[1]
                vk_t_list = parallel_get_jk(flat_diagK_ao_T, get_jk, hermi=0,with_j=False)[1]

            # JK grad builds
            with log_time("Diag Grad K Builds (1)"):
                #vk_grad_list = grad_obj.get_jk(dm=flat_diagK_ao.transpose(0,2,1) + flat_diagK_ao_T.transpose(0,2,1), hermi=0, with_j=False) [1]
                #vk_grad_t_list = grad_obj.get_jk(dm=flat_diagK_ao_T.transpose(0,2,1), hermi=0, with_j=False) [1]
                
                # gradient JK builds for diagonals
                vk_grad_list = parallel_get_jk(flat_diagK_ao.transpose(0,2,1), grad_obj.get_jk, hermi=0, with_j=False)[1]

            # Unflatten
            vk_unflat = vk_list.reshape(*orig_shape, *flat_diagK_ao.shape[1:])  # (3, 4, 5, 6)
            vk = unstack_tril(vk_unflat,hermitian=hermitian)
            
            vk_t_unflat = vk_t_list.reshape(*orig_shape, *flat_diagK_ao.shape[1:])  # (3, 4, 5, 6)
            vk_t = unstack_tril(vk_t_unflat,hermitian=hermitian)
            
            vk_grad_unflat = vk_grad_list.reshape(*orig_shape, *vk_grad_list.shape[1:])  
            vk_grad = unstack_tril(vk_grad_unflat,hermitian=hermitian)
            
            #vk_grad_t_unflat = vk_grad_t_list.reshape(*orig_shape, *vk_grad_list.shape[1:])  
            #vk_grad_t = unstack_tril(vk_grad_t_unflat,hermitian=hermitian)

        else:
            vk, vk_t, vk_grad = None, None, None
    
    # Function OUTPUT
    ed_return, svd_return, diag_return = None, None, None
    
    if lowrank_vecs['has_ed']:
        ed_return = (lr_vecs, lr_vecs_ao, vhf, vhf_grad, vhf_grad_t)
        
    if lowrank_vecs['has_svd']:
        svd_return = (svd_lvecs, svd_rvecs, svd_lvecs_ao, svd_rvecs_ao, vj_left, vj_right, vj_l_grad, vj_r_grad)

    if use_diag and not sao_diag:
        diag_return = (vj, vj_grad, vk, vk_t, vk_grad)

    return ed_return, svd_return, diag_return


###############################################################################
def select_lowrank(evals, evecs, norb, 
                   rightvecs=None,
                   truncation_style='eigval',nvecs=10, eval_thr=0.1, min_nvec=0):
    """
    Function to select low-rank vectors from the eigendecomposition
    """
    # Check if right eigenvectors are given
    if rightvecs is None:
        rightvecs = evecs.T

    # Sort the eigenstates by the square of their eigenvalue
    idx = (-np.power(evals, 2)).argsort()
    evals_sort = evals[idx]
    evecs_sort = evecs[:,idx]
    rightvecs_sort = rightvecs[idx,:]

    # Truncate through either eigvals or a given number of vectors
    if truncation_style == 'eigval':
        nvecs = len(evals_sort[np.power(evals_sort,2) > eval_thr])
        nvecs = max(min_nvec, nvecs)
        
    #norb = np.sqrt(evecs_sort.shape[0],dtype=int)
    vals_trunc = evals_sort[:nvecs]
    vecs_trunc = evecs_sort[:,:nvecs].reshape((norb, norb, nvecs))
    rightvecs_trunc = rightvecs_sort[:nvecs,:].reshape((nvecs,norb, norb))

    return vals_trunc, vecs_trunc, rightvecs_trunc

def select_lowrank_ham(evals, evecs, joint, norb,
                       rdm2, rdm1, ovlp, save_diag,
                       mol, training_energy, Jdiag_only,
                       rightvecs=None,
                       density_fit=True,
                       truncation_style='ham',
                       ham_thr=0.001, min_nvec=0
                       ):
    """
    Select a low rank decomposition of the RDM based on the error on
    subspace hamiltonian
    """
    # Check if right eigenvectors are given
    if rightvecs is None:
        rightvecs = evecs.T

    """
    lowrank_hamiltonian(mol, one_RDM, S, lowrank_vecs, diagonals=None,
                            sao_basis=None, df_basis=None, 
                            Jdiag_only=True, sao_diag=True,
                            hermitian=True,
                            debug=True)
    """
    # Sort the eigenstates by the square of their eigenvalue
    idx = (-np.power(evals, 2)).argsort()
    evals_sort = evals[idx]
    evecs_sort = evecs[:,idx]
    rightvecs_sort = rightvecs[idx,:]

    # Prepare rdms in a suitable format
    one_RDM = np.zeros([1,1,norb,norb])
    one_RDM[0,0,:,:] = rdm1
    
    S = np.array([[ovlp]])
    
    # Exact element of subspace Hamiltonian
    ham_training = ovlp * training_energy
    
    # For direct contraction
    h1, h2 = get_integrals(mol, get_basis(mol))

    # Iterate over subset
    ham_err = [1000,1000]
    nvecs = 0
    while ((abs(ham_err[-1]) > ham_thr or abs(ham_err[-2]) > ham_thr) or nvecs < min_nvec+2) and nvecs <= norb*norb:
        # Truncate
        vals_trunc = evals_sort[:nvecs]
        vecs_trunc = evecs_sort[:,:nvecs].reshape((norb, norb, nvecs))
        rightvecs_trunc = rightvecs_sort[:nvecs,:].reshape((nvecs,norb, norb))

        #lowrank_vecs = {(0,0):(vals_trunc,vecs_trunc, rightvecs_trunc)}
        
        # TODO: Instead of reconstructing the RDM from scratch; just 
        # add the contribution iteratively in the loop for each new vector
        
        # Diagonal of the remainder
        rdm2_reconstructed = reconstruct_rdm2_joint((vals_trunc,vecs_trunc, rightvecs_trunc),
                                                    joint=joint)
        
        if save_diag:
            remainder = rdm2 - rdm2_reconstructed
            
            # Fix the diagonals
            for (i,j) in itertools.product(range(norb), range(norb)):
                rdm2_reconstructed[ i, i, j, j] += remainder[ i, i, j, j]
                if not Jdiag_only and i!= j:
                    rdm2_reconstructed[ i, j, i, j] += remainder[ i, j, i, j]
                    rdm2_reconstructed[ i, j, j, i] += remainder[ i, j, j, i]

        # Compute subspace Hamiltonian
        ham_new = lib.einsum('ij,ij->',h1,rdm1) + 0.5*lib.einsum('ijkl,ijkl->',h2,rdm2_reconstructed)

        # Since rdm2 is already reconstructed, it's better to use that for now
        #ham_new = lowrank_hamiltonian(mol, one_RDM, S, diagonals, 
        #                              lowrank_vecs, sao_basis=None,
        #                              Jdiag_only=Jdiag_only, 
        #                              density_fit=density_fit)[0,0]
        
        # Compute error and go to next iteration to see if it is good enough
        if truncation_style == 'ham':
            ham_err.append(ham_training - ham_new)
        elif truncation_style == 'ham_en':
            ham_err.append(training_energy - ham_new/ovlp)

        #print(nvecs, ovlp, training_energy)
        #print(nvecs, ham_new, ham_training, ham_err[-1])
        nvecs += 1
        
    # Truncated decomposition
    if nvecs > norb*norb:
        nvec_select = norb*norb 
    else:
        #nvec_select = max(min_nvec, nvecs-2)
        nvec_select = nvecs-2
        
    print(nvec_select)
    vals_trunc = evals_sort[:nvec_select]
    vecs_trunc = evecs_sort[:,:nvec_select].reshape((norb, norb, nvec_select))
    rightvecs_trunc = rightvecs_sort[:nvec_select,:].reshape((nvec_select,norb, norb))

    return vals_trunc, vecs_trunc, rightvecs_trunc
    
###############################################################################
def stack_lowrank(vecs_lowrank, hermitian=True):
    """
    Function to group dynamically chosen low-rank eigenstates for different
    bra,ket pairs into a compound index for efficient inference
    """
    # Prelim
    nbra = list(vecs_lowrank.keys())[-1][0]+1
    norb = vecs_lowrank[(0,0)][1].shape[1]
    
    # Store the locations of bra,ket pairs in the composite index
    pair_loc = {}
    
    # Start stacking
    vecs_lr = []
    vals_lr = []
    nvec_tot = 0
    
    # Have a separate on for SVD vectors that only needs J builds
    pair_svd_loc = {}
    vecs_svd_lr = []
    rightvecs_svd_lr = []
    vals_svd_lr = []
    nsvd_tot = 0
    
    for i in range(nbra):
        
        # Only iterarte through lower triangular indices
        if hermitian:
            jmax = i+1
        else:
            jmax = nbra
            
        for j in range(jmax):
            lr_i = vecs_lowrank[(i,j)]

            nvec_i = lr_i[0].shape[-1]
            
            # Joint ED
            if lr_i[-1]:
                vecs_lr.append(lr_i[1].transpose(2,0,1))
                vals_lr.append(lr_i[0])
                
                pair_loc[(i,j)] = [nvec_tot, nvec_tot + nvec_i]
                
                nvec_tot += nvec_i     
            
            # Coulomb SVD
            else:
                vecs_svd_lr.append(lr_i[1].transpose(2,0,1))
                rightvecs_svd_lr.append(lr_i[2])
                vals_svd_lr.append(lr_i[0])
                
                pair_svd_loc[(i,j)] = [nsvd_tot, nsvd_tot + nvec_i]
                
                nsvd_tot += nvec_i                  
                
    # Check if any (t)RDM used ED
    has_ed = True
    if len(vecs_lr) == 0:
        has_ed = False
      
    # Check if any (t)RDM used SVD
    has_svd = True
    if len(vecs_svd_lr) == 0:
        has_svd = False
        
    # Set up the final dictionary
    stacked_lowrank = {}
    stacked_lowrank['hermitian'] = hermitian

    if has_ed:
        # Joint ED vectors
        vecs_stacked = np.concatenate(vecs_lr,axis=0)
        vals_stacked = np.concatenate(vals_lr)
        
        stacked_lowrank['vals'] = vals_stacked
        stacked_lowrank['vecs'] = vecs_stacked
        stacked_lowrank['pairloc'] = pair_loc
                
    if has_svd:
        # Coulomb SVD vectors
        vecs_svd_stacked = np.concatenate(vecs_svd_lr,axis=0)
        rightvecs_svd_stacked = np.concatenate(rightvecs_svd_lr,axis=0)
        vals_svd_stacked = np.concatenate(vals_svd_lr)    
        
        stacked_lowrank['vals_svd'] = vals_svd_stacked
        stacked_lowrank['vecs_svd'] = vecs_svd_stacked
        stacked_lowrank['rightvecs_svd'] = rightvecs_svd_stacked
        stacked_lowrank['pairloc_svd'] = pair_svd_loc
        
    return stacked_lowrank, has_svd, has_ed


def stack_tril(arr, hermitian=True):
    """
    Stack selected blocks from a (n, n, x, x) array into a compact (m, x, x) array.
    
    If hermitian=True, stacks only the lower triangle (i >= j),
    assuming the array is Hermitian in its (n, n) block structure.
    
    If hermitian=False, stacks the full (i, j) grid in row-major order.
    
    Parameters:
        arr : np.ndarray
            Input array of shape (n, n, x, x)
        hermitian : bool
            Whether to restrict to lower-triangular blocks only

    Returns:
        stacked : np.ndarray
            Stacked array of shape (m, x, x) where m = n*(n+1)//2 if Hermitian,
            or m = n*n if not.
    """
    n, _, x, _ = arr.shape
    packed = []
    for i in range(n):
        jmax = i + 1 if hermitian else n
        for j in range(jmax):
            packed.append(arr[i, j])
    return np.array(packed)


def unstack_tril(packed, hermitian=True):
    """
    Unpacks a stacked array of shape (m, ...) into shape (n, n, ...), where the first
    axis was previously packed using only the lower triangle (if hermitian=True) or the full (n,n) grid.

    Parameters:
        packed : np.ndarray
            Input array with shape (m, ...) where m = n*(n+1)//2 (hermitian) or n*n (full)
        hermitian : bool
            Whether the packed data was from the lower triangle only

    Returns:
        arr : np.ndarray
            Output array of shape (n, n, ...)
    """
    # If the input already appears to be in full (n, n, ...) grid form, just
    # return it (but ensure Hermitian symmetry is enforced when requested).
    if packed.ndim >= 2 and packed.shape[0] == packed.shape[1]:
        arr = packed.copy()
        if hermitian:
            n = arr.shape[0]
            for i in range(n):
                for j in range(i + 1, n):
                    try:
                        arr[j, i] = arr[i, j].conj()
                    except Exception:
                        arr[j, i] = arr[i, j]
        return arr

    m = packed.shape[0]
    rest_shape = packed.shape[1:]

    # The packed length m can sometimes be both a triangular number and a perfect square
    # (e.g., m=36 -> triangular for n=8, square for n=6). Use the `hermitian` flag to
    # disambiguate: when hermitian=True prefer triangular (lower-triangle) packing,
    # otherwise prefer full-grid (n*n) packing.

    # Check triangular possibility: solve n(n+1)/2 == m
    tri_n = int((math.isqrt(1 + 8 * m) - 1) // 2)
    is_tri = (tri_n * (tri_n + 1) // 2 == m)

    # Check square possibility: m == n*n
    sq_n = int(math.isqrt(m))
    is_sq = (sq_n * sq_n == m)

    # Prefer triangular when requested or when square interpretation is impossible
    if is_tri and (hermitian or not is_sq):
        n = tri_n
        arr = np.zeros((n, n) + rest_shape, dtype=packed.dtype)
        idx = 0
        for i in range(n):
            for j in range(i + 1):  # j <= i
                val = packed[idx]
                arr[i, j] = val
                # Mirror to the upper triangle to restore full matrix symmetry
                try:
                    arr[j, i] = val.conj()
                except Exception:
                    arr[j, i] = val
                idx += 1
        return arr

    # If square packing fits (n*n) interpret as full-grid
    if is_sq:
        n = sq_n
        arr = packed.reshape((n, n) + rest_shape).copy()
        if hermitian:
            for i in range(n):
                for j in range(i + 1, n):
                    try:
                        arr[j, i] = arr[i, j].conj()
                    except Exception:
                        arr[j, i] = arr[i, j]
        return arr

    # If none of the interpretations matched, raise an informative error
    raise ValueError(f"Invalid packed shape for unstacking: not triangular nor square (m={m})")


def stack_diagonal(diagonals, hermitian=True):
    """
    Stack 2(t)RDM diagonals for a vectorized inference
    """
    
    diag_J = diagonals[:,:,0]
    diag_K = diagonals[:,:,1] + diagonals[:,:,2]
    
    stacked_diagJ = stack_tril(diag_J, hermitian=hermitian)
    stacked_diagK = stack_tril(diag_K, hermitian=hermitian)
                
    return (stacked_diagJ, stacked_diagK)

def unpack_vec(vecs,pair_loc,hermitian=True,nbra=None):
    """
    Function to unpack vectors stacked using "stack_lowrank" function

    """
    if nbra is None:
        nbra = list(pair_loc.keys())[-1][0]+1
    norb = vecs.shape[1]
    nvec_max = np.max([j-i for i,j in pair_loc.values()])
    
    vecs_unpacked = np.zeros([nbra, nbra, nvec_max,norb,norb])
    
    """
    for i in range(nbra):
        # Only iterarte through lower triangular indices
        if hermitian:
            jmax = i+1
        else:
            jmax = nbra
            
        for j in range(jmax):
            
            # Check key
            if (i,j) in pair_loc:
                st, en = pair_loc[(i,j)]
                vecs_unpacked[i,j,:(en-st)] = vecs[st:en]
    """
    # Precompute index arrays for batch assignment
    for (i, j), (start, end) in pair_loc.items():
        nv = end - start
        vecs_unpacked[i, j, :nv] = vecs[start:end]


    return vecs_unpacked


def unpack_grad_vec(vecs,pair_loc,hermitian=True,nbra=None):
    """
    Function to unpack vectors stacked using "stack_lowrank" function

    """
    if nbra is None:
        nbra = list(pair_loc.keys())[-1][0]+1
    norb = vecs.shape[2]
    nvec_max = np.max([j-i for i,j in pair_loc.values()])
    
    vecs_unpacked = np.zeros([nbra, nbra, nvec_max, 3, norb, norb])
    
    """
    for i in range(nbra):
        # Only iterarte through lower triangular indices
        if hermitian:
            jmax = i+1
        else:
            jmax = nbra
            
        for j in range(jmax):
            
            # Check key
            if (i,j) in pair_loc:
                st, en = pair_loc[(i,j)]
                vecs_unpacked[i,j,:(en-st)] = vecs[st:en]
    """
    
    # Precompute index arrays for batch assignment
    for (i, j), (start, end) in pair_loc.items():
        nv = end - start
        vecs_unpacked[i, j, :nv] = vecs[start:end]

    """
    # Check if auxbasis response is computed
    #try:
    print('auxbasis unpacking')
    auxvec = vecs.aux
    nat = auxvec.shape[-2]
    
    aux_unpacked = np.zeros([nbra, nbra, nbra, nbra, nvec_max, nat, 3])

    # Precompute index arrays for batch assignment
    for (i, j), (start, end) in pair_loc.items():
        nv = end - start
        aux_unpacked[i,i,j, j, :nv] = auxvec[start:end,start:end]

    vecs_unpacked = lib.tag_array(vecs_unpacked, aux=np.array(aux_unpacked))

    #except:
    #    print('No auxbasis')
    #    None
    """
    return vecs_unpacked


def unpack_grad_aux(vecs,pair_loc,vals,hermitian=True):
    """
    Function to unpack vectors stacked using "stack_lowrank" function

    """
    nbra = list(pair_loc.keys())[-1][0]+1

    auxvec = vecs
    nat = auxvec.shape[-2]
    
    aux_unpacked = np.zeros([nbra, nbra, nat, 3])

    # Precompute index arrays for batch assignment
    for (i, j), (start, end) in pair_loc.items():
        nv = end - start
        aux_unpacked[i,j] = np.einsum('aamn,a->mn',auxvec[start:end,start:end],vals[i,j,:nv],optimize='optimal')

    return aux_unpacked

def unpack_lowrank(stacked_lowrank,hermitian=True):
    """
    For testing; function to unpack both eigenvectors and eigenvectors
    from the stacked_lowrank dictionary
    """
    
    vals_stacked = stacked_lowrank['vals']
    vecs_stacked = stacked_lowrank['vecs']
    pair_loc = stacked_lowrank['pairloc']
    
    # Prelim
    nbra = list(pair_loc.keys())[-1][0]+1
    
    unpacked_vecs = {}
    if hermitian:
        for i in range(nbra):
            for j in range(i+1):
                st, en = pair_loc[(i,j)]
                vals_i = vals_stacked[st:en]
                vecs_i = vecs_stacked[st:en].transpose(1,2,0)
                
                unpacked_vecs[(i,j)] = vals_i, vecs_i

    else:
        for i, j in itertools.product(range(nbra), range(nbra)):
            st, en = pair_loc[(i,j)]
            vals_i = vals_stacked[st:en]
            vecs_i = vecs_stacked[st:en].transpose(1,2,0)

            unpacked_vecs[(i,j)] = vals_i, vecs_i
        
    return unpacked_vecs
    
# Attribute function to vectorize low-rank vectors for EVCont solver classes
def vectorize_lowrank(self, hermitian=True):
    
    # Make sure a low-rank decomposition has been performed
    assert len(self.vecs_lowrank.items()) != 0
    
    # Find the largest number of vectors for each bra,ket pair
    nbra = self.overlap.shape[0]
    norb = self.one_rdm.shape[-1]
    # Determine maximum vectors per pair quickly
    nvec_max = max((lr[0].shape[-1] for lr in self.vecs_lowrank.values()), default=0)

    # Use stack_lowrank to get packed/staged arrays and flags
    stacked, has_svd, has_ed = stack_lowrank(self.vecs_lowrank, hermitian=hermitian)

    # Allocate padded arrays for per-pair fast indexing
    vals_lr = np.zeros((nbra, nbra, nvec_max), dtype=float)
    vecs_lr = np.zeros((nbra, nbra, nvec_max, norb, norb), dtype=float)
    # store number of vectors per pair explicitly to make unpack lossless
    nvecs_per_pair = np.zeros((nbra, nbra), dtype=int)
    # Only allocate rightvecs if any pair uses ED (joint decomposition)
    rightvecs_lr = None
    if has_ed:
        rightvecs_lr = np.zeros((nbra, nbra, nvec_max, norb, norb), dtype=float)

    # Fill padded arrays from the original per-pair dict (single pass)
    for (i, j), lr_i in self.vecs_lowrank.items():
        vals = lr_i[0]
        vecs = lr_i[1]
        rvecs = lr_i[2]
        nvec_i = vals.shape[-1]

        if nvec_i == 0:
            continue

        vals_lr[i, j, :nvec_i] = vals
        nvecs_per_pair[i, j] = nvec_i
        # stored as (norb, norb, nvec) in original, need to transpose to (nvec, norb, norb)
        vecs_lr[i, j, :nvec_i] = vecs.transpose(2, 0, 1)
        if rightvecs_lr is not None:
            rightvecs_lr[i, j, :nvec_i] = rvecs
    
    # Vectorize diagonal corrections: store both the packed J and the packed K-sum
    # for compatibility, but also keep packed components of diag[1] and diag[2]
    if ( getattr(self, 'diagonal_lr', None) is not None ) and ( not np.isnan(self.diagonal_lr).all() ):
        diagJ = self.diagonal_lr[:, :, 0]
        diagK1 = self.diagonal_lr[:, :, 1]
        diagK2 = self.diagonal_lr[:, :, 2]

        stacked_diagJ = stack_tril(diagJ, hermitian=hermitian)
        stacked_diagKsum = stack_tril(diagK1 + diagK2, hermitian=hermitian)
        # Also store separated components for lossless roundtrip
        stacked_diagK1 = stack_tril(diagK1, hermitian=hermitian)
        stacked_diagK2 = stack_tril(diagK2, hermitian=hermitian)

        self.diagonal_vectorized = np.stack((stacked_diagJ, stacked_diagKsum))
        # components stored separately to enable exact reconstruction
        self.diagonal_vectorized_components = np.stack((stacked_diagK1, stacked_diagK2))
    else:
        self.diagonal_vectorized = None
        self.diagonal_vectorized_components = None
    #self.diagonal_K = diagK

    # Set this low-rank description
    # Build final dict
    self.lowrank_vectorized = {
        'vals': vals_lr,
        'vecs': vecs_lr,
        'hermitian': hermitian,
        'has_ed': has_ed,
        'has_svd': has_svd,
        'nvecs': nvecs_per_pair,
    }

    if has_ed:
        self.lowrank_vectorized['rightvecs'] = rightvecs_lr
        self.lowrank_vectorized['vecs_stacked'] = stacked['vecs']
        self.lowrank_vectorized['pairloc'] = stacked['pairloc']

    if has_svd:
        self.lowrank_vectorized['rightvecs_stacked'] = stacked['rightvecs_svd']
        self.lowrank_vectorized['vecs_svd_stacked'] = stacked['vecs_svd']
        self.lowrank_vectorized['pairloc_svd'] = stacked['pairloc_svd']
    
    
###############################################################################

def unpack_vectorized_lowrank(self):
    """
    Reconstruct per-(bra,ket) lowrank dict (`self.vecs_lowrank`) and
    `self.diagonal_lr` from the vectorized representations
    (`self.lowrank_vectorized` and `self.diagonal_vectorized`). This
    enables appending new training points when the object is already
    vectorized. Mirrors the helper previously attached to the FCI_EVCont_obj.
    """
    if not getattr(self, 'lowrank_vectorized', None):
        return

    lv = self.lowrank_vectorized
    vals = lv.get('vals')
    vecs = lv.get('vecs')
    rightvecs = lv.get('rightvecs', None)
    pairloc = lv.get('pairloc', {})
    pairloc_svd = lv.get('pairloc_svd', {})
    hermitian = lv.get('hermitian', True)
    nvecs_per_pair = lv.get('nvecs', None)

    # Basic shapes
    nbra = vals.shape[0]
    norb = vecs.shape[-1]

    # Reconstruct per-pair vecs_lowrank
    vecs_lowrank = {}
    for i, j in itertools.product(range(nbra), range(nbra)):
        vals_ij = vals[i, j]
        vecs_ij = vecs[i, j]

        # Determine nvec for this pair: prefer explicit stored count if present
        if nvecs_per_pair is not None:
            nvec = int(nvecs_per_pair[i, j])
        else:
            mask = np.any(np.abs(vecs_ij) > 1e-12, axis=(1, 2))
            nvec = int(mask.sum())

        if nvec == 0:
            # keep zero-length arrays for consistency
            vals_i = np.zeros((0,))
            vecs_i = np.zeros((norb, norb, 0))
            right_i = np.zeros((0, norb, norb)) if rightvecs is not None else np.zeros((0, norb, norb))
        else:
            vals_i = vals_ij[:nvec].copy()
            # stored in vectorize_lowrank as (nvec, norb, norb)
            vecs_i = vecs_ij[:nvec].transpose(1, 2, 0).copy()
            if rightvecs is not None:
                right_i = rightvecs[i, j, :nvec].copy()
            else:
                # try to recover from stacked SVD/rightvecs_stacked if present
                if 'rightvecs_stacked' in lv and (i, j) in pairloc_svd:
                    st, en = pairloc_svd[(i, j)]
                    right_i = lv['rightvecs_stacked'][st:en].copy()
                elif 'rightvecs_stacked' in lv and (j, i) in pairloc_svd:
                    st, en = pairloc_svd[(j, i)]
                    right_i = lv['rightvecs_stacked'][st:en].copy()
                else:
                    right_i = np.zeros((nvec, norb, norb))

        # Determine whether this pair used joint ED (present in pairloc in either order)
        use_joint = (i, j) in pairloc or (j, i) in pairloc

        vecs_lowrank[(i, j)] = (vals_i, vecs_i, right_i, use_joint)

    # Reconstruct diagonals from diagonal_vectorized if available
    if getattr(self, 'diagonal_vectorized', None) is not None:
        diag_stack = self.diagonal_vectorized
        # diag_stack shape (2, m, norb, norb)
        diagJ_packed = diag_stack[0]
        diagKsum_packed = diag_stack[1]

        # If explicit components were stored at vectorization time, use them to reconstruct exact diag[1] and diag[2]
        comp = getattr(self, 'diagonal_vectorized_components', None)
        if comp is not None:
            diagK1_packed = comp[0]
            diagK2_packed = comp[1]

            diagJ_unpacked = unstack_tril(diagJ_packed, hermitian=hermitian)
            diagK1_unpacked = unstack_tril(diagK1_packed, hermitian=hermitian)
            diagK2_unpacked = unstack_tril(diagK2_packed, hermitian=hermitian)

            diagonal_lr = np.zeros((nbra, nbra, 3, norb, norb), dtype=diagJ_unpacked.dtype)
            diagonal_lr[:, :, 0, :, :] = diagJ_unpacked
            diagonal_lr[:, :, 1, :, :] = diagK1_unpacked
            diagonal_lr[:, :, 2, :, :] = diagK2_unpacked

        else:
            # Fallback: only K-sum available
            diagJ_unpacked = unstack_tril(diagJ_packed, hermitian=hermitian)
            diagK_unpacked = unstack_tril(diagKsum_packed, hermitian=hermitian)

            # Recreate original 3-component diagonal array: [J, K_sum, 0]
            diagonal_lr = np.zeros((nbra, nbra, 3, norb, norb), dtype=diagJ_unpacked.dtype)
            diagonal_lr[:, :, 0, :, :] = diagJ_unpacked
            # diagK_unpacked contains sum of original diag[1] + diag[2]
            diagonal_lr[:, :, 1, :, :] = diagK_unpacked
            diagonal_lr[:, :, 2, :, :] = 0.0

    else:
        diagonal_lr = None

    # Assign back to self so append_to_rdms can operate on the per-pair structures
    self.vecs_lowrank = vecs_lowrank
    self.diagonal_lr = diagonal_lr


        
def rdm2_from_rdm1(rdm1, ovlp):
    """
    1-body contribution to the 2-(transition) reduced density matrices
    """
    rdm1_contribution = ( np.einsum('ij,kl->jilk', rdm1, rdm1) - 0.5 * np.einsum('kj,il->jilk', rdm1, rdm1) ) * 1/ovlp
    return rdm1_contribution

def build_diag_mask(norb):
    """
    Function that returns a mask array for diagonal matrices of
    4D tensor with dimensions norb^4
    """
    # Build training overlaps and (t)RDMs (note that hermiticity should be used for performant code, as well as no norb^4 objects stored).
    diag_mask = np.zeros((norb, norb, norb, norb))
    for (i,j) in itertools.product(range(norb), range(norb)):
        diag_mask[i,i,j,j] = diag_mask[i,j,i,j] = diag_mask[i,j,j,i] = 1.0

    return diag_mask 


