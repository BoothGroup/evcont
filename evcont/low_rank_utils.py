#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 17:13:27 2025

Low-rank decomposition of 2-body (transition) reduced density matrices

Mixed decomposition:
    Joint ED: Joint eigenvalue decomposition where each low-rank vector 
              contribute to both Coulomb and exchange channels. 
    Coulomb SVD: SVD in the Coulomb grouping of the state indices 

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

from multiprocessing import Process, Pipe

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
# Iterative solver with timeout fallback to full diagonalization
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

########################################################################
@timeit
def reduce_2rdm(rdm1, rdm2, ovlp, 
                truncation_style='eigval',nvecs=10, eval_thr=0.1, ham_thr=0.001,
                save_diag=False, Jdiag_only=True,
                relax_amp=True, opt_no_diag=True, relax_after=True,
                use_svd=False, svd_weight = 1.0,
                iterative=False, nit=None, max_iter_time=10000,
                mol=None,train_en=None,
                min_eval=None):
    """
    Function to compress the 2-transition-RDM between a pair of training
    states using joint decomposition and diagonal corrections, and
    a variety of truncation criteria.

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

        save_diag (bool): Whether to save the diagonal corrections to make low-rank 2RDM diagonal elements exact.
        Jdiag_only (bool): Whether only Coulomb diagonals are used in the inference when determining the 
                          truncation

        # Parameters for amplitude relaxation of joint decomposition
        relax_amp (bool): Whether to perform amplitude relaxation after selecting the low-rank vectors
                         in the joint decomposition.
        opt_no_diag (bool): Whether to remove diagonal elements from the relaxation. 
                            Only applicable if relax_amp is True and save_diag is True.
        relax_after (bool): Whether to perform amplitude relaxation after selecting the low-rank 
                            vectors based on Hamiltonian error vs relaxing during the selection process. 
                            Only applicable if relax_amp is True and truncation_style is 'ham' or 'ham_en'.
        
        # Parameters relating to SVD decomposition
        use_svd (bool): Whether to perform SVD in addition to the joint decomposition and choose the more compact representation.
        svd_weight (float): Weight for choosing the SVD decomposition over joint ED.

        # Parameters for iterative diagonalization of the 2RDM for larger systems
        iterative (bool): Whether to use iterative diagonalization for the low-rank decomposition.
        nit (int): Number of eigenvalues/vectors to compute in the iterative diagonalization. Only applicable if iterative is True.
        max_iter_time (float): Maximum time in seconds to allow for the iterative diagonalization 
                                before falling back to full diagonalization.

        # Parameters relevant for Hamiltonian error truncation
        mol (pyscf Mole object): Molecule object that is used for computing the Hamiltonian error
        train_en (float):  Energy of the training geometry used for the truncation
        min_eval (float or None): When using Hamiltonian-based truncation, include all tied vectors
                   whose eigenvalue magnitude equals the boundary eigenvalue magnitude. If provided,
                   ties are detected against this value; otherwise the boundary value is used.

    Output:
        lowrank_vecs (vals_trunc, vecs_trunc): Low rank eigenvalues and eigenvectors of the 2tRDM decomposition.
        diagonals (np.array([3,n,n])): Diagonal correction matrices to be added to the low-rank reconstructed 2RDM.
        joint (bool): Whether the low-rank vectors are from the joint decomposition (True) or SVD (False)

    """

    # Matrix to decompose
    mat_decomp = rdm2
    
    norb = rdm1.shape[0]
    norb_sq = norb * norb

    # Ensure the matrix is hermitian
    if not np.allclose(rdm2.reshape((norb_sq, norb_sq)), rdm2.reshape((norb_sq,norb_sq)).T):
        print('Warning: 2RDM was not Hermitian.')
        # Hermitise
        rdm2 = 0.5 * (rdm2 + np.einsum('...abcd->...cdab',rdm2.conj()))

    # Matrix to decompose in the joint decomposition
    # Refactor the 2(t)RDM such that its eigenvectors solely correponds to Coulomb grouping
    mat_decomp = 4/3*mat_decomp + 2/3*np.einsum('ijkl->ilkj',mat_decomp)
    
    # Check the nit is given is iterative is True
    if iterative and nit is None:
        print('Error in reduce_rdm: nit is not given for iterative diagonalization')
        sys.exit()
        
    # Diagonalize
    if not iterative:
        evals, evecs = scipy.linalg.eigh(mat_decomp.reshape((norb_sq, norb_sq)))
    else:
        evals, evecs = try_iterative_diag(mat_decomp.reshape((norb_sq, norb_sq)),
                                            k=nit, 
                                            which='LM', 
                                            max_time=max_iter_time)
        
    rightvecs = None
    joint = True # Joint decomp
    
    ########################################################################
    #### Select low rank vectors
    
    # Choose at least one vector (lower bound for dynamic truncation)
    min_nvecs = 1

    # Unless the 2RDM is close to zero
    if np.linalg.norm(rdm2) < 1e-8:
        print("Warning in reduce_2rdm: RDM2 is close to zero; setting base_count to 0 to avoid selecting vectors based on noise.")
        min_nvecs = 0
    
    # Make sure the mol and training energy is given for this truncation
    if truncation_style in ['ham','ham_en']:
        if mol is None or train_en is None:
            print('Error in reduce_2rdm: Insufficient input for decomposition based on Hamiltonian error.')
            sys.exit()
            
    elif truncation_style not in ['eigval','nvec']:
        print('Unknown truncation_style in reduce_2rdm: %s'%truncation_style)
        sys.exit()    
        
    # Fixed truncation based on 'nvecs' parameter
    # Or dynamic truncation based on the eigenvalue magnitude / Hamiltonian error
    if truncation_style in ['eigval','nvec']:

        lowrank_vecs_joint = select_lowrank(evals, evecs, norb, rightvecs=rightvecs, 
                                            truncation_style=truncation_style, 
                                            nvecs=nvecs, eval_thr=eval_thr, min_nvec=min_nvecs,
                                            relax_amp=relax_amp, rdm2=rdm2,
                                            remove_diagopt=(opt_no_diag and save_diag),
                                            jdiag_only=Jdiag_only)
        
    elif truncation_style in ['ham','ham_en']:

        ham_selector = select_lowrank_ham_relaxed if (joint and relax_amp and not relax_after) else select_lowrank_ham
        out_ham = ham_selector(evals, evecs, joint, norb,
                                rdm2, rdm1, ovlp,save_diag,
                                mol, train_en, Jdiag_only,
                                rightvecs=rightvecs,
                                truncation_style=truncation_style,
                                ham_thr=ham_thr, min_nvec=min_nvecs,
                                min_eval=min_eval,
                                relax_amp=relax_amp,
                                remove_diagopt=(opt_no_diag and save_diag))
        lowrank_vecs_joint = out_ham[:-1]
        #ham_err_joint = out_ham[-1]
        
    # Choose between joint decomposition and SVD based on the compactness of the representation (weighted by svd_weight) and/or Hamiltonian error
    if not use_svd:
        lowrank_vecs = lowrank_vecs_joint

    else:

        if not iterative:
            evecs2, evals2, rightvecs2 = scipy.linalg.svd(rdm2.reshape((norb_sq, norb_sq)))
        else:
            evecs2, evals2, rightvecs2 = svds(rdm2.reshape((norb_sq, norb_sq)), k=nit, which='LM')

        if truncation_style in ['eigval','nvec']:

            lowrank_vecs_svd = select_lowrank(evals2, evecs2, norb, rightvecs=rightvecs2, 
                                            truncation_style=truncation_style, 
                                            nvecs=nvecs, eval_thr=eval_thr, min_nvec=min_nvecs, 
                                            relax_amp=False)
        
        elif truncation_style in ['ham','ham_en']:
                
            out_ham = select_lowrank_ham(evals2, evecs2, False, norb,
                                    rdm2, rdm1, ovlp,save_diag,
                                    mol, train_en, Jdiag_only,
                                    rightvecs=rightvecs2,
                                    truncation_style=truncation_style,
                                    ham_thr=ham_thr, min_nvec=min_nvecs,
                                    min_eval=min_eval, relax_amp=False)
            lowrank_vecs_svd = out_ham[:-1]
            #ham_err_svd = out_ham[-1]
        
        # Check which one is more compact
        if truncation_style in ['ham','ham_en']:
            if len(lowrank_vecs_joint[0])*svd_weight < len(lowrank_vecs_svd[0]):
                lowrank_vecs = lowrank_vecs_joint
                #ham_err = ham_err_joint
            else:
                print('**Using SVD')
                lowrank_vecs = lowrank_vecs_svd
                joint = False
                #ham_err = ham_err_svd
                
        # TODO: Add considerations for norm error; not just compactness
        # SVD as well
        elif truncation_style in ['eigval']:
            if len(lowrank_vecs_joint[0])*svd_weight < len(lowrank_vecs_svd[0]):
                lowrank_vecs = lowrank_vecs_joint
            else:
                print('**Using SVD')
                lowrank_vecs = lowrank_vecs_svd
                joint = False
                #ham_err = ham_err_svd
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
        
        remainder = rdm2 - reconstruct_rdm2_joint(lowrank_vecs,joint=joint)
        
        # Save diagonals of the remainder
        diagonals = np.zeros([3,norb,norb])
        for (i,j) in itertools.product(range(norb), range(norb)):
            diagonals[0, i, j] = remainder[ i, i, j, j]
            if (not Jdiag_only) and i != j:
                diagonals[1, i, j] = remainder[ i, j, i, j]
                diagonals[2, i, j] = remainder[ i, j, j, i]

    reconstructed_rdm2 = reconstruct_rdm2_joint(lowrank_vecs, diagonals, joint=joint)
    if mol is not None:
        # Compute the energy error as well
        h1, h2 = get_integrals(mol, get_basis(mol))
        ham_inferred = 0.5*np.einsum('pqrs,pqrs->', reconstructed_rdm2, h2,optimize='optimal') + \
                      np.einsum('pq,pq->', rdm1, h1)
        # Recompute train hamiltonian
        train_ham = 0.5*np.einsum('pqrs,pqrs->', rdm2, h2,optimize='optimal') + \
                      np.einsum('pq,pq->', rdm1, h1)
        
        #ham_error = ham_inferred - train_en*ovlp
        ham_error = ham_inferred - train_ham
        #print(f'-- Verifying en_train*ovlp - ham_train: {train_en*ovlp - train_ham:.1e} or en_train*|ovlp| - ham_train: {train_en*np.abs(ovlp) - train_ham:.1e}')
        print(f'-- Norm error: {np.linalg.norm(reconstructed_rdm2 - rdm2):.1e}, Hamiltonian error: {ham_error:.1e}, nvecs: {len(lowrank_vecs[0])}')
        
    else:
        print(f'-- Norm error: {np.linalg.norm(reconstructed_rdm2 - rdm2):.1e}, nvecs: {len(lowrank_vecs[0])}')
    
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
                rdm2_i[ i, j, j, i] += diagonals[2, i, j] 
    
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
                  df_response=False, verbose=False):
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
    if lowrank_vecs['has_ed']:
        norb = lowrank_vecs['vecs_stacked'].shape[-1]
    else:
        norb = lowrank_vecs['vecs_svd_stacked'].shape[-1]
    nvec = lowrank_vecs['vals'].shape[2]
    ntrain = lowrank_vecs['vals'].shape[0]

    ######################################################
    ######### COMPUTE PRELIMINARIES & FOCK BUILDS
    ######################################################
    # Initiate grad object
    grad_obj = mf_grad(mf)
    # Set verbose level for detailed timing output (6 or higher shows timer_debug1)
    if verbose:
        grad_obj.verbose = 6
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
            # Stick to PySCF's own threading; joblib overhead slowed plain get_jk
            vj_list, vk_list = get_jk(dm=lr_vecs_ao.transpose(0,2,1), hermi=0)

        vhf = vj_list - 0.5*vk_list
        
        # Grad JK builds
        # TODO: Add auxbasis_response in the future, for now ignore it
        with log_time("JK Grad Builds (2)"):
            lr_vecs_ao = np.ascontiguousarray(lr_vecs_ao)
            lr_vecs_ao_T = np.ascontiguousarray(lr_vecs_ao.transpose(0,2,1))

            vj_grad_list, vk_grad_list = grad_obj.get_jk(dm=lr_vecs_ao, hermi=0) 
            vj_grad_list_t, vk_grad_list_t = grad_obj.get_jk(dm=lr_vecs_ao_T, hermi=0) 


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
            
            svd_vecs_ao = np.ascontiguousarray(svd_vecs_ao)
            svd_rightvecs_ao = np.ascontiguousarray(svd_rightvecs_ao)

        # J builds
        with log_time("J Builds (2)"):
            vj_r_list, _ = get_jk(dm=svd_rightvecs_ao, hermi=0, with_k=False)
            vj_l_list, _ = get_jk(dm=svd_vecs_ao, hermi=0, with_k=False)

        # Grad JK builds
        # TODO: Add auxbasis_response in the future, for now ignore it
        with log_time("J Grad Builds (2)"):
            vj_lgrad_list = grad_obj.get_j(dm=svd_vecs_ao, hermi=0) 
            vj_rgrad_list = grad_obj.get_j(dm=svd_rightvecs_ao, hermi=0)

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

        # Flatten
        orig_shape = diagJ_ao.shape[:2]
        flat_diagJ_ao = diagJ_ao.reshape(orig_shape[0] * orig_shape[1], *diagJ_ao.shape[2:])
        
        # JK Builds
        with log_time("Diag J Builds (1)"):
            vj_list = get_jk(dm=flat_diagJ_ao, hermi=0, with_k=False)[0]

        # JK grad builds
        with log_time("Diag Grad J Builds (1)"):
            vj_grad_list = grad_obj.get_j(dm=flat_diagJ_ao.transpose(0,2,1), hermi=0)

        # Unflatten
        vj_unflat = vj_list.reshape(*orig_shape, *vj_list.shape[1:])
        vj = unstack_tril(vj_unflat,hermitian=hermitian)
        
        vj_grad_unflat = vj_grad_list.reshape(*orig_shape, *vj_grad_list.shape[1:])  
        vj_grad = unstack_tril(vj_grad_unflat,hermitian=hermitian)
    
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
                vk_list = get_jk(dm=flat_diagK_ao, hermi=0, with_j=False)[1]
                vk_t_list = get_jk(dm=flat_diagK_ao_T, hermi=0, with_j=False)[1]

            # JK grad builds
            with log_time("Diag Grad K Builds (1)"):
                vk_grad_list = grad_obj.get_jk(dm=flat_diagK_ao.transpose(0,2,1), hermi=0, with_j=False)[1]

            # Unflatten
            vk_unflat = vk_list.reshape(*orig_shape, *flat_diagK_ao.shape[1:])  # (3, 4, 5, 6)
            vk = unstack_tril(vk_unflat,hermitian=hermitian)
            
            vk_t_unflat = vk_t_list.reshape(*orig_shape, *flat_diagK_ao.shape[1:])  # (3, 4, 5, 6)
            vk_t = unstack_tril(vk_t_unflat,hermitian=hermitian)
            
            vk_grad_unflat = vk_grad_list.reshape(*orig_shape, *vk_grad_list.shape[1:])  
            vk_grad = unstack_tril(vk_grad_unflat,hermitian=hermitian)

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
# Amplitude relaxation for joint decomposition
def relax_coef(rdm2, vec, diag=False, jdiag_only=True):
    # Basis functions
    B = np.einsum('ija,kla->aijkl', vec, vec, optimize='optimal') \
        - 0.5 * np.einsum('ila,kja->aijkl', vec, vec, optimize='optimal')

    # Build masked tensors first (target and basis), then form normal equations.
    if diag:
        norb_loc = rdm2.shape[0]
        mask = np.ones((norb_loc, norb_loc, norb_loc, norb_loc), dtype=rdm2.dtype)

        for (i, j) in itertools.product(range(norb_loc), range(norb_loc)):
            mask[i, i, j, j] = 0.0
            if not jdiag_only and i != j:
                mask[i, j, i, j] = 0.0
                mask[i, j, j, i] = 0.0

        rdm2_opt = rdm2 * mask
        B_opt = B * mask[None, :, :, :, :]
    else:
        rdm2_opt = rdm2
        B_opt = B

    # Gram matrix and projection in the masked space
    G = np.einsum('aijkl,bijkl->ab', B_opt, B_opt, optimize='optimal')
    b = np.einsum('ijkl,aijkl->a', rdm2_opt, B_opt, optimize='optimal')

    # Solve the linear system
    coef, res, rank, sval = np.linalg.lstsq(G, b, rcond=None)

    return coef

def select_lowrank(evals, evecs, norb, 
                   rightvecs=None,
                   truncation_style='eigval',nvecs=10, eval_thr=0.1, min_nvec=0,
                   relax_amp=True, rdm2=None, remove_diagopt=True, jdiag_only=True):
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

    # Relax amplitudes
    if relax_amp:
        vals_new = relax_coef(rdm2, vecs_trunc, diag=remove_diagopt, jdiag_only=jdiag_only)
        print('Relaxing amplitudes; norm of change in vals:', np.linalg.norm(vals_new - vals_trunc))
        vals_trunc = vals_new

    return vals_trunc, vecs_trunc, rightvecs_trunc

def select_lowrank_ham(evals, evecs, joint, norb,
                       rdm2, rdm1, ovlp, save_diag,
                       mol, training_energy, Jdiag_only,
                       rightvecs=None,
                       density_fit=True,
                       truncation_style='ham',
                       ham_thr=0.001, 
                       min_nvec=1, min_eval=None,
                       relax_amp=True, 
                       remove_diagopt=False,
                       ):
    """
    Select a low rank decomposition of the RDM based on the error on
    subspace hamiltonian.
    
    Strategy:
    1. Sort eigenstates by the square of their eigenvalue
    2. Choose a base set based on min_nvec and min_eval (using magnitude to handle negative evals)
    3. Compute initial rdm2_cum from this base set using reconstruct_rdm2_joint
    4. Continue adding vectors and checking convergence on hamiltonian error until converged
    """
    # Check if right eigenvectors are given
    if rightvecs is None:
        rightvecs = evecs.T

    # Sort the eigenstates by the square of their eigenvalue
    idx = (-np.power(evals, 2)).argsort()
    evals_sort = evals[idx]
    evecs_sort = evecs[:,idx]
    rightvecs_sort = rightvecs[idx,:]

    # For direct contraction
    h1, h2 = get_integrals(mol, get_basis(mol))
    e1 = lib.einsum('ij,ij->', h1, rdm1)
    
    # Exact element of subspace Hamiltonian - compute explicitly instead of using train_en*ovlp
    ham_training = 0.5*lib.einsum('pqrs,pqrs->', rdm2, h2, optimize='optimal') + \
                   lib.einsum('pq,pq->', rdm1, h1)
    #ham_training = ovlp * training_energy

    max_nvec = min(norb * norb, evals_sort.shape[0])
    
    # Step 2: Determine base set from min_nvec and min_eval
    base_count = min_nvec
    if min_eval is not None:
        # Include all vectors with eigenvalue magnitude^2 >= min_eval^2
        # evals_sort is already sorted by magnitude^2 descending
        min_eval_mag2 = np.power(min_eval, 2)
        eval_mag2 = np.power(evals_sort, 2)
        
        # Find how many vectors have magnitude^2 >= threshold
        count_above_threshold = np.sum(eval_mag2 >= min_eval_mag2)
        base_count = max(min_nvec, count_above_threshold)
    
    # Step 3: Compute initial rdm2_cum from base set
    base_vals = evals_sort[:base_count]
    base_vecs = evecs_sort[:, :base_count].reshape((norb, norb, base_count))
    base_rightvecs = rightvecs_sort[:base_count, :].reshape((base_count, norb, norb))
    
    rdm2_cum = reconstruct_rdm2_joint((base_vals, base_vecs, base_rightvecs), 
                                       diagonals=None, joint=joint)
    
    # Apply diagonal corrections to base set if requested
    if save_diag:
        for (i, j) in itertools.product(range(norb), range(norb)):
            rdm2_cum[i, i, j, j] = rdm2[i, i, j, j]
            if not Jdiag_only and i != j:
                rdm2_cum[i, j, i, j] = rdm2[i, j, i, j]
                rdm2_cum[i, j, j, i] = rdm2[i, j, j, i]
    
    # Compute Hamiltonian error for the base set
    e2_base = 0.5 * lib.einsum('ijkl,ijkl->', h2, rdm2_cum,optimize='optimal')
    ham_base = e1 + e2_base
    
    if truncation_style == 'ham':
        ham_err_base = ham_training - ham_base
    elif truncation_style == 'ham_en':
        ham_err_base = training_energy - ham_base / ovlp
    
    # Step 4: Continue from base_count, checking convergence
    ham_err_list = [ham_err_base]
    nvec_select = base_count 

    for k in range(base_count, max_nvec):
        # Add contribution from k-th vector
        v_left = evecs_sort[:, k].reshape(norb, norb)
        v_right = rightvecs_sort[k, :].reshape(norb, norb)
        contrib = evals_sort[k] * np.einsum('ij,kl->ijkl', v_left, v_right.conj())
        if joint:
            contrib -= 0.5 * evals_sort[k] * np.einsum('kj,il->ijkl', v_left, v_right.conj())
        rdm2_cum += contrib

        # Apply diagonal corrections if requested
        if save_diag:
            for (i, j) in itertools.product(range(norb), range(norb)):
                rdm2_cum[i, i, j, j] = rdm2[i, i, j, j]
                if (not Jdiag_only) and i != j:
                    rdm2_cum[i, j, i, j] = rdm2[i, j, i, j]
                    rdm2_cum[i, j, j, i] = rdm2[i, j, j, i]

        # Compute Hamiltonian and error
        e2 = 0.5 * lib.einsum('ijkl,ijkl->', h2, rdm2_cum,optimize='optimal')
        ham_new = e1 + e2
        
        if truncation_style == 'ham':
            ham_err = ham_training - ham_new
        elif truncation_style == 'ham_en':
            ham_err = training_energy - ham_new / ovlp
        
        ham_err_list.append(ham_err)
        
        # Check for convergence (two-step robustness check)
        # Need at least 2 errors to compare, and both current and previous must be below threshold
        if len(ham_err_list) >= 2:
            if abs(ham_err_list[-1]) <= ham_thr and abs(ham_err_list[-2]) <= ham_thr:
                # Both current and previous errors are below threshold
                # This means convergence was achieved at the previous iteration k-1
                nvec_select = k
                break

        # Check for last iteration to set to max if never converged
        if k == max_nvec - 1:
            nvec_select = max_nvec

    # Return truncated decomposition
    vals_trunc = evals_sort[:nvec_select]
    vecs_trunc = evecs_sort[:, :nvec_select].reshape((norb, norb, nvec_select))
    rightvecs_trunc = rightvecs_sort[:nvec_select, :].reshape((nvec_select, norb, norb))

    # Relax amplitudes
    if relax_amp:
        vals_new = relax_coef(rdm2, vecs_trunc, diag=remove_diagopt, jdiag_only=Jdiag_only)
        print('Relaxing amplitudes; norm of change in vals:', np.linalg.norm(vals_new - vals_trunc))
        vals_trunc = vals_new

    return vals_trunc, vecs_trunc, rightvecs_trunc, ham_err_list[-1] if ham_err_list else 0.0


def select_lowrank_ham_relaxed(evals, evecs, joint, norb,
                               rdm2, rdm1, ovlp, save_diag,
                               mol, training_energy, Jdiag_only,
                               rightvecs=None,
                               density_fit=True,
                               truncation_style='ham',
                               ham_thr=0.001,
                               min_nvec=1, min_eval=None,
                               relax_amp=True,
                               remove_diagopt=False,
                               ):
    """
    Select a joint low-rank decomposition based on Hamiltonian error while
    re-optimizing the selected amplitudes after each truncation step.

    This path is intended only for the joint-decomposition case with
    amplitude relaxation enabled.
    """
    # This selector is intentionally narrow to keep behavior predictable.
    if not joint:
        raise ValueError('select_lowrank_ham_relaxed only supports joint=True')
    if not relax_amp:
        raise ValueError('select_lowrank_ham_relaxed requires relax_amp=True')

    # For Hermitian ED, right vectors are the transpose if not supplied.
    if rightvecs is None:
        rightvecs = evecs.T

    # Process candidates from largest |eigenvalue|^2 to smallest.
    idx = (-np.power(evals, 2)).argsort()
    evals_sort = evals[idx]
    evecs_sort = evecs[:, idx]
    rightvecs_sort = rightvecs[idx, :]

    # Precompute one-electron and exact training Hamiltonian pieces once.
    h1, h2 = get_integrals(mol, get_basis(mol))
    e1 = lib.einsum('ij,ij->', h1, rdm1)
    ham_training = 0.5 * lib.einsum('pqrs,pqrs->', rdm2, h2, optimize='optimal') + \
                   lib.einsum('pq,pq->', rdm1, h1)

    max_nvec = min(norb * norb, evals_sort.shape[0])

    base_count = min_nvec
    if min_eval is not None:
        # Optionally force all vectors above a minimum |eigenvalue| into the base set.
        min_eval_mag2 = np.power(min_eval, 2)
        eval_mag2 = np.power(evals_sort, 2)
        count_above_threshold = np.sum(eval_mag2 >= min_eval_mag2)
        base_count = max(min_nvec, count_above_threshold)

    def build_relaxed_prefix(prefix_count):
        # Build the current prefix basis and re-fit amplitudes on the full target rdm2.
        vecs_prefix = evecs_sort[:, :prefix_count].reshape((norb, norb, prefix_count))
        rightvecs_prefix = rightvecs_sort[:prefix_count, :].reshape((prefix_count, norb, norb))
        vals_prefix = relax_coef(
            rdm2,
            vecs_prefix,
            diag=remove_diagopt,
            jdiag_only=Jdiag_only,
        )

        # Reconstruct the relaxed 2RDM from fitted amplitudes.
        rdm2_prefix = reconstruct_rdm2_joint(
            (vals_prefix, vecs_prefix, rightvecs_prefix),
            diagonals=None,
            joint=True,
        )

        # If diagonal terms are saved separately in production, mirror that here
        # so selection is based on the same Hamiltonian expression used downstream.
        if save_diag:
            for (i, j) in itertools.product(range(norb), range(norb)):
                rdm2_prefix[i, i, j, j] = rdm2[i, i, j, j]
                if not Jdiag_only and i != j:
                    rdm2_prefix[i, j, i, j] = rdm2[i, j, i, j]
                    rdm2_prefix[i, j, j, i] = rdm2[i, j, j, i]

        # Evaluate the Hamiltonian error of this relaxed prefix.
        ham_prefix = e1 + 0.5 * lib.einsum('ijkl,ijkl->', h2, rdm2_prefix, optimize='optimal')
        if truncation_style == 'ham':
            ham_err = ham_training - ham_prefix
        elif truncation_style == 'ham_en':
            ham_err = training_energy - ham_prefix / ovlp
        else:
            raise ValueError(f'Unknown truncation_style in select_lowrank_ham_relaxed: {truncation_style}')

        return vals_prefix, vecs_prefix, rightvecs_prefix, ham_err

    ham_err_list = []
    vals_trunc = np.zeros((0,))
    vecs_trunc = np.zeros((norb, norb, 0))
    rightvecs_trunc = np.zeros((0, norb, norb))
    nvec_select = base_count

    # Grow the prefix one vector at a time, relax, then test Hamiltonian error.
    for prefix_count in range(base_count, max_nvec + 1):
        vals_prefix, vecs_prefix, rightvecs_prefix, ham_err = build_relaxed_prefix(prefix_count)
        ham_err_list.append(ham_err)
        vals_trunc = vals_prefix
        vecs_trunc = vecs_prefix
        rightvecs_trunc = rightvecs_prefix
        nvec_select = prefix_count

        # Two-step robustness check: require two consecutive below-threshold errors.
        if len(ham_err_list) >= 2:
            if abs(ham_err_list[-1]) <= ham_thr and abs(ham_err_list[-2]) <= ham_thr:
                nvec_select = prefix_count
                break

    # Defensive slicing; arrays are already prefix-sized, but this keeps output explicit.
    vals_trunc = vals_trunc[:nvec_select]
    vecs_trunc = vecs_trunc[:, :, :nvec_select]
    rightvecs_trunc = rightvecs_trunc[:nvec_select]

    return vals_trunc, vecs_trunc, rightvecs_trunc, ham_err_list[-1] if ham_err_list else 0.0
    
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
    elif len(np.concatenate(vecs_lr)) == 0:
        has_ed = False
      
    # Check if any (t)RDM used SVD
    has_svd = True
    if len(vecs_svd_lr) == 0:
        has_svd = False
    elif len(np.concatenate(vecs_svd_lr)) == 0:
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
        #'vecs': vecs_lr,
        'hermitian': hermitian,
        'has_ed': has_ed,
        'has_svd': has_svd,
        'nvecs': nvecs_per_pair,
        'norb': norb,
        'ntrain': nbra
    }

    if has_ed:
        #self.lowrank_vectorized['rightvecs'] = rightvecs_lr
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


