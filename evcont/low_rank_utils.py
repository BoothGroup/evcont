#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:12:27 2024

Functions to be used when decomposing the 2-(transition)-cumulant
into low rank vectors to reduce memory and CPU scaling due
to 2-(transition)-RDMs

TODO: Adapt this into EVCont objects (start with CASSCF?)
TODO: Write relevant functions for testing stage (N^3 cost for constructing subspace Hamiltonian)

@author: KAtalar
"""

import numpy as np
import sys
import scipy
import itertools
from pyscf import scf, gto, ao2mo, fci, lib

from evcont.electron_integral_utils import get_loewdin_trafo

# Threshold on the overlap between states below which cumulant expansion is
# assumed to be not valid. In that case, the 2tRDM itself is decomposed
OVLP_THR = 1e-6

def rdm2_from_rdm1(rdm1, ovlp):
    """
    1-body contribution to the 2-(transition) reduced density matrices
    """
    # 
    rdm1_contribution = ( np.einsum('ij,kl->jilk', rdm1, rdm1) - 0.5 * np.einsum('kj,il->jilk', rdm1, rdm1) ) * 1/ovlp

    return rdm1_contribution

# TODO: For HF (or determinant solvers), add the option to discard the cumulant 
# and construct rdm2 from rdm1 using the above function

def reduce_2rdm(rdm1, rdm2, ovlp, 
                truncation_style='eigval',nvecs=10, eval_thr=10, ham_thr=0.1,
                diag_mask=None):
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
                                        Hamiltonian matrix elements is less than 'ham_thr'}
        nvecs (int): Number of low rank vectors to include
        eval_thr (float): Threshold to choose vectors based on their eval**2
        ham_thr (float): Threshold to choose vectors based on their H matrix elements (Hartree units)

        diag_mas (np.array([n,n,n,n])): Mask choosing diagonal matrices of (n,n,n,n) tensor. Computed 
                                        OTF if not given

    Output:
        diagonals (np.array([3,n,n])): Diagonal matrices of 2-transition-cumulants
        lowrank_vecs (vals_trunc, vecs_trunc): Low rank eigenvalues and eigenvectors of off-diagonal 2-cumulant
    """
    # 2-(transition)-cumulant
    cum2 = rdm2 - rdm2_from_rdm1(rdm1, ovlp)

    # Save diagonal terms in the cumulant 
    norb = rdm1.shape[0]
    norb_sq = norb * norb

    diagonals = np.zeros([3,norb,norb])
    for (i,j) in itertools.product(range(norb), range(norb)):
        diagonals[0, i, j] = cum2[ i, i, j, j]
        if i != j:
            diagonals[1, i, j] = cum2[ i, j, i, j]
            diagonals[2, i, j] = cum2[ i, j, j, i]

    # If not given to the function, compute the diagonal mask
    if diag_mask is None:
        diag_mask = build_diag_mask(norb)

    # Remove diagonals of 2-cumulant before decomposing
    mat_decomp = cum2.copy()
    mat_decomp -= diag_mask*mat_decomp

    # Check that it is hermitian and diagonalize the matrix
    assert(np.allclose(mat_decomp.reshape((norb_sq, norb_sq)), mat_decomp.reshape((norb_sq,norb_sq)).T))
    evals, evecs = scipy.linalg.eigh(mat_decomp.reshape((norb_sq, norb_sq)))

    # Select low rank vectors
    if truncation_style in ['eigval','nvec']:
        lowrank_vecs = select_lowrank(evals, evecs, norb, truncation_style=truncation_style, nvecs=nvecs, eval_thr=eval_thr)

    elif truncation_style == 'ham':
        print('Error in reduce_2rdm: Truncation based on subspace Hamiltonian elements has not been implemented yet.')
        sys.exit()

    else:
        print('Unknown truncation_style in reduce_2rdm: %s'%truncation_style)
        sys.exit()

    return diagonals, lowrank_vecs

def lowrank_hamiltonian(mol, one_RDM, S, cum_diagonal, lowrank_vecs, sao_basis=None):
    """
    Construct subspace Hamiltonian using diagonals and low rank vectors of 
    2-transition-cumulant -- O(N^3) scaling
    
    Input:
        mol (Mole object): pySCF mole object at the test geometry
        
    """
    ### Preliminaries
    ntrain = S.shape[0]
    subspace_h = np.zeros((ntrain, ntrain))

    # Initiate the mean field object to use DF integrals (no need to use kernel)
    mf = scf.RHF(mol).density_fit(auxbasis='weigend')
    
    # SAO basis
    if sao_basis is None:
        sao_basis = get_loewdin_trafo(mol.intor("int1e_ovlp"))
    
    # 1-electron integrals with DF
    h1_ao = mf.get_hcore()
    h1e_sao = np.einsum('ai,ab,bj->ij', sao_basis, h1_ao, sao_basis)
    
    subspace_h = np.einsum('...kl,kl->...', one_RDM, h1e_sao)
    # 1-RDM in AO basis
    training_1rdm_ao = np.einsum('ai,...ij,bj->...ab', sao_basis, one_RDM, sao_basis)

    # 1-body Coulomb and exchange matrices
    vj, vk = mf.with_df.get_jk(dm = training_1rdm_ao.transpose([0,1,3,2]), hermi=0)
    subspace_h += 0.5 * np.einsum('...ij,...ij->...', vj, training_1rdm_ao) / S
    subspace_h -= 0.25 * np.einsum('...ij,...ij->...', vk, training_1rdm_ao) / S

    # 2-electron integrals with DF
    Lpq_sao = ao2mo._ao2mo.nr_e2(mf.with_df._cderi, sao_basis,
        (0, sao_basis.shape[1], 0, sao_basis.shape[1]),aosym="s2",mosym="s2")
    Lpq_sao = lib.unpack_tril(Lpq_sao)
    
    # Construct the subspace Hamiltonian
    # Would be good to optimize so as to contract for the whole hamiltonian in one go...?
    for bra in range(ntrain):
        for ket in range(ntrain):
            
            ### 1-body contriubtions
            
            #subspace_h[bra, ket] = np.einsum('kl,kl->', one_RDM[bra, ket, :, :], h1e_sao)
            
            # 1-RDM in AO basis
            #training_1rdm_ao = np.einsum('ai,ij,bj->ab', sao_basis, one_RDM[bra, ket, :, :], sao_basis)
            #training_1rdm_ao_conj = np.einsum('ai,ij,bj->ab', sao_basis, one_RDM[ket, bra, :, :], sao_basis)
        
            # Coulomb and exchange matrices
            #vj, vk = mf.get_jk(dm = training_1rdm_ao, hermi=0)
            
            #training_1rdm_ao = one_RDM_ao[bra, ket,:,:]
            #training_1rdm_ao_conj = one_RDM_ao[ket, bra,:,:]
            
            #vj, vk = vj_all[bra,ket,:,:], vk_all[bra,ket,:,:]
            
            # Add the 1-body contributions of 2-RDM into subspace Hamiltonian
            #subspace_h[bra, ket] += 0.5 * np.einsum('ij,ij->', vj, training_1rdm_ao) * 1/S[bra, ket]
            #subspace_h[bra, ket] -= 0.25 * np.einsum('ij,ij->', vk, training_1rdm_ao_conj) * 1/S[bra, ket]

            ### Low rank 2-body cumulant contributions
            
            half_cont_cumulant = np.einsum('Pij,ija->Pa', Lpq_sao, lowrank_vecs[(bra, ket)][1])
            subspace_h[bra, ket] += 0.5 * np.einsum('Pa,a,Pa->', half_cont_cumulant, lowrank_vecs[(bra, ket)][0], half_cont_cumulant)

            # remove the i=j part of the ijij and ijji diagonals to avoid double counting
            double_count = np.diag(np.einsum('iia,a,iia->i', lowrank_vecs[(bra, ket)][1], lowrank_vecs[(bra, ket)][0], lowrank_vecs[(bra, ket)][1]))
            lowrank_iijj = np.einsum('iia,a,jja->ij', lowrank_vecs[(bra, ket)][1], lowrank_vecs[(bra, ket)][0], lowrank_vecs[(bra, ket)][1])
            lowrank_ijij = np.einsum('ija,a,ija->ij', lowrank_vecs[(bra, ket)][1], lowrank_vecs[(bra, ket)][0], lowrank_vecs[(bra, ket)][1]) - double_count
            lowrank_ijji = np.einsum('ija,a,jia->ij', lowrank_vecs[(bra, ket)][1], lowrank_vecs[(bra, ket)][0], lowrank_vecs[(bra, ket)][1]) - double_count

            # Add in (last two contractions can be combined due to permutational invariance?)
            subspace_h[bra, ket] += 0.5 * np.einsum('ij,Pii,Pjj->',cum_diagonal[bra, ket, 0, :, :] - lowrank_iijj, Lpq_sao, Lpq_sao)
            subspace_h[bra, ket] += 0.5 * np.einsum('ij,Pij,Pij->',cum_diagonal[bra, ket, 1, :, :] - lowrank_ijij, Lpq_sao, Lpq_sao)
            subspace_h[bra, ket] += 0.5 * np.einsum('ij,Pij,Pji->',cum_diagonal[bra, ket, 2, :, :] - lowrank_ijji, Lpq_sao, Lpq_sao)

    return subspace_h

def select_lowrank(evals, evecs, norb, truncation_style='eigval',nvecs=10, eval_thr=10):
    """
    
    """
    # Sort the eigenstates by the square of their eigenvalue
    idx = (-np.power(evals, 2)).argsort()
    evals_sort = evals[idx]
    evecs_sort = evecs[:,idx]

    # Truncate through either eigvals or a given number of vectors
    if truncation_style == 'eigval':
        nvecs = len(evals_sort[np.power(evals_sort,2) > eval_thr])

    #norb = np.sqrt(evecs_sort.shape[0],dtype=int)
    vals_trunc = evals_sort[:nvecs]
    vecs_trunc = evecs_sort[:,:nvecs].reshape((norb, norb, nvecs))

    return vals_trunc, vecs_trunc

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

