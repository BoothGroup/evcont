#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 17:13:27 2025

Low-rank decomposition of 2-body (transition) reduced density matrices

NEW IMPLEMENTATION: Joint decomposition where each low-rank vector contribute
to both Coulomb and exchange channels. 

Other implementations will be developed 

@author: Kemal Atalar
"""

import numpy as np
import sys
import scipy
import itertools
import pyscf
from pyscf import scf, gto, ao2mo, fci, lib, df

from evcont.electron_integral_utils import get_loewdin_trafo, get_integrals

def reduce_2rdm(rdm1, rdm2, ovlp, 
                truncation_style='eigval',nvecs=10, eval_thr=0.1, ham_thr=0.001,
                diag_mask=None, save_diag=False,
                mol=None,train_en=None):
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
    
    # Check that it is hermitian and diagonalize the matrix
    assert(np.allclose(mat_decomp.reshape((norb_sq, norb_sq)), mat_decomp.reshape((norb_sq,norb_sq)).T))
    evals, evecs = scipy.linalg.eigh(mat_decomp.reshape((norb_sq, norb_sq)))
    
    ########################################################################
    #### Select low rank vectors
    
    # Choose at least one vector (lower bound for dynamic truncation)
    min_nvecs = 1
    
    # Fixed truncation based on 'nvecs' parameter
    # Or dynamic truncation based on the eigenvalue magnitude
    if truncation_style in ['eigval','nvec']:
        lowrank_vecs = select_lowrank(evals, evecs, norb, truncation_style=truncation_style, 
                                      nvecs=nvecs, eval_thr=eval_thr, min_nvec=min_nvecs)
        
    
    elif truncation_style in ['ham','ham_en']:
        
        print('Error in reduce_2rdm: Hamiltonian error not implemented yet.')
        sys.exit()
        #### NEED TO CHANGE FOR THE NEW IMPLEMENTATION
        diagonals = None
        
        # Make sure the mol and training energy is given for this truncation
        if mol is None or train_en is None:
            print('Error in reduce_2rdm: Insufficient input for decomposition based on Hamiltonian error.')
            sys.exit()
            
        lowrank_vecs = select_lowrank_ham(evals, evecs, diagonals, norb,
                               rdm1, ovlp,
                               mol, train_en,
                               truncation_style=truncation_style,
                               ham_thr=ham_thr, min_nvec=min_nvecs)
        #print('Error in reduce_2rdm: Truncation based on subspace Hamiltonian elements has not been implemented yet.')
        #sys.exit()

    else:
        print('Unknown truncation_style in reduce_2rdm: %s'%truncation_style)
        sys.exit()    
    ########################################################################
    
    if not save_diag:
        diagonals = None

        
    else:
        #print('Error in reduce_2rdm: Saving diagonals not implemented yet.')
        #sys.exit()
        
        remainder = rdm2 - reconstruct_rdm2_joint(lowrank_vecs)
        
        # Save diagonals of the remainder
        diagonals = np.zeros([3,norb,norb])
        for (i,j) in itertools.product(range(norb), range(norb)):
            diagonals[0, i, j] = remainder[ i, i, j, j]
            if i != j:
                diagonals[1, i, j] = remainder[ i, j, i, j]
                diagonals[2, i, j] = remainder[ i, j, j, i]

        # If not given to the function, compute the diagonal mask
        #if diag_mask is None:
        #    diag_mask = build_diag_mask(norb)
        #mat_decomp -= diag_mask*mat_decomp

    print(np.linalg.norm(reconstruct_rdm2_joint(lowrank_vecs, diagonals) - rdm2))

          
    return lowrank_vecs, diagonals

def reconstruct_rdm2_joint(lowrank_vecs, diagonals=None):
    """
    Reconstructing the 2RDM
    """
    lr_vals, lr_vecs = lowrank_vecs
    rdm2_i = np.einsum('ija,a,kla->ijkl',lr_vecs, lr_vals, lr_vecs,optimize='optimal')
    # Add exchange part as well
    rdm2_i -= 0.5*np.einsum('kja,a,ila->ijkl',lr_vecs, lr_vals, lr_vecs,optimize='optimal')
    
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
def lowrank_hamiltonian(mol, one_RDM, S, lowrank_vecs, diagonals=None,
                        sao_basis=None, df_basis='weigend', 
                        use_diag=False):
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

    subspace_h = np.zeros((ntrain, ntrain))
    
    # Initiate the mean field object to use DF integrals (no need to use kernel)
    #mol.symmetry = False
    mf = scf.RHF(mol).density_fit(auxbasis=df_basis)
    
    # AO to SAO basis transformation
    if sao_basis is None:
        sao_basis = get_loewdin_trafo(mol.intor("int1e_ovlp"))
    
    
    # 1-electron integrals with DF
    h1_ao = mf.get_hcore()
    h1e_sao = np.einsum('ai,ab,bj->ij', sao_basis, h1_ao, sao_basis)
    
    ### 1-body contributions
    subspace_h = np.einsum('...kl,kl->...', one_RDM, h1e_sao)

    # Check if low-rank vectors have been vectorized
    if ('vals' in lowrank_vecs):
        vectorized = True
    else:
        vectorized = False
        
        
    # Construct the subspace Hamiltonian

    if not vectorized:
        # Ideally this should all be vectorized where 
        # lowrank vectors are of the form ([nbra, nket, nvec, nao, nao])
        # but for now let's do them separately in a loop for testing
        for bra in range(ntrain):
            for ket in range(ntrain):
                
                nvec = lowrank_vecs[(bra,ket)][0].shape[0]
    
                # Transform the low-rank vecs into
                lr_vecs_ao = ao2mo._ao2mo.nr_e2(lowrank_vecs[(bra, ket)][1].transpose((2,0,1)), sao_basis.T,
                (0, norb, 0, norb), aosym='s1', mosym='s1')
                lr_vecs_ao = lr_vecs_ao.reshape((nvec,norb,norb))
    
                # JK build
                vj_list, vk_list = mf.with_df.get_jk(dm=lr_vecs_ao.transpose(0,2,1), hermi=0)  # Specify hermiticity per case
                subspace_h[bra,ket] += 0.5*np.einsum('aij,aij,a->', vj_list - 0.5 * vk_list, lr_vecs_ao, lowrank_vecs[(bra, ket)][0])
    
                if use_diag:
                    print('Error in lowrank_hamiltonian: Diagonal contraction not implemented')
                    sys.exit()
                    
    else:
        nvec = lowrank_vecs['vals'].shape[2]
        
        # Group bra, ket and nvec indices together 
        # (this can be reduced further since bra and ket are related, i.e. use utril)
        lr_vecs_grouped = lowrank_vecs['vecs'].reshape([ntrain*ntrain*nvec,norb, norb])
        
        # Transform the low-rank vecs into
        lr_vecs_ao = ao2mo._ao2mo.nr_e2(lr_vecs_grouped, sao_basis.T,
        (0, norb, 0, norb), aosym='s1', mosym='s1')
        lr_vecs_ao = lr_vecs_ao.reshape((ntrain*ntrain*nvec,norb,norb))

        # JK build
        vj_list, vk_list = mf.with_df.get_jk(dm=lr_vecs_ao.transpose(0,2,1), hermi=0)  # Specify hermiticity per case
        vhf = vj_list - 0.5*vk_list

        # Reindex to separate bra, ket, nvec indices
        vhf = vhf.reshape([ntrain,ntrain,nvec, norb,norb])
        lr_vecs_ao = lr_vecs_ao.reshape([ntrain,ntrain,nvec, norb,norb])
        
        # Contruction for subspace Hamiltonian
        subspace_h += 0.5*np.einsum('xyaij,xyaij,xya->xy', vhf, lr_vecs_ao, lowrank_vecs['vals'])
    

                
    return subspace_h

def select_lowrank(evals, evecs, norb, 
                   truncation_style='eigval',nvecs=10, eval_thr=0.1, min_nvec=0):
    """
    Function to select low-rank vectors from the eigendecomposition
    """
    # Sort the eigenstates by the square of their eigenvalue
    idx = (-np.power(evals, 2)).argsort()
    evals_sort = evals[idx]
    evecs_sort = evecs[:,idx]

    # Truncate through either eigvals or a given number of vectors
    if truncation_style == 'eigval':
        nvecs = len(evals_sort[np.power(evals_sort,2) > eval_thr])
        nvecs = max(min_nvec, nvecs)
        
    #norb = np.sqrt(evecs_sort.shape[0],dtype=int)
    vals_trunc = evals_sort[:nvecs]
    vecs_trunc = evecs_sort[:,:nvecs].reshape((norb, norb, nvecs))

    return vals_trunc, vecs_trunc

def select_lowrank_ham(evals, evecs, diagonal, norb,
                       rdm1, ovlp,
                       mol, training_energy,
                       truncation_style='ham',
                       ham_thr=0.001, min_nvec=0
                       ):
    """
    Select a low rank decomposition of the RDM based on the error on
    subspace hamiltonian
    """
    
    # Sort the eigenstates by the square of their eigenvalue
    idx = (-np.power(evals, 2)).argsort()
    evals_sort = evals[idx]
    evecs_sort = evecs[:,idx]
    
    # Prepare rdms in a suitable format
    one_RDM = np.zeros([1,1,norb,norb])
    one_RDM[0,0,:,:] = rdm1
    
    cum_diagonal = np.zeros([1,1,3,norb,norb])
    cum_diagonal[0,0,:,:,:] = diagonal
    
    S = np.array([[ovlp]])
    
    # Exact element of subspace Hamiltonian
    ham_training = ovlp * training_energy
    
    # Iterate over subset
    ham_err = [1000,1000]
    nvecs = 0
    while (abs(ham_err[-1]) > ham_thr or abs(ham_err[-2]) > ham_thr) and nvecs <= norb*norb:
        # Truncate
        vals_trunc = evals_sort[:nvecs]
        vecs_trunc = evecs_sort[:,:nvecs].reshape((norb, norb, nvecs))
        lowrank_vecs = {(0,0):(vals_trunc,vecs_trunc)}
        
        # Compute subspace Hamiltonian
        ham_new = lowrank_hamiltonian(mol, one_RDM, S, cum_diagonal, 
                                      lowrank_vecs, sao_basis=None)[0,0]
        
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
        nvec_select = max(min_nvec, nvecs-2)
        
    vals_trunc = evals_sort[:nvec_select]
    vecs_trunc = evecs_sort[:,:nvec_select].reshape((norb, norb, nvec_select))
    
    return vals_trunc, vecs_trunc
    

# Attribute function to vectorize low-rank vectors for EVCont solver classes
def vectorize_lowrank(self):
    
    # Make sure a low-rank decomposition has been performed
    assert len(self.vecs_lowrank.items()) != 0
    
    # Find the largest number of vectors for each bra,ket pair
    nbra = self.overlap.shape[0]
    norb = self.one_rdm.shape[-1]
    nvec_max = 0
    for i, j in itertools.product(range(nbra), range(nbra)):
        nvec_max = max(nvec_max, self.vecs_lowrank[(i,j)][0].shape[-1])
        
    # Convert the dictionary of states into a np.array
    vecs_lr = np.zeros([nbra, nbra, nvec_max, norb, norb])
    vals_lr = np.zeros([nbra, nbra, nvec_max])
    for i, j in itertools.product(range(nbra), range(nbra)):
        lr_i = self.vecs_lowrank[(i,j)]
        nvec_i = lr_i[0].shape[-1]
        vecs_lr[i,j,:nvec_i] = lr_i[1].transpose(2,0,1) 
        vals_lr[i,j,:nvec_i] = lr_i[0]
        
    # Set this low-rank description
    self.lowrank_vectorized = {}
    self.lowrank_vectorized['vals'] = vals_lr
    self.lowrank_vectorized['vecs'] = vecs_lr
        
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

