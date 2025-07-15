#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 15:02:20 2024

Script to test low rank construction of eigenvector continuation

@author: Kemal Atalar
"""

import numpy as np
import time

from pyscf import gto, fci, scf, lib, ao2mo, mcscf, df

from evcont.FCI_EVCont import FCI_EVCont_obj
#from evcont.CASCI_EVCont import CAS_EVCont_obj

from evcont.electron_integral_utils import get_basis, get_integrals, get_loewdin_trafo

#from evcont.ab_initio_gradients_loewdin import get_multistate_energy_with_grad_and_NAC

from evcont.ab_initio_eigenvector_continuation import (
    approximate_multistate_lowrank_OAO,
    approximate_multistate_OAO,
    approximate_multistate
)

from evcont.ab_initio_gradients_loewdin import (
    get_lowrank_en_with_grad_and_NAC    ,
    get_multistate_energy_with_grad_and_NAC,
    get_grad_elec_OAO_customERI
)

#from evcont.FCI_NAC import get_FCI_energy_with_grad_and_NAC_withsym
#from pyscf.mcscf import CASCI
#import pickle

import matplotlib.pylab as plt
#import matplotlib as mpl
plt.style.use('default')

nroots_evcont = 1
cibasis = 'canonical'
#cibasis = 'OAO'

df_basis = 'weigend'
df_basis = 'cc-pvdz-jkfit'

natom = 6

cont_solver = 'CAS'
cont_solver = 'FCI'

cassolver='SS-CASSCF'
#cassolver='CASCI'
ncas, neleca = 2,2
figsave = True

#plot_extensive = False
fix_singlet = False
#withMolcas = False

# Whether to use FCI as a comparison if a CAS solver is used
fci_done = False

fix_sym = 'A1g'
fix_sym = None

if fix_sym == None:
    mol_sym = False
else:
    mol_sym = True
    
#lowrank_kwargs = {'truncation_style':'nvec', 'nvecs':5}
lowrank_kwargs = {'truncation_style':'eigval', 'eval_thr':1e-15}
#lowrank_kwargs = {'truncation_style':'eigval', 'eval_thr':1e-3}
#lowrank_kwargs = {'truncation_style':'ham', 'ham_thr':0.002}
#lowrank_kwargs = {'truncation_style':'ham_en', 'ham_thr':0.0002}

vectorize = True

#test_range = np.linspace(0.8, 3.0,40)
test_range = np.linspace(0.8, 3.0,15)
#test_range = np.linspace(0.4, 1.5,20)

def get_mol(positions):
    mol = gto.Mole()

    mol.build(
        atom=[("H", pos) for pos in positions],
        #basis="sto-3g",
        basis="sto-6g",
        #basis="6-31g",
        symmetry=mol_sym,
        unit="Bohr",
        verbose=0
    )

    return mol

mol_dummy = get_mol([(x, 0.0, 0.0) for x in test_range[0] * np.arange(natom)])

# Set fci solver to be used
if fix_sym is None:
    myci = fci.direct_spin0.FCI()
else:
    myci = fci.direct_spin0_symm.FCI(mol_dummy)
    myci.wfnsym = fix_sym
    
#myci = fci.direct_spin0.FCI()
#myci = fci.direct_spin1.FCISolver()
    

if fix_singlet:
    fci.addons.fix_spin_(myci,ss=0) # Fix spin

equilibrium_dist = 1.78596

equilibrium_pos = np.array([(x * equilibrium_dist, 0.0, 0.0) for x in range(10)])

trainig_dists = [0.97, 1.76]#, 2.60]
#trainig_dists = np.linspace(0.97,2.60,5)

if cont_solver == 'FCI':
    continuation_object = FCI_EVCont_obj(nroots=nroots_evcont,
                                         cibasis=cibasis,cisolver=myci,
                                         irrep_name=fix_sym,
                                         lowrank=True,
                                         **lowrank_kwargs)
    
    continuation_object_full = FCI_EVCont_obj(nroots=nroots_evcont,
                                         cibasis=cibasis,cisolver=myci,
                                         irrep_name=fix_sym)

else:
    continuation_object = CAS_EVCont_obj(ncas, neleca,nroots=nroots_evcont,
                                         solver=cassolver,
                                         lowrank=True,
                                         **lowrank_kwargs)
    
    continuation_object_full = CAS_EVCont_obj(ncas, neleca,nroots=nroots_evcont,
                                         solver=cassolver)
 
trn_geometries = []
# Generate training data + prepare training models
for i, dist in enumerate(trainig_dists):
    positions = [(x, 0.0, 0.0) for x in dist * np.arange(natom)]
    # Add the geometry to set of training points
    trn_geometries.append(positions)
    
    # Build molecule    
    mol = get_mol(positions)
    if cont_solver == 'CAS':
        continuation_object.append_to_rdms(mol,debug=False)
    else:
        continuation_object.append_to_rdms(mol)

    #continuation_object.append_to_rdms_new(mol)
    continuation_object_full.append_to_rdms(mol)

# If vectorize
if vectorize:
    continuation_object.vectorize_lowrank(hermitian=True)
    vecs_lr = continuation_object.lowrank_vectorized
else:
    vecs_lr = continuation_object.vecs_lowrank
    
# Save
i = 'final'
np.save("overlap_{}.npy".format(i), continuation_object.overlap)
np.save("one_rdm_{}.npy".format(i), continuation_object.one_rdm)

np.save("diagonal_lr_{}.npy".format(i), continuation_object.diagonal_lr)
np.save("lowrank_vecs_{}.npy".format(i), continuation_object.vecs_lowrank)

np.save('trn_geometries_{}.npy'.format(i), trn_geometries)

lr_tot = 0.; lr_n_eval = 0
train_lowrank_en = []
train_en = []
for i, test_dist in enumerate(trainig_dists):
    print(i)
    positions = [(x, 0.0, 0.0) for x in test_dist * np.arange(natom)]
    
    mol = get_mol(positions)
    h1, h2 = get_integrals(mol, get_basis(mol))
    
    # Continuation
    start = time.time()

    en_continuation_ms, vec = approximate_multistate_lowrank_OAO(
        mol, 
        continuation_object.one_rdm,  
        vecs_lr,
        None, #continuation_object.diagonal_lr, 
        continuation_object.overlap,
        nroots=nroots_evcont
    )
    lr_tot += (time.time()-start); lr_n_eval += 1
    
    train_lowrank_en += [en_continuation_ms]

    en_continuation_ms, vec = approximate_multistate_OAO(
        mol,
        continuation_object_full.one_rdm,
        continuation_object_full.two_rdm,
        continuation_object_full.overlap,
        nroots=nroots_evcont
    )
    
    train_en += [en_continuation_ms]
    
#1/0

# Prediction on test dataset and comparison against FCI results
fci_en = np.zeros([len(test_range),nroots_evcont])
ref_en = np.zeros([len(test_range),nroots_evcont])
hf_en = np.zeros([len(test_range)])
cont_en = np.zeros([len(test_range),nroots_evcont+1])
cont_lowrank_en = np.zeros([len(test_range),nroots_evcont+1])
for i, test_dist in enumerate(test_range):
    print(i)
    positions = [(x, 0.0, 0.0) for x in test_dist * np.arange(natom)]
    
    mol = get_mol(positions)
    h1, h2 = get_integrals(mol, get_basis(mol,'canonical'))
    
    print('   low rank - start')
    # Continuation
    start = time.time()
    en_continuation_ms, vec = approximate_multistate_lowrank_OAO(
        mol, 
        continuation_object.one_rdm, 
        vecs_lr,
        None, #continuation_object.diagonal_lr, 
        continuation_object.overlap,
        nroots=nroots_evcont+1,
        df_basis=df_basis
    )
    
    out = get_lowrank_en_with_grad_and_NAC(mol, continuation_object.one_rdm, 
                                           continuation_object.overlap,
                                           vecs_lr, None, 
                                           nroots=nroots_evcont+1,
                                           df_basis=df_basis)
    
    lr_tot += (time.time()-start); lr_n_eval += 1
    
    print('   low rank - finish - %.1f sec'%(time.time()-start))

    #cont_lowrank_en += [en_continuation_ms]
    cont_lowrank_en[i,:] = en_continuation_ms
    
    ## HF and FCI

    mf = scf.RHF(mol).density_fit(auxbasis=df_basis)
    # Note that in performant code, we don't actually need to run HF at the training points,
    # just have access to the get_jk function.
    ehf = mf.scf()
    hf_en[i] = ehf
    assert(mf.converged)

    # Do FCI for the exact energy
    h1_ao = mf.get_hcore()
    Lpq_ao = lib.unpack_tril(mf.with_df._cderi)
    Lpq_mo = lib.einsum('pi,qj,Lpq->Lij', mf.mo_coeff, mf.mo_coeff, Lpq_ao)
    df_eri = lib.einsum('Pij,Pkl->ijkl', Lpq_mo, Lpq_mo)
    h1e_mo = np.einsum('ai,ab,bj->ij', mf.mo_coeff, h1_ao, mf.mo_coeff)
    #print(h1e_mo, df_eri, mol.nao, mol.nelec)

    # DF-ERI gradients
    auxmol = df.addons.make_auxmol(mol, df_basis)
    
    # ints_3c is the 3-center integral tensor (ij|P), where i and j are the
    # indices of AO basis and P is the auxiliary basis
    ints_3c2e = df.incore.aux_e2(mol, auxmol, intor='int3c2e')
    #ints_2c2e is the (P|Q) integrals
    ints_2c2e = auxmol.intor('int2c2e')
    vals, vecs = np.linalg.eigh(ints_2c2e)
    assert(len(vals[vals < 1.e-15]) == 0) # PSD
    metric = np.array(np.dot(vecs * (1 / np.sqrt(vals)) , vecs.conj().T))
    cd_array = np.einsum('PQ,ijP->Qij', metric, ints_3c2e)
    
    # We can get the integrals ( d/dx i, j | P)
    ints_3c2e_ip1 = df.incore.aux_e2(mol, auxmol, intor='int3c2e_ip1', comp=3)
    # Use the same metric as before
    deriv_cderi = np.einsum('PQ,xijP -> xijQ', metric, ints_3c2e_ip1)
    # To reconstruct the full 4c derivative integrals, we need to contract with the previous cderi integrals
    df_grad_4c_ints = np.einsum('xijP,Pkl->xijkl', deriv_cderi, cd_array)
    
    h2_ao_deriv = df_grad_4c_ints
    h2_ao = lib.einsum('Pij,Pkl->ijkl', Lpq_ao, Lpq_ao)
    
    h2_ao_nondf = mol.intor("int2e")
    h2_ao_deriv_nondf = mol.intor("int2e_ip1", comp=3)
    
    print("Max error in 4c ERI derivative:", np.max(np.abs(h2_ao_deriv - h2_ao_deriv_nondf)))
    
    grad_nuc = df.grad.RHF(mf).grad_nuc()    
    #grad_nuc = grad.RHF(scf.RHF(mol)).grad_nuc()


    # Only do FCI if number of orbitals is less than 16
    if mol.nao < 16 and (cont_solver == 'FCI' or fci_done):
        e_fci, c_fci = myci.kernel(h1e_mo, df_eri, mol.nao, mol.nelec, nroots=nroots_evcont)
        e_fci += mol.energy_nuc()
        fci_en[i,:] = e_fci
        
        # Gradients
        mc = mcscf.CASCI(mf, ncas=mf.mo_coeff.shape[0], nelecas=mf.mo_coeff.shape[0])
        out_mc = mc.kernel()
        e_fci = out_mc[0]
        
        assert mc.converged
        
        grad_method = mc.Gradients()
        grad_ref = grad_method.kernel(state=0) #- grad_method.grad_nuc()
        
    else:
        fci_done = False

    if cont_solver != 'FCI':
        # CAS reference
        #mf2 = scf.RHF(mol.copy()).run()
        mf2 = mf
        if cassolver == 'CASCI':
            mc = mcscf.CASCI(mf2, ncas, neleca)
            mc.fcisolver.nroots = nroots_evcont
            e_cas = mc.kernel()[1]
            ref_en[i,:] = e_cas + mol.energy_nuc()
        elif cassolver == 'SA-CASSCF':
            mc_sa = mcscf.CASSCF(mf2, ncas, neleca)#.state_average_([1/nroots_evcont]*nroots_evcont)
            mc_sa.kernel()
            mc = mcscf.CASCI(mf2, ncas, neleca)
            mc.casci(mc_sa.mo_coeff)
            mc.fcisolver.nroots = nroots_evcont
            e_cas = mc.kernel()[1]
            ref_en[i,:] = e_cas + mol.energy_nuc()
        elif cassolver == 'SS-CASSCF':
            e_cas = []
            for istate in range(nroots_evcont):
                mc_ss = mcscf.CASSCF(mf2, ncas, neleca).state_specific_(istate)
                mc_ss.kernel()
                mc = mcscf.CASCI(mf2, ncas, neleca).state_specific_(istate)
                mc.casci(mc_ss.mo_coeff)
                #mc.fcisolver.nroots = nroots_evcont
                e_cas.append(mc.kernel()[0])
            ref_en[i,:] = np.array(e_cas) #+ mol.energy_nuc()

    else:
        ref_en[i,:] = e_fci
        
    # Full continuation
    print('   full')
    # Find h1 and eris in SAO basis
    sao_basis = get_loewdin_trafo(mol.intor("int1e_ovlp"))
    h1e_sao = np.einsum('ai,ab,bj->ij', sao_basis, h1_ao, sao_basis)
    #Lpq_sao = lib.einsum('pi,qj,Lpq->Lij', sao_basis, sao_basis, Lpq_ao)
    Lpq_sao = ao2mo._ao2mo.nr_e2(mf.with_df._cderi, sao_basis,
        (0, sao_basis.shape[1], 0, sao_basis.shape[1]),aosym="s2",mosym="s2")
    Lpq_sao = lib.unpack_tril(Lpq_sao)
    df_eri_sao = lib.einsum('Pij,Pkl->ijkl', Lpq_sao, Lpq_sao)
    
    en_continuation_ms, vec = approximate_multistate(
        h1e_sao,
        df_eri_sao,
        continuation_object_full.one_rdm,
        continuation_object_full.two_rdm,
        continuation_object_full.overlap,
        nroots=nroots_evcont+1
    )
    
    # Get grad
    grad_cont = []
    for i_state in range(nroots_evcont+1):
        vec_i = vec[i_state,:]
        vec_j = vec_i
        
        one_rdm_predicted = np.tensordot(np.outer(vec_i, vec_j), continuation_object_full.one_rdm, axes=2)
        two_rdm_predicted = np.tensordot(np.outer(vec_i, vec_j), continuation_object_full.two_rdm, axes=2)
            
        grad_i = get_grad_elec_OAO_customERI(mol, h2_ao, h2_ao_deriv, 
                                    one_rdm_predicted,
                                    two_rdm_predicted)
        grad_cont.append(grad_i + grad_nuc)
    
    out_full = get_multistate_energy_with_grad_and_NAC(mol,
                                                       continuation_object_full.one_rdm,
                                                       continuation_object_full.two_rdm,
                                                       continuation_object_full.overlap,
                                                       nroots=nroots_evcont+1)
    
    cont_en[i,:] = en_continuation_ms + mol.energy_nuc()
    
    if cont_solver == 'CAS':
        if fci_done:
            print(ehf, e_fci, ref_en[i], cont_en[i], cont_lowrank_en[i], mol.energy_nuc())
        else:
            print(ehf, ref_en[i], cont_en[i], cont_lowrank_en[i], mol.energy_nuc())

    else:
        print(ehf, ref_en[i,:], cont_en[i], cont_lowrank_en[i])
        print('grad', np.linalg.norm(grad_ref-out[2][0]),np.linalg.norm(grad_cont[0]-out[2][0]))
        #print(' \n', grad_ref, '\n', out[2][0],'\n', grad_cont[0] )

print('Time per low-rank (s): %.2f'%(lr_tot/lr_n_eval))

# PLOT
fig, [ax1,ax2,ax3] = plt.subplots(nrows=3,sharex=True,figsize=[4,7],height_ratios=[3,1.5,1.5],
                                 gridspec_kw={'hspace':0.,'wspace':0.})

ax1.plot(test_range, hf_en,'orange',label='HF')
if nroots_evcont > 1:
    if (cont_solver == 'FCI' or fci_done):
        ax1.plot(test_range,fci_en,'k',label=['FCI']+[None]*(nroots_evcont-1))
    if cont_solver != 'FCI':
        ax1.plot(test_range,ref_en,'green',label=[cont_solver]+[None]*(nroots_evcont-1))
    ax1.plot(test_range,cont_en,'b',label=['full evcont']+[None]*(cont_en.shape[-1]-1))
    ax1.plot(test_range,cont_lowrank_en,'--r',label=['low rank evcont']+[None]*(cont_lowrank_en.shape[-1]-1))
else:
    if (cont_solver == 'FCI' or fci_done):
        ax1.plot(test_range,fci_en,'k',label='FCI')
    if cont_solver != 'FCI':
        ax1.plot(test_range,ref_en,'green',label=cont_solver)
    ax1.plot(test_range,cont_en,'b',label='full evcont')
    ax1.plot(test_range,cont_lowrank_en,'--r',label='low rank evcont')
   

ax1.plot(trainig_dists,train_en,'xb')
ax1.plot(trainig_dists,train_lowrank_en,'xr')
ax1.legend()

if (cont_solver == 'FCI' or fci_done):
    ax2.plot(test_range,cont_en[:,:nroots_evcont] - fci_en,'b')
    ax2.plot(test_range,cont_lowrank_en[:,:nroots_evcont] - fci_en,'--r')

if cont_solver != 'FCI':
    ax3.plot(test_range,cont_en[:,:nroots_evcont] - ref_en,'b')
    ax3.plot(test_range,cont_lowrank_en[:,:nroots_evcont] - ref_en,'--r')
    ax3.set_ylabel(r'$E_{cont}$ - $E_{%s}$ (Ha)'%cont_solver)
else:
    
    ax3.plot(test_range,cont_lowrank_en - cont_en,'--r')
    ax3.set_ylabel(r'$E_{cont}$ - $E_{lowrank}$ (Ha)')
    
ax1.set_ylabel('Energy (Ha)')
ax2.set_ylabel(r'$E_{cont}$ - $E_{FCI}$ (Ha)')
ax3.set_xlabel('Atomic separation ($a_0$)')

if figsave:
    plt.savefig('H%i_%s_roots%i_%s'%(natom,cont_solver,nroots_evcont,lowrank_kwargs['truncation_style'])+'.png',bbox_inches='tight',dpi=500)
else:
    plt.show()

from matplotlib.patches import Patch

# Expansion limit
key_clrs = ['tab:blue', 'tab:orange']
no_vec_dic = {}
clr_dic = {}
for key, item in continuation_object.vecs_lowrank.items():
    #print(key, item[0].shape)
    if item[-1]:
        kclr = key_clrs[0]
    else:
        kclr = key_clrs[1]
    if key[1] >= key[0]:
        no_vec_dic[','.join([str(i) for i in key])] = item[0].shape[0]
        clr_dic[','.join([str(i) for i in key])] = kclr

fig, ax = plt.subplots(figsize=[4,7])
ax.grid(alpha=0.5)

# Extract keys, values, and corresponding colors
labels, values = zip(*no_vec_dic.items())
colors = [clr_dic[label] for label in labels]

# Plot with specified colors
ax.barh(labels, values, color=colors)
#D = {u'Label1':26, u'Label2': 17, u'Label3':30}
#ax.barh(*zip(*no_vec_dic.items()))
ax.set_xlabel('Number of vectors (max %i)'%(continuation_object.one_rdm.shape[-1]**2))
ax.set_ylabel('(bra, ket) index')

# Add legend
legend_elements = [
    Patch(facecolor=key_clrs[0], label='Joint ED'),
    Patch(facecolor=key_clrs[1], label='Coulomb SVD')
]
ax.legend(handles=legend_elements, loc='best')

if figsave:
    plt.savefig('nvecs_H%i_%s_roots%i_%s'%(natom,cont_solver,nroots_evcont,lowrank_kwargs['truncation_style'])+'.png',bbox_inches='tight',dpi=500)
else:
    plt.show()
