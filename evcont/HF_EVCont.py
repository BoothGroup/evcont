#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 14:13:32 2024

HF Eigenvector Continuation 

@author: katalar
"""

import numpy as np

from evcont.electron_integral_utils import get_basis, get_integrals

from pyscf import scf, ao2mo, fci

from pygnme import wick, utils

from evcont.ab_initio_gradients_loewdin import get_loewdin_trafo

from mpi4py import MPI

from tqdm import tqdm
import sys

rank = MPI.COMM_WORLD.Get_rank()

# Some stuff required for pygnme interface
def owndata(x):
    # CARMA requires numpy arrays to have data ownership
    if not x.flags["OWNDATA"]:
        y = np.zeros(x.shape, order="C")
        y[:] = x
        x = y
    assert x.flags["OWNDATA"]
    return x

class HF_EVCont_obj:
    """
    HF_EVCont_obj holds the data structure for the continuation from HF states.
    """

    def __init__(
        self,
        hfsolver=scf.RHF,
        nroots=1,
    ):
        """
        Initializes the HF_EVCont_obj class.

        Args:
            hfsolver: The pyscf Hartree-Fock solver routine.
            nroots: Number of states to be solved.  Default is 1, the ground state.
                
        Attributes:
            states (list): The HF training states.
            ens (list): The HF training energies.
            mol_index (list): The molecule indices of the FCI training states
            overlap (ndarray): Overlap matrix.
            one_rdm (ndarray): One-electron t-RDM.
            two_rdm (ndarray): Two-electron t-RDM.
        """
        self.hfsolver = hfsolver
        
        self.nroots = nroots
        # No excited states with vanilla HF; will be extended later on 
        # for multideterminantal solvers
        assert nroots == 1
        
        # Initialize attributes
        self.cis = []
        self.ens = []
        self.mol_index = []
        self.overlap = None
        self.one_rdm = None
        self.two_rdm = None

        # User RDMs or on-the-fly computation
        self.use_rdm = None

        # Preliminaries for OTF Hamiltonian generation
        self.ovlp = [] # Single particle overlap (not to be confused with overlap between states)
        self.sao = []
        self.mo2sao = [] # Canonical to SAO transformation
        self.invsao = [] # Inverse of AO to SAO transformation
        self.mo_coeffs = []
        self.mols = []
        self.nocc = []
        self.norbs = []

    def add_state(self, mol):
        """ 
        Compute the wavefunctions and store them in this object for on-the-fly continuation
        later on.
        ALTERNATIVE to append_to_rdms

        Args:
            mol (object): Molecular object of the training geometry.

        Raises:
            AssertionError: If the mean-field calculation is not converged.
        """
        # Some checks
        if self.use_rdm is None:
            use_rdm = False
        elif self.use_rdm:
            print('Error in add_state: already using append_to_rdms')
            sys.exit()

        # Relevant matrices for SAO basis
        ovlp = mol.intor("int1e_ovlp")
        sao = get_loewdin_trafo(ovlp)
        invsao = np.linalg.inv(sao)

        # Run HF
        mf = self.hfsolver(mol)
        ehf = mf.scf()
        assert mf.converged

        self.ens.append(ehf)
        
        # Rotation matrix
        mf2sao = np.einsum('ji,jk,kl->il',mf.mo_coeff, ovlp, sao)
        
        nocc = np.sum(mf.mo_occ > 0)
        # Add to the EVCont object
        self.ovlp.append(ovlp)
        self.sao.append(sao)
        self.mo2sao.append(mf2sao)
        self.invsao.append(invsao)
        self.mo_coeffs.append(mf.mo_coeff)
        self.mols.append(mol)
        self.nocc.append(nocc)
        self.norbs.append(mf.mo_coeff.shape[0])

    def otf_hamiltonian(self, h1, h2):
        """ 
        Generate subspace Hamiltonian on the fly from precomputed training states (self.cascis)
        Note: Still need to test if the MPI version works

        Args:
            h1 (np.array): 1-electron integrals at the test geometry in SAO basis.
            h2 (np.array): 2-electron integrals at the test geometry in SAO basis.
        """

        nwf = len(self.mols)
        H = np.zeros([nwf,nwf])
        S = np.zeros([nwf,nwf])
        
        #time_pre_bra = 0.
        #time_pre_ket_worb = 0.
        #n_bra_pre = 0
        #n_ket_pre = 0

        #st = time()

        # Iterate over bra states
        for a in range(nwf):

            #MPI.COMM_WORLD.Bcast(casci_bra.ci)
            #MPI.COMM_WORLD.Bcast(casci_bra.mo_coeff)
            
            #st_bra = time()
            
            ovlp_bra = self.ovlp[a]
            basis_OAO_bra = self.sao[a]
            mo_coeff_bra = self.mo_coeffs[a]
            #trafo_bra = self.mo2sao[a]
            #norb_bra = self.norbs[a]
            mol_bra = self.mols[a]
            
            # Pick single Slater-determinant
            nelec = mol_bra.nelec[0]
            ncore = self.nocc[a]-1
            ncas = 1


            bra_ref_state = wick.reference_state[float](
                mo_coeff_bra.shape[0],
                mo_coeff_bra.shape[0],
                nelec,
                ncas,
                ncore,
                owndata(mo_coeff_bra),
            )

            bra_occ_strings = utils.fci_bitset_list(
               1,1
            )

            # Transform 1- and 2-electron integrals into AO basis of bra
            inv_basis_OAO_bra = self.invsao[a]

            #time_pre_bra += time()-st_bra
            #n_bra_pre += 1

            h1e = np.einsum(
                "ia,jb,ij->ab", inv_basis_OAO_bra, inv_basis_OAO_bra, h1, optimize="optimal"
            )

            h2e = np.einsum(
                "ia,jb,kc,ld,ijkl->abcd",
                inv_basis_OAO_bra,
                inv_basis_OAO_bra,
                inv_basis_OAO_bra,
                inv_basis_OAO_bra,
                h2,
                optimize="optimal",
            )

            MPI.COMM_WORLD.Bcast(h1e)
            MPI.COMM_WORLD.Bcast(h2e)

            # Iterate over ket states
            #for b, casci_ket in enumerate(states):
            for b in range(a, nwf):
                #st_ket = time()

                ovlp_ket = self.ovlp[b]
                mo_coeff_ket = self.mo_coeffs[b]
                trafo_ket = self.mo2sao[b]
                mol_ket = self.mols[b]

                trafo_ket_bra = basis_OAO_bra.dot(trafo_ket)

                ket_ref_state = wick.reference_state[float](
                    mo_coeff_ket.shape[0],
                    mo_coeff_ket.shape[0],
                    nelec,
                    ncas,
                    ncore,
                    owndata(trafo_ket_bra),
                )

                ket_occ_strings = utils.fci_bitset_list(
                    1,1
                )

                orbitals = wick.wick_orbitals[float, float](
                    bra_ref_state, ket_ref_state, owndata(ovlp_bra)
                )

                mb = wick.wick_rscf[float, float, float](orbitals, 0.0)

                #time_pre_ket_worb += time()-st_ket
                #n_ket_pre += 1

                # Add one- and two-body contributions
                h1e = owndata(h1e)
                h2e = owndata(h2e.reshape(h1e.shape[0]**2, h1e.shape[0]**2))
                mb.add_one_body(h1e)
                mb.add_two_body(h2e)

                overlap_accumulate = 0.0
                hamiltonian_accumulate = 0.0

                all_ids = np.array(
                    [
                        [iabra, ibbra, iaket, ibket]
                        for iabra in range(len(bra_occ_strings))
                        for ibbra in range(len(bra_occ_strings))
                        for iaket in range(len(ket_occ_strings))
                        for ibket in range(len(ket_occ_strings))
                    ]
                )

                n_ranks = MPI.COMM_WORLD.Get_size()

                all_ids_local = np.array_split(all_ids, n_ranks)[rank]

                if rank == 0:
                    pbar = tqdm(total=len(all_ids_local))

                for ids in all_ids_local:
                    iabra, ibbra, iaket, ibket = ids

                    # Compute S and H contribution for this pair of determinants
                    stmp, htmp = mb.evaluate(bra_occ_strings[iabra], 
                                             bra_occ_strings[ibbra],
                                             ket_occ_strings[iaket],
                                             ket_occ_strings[ibket],
                                             )

                    hamiltonian_accumulate += htmp #* casci_bra.ci[iabra, ibbra] *  casci_ket.ci[iaket, ibket]
                    overlap_accumulate += stmp #* casci_bra.ci[iabra, ibbra] * casci_ket.ci[iaket, ibket]

                    if rank == 0:
                        pbar.update(1)

                if rank == 0:
                    pbar.close()

                overlap_accumulate = MPI.COMM_WORLD.allreduce(
                    overlap_accumulate, op=MPI.SUM
                )
                hamiltonian_accumulate = MPI.COMM_WORLD.allreduce(
                    hamiltonian_accumulate, op=MPI.SUM
                )

                if rank == 0:
                    #print(hamiltonian_accumulate, overlap_accumulate)

                    H[a,b] = hamiltonian_accumulate
                    S[a,b] = overlap_accumulate

                    H[b,a] = np.conj(hamiltonian_accumulate)
                    S[b,a] = np.conj(overlap_accumulate)
                
        #print('----------------------------------------------')
        #print('Time per Hamiltonian: %.5f'%(time()-st))
        #print('Time available for precomputation: %.5f'%(time_pre_bra+time_pre_ket_worb))
        #print('Bra preparation time: %.5f, (%.5f per each bra)'%(time_pre_bra, time_pre_bra/n_bra_pre))
        #print('Ket preparation time with orbital object definition: %.5f, (%.5f per each ket)'%(time_pre_ket_worb, time_pre_ket_worb/n_ket_pre))
        #print('----------------------------------------------')
        #print('OTF-Hamiltonian')
        #print(H)
        #print('OTF-Overlap')
        #print(S)

        return H, S

    def append_to_rdms(self, mol):
        """
        Append a new training geometry by growing the t-RDMs.

        Args:
            mol (object): Molecular object of the training geometry.

        """
        # Some checks
        if self.use_rdm is None:
            use_rdm = True
        elif not self.use_rdm:
            print('Error in append_to_rdms: already using add_state')
            sys.exit()

        # Relevant matrices for SAO basis
        ovlp = mol.intor("int1e_ovlp")
        sao = get_loewdin_trafo(ovlp)
        invsao = np.linalg.inv(sao)

        # Run HF
        mf = self.hfsolver(mol)
        ehf = mf.scf()
        assert mf.converged
        
        # Rotation matrix
        mf2sao = np.einsum('ji,jk,kl->il',mf.mo_coeff, ovlp, sao)
        
        #trafo_basis = sao.T.dot(ovlp).dot(mf.mo_coeff)
        trafo_basis = mf2sao.T

        # Add to the EVCont object
        self.ovlp.append(ovlp)
        self.sao.append(sao)
        self.mo2sao.append(trafo_basis)
        self.invsao.append(invsao)
        self.mo_coeffs.append(mf.mo_coeff)
        self.mols.append(mol)
        nocc = np.sum(mf.mo_occ > 0)
        self.nocc.append(nocc)
        self.norbs.append(mf.mo_coeff.shape[0])

        # Construct h1 and h2
        #h1 = np.linalg.multi_dot((basis.T, scf.hf.get_hcore(mol), basis))
        #h2 = ao2mo.restore(1, ao2mo.kernel(mol, basis), basis.shape[1])
        
        # In the CI form for trans_rdm12 function later on
        no_det = fci.cistring.make_strings(range(mol.nao),mol.nelec[0]).shape[0]
        ci_hf = np.zeros([no_det,no_det])
        ci_hf[0,0] = 1.
        
        ci = fci.addons.transform_ci(ci_hf, mol.nelec, mf2sao)

        # Setting molecular index
        if len(self.mol_index) == 0:
            mindex = 0
        else:
            mindex = max(self.mol_index) + 1
            
        # Iterate over ground and excited states include them in the training
        # NOTE: doesn't work for excited states yet
        for ind in range(self.nroots):
                            
            self.cis.append(ci)
    
            self.ens.append(ehf + mol.energy_nuc())
            self.mol_index.append(mindex)
    
            a = -1 # Last element

            ovlp_bra = self.ovlp[a]
            basis_OAO_bra = self.sao[a]
            mo_coeff_bra = self.mo_coeffs[a]
            trafo_bra = self.mo2sao[a]
            #norb_bra = self.norbs[a]
            mol_bra = self.mols[a]
            
            # Pick single Slater-determinant
            nelec = mol_bra.nelec[0]
            ncore = self.nocc[a]-1
            ncas = 1


            bra_ref_state = wick.reference_state[float](
                mo_coeff_bra.shape[0],
                mo_coeff_bra.shape[0],
                nelec,
                ncas,
                ncore,
                owndata(mo_coeff_bra),
            )

            bra_occ_strings = utils.fci_bitset_list(
               1,1
            )

            # Initialize
            overlap_new = np.ones((len(self.cis), len(self.cis)))
            if self.overlap is not None:
                overlap_new[:-1, :-1] = self.overlap
            one_rdm_new = np.ones((len(self.cis), len(self.cis), mol.nao, mol.nao))
            if self.one_rdm is not None:
                one_rdm_new[:-1, :-1, :, :] = self.one_rdm
            two_rdm_new = np.ones(
                (len(self.cis), len(self.cis), mol.nao, mol.nao, mol.nao, mol.nao)
            )
            if self.two_rdm is not None:
                two_rdm_new[:-1, :-1, :, :, :, :] = self.two_rdm
                
            for b in range(len(self.cis)):
                #ovlp = self.cis[-1].flatten().conj().dot(self.cis[i].flatten())
                #overlap_new[-1, i] = ovlp
                #overlap_new[i, -1] = ovlp.conj()
                ovlp_ket = self.ovlp[b]
                mo_coeff_ket = self.mo_coeffs[b]
                trafo_ket = self.mo2sao[b]
                mol_ket = self.mols[b]

                trafo_ket_bra = basis_OAO_bra.dot(trafo_ket)

                ket_ref_state = wick.reference_state[float](
                    mo_coeff_ket.shape[0],
                    mo_coeff_ket.shape[0],
                    nelec,
                    ncas,
                    ncore,
                    owndata(trafo_ket_bra),
                )

                ket_occ_strings = utils.fci_bitset_list(
                    1,1
                )

                orbitals = wick.wick_orbitals[float, float](
                    bra_ref_state, ket_ref_state, owndata(ovlp_bra)
                )

                mb = wick.wick_rscf[float, float, float](orbitals, 0.0)

                rdm1_tmp = np.zeros((mo_coeff_ket.shape[0], mo_coeff_ket.shape[0]))
                rdm1 = np.zeros((mo_coeff_ket.shape[0], mo_coeff_ket.shape[0]))
                rdm2_tmp = np.zeros(
                    (
                        mo_coeff_ket.shape[0] * mo_coeff_ket.shape[0],
                        mo_coeff_ket.shape[0] * mo_coeff_ket.shape[0],
                    )
                )
                rdm2 = np.zeros(
                    (
                        mo_coeff_ket.shape[0],
                        mo_coeff_ket.shape[0],
                        mo_coeff_ket.shape[0],
                        mo_coeff_ket.shape[0],
                    )
                )

                overlap_accumulate = 0.0

                all_ids = np.array(
                    [
                        [iabra, ibbra, iaket, ibket]
                        for iabra in range(len(bra_occ_strings))
                        for ibbra in range(len(bra_occ_strings))
                        for iaket in range(len(ket_occ_strings))
                        for ibket in range(len(ket_occ_strings))
                    ]
                )

                n_ranks = MPI.COMM_WORLD.Get_size()

                all_ids_local = np.array_split(all_ids, n_ranks)[rank]

                if rank == 0:
                    pbar = tqdm(total=len(all_ids_local))

                for ids in all_ids_local:
                    iabra, ibbra, iaket, ibket = ids

                    o = mb.evaluate_rdm12(
                        bra_occ_strings[iabra], 
                        bra_occ_strings[ibbra],
                        ket_occ_strings[iaket],
                        ket_occ_strings[ibket],
                        1.0,
                        rdm1_tmp,
                        rdm2_tmp,
                    )
                    overlap_accumulate += (
                        o #* casci_bra.ci[iabra, ibbra] * casci_ket.ci[iaket, ibket]
                    )

                    rdm1 += (
                        rdm1_tmp #* casci_bra.ci[iabra, ibbra] * casci_ket.ci[iaket, ibket]
                    )
                    rdm2 += (
                        rdm2_tmp.reshape(rdm2.shape)
                        #* casci_bra.ci[iabra, ibbra]
                        #* casci_ket.ci[iaket, ibket]
                    )

                    if rank == 0:
                        pbar.update(1)

                if rank == 0:
                    pbar.close()

                overlap_accumulate = MPI.COMM_WORLD.allreduce(
                    overlap_accumulate, op=MPI.SUM
                )

                MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, rdm1, op=MPI.SUM)
                MPI.COMM_WORLD.Allreduce(MPI.IN_PLACE, rdm2, op=MPI.SUM)
       
                rdm1 = np.einsum(
                    "...ij,ai,bj->...ab", rdm1, trafo_ket, trafo_bra, optimize="optimal"
                )
                rdm2 = np.einsum(
                    "...ijkl,ai,bj,ck,dl->...abcd",
                    rdm2,
                    trafo_bra,
                    trafo_ket,
                    trafo_bra,
                    trafo_ket,
                    optimize="optimal",
                )
                overlap_new[-1, b] = overlap_accumulate
                overlap_new[b, -1] = np.conj(overlap_accumulate)
                one_rdm_new[-1, b, :, :] = rdm1
                one_rdm_new[b, -1, :, :] = rdm1.conj()
                two_rdm_new[-1, b, :, :, :, :] = rdm2
                two_rdm_new[b, -1, :, :, :, :] = rdm2.conj()
    
            self.overlap = overlap_new
            self.one_rdm = one_rdm_new
            self.two_rdm = two_rdm_new


    def append_to_rdms_old(self, mol):
        """
        Append a new training geometry by growing the t-RDMs.

        Args:
            mol (object): Molecular object of the training geometry.

        """
        # Some checks
        if self.use_rdm is None:
            use_rdm = True
        elif not self.use_rdm:
            print('Error in append_to_rdms: already using add_state')
            sys.exit()

        # Relevant matrices for SAO basis
        ovlp = mol.intor("int1e_ovlp")
        sao = get_loewdin_trafo(ovlp)
        invsao = np.linalg.inv(sao)

        # Run HF
        mf = self.hfsolver(mol)
        ehf = mf.scf()
        assert mf.converged
        
        # Rotation matrix
        mf2sao = np.einsum('ji,jk,kl->il',mf.mo_coeff, ovlp, sao)
        
        #trafo_basis = sao.T.dot(ovlp).dot(mf.mo_coeff)
        trafo_basis = mf2sao.T

        # Add to the EVCont object
        self.ovlp.append(ovlp)
        self.sao.append(sao)
        self.mo2sao.append(trafo_basis)
        self.invsao.append(invsao)
        self.mo_coeffs.append(mf.mo_coeff)
        self.mols.append(mol)
        nocc = np.sum(mf.mo_occ > 0)
        self.nocc.append(nocc)
        self.norbs.append(mf.mo_coeff.shape[0])

        # Construct h1 and h2
        #h1 = np.linalg.multi_dot((basis.T, scf.hf.get_hcore(mol), basis))
        #h2 = ao2mo.restore(1, ao2mo.kernel(mol, basis), basis.shape[1])
        
        # In the CI form for trans_rdm12 function later on
        no_det = fci.cistring.make_strings(range(mol.nao),mol.nelec[0]).shape[0]
        ci_hf = np.zeros([no_det,no_det])
        ci_hf[0,0] = 1.
        
        ci = fci.addons.transform_ci(ci_hf, mol.nelec, mf2sao)

        # Setting molecular index
        if len(self.mol_index) == 0:
            mindex = 0
        else:
            mindex = max(self.mol_index) + 1
            
        # Iterate over ground and excited states include them in the training
        # NOTE: doesn't work for excited states yet
        for ind in range(self.nroots):
                            
            self.cis.append(ci)
    
            self.ens.append(ehf + mol.energy_nuc())
            self.mol_index.append(mindex)
    
            # Initialize
            overlap_new = np.ones((len(self.cis), len(self.cis)))
            if self.overlap is not None:
                overlap_new[:-1, :-1] = self.overlap
            one_rdm_new = np.ones((len(self.cis), len(self.cis), mol.nao, mol.nao))
            if self.one_rdm is not None:
                one_rdm_new[:-1, :-1, :, :] = self.one_rdm
            two_rdm_new = np.ones(
                (len(self.cis), len(self.cis), mol.nao, mol.nao, mol.nao, mol.nao)
            )
            if self.two_rdm is not None:
                two_rdm_new[:-1, :-1, :, :, :, :] = self.two_rdm
                
            for i in range(len(self.cis)):
                ovlp = self.cis[-1].flatten().conj().dot(self.cis[i].flatten())
                overlap_new[-1, i] = ovlp
                overlap_new[i, -1] = ovlp.conj()


                rdm1, rdm2 = fci.direct_spin1.trans_rdm12(
                    self.cis[-1], self.cis[i], mol.nao, mol.nelec
                )
                rdm1_conj, rdm2_conj = fci.direct_spin1.trans_rdm12(
                    self.cis[i], self.cis[-1], mol.nao, mol.nelec
                )
                one_rdm_new[-1, i, :, :] = rdm1
                one_rdm_new[i, -1, :, :] = rdm1_conj
                #one_rdm_new[i, -1, :, :] = rdm1.conj()
                two_rdm_new[-1, i, :, :, :, :] = rdm2
                two_rdm_new[i, -1, :, :, :, :] = rdm2_conj
                #two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()
    
            self.overlap = overlap_new
            self.one_rdm = one_rdm_new
            self.two_rdm = two_rdm_new


if __name__ == '__main__':
    
    # Quick testing of the implementation
    from pyscf import gto
    from evcont.ab_initio_eigenvector_continuation import approximate_multistate_OAO
    
    # INPUT PARAMETERS
    natom = 8
    # Training geometries
    #training_r = [1.0, 1.75, 2.6]
    training_r = [1.0, 1.3, 1.75, 2.2, 2.6]
    # Test geometeries
    r_scan = np.linspace(0.7, 3.5, 30)
    
    def chain(atom, natom, bond_length, numbering=None):
        """Open boundary condition version of 1D ring"""
        atoms = []
        if isinstance(atom, str):
            atom = [atom]
        for i in range(natom):
            atom_i = atom[i % len(atom)]
            if numbering is not None:
                atom_i += str(int(numbering) + i)
            atoms.append([atom_i, np.asarray([i * bond_length, 0.0, 0.0])])
        return atoms

    # Initialize EVcont object
    continuation_object = HF_EVCont_obj(hfsolver=scf.RHF)
    
    # Training
    training_energies = []
    ntrain = len(training_r)
    # Loop over training geometries
    for i, r in enumerate(training_r):
        print('Training', i)

        mol = gto.Mole()
        mol.build(atom = chain("H", natom, bond_length=r),basis = "sto-3g", unit = 'B', verbose=1)
    
        continuation_object.append_to_rdms(mol)
        training_energies.append(continuation_object.ens[-1] - mol.energy_nuc())
    
    # Testing
    hf_en = np.zeros([len(r_scan)])
    fci_en = np.zeros([len(r_scan)])   
    cont_en = np.zeros([len(r_scan)])

    for i, r in enumerate(r_scan):
        print(i)
        
        mol = gto.Mole()
        mol.build(atom = chain("H", natom, bond_length=r),basis = "sto-3g", unit = 'B', verbose=1)

        #h1, h2 = get_integrals(mol, get_basis(mol))
        
        # HF reference
        mf = scf.RHF(mol)
        ehf = mf.scf()
        hf_en[i] = ehf
        
        # FCI reference - TODO
        
        
        # Continuation
        en_continuation_ms, vec = approximate_multistate_OAO(
            mol,
            continuation_object.one_rdm,
            continuation_object.two_rdm,
            continuation_object.overlap,
        )
        
        cont_en[i] = en_continuation_ms

    # Plot
    import matplotlib.pylab as plt
    
    fig, [ax1,ax2] = plt.subplots(nrows=2,figsize=[8,8],height_ratios=[3,2])
    ax1.plot(r_scan,hf_en,label='HF')
    ax1.plot(r_scan,cont_en,label='HF-Continuation')
    ax1.plot(training_r,training_energies,'xr')
    
    ax2.plot(r_scan,cont_en-hf_en)
    
    
    ax1.set_ylabel(r'E (Ha)',fontsize=15)
    ax2.set_ylabel(r'E$_{cont}$ - E$_{HF}$ (Ha)',fontsize=15)
    ax2.set_xlabel(r'H Bond length (a$_0$)',fontsize=15)
    
    ax1.legend(fontsize=15)
    
    plt.savefig('hf_continuation.png')
