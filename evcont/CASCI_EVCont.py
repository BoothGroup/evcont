import numpy as np

from evcont.electron_integral_utils import get_basis, get_integrals

from evcont.low_rank_utils import reduce_2rdm
from pygnme import wick, utils

#from pyscf.mcscf.casci import CASCI
from pyscf import scf, mcscf

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


# Old test function (for CAS spanning the full space)
# def append_to_rdms_complete_space(cascis, overlap=None, one_rdm=None, two_rdm=None):
#     n_cascis = len(cascis)
#     casci_bra = cascis[-1]
#     casci_bra.kernel()
#     mo_coeff_bra = casci_bra.mo_coeff
#     mol_bra = casci_bra.mol

#     basis_MO_bra = mo_coeff_bra[:, casci_bra.ncore : casci_bra.ncore + casci_bra.ncas]
#     ovlp_bra = mol_bra.intor_symmetric("int1e_ovlp")
#     basis_OAO_bra = get_basis(mol_bra)
#     trafo_bra = basis_OAO_bra.T.dot(ovlp_bra).dot(basis_MO_bra)

#     overlap_new = np.ones((n_cascis, n_cascis))
#     if overlap is not None:
#         overlap_new[:-1, :-1] = overlap
#     one_rdm_new = np.ones(
#         (n_cascis, n_cascis, mo_coeff_bra.shape[0], mo_coeff_bra.shape[0])
#     )
#     if one_rdm is not None:
#         one_rdm_new[:-1, :-1, :, :] = one_rdm
#     two_rdm_new = np.ones(
#         (
#             n_cascis,
#             n_cascis,
#             mo_coeff_bra.shape[0],
#             mo_coeff_bra.shape[0],
#             mo_coeff_bra.shape[0],
#             mo_coeff_bra.shape[0],
#         )
#     )
#     if two_rdm is not None:
#         two_rdm_new[:-1, :-1, :, :, :, :] = two_rdm
#     for i in range(n_cascis):
#         casci_ket = cascis[i]
#         mo_coeff_ket = casci_ket.mo_coeff
#         mol_ket = casci_ket.mol

#         basis_MO_ket = mo_coeff_ket[
#             :, casci_ket.ncore : casci_ket.ncore + casci_ket.ncas
#         ]
#         ovlp_ket = mol_ket.intor_symmetric("int1e_ovlp")
#         basis_OAO_ket = get_basis(mol_ket)
#         trafo_ket = basis_OAO_ket.T.dot(ovlp_ket).dot(basis_MO_ket)

#         trafo_ket_bra = trafo_ket.T.dot(trafo_bra)

#         ket = casci_ket.fcisolver.transform_ci_for_orbital_rotation(
#             casci_ket.ci, casci_ket.ncas, casci_ket.nelecas, trafo_ket_bra
#         )

#         ovlp = casci_bra.ci.flatten().conj().dot(ket.flatten())
#         overlap_new[-1, i] = ovlp
#         overlap_new[i, -1] = ovlp.conj()

#         rdm1, rdm2 = casci_bra.fcisolver.trans_rdm12(
#             casci_bra.ci, ket, casci_bra.ncas, casci_bra.nelecas
#         )
#         rdm1, rdm2 = transform_integrals(rdm1, rdm2, trafo_bra)

#         one_rdm_new[-1, i, :, :] = rdm1
#         one_rdm_new[i, -1, :, :] = rdm1.conj()
#         two_rdm_new[-1, i, :, :, :, :] = rdm2
#         two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()
#     return overlap_new, one_rdm_new, two_rdm_new


class CAS_EVCont_obj:
    """
    CAS_EVCont_obj holds the data structure for the continuation from CAS states.
    """

    def __init__(self, ncas, neleca, #casci_solver=CASCI, 
                nroots=1, solver='SS-CASSCF',
                lowrank=False,
                **kwargs):
        """
        Initialize the CAS_EVCont_obj.

        Args:
            ncas (int): Number of CAS orbitals.
            neleca (int): Number of active space electrons.
            casci_solver (object): CASCI solver object from PySCF (can also be CASSCF).

        Attributes:
            ncas (int): Number of CAS orbitals.
            neleca (int): Number of alpha electrons.
            cascis (list): List to store CASCI objects.
            overlap (ndarray): Overlap matrix.
            one_rdm (ndarray): One-electron t-RDM.
            two_rdm (ndarray): Two-electron t-RDM.
            casci_solver (object): CASCI solver object.
        """

        self.ncas = ncas
        self.neleca = neleca

        self.cascis = []
        self.overlap = None
        self.one_rdm = None
        self.two_rdm = None

        #self.casci_solver = casci_solver
        self.nroots = nroots

        if solver in ['CASCI','SS-CASSCF','SA-CASSCF']:
            self.solver = solver
        else:
            print('Wrong solver in CAS_EVCont_obj')
            sys.exit()

        # Set flags for using add_state vs append_to_rdms
        # (to prevent double addition into self.cascis or missing states in tRDMs)
        self.use_rdm = None
        
        ### Initialize low-rank attributes
        self.lowrank = lowrank
        if lowrank:
            #self.truncation_style = kwargs['truncation_style']
            self.kwargs = kwargs
            
        # Diagonals of 2-cumulants ([nbra, nket, 3, norb, norb])
        self.cum_diagonal = None 
        # Low rank eigendecomposition of the rest of 2-cumulant
        # dictionary[(nbra, nket)] = (vals_trunc, vecs_trunc)
        self.vecs_lowrank = {}

        # Precomputation for OTF Hamiltonian
        self.precompute = False
        self.inv_OAO_all = []
        self.mb_all = None
        self.occ_strings_all = []

    def otf_hamiltonian_old(self, h1, h2):
        """ 
        OLD VERSION WITH INTERMEDIATE RDM COMPUTATION
        Generate subspace Hamiltonian on the fly from precomputed training states (self.cascis)
        Note: Still need to test if the MPI version works

        Args:
            h1 (np.array): 1-electron integrals at the test geometry.
            h2 (np.array): 2-electron integrals at the test geometry.
        """
        states = self.cascis

        nwf = len(states)
        H = np.zeros([nwf,nwf])
        S = np.zeros([nwf,nwf])
        
        # Iterate over bra states
        for a, casci_bra in enumerate(states):

            MPI.COMM_WORLD.Bcast(casci_bra.ci)
            MPI.COMM_WORLD.Bcast(casci_bra.mo_coeff)

            mo_coeff_bra = casci_bra.mo_coeff
            mol_bra = casci_bra.mol

            ovlp_bra = mol_bra.intor_symmetric("int1e_ovlp")
            basis_OAO_bra = get_basis(mol_bra)
            trafo_bra = basis_OAO_bra.T.dot(ovlp_bra).dot(mo_coeff_bra)

            bra_ref_state = wick.reference_state[float](
                mo_coeff_bra.shape[0],
                mo_coeff_bra.shape[0],
                mol_bra.nelec[0],
                casci_bra.ncas,
                casci_bra.ncore,
                owndata(mo_coeff_bra),
            )

            bra_occ_strings = utils.fci_bitset_list(
                mol_bra.nelec[0] - casci_bra.ncore, casci_bra.ncas
            )

            # Iterate over ket states
            for b, casci_ket in enumerate(states):

                # Prepare ket state                
                mo_coeff_ket = casci_ket.mo_coeff
                mol_ket = casci_ket.mol

                ovlp_ket = mol_ket.intor_symmetric("int1e_ovlp")
                basis_OAO_ket = get_basis(mol_ket)
                trafo_ket = basis_OAO_ket.T.dot(ovlp_ket).dot(mo_coeff_ket)

                trafo_ket_bra = basis_OAO_bra.dot(trafo_ket)

                ket_ref_state = wick.reference_state[float](
                    mo_coeff_ket.shape[0],
                    mo_coeff_ket.shape[0],
                    mol_ket.nelec[0],
                    casci_ket.ncas,
                    casci_ket.ncore,
                    owndata(trafo_ket_bra),
                )

                orbitals = wick.wick_orbitals[float, float](
                    bra_ref_state, ket_ref_state, owndata(ovlp_bra)
                )

                wick_mb = wick.wick_rscf[float, float, float](orbitals, 0.0)

                ket_occ_strings = utils.fci_bitset_list(
                    mol_ket.nelec[0] - casci_ket.ncore, casci_ket.ncas
                )

                # Generate temporary RDMs between bra and ket states
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
                    stringabra = bra_occ_strings[iabra]
                    stringbbra = bra_occ_strings[ibbra]
                    stringaket = ket_occ_strings[iaket]
                    stringbket = ket_occ_strings[ibket]

                    rdm1_tmp.fill(0.0)
                    rdm2_tmp.fill(0.0)
                    o = wick_mb.evaluate_rdm12(
                        stringabra,
                        stringbbra,
                        stringaket,
                        stringbket,
                        1.0,
                        rdm1_tmp,
                        rdm2_tmp,
                    )
                    overlap_accumulate += (
                        o * casci_bra.ci[iabra, ibbra] * casci_ket.ci[iaket, ibket]
                    )

                    rdm1 += (
                        rdm1_tmp * casci_bra.ci[iabra, ibbra] * casci_ket.ci[iaket, ibket]
                    )
                    rdm2 += (
                        rdm2_tmp.reshape(rdm2.shape)
                        * casci_bra.ci[iabra, ibbra]
                        * casci_ket.ci[iaket, ibket]
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
                
                if rank == 0:
                    '''
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
                    '''
                    #'''
                    # Alternatively pass the transformation to integrals (working)
                    h1e = np.einsum(
                        "ai,bj,ab->ij", trafo_ket, trafo_bra, h1, optimize="optimal"
                    )

                    h2e = np.einsum(
                        "ai,bj,ck,dl,abcd->ijkl",
                        trafo_bra,
                        trafo_ket,
                        trafo_bra,
                        trafo_ket,
                        h2,
                        optimize="optimal",
                    )
                    #'''
                    H[a,b] = np.einsum("...kl,kl", rdm1, h1e, optimize="optimal") + 0.5 * np.einsum(
                        "...klmn,klmn", rdm2, h2e, optimize="optimal"
                    )

                    # Alternative contraction (works)
                    #H[a,b] = np.einsum( "...ij,ai,bj,ab", rdm1, trafo_ket, trafo_bra, h1, optimize="optimal") \
                    #+ 0.5 * np.einsum("...ijkl,ai,bj,ck,dl,abcd",
                    #    rdm2,
                    #    trafo_bra,
                    #    trafo_ket,
                    #    trafo_bra,
                    #    trafo_ket,
                    #    h2,
                    #    optimize="optimal"
                    #)
                    

                    S[a,b] = overlap_accumulate
                
        return H, S

    def otf_hamiltonian(self, h1, h2):
        """ 
        Generate subspace Hamiltonian on the fly from precomputed training states (self.cascis)
        Note: Still need to test if the MPI version works

        Args:
            h1 (np.array): 1-electron integrals at the test geometry in SAO basis.
            h2 (np.array): 2-electron integrals at the test geometry in SAO basis.
        """
        states = self.cascis

        nwf = len(states)
        H = np.zeros([nwf,nwf])
        S = np.zeros([nwf,nwf])
        
        #time_pre_bra = 0.
        #time_pre_ket_worb = 0.
        #n_bra_pre = 0
        #n_ket_pre = 0

        #st = time()

        # Iterate over bra states
        for a, casci_bra in enumerate(states):

            MPI.COMM_WORLD.Bcast(casci_bra.ci)
            MPI.COMM_WORLD.Bcast(casci_bra.mo_coeff)
            
            #st_bra = time()
            
            mo_coeff_bra = casci_bra.mo_coeff
            mol_bra = casci_bra.mol

            ovlp_bra = mol_bra.intor_symmetric("int1e_ovlp")
            basis_OAO_bra = get_basis(mol_bra)
            trafo_bra = basis_OAO_bra.T.dot(ovlp_bra).dot(mo_coeff_bra)
            #print('bra',ovlp_bra.shape,basis_OAO_bra.shape,trafo_bra.shape)

            bra_ref_state = wick.reference_state[float](
                mo_coeff_bra.shape[0],
                mo_coeff_bra.shape[0],
                mol_bra.nelec[0],
                casci_bra.ncas,
                casci_bra.ncore,
                owndata(mo_coeff_bra),
            )

            bra_occ_strings = utils.fci_bitset_list(
                mol_bra.nelec[0] - casci_bra.ncore, casci_bra.ncas
            )

            # Generate and transform 1- and 2-electron integrals
            # AO(test) to AO(bra) transformation
            #basis_test_bra = np.dot(get_basis(mol),np.linalg.inv(basis_OAO_bra))
            #h1e, h2e = get_integrals(mol, basis_test_bra)

            # Transform 1- and 2-electron integrals into AO basis of bra
            inv_basis_OAO_bra = np.linalg.inv(basis_OAO_bra)

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
                casci_ket = states[b]
                #st_ket = time()

                # Prepare ket state                
                mo_coeff_ket = casci_ket.mo_coeff
                mol_ket = casci_ket.mol

                ovlp_ket = mol_ket.intor_symmetric("int1e_ovlp")
                basis_OAO_ket = get_basis(mol_ket)
                trafo_ket = basis_OAO_ket.T.dot(ovlp_ket).dot(mo_coeff_ket)

                trafo_ket_bra = basis_OAO_bra.dot(trafo_ket)
                #print('bra',ovlp_ket.shape,basis_OAO_ket.shape,trafo_ket.shape, trafo_ket_bra.shape)

                ket_ref_state = wick.reference_state[float](
                    mo_coeff_ket.shape[0],
                    mo_coeff_ket.shape[0],
                    mol_ket.nelec[0],
                    casci_ket.ncas,
                    casci_ket.ncore,
                    owndata(trafo_ket_bra),
                )

                ket_occ_strings = utils.fci_bitset_list(
                    mol_ket.nelec[0] - casci_ket.ncore, casci_ket.ncas
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
                                             1.0)

                    hamiltonian_accumulate += htmp * casci_bra.ci[iabra, ibbra] *  casci_ket.ci[iaket, ibket]
                    overlap_accumulate += stmp * casci_bra.ci[iabra, ibbra] * casci_ket.ci[iaket, ibket]

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
        return H, S

    def otf_hamiltonia_precomputed(self, h1, h2):
        """ 
        Generate subspace Hamiltonian on the fly from precomputed training states (self.cascis)
        Using precomputed quantities for speedup
        Note: Still need to test if the MPI version works

        Args:
            h1 (np.array): 1-electron integrals at the test geometry in SAO basis.
            h2 (np.array): 2-electron integrals at the test geometry in SAO basis.
        """

        if self.precompute == False:
            print('Precomputations were not available. Precomputing now.')
            self.precompute_for_otf()

        states = self.cascis
        inv_OAO_all = self.inv_OAO_all
        occ_strings_all = self.occ_strings_all
        #mb_all = self.mb_all

        #print(mb_all[0][0])
        nwf = len(states)
        H = np.zeros([nwf,nwf])
        S = np.zeros([nwf,nwf])
        
        #mb_all = [[i,j] for i in range(nwf) for j in range(nwf)]

        #time_pre_bra = 0.
        #time_pre_ket_worb = 0.
        #n_bra_pre = 0
        #n_ket_pre = 0

        #st = time()

        # Iterate over bra states
        for a, casci_bra in enumerate(states):

            MPI.COMM_WORLD.Bcast(casci_bra.ci)            
            #MPI.COMM_WORLD.Bcast(inv_OAO_all[a])         
            #MPI.COMM_WORLD.Bcast(occ_strings_all[a])  
            #MPI.COMM_WORLD.Bcast(mb_all[a,:])  

            bra_occ_strings = occ_strings_all[a]

            # Generate and transform 1- and 2-electron integrals
            # AO(test) to AO(bra) transformation
            #basis_test_bra = np.dot(get_basis(mol),np.linalg.inv(basis_OAO_bra))
            #h1e, h2e = get_integrals(mol, basis_test_bra)

            # Transform 1- and 2-electron integrals into AO basis of bra
            inv_basis_OAO_bra = inv_OAO_all[a]

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
                casci_ket = states[b]
                #st_ket = time()

                ket_occ_strings = occ_strings_all[b]

                
                #mb = wick.wick_rscf[float, float, float](orbitals, 0.0)

                mb = self.mb_all[a][b]

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
                                             1.0)

                    hamiltonian_accumulate += htmp * casci_bra.ci[iabra, ibbra] *  casci_ket.ci[iaket, ibket]
                    overlap_accumulate += stmp * casci_bra.ci[iabra, ibbra] * casci_ket.ci[iaket, ibket]

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
        return H, S

    def precompute_for_otf(self):
        """ 
        Precompute and setup the on-the-fly computation of the subspace Hamiltonian 
        beforehand to save time during test evaluations
        """

        assert len(self.cascis) != 0
        self.precompute = True

        states = self.cascis

        nwf = len(states)

        #self.mb_all = np.ones((nwf,nwf))*np.nan
        self.mb_all = []
        self.orbitals_all = []

        # Iterate over bra states
        for a, casci_bra in enumerate(states):
            
            mb_a = []
            orb_a = []
            MPI.COMM_WORLD.Bcast(casci_bra.ci)
            MPI.COMM_WORLD.Bcast(casci_bra.mo_coeff)
            
            #st_bra = time()
            
            mo_coeff_bra = casci_bra.mo_coeff
            mol_bra = casci_bra.mol

            ovlp_bra = mol_bra.intor_symmetric("int1e_ovlp")
            basis_OAO_bra = get_basis(mol_bra)
            trafo_bra = basis_OAO_bra.T.dot(ovlp_bra).dot(mo_coeff_bra)
            #print('bra',ovlp_bra.shape,basis_OAO_bra.shape,trafo_bra.shape)

            bra_ref_state = wick.reference_state[float](
                mo_coeff_bra.shape[0],
                mo_coeff_bra.shape[0],
                mol_bra.nelec[0],
                casci_bra.ncas,
                casci_bra.ncore,
                owndata(mo_coeff_bra),
            )

            bra_occ_strings = utils.fci_bitset_list(
                mol_bra.nelec[0] - casci_bra.ncore, casci_bra.ncas
            )
            self.occ_strings_all.append(bra_occ_strings)

            # Transform 1- and 2-electron integrals into AO basis of bra
            inv_basis_OAO_bra = np.linalg.inv(basis_OAO_bra)
            self.inv_OAO_all.append(inv_basis_OAO_bra)

            # Iterate over ket states
            #for b, casci_ket in enumerate(states):
            #for b in range(a, nwf):
            for b in range(nwf):
                casci_ket = states[b]
                #st_ket = time()

                # Prepare ket state                
                mo_coeff_ket = casci_ket.mo_coeff
                mol_ket = casci_ket.mol

                ovlp_ket = mol_ket.intor_symmetric("int1e_ovlp")
                basis_OAO_ket = get_basis(mol_ket)
                trafo_ket = basis_OAO_ket.T.dot(ovlp_ket).dot(mo_coeff_ket)

                trafo_ket_bra = basis_OAO_bra.dot(trafo_ket)
                #print('bra',ovlp_ket.shape,basis_OAO_ket.shape,trafo_ket.shape, trafo_ket_bra.shape)

                ket_ref_state = wick.reference_state[float](
                    mo_coeff_ket.shape[0],
                    mo_coeff_ket.shape[0],
                    mol_ket.nelec[0],
                    casci_ket.ncas,
                    casci_ket.ncore,
                    owndata(trafo_ket_bra),
                )

                ket_occ_strings = utils.fci_bitset_list(
                    mol_ket.nelec[0] - casci_ket.ncore, casci_ket.ncas
                )
                orbitals = wick.wick_orbitals[float, float](
                    bra_ref_state, ket_ref_state, owndata(ovlp_bra)
                )
                orb_a.append(orbitals)
                mb = wick.wick_rscf[float, float, float](orbitals, 0.0)
                mb_a.append(mb)

                #self.mb_all[a,b] = mb
                #self.mb_all[b,a] = mb

            self.mb_all.append(mb_a)
            self.orbitals_all.append(orb_a)
        #self.mb_all = np.array(self.mb_all)

        return 1

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

        # Run mean field calculations for the orbitals
        mf = scf.RHF(mol.copy())
        mf.kernel()

        assert mf.converged

        MPI.COMM_WORLD.Bcast(mf.mo_coeff)

        # Specificy the CAS solver for the current state
        if self.solver == 'SA-CASSCF':
            mc = mcscf.CASSCF(mf, self.ncas, self.neleca).state_average_([1/self.nroots]*self.nroots)
            mc.kernel()
            mo_sacasscf = mc.mo_coeff

        # Iterate over different states
        for istate in range(self.nroots):

            if self.solver == 'CASCI':
                casci_bra = mcscf.CASCI(mf, self.ncas, self.neleca).state_specific_(istate)
            elif self.solver == 'SS-CASSCF':
                cas_ss = mcscf.CASSCF(mf, self.ncas, self.neleca).state_specific_(istate)
                cas_ss.kernel()
                casci_bra = mcscf.CASCI(mf, self.ncas, self.neleca).state_specific_(istate)
                casci_bra.casci(cas_ss.mo_coeff)
            else:
                casci_bra = mcscf.CASCI(mf, self.ncas, self.neleca).state_specific_(istate)
                casci_bra.casci(mo_sacasscf)

            self.cascis.append(casci_bra)

    def append_to_rdms_new(self, mol):
        """
        Append a new training geometry. See pygnme examples for more information about
        the evaluation of the t-RDMs.

        Args:
            mol (object): Molecular object of the training geometry.

        Raises:
            AssertionError: If the mean-field calculation is not converged.
        """
        # Some checks
        if self.use_rdm is None:
            use_rdm = True
        elif not self.use_rdm:
            print('Error in append_to_rdms: already using add_state')
            sys.exit()
            
        lowrank = self.lowrank

        # Run mean field calculations for the orbitals
        #mf = mol.copy().RHF()
        mf = scf.RHF(mol.copy())
        mf.kernel()

        assert mf.converged

        MPI.COMM_WORLD.Bcast(mf.mo_coeff)

        # Specificy the CAS solver for the current state
        #casci_bra_all = self.casci_solver(mf, self.ncas, self.neleca)
        #casci_bra_all.fcisolver.nroots = self.nroots

        if self.solver == 'SA-CASSCF':
            mc = mcscf.CASSCF(mf, self.ncas, self.neleca).state_average_([1/self.nroots]*self.nroots)
            mc.kernel()
            mo_sacasscf = mc.mo_coeff

        # Iterate over different states
        for istate in range(self.nroots):

            # Read the DM representation from existing training states
            overlap = self.overlap
            one_rdm = self.one_rdm
            if not lowrank:
                two_rdm = self.two_rdm
            else:
                cum_diagonal = self.cum_diagonal
                vecs_lowrank = self.vecs_lowrank
                
            # CAS solver
            if self.solver == 'CASCI':
                casci_bra = mcscf.CASCI(mf, self.ncas, self.neleca).state_specific_(istate)
            elif self.solver == 'SS-CASSCF':
                cas_ss = mcscf.CASSCF(mf, self.ncas, self.neleca).state_specific_(istate)
                cas_ss.kernel()
                casci_bra = mcscf.CASCI(mf, self.ncas, self.neleca).state_specific_(istate)
                casci_bra.casci(cas_ss.mo_coeff)
            else:
                casci_bra = mcscf.CASCI(mf, self.ncas, self.neleca).state_specific_(istate)
                casci_bra.casci(mo_sacasscf)

            self.cascis.append(casci_bra)

            cascis = self.cascis
            n_cascis = len(cascis)

            out = casci_bra.kernel()
            e = out[0]
            
            assert np.all(casci_bra.fcisolver.converged)

            if hasattr(casci_bra, "converged"):
                assert casci_bra.converged

            MPI.COMM_WORLD.Bcast(casci_bra.ci)
            MPI.COMM_WORLD.Bcast(casci_bra.mo_coeff)

            mo_coeff_bra = casci_bra.mo_coeff
            mol_bra = casci_bra.mol

            ovlp_bra = mol_bra.intor_symmetric("int1e_ovlp")
            basis_OAO_bra = get_basis(mol_bra)
            trafo_bra = basis_OAO_bra.T.dot(ovlp_bra).dot(mo_coeff_bra)

            bra_ref_state = wick.reference_state[float](
                mo_coeff_bra.shape[0],
                mo_coeff_bra.shape[0],
                mol_bra.nelec[0],
                casci_bra.ncas,
                casci_bra.ncore,
                owndata(mo_coeff_bra),
            )

            if rank == 0:
                overlap_new = np.zeros((n_cascis, n_cascis))
                if overlap is not None:
                    overlap_new[:-1, :-1] = overlap
                one_rdm_new = np.zeros(
                    (n_cascis, n_cascis, mo_coeff_bra.shape[0], mo_coeff_bra.shape[0])
                )
                if one_rdm is not None:
                    one_rdm_new[:-1, :-1, :, :] = one_rdm
                two_rdm_new = np.zeros(
                    (
                        n_cascis,
                        n_cascis,
                        mo_coeff_bra.shape[0],
                        mo_coeff_bra.shape[0],
                        mo_coeff_bra.shape[0],
                        mo_coeff_bra.shape[0],
                    )
                )
                
                # Only define two_rdm if not lowrank
                if not lowrank:
                    two_rdm_new = np.zeros(
                        (
                            n_cascis,
                            n_cascis,
                            mo_coeff_bra.shape[0],
                            mo_coeff_bra.shape[0],
                            mo_coeff_bra.shape[0],
                            mo_coeff_bra.shape[0],
                        )
                    )
                    if two_rdm is not None:
                        two_rdm_new[:-1, :-1, :, :, :, :] = two_rdm
                        
                else:
                    cum_diagonal_new = np.ones(
                        (n_cascis,
                         n_cascis, 3, mo_coeff_bra.shape[0], mo_coeff_bra.shape[0])
                    )
                    if cum_diagonal is not None:
                        cum_diagonal_new[:-1, :-1, :, :, :] = cum_diagonal
                    
            else:
                overlap_new = one_rdm_new = two_rdm_new = None

            bra_occ_strings = utils.fci_bitset_list(
                mol_bra.nelec[0] - casci_bra.ncore, casci_bra.ncas
            )

            for i in range(n_cascis):
                casci_ket = cascis[i]
                mo_coeff_ket = casci_ket.mo_coeff
                mol_ket = casci_ket.mol

                ovlp_ket = mol_ket.intor_symmetric("int1e_ovlp")
                basis_OAO_ket = get_basis(mol_ket)
                trafo_ket = basis_OAO_ket.T.dot(ovlp_ket).dot(mo_coeff_ket)

                trafo_ket_bra = basis_OAO_bra.dot(trafo_ket)

                ket_ref_state = wick.reference_state[float](
                    mo_coeff_ket.shape[0],
                    mo_coeff_ket.shape[0],
                    mol_ket.nelec[0],
                    casci_ket.ncas,
                    casci_ket.ncore,
                    owndata(trafo_ket_bra),
                )

                orbitals = wick.wick_orbitals[float, float](
                    bra_ref_state, ket_ref_state, owndata(ovlp_bra)
                )

                wick_mb = wick.wick_rscf[float, float, float](orbitals, 0.0)

                ket_occ_strings = utils.fci_bitset_list(
                    mol_ket.nelec[0] - casci_ket.ncore, casci_ket.ncas
                )

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
                    stringabra = bra_occ_strings[iabra]
                    stringbbra = bra_occ_strings[ibbra]
                    stringaket = ket_occ_strings[iaket]
                    stringbket = ket_occ_strings[ibket]

                    rdm1_tmp.fill(0.0)
                    rdm2_tmp.fill(0.0)
                    o = wick_mb.evaluate_rdm12(
                        stringabra,
                        stringbbra,
                        stringaket,
                        stringbket,
                        1.0,
                        rdm1_tmp,
                        rdm2_tmp,
                    )
                    overlap_accumulate += (
                        o * casci_bra.ci[iabra, ibbra] * casci_ket.ci[iaket, ibket]
                    )

                    rdm1 += (
                        rdm1_tmp * casci_bra.ci[iabra, ibbra] * casci_ket.ci[iaket, ibket]
                    )
                    rdm2 += (
                        rdm2_tmp.reshape(rdm2.shape)
                        * casci_bra.ci[iabra, ibbra]
                        * casci_ket.ci[iaket, ibket]
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

                if rank == 0:
                    overlap_new[-1, i] = overlap_accumulate
                    overlap_new[i, -1] = overlap_accumulate.conj()
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

                    one_rdm_new[-1, i, :, :] = rdm1
                    one_rdm_new[i, -1, :, :] = rdm1.conj()
                    
                    if not lowrank:
                        two_rdm_new[-1, i, :, :, :, :] = rdm2
                        two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()
                        #two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()
                    
                    # Low rank
                    else:
                        # Get low rank representation
                        diagonals, lowrank_vecs = \
                            reduce_2rdm(rdm1, rdm2, overlap_accumulate, 
                                        mol=mol, train_en=e,
                                        **self.kwargs)
                        
                        diagonals_conj, lowrank_vecs_conj = \
                            reduce_2rdm(rdm1.conj(), rdm2.conj(), overlap_accumulate,        
                                        mol=mol, train_en=e,
                                        **self.kwargs)
                        #    reduce_2rdm(rdm1_conj, rdm2_conj, ovlp,        
                        #                mol=mol, train_en=e,
                        #                **self.kwargs)
                        
                        cum_diagonal_new[-1, i, :, :, :] = diagonals
                        cum_diagonal_new[i, -1, :, :, :] = diagonals_conj
                        
                        vecs_lowrank[(n_cascis-1, i)] = lowrank_vecs
                        vecs_lowrank[(i,n_cascis-1)] = lowrank_vecs_conj
                        

            self.overlap = overlap_new
            self.one_rdm = one_rdm_new
            if not lowrank:
                self.two_rdm = two_rdm_new
            else:
                self.cum_diagonal = cum_diagonal_new
                self.vecs_lowrank = vecs_lowrank

    def append_to_rdms(self, mol):
        """
        Append a new training geometry. See pygnme examples for more information about
        the evaluation of the t-RDMs.

        Args:
            mol (object): Molecular object of the training geometry.

        Raises:
            AssertionError: If the mean-field calculation is not converged.
        """
        # Some checks
        if self.use_rdm is None:
            use_rdm = True
        elif not self.use_rdm:
            print('Error in append_to_rdms: already using add_state')
            sys.exit()

        # Run mean field calculations for the orbitals
        #mf = mol.copy().RHF()
        mf = scf.RHF(mol.copy())
        mf.kernel()

        assert mf.converged

        MPI.COMM_WORLD.Bcast(mf.mo_coeff)

        # Specificy the CAS solver for the current state
        #casci_bra_all = self.casci_solver(mf, self.ncas, self.neleca)
        #casci_bra_all.fcisolver.nroots = self.nroots

        if self.solver == 'SA-CASSCF':
            mc = mcscf.CASSCF(mf, self.ncas, self.neleca).state_average_([1/self.nroots]*self.nroots)
            mc.kernel()
            mo_sacasscf = mc.mo_coeff

        # Iterate over different states
        for istate in range(self.nroots):

            # Read the DM representation from existing training states
            overlap = self.overlap
            one_rdm = self.one_rdm
            two_rdm = self.two_rdm

            if self.solver == 'CASCI':
                casci_bra = mcscf.CASCI(mf, self.ncas, self.neleca).state_specific_(istate)
            elif self.solver == 'SS-CASSCF':
                cas_ss = mcscf.CASSCF(mf, self.ncas, self.neleca).state_specific_(istate)
                cas_ss.kernel()
                casci_bra = mcscf.CASCI(mf, self.ncas, self.neleca).state_specific_(istate)
                casci_bra.casci(cas_ss.mo_coeff)
            else:
                casci_bra = mcscf.CASCI(mf, self.ncas, self.neleca).state_specific_(istate)
                casci_bra.casci(mo_sacasscf)

            self.cascis.append(casci_bra)

            cascis = self.cascis
            n_cascis = len(cascis)

            casci_bra.kernel()

            assert np.all(casci_bra.fcisolver.converged)

            if hasattr(casci_bra, "converged"):
                assert casci_bra.converged

            MPI.COMM_WORLD.Bcast(casci_bra.ci)
            MPI.COMM_WORLD.Bcast(casci_bra.mo_coeff)

            mo_coeff_bra = casci_bra.mo_coeff
            mol_bra = casci_bra.mol

            ovlp_bra = mol_bra.intor_symmetric("int1e_ovlp")
            basis_OAO_bra = get_basis(mol_bra)
            trafo_bra = basis_OAO_bra.T.dot(ovlp_bra).dot(mo_coeff_bra)

            bra_ref_state = wick.reference_state[float](
                mo_coeff_bra.shape[0],
                mo_coeff_bra.shape[0],
                mol_bra.nelec[0],
                casci_bra.ncas,
                casci_bra.ncore,
                owndata(mo_coeff_bra),
            )

            if rank == 0:
                overlap_new = np.zeros((n_cascis, n_cascis))
                if overlap is not None:
                    overlap_new[:-1, :-1] = overlap
                one_rdm_new = np.zeros(
                    (n_cascis, n_cascis, mo_coeff_bra.shape[0], mo_coeff_bra.shape[0])
                )
                if one_rdm is not None:
                    one_rdm_new[:-1, :-1, :, :] = one_rdm
                two_rdm_new = np.zeros(
                    (
                        n_cascis,
                        n_cascis,
                        mo_coeff_bra.shape[0],
                        mo_coeff_bra.shape[0],
                        mo_coeff_bra.shape[0],
                        mo_coeff_bra.shape[0],
                    )
                )
                if two_rdm is not None:
                    two_rdm_new[:-1, :-1, :, :, :, :] = two_rdm
            else:
                overlap_new = one_rdm_new = two_rdm_new = None

            bra_occ_strings = utils.fci_bitset_list(
                mol_bra.nelec[0] - casci_bra.ncore, casci_bra.ncas
            )

            for i in range(n_cascis):
                casci_ket = cascis[i]
                mo_coeff_ket = casci_ket.mo_coeff
                mol_ket = casci_ket.mol

                ovlp_ket = mol_ket.intor_symmetric("int1e_ovlp")
                basis_OAO_ket = get_basis(mol_ket)
                trafo_ket = basis_OAO_ket.T.dot(ovlp_ket).dot(mo_coeff_ket)

                trafo_ket_bra = basis_OAO_bra.dot(trafo_ket)

                ket_ref_state = wick.reference_state[float](
                    mo_coeff_ket.shape[0],
                    mo_coeff_ket.shape[0],
                    mol_ket.nelec[0],
                    casci_ket.ncas,
                    casci_ket.ncore,
                    owndata(trafo_ket_bra),
                )

                orbitals = wick.wick_orbitals[float, float](
                    bra_ref_state, ket_ref_state, owndata(ovlp_bra)
                )

                wick_mb = wick.wick_rscf[float, float, float](orbitals, 0.0)

                ket_occ_strings = utils.fci_bitset_list(
                    mol_ket.nelec[0] - casci_ket.ncore, casci_ket.ncas
                )

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
                    stringabra = bra_occ_strings[iabra]
                    stringbbra = bra_occ_strings[ibbra]
                    stringaket = ket_occ_strings[iaket]
                    stringbket = ket_occ_strings[ibket]

                    rdm1_tmp.fill(0.0)
                    rdm2_tmp.fill(0.0)
                    o = wick_mb.evaluate_rdm12(
                        stringabra,
                        stringbbra,
                        stringaket,
                        stringbket,
                        1.0,
                        rdm1_tmp,
                        rdm2_tmp,
                    )
                    overlap_accumulate += (
                        o * casci_bra.ci[iabra, ibbra] * casci_ket.ci[iaket, ibket]
                    )

                    rdm1 += (
                        rdm1_tmp * casci_bra.ci[iabra, ibbra] * casci_ket.ci[iaket, ibket]
                    )
                    rdm2 += (
                        rdm2_tmp.reshape(rdm2.shape)
                        * casci_bra.ci[iabra, ibbra]
                        * casci_ket.ci[iaket, ibket]
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

                if rank == 0:
                    overlap_new[-1, i] = overlap_accumulate
                    overlap_new[i, -1] = overlap_accumulate.conj()
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

                    one_rdm_new[-1, i, :, :] = rdm1
                    one_rdm_new[i, -1, :, :] = rdm1.conj()
                    two_rdm_new[-1, i, :, :, :, :] = rdm2
                    two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()
            self.overlap = overlap_new
            self.one_rdm = one_rdm_new
            self.two_rdm = two_rdm_new

    def states_to_rdms(self):
        """
        Construct transition RDMs between given training states (self.cascis)

        Raises:
            AssertionError: If the mean-field calculation is not converged.
        """
        # Some checks
        assert len(self.cascis) > 0

        assert self.two_rdm is None and self.one_rdm is None

        states = self.cascis

        n_cascis = len(states)
        # Iterate over bra states
        for a, casci_bra in enumerate(states):

            MPI.COMM_WORLD.Bcast(casci_bra.ci)
            MPI.COMM_WORLD.Bcast(casci_bra.mo_coeff)
            
            #st_bra = time()
            
            mo_coeff_bra = casci_bra.mo_coeff
            mol_bra = casci_bra.mol

            ovlp_bra = mol_bra.intor_symmetric("int1e_ovlp")
            basis_OAO_bra = get_basis(mol_bra)
            trafo_bra = basis_OAO_bra.T.dot(ovlp_bra).dot(mo_coeff_bra)
            #print('bra',ovlp_bra.shape,basis_OAO_bra.shape,trafo_bra.shape)

            bra_ref_state = wick.reference_state[float](
                mo_coeff_bra.shape[0],
                mo_coeff_bra.shape[0],
                mol_bra.nelec[0],
                casci_bra.ncas,
                casci_bra.ncore,
                owndata(mo_coeff_bra),
            )

            bra_occ_strings = utils.fci_bitset_list(
                mol_bra.nelec[0] - casci_bra.ncore, casci_bra.ncas
            )

            if rank == 0:
                overlap_new = np.zeros((n_cascis, n_cascis))

                one_rdm_new = np.zeros(
                    (n_cascis, n_cascis, mo_coeff_bra.shape[0], mo_coeff_bra.shape[0])
                )

                two_rdm_new = np.zeros(
                    (
                        n_cascis,
                        n_cascis,
                        mo_coeff_bra.shape[0],
                        mo_coeff_bra.shape[0],
                        mo_coeff_bra.shape[0],
                        mo_coeff_bra.shape[0],
                    )
                )
 
            else:
                overlap_new = one_rdm_new = two_rdm_new = None

            # Iterate over ket states
            #for b, casci_ket in enumerate(states):
            for b in range(n_cascis):
                casci_ket = states[b]
                #st_ket = time()

                # Prepare ket state                
                mo_coeff_ket = casci_ket.mo_coeff
                mol_ket = casci_ket.mol

                ovlp_ket = mol_ket.intor_symmetric("int1e_ovlp")
                basis_OAO_ket = get_basis(mol_ket)
                trafo_ket = basis_OAO_ket.T.dot(ovlp_ket).dot(mo_coeff_ket)

                trafo_ket_bra = basis_OAO_bra.dot(trafo_ket)
                #print('bra',ovlp_ket.shape,basis_OAO_ket.shape,trafo_ket.shape, trafo_ket_bra.shape)

                ket_ref_state = wick.reference_state[float](
                    mo_coeff_ket.shape[0],
                    mo_coeff_ket.shape[0],
                    mol_ket.nelec[0],
                    casci_ket.ncas,
                    casci_ket.ncore,
                    owndata(trafo_ket_bra),
                )

                ket_occ_strings = utils.fci_bitset_list(
                    mol_ket.nelec[0] - casci_ket.ncore, casci_ket.ncas
                )
                orbitals = wick.wick_orbitals[float, float](
                    bra_ref_state, ket_ref_state, owndata(ovlp_bra)
                )

                wick_mb = wick.wick_rscf[float, float, float](orbitals, 0.0)

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
                    stringabra = bra_occ_strings[iabra]
                    stringbbra = bra_occ_strings[ibbra]
                    stringaket = ket_occ_strings[iaket]
                    stringbket = ket_occ_strings[ibket]

                    rdm1_tmp.fill(0.0)
                    rdm2_tmp.fill(0.0)
                    o = wick_mb.evaluate_rdm12(
                        stringabra,
                        stringbbra,
                        stringaket,
                        stringbket,
                        1.0,
                        rdm1_tmp,
                        rdm2_tmp,
                    )
                    overlap_accumulate += (
                        o * casci_bra.ci[iabra, ibbra] * casci_ket.ci[iaket, ibket]
                    )

                    rdm1 += (
                        rdm1_tmp * casci_bra.ci[iabra, ibbra] * casci_ket.ci[iaket, ibket]
                    )
                    rdm2 += (
                        rdm2_tmp.reshape(rdm2.shape)
                        * casci_bra.ci[iabra, ibbra]
                        * casci_ket.ci[iaket, ibket]
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

                if rank == 0:
                    overlap_new[a, b] = overlap_accumulate
                    overlap_new[b, a] = overlap_accumulate.conj()
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

                    one_rdm_new[a, b, :, :] = rdm1
                    one_rdm_new[b, a, :, :] = rdm1.conj()
                    two_rdm_new[a, b, :, :, :, :] = rdm2
                    two_rdm_new[b, a, :, :, :, :] = rdm2.conj()
        self.overlap = overlap_new
        self.one_rdm = one_rdm_new
        self.two_rdm = two_rdm_new

    def prune_datapoints(self, keep_ids):
        """
        Prunes training points from the continuation object based on the given keep_ids.

        Args:
            keep_ids (list): List of indices to keep.

        Returns:
            None
        """
        if self.overlap is not None:
            self.overlap = self.overlap[np.ix_(keep_ids, keep_ids)]
        if self.one_rdm is not None:
            self.one_rdm = self.one_rdm[np.ix_(keep_ids, keep_ids)]
        if self.two_rdm is not None:
            self.two_rdm = self.two_rdm[np.ix_(keep_ids, keep_ids)]
        self.cascis = [self.cascis[i] for i in keep_ids]
