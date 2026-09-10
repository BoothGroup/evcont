import numpy as np
import pickle

from evcont.electron_integral_utils import get_basis, get_integrals
from evcont.basis.basis_utils import AbstractBasisMixin, run_hf

from evcont.low_rank_utils import reduce_2rdm, vectorize_lowrank
from evcont.solver_evaluation import EVContEvaluationMixin, maintain_two_rdm_compression
from evcont.solver_persistence import EVContPersistenceMixin

from pygnme import wick, utils

#from pyscf.mcscf.casci import CASCI
from pyscf import mcscf, gto

from mpi4py import MPI

from tqdm import tqdm
import sys
import os, re

###########################################################################
# Load Quantel if available
try:
    from quantel.ints.pyscf_integrals import PySCFMolecule, PySCFIntegrals
    from quantel.wfn.ss_casscf import SS_CASSCF
    from quantel.opt.mode_controlling import ModeControl

    QUANTEL_FOUND = True
except:
    QUANTEL_FOUND = False
###########################################################################

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


class CAS_EVCont_obj(
    AbstractBasisMixin, EVContEvaluationMixin, EVContPersistenceMixin
):
    """
    CAS_EVCont_obj holds the data structure for the continuation from CAS states.
    """

    def __init__(self, ncas, neleca, 
                nroots=1, solver='SS-CASSCF',
                software='pyscf', quantel_path=None, solutions_to_reconverge=None,
                lowrank=False,
                abstract_basis="meta-lowdin",
                abstract_basis_ref=None,
                abstract_basis_ref_mol=None,
                abstract_basis_kwargs=None,
                compress_two_rdm=False,
                **kwargs):
        """
        Initialize the CAS_EVCont_obj.

        Args:
            ncas (int): Number of CAS orbitals.
            neleca (int): Number of active space electrons.
            nroots (int): Number of states to be continued.
            solver (object): CAS solver type. Options: CASCI, SA-CASSCF, SS-CASSCF.
            software (str): Software backend to use ('pyscf' or 'quantel').
            quantel_path (str): Path to Quantel solution to continue from.
            solutions_to_reconverge (list): List of solutions to reconverge in the Quantel path.
            lowrank (bool): Whether to use low-rank approximation for 2-body t-RDMs.
            **kwargs: Additional keyword arguments for low-rank settings.

        Attributes:
            ncas (int): Number of CAS orbitals.
            neleca (int): Number of alpha electrons.

            overlap (ndarray): Overlap matrix.
            one_rdm (ndarray): One-electron t-RDM.
            two_rdm (ndarray): Two-electron t-RDM. (if not using low-rank)
            vecs_lowrank (dict): Low-rank decomposition of 2-body t-RDMs. (if using low-rank)

            mols (list): List of molecule objects for each state.
            mo_coeffs (list): List of MO coefficient matrices for each state.
            cis (list): List of CI vectors for each state.
            trafos (list): List of transformation matrices for each state.

            
        """

        self.ncas = ncas
        self.neleca = neleca
        self.abstract_basis = abstract_basis
        self.abstract_basis_ref = abstract_basis_ref
        self.abstract_basis_ref_mol = abstract_basis_ref_mol
        self.abstract_basis_kwargs = dict(abstract_basis_kwargs or {})
        self.compress_two_rdm = bool(compress_two_rdm)

        self.overlap = None
        self.one_rdm = None
        self.two_rdm = None

        # OBSOLETE: Keeping for the old routines, new routines use mo_coeffs and cis
        self.cascis = []

        self.mols = []
        self.mo_coeffs = []
        self.cis = []
        self.trafos = []

        #self.casci_solver = casci_solver
        self.nroots = nroots

        # Checks and sets solver/software related attributes
        self._input_checks(solver, software, nroots, quantel_path, solutions_to_reconverge)

        # Use each determinant as a separate state
        # EXPERIMENTAL: will turn into an input in the future
        self.uncontracted = False

        # Internal: Set flags for using add_state vs append_to_rdms
        # (to prevent double addition into self.cascis or missing states in tRDMs)
        self.use_rdm = None
        
        ### Initialize low-rank attributes
        ### Initialize low-rank attributes
        self.lowrank = lowrank
        if lowrank:
            #self.truncation_style = kwargs['truncation_style']
            self.kwargs = kwargs
            
        # Diagonals of 2-cumulants ([nbra, nket, 3, norb, norb])
        self.diagonal_lr = None 
        # Low rank eigendecomposition of the rest of 2-rdm
        # Old version: dictionary[(nbra, nket)] = (vals_trunc, vecs_trunc)
        # New version: dictionary['vals': np.array([nbra, nket, nvec]),
        #                         'vecs': np.array([nbra, nket, nvec, nao, nao])]
        self.vecs_lowrank = {}

        # Precomputation for OTF Hamiltonian
        self.precompute = False
        self.inv_OAO_all = []
        self.mb_all = None
        self.occ_strings_all = []

    def _input_checks(self, solver, software, nroots, quantel_path, solutions_to_reconverge):
        if solver in ['CASCI','SS-CASSCF','SA-CASSCF', 'casci','ss-casscf','sa-casscf']:
            self.solver = solver
        elif solver in ['CASSCF', 'casscf']:
            if nroots == 1:
                self.solver = 'SS-CASSCF'
            else:
                print('Warning: Solver should specificy state-averaged vs state-specific. Defaulting to state-averaged solver.')
                self.solver = 'SA-CASSCF'
        else:
            print(f'Unknown solver "{solver}" in CAS_EVCont_obj')
            sys.exit()

        # Check for the software
        if software in ['pyscf']:
            self.software = software
        elif software in ['quantel']:
            if QUANTEL_FOUND:
                if solver in ['SS-CASSCF']:
                    self.software = software
                    self.quantel_path = quantel_path
                    self.solutions_to_reconverge = solutions_to_reconverge
                else:
                    print('Unsupported solver for Quantel backend.')
                    sys.exit()
            else:
                print('Quantel package not found. Install Quantel or use pyscf as software backend.')
                sys.exit()

    def vectorize_lowrank(self,hermitian=True):        
        vectorize_lowrank(self,hermitian=hermitian)
        
    @maintain_two_rdm_compression
    def append_to_rdms(self, mol, state=None, quantel_tag='ref', debug=False):
        """
        Append a new training geometry. See pygnme examples for more information about
        the evaluation of the t-RDMs.

        Args:
            mol (object): Molecular object of the training geometry.
            state (list, optional): List of precomputed states to be added. If None, new states will be computed.
            quantel_tag (str, optional): Tag for quantel states if using quantel software. Default is 'ref' folder in self.quantel_path
            debug (bool, optional): If True, print debug information. Defaults to False.

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

        ## Preliminaries before state iterations
        # AO-SAO transformation
        ovlp_bra = mol.intor_symmetric("int1e_ovlp")
        basis_OAO_bra = self.get_abstract_basis(mol)
    
        if self.software == 'pyscf' and state is None:
            # Run mean field calculations for the orbitals
            mf = run_hf(mol.copy())

            #MPI.COMM_WORLD.Bcast(mf.mo_coeff)
            
            if self.solver == 'SA-CASSCF':
                cas_sa = mcscf.CASSCF(mf, self.ncas, self.neleca)
                if self.nroots > 1:
                    cas_sa = cas_sa.state_average_([1/self.nroots]*self.nroots)
                cas_sa.kernel()
                #mo_sacasscf = cas_sa.mo_coeff
                assert cas_sa.converged

            elif self.solver == 'CASCI':
                mc_casci = mcscf.CASCI(mf, self.ncas, self.neleca)
                mc_casci.fcisolver.nroots = self.nroots
                mc_casci.kernel()
                
                assert mc_casci.converged

        elif self.software == 'quantel' and state is None:
            # Quantel molecule object
            mol_q = PySCFMolecule(mol.atom, mol.basis, mol.unit)
            ints = PySCFIntegrals(mol_q)
            #metric = ints.overlap_matrix()
            #hcore  = ints.oei_matrix()

            def convert_to_mcscf(mol,wfn, ncas, neleca):
                mc = mcscf.CASCI(mol, ncas, neleca) 
                mc.fcisolver.max_cycle = 1
                mc.casci(wfn.mo_coeff,ci0=wfn.mat_ci[:,0])
                return mc
            
            # Save the new tag path
            # Create new geometry directory
            new_tag = create_next_geom_dir(self.quantel_path)

            # Save the geometry and integrals
            mol_q.tofile(os.path.join(self.quantel_path, new_tag, 'molecule.xyz'))

            h1_q = ints.oei_ao_to_mo(basis_OAO_bra, basis_OAO_bra)
            h2_q = np.einsum('pi,qj,pqrs,rk,sl->ijkl', basis_OAO_bra, basis_OAO_bra, ints.tei_array(), basis_OAO_bra, basis_OAO_bra,optimize=True)

            np.savetxt(os.path.join(self.quantel_path, new_tag, 'oei.dat'), h1_q)
            np.save(os.path.join(self.quantel_path, new_tag, 'tei.npy'), h2_q)

        # Iterate over different states
        if state is None:
            if self.software == 'quantel':
                nroots = len(self.solutions_to_reconverge)
            else:   
                nroots = self.nroots
        else:
            nroots = len(state)

        for istate in range(nroots):

            # Read the DM representation from existing training states
            overlap = self.overlap
            one_rdm = self.one_rdm
            if not lowrank:
                two_rdm = self.two_rdm
            else:
                diagonal_lr = self.diagonal_lr
                vecs_lowrank = self.vecs_lowrank

            if state is None and self.software == 'pyscf':
                if self.solver == 'CASCI':
                    #casci_bra = mcscf.CASCI(mf, self.ncas, self.neleca).state_specific_(istate)
                    mo_coeff_bra = mc_casci.mo_coeff
                    mol_bra = mc_casci.mol

                    if self.nroots > 1:
                        ci_bra = mc_casci.ci[istate]
                        e = mc_casci.e_tot[istate]
                    else:
                        ci_bra = mc_casci.ci
                        e = mc_casci.e_tot

                    ncas = mc_casci.ncas
                    ncore = mc_casci.ncore

                elif self.solver == 'SA-CASSCF':
                    # casci_bra = mcscf.CASCI(mf, self.ncas, self.neleca).state_specific_(istate)
                    # casci_bra.casci(mo_sacasscf)
                    mo_coeff_bra = cas_sa.mo_coeff
                    mol_bra = cas_sa.mol
                    if self.nroots > 1:
                        ci_bra = cas_sa.ci[istate]
                        e = cas_sa.e_states[istate]
                    else:
                        ci_bra = cas_sa.ci
                        e = cas_sa.e_tot

                    ncas = cas_sa.ncas
                    ncore = cas_sa.ncore

                elif self.solver == 'SS-CASSCF':
                    cas_ss = mcscf.CASSCF(mf, self.ncas, self.neleca).state_specific_(istate)
                    cas_ss.kernel()
                    #casci_bra = mcscf.CASCI(mf, self.ncas, self.neleca).state_specific_(istate)
                    #casci_bra.casci(cas_ss.mo_coeff)

                    mo_coeff_bra = cas_ss.mo_coeff
                    mol_bra = cas_ss.mol
                    ci_bra = cas_ss.ci
            
                    e = cas_ss.e_tot

                    assert cas_ss.converged

                    ncas = cas_ss.ncas
                    ncore = cas_ss.ncore

                #else:


            elif self.software == 'quantel' and state is None:
                # Quantel molecule object
                state_ind = self.solutions_to_reconverge[istate]+1
                wfn = SS_CASSCF(ints, (self.ncas,self.neleca))
                wfn.initialise(np.genfromtxt(os.path.join(self.quantel_path, quantel_tag, f'{state_ind:04d}.mo_coeff')),
                               np.genfromtxt(os.path.join(self.quantel_path, quantel_tag, f'{state_ind:04d}.mat_ci')))
                
                # Reconverge solution at the new geometry
                ModeControl().run(wfn)

                # Save the reconverged wavefunction
                state_path = os.path.join(self.quantel_path, new_tag)
                np.savetxt(os.path.join(state_path,f'{state_ind:04d}.mo_coeff'), wfn.mo_coeff)
                np.savetxt(os.path.join(state_path, f'{state_ind:04d}.mat_ci'), wfn.mat_ci)

                casci_bra = convert_to_mcscf(mol,wfn, self.ncas, self.neleca)
                mo_coeff_bra = casci_bra.mo_coeff
                mol_bra = casci_bra.mol
                ci_bra = casci_bra.ci

                #e = wfn.energy
                out = casci_bra.kernel()
                e = out[0]

                ncas = casci_bra.ncas
                ncore = casci_bra.ncore
                nelec = mol_bra.nelec

            else:
                casci_bra = state[istate]

                mo_coeff_bra = casci_bra.mo_coeff
                mol_bra = casci_bra.mol
                ci_bra = casci_bra.ci

                out = casci_bra.kernel()
                e = out[0]

                assert np.all(casci_bra.fcisolver.converged)

                if hasattr(casci_bra, "converged"):
                    assert casci_bra.converged

                ncas = casci_bra.ncas
                ncore = casci_bra.ncore
                nelec = mol_bra.nelec

            nelec = mol_bra.nelec

            # Check if this new state is already stored (mo_coeff_bra and ci_bra are within a threshold of any element of self.mo_coeffs and self.cis)
            state_exists = False
            for i in range(len(self.cis)):
                mo_coeff_existing = self.mo_coeffs[i]
                ci_existing = self.cis[i]

                mo_diff = np.linalg.norm(mo_coeff_existing - mo_coeff_bra)
                ci_diff = np.linalg.norm(ci_existing - ci_bra)

                if mo_diff < 1e-6 and ci_diff < 1e-6:
                    print('Warning: The appended state is already stored in the training set. Skipping addition.')
                    state_exists = True
                    break

            if state_exists:
                continue

            # New version: store MO coeffs and CI vectors separately
            self.mo_coeffs.append(mo_coeff_bra)
            self.cis.append(ci_bra)
            self.mols.append(mol_bra)
            
            trafo_bra = basis_OAO_bra.T.dot(ovlp_bra).dot(mo_coeff_bra)

            self.trafos.append(trafo_bra)

            mo_coeffs = self.mo_coeffs
            cis = self.cis
            mols = self.mols
            trafos = self.trafos
            n_cascis = len(cis)

            MPI.COMM_WORLD.Bcast(ci_bra)
            MPI.COMM_WORLD.Bcast(mo_coeff_bra)

            bra_ref_state = wick.reference_state[float](
                mo_coeff_bra.shape[0],
                mo_coeff_bra.shape[0],
                mol_bra.nelec[0],
                ncas,
                ncore,
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
                    diagonal_lr_new = np.ones(
                        (n_cascis,
                         n_cascis, 3, mo_coeff_bra.shape[0], mo_coeff_bra.shape[0])
                    )
                    if diagonal_lr is not None:
                        diagonal_lr_new[:-1, :-1, :, :, :] = diagonal_lr
                    
            else:
                overlap_new = one_rdm_new = two_rdm_new = None

            bra_occ_strings = utils.fci_bitset_list(
                mol_bra.nelec[0] - ncore, ncas
            )

            for i in range(n_cascis):
                mo_coeff_ket = mo_coeffs[i]
                ci_ket = cis[i]

                trafo_ket = trafos[i]

                trafo_ket_bra = basis_OAO_bra.dot(trafo_ket)

                ket_ref_state = wick.reference_state[float](
                    mo_coeff_ket.shape[0],
                    mo_coeff_ket.shape[0],
                    nelec[0],
                    ncas,
                    ncore,
                    owndata(trafo_ket_bra),
                )

                orbitals = wick.wick_orbitals[float, float](
                    bra_ref_state, ket_ref_state, owndata(ovlp_bra)
                )

                wick_mb = wick.wick_rscf[float, float, float](orbitals, 0.0)

                ket_occ_strings = utils.fci_bitset_list(
                    nelec[0] - ncore, ncas
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
                        o * ci_bra[iabra, ibbra] * ci_ket[iaket, ibket]
                    )

                    rdm1 += (
                        rdm1_tmp * ci_bra[iabra, ibbra] * ci_ket[iaket, ibket]
                    )
                    rdm2 += (
                        rdm2_tmp.reshape(rdm2.shape)
                        * ci_bra[iabra, ibbra]
                        * ci_ket[iaket, ibket]
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

                    if debug:
                        np.save('rdm2_%i_%i.npy'%(n_cascis-1, i),rdm2)

                    one_rdm_new[-1, i, :, :] = rdm1
                    one_rdm_new[i, -1, :, :] = rdm1.conj().T
                    
                    if not lowrank:
                        two_rdm_new[-1, i, :, :, :, :] = rdm2
                        two_rdm_new[i, -1, :, :, :, :] = np.einsum('ijkl->lkji',rdm2.conj())
                    
                    # Low rank
                    else:
                        # Get low rank representation
                        print('States: %i %i, overlap: %f' % (n_cascis-1, i, overlap_accumulate))

                        lowrank_vecs, diagonals, use_joint = \
                            reduce_2rdm(rdm1, rdm2, overlap_accumulate, 
                                        mol=mol, train_en=e,
                                        abstract_basis=self.abstract_basis,
                                        basis_ref=self.abstract_basis_ref,
                                        basis_kwargs=self.abstract_basis_kwargs,
                                        **self.kwargs)
                        
                        diagonal_lr_new[-1, i, :, :, :] = diagonals
                        try:
                            # This gives an error if diagonals are not saved and set to None by reduce_2rdm
                            diagonal_lr_new[i, -1, :, :, :] = diagonals.conj()
                        except:
                            diagonal_lr_new[i, -1, :, :, :] = diagonals

                        #diagonal_lr_new[i, -1, :, :, :] = diagonals_conj
                        
                        # Data structure [(bra,ket)]; eval, leftvec, rightvec, use_joint
                        vecs_lowrank[(n_cascis-1, i)] = lowrank_vecs[0], lowrank_vecs[1], lowrank_vecs[2], use_joint
                        vecs_lowrank[(i,n_cascis-1)] = lowrank_vecs[0].conj(), lowrank_vecs[1].conj(), lowrank_vecs[2].conj(), use_joint
                        

            self.overlap = overlap_new
            self.one_rdm = one_rdm_new
            if not lowrank:
                self.two_rdm = two_rdm_new
            else:
                self.diagonal_lr = diagonal_lr_new
                self.vecs_lowrank = vecs_lowrank

    # Experimental: Uncontracted CAS continuation. Can be combined with append_to_rdms once tested
    @maintain_two_rdm_compression
    def append_to_rdms_separate_determinants(self, mol, state=None, debug=False):
        """
        Append a new training geometry with each determinant as a separate state.
        Modified version that creates separate states for each determinant instead of summing them.

        Args:
            mol (object): Molecular object of the training geometry.

        Raises:
            AssertionError: If the mean-field calculation is not converged.
        """
        # Some checks
        if self.use_rdm is None:
            use_rdm = True
        elif not self.use_rdm:
            print('Error in append_to_rdms_separate_determinants: already using add_state')
            sys.exit()

        lowrank = self.lowrank

        # Run mean field calculations for the orbitals
        mf = run_hf(mol.copy())

        MPI.COMM_WORLD.Bcast(mf.mo_coeff)

        if state is None:
            if self.solver == 'SA-CASSCF':
                cas_sa = mcscf.CASSCF(mf, self.ncas, self.neleca).state_average_([1/self.nroots]*self.nroots)
                cas_sa.kernel()
                assert cas_sa.converged

            elif self.solver == 'CASCI':
                mc_casci = mcscf.CASCI(mf, self.ncas, self.neleca)
                mc_casci.fcisolver.nroots = self.nroots
                mc_casci.kernel()
                assert mc_casci.converged

        # Iterate over different states
        if state is None:
            nroots = self.nroots
        else:
            nroots = len(state)

        for istate in range(nroots):
            # Read the DM representation from existing training states
            overlap = self.overlap
            one_rdm = self.one_rdm
            if not lowrank:
                two_rdm = self.two_rdm
            else:
                diagonal_lr = self.diagonal_lr
                vecs_lowrank = self.vecs_lowrank

            if state is None:
                if self.solver == 'CASCI':
                    mo_coeff_bra = mc_casci.mo_coeff
                    mol_bra = mc_casci.mol
                    ci_bra = mc_casci.ci[istate]
                    e = mc_casci.e_tot[istate]
                    ncas = mc_casci.ncas
                    ncore = mc_casci.ncore

                elif self.solver == 'SA-CASSCF':
                    mo_coeff_bra = cas_sa.mo_coeff
                    mol_bra = cas_sa.mol
                    ci_bra = cas_sa.ci[istate]
                    e = cas_sa.e_states[istate]
                    ncas = cas_sa.ncas
                    ncore = cas_sa.ncore

                elif self.solver == 'SS-CASSCF':
                    cas_ss = mcscf.CASSCF(mf, self.ncas, self.neleca).state_specific_(istate)
                    cas_ss.kernel()
                    mo_coeff_bra = cas_ss.mo_coeff
                    mol_bra = cas_ss.mol
                    ci_bra = cas_ss.ci
                    e = cas_ss.e_tot
                    assert cas_ss.converged
                    ncas = cas_ss.ncas
                    ncore = cas_ss.ncore
            else:
                casci_bra = state[istate]
                mo_coeff_bra = casci_bra.mo_coeff
                mol_bra = casci_bra.mol
                ci_bra = casci_bra.ci
                out = casci_bra.kernel()
                e = out[0]
                assert np.all(casci_bra.fcisolver.converged)
                if hasattr(casci_bra, "converged"):
                    assert casci_bra.converged
                ncas = casci_bra.ncas
                ncore = casci_bra.ncore

            nelec = mol_bra.nelec

            # Get determinant strings for this state
            bra_occ_strings = utils.fci_bitset_list(mol_bra.nelec[0] - ncore, ncas)
            

            # Efficiently handle the single-determinant case: only process the nonzero element
            nz = np.argwhere(np.abs(ci_bra) > 1e-10)
            print(f"State {istate}: Found {len(nz)} significant determinants")
            for det_idx, (iabra, ibbra) in enumerate(nz):
                ovlp_bra = mol_bra.intor_symmetric("int1e_ovlp")
                basis_OAO_bra = self.get_abstract_basis(mol_bra)
                trafo_bra = basis_OAO_bra.T.dot(ovlp_bra).dot(mo_coeff_bra)

                self.mo_coeffs.append(mo_coeff_bra)
                ci_single_det = np.zeros_like(ci_bra)
                ci_single_det[iabra, ibbra] = 1.0
                self.cis.append(ci_single_det)
                self.mols.append(mol_bra)
                self.trafos.append(trafo_bra)

                mo_coeffs = self.mo_coeffs
                cis = self.cis
                mols = self.mols
                trafos = self.trafos
                n_cascis = len(cis)

                MPI.COMM_WORLD.Bcast(ci_single_det)
                MPI.COMM_WORLD.Bcast(mo_coeff_bra)

                bra_ref_state = wick.reference_state[float](
                    mo_coeff_bra.shape[0],
                    mo_coeff_bra.shape[0],
                    mol_bra.nelec[0],
                    ncas,
                    ncore,
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
                        diagonal_lr_new = np.ones(
                            (n_cascis, n_cascis, 3, mo_coeff_bra.shape[0], mo_coeff_bra.shape[0])
                        )
                        if diagonal_lr is not None:
                            diagonal_lr_new[:-1, :-1, :, :, :] = diagonal_lr
                else:
                    overlap_new = one_rdm_new = two_rdm_new = None

                # Only need to process the single nonzero determinant for bra
                iabra_bra, ibbra_bra = iabra, ibbra
                for i in range(n_cascis):
                    mo_coeff_ket = mo_coeffs[i]
                    ci_ket = cis[i]
                    trafo_ket = trafos[i]

                    trafo_ket_bra = basis_OAO_bra.dot(trafo_ket)

                    ket_ref_state = wick.reference_state[float](
                        mo_coeff_ket.shape[0],
                        mo_coeff_ket.shape[0],
                        nelec[0],
                        ncas,
                        ncore,
                        owndata(trafo_ket_bra),
                    )

                    orbitals = wick.wick_orbitals[float, float](
                        bra_ref_state, ket_ref_state, owndata(ovlp_bra)
                    )

                    wick_mb = wick.wick_rscf[float, float, float](orbitals, 0.0)

                    ket_occ_strings = utils.fci_bitset_list(nelec[0] - ncore, ncas)

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

                    # Only process the nonzero element in ci_ket
                    nz_ket = np.argwhere(np.abs(ci_ket) > 1e-10)
                    for iabra_ket, ibbra_ket in nz_ket:
                        stringabra = bra_occ_strings[iabra_bra]
                        stringbbra = bra_occ_strings[ibbra_bra]
                        stringaket = ket_occ_strings[iabra_ket]
                        stringbket = ket_occ_strings[ibbra_ket]

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
                        overlap_accumulate += o * ci_ket[iabra_ket, ibbra_ket]
                        rdm1 += rdm1_tmp * ci_ket[iabra_ket, ibbra_ket]
                        rdm2 += rdm2_tmp.reshape(rdm2.shape) * ci_ket[iabra_ket, ibbra_ket]

                    overlap_accumulate = MPI.COMM_WORLD.allreduce(overlap_accumulate, op=MPI.SUM)
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

                        if debug:
                            np.save('rdm2_det_%i_%i_%i.npy'%(istate, det_idx, i), rdm2)
                        
                        one_rdm_new[-1, i, :, :] = rdm1
                        one_rdm_new[i, -1, :, :] = rdm1.conj().T

                        if not lowrank:
                            two_rdm_new[-1, i, :, :, :, :] = rdm2
                            two_rdm_new[i, -1, :, :, :, :] = np.einsum('ijkl->klij', rdm2.conj())
                        else:
                            print(f"Determinant {det_idx} of state {istate}, overlap with state {i}: {overlap_accumulate}")
                            lowrank_vecs, diagonals, use_joint = reduce_2rdm(
                                rdm1, rdm2, overlap_accumulate, 
                                mol=mol, train_en=e,
                                abstract_basis=self.abstract_basis,
                                basis_ref=self.abstract_basis_ref,
                                basis_kwargs=self.abstract_basis_kwargs,
                                **self.kwargs
                            )

                            diagonal_lr_new[-1, i, :, :, :] = diagonals
                            try:
                                diagonal_lr_new[i, -1, :, :, :] = diagonals.conj()
                            except:
                                diagonal_lr_new[i, -1, :, :, :] = diagonals

                            vecs_lowrank[(n_cascis-1, i)] = lowrank_vecs[0], lowrank_vecs[1], lowrank_vecs[2], use_joint
                            vecs_lowrank[(i, n_cascis-1)] = lowrank_vecs[0].conj(), lowrank_vecs[1].conj(), lowrank_vecs[2].conj(), use_joint

                self.overlap = overlap_new
                self.one_rdm = one_rdm_new
                if not lowrank:
                    self.two_rdm = two_rdm_new
                else:
                    if getattr(self, 'kwargs', {}).get('save_diag', True):
                        self.diagonal_lr = diagonal_lr_new
                    else:
                        self.diagonal_lr = None
                    self.vecs_lowrank = vecs_lowrank

                overlap = overlap_new
                one_rdm = one_rdm_new
                if not lowrank:
                    two_rdm = two_rdm_new
                else:
                    diagonal_lr = diagonal_lr_new

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
            basis_OAO_bra = self.get_abstract_basis(mol_bra)
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
                basis_OAO_ket = self.get_abstract_basis(mol_ket)
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
            basis_OAO_bra = self.get_abstract_basis(mol_bra)
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
                basis_OAO_ket = self.get_abstract_basis(mol_ket)
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
        mf = run_hf(mol.copy())

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

    @maintain_two_rdm_compression
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
        
        if rank == 0:
            overlap_new = np.zeros((n_cascis, n_cascis))
            one_rdm_new = np.zeros((n_cascis, n_cascis, states[0].mo_coeff.shape[0], states[0].mo_coeff.shape[0]))
            if not self.lowrank:
                two_rdm_new = np.zeros((n_cascis, n_cascis,
                                         states[0].mo_coeff.shape[0],
                                         states[0].mo_coeff.shape[0],
                                         states[0].mo_coeff.shape[0],
                                         states[0].mo_coeff.shape[0]))
            else:
                diagonal_lr_new = np.ones((n_cascis, n_cascis, 3,
                                           states[0].mo_coeff.shape[0],
                                           states[0].mo_coeff.shape[0]))
                vecs_lowrank = {}
        else:
            overlap_new = one_rdm_new = two_rdm_new = None
            if self.lowrank:
                diagonal_lr_new = None
                vecs_lowrank = None
                    
                
        # Iterate over bra states
        for a, casci_bra in enumerate(states):

            MPI.COMM_WORLD.Bcast(casci_bra.ci)
            MPI.COMM_WORLD.Bcast(casci_bra.mo_coeff)
            
            #st_bra = time()
            
            mo_coeff_bra = casci_bra.mo_coeff
            mol_bra = casci_bra.mol

            ovlp_bra = mol_bra.intor_symmetric("int1e_ovlp")
            basis_OAO_bra = self.get_abstract_basis(mol_bra)
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

            # Iterate over ket states
            #for b, casci_ket in enumerate(states):
            for b in range(n_cascis):
                casci_ket = states[b]
                #st_ket = time()

                # Prepare ket state                
                mo_coeff_ket = casci_ket.mo_coeff
                mol_ket = casci_ket.mol

                ovlp_ket = mol_ket.intor_symmetric("int1e_ovlp")
                basis_OAO_ket = self.get_abstract_basis(mol_ket)
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
                
                    if not self.lowrank:
                        two_rdm_new[a, b, :, :, :, :] = rdm2
                        two_rdm_new[b, a, :, :, :, :] = np.einsum('ijkl->klij', rdm2.conj())
                    else:
                        lowrank_vecs, diagonals, use_joint = reduce_2rdm(
                            rdm1,
                            rdm2,
                            overlap_accumulate,
                            mol=mol_bra,
                            train_en=casci_bra.e_tot,
                            abstract_basis=self.abstract_basis,
                            basis_ref=self.abstract_basis_ref,
                            basis_kwargs=self.abstract_basis_kwargs,
                            **self.kwargs,
                        )
                
                        diagonal_lr_new[a, b, :, :, :] = diagonals
                        try:
                            diagonal_lr_new[b, a, :, :, :] = diagonals.conj()
                        except:
                            diagonal_lr_new[b, a, :, :, :] = diagonals
                
                        vecs_lowrank[(a, b)] = (
                            lowrank_vecs[0],
                            lowrank_vecs[1],
                            lowrank_vecs[2],
                            use_joint,
                        )
                        vecs_lowrank[(b, a)] = (
                            lowrank_vecs[0].conj(),
                            lowrank_vecs[1].conj(),
                            lowrank_vecs[2].conj(),
                            use_joint,
                        )

        self.overlap = overlap_new
        self.one_rdm = one_rdm_new
        if not self.lowrank:
            self.two_rdm = two_rdm_new
        else:
            self.diagonal_lr = diagonal_lr_new
            self.vecs_lowrank = vecs_lowrank

    def prune_datapoints(self, keep_ids):
        """
        Prune training points (states/geometries) from the continuation object.

        This adapts to the newer storage model where individual state data are
        held in parallel lists (mo_coeffs, cis, trafos, mols) and low-rank data
        are stored in dictionaries keyed by (i,j) pairs.

        Parameters
        ----------
        keep_ids : sequence[int] or sequence[bool]
            Indices to keep (integer list) or a boolean mask of length n_states.
            Order is preserved as given.
        """
        # Normalize keep_ids: allow boolean mask
        import numpy as _np
        if isinstance(keep_ids, (list, tuple)) and len(keep_ids) > 0 and isinstance(keep_ids[0], (bool, _np.bool_)):
            keep_ids = [i for i, flag in enumerate(keep_ids) if flag]
        else:
            keep_ids = list(keep_ids)

        if len(keep_ids) == 0:
            raise ValueError("prune_datapoints: keep_ids is empty; refusing to drop all datapoints.")

        # Ensure indices are within range
        n_states = len(self.mo_coeffs)
        if any((i < 0 or i >= n_states) for i in keep_ids):
            raise IndexError("prune_datapoints: keep_ids contains out-of-range indices.")

        # Deduplicate while preserving order
        seen = set(); ordered_keep = []
        for i in keep_ids:
            if i not in seen:
                ordered_keep.append(i); seen.add(i)
        keep_ids = ordered_keep

        # Core square matrices/tensors
        if self.overlap is not None:
            self.overlap = self.overlap[_np.ix_(keep_ids, keep_ids)]
        if self.one_rdm is not None:
            # shape (n,n,nao,nao)
            self.one_rdm = self.one_rdm[_np.ix_(keep_ids, keep_ids)]
        if self.two_rdm is not None:
            self.two_rdm = self.two_rdm[_np.ix_(keep_ids, keep_ids)]
        if self.lowrank and self.diagonal_lr is not None:
            self.diagonal_lr = self.diagonal_lr[_np.ix_(keep_ids, keep_ids)]

        # Parallel lists of per-state data
        self.mo_coeffs = [self.mo_coeffs[i] for i in keep_ids]
        self.cis       = [self.cis[i] for i in keep_ids]
        self.trafos    = [self.trafos[i] for i in keep_ids]
        if hasattr(self, 'mols') and self.mols is not None and len(self.mols) == n_states:
            self.mols = [self.mols[i] for i in keep_ids]

        # Precompute-related arrays (if present)
        # inv_OAO_all, occ_strings_all, mb_all, orbitals_all created in precompute_for_otf
        if getattr(self, 'precompute', False):
            if hasattr(self, 'inv_OAO_all') and len(self.inv_OAO_all) == n_states:
                self.inv_OAO_all = [self.inv_OAO_all[i] for i in keep_ids]
            if hasattr(self, 'occ_strings_all') and len(self.occ_strings_all) == n_states:
                self.occ_strings_all = [self.occ_strings_all[i] for i in keep_ids]
            if hasattr(self, 'mb_all') and len(self.mb_all) == n_states:
                self.mb_all = [self.mb_all[i] for i in keep_ids]
            if hasattr(self, 'orbitals_all') and len(self.orbitals_all) == n_states:
                self.orbitals_all = [self.orbitals_all[i] for i in keep_ids]

        # Low-rank dictionary remapping
        if self.lowrank and self.vecs_lowrank is not None:
            vecs_lowrank_new = {}
            for new_i, old_i in enumerate(keep_ids):
                for new_j, old_j in enumerate(keep_ids):
                    key_old = (old_i, old_j)
                    if key_old in self.vecs_lowrank:
                        vecs_lowrank_new[(new_i, new_j)] = self.vecs_lowrank[key_old]
            self.vecs_lowrank = vecs_lowrank_new

        # Sanity: update counts if stored elsewhere
        # (No explicit n_states attribute; len(self.mo_coeffs) is authoritative.)
        return

    def save(self, filename):
        """
        Save the attributes of this CAS_EVCont_obj to a file.
        
        Args:
            filename (str): Path to the output file (will be saved as pickle)
        
        Returns:
            None
        """
        # Collect all the essential attributes
        cas_data = {
            # Basic parameters
            'ncas': self.ncas,
            'neleca': self.neleca,
            'nroots': self.nroots,
            'solver': self.solver,
            'lowrank': self.lowrank,
            'software': self.software,
            'quantel_path': getattr(self, 'quantel_path', None),
            'solutions_to_reconverge': getattr(self, 'solutions_to_reconverge', None),
            'abstract_basis': self.abstract_basis,
            'abstract_basis_ref': self.abstract_basis_ref,
            'abstract_basis_ref_mol': (
                None
                if self.abstract_basis_ref_mol is None
                else {
                    'atom': self.abstract_basis_ref_mol.atom,
                    'basis': self.abstract_basis_ref_mol.basis,
                    'unit': self.abstract_basis_ref_mol.unit,
                }
            ),
            'abstract_basis_kwargs': self.abstract_basis_kwargs,
            
            # RDM and overlap data
            'overlap': self.overlap,
            'one_rdm': self.one_rdm,
            'two_rdm': self.two_rdm,
            'compress_two_rdm': self.compress_two_rdm,
            
            # Low-rank specific data (if applicable)
            'diagonal_lr': self.diagonal_lr if self.lowrank else None,
            'vecs_lowrank': self.vecs_lowrank if self.lowrank else None,
            'kwargs': self.kwargs if self.lowrank else None,
            
            # State information
            'mo_coeffs': self.mo_coeffs,
            'cis': self.cis,
            'trafos': self.trafos,
            
            # Additional flags
            'uncontracted': self.uncontracted,
            'use_rdm': self.use_rdm,
            'precompute': self.precompute,

            # Save pyscf Molecule object geometries (not the full mols since it's not pickleable)
            'molecule_geometries': [mol.atom for mol in self.mols],
            'molecule_basis': [mol.basis for mol in self.mols],
            'molecule_unit': [mol.unit for mol in self.mols],
        }
        
        
        # Save to pickle file
        with open(filename, 'wb') as f:
            pickle.dump(cas_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        if rank == 0:
            print(f"CAS object saved to {filename}")

    @classmethod
    def load(cls, filename):
        """
        Load CAS_EVCont_obj attributes from a file and reinitialize the object.
        
        Args:
            filename (str): Path to the input file (pickle format)
        
        Returns:
            CAS_EVCont_obj: Reinitialized CAS_EVCont_obj instance
        """
        # Load the saved data
        with open(filename, 'rb') as f:
            cas_data = pickle.load(f)
        
        # Reinitialize the CAS object with basic parameters
        # Extract optional parameters with defaults for backward compatibility
        software = cas_data.get('software', 'pyscf')
        quantel_path = cas_data.get('quantel_path', None)
        solutions_to_reconverge = cas_data.get('solutions_to_reconverge', None)
        abstract_basis = cas_data.get('abstract_basis', 'meta-lowdin')
        abstract_basis_ref = cas_data.get('abstract_basis_ref', None)
        abstract_basis_ref_mol_data = cas_data.get('abstract_basis_ref_mol')
        abstract_basis_ref_mol = None
        if abstract_basis_ref_mol_data is not None:
            abstract_basis_ref_mol = gto.M(
                atom=abstract_basis_ref_mol_data['atom'],
                basis=abstract_basis_ref_mol_data['basis'],
                unit=abstract_basis_ref_mol_data['unit'],
                verbose=0,
            )
        abstract_basis_kwargs = cas_data.get('abstract_basis_kwargs', None)
        
        if cas_data['lowrank']:
            cas_obj = cls(
                cas_data['ncas'], 
                cas_data['neleca'],
                nroots=cas_data['nroots'],
                solver=cas_data['solver'],
                software=software,
                quantel_path=quantel_path,
                solutions_to_reconverge=solutions_to_reconverge,
                lowrank=True,
                abstract_basis=abstract_basis,
                abstract_basis_ref=abstract_basis_ref,
                abstract_basis_ref_mol=abstract_basis_ref_mol,
                abstract_basis_kwargs=abstract_basis_kwargs,
                compress_two_rdm=cas_data.get('compress_two_rdm', False),
                **cas_data['kwargs']
            )
        else:
            cas_obj = cls(
                cas_data['ncas'], 
                cas_data['neleca'],
                nroots=cas_data['nroots'],
                solver=cas_data['solver'],
                software=software,
                quantel_path=quantel_path,
                solutions_to_reconverge=solutions_to_reconverge,
                lowrank=False,
                abstract_basis=abstract_basis,
                abstract_basis_ref=abstract_basis_ref,
                abstract_basis_ref_mol=abstract_basis_ref_mol,
                abstract_basis_kwargs=abstract_basis_kwargs,
                compress_two_rdm=cas_data.get('compress_two_rdm', False),
            )
        
        # Restore RDM and overlap data
        cas_obj.overlap = cas_data['overlap']
        cas_obj.one_rdm = cas_data['one_rdm']
        cas_obj.two_rdm = cas_data['two_rdm']
        
        # Restore low-rank data if applicable
        if cas_data['lowrank']:
            cas_obj.diagonal_lr = cas_data['diagonal_lr']
            cas_obj.vecs_lowrank = cas_data['vecs_lowrank']
        
        # Restore state information
        cas_obj.mo_coeffs = cas_data['mo_coeffs']
        cas_obj.cis = cas_data['cis']
        cas_obj.trafos = cas_data['trafos']
        
        # Restore additional flags
        cas_obj.uncontracted = cas_data['uncontracted']
        cas_obj.use_rdm = cas_data['use_rdm']
        cas_obj.precompute = cas_data['precompute']

        # Restore molecule objects if geometries were saved
        if 'molecule_geometries' in cas_data:
            cas_obj.mols = []
            for geom, basis, unit in zip(cas_data['molecule_geometries'], cas_data['molecule_basis'], cas_data['molecule_unit']):
                mol = gto.Mole()
                mol.build(atom=geom, basis=basis, unit=unit, verbose=0)
                cas_obj.mols.append(mol)
        
        if rank == 0:
            print(f"CAS object loaded from {filename}")
            print(f"  ncas={cas_obj.ncas}, neleca={cas_obj.neleca}, nroots={cas_obj.nroots}")
            print(f"  solver={cas_obj.solver}, lowrank={cas_obj.lowrank}, software={cas_obj.software}")
            print(f"  Number of states: {len(cas_obj.cis)}")
            if cas_obj.software == 'quantel':
                print(f"  quantel_path={cas_obj.quantel_path}")
                print(f"  solutions_to_reconverge={cas_obj.solutions_to_reconverge}")
        
        return cas_obj


# Quantel specific parser functions
def create_next_geom_dir(quantel_path):
    """
    Create the next geometry directory in the Quantel path.
    Args:
        quantel_path (str): Path to the Quantel directory.  
    Returns:
        str: Name of the newly created geometry directory.
    """
    nums = []
    for name in os.listdir(quantel_path):
        if os.path.isdir(os.path.join(quantel_path, name)):
            m = re.fullmatch(r'geom(\d+)', name)
            if m:
                nums.append(int(m.group(1)))
    next_n = (max(nums) + 1) if nums else 1
    new_name = f"geom{next_n}"
    os.makedirs(os.path.join(quantel_path, new_name))
    return new_name
