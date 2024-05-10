import numpy as np
import sys

from evcont.electron_integral_utils import get_basis, get_integrals

from pyscf import scf, ao2mo, fci, symm

from pyscf.fci.addons import transform_ci

from evcont.ab_initio_gradients_loewdin import get_loewdin_trafo

from evcont.low_rank_utils import reduce_2rdm

class FCI_EVCont_obj:
    """
    FCI_EVCont_obj holds the data structure for the continuation from FCI states.
    """

    def __init__(
        self,
        cisolver=fci.direct_spin0.FCI(),
        cibasis='canonical',
        nroots=1,
        roots_train=None,
        irrep_name=None,
        lowrank=False,
        **kwargs
    ):
        """
        Initializes the FCI_EVCont_obj class.

        Args:
            cisolver: The pyscf cisolver routine.
            cibasis: The basis for solving ci for each mol
                    Note that after computation, the basis is converted to OAO
            nroots: Number of states to be solved.  Default is 1, the ground state.
            roots_train (list): Indices of states to include in the continuation
            irrep_name (string): If not None, only include states corresponding 
                                to the given symmetry irreducible representation
                
        Attributes:
            fcivecs (list): The FCI training states.
            ens (list): The FCI training energies.
            mol_index (list): The molecule indices of the FCI training states
            overlap (ndarray): Overlap matrix.
            one_rdm (ndarray): One-electron t-RDM.
            two_rdm (ndarray): Two-electron t-RDM.
        """
        self.cisolver = cisolver
        self.cibasis = cibasis
        
        self.nroots = nroots
        if roots_train == None:
            self.roots_train = list(range(nroots))
        else:
            self.roots_train = roots_train
            assert isinstance(roots_train,list)

        # Symmetry
        if irrep_name == None:
            self.use_symmetry = False
            self.irrep_name = None
        else:
            self.use_symmetry = True
            self.irrep_name = irrep_name
            
            # Need canonical basis
            assert cibasis == 'canonical'
        
        # Initialize attributes
        self.fcivecs = []
        self.ens = []
        self.ens_nuc = []
        self.mol_index = []
        self.overlap = None
        self.one_rdm = None
        self.two_rdm = None
        
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
    
    def append_to_rdms(self, mol):
        """
        Append a new training geometry by growing the t-RDMs.

        Args:
            mol (object): Molecular object of the training geometry.

        """
        # Relevant matrices for SAO basis
        #S = mol.intor("int1e_ovlp")
        #ao_mo_trafo = get_loewdin_trafo(S)
        
        basis = get_basis(mol,basis_type=self.cibasis)
        h1, h2 = get_integrals(mol, basis)
        
        nroots_train = max(self.roots_train)+1

        # Get FCI energies and wavefunctions in SAO basis
        if not self.use_symmetry:
            e_all, fcivec_all = self.cisolver.kernel(h1, h2, mol.nao, mol.nelec,
                                              nroots=nroots_train)
            
        else:
            orbsym = symm.label_orb_symm(mol, mol.irrep_id, mol.symm_orb, basis)
            
            e_all, fcivec_all = self.cisolver.kernel(h1, h2, mol.nao, mol.nelec, 
                                                     nroots=nroots_train, orbsym=orbsym,)
        
        # If ground state
        if nroots_train == 1:
            e_all = [e_all]
            fcivec_all = [fcivec_all]

        # Transform to OAO basis
        if self.cibasis != 'OAO':
            S = mol.intor("int1e_ovlp")
            basis_oao = get_basis(mol)

            u = np.einsum('ji,jk,kl->il',basis,S,basis_oao)
            
            fcivec_all = [transform_ci(fcivec_i,mol.nelec,u) for fcivec_i in fcivec_all]

        # Fix gauge; probably not necessary
        for fcivec_i in fcivec_all:
            # Set maximum element to be positive
            idx = np.unravel_index(np.argmax(np.abs(fcivec_i.real)),fcivec_i.shape)
            fcivec_i *= np.sign(fcivec_i[idx])
            
        # Setting molecular index
        if len(self.mol_index) == 0:
            mindex = 0
        else:
            mindex = max(self.mol_index) + 1
            
        # Iterate over ground and excited states include them in the training
        # if their index is in self.roots_train
        for ind in range(len(e_all)):
            if ind in self.roots_train:
                
                fcivec = fcivec_all[ind]
                e = e_all[ind]
                
                self.fcivecs.append(fcivec)
        
                self.ens.append(e)
                self.ens_nuc.append(mol.energy_nuc())
                self.mol_index.append(mindex)
                            
                new_ntrain = len(self.fcivecs)
                
                overlap_new = np.ones((new_ntrain, new_ntrain))
                if self.overlap is not None:
                    overlap_new[:-1, :-1] = self.overlap
                one_rdm_new = np.ones((len(self.fcivecs), len(self.fcivecs), mol.nao, mol.nao))
                if self.one_rdm is not None:
                    one_rdm_new[:-1, :-1, :, :] = self.one_rdm
                
                # Only define two_rdm if not lowrank
                if not self.lowrank:
                    two_rdm_new = np.ones(
                        (len(self.fcivecs), len(self.fcivecs), mol.nao, mol.nao, mol.nao, mol.nao)
                    )
                    if self.two_rdm is not None:
                        two_rdm_new[:-1, :-1, :, :, :, :] = self.two_rdm
                        
                else:
                    cum_diagonal_new = np.ones(
                        (len(self.fcivecs), len(self.fcivecs), 3, mol.nao, mol.nao)
                    )
                    if self.cum_diagonal is not None:
                        cum_diagonal_new[:-1, :-1, :, :, :] = self.cum_diagonal
                    
                        
                # Iterate over training states to add RDMs to the existing states
                for i in range(len(self.fcivecs)):
                    ovlp = self.fcivecs[-1].flatten().conj().dot(self.fcivecs[i].flatten())
                    overlap_new[-1, i] = ovlp
                    overlap_new[i, -1] = ovlp.conj()
                    rdm1, rdm2 = self.cisolver.trans_rdm12(
                        self.fcivecs[-1], self.fcivecs[i], mol.nao, mol.nelec
                    )
                    rdm1_conj, rdm2_conj = self.cisolver.trans_rdm12(
                        self.fcivecs[i], self.fcivecs[-1], mol.nao, mol.nelec
                    )
                    one_rdm_new[-1, i, :, :] = rdm1
                    one_rdm_new[i, -1, :, :] = rdm1_conj
                    #one_rdm_new[i, -1, :, :] = rdm1.conj()
                    
                    if not self.lowrank:
                        two_rdm_new[-1, i, :, :, :, :] = rdm2
                        two_rdm_new[i, -1, :, :, :, :] = rdm2_conj
                        #two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()
                    
                    # Low rank
                    else:
                        # Get low rank representation
                        diagonals, lowrank_vecs = \
                            reduce_2rdm(rdm1, rdm2, ovlp, 
                                        mol=mol, train_en=e,
                                        **self.kwargs)
                        
                        diagonals_conj, lowrank_vecs_conj = \
                            reduce_2rdm(rdm1_conj, rdm2_conj, ovlp,        
                                        mol=mol, train_en=e,
                                        **self.kwargs)
                        
                        cum_diagonal_new[-1, i, :, :, :] = diagonals
                        cum_diagonal_new[i, -1, :, :, :] = diagonals_conj
                        
                        self.vecs_lowrank[(new_ntrain-1, i)] = lowrank_vecs
                        self.vecs_lowrank[(i, new_ntrain-1)] = lowrank_vecs_conj
                        
        
                self.overlap = overlap_new
                self.one_rdm = one_rdm_new
                if not self.lowrank:
                    self.two_rdm = two_rdm_new
                else:
                    self.cum_diagonal = cum_diagonal_new

    def prune_datapoints(self, keep_ids):
        """
        Prunes training points from the continuation object based on the given keep_ids.

        Args:
            keep_ids (list): List of indices to keep.

        Returns:
            None
        """

        if self.nroots > 1 or self.lowrank:
            print('Error in prune_datapoints: Pruning has not been implemented for excited states or low rank implementations')
            sys.exit()
            
        if self.overlap is not None:
            self.overlap = self.overlap[np.ix_(keep_ids, keep_ids)]
        if self.one_rdm is not None:
            self.one_rdm = self.one_rdm[np.ix_(keep_ids, keep_ids)]
        if self.two_rdm is not None:
            self.two_rdm = self.two_rdm[np.ix_(keep_ids, keep_ids)]
        self.fcivecs = [self.fcivecs[i] for i in keep_ids]
        self.ens = [self.ens[i] for i in keep_ids]
