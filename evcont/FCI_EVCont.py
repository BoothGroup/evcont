import numpy as np

from evcont.electron_integral_utils import get_basis, get_integrals

from pyscf import scf, ao2mo, fci, symm

from pyscf.fci.addons import transform_ci

from evcont.ab_initio_gradients_loewdin import get_loewdin_trafo

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
        lowrank=False
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
        self.mol_index = []
        self.overlap = None
        self.one_rdm = None
        self.two_rdm = None

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
        
                self.ens.append(e + mol.energy_nuc())
                self.mol_index.append(mindex)
        
                overlap_new = np.ones((len(self.fcivecs), len(self.fcivecs)))
                if self.overlap is not None:
                    overlap_new[:-1, :-1] = self.overlap
                one_rdm_new = np.ones((len(self.fcivecs), len(self.fcivecs), mol.nao, mol.nao))
                if self.one_rdm is not None:
                    one_rdm_new[:-1, :-1, :, :] = self.one_rdm
                two_rdm_new = np.ones(
                    (len(self.fcivecs), len(self.fcivecs), mol.nao, mol.nao, mol.nao, mol.nao)
                )
                if self.two_rdm is not None:
                    two_rdm_new[:-1, :-1, :, :, :, :] = self.two_rdm
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
                    two_rdm_new[-1, i, :, :, :, :] = rdm2
                    two_rdm_new[i, -1, :, :, :, :] = rdm2_conj
                    #two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()
        
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
        self.fcivecs = [self.fcivecs[i] for i in keep_ids]
        self.ens = [self.ens[i] for i in keep_ids]
