import numpy as np

from evcont.electron_integral_utils import get_basis, get_integrals

from pyscf import fci


class FCI_EVCont_obj:
    """
    FCI_EVCont_obj holds the data structure for the continuation from FCI states.
    """

    def __init__(
        self,
        cisolver=fci.direct_spin0.FCI(),
    ):
        """
        Initializes the FCI_EVCont_obj class.

        Args:
            cisolver: The pyscf cisolver routine.

        Attributes:
            fcivecs (list): The FCI training states.
            ens (list): The FCI training energies.
            overlap (ndarray): Overlap matrix.
            one_rdm (ndarray): One-electron t-RDM.
            two_rdm (ndarray): Two-electron t-RDM.
        """
        self.cisolver = cisolver

        self.fcivecs = []
        self.ens = []
        self.overlap = None
        self.one_rdm = None
        self.two_rdm = None

    def append_to_rdms(self, mol):
        """
        Append a new training geometry by growing the t-RDMs.

        Args:
            mol (object): Molecular object of the training geometry.

        """
        h1, h2 = get_integrals(mol, get_basis(mol))
        e, fcivec = self.cisolver.kernel(h1, h2, mol.nao, mol.nelec)

        self.fcivecs.append(fcivec)

        self.ens.append(e + mol.energy_nuc())

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
            one_rdm_new[-1, i, :, :] = rdm1
            one_rdm_new[i, -1, :, :] = rdm1.conj()
            two_rdm_new[-1, i, :, :, :, :] = rdm2
            two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()

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
