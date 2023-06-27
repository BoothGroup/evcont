from pyscf import ao2mo

import numpy as np

from EVCont.electron_integral_utils import get_basis, transform_integrals

from EVCont.ab_initio_eigenvector_continuation import approximate_ground_state

from EVCont.customCASCI import CustomCASCI


class EVContSolver:
    def __init__(self, one_rdm, two_rdm, overlap):
        self.one_rdm = one_rdm
        self.two_rdm = two_rdm
        self.overlap = overlap

    def get_MO_OAO_transformation(self):
        basis_MO = self.mo_coeff
        ovlp = self.mol.intor_symmetric("int1e_ovlp")
        basis_OAO = get_basis(self.mol)
        return basis_OAO.T.dot(ovlp).dot(basis_MO)

    def kernel(self, h1, h2, norb, nelec, ecore=0.0):
        MO_OAO_trafo = self.get_MO_OAO_transformation()

        # Likely there are faster ways to do this...
        h2_full = ao2mo.restore(1, h2, norb)
        h1_transformed, h2_transformed = transform_integrals(h1, h2_full, MO_OAO_trafo)

        en, coeff = approximate_ground_state(
            h1_transformed, h2_transformed, self.one_rdm, self.two_rdm, self.overlap
        )

        return ecore + en, coeff

    def make_rdm1(self, fcivec, norb, nelec):
        one_rdm = np.zeros((norb, norb))

        MO_OAO_trafo = self.get_MO_OAO_transformation()

        # TODO: Numpyfy this!
        for i in range(len(fcivec)):
            for j in range(len(fcivec)):
                one_rdm += fcivec[i].conj() * fcivec[j] * self.one_rdm[i, j, :, :]

        one_rdm_transformed = MO_OAO_trafo.T.dot(one_rdm.dot(MO_OAO_trafo))

        return one_rdm_transformed

    def make_rdm12(self, fcivec, norb, nelec):
        one_rdm = np.zeros((norb, norb))
        two_rdm = np.zeros((norb, norb, norb, norb))

        MO_OAO_trafo = self.get_MO_OAO_transformation()

        # TODO: Numpyfy this!
        for i in range(len(fcivec)):
            for j in range(len(fcivec)):
                one_rdm += fcivec[i].conj() * fcivec[j] * self.one_rdm[i, j, :, :]
                two_rdm += fcivec[i].conj() * fcivec[j] * self.two_rdm[i, j, :, :, :, :]

        one_rdm_transformed, two_rdm_transformed = transform_integrals(
            one_rdm, two_rdm, MO_OAO_trafo.T
        )

        return one_rdm_transformed, two_rdm_transformed


class EVContCI(CustomCASCI):
    def __init__(self, mf_or_mol, ncas, nelecas, one_rdm, two_rdm, overlap, ncore=None):
        super().__init__(mf_or_mol, ncas, nelecas, ncore=ncore)
        self.fcisolver = EVContSolver(one_rdm, two_rdm, overlap)
