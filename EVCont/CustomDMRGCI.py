from pyscf import ao2mo

import numpy as np

from .electron_integral_utils import get_basis, get_integrals, transform_integrals

from .ab_initio_eigenvector_continuation import approximate_ground_state

from .EVContCI import EVContCI


def default_solver_fun(h1, h2, nelec):
    return converge_dmrg(
        h1,
        h2,
        nelec,
        "MPS",
        tolerance=1.0e-4,
    )


class DMRGSolver:
    def __init__(self, computational_basis, converge_dmrg_fun=default_solver_fun):
        self.basis = computational_basis
        self.converge_dmrg_fun = converge_dmrg_fun

    def get_MO_computational_transformation(self):
        basis_MO = self.mo_coeff
        ovlp = self.mol.intor_symmetric("int1e_ovlp")
        computational_basis = get_basis(self.mol, basis_type=self.basis)
        return computational_basis.T.dot(ovlp).dot(basis_MO)

    def kernel(self, h1, h2, norb, nelec, ecore=0.0):
        MO_computational_transformation = self.get_MO_computational_transformation()

        # Likely there are faster ways to do this...
        h2_full = ao2mo.restore(1, h2, norb)
        h1_transformed, h2_transformed = transform_integrals(
            h1, h2_full, MO_computational_transformation
        )

        state, en = self.converge_dmrg_fun(
            h1_transformed,
            h2_full,
            nelec,
        )

        return ecore + en, state

    def make_rdm1(self, fcivec, norb, nelec):
        mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2)
        mps_solver.initialize_system(norb)

        MO_computational_trafo = self.get_MO_computational_transformation()

        one_rdm = np.array(mps_solver.get_1pdm(state, bra=state))

        one_rdm_transformed = MO_OAO_trafo.T.dot(one_rdm.dot(MO_OAO_trafo))

        return one_rdm_transformed, two_rdm_transformed

    def make_rdm12(self, state, norb, nelec):
        mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2)
        mps_solver.initialize_system(norb)

        MO_computational_trafo = self.get_MO_computational_transformation()

        one_rdm = np.array(mps_solver.get_1pdm(state, bra=state))
        two_rdm = np.array(
            np.transpose(mps_solver.get_2pdm(state, bra=state), (0, 3, 1, 2))
        )

        one_rdm_transformed, two_rdm_transformed = transform_integrals(
            one_rdm, two_rdm, MO_computational_trafo.T
        )

        return one_rdm_transformed, two_rdm_transformed


class CustomDMRGCI(EVContCI):
    def __init__(
        self,
        mf_or_mol,
        ncas,
        nelecas,
        computational_basis,
        converge_dmrg_fun=default_solver_fun,
        ncore=None,
    ):
        super().super().__init__(
            mf_or_mol, ncas, nelecas, converge_dmrg_fun=converge_dmrg_fun, ncore=ncore
        )
        self.fcisolver = DMRGSolver(computational_basis)
