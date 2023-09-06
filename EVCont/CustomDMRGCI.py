from pyscf import ao2mo

import numpy as np

from EVCont.electron_integral_utils import transform_integrals, get_basis

from EVCont.converge_dmrg import converge_dmrg

from EVCont.customCASCI import CustomCASCI

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from mpi4py import MPI


def default_solver_fun(h1, h2, nelec):
    return converge_dmrg(
        h1,
        h2,
        nelec,
        "MPS",
        tolerance=1.0e-4,
    )


class DMRGSolver:
    def __init__(
        self,
        computational_basis,
        converge_dmrg_fun=default_solver_fun,
        reorder_orbitals=False,
        mem=5,
    ):
        self.basis = computational_basis
        self.converge_dmrg_fun = converge_dmrg_fun
        self.reorder_orbitals = reorder_orbitals
        self.mem = mem

    def kernel(self, h1, h2, norb, nelec, ecore=0.0):
        MPI.COMM_WORLD.Bcast(self.mo_coeff)  # Just to be safe...

        basis_MO = self.mo_coeff
        ovlp = self.mol.intor_symmetric("int1e_ovlp")
        self.comp_basis = get_basis(self.mol, basis_type=self.basis)

        self.MO_computational_transformation = self.comp_basis.T.dot(ovlp).dot(basis_MO)
        MPI.COMM_WORLD.Bcast(self.MO_computational_transformation)

        # Likely there are faster ways to do this...
        h2_full = ao2mo.restore(1, h2, norb)
        h1_transformed, h2_transformed = transform_integrals(
            h1, h2_full, self.MO_computational_transformation
        )

        if self.reorder_orbitals:
            mps_solver = DMRGDriver(
                symm_type=SymmetryTypes.SU2,
                mpi=(MPI.COMM_WORLD.size > 1),
                stack_mem=self.mem << 30,
            )
            mps_solver.initialize_system(norb, n_elec=np.sum(nelec), spin=self.mol.spin)

            orbital_reordering = mps_solver.orbital_reordering(
                h1_transformed, h2_transformed
            )

            self.comp_basis = self.comp_basis[:, orbital_reordering]
            self.MO_computational_transformation = self.MO_computational_transformation[
                orbital_reordering, :
            ]
            h1_transformed, h2_transformed = transform_integrals(
                h1, h2_full, self.MO_computational_transformation
            )

        state, en = self.converge_dmrg_fun(
            h1_transformed,
            h2_transformed,
            nelec,
        )

        return ecore + en, state

    def make_rdm1(self, state, norb, nelec):
        MPI.COMM_WORLD.Bcast(self.mo_coeff)  # Just to be safe...
        mps_solver = DMRGDriver(
            symm_type=SymmetryTypes.SU2,
            mpi=(MPI.COMM_WORLD.size > 1),
            stack_mem=self.mem << 30,
        )
        mps_solver.initialize_system(
            norb, n_elec=np.sum(nelec), spin=(nelec[0] - nelec[1])
        )

        one_rdm = np.array(mps_solver.get_1pdm(state, bra=state))

        one_rdm_transformed = self.MO_computational_transformation.T.dot(
            one_rdm.dot(self.MO_computational_transformation)
        )

        return one_rdm_transformed

    def make_rdm12(self, state, norb, nelec):
        MPI.COMM_WORLD.Bcast(self.mo_coeff)  # Just to be safe...
        mps_solver = DMRGDriver(
            symm_type=SymmetryTypes.SU2,
            mpi=(MPI.COMM_WORLD.size > 1),
            stack_mem=self.mem << 30,
        )
        mps_solver.initialize_system(
            norb, n_elec=np.sum(nelec), spin=(nelec[0] - nelec[1])
        )

        one_rdm = np.array(mps_solver.get_1pdm(state, bra=state))
        two_rdm = np.array(
            np.transpose(mps_solver.get_2pdm(state, bra=state), (0, 3, 1, 2))
        )

        one_rdm_transformed, two_rdm_transformed = transform_integrals(
            one_rdm, two_rdm, self.MO_computational_transformation.T
        )

        return one_rdm_transformed, two_rdm_transformed


class CustomDMRGCI(CustomCASCI):
    def __init__(
        self,
        mf_or_mol,
        ncas,
        nelecas,
        computational_basis,
        converge_dmrg_fun=default_solver_fun,
        ncore=None,
        reorder_orbitals=False,
    ):
        super().__init__(mf_or_mol, ncas, nelecas, ncore=ncore)
        self.fcisolver = DMRGSolver(
            computational_basis,
            converge_dmrg_fun=converge_dmrg_fun,
            reorder_orbitals=reorder_orbitals,
        )
