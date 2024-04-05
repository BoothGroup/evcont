import numpy as np

from evcont.electron_integral_utils import get_basis

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
                nroots=1, solver='SS-CASSCF'):
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


        #self.casci_solver.fcisolver.nroots = nroots

    def append_to_rdms(self, mol):
        """
        Append a new training geometry. See pygnme examples for more information about
        the evaluation of the t-RDMs.

        Args:
            mol (object): Molecular object of the training geometry.

        Raises:
            AssertionError: If the mean-field calculation is not converged.
        """
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
