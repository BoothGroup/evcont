import numpy as np

from EVCont.electron_integral_utils import get_basis, transform_integrals

from pygnme import wick, utils


def owndata(x):
    # CARMA requires numpy arrays to have data ownership
    if not x.flags["OWNDATA"]:
        y = np.zeros(x.shape, order="C")
        y[:] = x
        x = y
    assert x.flags["OWNDATA"]
    return x


def append_to_rdms_complete_space(cascis, overlap=None, one_rdm=None, two_rdm=None):
    n_cascis = len(cascis)
    casci_bra = cascis[-1]
    casci_bra.kernel()
    mo_coeff_bra = casci_bra.mo_coeff
    mol_bra = casci_bra.mol

    basis_MO_bra = mo_coeff_bra[:, casci_bra.ncore : casci_bra.ncore + casci_bra.ncas]
    ovlp_bra = mol_bra.intor_symmetric("int1e_ovlp")
    basis_OAO_bra = get_basis(mol_bra)
    trafo_bra = basis_OAO_bra.T.dot(ovlp_bra).dot(basis_MO_bra)

    overlap_new = np.ones((n_cascis, n_cascis))
    if overlap is not None:
        overlap_new[:-1, :-1] = overlap
    one_rdm_new = np.ones(
        (n_cascis, n_cascis, mo_coeff_bra.shape[0], mo_coeff_bra.shape[0])
    )
    if one_rdm is not None:
        one_rdm_new[:-1, :-1, :, :] = one_rdm
    two_rdm_new = np.ones(
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
    for i in range(n_cascis):
        casci_ket = cascis[i]
        mo_coeff_ket = casci_ket.mo_coeff
        mol_ket = casci_ket.mol

        basis_MO_ket = mo_coeff_ket[
            :, casci_ket.ncore : casci_ket.ncore + casci_ket.ncas
        ]
        ovlp_ket = mol_ket.intor_symmetric("int1e_ovlp")
        basis_OAO_ket = get_basis(mol_ket)
        trafo_ket = basis_OAO_ket.T.dot(ovlp_ket).dot(basis_MO_ket)

        trafo_ket_bra = trafo_ket.T.dot(trafo_bra)

        ket = casci_ket.fcisolver.transform_ci_for_orbital_rotation(
            casci_ket.ci, casci_ket.ncas, casci_ket.nelecas, trafo_ket_bra
        )

        ovlp = casci_bra.ci.flatten().conj().dot(ket.flatten())
        overlap_new[-1, i] = ovlp
        overlap_new[i, -1] = ovlp.conj()

        rdm1, rdm2 = casci_bra.fcisolver.trans_rdm12(
            casci_bra.ci, ket, casci_bra.ncas, casci_bra.nelecas
        )
        rdm1, rdm2 = transform_integrals(rdm1, rdm2, trafo_bra)

        one_rdm_new[-1, i, :, :] = rdm1
        one_rdm_new[i, -1, :, :] = rdm1.conj()
        two_rdm_new[-1, i, :, :, :, :] = rdm2
        two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()
    return overlap_new, one_rdm_new, two_rdm_new


def append_to_rdms(cascis, overlap=None, one_rdm=None, two_rdm=None):
    n_cascis = len(cascis)
    casci_bra = cascis[-1]
    casci_bra.kernel()
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
            bra_ref_state, ket_ref_state, owndata(mol_bra.get_ovlp())
        )

        wick_mb = wick.wick_rscf[float, float, float](orbitals, 0.0)

        ket_occ_strings = utils.fci_bitset_list(
            mol_ket.nelec[0] - casci_ket.ncore, casci_ket.ncas
        )

        rdm1_tmp = np.zeros((one_rdm_new.shape[2], one_rdm_new.shape[3]))
        rdm1 = np.zeros((one_rdm_new.shape[2], one_rdm_new.shape[3]))
        rdm2_tmp = np.zeros(
            (
                two_rdm_new.shape[2] * two_rdm_new.shape[3],
                two_rdm_new.shape[4] * two_rdm_new.shape[5],
            )
        )
        rdm2 = np.zeros(
            (
                two_rdm_new.shape[2],
                two_rdm_new.shape[3],
                two_rdm_new.shape[4],
                two_rdm_new.shape[5],
            )
        )
        overlap_accumulate = 0.0
        for iabra, stringabra in enumerate(bra_occ_strings):
            for ibbra, stringbbra in enumerate(bra_occ_strings):
                for iaket, stringaket in enumerate(ket_occ_strings):
                    for ibket, stringbket in enumerate(ket_occ_strings):
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
                            rdm1_tmp
                            * casci_bra.ci[iabra, ibbra]
                            * casci_ket.ci[iaket, ibket]
                        )
                        rdm2 += (
                            rdm2_tmp.reshape(rdm2.shape)
                            * casci_bra.ci[iabra, ibbra]
                            * casci_ket.ci[iaket, ibket]
                        )

        overlap_new[-1, i] = overlap_accumulate
        overlap_new[i, -1] = overlap_accumulate.conj()

        rdm1 = np.einsum("...ij,ai->...aj", rdm1, trafo_ket)
        rdm1 = np.einsum("...aj,bj->...ab", rdm1, trafo_bra)
        rdm2 = np.einsum("...ijkl,ai->...ajkl", rdm2, trafo_bra)
        rdm2 = np.einsum("...ajkl,bj->...abkl", rdm2, trafo_ket)
        rdm2 = np.einsum("...abkl,ck->...abcl", rdm2, trafo_bra)
        rdm2 = np.einsum("...abcl,dl->...abcd", rdm2, trafo_ket)

        one_rdm_new[-1, i, :, :] = rdm1
        one_rdm_new[i, -1, :, :] = rdm1.conj()
        two_rdm_new[-1, i, :, :, :, :] = rdm2
        two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()
    return overlap_new, one_rdm_new, two_rdm_new
