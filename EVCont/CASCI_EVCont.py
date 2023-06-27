import numpy as np

from EVCont.electron_integral_utils import get_basis, transform_integrals


def append_to_rdms(cascis, overlap=None, one_rdm=None, two_rdm=None):
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
