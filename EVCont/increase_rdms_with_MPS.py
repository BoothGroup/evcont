import numpy as np

from pyscf import gto, md, fci
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from .MPS_orb_rotation import converge_orbital_rotation_mps

from .electron_integral_utils import get_basis, transform_integrals



def append_to_rdms(mols, overlap=None, one_rdm=None, two_rdm=None):
    mol_bra = mols[-1]
    computational_basis_bra = np.load("basis_{}.npy".format(str(hash(mol_bra))))
    norb = computational_basis_bra.shape[0]

    mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2)
    mps_solver.initialize_system(norb)

    bra = mps_solver.load_mps(str(hash(mol_bra))).deep_copy("bra_state")
    ovlp_bra = mol_bra.intor_symmetric('int1e_ovlp')
    oao_basis_bra = get_basis(mol_bra, "OAO")


    overlap_new = np.ones((len(mols), len(mols)))
    if overlap is not None:
        overlap_new[:-1,:-1] = overlap
    one_rdm_new = np.ones((len(mols), len(mols), norb, norb))
    if one_rdm is not None:
        one_rdm_new[:-1,:-1,:, :] = one_rdm
    two_rdm_new = np.ones((len(mols), len(mols), norb, norb, norb, norb))
    if two_rdm is not None:
        two_rdm_new[:-1,:-1,:,:,:,:] = two_rdm

    for i in range(len(mols)):
        mol_ket = mols[i]
        ket = mps_solver.load_mps(str(hash(mol_ket)))
        computational_basis_ket = np.load("basis_{}.npy".format(str(hash(mol_ket))))
        ovlp_ket = mol_ket.intor_symmetric('int1e_ovlp')
        oao_basis_ket = get_basis(mol_ket, "OAO")

        # Transform ket into MO basis of bra
        computational_to_OAO_ket = oao_basis_ket.T.dot(ovlp_ket).dot(computational_basis_ket)
        orbital_rotation = computational_basis_bra.T.dot(ovlp_bra).dot(oao_basis_bra).dot(computational_to_OAO_ket)

        if not np.allclose(orbital_rotation, np.eye(norb)):
            transformed_bra_state = converge_orbital_rotation_mps(bra, orbital_rotation)
        else:
            transformed_bra_state = bra

        ovlp = np.array(mps_solver.expectation(transformed_bra_state, mps_solver.get_identity_mpo(), ket))
        o_RDM = np.array(mps_solver.get_1pdm(ket, bra=transformed_bra_state))
        t_RDM = np.array(np.transpose(mps_solver.get_2pdm(ket, bra=transformed_bra_state), (0,3,1,2)))
        rdm1, rdm2 = transform_integrals(o_RDM, t_RDM, computational_to_OAO_ket)

        overlap_new[-1, i] = ovlp
        overlap_new[i, -1] = ovlp.conj()
        one_rdm_new[-1, i, :, :] = rdm1
        one_rdm_new[i, -1, :, :] = rdm1.conj()
        two_rdm_new[-1, i, :, :, :, :] = rdm2
        two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()

    return overlap_new, one_rdm_new, two_rdm_new
