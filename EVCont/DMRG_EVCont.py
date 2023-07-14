import numpy as np

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from EVCont.electron_integral_utils import get_basis, get_integrals, transform_integrals
from EVCont.converge_dmrg import converge_dmrg

from EVCont.MPS_orb_rotation import converge_orbital_rotation_mps

from mpi4py import MPI


rank = MPI.COMM_WORLD.rank


def default_solver_fun(h1, h2, nelec, tag):
    return converge_dmrg(
        h1, h2, nelec, tag, tolerance=1.0e-5, mpi=(MPI.COMM_WORLD.size > 1)
    )


def append_to_rdms_rerun(
    mols,
    overlap=None,
    one_rdm=None,
    two_rdm=None,
    computational_basis="split",
    reorder_orbitals=True,
    converge_dmrg_fun=default_solver_fun,
):
    mol_bra = mols[-1]

    basis = get_basis(mol_bra, basis_type=computational_basis)
    h1, h2 = get_integrals(mol_bra, basis)

    norb = h1.shape[0]
    nelec = np.sum(mol_bra.nelec)

    mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2)
    mps_solver.initialize_system(norb, n_elec=nelec)

    if reorder_orbitals:
        orbital_reordering = mps_solver.orbital_reordering(h1, h2)

        basis = basis[:, orbital_reordering]

    h1, h2 = get_integrals(mol_bra, basis)

    bra, en = converge_dmrg_fun(h1, h2, nelec, "MPS_{}".format(len(mols) - 1))

    if rank == 0:
        np.save("basis_{}.npy".format(len(mols) - 1), basis)

    mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2)
    mps_solver.initialize_system(norb)

    ovlp_bra = mol_bra.intor_symmetric("int1e_ovlp")
    oao_basis_bra = get_basis(mol_bra, "OAO")

    overlap_new = np.ones((len(mols), len(mols)))
    if overlap is not None:
        overlap_new[:-1, :-1] = overlap
    one_rdm_new = np.ones((len(mols), len(mols), norb, norb))
    if one_rdm is not None:
        one_rdm_new[:-1, :-1, :, :] = one_rdm
    two_rdm_new = np.ones((len(mols), len(mols), norb, norb, norb, norb))
    if two_rdm is not None:
        two_rdm_new[:-1, :-1, :, :, :, :] = two_rdm

    for i, mol_ket in enumerate(mols):
        ket = mps_solver.load_mps("MPS_{}".format(i))
        computational_basis_ket = np.load("basis_{}.npy".format(i))
        ovlp_ket = mol_ket.intor_symmetric("int1e_ovlp")
        oao_basis_ket = get_basis(mol_ket, "OAO")

        # Transform ket into computational basis of bra
        computational_to_OAO_ket = oao_basis_ket.T.dot(ovlp_ket).dot(
            computational_basis_ket
        )
        computational_to_OAO_bra = oao_basis_bra.T.dot(ovlp_bra).dot(basis)
        orbital_rotation = (computational_to_OAO_bra.T.dot(computational_to_OAO_ket)).T

        if i != len(mols) - 1:
            h1, h2 = get_integrals(
                mol_ket, computational_basis_ket.dot(orbital_rotation)
            )
            transformed_ket, en = converge_dmrg_fun(
                h1, h2, nelec, "MPS_{}_{}".format(len(mols) - 1, i)
            )
        else:
            transformed_ket = ket

        ovlp = np.array(
            mps_solver.expectation(bra, mps_solver.get_identity_mpo(), transformed_ket)
        )
        o_RDM = np.array(mps_solver.get_1pdm(transformed_ket, bra=bra))
        t_RDM = np.array(
            np.transpose(mps_solver.get_2pdm(transformed_ket, bra=bra), (0, 3, 1, 2))
        )

        rdm1, rdm2 = transform_integrals(o_RDM, t_RDM, computational_to_OAO_bra)

        overlap_new[-1, i] = ovlp
        overlap_new[i, -1] = ovlp.conj()
        one_rdm_new[-1, i, :, :] = rdm1
        one_rdm_new[i, -1, :, :] = rdm1.conj()
        two_rdm_new[-1, i, :, :, :, :] = rdm2
        two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()

    return overlap_new, one_rdm_new, two_rdm_new


def append_to_rdms_orbital_rotation(
    mols,
    overlap=None,
    one_rdm=None,
    two_rdm=None,
    computational_basis="split",
    reorder_orbitals=True,
    converge_dmrg_fun=default_solver_fun,
):
    mol_bra = mols[-1]

    basis = get_basis(mol_bra, basis_type=computational_basis)
    MPI.COMM_WORLD.Bcast(basis)

    h1, h2 = get_integrals(mol_bra, basis)

    norb = h1.shape[0]
    nelec = np.sum(mol_bra.nelec)

    mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2)
    mps_solver.initialize_system(norb, n_elec=nelec)

    if reorder_orbitals:
        orbital_reordering = mps_solver.orbital_reordering(h1, h2)

        basis = basis[:, orbital_reordering]

    h1, h2 = get_integrals(mol_bra, basis)

    bra, en = converge_dmrg_fun(h1, h2, nelec, "MPS_{}".format(len(mols) - 1))

    if rank == 0:
        np.save("basis_{}.npy".format(len(mols) - 1), basis)

    mps_solver = DMRGDriver(symm_type=SymmetryTypes.SU2)
    mps_solver.initialize_system(norb)

    ovlp_bra = mol_bra.intor_symmetric("int1e_ovlp")
    oao_basis_bra = get_basis(mol_bra, "OAO")
    MPI.COMM_WORLD.Bcast(oao_basis_bra)

    overlap_new = np.ones((len(mols), len(mols)))
    if overlap is not None:
        overlap_new[:-1, :-1] = overlap
    one_rdm_new = np.ones((len(mols), len(mols), norb, norb))
    if one_rdm is not None:
        one_rdm_new[:-1, :-1, :, :] = one_rdm
    two_rdm_new = np.ones((len(mols), len(mols), norb, norb, norb, norb))
    if two_rdm is not None:
        two_rdm_new[:-1, :-1, :, :, :, :] = two_rdm

    for i, mol_ket in enumerate(mols):
        ket = mps_solver.load_mps("MPS_{}".format(i))
        computational_basis_ket = np.load("basis_{}.npy".format(i))
        ovlp_ket = mol_ket.intor_symmetric("int1e_ovlp")
        oao_basis_ket = get_basis(mol_ket, "OAO")

        # Transform ket into computational basis of bra
        computational_to_OAO_ket = oao_basis_ket.T.dot(ovlp_ket).dot(
            computational_basis_ket
        )
        computational_to_OAO_bra = oao_basis_bra.T.dot(ovlp_bra).dot(basis)
        orbital_rotation = (computational_to_OAO_bra.T.dot(computational_to_OAO_ket)).T

        init_bond_dim = max(ket.info.bond_dim, ket.info.bond_dim)

        if i != len(mols) - 1:
            transformed_ket = converge_orbital_rotation_mps(
                ket,
                orbital_rotation,
                tag="new",
                convergence_thresh=1.0e-6,
                init_bond_dim=init_bond_dim,
                iprint=0,
            )
        else:
            transformed_ket = ket

        ovlp = np.array(
            mps_solver.expectation(bra, mps_solver.get_identity_mpo(), transformed_ket)
        )
        o_RDM = np.array(mps_solver.get_1pdm(transformed_ket, bra=bra))
        t_RDM = np.array(
            np.transpose(mps_solver.get_2pdm(transformed_ket, bra=bra), (0, 3, 1, 2))
        )

        rdm1, rdm2 = transform_integrals(o_RDM, t_RDM, computational_to_OAO_bra)

        overlap_new[-1, i] = ovlp
        overlap_new[i, -1] = ovlp.conj()
        one_rdm_new[-1, i, :, :] = rdm1
        one_rdm_new[i, -1, :, :] = rdm1.conj()
        two_rdm_new[-1, i, :, :, :, :] = rdm2
        two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()

    return overlap_new, one_rdm_new, two_rdm_new
