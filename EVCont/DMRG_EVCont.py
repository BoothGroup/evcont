import numpy as np

from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from EVCont.electron_integral_utils import get_basis, get_integrals, transform_integrals
from EVCont.converge_dmrg import converge_dmrg

from EVCont.MPS_orb_rotation import converge_orbital_rotation_mps

from mpi4py import MPI


rank = MPI.COMM_WORLD.rank
n_ranks = MPI.COMM_WORLD.Get_size()


def append_to_rdms_rerun(
    mols,
    tags,
    overlap=None,
    one_rdm=None,
    two_rdm=None,
    computational_basis="split",
    reorder_orbitals=True,
    converge_dmrg_fun=converge_dmrg,
    enforce_symmetric=True,
):
    mol_bra = mols[-1]

    basis = get_basis(mol_bra, basis_type=computational_basis)
    h1, h2 = get_integrals(mol_bra, basis)

    norb = h1.shape[0]
    nelec = np.sum(mol_bra.nelec)

    mps_solver = DMRGDriver(
        symm_type=SymmetryTypes.SU2,
        mpi=(MPI.COMM_WORLD.size > 1),
        stack_mem=4 << 30,
    )
    mps_solver.initialize_system(norb, n_elec=nelec, spin=mol_bra.spin)

    if reorder_orbitals:
        orbital_reordering = mps_solver.orbital_reordering(h1, h2)

        basis = basis[:, orbital_reordering]

    MPI.COMM_WORLD.Bcast(basis, root=0)

    h1, h2 = get_integrals(mol_bra, basis)

    MPI.COMM_WORLD.Bcast(h1, root=0)

    h2_slice = np.empty((h2.shape[2], h2.shape[3]))

    for i in range(h2.shape[0]):
        for j in range(h2.shape[1]):
            np.copyto(h2_slice, h2[i, j, :, :])
            MPI.COMM_WORLD.Bcast(h2_slice, root=0)
            np.copyto(h2[i, j, :, :], h2_slice)

    bra, en = converge_dmrg_fun(h1, h2, mol_bra.nelec, "MPS_{}".format(tags[-1]))

    if rank == 0:
        np.save("basis_{}.npy".format(tags[-1]), basis)

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
        ket = mps_solver.load_mps("MPS_{}".format(tags[i]))
        computational_basis_ket = np.load("basis_{}.npy".format(tags[i]))
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

            MPI.COMM_WORLD.Bcast(h1, root=0)

            h2_slice = np.empty((h2.shape[2], h2.shape[3]))

            for i in range(h2.shape[0]):
                for j in range(h2.shape[1]):
                    np.copyto(h2_slice, h2[i, j, :, :])
                    MPI.COMM_WORLD.Bcast(h2_slice, root=0)
                    np.copyto(h2[i, j, :, :], h2_slice)

            transformed_ket, en = converge_dmrg_fun(
                h1, h2, mol_ket.nelec, "MPS_{}_{}".format(tags[-1], tags[i])
            )
        else:
            transformed_ket = ket

        ovlp = (
            np.array(
                mps_solver.expectation(
                    bra, mps_solver.get_identity_mpo(), transformed_ket
                )
            )
            / n_ranks
        )
        o_RDM = np.array(mps_solver.get_1pdm(transformed_ket, bra=bra))
        t_RDM = np.array(
            np.transpose(mps_solver.get_2pdm(transformed_ket, bra=bra), (0, 3, 1, 2))
        )

        rdm1, rdm2 = transform_integrals(o_RDM, t_RDM, computational_to_OAO_bra)

        overlap_new[-1, i] = ovlp
        one_rdm_new[-1, i, :, :] = rdm1
        two_rdm_new[-1, i, :, :, :, :] = rdm2

        if enforce_symmetric:
            overlap_new[i, -1] = ovlp.conj()
            one_rdm_new[i, -1, :, :] = rdm1.conj()
            two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()

    if not enforce_symmetric:
        for i, mol_ket in enumerate(mols[:-1]):
            ket = mps_solver.load_mps("MPS_{}".format(tags[i]))
            computational_basis_ket = np.load("basis_{}.npy".format(tags[i]))
            ovlp_ket = mol_ket.intor_symmetric("int1e_ovlp")
            oao_basis_ket = get_basis(mol_ket, "OAO")

            # Transform ket into computational basis of bra
            computational_to_OAO_ket = oao_basis_ket.T.dot(ovlp_ket).dot(
                computational_basis_ket
            )
            computational_to_OAO_bra = oao_basis_bra.T.dot(ovlp_bra).dot(basis)
            orbital_rotation = (
                computational_to_OAO_bra.T.dot(computational_to_OAO_ket)
            ).T

            if i != len(mols) - 1:
                h1, h2 = get_integrals(mol_bra, (basis.dot(orbital_rotation.T)))

                MPI.COMM_WORLD.Bcast(h1, root=0)

                h2_slice = np.empty((h2.shape[2], h2.shape[3]))

                for i in range(h2.shape[0]):
                    for j in range(h2.shape[1]):
                        np.copyto(h2_slice, h2[i, j, :, :])
                        MPI.COMM_WORLD.Bcast(h2_slice, root=0)
                        np.copyto(h2[i, j, :, :], h2_slice)

                transformed_bra, en = converge_dmrg_fun(
                    h1, h2, mol_ket.nelec, "MPS_{}_{}".format(tags[i], tags[-1])
                )
            else:
                transformed_bra = bra

            ovlp = (
                np.array(
                    mps_solver.expectation(
                        transformed_bra, mps_solver.get_identity_mpo(), ket
                    )
                )
                / n_ranks
            )
            o_RDM = np.array(mps_solver.get_1pdm(ket, bra=transformed_bra))
            t_RDM = np.array(
                np.transpose(
                    mps_solver.get_2pdm(ket, bra=transformed_bra), (0, 3, 1, 2)
                )
            )

            rdm1, rdm2 = transform_integrals(o_RDM, t_RDM, computational_to_OAO_ket)

            overlap_new[i, -1] = ovlp.conj()
            one_rdm_new[i, -1, :, :] = rdm1.conj()
            two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()

    return overlap_new, one_rdm_new, two_rdm_new


def append_to_rdms_orbital_rotation(
    mols,
    tags,
    overlap=None,
    one_rdm=None,
    two_rdm=None,
    computational_basis="split",
    reorder_orbitals=True,
    converge_dmrg_fun=converge_dmrg,
    rotation_thresh=1.0e-6,
):
    mol_bra = mols[-1]

    basis = get_basis(mol_bra, basis_type=computational_basis)

    h1, h2 = get_integrals(mol_bra, basis)

    MPI.COMM_WORLD.Bcast(h1, root=0)

    h2_slice = np.empty((h2.shape[2], h2.shape[3]))

    for i in range(h2.shape[0]):
        for j in range(h2.shape[1]):
            np.copyto(h2_slice, h2[i, j, :, :])
            MPI.COMM_WORLD.Bcast(h2_slice, root=0)
            np.copyto(h2[i, j, :, :], h2_slice)

    norb = h1.shape[0]
    nelec = np.sum(mol_bra.nelec)

    if rank == 0:
        mps_solver = DMRGDriver(
            symm_type=SymmetryTypes.SU2, mpi=None, stack_mem=4 << 30
        )
        mps_solver.initialize_system(norb, n_elec=nelec, spin=mol_bra.spin)

        if reorder_orbitals:
            orbital_reordering = mps_solver.orbital_reordering(h1, h2)

            basis = basis[:, orbital_reordering]

    MPI.COMM_WORLD.Bcast(basis, root=0)

    h1, h2 = get_integrals(mol_bra, basis)

    MPI.COMM_WORLD.Bcast(h1, root=0)

    h2_slice = np.empty((h2.shape[2], h2.shape[3]))

    for i in range(h2.shape[0]):
        for j in range(h2.shape[1]):
            np.copyto(h2_slice, h2[i, j, :, :])
            MPI.COMM_WORLD.Bcast(h2_slice, root=0)
            np.copyto(h2[i, j, :, :], h2_slice)

    overlap_new = np.ones((len(mols), len(mols)))
    if overlap is not None:
        overlap_new[:-1, :-1] = overlap
    one_rdm_new = np.ones((len(mols), len(mols), norb, norb))
    if one_rdm is not None:
        one_rdm_new[:-1, :-1, :, :] = one_rdm
    two_rdm_new = np.ones((len(mols), len(mols), norb, norb, norb, norb))
    if two_rdm is not None:
        two_rdm_new[:-1, :-1, :, :, :, :] = two_rdm

    converge_dmrg_fun(h1, h2, mol_bra.nelec, "MPS_{}".format(tags[-1]))

    if rank == 0:
        mps_solver = DMRGDriver(
            symm_type=SymmetryTypes.SU2, mpi=None, stack_mem=4 << 30
        )
        mps_solver.initialize_system(norb, n_elec=nelec, spin=mol_bra.spin)

        bra = mps_solver.load_mps("MPS_{}".format(tags[-1]))

        np.save("basis_{}.npy".format(tags[-1]), basis)

        ovlp_bra = mol_bra.intor_symmetric("int1e_ovlp")
        oao_basis_bra = get_basis(mol_bra, "OAO")

        for i, mol_ket in enumerate(mols):
            ket = mps_solver.load_mps("MPS_{}".format(tags[i]))
            computational_basis_ket = np.load("basis_{}.npy".format(tags[i]))
            ovlp_ket = mol_ket.intor_symmetric("int1e_ovlp")
            oao_basis_ket = get_basis(mol_ket, "OAO")

            # Transform ket into computational basis of bra
            computational_to_OAO_ket = oao_basis_ket.T.dot(ovlp_ket).dot(
                computational_basis_ket
            )
            computational_to_OAO_bra = oao_basis_bra.T.dot(ovlp_bra).dot(basis)
            orbital_rotation = (
                computational_to_OAO_bra.T.dot(computational_to_OAO_ket)
            ).T

            init_bond_dim = max(bra.info.bond_dim, ket.info.bond_dim)

            if i != len(mols) - 1:
                transformed_ket = converge_orbital_rotation_mps(
                    ket,
                    orbital_rotation,
                    tag="MPS_{}_{}".format(tags[-1], tags[i]),
                    convergence_thresh=rotation_thresh,
                    init_bond_dim=init_bond_dim,
                    iprint=0,
                )
            else:
                transformed_ket = ket

            ovlp = np.array(
                mps_solver.expectation(
                    bra, mps_solver.get_identity_mpo(), transformed_ket
                )
            )
            o_RDM = np.array(mps_solver.get_1pdm(transformed_ket, bra=bra))
            t_RDM = np.array(
                np.transpose(
                    mps_solver.get_2pdm(transformed_ket, bra=bra), (0, 3, 1, 2)
                )
            )

            rdm1, rdm2 = transform_integrals(o_RDM, t_RDM, computational_to_OAO_bra)

            overlap_new[-1, i] = ovlp
            overlap_new[i, -1] = ovlp.conj()
            one_rdm_new[-1, i, :, :] = rdm1
            one_rdm_new[i, -1, :, :] = rdm1.conj()
            two_rdm_new[-1, i, :, :, :, :] = rdm2
            two_rdm_new[i, -1, :, :, :, :] = rdm2.conj()

    MPI.COMM_WORLD.Bcast(overlap_new)
    MPI.COMM_WORLD.Bcast(one_rdm_new)

    two_rdm_slice = np.empty(
        (
            two_rdm_new.shape[0],
            two_rdm_new.shape[1],
            two_rdm_new.shape[4],
            two_rdm_new.shape[5],
        )
    )

    for i in range(two_rdm_new.shape[2]):
        for j in range(two_rdm_new.shape[3]):
            np.copyto(two_rdm_slice, two_rdm_new[:, :, i, j, :, :])
            MPI.COMM_WORLD.Bcast(two_rdm_slice, root=0)
            np.copyto(two_rdm_new[:, :, i, j, :, :], two_rdm_slice)

    return overlap_new, one_rdm_new, two_rdm_new


def append_to_rdms_OAO_basis(
    mols,
    tags,
    overlap=None,
    one_rdm=None,
    two_rdm=None,
    converge_dmrg_fun=converge_dmrg,
):
    mol_bra = mols[-1]

    h1, h2 = get_integrals(mol_bra, get_basis(mol_bra, basis_type="OAO"))

    MPI.COMM_WORLD.Bcast(h1, root=0)

    h2_slice = np.empty((h2.shape[2], h2.shape[3]))

    for i in range(h2.shape[0]):
        for j in range(h2.shape[1]):
            np.copyto(h2_slice, h2[i, j, :, :])
            MPI.COMM_WORLD.Bcast(h2_slice, root=0)
            np.copyto(h2[i, j, :, :], h2_slice)

    norb = h1.shape[0]
    nelec = np.sum(mol_bra.nelec)

    mps_solver = DMRGDriver(
        symm_type=SymmetryTypes.SU2,
        mpi=(MPI.COMM_WORLD.size > 1),
        stack_mem=10 << 30,
    )
    mps_solver.initialize_system(norb, n_elec=nelec, spin=mol_bra.spin)

    converge_dmrg_fun(h1, h2, mol_bra.nelec, "MPS_{}".format(tags[-1]))

    bra = mps_solver.load_mps("MPS_{}".format(tags[-1]))

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
        ket = mps_solver.load_mps("MPS_{}".format(tags[i]))

        ovlp = (
            np.array(mps_solver.expectation(bra, mps_solver.get_identity_mpo(), ket))
            / n_ranks
        )
        o_RDM = np.array(mps_solver.get_1pdm(ket, bra=bra))
        t_RDM = np.array(np.transpose(mps_solver.get_2pdm(ket, bra=bra), (0, 3, 1, 2)))

        overlap_new[-1, i] = ovlp
        overlap_new[i, -1] = ovlp.conj()
        one_rdm_new[-1, i, :, :] = o_RDM
        one_rdm_new[i, -1, :, :] = o_RDM.conj()
        two_rdm_new[-1, i, :, :, :, :] = t_RDM
        two_rdm_new[i, -1, :, :, :, :] = t_RDM.conj()

    return overlap_new, one_rdm_new, two_rdm_new


class DMRG_EVCont_obj:
    def __init__(
        self,
        dmrg_converge_fun=converge_dmrg,
        append_method=append_to_rdms_OAO_basis,
    ):
        self.solver = dmrg_converge_fun
        self.append_method = append_method

        self.mols = []
        self.tags = []
        self.max_tag = 0
        self.overlap = None
        self.one_rdm = None
        self.two_rdm = None

    def append_to_rdms(self, mol):
        self.mols.append(mol)
        self.tags.append(self.max_tag)
        self.max_tag += 1
        self.overlap, self.one_rdm, self.two_rdm = self.append_method(
            self.mols,
            self.tags,
            overlap=self.overlap,
            one_rdm=self.one_rdm,
            two_rdm=self.two_rdm,
            converge_dmrg_fun=self.solver,
        )

    def prune_datapoints(self, keep_ids):
        if self.overlap is not None:
            self.overlap = self.overlap[np.ix_(keep_ids, keep_ids)]
        if self.one_rdm is not None:
            self.one_rdm = self.one_rdm[np.ix_(keep_ids, keep_ids)]
        if self.two_rdm is not None:
            self.two_rdm = self.two_rdm[np.ix_(keep_ids, keep_ids)]
        self.mols = [self.mols[i] for i in keep_ids]
        self.tags = [self.tags[i] for i in keep_ids]
