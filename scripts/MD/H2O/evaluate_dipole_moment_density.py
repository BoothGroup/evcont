import numpy as np

from pyscf import gto, md, fci, scf, mcscf

from pyscf.scf import hf

import numpy as np

from EVCont.electron_integral_utils import get_basis, get_integrals

from EVCont.ab_initio_eigenvector_continuation import approximate_ground_state


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", geometry[0]), ("H", geometry[1]), ("O", geometry[2])],
        basis="6-31G",
        symmetry=True,
        unit="Bohr",
    )

    return mol


overlap = np.load("overlap.npy")
one_rdm = np.load("one_rdm.npy")
two_rdm = np.load("two_rdm.npy")

num_points = overlap.shape[0]

trajectory = np.load("traj_EVCont_{}.npy".format(num_points - 1))


for i in range(num_points):
    open("dipole_moment_{}.txt".format(i), "w")
    open("atom_charges_{}.txt".format(i), "w")


for i, pos in enumerate(trajectory):
    print(i)
    mol = get_mol(pos)

    basis = get_basis(mol)
    h1, h2 = get_integrals(mol, basis)
    for j in range(num_points):
        red_one_rdm = one_rdm[: j + 1, : j + 1, :, :]
        en, vec = approximate_ground_state(
            h1,
            h2,
            (red_one_rdm),
            (two_rdm[: j + 1, : j + 1, :, :, :, :]),
            (overlap[: j + 1, : j + 1]),
            hermitian=True,
        )

        predicted_one_rdm = np.sum(
            red_one_rdm * vec.reshape((-1, 1, 1)) * vec.reshape((-1, 1, 1, 1)),
            axis=(0, 1),
        )

        predicted_one_rdm = basis.dot(predicted_one_rdm).dot(basis.T)

        dipole_moment = hf.dip_moment(mol, predicted_one_rdm)
        atomic_charges = hf.mulliken_meta(mol, predicted_one_rdm)[1]

        with open("dipole_moment_{}.txt".format(j), "a") as fl:
            for el in dipole_moment:
                fl.write("{}  ".format(el))
            fl.write("\n")
        with open("atom_charges_{}.txt".format(j), "a") as fl:
            for el in atomic_charges:
                fl.write("{}  ".format(el))
            fl.write("\n")
