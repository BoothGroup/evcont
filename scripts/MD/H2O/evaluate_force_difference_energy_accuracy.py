import numpy as np

from pyscf import gto, md, fci, scf, mcscf

import numpy as np

from EVCont.electron_integral_utils import get_basis, get_integrals

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad


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


with open("force_error_6_31_G.txt", "w") as fl:
    for i in range(num_points):
        fl.write("Force error ({} trn. geometries)  ".format(i + 1))
    fl.write("\n")
with open("predicted_energies_6_31_G.txt", "w") as fl:
    fl.write("Energy exact")
    for i in range(num_points):
        fl.write("  Energy predicted ({} trn. geometries)".format(i + 1))
    fl.write("\n")


for i, pos in enumerate(trajectory):
    print(i)
    mol = get_mol(pos)
    nuc_en = mol.energy_nuc()

    mf = scf.hf_symm.RHF(mol)
    en_exact, grad_exact = (
        mcscf.CASCI(mf, mol.nao, np.sum(mol.nelec)).nuc_grad_method().as_scanner()(mol)
    )

    with open("predicted_energies_6_31_G.txt", "a") as fl:
        fl.write("{}".format(en_exact))
        with open("force_error_6_31_G.txt", "a") as fl2:
            h1, h2 = get_integrals(mol, get_basis(mol))
            for j in range(num_points):
                en, grad = get_energy_with_grad(
                    mol,
                    (one_rdm[: j + 1, : j + 1, :, :]),
                    (two_rdm[: j + 1, : j + 1, :, :, :, :]),
                    (overlap[: j + 1, : j + 1]),
                    hermitian=True,
                )
                fl.write("  {}".format(en))
                fl2.write("{}  ".format(np.sum(abs(grad - grad_exact) ** 2)))
            fl.write("\n")
            fl2.write("\n")
