import numpy as np

from pyscf import gto, md, fci, mcscf, scf

import numpy as np

from EVCont.electron_integral_utils import get_basis, get_integrals

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad


ncas = 8
neleca = 4


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", geometry[0]), ("H", geometry[1]), ("O", geometry[2])],
        basis="cc-pvtz",
        symmetry=True,
        unit="Bohr",
    )

    return mol


overlap = np.load("overlap_vtz.npy")
one_rdm = np.load("one_rdm_vtz.npy")
two_rdm = np.load("two_rdm_vtz.npy")

num_points = overlap.shape[0]

trajectory = np.load("traj_EVCont_{}_vtz.npy".format(num_points - 1))


with open("force_error_vtz.txt", "w") as fl:
    for i in range(num_points):
        fl.write("Force error ({} trn. geometries)  ".format(i + 1))
    fl.write("\n")
with open("predicted_energies_vtz.txt", "w") as fl:
    fl.write("Energy exact")
    for i in range(num_points):
        fl.write("  Energy predicted ({} trn. geometries)".format(i + 1))
    fl.write("\n")


for i, pos in enumerate(trajectory):
    print(i)
    mol = get_mol(pos)
    nuc_en = mol.energy_nuc()

    mf = scf.hf_symm.RHF(mol)
    en_casci, grad_casci = (
        mcscf.CASCI(mf, ncas, neleca).nuc_grad_method().as_scanner()(mol)
    )

    with open("predicted_energies_vtz.txt", "a") as fl:
        fl.write("{}".format(en_casci))
        with open("force_error_vtz.txt", "a") as fl2:
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
                fl2.write("{}  ".format(np.sum(abs(grad - grad_casci) ** 2)))
            fl.write("\n")
            fl2.write("\n")
