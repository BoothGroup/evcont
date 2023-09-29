import numpy as np

from pyscf import gto, mcscf

from evcont.electron_integral_utils import get_basis, get_integrals

from evcont.ab_initio_gradients_loewdin import get_energy_with_grad

"""
Evaluates the energy and force along the converged trajectory in the cc-pVDZ basis
(for continuation and CASCI).
"""


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", geometry[0]), ("H", geometry[1]), ("O", geometry[2])],
        basis="cc-pVDZ",
        symmetry=False,
        unit="Bohr",
    )

    return mol


ncas = 8
neleca = 4

overlap = np.load("overlap.npy")
one_rdm = np.load("one_rdm.npy")
two_rdm = np.load("two_rdm.npy")

num_points = overlap.shape[0]

# Load trajectory
trajectory = np.load("traj_EVCont_{}.npy".format(overlap.shape[0] - 1))


with open("force_error_vdz_active.txt", "w") as fl:
    for i in range(num_points):
        fl.write("Force error ({} trn. geometries)  ".format(i + 1))
    fl.write("\n")
with open("predicted_energies_vdz_active.txt", "w") as fl:
    fl.write("Energy CAS")
    for i in range(num_points):
        fl.write("  Energy predicted ({} trn. geometries)".format(i + 1))
    fl.write("\n")


for i, pos in enumerate(trajectory):
    print(i)
    mol = get_mol(pos)
    nuc_en = mol.energy_nuc()

    mf = mol.RHF()
    en_exact, grad_exact = (
        mcscf.CASCI(mf, ncas, neleca).nuc_grad_method().as_scanner()(mol)
    )

    with open("predicted_energies_vdz_active.txt", "a") as fl:
        fl.write("{}".format(en_exact))
        with open("force_error_vdz_active.txt", "a") as fl2:
            h1, h2 = get_integrals(mol, get_basis(mol))
            for j in range(num_points):
                en, grad = get_energy_with_grad(
                    mol,
                    one_rdm[: j + 1, : j + 1],
                    two_rdm[: j + 1, : j + 1],
                    overlap[: j + 1, : j + 1],
                    hermitian=True,
                )
                fl.write("  {}".format(en))
                fl2.write("{}  ".format(np.sum(abs(grad - grad_exact) ** 2)))
            fl.write("\n")
            fl2.write("\n")
