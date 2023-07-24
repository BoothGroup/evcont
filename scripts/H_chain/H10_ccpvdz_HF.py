from pyscf import gto, md, fci, scf, mcscf, ao2mo

import numpy as np


basis = "cc-pVdZ"


def get_mol(dist):
    mol = gto.Mole()

    mol.build(
        atom=[("H", (x, 0.0, 0.0)) for x in dist * np.arange(10)],
        basis=basis,
        symmetry=True,
        unit="Bohr",
    )

    return mol


open("HF_surface.txt", "w").close()

dist_list = np.linspace(1.0, 3.6, 27)

for d in dist_list:
    mol = get_mol(d)
    mf = scf.HF(mol)
    mf.kernel()

    with open("HF_surface.txt", "a") as fl:
        fl.write("{}\n".format(mf.e_tot))
