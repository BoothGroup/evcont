from pyscf import gto, md, fci, scf, mcscf, ao2mo

import numpy as np

from pyscf.mcscf.casci import CASCI


ncas = 8
neleca = 4

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


def get_CASCI_surface(mfs):
    ens = []
    for mf in mfs:
        tmp = CASCI(mf, ncas, neleca)
        tmp.canonicalization = False
        e = tmp.kernel()[0]
        ens.append(e)
    return np.array(ens)


dist_list = np.linspace(1.0, 3.6, 27)

test_mfs = []
for d in dist_list:
    mol = get_mol(d)
    mf = scf.HF(mol)
    mf.kernel()
    test_mfs.append(mf)

np.savetxt("CASCI_surface.txt", get_CASCI_surface(test_mfs))
