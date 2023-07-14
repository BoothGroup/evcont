from pyscf import gto, md, fci, scf, mcscf, ao2mo

import numpy as np

# from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad
from EVCont.electron_integral_utils import get_basis, get_integrals, transform_integrals


from EVCont.ab_initio_eigenvector_continuation import approximate_ground_state

from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad

from EVCont.EVContCI import EVContCI

from EVCont.customCASCI import CustomCASCI

from EVCont.CASCI_EVCont import append_to_rdms, append_to_rdms_complete_space

from pyscf.mcscf.casci import CASCI


ncas = 8
neleca = 4

basis = "cc-pVdZ"


def get_mol(dist):
    mol = gto.Mole()

    mol.build(
        atom=[("H", (x, 0.0, 0.0)) for x in dist * np.arange(10)],
        basis="cc-pVdZ",
        symmetry=True,
        unit="Bohr",
    )

    return mol


def get_potential_energy_curve(overlap, one_rdm, two_rdm, mfs):
    ens = []
    for mf in mfs:
        mol = mf.mol
        h1, h2 = get_integrals(mol, get_basis(mol))
        e, _ = approximate_ground_state(
            h1, h2, one_rdm, two_rdm, overlap, hermitian=True
        )
        ens.append(e + mol.energy_nuc())
    return np.array(ens)


def get_CASCI_surface(mfs):
    ens = []
    for mf in mfs:
        tmp = CASCI(mf, ncas, neleca)
        tmp.canonicalization = False
        e = tmp.kernel()[0]
        ens.append(e)
    return np.array(ens)


dist_list = [1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.4, 2.8, 3.2, 3.6]

test_mfs = []
for d in dist_list:
    mol = get_mol(d)
    mf = scf.HF(mol)
    mf.kernel()
    test_mfs.append(mf)

CASCI_surface = get_CASCI_surface(test_mfs)

np.savetxt("CASCI_surface.txt", CASCI_surface)


cascis = [CASCI(test_mfs[0], ncas, neleca)]

overlap, one_rdm, two_rdm = append_to_rdms(cascis)

a = get_potential_energy_curve(overlap, one_rdm, two_rdm, test_mfs)

with open("continued_surface.txt", "w") as fl:
    for a_el in a:
        fl.write("{}".format(a_el))
        if a_el != a[-1]:
            fl.write("  ")
        else:
            fl.write("\n")


cascis.append(CASCI(test_mfs[-1], ncas, neleca))

overlap, one_rdm, two_rdm = append_to_rdms(cascis, overlap, one_rdm, two_rdm)

b = get_potential_energy_curve(overlap, one_rdm, two_rdm, test_mfs)

with open("continued_surface.txt", "a") as fl:
    for el in b:
        fl.write("{}".format(el))
        if el != b[-1]:
            fl.write("  ")
        else:
            fl.write("\n")


cascis.append(CASCI(test_mfs[4], ncas, neleca))

overlap, one_rdm, two_rdm = append_to_rdms(cascis, overlap, one_rdm, two_rdm)

c = get_potential_energy_curve(overlap, one_rdm, two_rdm, test_mfs)

with open("continued_surface.txt", "a") as fl:
    for el in c:
        fl.write("{}".format(el))
        if el != c[-1]:
            fl.write("  ")
        else:
            fl.write("\n")


cascis.append(CASCI(test_mfs[7], ncas, neleca))

overlap, one_rdm, two_rdm = append_to_rdms(cascis, overlap, one_rdm, two_rdm)

d = get_potential_energy_curve(overlap, one_rdm, two_rdm, test_mfs)

with open("continued_surface.txt", "a") as fl:
    for el in d:
        fl.write("{}".format(el))
        if el != d[-1]:
            fl.write("  ")
        else:
            fl.write("\n")

cascis.append(CASCI(test_mfs[2], ncas, neleca))

overlap, one_rdm, two_rdm = append_to_rdms(cascis, overlap, one_rdm, two_rdm)

e = get_potential_energy_curve(overlap, one_rdm, two_rdm, test_mfs)

with open("continued_surface.txt", "a") as fl:
    for el in e:
        fl.write("{}".format(el))
        if el != e[-1]:
            fl.write("  ")
        else:
            fl.write("\n")

np.save("overlap.npy", overlap)
np.save("one_rdm.npy", one_rdm)
np.save("two_rdm.npy", two_rdm)
