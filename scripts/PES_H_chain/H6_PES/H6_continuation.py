from pyscf import gto, fci


from evcont.electron_integral_utils import get_basis, get_integrals

from evcont.FCI_EVCont import FCI_EVCont_obj

from evcont.ab_initio_eigenvector_continuation import approximate_ground_state_OAO

import numpy as np

"""
Prediction of the PES for a 6-atom H chain from different training points as depicted in
the schematic (Fig. (1) of the manuscript)
"""

n_atoms = 6

continuation_object = FCI_EVCont_obj()


def get_mol(dist):
    mol = gto.Mole()

    mol.build(
        atom=[
            ("H", (x, 0.0, 0.0))
            for x in (np.arange(n_atoms) - np.median(np.arange(n_atoms))) * dist
        ],
        basis="sto-6g",
        symmetry=False,
        unit="Bohr",
    )

    return mol


for i, trn_dist in enumerate([1.0, 1.8, 2.6]):
    mol = get_mol(trn_dist)

    continuation_object.append_to_rdms(mol)

    np.savetxt(
        "GS_dist_{}.txt".format(trn_dist), continuation_object.fcivecs[-1].flatten()
    )

    np.savetxt(
        "en_dist_{}.txt".format(trn_dist), np.atleast_1d(continuation_object.ens[-1])
    )

    open("predicted_surface_{}_datapoints.txt".format(i + 1), "w").close()
    for test_dist in np.linspace(0.8, 3.0):
        mol = get_mol(test_dist)
        en, _ = approximate_ground_state_OAO(
            mol,
            continuation_object.one_rdm,
            continuation_object.two_rdm,
            continuation_object.overlap,
        )
        with open("predicted_surface_{}_datapoints.txt".format(i + 1), "a") as fl:
            fl.write("{}  {}\n".format(test_dist, en))

    test_dist = 2.2
    mol = get_mol(test_dist)
    _, c = approximate_ground_state_OAO(
        mol,
        continuation_object.one_rdm,
        continuation_object.two_rdm,
        continuation_object.overlap,
    )
    np.savetxt("continuation_gs_{}_datapoints.txt".format(i + 1), c)


open("exact_surface.txt", "w").close()
open("HF_surface.txt", "w").close()
for i, test_dist in enumerate(np.linspace(0.8, 3.0)):
    mol = get_mol(test_dist)
    h1, h2 = get_integrals(mol, get_basis(mol))
    e, fcivec = fci.direct_spin0.FCI().kernel(h1, h2, mol.nao, mol.nelec)

    with open("exact_surface.txt", "a") as fl:
        fl.write("{}  {}\n".format(test_dist, e + mol.energy_nuc()))

    mf = mol.RHF()
    mf.kernel()

    with open("HF_surface.txt", "a") as fl:
        fl.write("{}  {}\n".format(test_dist, e + mol.energy_nuc()))
