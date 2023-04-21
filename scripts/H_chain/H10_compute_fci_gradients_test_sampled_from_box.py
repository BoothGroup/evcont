import numpy as np

from pyscf import scf, gto, ao2mo, fci, lo, mcscf

from scipy.linalg import eig

import sys

box_edge = float(sys.argv[1])

n_data_points = 1000

norb = 10
nelec = 10


def get_mol(pos):
    mol = gto.Mole()

    mol.build(
        atom = [('H', x) for x in pos],
        basis = 'sto-6g',
        symmetry = True,
        unit="Bohr"
    )

    return mol


equilibrium_dist = 1.78596

equilibrium_pos = np.array([(x*equilibrium_dist, 0., 0.) for x in range(10)])

rng = np.random.default_rng(1)

r_list = np.around(np.linspace(0.025, 1., num=40),3)

open("H10_en_{}.txt".format(box_edge), "w").close()

grads = []
for i in range(n_data_points):
    # Sample position
    shifts = (rng.random(size=(10, 3)) - 0.5) * 2 * box_edge
    sampled_pos = equilibrium_pos + shifts
    mol = get_mol(sampled_pos)
    en, grad = mcscf.CASCI(mol, 10, 10).nuc_grad_method().as_scanner()(mol)
    with open("H10_en_{}.txt".format(box_edge), "a") as fl:
        fl.write("{}\n".format(en))
    grads.append(grad)

np.save("H10_grad_{}.npy".format(box_edge), np.array(grads))




