import numpy as np

from scipy.linalg import eigh

from pyscf import scf, gto, ao2mo, fci, lo, cc

import sys

from EVCont.ab_initio_eigenvector_continuation import approximate_ground_state
from EVCont.ab_initio_gradients_loewdin import get_energy_with_grad


box_edge = float(sys.argv[1])

n_data_points = 1000
seed = 1

norb = nelec = 10


def get_ham(positions):
    mol = gto.Mole()

    mol.build(
        atom = [('H', pos) for pos in positions],
        basis = 'sto-6g',
        symmetry = True,
        unit="Bohr"
    )

    return mol

rng = np.random.default_rng(seed)

equilibrium_dist = 1.78596

equilibrium_pos = np.array([(x*equilibrium_dist, 0., 0.) for x in range(10)])

S = np.load("/home/yannic/uni/mountpoints/create/data/AC/H10/H10_box_displacement_r_0.025_local/S.npy")
one_RDM = np.load("/home/yannic/uni/mountpoints/create/data/AC/H10/H10_box_displacement_r_0.025_local/one_RDM.npy")
two_RDM = np.load("/home/yannic/uni/mountpoints/create/data/AC/H10/H10_box_displacement_r_0.025_local/two_RDM.npy")


open("H10_en_predicted_{}.txt".format(box_edge), "w").close()

grads = []
for i in range(n_data_points):
    # Sample position
    shifts = (rng.random(size=(10, 3)) - 0.5) * 2 * box_edge
    sampled_pos = equilibrium_pos + shifts
    mol = get_ham(sampled_pos)
    en_approx, grad = get_energy_with_grad(mol, one_RDM, two_RDM, S)

    with open("H10_en_predicted_{}.txt".format(box_edge), "a") as fl:
        fl.write("{}\n".format(en_approx))
    grads.append(grad)


np.save("H10_grad_predicted_{}.npy".format(box_edge), np.array(grads))
