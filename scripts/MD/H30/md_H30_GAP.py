from pyscf import md, gto, scf, lib, grad

import numpy as np

from dscribe.descriptors import SOAP
from ase import Atoms

from dscribe.kernels import AverageKernel


from evcont.electron_integral_utils import get_basis, get_integrals


"""
MD simulation with GAP predictions (using the same training data from previous continuation
runs)
"""

norb = nelec = number_atoms = 30

soap = SOAP(species=["H"], periodic=False, r_cut=10.0, n_max=10, l_max=20)


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[("H", pos) for pos in geometry],
        basis="sto-6g",
        symmetry=False,
        unit="Bohr",
    )

    return mol


def energy_grad(geometry, weights, features):
    h_chain = Atoms("H" * number_atoms, [atom_pos for atom_pos in geometry])
    new_feat_der, new_feat = soap.derivatives(h_chain, attach=True)
    kern_nn = np.mean(np.einsum("ijk,lk->ijl", features, new_feat), axis=(1, 2))
    kern_der = np.mean(
        np.einsum("ijk,labk->abijl", features, new_feat_der), axis=(3, 4)
    )

    norm_a = np.mean(np.einsum("ij,kj->ik", new_feat, new_feat))
    norm_a_der = np.mean(
        np.einsum("ij,kabj->abik", new_feat, new_feat_der), axis=(2, 3)
    )
    norm_a_der += np.mean(
        np.einsum("iabj,kj->abik", new_feat_der, new_feat), axis=(2, 3)
    )

    norm_b = np.mean(np.einsum("ijk,imk->ijm", features, features), axis=(1, 2))
    kern = kern_nn / np.sqrt(norm_a * norm_b)
    kern_der = kern_der / np.sqrt(norm_a * norm_b) - 0.5 * kern_nn * np.expand_dims(
        norm_a_der, axis=-1
    ) * norm_b / (np.sqrt(norm_a * norm_b) ** 3)

    GP_en = weights.dot(np.squeeze(kern))

    GP_grad = np.einsum("jki,i->jk", kern_der, weights)

    return GP_en, GP_grad


def get_scanner(mol, weights, features):
    class Base:
        converged = True

    class Scanner(lib.GradScanner):
        def __init__(self):
            self.mol = mol
            self.base = Base()
            # self.converged = True

        def __call__(self, mol):
            positions = np.array([mol.atom[i][1] for i in range(norb)])
            GP_en, GP_grad = energy_grad(positions, weights, features)
            return GP_en + mol.energy_nuc(), GP_grad + grad.RHF(scf.RHF(mol)).grad_nuc()

    return Scanner()


def train_GP(ens, features):
    ens = np.array(ens)
    avg_kernel = AverageKernel(metric="linear", normalize_kernel=True)

    kernel_mat = avg_kernel.create(features)

    features = np.array(features)

    # Fit the GP with a moderately small degree of noise
    noise_term = np.eye(kernel_mat.shape[0]) * 1.0e-8

    # We only need the weights later on so calling lstsq is enough
    weights = np.linalg.lstsq(kernel_mat + noise_term, ens)[0]

    return weights


def get_trajectory(
    init_mol,
    weights,
    features,
    dt=10.0,
    steps=10,
    init_veloc=None,
    trajectory_output=None,
    energy_output=None,
):
    scanner_fun = get_scanner(init_mol, weights, features)
    frames = []
    myintegrator = md.NVE(
        scanner_fun,
        dt=dt,
        steps=steps,
        veloc=init_veloc,
        incore_anyway=True,
        frames=frames,
        trajectory_output=trajectory_output,
        energy_output=energy_output,
    )
    myintegrator.run()
    trajectory = [frame.coord for frame in frames]

    return np.array(trajectory)


init_dist = 1.9

steps = 300
dt = 5


energies = []
features = []


geometry = np.array([[0, 0, init_dist * i] for i in range(nelec)])
init_mol = get_mol(geometry).copy()


one_rdm = np.load("one_rdm.npy")
two_rdm = np.load("two_rdm.npy")
overlap = np.load("overlap.npy")

# Restore training geometries
trn_times = list(np.atleast_1d(np.loadtxt("trn_times.txt").astype(int)))

trajs = [np.load("traj_EVCont_{}.npy".format(i)) for i in range(len(trn_times))]

trn_geometries = [trajs[0][0]] + [
    trajs[k][trn_times[k + 1]] for k in range(len(trajs) - 1)
]

for i in range(one_rdm.shape[0]):
    geometry = trn_geometries[i]
    mol = get_mol(geometry)
    h1, h2 = get_integrals(mol, get_basis(mol))

    en = np.sum(one_rdm[i, i, :, :] * h1) + 0.5 * np.sum(two_rdm[i, i, :, :, :, :] * h2)

    energies.append(en)

    features.append(soap.create(Atoms("H" * number_atoms, geometry)))

    weights = train_GP(np.array(energies), np.array(features))

i = one_rdm.shape[0] - 1
traj = get_trajectory(
    init_mol.copy(),
    weights,
    features,
    steps=steps,
    dt=dt,
    trajectory_output=open("traj_GAP_{}.xyz".format(i), "w"),
    energy_output=open("ens_GAP_{}.xyz".format(i), "w"),
)

np.save("traj_GAP_{}.npy".format(i), traj)
