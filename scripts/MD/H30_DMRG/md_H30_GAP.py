from pyscf import md, gto, scf, mcscf, lo, ao2mo, fci, lib, grad

import numpy as np

from dscribe.descriptors import SOAP
from ase import Atoms

from dscribe.kernels import AverageKernel

from EVCont.converge_dmrg import converge_dmrg

from EVCont.electron_integral_utils import get_basis, get_integrals


norb = nelec = number_atoms = 30

soap = SOAP(species=["H"], periodic=False, r_cut=10.0, n_max=10, l_max=30)


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
    avg_kernel = AverageKernel(metric="linear", normalize_kernel=True)

    kernel_mat = avg_kernel.create(features)

    features = np.array(features)

    # Fitting the GP, we set some rather randomly chosen small noise
    noise_term = np.eye(kernel_mat.shape[0]) * 1.0e-15
    # We only need the weights later on so calling lstsq is enough
    weights = np.linalg.lstsq(kernel_mat + noise_term, ens)[0]

    return weights


def get_trajectory(init_mol, weights, features, dt=100.0, steps=10, init_veloc=None):
    scanner_fun = get_scanner(init_mol, weights, features)
    frames = []
    myintegrator = md.NVE(
        scanner_fun,
        dt=dt,
        steps=steps,
        veloc=init_veloc,
        incore_anyway=True,
        frames=frames,
    )
    myintegrator.run()
    trajectory = [frame.coord for frame in frames]

    return np.array(trajectory)


init_dist = 1.9

steps = 500
dt = 2


energies = []
features = []


geometry = np.array([[0, 0, init_dist * i] for i in range(nelec)])
init_mol = get_mol(geometry).copy()


one_rdm = np.load("one_rdm.npy")
two_rdm = np.load("two_rdm.npy")

for i in range(0, 11):
    if i != 0:
        geometry = np.load("trn_geometry_{}.npy".format(i))
    mol = get_mol(geometry)
    h1, h2 = get_integrals(init_mol, get_basis(init_mol))

    en = np.sum(one_rdm[i, i, :, :] * h1 + 0.5 * two_rdm[i, i, :, :, :, :] * h2)

    energies.append(en)
    features.append(soap.create(Atoms("H" * number_atoms, geometry)))

    weights = train_GP(np.array(energies), np.array(features))

    traj = get_trajectory(init_mol.copy(), weights, features, steps=steps, dt=dt)

    np.save("traj_SOAP_{}.npy".format(i), traj)
