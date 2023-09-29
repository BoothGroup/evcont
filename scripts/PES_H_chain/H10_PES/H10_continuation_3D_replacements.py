import numpy as np

from pyscf import gto

import sys

from evcont.ab_initio_gradients_loewdin import get_energy_with_grad

from evcont.FCI_EVCont import FCI_EVCont_obj

from pyscf.mcscf import CASCI

from dscribe.descriptors import SOAP
from ase import Atoms

from dscribe.kernels import AverageKernel

"""
Script to generate the data from Fig. (2), generating the EVCont, HF and GAP predictions
of PES and force field for distorted 10-atom Hydrogen chains based on symmetrically
stretched training configurations.
"""

# Radius by which each atom is displaced w.r.t. the equilibrium config
radius = float(sys.argv[1])

n_data_points = 1000
seed = 1

norb = nelec = 10


def get_mol(positions):
    mol = gto.Mole()

    mol.build(
        atom=[("H", pos) for pos in positions],
        basis="sto-6g",
        symmetry=False,
        unit="Bohr",
    )

    return mol


# Computes the gradient and the energy of the GAP prediction
def get_GAP_energy_and_grad(w, trn_features, soap_object, atoms):
    # Compute new SOAP features + derivatives
    new_feat_der, new_feat = soap_object.derivatives(atoms, attach=True)

    # non-normalized kernel
    kern_nn = np.mean(np.einsum("ijk,lk->ijl", trn_features, new_feat), axis=(1, 2))

    kern_nn_derivative = np.mean(
        np.einsum("ijk,labk->abijl", trn_features, new_feat_der), axis=(3, 4)
    )

    norm_a = np.mean(np.einsum("ij,kj->ik", new_feat, new_feat))
    norm_a_derivative = np.mean(
        np.einsum("ij,kabj->abik", new_feat, new_feat_der), axis=(2, 3)
    )
    norm_a_derivative += np.mean(
        np.einsum("iabj,kj->abik", new_feat_der, new_feat), axis=(2, 3)
    )

    norm_b = np.mean(np.einsum("ijk,imk->ijm", trn_features, trn_features), axis=(1, 2))

    kern = kern_nn / np.sqrt(norm_a * norm_b)

    kern_derivative = kern_nn_derivative / np.sqrt(
        norm_a * norm_b
    ) - 0.5 * kern_nn * np.expand_dims(norm_a_derivative, axis=-1) * norm_b / (
        np.sqrt(norm_a * norm_b) ** 3
    )

    en_predicted = weights.dot(np.squeeze(kern))
    grad_predicted = np.einsum("jki,i->jk", kern_derivative, weights)

    return en_predicted, grad_predicted


rng = np.random.default_rng(seed)

equilibrium_dist = 1.78596

equilibrium_pos = np.array([(x * equilibrium_dist, 0.0, 0.0) for x in range(10)])

training_stretches = np.array([0.0, 0.5, -0.5, 1.0, -1.0])

trainig_dists = equilibrium_dist + training_stretches

continuation_object = FCI_EVCont_obj()

# Prepare SOAP
soap = SOAP(species=["H"], periodic=False, r_cut=10.0, n_max=10, l_max=20)
soap_features_trn = []

# Generate training data + prepare training models
for i, dist in enumerate(trainig_dists):
    positions = [(x, 0.0, 0.0) for x in dist * np.arange(10)]
    mol = get_mol(positions)
    continuation_object.append_to_rdms(mol)

    # Get SOAP features
    h_chain = Atoms("H" * 10, positions)
    soap_features_trn.append(soap.create(h_chain))

# Train GAP model
avg_kernel = AverageKernel(metric="linear", normalize_kernel=True)

kernel_mat = avg_kernel.create(soap_features_trn)
soap_features_trn = np.array(soap_features_trn)

# Fitting the GP, we set some rather randomly chosen small noise
noise_term = np.eye(kernel_mat.shape[0]) * 1.0e-15
# We only need the weights later on so calling lstsq is enough
weights = np.linalg.lstsq(kernel_mat + noise_term, continuation_object.ens)[0]


with open("H10_predicted_energies_{}.txt".format(radius), "w") as fl:
    ("FCI  HF  Continuation  GAP\n")
open("H10_continuation_gradients_{}.txt".format(radius), "w").close()
open("H10_FCI_gradients_{}.txt".format(radius), "w").close()
open("H10_HF_gradients_{}.txt".format(radius), "w").close()
open("H10_GAP_gradients_{}.txt".format(radius), "w").close()

for i in range(n_data_points):
    # Sample a new test geometry
    displacement_theta = rng.random(size=(10)) * np.pi
    displacement_phi = rng.random(size=(10)) * 2 * np.pi

    sampled_displacement_x = (
        radius * np.sin(displacement_theta) * np.cos(displacement_phi)
    )
    sampled_displacement_y = (
        radius * np.sin(displacement_theta) * np.sin(displacement_phi)
    )
    sampled_displacement_z = radius * np.cos(displacement_theta)

    sampled_displacement = np.stack(
        (sampled_displacement_x, sampled_displacement_y, sampled_displacement_z),
        axis=-1,
    )

    sampled_pos = equilibrium_pos + sampled_displacement
    mol = get_mol(sampled_pos)

    # Predictions from continuation
    en_continuation, grad_continuation = get_energy_with_grad(
        mol,
        continuation_object.one_rdm,
        continuation_object.two_rdm,
        continuation_object.overlap,
    )

    # Predictions from Hartree-Fock
    hf_energy, hf_grad = mol.RHF().nuc_grad_method().as_scanner()(mol)

    # Fci reference values
    en_exact, grad_exact = CASCI(mol.RHF(), 10, 10).nuc_grad_method().as_scanner()(mol)

    # Gap prediction
    h_chain = Atoms("H" * 10, [atom_pos for atom_pos in sampled_pos])
    en_GAP, grad_GAP = get_GAP_energy_and_grad(
        weights, soap_features_trn, soap, h_chain
    )

    with open("H10_predicted_energies_{}.txt".format(radius), "a") as fl:
        fl.write(
            "{}  {}  {}  {}\n".format(en_exact, hf_energy, en_continuation, en_GAP)
        )

    with open("H10_FCI_gradients_{}.txt".format(radius), "a") as fl:
        np.savetxt(fl, grad_exact.flatten(), newline=" ")
        fl.write("\n")

    with open("H10_HF_gradients_{}.txt".format(radius), "a") as fl:
        np.savetxt(fl, hf_grad.flatten(), newline=" ")
        fl.write("\n")

    with open("H10_continuation_gradients_{}.txt".format(radius), "a") as fl:
        np.savetxt(fl, grad_continuation.flatten(), newline=" ")
        fl.write("\n")

    with open("H10_GAP_gradients_{}.txt".format(radius), "a") as fl:
        np.savetxt(fl, grad_GAP.flatten(), newline=" ")
        fl.write("\n")
