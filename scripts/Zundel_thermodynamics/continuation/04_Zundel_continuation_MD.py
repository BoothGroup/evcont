from pyscf import gto, md

from pyscf.scf import hf

import numpy as np

from pyscf.data import nist


from evcont.MD_utils import get_scanner


from evcont.ab_initio_eigenvector_continuation import (
    get_basis,
)


import sys

"""
Single MD trajectory simulation with the continuation framework, this also calculates
the predicted dipole moment and Mulliken charges. Initial velocity is drawn from
Maxwell-Boltzmann distribution at room temperature according to specified seed. Assumes that
the initial geometry is specified in the file "../../init_geometry.npy". Script assumes that training
configurations are specified in "../trn_geometries.txt" as produced by the script
"01_Zundel_continuation_trn_set_generation.py", that MPSs have been constructed by
the script "02_Zundel_continuation_run_DMRG.py", and that t-RDMs/overlaps have been constructed with
the script "03_Zundel_continuation_evaluate_MPS_t_RDMs.py".
"""

seed = int(sys.argv[1])


def get_mol(geometry):
    mol = gto.Mole()

    mol.build(
        atom=[
            ("O", geometry[0]),
            ("H", geometry[1]),
            ("H", geometry[2]),
            ("H", geometry[3]),
            ("O", geometry[4]),
            ("H", geometry[5]),
            ("H", geometry[6]),
        ],
        basis="6-31G",
        symmetry=False,
        unit="Angstrom",
        charge=1,
    )

    return mol


init_geometry = np.load("../../init_geometry.npy")

rng = np.random.default_rng(seed)

mol = get_mol(init_geometry)


init_mol = mol.copy()


steps = 10000
dt = 25


def dip_moment(mol, dm, unit="Debye"):
    if not (isinstance(dm, np.ndarray) and dm.ndim == 2):
        # UHF density matrices
        dm = dm[0] + dm[1]

    center_of_mass = np.sum(
        mol.atom_mass_list() * mol.atom_coords().T, axis=-1
    ) / np.sum(mol.atom_mass_list())

    with mol.with_common_orig(center_of_mass):
        ao_dip = mol.intor_symmetric("int1e_r", comp=3)
    el_dip = np.einsum("xij,ji->x", ao_dip, dm).real

    charges = mol.atom_charges()
    coords = mol.atom_coords() - center_of_mass
    nucl_dip = np.einsum("i,ix->x", charges, coords)
    mol_dip = nucl_dip - el_dip

    if unit.upper() == "DEBYE":
        mol_dip *= nist.AU2DEBYE

    return mol_dip


with open("../trn_geometries.txt", "r") as fl:
    lines = fl.readlines()
    no_trn_points = len(lines)

inds_trn = np.tril_indices(no_trn_points)

overlap = np.zeros((no_trn_points, no_trn_points))

for i in range(len(inds_trn[0])):
    overlap[inds_trn[0][i], inds_trn[1][i]] = np.load(
        "../MPS_cross_{}_{}/ovlp.npy".format(inds_trn[0][i], inds_trn[1][i])
    )

overlap[np.triu_indices(no_trn_points)] = (overlap.T)[np.triu_indices(no_trn_points)]

one_rdm = np.zeros((no_trn_points, no_trn_points, mol.nao, mol.nao))

for i in range(len(inds_trn[0])):
    one_rdm[inds_trn[0][i], inds_trn[1][i], :, :] = np.load(
        "../MPS_cross_{}_{}/one_rdm.npy".format(inds_trn[0][i], inds_trn[1][i])
    )

one_rdm[np.triu_indices(no_trn_points)[0], np.triu_indices(no_trn_points)[1], :, :] = (
    np.transpose(one_rdm, [1, 0, 2, 3])
)[np.triu_indices(no_trn_points)[0], np.triu_indices(no_trn_points)[1], :, :]

inds_orb = np.tril_indices(mol.nao * mol.nao)

two_rdm = np.zeros((len(inds_trn[0]), len(inds_orb[0])))

for i in range(len(inds_trn[0])):
    two_rdm[i, :] = np.load(
        "../MPS_cross_{}_{}/two_rdm.npy".format(inds_trn[0][i], inds_trn[1][i])
    )


scanner_fun = get_scanner(init_mol, one_rdm, two_rdm, overlap, hermitian=True)


open("dipole_moment_continuation.txt", "w").close()
open("atom_charges_continuation.txt", "w").close()

veloc = md.distributions.MaxwellBoltzmannVelocity(mol, T=298.15, rng=rng)


def callback(locals):
    mol = locals["mol"]

    basis = get_basis(mol)

    predicted_one_rdm = locals["scanner"].base.predicted_one_rdm

    predicted_one_rdm_ao = basis.dot(predicted_one_rdm).dot(basis.T)

    dipole_moment = dip_moment(mol, predicted_one_rdm_ao)
    atomic_charges = hf.mulliken_meta(mol, predicted_one_rdm_ao)[1]

    with open("dipole_moment_continuation.txt", "a") as fl:
        for el in dipole_moment:
            fl.write("{}  ".format(el))
        fl.write("\n")
    with open("atom_charges_continuation.txt", "a") as fl:
        for el in atomic_charges:
            fl.write("{}  ".format(el))
        fl.write("\n")


frames = []
scanner_fun.mol = init_mol.copy()
myintegrator = md.integrators.NVTBerendson(
    scanner_fun,
    298.15,
    taut=250,
    steps=steps,
    dt=dt,
    incore_anyway=True,
    frames=frames,
    veloc=veloc,
    trajectory_output="trajectory.xyz",
    data_output="energy.xyz",
    callback=callback,
)
myintegrator.run()


np.save("trajectory.npy", np.array([frame.coord for frame in frames]))
