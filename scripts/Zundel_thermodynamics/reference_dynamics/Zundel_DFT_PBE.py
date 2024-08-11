from pyscf import gto, md, dft

from pyscf.scf import hf

from pyscf.data import nist


import numpy as np

import sys

"""
Single MD trajectory simulation with DFT (PBE exchange correlation function), this also calculates
the predicted dipole moment and Mulliken charges. Initial velocity is drawn from
Maxwell-Boltzmann distribution at room temperature according to specified seed. Assumes that
the initial geometry is specified in the file "../../init_geometry.npy".
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

mf = dft.RKS(init_mol, xc="pbe")

mf.max_cycle = 100

steps = 10000
dt = 25


scanner_fun = mf.nuc_grad_method().as_scanner()

open("dipole_moment_DFT_PBE.txt", "w").close()
open("atom_charges_DFT_PBE.txt", "w").close()

veloc = md.distributions.MaxwellBoltzmannVelocity(mol, T=298.15, rng=rng)


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


def callback(locals):
    mol = locals["mol"]
    dft_obj = locals["scanner"].base

    one_rdm = dft_obj.make_rdm1()
    dipole_moment = dip_moment(mol, one_rdm)
    atomic_charges = hf.mulliken_meta(mol, one_rdm)[1]

    with open("dipole_moment_DFT_PBE.txt", "a") as fl:
        for el in dipole_moment:
            fl.write("{}  ".format(el))
        fl.write("\n")
    with open("atom_charges_DFT_PBE.txt", "a") as fl:
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
