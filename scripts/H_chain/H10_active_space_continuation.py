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

from mpi4py import MPI


ncas = 8
neleca = 4

basis = "cc-pVdZ"

rank = MPI.COMM_WORLD.Get_rank()


def get_mol(dist):
    mol = gto.Mole()

    mol.build(
        atom=[("H", (x, 0.0, 0.0)) for x in dist * np.arange(10)],
        basis=basis,
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


dist_list = np.linspace(1.0, 3.6, 27)

if rank == 0:
    test_mfs = []
    for d in dist_list:
        mol = get_mol(d)
        mf = scf.HF(mol)
        mf.kernel()
        test_mfs.append(mf)

    with open("continued_surface.txt", "w") as fl:
        for d in dist_list:
            fl.write("{}".format(d))
            if d != dist_list[-1]:
                fl.write("  ")
            else:
                fl.write("\n")

MPI.COMM_WORLD.barrier()

cascis = []

overlap = one_rdm = two_rdm = None
for trn_dist in [1.0, 3.6, 1.8, 2.6, 1.4, 3.2, 1.2, 1.6, 2.0, 2.8]:
    mol = get_mol(trn_dist)
    mf = scf.HF(mol)
    mf.kernel()
    MPI.COMM_WORLD.Bcast(mf.mo_coeff)
    cascis.append(CASCI(mf, ncas, neleca))
    overlap, one_rdm, two_rdm = append_to_rdms(
        cascis, overlap=overlap, one_rdm=one_rdm, two_rdm=two_rdm
    )

    if rank == 0:
        np.save("overlap.npy", overlap)
        np.save("one_rdm.npy", one_rdm)
        np.save("two_rdm.npy", two_rdm)

        pot_energy_surface = get_potential_energy_curve(
            overlap, one_rdm, two_rdm, test_mfs
        )

        with open("continued_surface.txt", "a") as fl:
            for el in pot_energy_surface:
                fl.write("{}".format(el))
                if el != pot_energy_surface[-1]:
                    fl.write("  ")
                else:
                    fl.write("\n")
    MPI.COMM_WORLD.barrier()
