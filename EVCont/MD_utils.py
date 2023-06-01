from pyscf import md, scf, lib, grad

import numpy as np

from .ab_initio_gradients_loewdin import get_energy_with_grad

from .electron_integral_utils import get_integrals, get_basis

def get_scanner(mol, one_rdm, two_rdm, overlap):
    class Base():
        converged = True

    class Scanner(lib.GradScanner):
        def __init__(self):
            self.mol = mol
            self.base = Base()
            # self.converged = True
        def __call__(self, mol):
            self.mol = mol
            if one_rdm is not None and two_rdm is not None and overlap is not None:
                return get_energy_with_grad(mol, one_rdm, two_rdm, overlap)
            else:
                return mol.energy_nuc(), grad.RHF(scf.RHF(mol)).grad_nuc()
    return Scanner()

def get_trajectory(init_mol, overlap, one_rdm, two_rdm, dt=100., steps=10, init_veloc=None):
    scanner_fun = get_scanner(init_mol, one_rdm, two_rdm, overlap)

    frames = []
    myintegrator = md.NVE(scanner_fun, dt=dt, steps=steps, veloc=init_veloc, incore_anyway=True, frames=frames)
    myintegrator.run()
    trajectory = [frame.coord for frame in frames]

    return np.array(trajectory)
