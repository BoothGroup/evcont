from pyscf import gto

import numpy as np

"""
Constructs the GS Zundel geometry in Cartesian coordinates in the order
(O, H, H, H, O, H, H), based on the values in https://pubs.aip.org/aip/jcp/article-abstract/122/4/044308/447592/Ab-initio-potential-energy-and-dipole-moment?redirectedFrom=PDF
(see also https://arxiv.org/pdf/1312.2897)
"""

mol = gto.Mole()
mol.atom = """
    H+
    O1 1 1.1950
    O2 1 1.1950 2 173.730
    H1 2 0.9686 1 115.849 3 295.302
    H2 2 0.9682 1 118.158 3 163.635
    H3 3 0.9686 1 115.849 2 295.302
    H4 3 0.9682 1 118.158 2 163.635
"""
mol.build(
    basis="6-31G",
    symmetry=True,
    unit="Angstrom",
    charge=1,
)

init_geometry = mol.atom_coords(unit="Angstrom")

geom_resorted = init_geometry[[1, 3, 4, 0, 2, 5, 6]]  # O, H, H, H, O, H, H


print(geom_resorted)


np.save(geom_resorted, "init_geometry.npy")
