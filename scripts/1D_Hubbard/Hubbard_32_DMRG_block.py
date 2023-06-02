import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes
import sys

U = float(sys.argv[1])
OBC = bool(int(sys.argv[2]))

L = n_elec = 32

driver = DMRGDriver(stack_mem=1 << 36, stack_mem_ratio=0.8, symm_type=SymmetryTypes.SU2)
driver.initialize_system(n_sites=L, n_elec=n_elec, spin=0)

t = 1

b = driver.expr_builder()

# hopping term
if OBC:
    b.add_term("(C+D)0",
        np.array([[[i, i + 1], [i + 1, i]] for i in range(L - 1)]).ravel(),
        [np.sqrt(2) * -t] * 2 * (L - 1))
else:
    b.add_term("(C+D)0",
        np.array([[[i, (i + 1)%L], [(i + 1)%L, i]] for i in range(L)]).ravel(),
        [np.sqrt(2) * -t] * 2 * (L - 1) + [np.sqrt(2) * t] * 2)

# onsite term
b.add_term("((C+(C+D)0)1+D)0",
    np.array([[i, ] * 4 for i in range(L)]).ravel(),
    [U] * L)

mpo = driver.get_mpo(b.finalize(), iprint=2)

ket = driver.get_random_mps(tag="GS", bond_dim=100, nroots=1)
bond_dims = [100]*5 + [250] * 10 + [500] * 10 + [1000 + i * 100 for i in range(16)]
noises = [1e-4] * 5 + [1e-5] * 10 + [1e-6] * 10 + [0]


en = driver.dmrg(mpo, ket, n_sweeps=1000, bond_dims=bond_dims, noises=noises, thrds=[1e-12]*46, iprint=1)

if OBC:
    bc_string = "OBC"
else:
    bc_string = "APBC"

with open("result.txt", "w") as fl:
    fl.write("U  BC  Energy\n")
    fl.write("{}  {}  {}\n".format(U, bc_string, en))

