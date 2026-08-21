import numpy as np
import pytest
from pyscf import gto


@pytest.fixture(scope="session")
def h2_molecule():
    def build(distance=1.4, basis="sto-3g"):
        return gto.M(
            atom=[("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, distance))],
            basis=basis,
            unit="Bohr",
            symmetry=False,
            verbose=0,
        )

    return build


@pytest.fixture
def rng():
    return np.random.default_rng(20260821)
