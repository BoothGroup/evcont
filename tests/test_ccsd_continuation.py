import numpy as np
import pytest
from pyscf import gto

from evcont.ccsd.CCSD_EVCont import CCSD_EVCont_obj


pytestmark = pytest.mark.filterwarnings(
    r"ignore::DeprecationWarning:ebcc\..*"
)

RTOL = 1.0e-9
ATOL = 1.0e-10
OTHER_BASIS_RESULTS = {
    "meta_lowdin": (-2.1778568522863244, False),
    "split_procrustes": (-2.1779376837205913, True),
    "least_change_atom_coordinate": (-2.1779376837206126, True),
    "least_change_frozen_mo": (-2.1779376837211197, True),
    "least_change_local": (-2.1779373106984865, False),
}


def _water(oh_distance):
    half_angle = 0.5 * np.deg2rad(104.5)
    x = oh_distance * np.sin(half_angle)
    z = oh_distance * np.cos(half_angle)
    return gto.M(
        atom=[
            ("O", (0.0, 0.0, 0.0)),
            ("H", (x, 0.0, z)),
            ("H", (-x, 0.0, z)),
        ],
        basis="6-31g",
        unit="Angstrom",
        verbose=0,
    )


def _h4(spacing):
    return gto.M(
        atom=[("H", (i * spacing, 0.0, 0.0)) for i in range(4)],
        basis="sto-3g",
        unit="Bohr",
        symmetry=False,
        verbose=0,
    )


def test_ccsd_continuation_matches_pre_refactor_baseline():
    continuation = CCSD_EVCont_obj(_water(0.958), abstract_basis="SAO")
    for distance in (0.90, 1.05, 1.20):
        continuation.append_to_rdms(_water(distance))

    np.testing.assert_allclose(
        continuation.ens,
        [-76.11696114888737, -76.13275519587776, -76.17526566839865],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        continuation.train_energies,
        [-76.10703408263961, -76.11334493391917, -76.07109723294019],
        rtol=RTOL,
        atol=ATOL,
    )
    np.testing.assert_allclose(
        continuation.overlap,
        [
            [1.0, 0.9772429949732743, 0.9205667656992705],
            [0.9772429949732743, 1.0, 0.9821003000443925],
            [0.9205667656992705, 0.9821003000443925, 1.0],
        ],
        rtol=RTOL,
        atol=ATOL,
    )

    energy, _ = continuation.approximate(_water(1.00))
    np.testing.assert_allclose(
        np.ravel(energy)[0], -76.11962855448573, rtol=RTOL, atol=ATOL
    )


@pytest.mark.parametrize(
    "abstract_basis, expected",
    OTHER_BASIS_RESULTS.items(),
    ids=OTHER_BASIS_RESULTS,
)
def test_ccsd_continuation_with_other_abstract_bases(abstract_basis, expected):
    expected_energy, requires_reference = expected
    basis_kwargs = (
        {"least_change_emit_warnings": False}
        if abstract_basis.startswith("least_change")
        else None
    )
    continuation = CCSD_EVCont_obj(
        _h4(1.5),
        abstract_basis=abstract_basis,
        abstract_basis_kwargs=basis_kwargs,
    )

    assert (continuation.abstract_basis_ref is not None) is requires_reference
    overlap_ao = continuation.comp_mol.intor_symmetric("int1e_ovlp")
    np.testing.assert_allclose(
        continuation.comp_basis_abstract.T
        @ overlap_ao
        @ continuation.comp_basis_abstract,
        np.eye(continuation.comp_mol.nao),
        rtol=1.0e-9,
        atol=1.0e-10,
    )

    for spacing in (1.2, 1.8):
        continuation.append_to_rdms(_h4(spacing))

    np.testing.assert_allclose(
        continuation.overlap,
        continuation.overlap.T.conj(),
        rtol=1.0e-9,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.diag(continuation.overlap), np.ones(2), rtol=1.0e-9, atol=1.0e-10
    )

    energy, _ = continuation.approximate(_h4(1.6))
    np.testing.assert_allclose(
        np.ravel(energy)[0], expected_energy, rtol=1.0e-7, atol=1.0e-8
    )
