import numpy as np
import pytest

from evcont.electron_integral_utils import compress_electron_exchange_symmetry
from evcont.excited_utils import make_rdm1, make_trdm1, oscillator_strength
from evcont.rdm_inference import (
    RDMResult,
    contract_rdm1,
    contract_rdm2,
    infer_rdms,
    make_rdm_result,
    normalize_rdm_request,
    resolve_state_pairs,
)


@pytest.mark.parametrize(
    ("rdm_request", "expected"),
    [
        (None, ()),
        (False, ()),
        (True, ("1rdm", "2rdm")),
        ("one", ("1rdm",)),
        (("2", "1rdm", "two"), ("2rdm", "1rdm")),
    ],
)
def test_normalize_rdm_request_aliases_and_deduplicates(rdm_request, expected):
    assert normalize_rdm_request(rdm_request) == expected


def test_normalize_rdm_request_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown RDM request"):
        normalize_rdm_request("three")


def test_resolve_state_pairs_supports_diagonal_all_and_validation():
    coefficients = np.eye(3)
    assert resolve_state_pairs(coefficients) == ((0, 0), (1, 1), (2, 2))
    assert len(resolve_state_pairs(coefficients, "all")) == 9
    assert resolve_state_pairs(coefficients, [(2, 0)]) == ((2, 0),)
    with pytest.raises(ValueError, match="Invalid state pairs"):
        resolve_state_pairs(coefficients, [(0, 3)])


def test_rdm_contractions_match_explicit_weighted_sums(rng):
    ntrain, norb = 4, 3
    one = rng.normal(size=(ntrain, ntrain, norb, norb))
    two = rng.normal(size=(ntrain, ntrain, norb, norb, norb, norb))
    bra = rng.normal(size=ntrain) + 1j * rng.normal(size=ntrain)
    ket = rng.normal(size=ntrain) + 1j * rng.normal(size=ntrain)

    np.testing.assert_allclose(contract_rdm1(one, bra, ket), np.einsum("a,abij,b->ij", bra.conj(), one, ket))
    np.testing.assert_allclose(contract_rdm2(two, bra, ket, norb), np.einsum("a,abijkl,b->ijkl", bra.conj(), two, ket))


def test_compressed_two_rdm_contraction_is_exact_for_diagonal_state(rng):
    ntrain, norb = 3, 2
    vector = rng.normal(size=ntrain)
    full = rng.normal(size=(ntrain, ntrain, norb**2, norb**2))
    full = 0.25 * (
        full
        + full.transpose(1, 0, 2, 3)
        + full.transpose(0, 1, 3, 2)
        + full.transpose(1, 0, 3, 2)
    )
    full = full.reshape((ntrain, ntrain) + (norb,) * 4)
    packed_data = []
    packed_both = []
    for i, j in zip(*np.tril_indices(ntrain)):
        packed_data.append(full[i, j])
        packed_both.append(compress_electron_exchange_symmetry(full[i, j]))

    expected = contract_rdm2(full, vector, vector, norb)
    np.testing.assert_allclose(contract_rdm2(np.asarray(packed_data), vector, vector, norb), expected)
    np.testing.assert_allclose(contract_rdm2(np.asarray(packed_both), vector, vector, norb), expected)


def test_infer_rdms_selects_pairs_and_transforms_to_ao(rng):
    coefficients = np.array([[1.0, 0.0], [0.6, 0.8]])
    one = rng.normal(size=(2, 2, 3, 3))
    two = rng.normal(size=(2, 2, 3, 3, 3, 3))
    ao_basis = rng.normal(size=(5, 3))

    result = infer_rdms(
        coefficients,
        one,
        two,
        return_rdms=True,
        state_pairs=[(1, 0)],
        basis="AO",
        ao_basis=ao_basis,
    )

    assert isinstance(result, RDMResult)
    assert result.state_pairs == ((1, 0),)
    assert result.basis == "AO"
    assert result.one.shape == (1, 5, 5)
    assert result.two.shape == (1, 5, 5, 5, 5)
    orbital_one = contract_rdm1(one, coefficients[1], coefficients[0])
    expected_one = ao_basis @ orbital_one @ ao_basis.T
    np.testing.assert_allclose(result.one[0], expected_one)


def test_lowrank_rdm_inference_warns_and_omits_two_rdm(rng):
    coefficients = np.eye(2)
    one = rng.normal(size=(2, 2, 2, 2))
    with pytest.warns(RuntimeWarning, match="not implemented"):
        result = infer_rdms(
            coefficients,
            one,
            return_rdms=True,
            basis="orbital",
            lowrank=True,
        )
    assert result.one is not None
    assert result.two is None


def test_make_rdm_result_requires_ao_basis_for_ao_output():
    with pytest.raises(ValueError, match="ao_basis is required"):
        make_rdm_result(np.eye(2)[None], None, [(0, 0)], basis="AO")


def test_excited_state_rdm_helpers_and_oscillator_strength(rng):
    one = rng.normal(size=(3, 3, 4, 4))
    left = np.array([0.2, -0.4, 0.7])
    right = np.array([0.6, 0.1, -0.2])

    expected = np.einsum("i,ijkl,j->kl", left, one, right)
    np.testing.assert_allclose(make_trdm1(None, one, left, right), expected)
    np.testing.assert_allclose(make_rdm1(None, one, left), make_trdm1(None, one, left, left))
    assert oscillator_strength(2.0, 1.5, np.array([1.0, 2.0, 2.0])) == pytest.approx(3.0)
