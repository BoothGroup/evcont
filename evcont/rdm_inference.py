"""Helpers for constructing state RDMs from continuation coefficients."""

from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np

from evcont.electron_integral_utils import restore_electron_exchange_symmetry


@dataclass(frozen=True)
class RDMResult:
    """Requested inferred RDMs and their state/basis metadata."""

    state_pairs: tuple
    basis: str
    one: np.ndarray | None = None
    two: np.ndarray | None = None

    @property
    def one_rdm(self):
        return self.one

    @property
    def two_rdm(self):
        return self.two


def normalize_rdm_request(return_rdms):
    """Normalize an RDM request to ``("1rdm", "2rdm")``-style names."""
    if return_rdms in (None, False):
        return ()
    if return_rdms is True:
        return ("1rdm", "2rdm")
    if isinstance(return_rdms, str):
        return_rdms = (return_rdms,)

    aliases = {
        "1": "1rdm", "1rdm": "1rdm", "one": "1rdm",
        "2": "2rdm", "2rdm": "2rdm", "two": "2rdm",
    }
    try:
        request = tuple(dict.fromkeys(aliases[str(item).lower()] for item in return_rdms))
    except KeyError as error:
        raise ValueError(f"Unknown RDM request: {error.args[0]}") from None
    return request


def resolve_state_pairs(coefficients, state_pairs="diagonal"):
    """Return validated zero-based state pairs."""
    nstates = np.atleast_2d(coefficients).shape[0]
    if state_pairs in (None, "diagonal"):
        pairs = tuple((state, state) for state in range(nstates))
    elif state_pairs == "all":
        pairs = tuple((bra, ket) for bra in range(nstates) for ket in range(nstates))
    else:
        pairs = tuple(tuple(pair) for pair in state_pairs)
    if any(len(pair) != 2 or min(pair) < 0 or max(pair) >= nstates for pair in pairs):
        raise ValueError(f"Invalid state pairs for {nstates} states: {pairs}")
    return pairs


def contract_rdm1(one_trdm, bra, ket):
    """Contract a training 1tRDM with a pair of subspace eigenvectors."""
    return np.einsum("a,abij,b->ij", bra.conj(), one_trdm, ket, optimize="optimal")


def contract_rdm2(two_trdm, bra, ket, norb):
    """Contract a training 2tRDM, including supported compressed forms."""
    weights = np.outer(bra.conj(), ket)
    if two_trdm.ndim in (2, 5):
        # Existing data-point compression stores the lower triangle. This
        # contraction is exact for state RDMs (bra == ket), the primary output.
        packed_weights = 2 * weights
        np.fill_diagonal(packed_weights, np.diag(weights))
        result = np.tensordot(
            packed_weights[np.tril_indices(len(bra))], two_trdm, axes=1
        )
    else:
        result = np.tensordot(weights, two_trdm, axes=2)
    if result.ndim != 4:
        result = restore_electron_exchange_symmetry(result, norb)
    return result


def _to_ao(rdm, ao_basis):
    if rdm.ndim == 3:
        return np.einsum(
            "pa,...ab,qb->...pq",
            ao_basis,
            rdm,
            ao_basis.conj(),
            optimize="optimal",
        )
    return np.einsum(
        "pa,qb,rc,sd,...abcd->...pqrs",
        ao_basis, ao_basis.conj(), ao_basis, ao_basis.conj(), rdm,
        optimize="optimal",
    )


def make_rdm_result(one, two, state_pairs, *, basis="AO", ao_basis=None):
    """Package already-contracted RDM arrays in the requested basis."""
    basis_name = str(basis)
    if basis_name.upper() == "AO":
        if ao_basis is None:
            raise ValueError("ao_basis is required when returning RDMs in the AO basis")
        one = None if one is None else _to_ao(np.asarray(one), ao_basis)
        two = None if two is None else _to_ao(np.asarray(two), ao_basis)
        basis_name = "AO"
    return RDMResult(
        state_pairs=tuple(state_pairs), basis=basis_name, one=one, two=two
    )


def infer_rdms(
    coefficients,
    one_trdm,
    two_trdm=None,
    *,
    return_rdms=("1rdm",),
    state_pairs="diagonal",
    basis="AO",
    ao_basis=None,
    lowrank=False,
):
    """Build only the requested state RDMs from continuation coefficients."""
    request = normalize_rdm_request(return_rdms)
    if not request:
        return None
    if lowrank and "2rdm" in request:
        warnings.warn(
            "2RDM output is not implemented for low-rank continuation; omitting it.",
            RuntimeWarning,
            stacklevel=2,
        )
        request = tuple(item for item in request if item != "2rdm")
    if "2rdm" in request and two_trdm is None:
        raise ValueError("A training 2tRDM is required for 2RDM inference")

    coefficients = np.atleast_2d(coefficients)
    pairs = resolve_state_pairs(coefficients, state_pairs)
    one = two = None
    if "1rdm" in request:
        one = np.stack([
            contract_rdm1(one_trdm, coefficients[bra], coefficients[ket])
            for bra, ket in pairs
        ])
    if "2rdm" in request:
        norb = one_trdm.shape[-1]
        two = np.stack([
            contract_rdm2(two_trdm, coefficients[bra], coefficients[ket], norb)
            for bra, ket in pairs
        ])

    return make_rdm_result(
        one, two, pairs, basis=basis, ao_basis=ao_basis
    )
