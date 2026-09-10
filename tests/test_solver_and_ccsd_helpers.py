from types import SimpleNamespace

import numpy as np
import pytest

import evcont.solver_evaluation as evaluation_module
from evcont.ccsd.CCSD_EVCont import (
    _mixed_rdms,
    _zero_amplitude_state,
)
from evcont.rdm_orthonormalization import (
    symmetrize_transition_rdm1,
    symmetrize_transition_rdm2,
)
from evcont.solver_evaluation import EVContEvaluationMixin


def test_ccsd_zero_amplitude_state_has_consistent_shapes_and_reference_rdms():
    state = _zero_amplitude_state(nocc=2, nvir=3)
    assert state.t1.shape == (2, 3)
    assert state.t2.shape == (2, 2, 3, 3)
    assert state.l1.shape == (3, 2)
    assert state.l2.shape == (3, 3, 2, 2)
    one, two = _mixed_rdms(state, state)
    np.testing.assert_allclose(np.trace(one), 4.0)
    np.testing.assert_allclose(one, np.diag([2.0, 2.0, 0.0, 0.0, 0.0]))
    assert two.shape == (5, 5, 5, 5)


def test_transition_rdm_symmetrizers_enforce_documented_permutations(rng):
    one = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    two = rng.normal(size=(3, 3, 3, 3)) + 1j * rng.normal(size=(3, 3, 3, 3))
    sym_one = symmetrize_transition_rdm1(one)
    sym_two = symmetrize_transition_rdm2(two)
    np.testing.assert_allclose(sym_one, sym_one.T.conj())
    np.testing.assert_allclose(sym_two, sym_two.transpose(1, 0, 3, 2).conj())


class DummyEvaluation(EVContEvaluationMixin):
    nroots = 2
    abstract_basis = "split_procrustes"
    abstract_basis_ref = "basis-ref"
    abstract_basis_ref_mol = "mol-ref"
    abstract_basis_ref_mf = "mf-ref"
    abstract_basis_kwargs = {"custom": 7}
    _basis_name = "split_procrustes"
    lowrank = False
    one_rdm = "one"
    two_rdm = "two"
    overlap = np.eye(2)


def test_evaluation_mixin_forwards_basis_reference_and_nroots(monkeypatch):
    captured = {}

    def fake(mol, one, two, overlap, **kwargs):
        captured.update(mol=mol, one=one, two=two, overlap=overlap, **kwargs)
        return "energies", "vectors"

    monkeypatch.setattr(evaluation_module, "approximate_multistate_abstract_basis", fake)
    result = DummyEvaluation().get_en("molecule", nroots=1, lindep=1e-8)

    assert result == ("energies", "vectors")
    assert captured == {
        "mol": "molecule",
        "one": "one",
        "two": "two",
        "overlap": DummyEvaluation.overlap,
        "nroots": 1,
        "abstract_basis": "split_procrustes",
        "basis_ref": "basis-ref",
        "basis_ref_mol": "mol-ref",
        "ref_mf": "mf-ref",
        "custom": 7,
        "lindep": 1e-8,
    }


def test_lowrank_option_filter_excludes_reduction_only_arguments():
    holder = DummyEvaluation()
    holder.kwargs = {
        "density_fit": False,
        "Jdiag_only": True,
        "nvecs": 20,
        "truncation_style": "nvec",
    }
    assert holder._lowrank_evaluation_kwargs({"lindep": 1e-6}) == {
        "density_fit": False,
        "Jdiag_only": True,
        "lindep": 1e-6,
    }
