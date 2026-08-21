import importlib
import sys
from types import ModuleType

import numpy as np


def _import_sci_helpers(monkeypatch):
    monkeypatch.setitem(sys.modules, "PostProcessStates", ModuleType("PostProcessStates"))
    monkeypatch.setitem(sys.modules, "shci", ModuleType("shci"))
    sys.modules.pop("evcont.sci.SCI_EVCont", None)
    return importlib.import_module("evcont.sci.SCI_EVCont")


def test_sci_rdm_text_readers_apply_expected_index_convention(tmp_path, monkeypatch):
    sci = _import_sci_helpers(monkeypatch)
    one_path = tmp_path / "one.txt"
    one_path.write_text("2\n0 1 0.25\n1 0 -0.5\n")
    two_path = tmp_path / "two.txt"
    two_path.write_text("2\n0 1 1 0 0.75\n")

    one = sci.read1RDM(one_path)
    two = sci.read2RDM(two_path)
    assert one[1, 0] == 0.25
    assert one[0, 1] == -0.5
    assert two[0, 1, 1, 0] == 0.75


def test_sci_state_labels_and_cache_names_are_geometry_specific(tmp_path, monkeypatch):
    sci = _import_sci_helpers(monkeypatch)
    assert sci._state_label(2, 1, 2, nroots=3) == (2, 1)
    assert sci._state_label(2, 4, 5, nroots=3) == (5, 1)
    path = sci._cache_name(tmp_path, "rdm1", 2, 1, 5, 0)
    assert path.endswith("rdm1.g2.s1.g5.s0.txt")


def test_sci_cleanup_only_removes_transition_outputs(tmp_path, monkeypatch):
    sci = _import_sci_helpers(monkeypatch)
    targets = [tmp_path / "spatialRDM.1.0.txt", tmp_path / "spatial1RDM.1.0.txt"]
    keep = tmp_path / "rdm1.g0.s0.g0.s0.txt"
    for path in targets + [keep]:
        path.write_text("data")
    sci._cleanup_transition_outputs(tmp_path)
    assert not any(path.exists() for path in targets)
    assert keep.exists()
