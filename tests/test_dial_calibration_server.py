"""test_dial_calibration_server -- the dial-calibration merge on /steer/axes.

research/dial_autocalibrate.py (a sibling module, NOT touched here) measures, per model, which dials
actually move a reply toward their own pole and how far each can be pushed before the model derails
(usable_range / usable_max / derail_point / range_valid -- see that module's docstring). This file is
GENERIC PLUMBING: it puts a CURATED distillation of that calibration in front of the Behavior page's
sliders, so a slider caps to what actually works on THIS model instead of one fixed default everywhere.

Two pieces, both in clozn_server.py:
  * _dial_calibration() -- reads a persisted ~/.clozn/dial_calibration.json (the SAME _pers(...) convention
    studio_personality.json/studio_memory.pt already use -- never research/runs/dial_autocalibrate.json,
    the raw research file with full curves/sample_replies). Missing/broken file -> {} (never raise).
  * _with_calibration() / the /steer/axes handler in Substrate._steer -- merges one calibration entry into
    one axis dict (built-in OR custom), additively.

No model, no GPU, no socket: _dial_calibration/_with_calibration are pure functions exercised directly;
Substrate._steer is exercised on a BARE instance built via object.__new__(cs.Substrate) (Substrate has no
__init__ of its own -- subclasses set self.steer/self.name/self._steer_ready post-construction, per its own
docstring -- so this needs no heavy QwenSubstrate/DreamSubstrate model load, the same
no-heavy-__init__ spirit as test_bridge_server.py's object.__new__(H) handler trick). CLOZN_DIR is
monkeypatched to a tmp_path so a real calibration file on this machine can never leak into a test.

Proves: (1) a missing calibration file is a STRICT no-op -- every axis (built-in AND custom) renders
byte-for-byte as it did before this feature existed, plus only the additive "calibrated": False; (2) a
calibrated, working dial gets its slider capped to usable_max and carries usable_range/derail_point/
works=True; (3) a calibrated dial that never found a usable dose falls back to its ORIGINAL declared max
and carries works=False; (4) the loader tolerates either the curated flat shape or the raw research-JSON
shape ({"dials": {...}}, "range_valid" instead of "works"); (5) custom (user-defined) dials get identical
treatment to built-ins; (6) a missing/corrupt calibration file never raises.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs   # noqa: E402
from clozn.behavior.steering import AXES   # noqa: E402


# --- isolate ~/.clozn (CLOZN_DIR) so a real calibration file on this machine never leaks into a test ------

@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "CLOZN_DIR", str(tmp_path))
    return tmp_path


def _write_calibration(tmp_path, table):
    with open(os.path.join(str(tmp_path), "dial_calibration.json"), "w", encoding="utf-8") as f:
        json.dump(table, f)


# --- a bare Substrate + fake steer object: _steer("/steer/axes", ...) only ever reads
# self.steer.strength/self.steer.custom/self._steer_ready/self.name for this path (no _ensure_steer(), no
# model) -- see module docstring. -----------------------------------------------------------------------

class FakeSteer:
    def __init__(self, strength=None, custom=None):
        self.strength = dict(strength or {})
        self.custom = dict(custom or {})


def _bare_substrate(strength=None, custom=None, name="faketest"):
    sub = object.__new__(cs.Substrate)
    sub.steer = FakeSteer(strength, custom)
    sub._steer_ready = True
    sub.name = name
    return sub


def _axes_by_name(result):
    return {a["name"]: a for a in result["axes"]}


# ==================================================================================== _dial_calibration()

def test_dial_calibration_missing_file_is_empty_dict(iso):
    assert cs._dial_calibration() == {}


def test_dial_calibration_corrupt_file_is_empty_dict_never_raises(iso, tmp_path):
    path = os.path.join(str(tmp_path), "dial_calibration.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write("{not valid json")
    assert cs._dial_calibration() == {}


def test_dial_calibration_reads_the_flat_curated_shape(iso, tmp_path):
    _write_calibration(tmp_path, {"warm": {"usable_max": 0.9, "usable_range": [0.25, 0.9],
                                            "derail_point": 1.2, "works": True}})
    assert cs._dial_calibration() == {
        "warm": {"usable_max": 0.9, "usable_range": [0.25, 0.9], "derail_point": 1.2, "works": True},
    }


def test_dial_calibration_tolerates_the_raw_research_json_shape(iso, tmp_path):
    """research/dial_autocalibrate.py's raw output nests dials under "dials" and calls the bool field
    "range_valid", not "works" -- the loader must degrade gracefully to that shape too, in case the curated
    file is ever just a copy of the raw research output."""
    _write_calibration(tmp_path, {"dials": {"candid": {"usable_max": None, "usable_range": [None, None],
                                                        "derail_point": None, "range_valid": False}}})
    assert cs._dial_calibration() == {
        "candid": {"usable_max": None, "usable_range": [None, None], "derail_point": None, "works": False},
    }


def test_dial_calibration_skips_a_malformed_entry_not_the_whole_table(iso, tmp_path):
    _write_calibration(tmp_path, {"warm": "not a dict", "candid": {"usable_max": 0.3, "works": True}})
    calib = cs._dial_calibration()
    assert "warm" not in calib
    assert calib["candid"]["usable_max"] == 0.3


# ==================================================================================== _with_calibration()

def test_with_calibration_no_entry_caps_max_to_the_enforced_uncalibrated_ceiling():
    """No calibration for this model -> the slider must advertise the ceiling the API will ENFORCE.

    This previously asserted the axis was returned untouched but for calibrated:False -- i.e. the
    slider still offered max 1.5 on a model nobody had ever swept, while EngineSteer.set() would
    later clamp it. A slider that offers a value the API silently refuses is a lie about the
    instrument, and on a real model (Qwen2.5-7B-Q4) dragging it there produced repetition loops."""
    from clozn.behavior.steering.engine_adapter import UNSAFE_STRENGTH
    axis = {"name": "warm", "poles": ("warm", "detached"), "value": 0.0, "max": 1.5}
    out = cs._with_calibration(dict(axis), None)
    assert out["calibrated"] is False
    assert out["max"] == UNSAFE_STRENGTH        # the enforced ceiling, not the declared 1.5
    assert out["declared_max"] == 1.5           # what it WOULD be if this model were calibrated
    assert "uncalibrated" in out                # and a reason a UI can actually show
    # everything else about the axis is untouched
    assert out["name"] == "warm" and out["poles"] == ("warm", "detached") and out["value"] == 0.0


def test_with_calibration_working_dial_caps_max_and_adds_hint_fields():
    axis = {"name": "warm", "poles": ("warm", "detached"), "value": 0.0, "max": 1.5}
    c = {"usable_max": 0.9, "usable_range": [0.25, 0.9], "derail_point": 1.2, "works": True}
    out = cs._with_calibration(dict(axis), c)
    assert out["max"] == 0.9
    assert out["usable_range"] == [0.25, 0.9]
    assert out["derail_point"] == 1.2
    assert out["works"] is True
    assert out["calibrated"] is True


def test_with_calibration_never_worked_falls_back_to_declared_max():
    axis = {"name": "candid", "poles": ("candid", "agreeable"), "value": 0.0, "max": 0.45}
    c = {"usable_max": None, "usable_range": [None, None], "derail_point": None, "works": False}
    out = cs._with_calibration(dict(axis), c)
    assert out["max"] == 0.45          # falls back -- the calibration never found a usable ceiling
    assert out["works"] is False
    assert out["calibrated"] is True


# ==================================================================================== /steer/axes end to end

def test_steer_axes_with_no_calibration_file_caps_only_the_over_ceiling_dials(iso):
    """No ~/.clozn/dial_calibration.json -> every dial is capped at the ENFORCED uncalibrated ceiling.

    Surgical by construction: a dial whose own declared max is already at or under the ceiling is
    returned untouched, because min(declared, ceiling) == declared. Only the optimistic 1.5 dials
    move. (This test previously asserted the opposite -- that an uncalibrated install was a strict
    no-op and warm still advertised 1.5 -- which is precisely the fail-open being fixed.)"""
    from clozn.behavior.steering.engine_adapter import UNSAFE_STRENGTH
    sub = _bare_substrate(strength={"warm": 0.3},
                          custom={"skeptical": {"poles": ["skeptical", "neutral"], "max": 0.5}})
    result = sub._steer("/steer/axes", {})
    axes = _axes_by_name(result)

    warm = axes["warm"]                        # declares no "max" -> the 1.5 default -> CAPPED
    assert warm["max"] == UNSAFE_STRENGTH
    assert warm["declared_max"] == 1.5
    assert warm["value"] == 0.3
    assert warm["poles"] == AXES["warm"]["poles"]
    assert warm["calibrated"] is False
    assert set(warm.keys()) == {"name", "poles", "value", "max", "calibrated",
                                "declared_max", "uncalibrated"}

    candid = axes["candid"]                    # 0.45 <= ceiling -> genuinely untouched
    assert candid["max"] == 0.45
    assert candid["calibrated"] is False

    skeptical = axes["skeptical"]              # 0.5 <= ceiling -> genuinely untouched
    assert skeptical["max"] == 0.5
    assert skeptical["custom"] is True
    assert skeptical["calibrated"] is False

    assert result["ready"] is True
    assert result["substrate"] == "faketest"


def test_steer_axes_calibrated_working_dial_caps_the_slider_and_adds_hints(iso, tmp_path):
    _write_calibration(tmp_path, {"warm": {"usable_max": 0.9, "usable_range": [0.25, 0.9],
                                            "derail_point": 1.2, "works": True}})
    axes = _axes_by_name(_bare_substrate()._steer("/steer/axes", {}))
    warm = axes["warm"]
    assert warm["max"] == 0.9                  # capped to the CALIBRATED ceiling, not the declared 1.5
    assert warm["usable_range"] == [0.25, 0.9]
    assert warm["derail_point"] == 1.2
    assert warm["works"] is True
    assert warm["calibrated"] is True
    # an uncalibrated sibling axis in the SAME response is untouched
    assert axes["candid"]["calibrated"] is False
    assert axes["candid"]["max"] == 0.45


def test_steer_axes_calibrated_dead_dial_falls_back_to_declared_max_and_flags_unworking(iso, tmp_path):
    _write_calibration(tmp_path, {"curious": {"usable_max": None, "usable_range": [None, None],
                                              "derail_point": None, "works": False}})
    curious = _axes_by_name(_bare_substrate()._steer("/steer/axes", {}))["curious"]
    assert curious["max"] == 1.5               # falls back -- calibration never found a usable ceiling
    assert curious["works"] is False
    assert curious["calibrated"] is True


def test_steer_axes_custom_dial_gets_calibration_merged_too(iso, tmp_path):
    _write_calibration(tmp_path, {"skeptical": {"usable_max": 0.3, "usable_range": [0.1, 0.3],
                                                "derail_point": 0.4, "works": True}})
    sub = _bare_substrate(custom={"skeptical": {"poles": ["skeptical", "neutral"], "max": 0.5}})
    skeptical = _axes_by_name(sub._steer("/steer/axes", {}))["skeptical"]
    assert skeptical["max"] == 0.3
    assert skeptical["usable_range"] == [0.1, 0.3]
    assert skeptical["works"] is True
    assert skeptical["calibrated"] is True
    assert skeptical["custom"] is True         # the custom marker itself must survive the merge


def test_steer_axes_tolerates_raw_research_json_shape_end_to_end(iso, tmp_path):
    _write_calibration(tmp_path, {"dials": {"warm": {"usable_max": 1.0, "usable_range": [0.25, 1.0],
                                                      "derail_point": None, "range_valid": True}}})
    warm = _axes_by_name(_bare_substrate()._steer("/steer/axes", {}))["warm"]
    assert warm["max"] == 1.0
    assert warm["works"] is True
    assert warm["calibrated"] is True
