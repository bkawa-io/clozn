"""test_dial_library_server -- deploying the curated dial LIBRARY as studio dials, distinct from "yours".

research/dial_library_shipped.json ships 33 human-curated, per-model-calibrated tone dials. 6 of them
share a name with a steering.AXES built-in (warm, playful, formal, concise, poetic, concrete) and are
already live + already capped by ~/.clozn/dial_calibration.json -- untouched by anything in this file. The
other 27 don't exist as a dial anywhere until research/deploy_dial_library.py registers them (needs the
loaded 7B/GPU -- NOT run here). This file covers everything about that deploy that IS model-free:

  * deploy_dial_library.py's pure planning functions -- load_shipped_library, library_only_dials (the
    33 -> 27 split, computed from steering.AXES, never a hardcoded count), already_deployed_names /
    existing_user_custom_names (both plain JSON-file reads), plan() (the to_add/skip_deployed/
    skip_collision partition), and the --check CLI path (main(["--check"]) touches NO files and loads NO
    model). deploy() itself (the GPU path: boots a substrate, calls steer.add_custom, writes
    studio_library.json) is intentionally NOT exercised here -- that needs the real backbone, and is the
    "GPU deploy-run pending" this test suite explicitly does not attempt.

  * clozn_server.py's read side: _library_dial_names() (reads ~/.clozn/studio_library.json's keys, {} /
    missing-file / corrupt-file tolerant, exactly like _dial_calibration()) and the /steer/axes handler's
    "library": true flag -- a SHIPPED dial must read as "library", a user's own custom dial must keep
    reading as "custom" ("yours" + deletable in the UI), and with NO studio_library.json at all (the
    pre-deploy / never-deployed case) every custom dial must render BYTE-FOR-BYTE as it did before this
    feature existed (Law-#6-style strict backward compat -- mirrors
    test_dial_calibration_server.py's own no-calibration-file regression test).

No model, no GPU, no socket: exercised the same way test_dial_calibration_server.py is -- a BARE Substrate
via object.__new__(cs.Substrate) + a FakeSteer stub (Substrate.__init__ itself, and its own
new studio_library.json load line, are NOT constructed here; that boot-time wiring is a 1-line addition
that mirrors the pre-existing studio_custom_<name>.json load immediately above it, and is exercised for
real only by the GPU deploy run). CLOZN_DIR is monkeypatched to a tmp_path so a real ~/.clozn on this
machine can never leak into a test.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)                                    # repo root, for `from clozn import ...`
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "calibration"))  # deploy_dial_library.py lives in scripts/calibration/

from clozn.server import app as cs          # noqa: E402
import deploy_dial_library as ddl  # noqa: E402
from clozn.behavior.steering import AXES          # noqa: E402


# --- isolate ~/.clozn (CLOZN_DIR) so a real studio_library.json/studio_custom_qwen.json on this machine
# never leaks into a test -- deploy_dial_library.py reads paths through cs._pers, which resolves off
# cs.CLOZN_DIR at call time, so patching the module clozn_server owns is enough for BOTH modules (they
# share the same imported clozn_server module object). ------------------------------------------------

@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(cs, "CLOZN_DIR", str(tmp_path))
    return tmp_path


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def _write_library(tmp_path, table):
    _write_json(os.path.join(str(tmp_path), "studio_library.json"), table)


def _write_user_customs(tmp_path, table):
    _write_json(os.path.join(str(tmp_path), f"studio_custom_{cs.EngineSubstrate.name}.json"), table)


# ==================================================================================== load_shipped_library

def test_load_shipped_library_reads_the_real_shipped_file():
    dials = ddl.load_shipped_library()
    assert isinstance(dials, list)
    assert len(dials) == 33
    names = {d["name"] for d in dials}
    assert len(names) == 33                          # every name unique
    for d in dials:
        assert set(d) >= {"name", "category", "pos", "neg", "ship_range"}
        assert len(d["ship_range"]) == 2


def test_load_shipped_library_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ddl.load_shipped_library(str(tmp_path / "nope.json"))


def test_load_shipped_library_malformed_shape_raises(tmp_path):
    p = tmp_path / "bad.json"
    _write_json(str(p), {"not_dials": []})
    with pytest.raises(ValueError):
        ddl.load_shipped_library(str(p))


# ==================================================================================== library_only_dials

def test_library_only_dials_splits_33_into_27_and_the_6_axes_overlap():
    dials = ddl.load_shipped_library()
    only = ddl.library_only_dials(dials)
    only_names = {d["name"] for d in only}
    assert len(only) == 27
    overlap = {d["name"] for d in dials} & set(AXES)
    assert overlap == {"warm", "playful", "formal", "concise", "poetic", "concrete"}
    assert only_names == {d["name"] for d in dials} - overlap
    assert only_names.isdisjoint(set(AXES))


def test_library_only_dials_is_computed_from_axes_not_hardcoded():
    fake = [{"name": "warm"}, {"name": "brand_new_dial"}, {"name": "candid"}]
    only = ddl.library_only_dials(fake)
    assert [d["name"] for d in only] == ["brand_new_dial"]   # warm + candid are both real AXES keys


# ==================================================================================== already_deployed_names

def test_already_deployed_names_missing_file_is_empty_set(iso):
    assert ddl.already_deployed_names() == set()


def test_already_deployed_names_reads_the_written_keys(iso, tmp_path):
    _write_library(tmp_path, {"editor": {"pos": "p", "neg": "n", "max": 0.5}})
    assert ddl.already_deployed_names() == {"editor"}


def test_already_deployed_names_corrupt_file_is_empty_set_never_raises(iso, tmp_path):
    with open(os.path.join(str(tmp_path), "studio_library.json"), "w", encoding="utf-8") as f:
        f.write("{not valid json")
    assert ddl.already_deployed_names() == set()


# ==================================================================================== existing_user_custom_names

def test_existing_user_custom_names_missing_file_is_empty_set(iso):
    assert ddl.existing_user_custom_names() == set()


def test_existing_user_custom_names_reads_the_written_keys(iso, tmp_path):
    _write_user_customs(tmp_path, {"skeptical": {"pos": "p", "neg": "n", "max": 0.6}})
    assert ddl.existing_user_custom_names() == {"skeptical"}


def test_existing_user_custom_names_corrupt_file_is_empty_set_never_raises(iso, tmp_path):
    with open(os.path.join(str(tmp_path), f"studio_custom_{cs.EngineSubstrate.name}.json"), "w",
              encoding="utf-8") as f:
        f.write("{not valid json")
    assert ddl.existing_user_custom_names() == set()


# ==================================================================================== plan()

def test_plan_with_nothing_on_disk_registers_everything(iso):
    to_register = [{"name": "editor"}, {"name": "coach"}]
    to_add, skip_deployed, skip_collision = ddl.plan(to_register)
    assert [d["name"] for d in to_add] == ["editor", "coach"]
    assert skip_deployed == []
    assert skip_collision == []


def test_plan_skips_already_deployed(iso, tmp_path):
    _write_library(tmp_path, {"editor": {"pos": "p", "neg": "n", "max": 0.5}})
    to_register = [{"name": "editor"}, {"name": "coach"}]
    to_add, skip_deployed, skip_collision = ddl.plan(to_register)
    assert [d["name"] for d in to_add] == ["coach"]
    assert skip_deployed == ["editor"]
    assert skip_collision == []


def test_plan_flags_a_user_custom_collision_and_does_not_add_it(iso, tmp_path):
    _write_user_customs(tmp_path, {"coach": {"pos": "p", "neg": "n", "max": 0.6}})
    to_register = [{"name": "editor"}, {"name": "coach"}]
    to_add, skip_deployed, skip_collision = ddl.plan(to_register)
    assert [d["name"] for d in to_add] == ["editor"]
    assert skip_deployed == []
    assert skip_collision == ["coach"]


def test_plan_deployed_takes_precedence_over_collision_for_the_same_name(iso, tmp_path):
    # a name already in studio_library.json is "deployed", full stop -- even if a stale/unrelated entry of
    # the same name also happens to sit in the user's file (shouldn't normally happen; deployed wins).
    _write_library(tmp_path, {"editor": {"pos": "p", "neg": "n", "max": 0.5}})
    _write_user_customs(tmp_path, {"editor": {"pos": "x", "neg": "y", "max": 0.9}})
    to_add, skip_deployed, skip_collision = ddl.plan([{"name": "editor"}])
    assert to_add == []
    assert skip_deployed == ["editor"]
    assert skip_collision == []


# ==================================================================================== argparse + --check

def test_build_arg_parser_defaults():
    args = ddl.build_arg_parser().parse_args([])
    assert args.check is False
    assert args.force is False


def test_build_arg_parser_flags():
    args = ddl.build_arg_parser().parse_args(["--check", "--force"])
    assert args.check is True
    assert args.force is True


def test_main_check_mode_touches_nothing_and_loads_no_model(iso, tmp_path, capsys):
    # a real, isolated ~/.clozn with nothing in it yet -- --check must run to completion (reading only
    # the real dial_library_shipped.json + the two small, absent JSON files) without constructing any
    # substrate/model and without creating any file under tmp_path.
    result = ddl.main(["--check"])
    assert result is None
    out = capsys.readouterr().out
    assert "27 library-only" in out
    assert "6 already steering.AXES built-ins" in out
    assert os.listdir(str(tmp_path)) == []   # --check writes nothing


def test_main_check_mode_reports_an_already_deployed_dial(iso, tmp_path, capsys):
    _write_library(tmp_path, {"editor": {"pos": "p", "neg": "n", "max": 0.5}})
    ddl.main(["--check"])
    out = capsys.readouterr().out
    assert "already deployed" in out
    assert "editor" in out


# ==================================================================================== clozn_server._library_dial_names

def test_library_dial_names_missing_file_is_empty_set(iso):
    assert cs._library_dial_names() == set()


def test_library_dial_names_reads_the_written_keys(iso, tmp_path):
    _write_library(tmp_path, {"editor": {"pos": "p", "neg": "n", "max": 0.5},
                              "coach": {"pos": "p", "neg": "n", "max": 1.5}})
    assert cs._library_dial_names() == {"editor", "coach"}


def test_library_dial_names_corrupt_file_is_empty_set_never_raises(iso, tmp_path):
    with open(os.path.join(str(tmp_path), "studio_library.json"), "w", encoding="utf-8") as f:
        f.write("{not valid json")
    assert cs._library_dial_names() == set()


def test_library_dial_names_non_dict_json_is_empty_set(iso, tmp_path):
    _write_json(os.path.join(str(tmp_path), "studio_library.json"), ["editor", "coach"])
    assert cs._library_dial_names() == set()


# ==================================================================================== /steer/axes end to end

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


def test_steer_axes_with_no_library_file_is_a_strict_noop(iso):
    """The core regression: with no ~/.clozn/studio_library.json, every custom dial must render EXACTLY
    as it did before "library" existed -- "custom": True, no "library" key at all -- and every built-in
    axis must carry no "library" key either. Mirrors test_dial_calibration_server.py's own
    no-calibration-file regression test, for this feature."""
    sub = _bare_substrate(strength={"warm": 0.3},
                          custom={"skeptical": {"poles": ["skeptical", "neutral"], "max": 0.5},
                                   "editor": {"poles": ["editor", "neutral"], "max": 0.5}})
    axes = _axes_by_name(sub._steer("/steer/axes", {}))

    warm = axes["warm"]
    assert "library" not in warm
    assert "custom" not in warm

    for name in ("skeptical", "editor"):
        a = axes[name]
        assert a["custom"] is True
        assert "library" not in a


def test_steer_axes_flags_a_deployed_library_dial_and_not_a_user_custom(iso, tmp_path):
    _write_library(tmp_path, {"editor": {"pos": "p", "neg": "n", "max": 0.5, "source": "library"}})
    sub = _bare_substrate(custom={"skeptical": {"poles": ["skeptical", "neutral"], "max": 0.6},
                                  "editor": {"poles": ["editor", "neutral"], "max": 0.5}})
    axes = _axes_by_name(sub._steer("/steer/axes", {}))

    editor = axes["editor"]
    assert editor["library"] is True
    assert "custom" not in editor        # never tagged "yours"

    skeptical = axes["skeptical"]
    assert skeptical["custom"] is True
    assert "library" not in skeptical    # untouched -- still a genuine user dial


def test_steer_axes_built_in_axis_never_gets_a_library_tag_even_if_name_collides(iso, tmp_path):
    # studio_library.json is keyed by name only; a built-in AXES name has no reason to appear there, but
    # even if it somehow did, the BUILT-IN loop never consults lib_names at all -- only the custom loop
    # does -- so a built-in can never be tagged "library" through this path.
    _write_library(tmp_path, {"warm": {"pos": "p", "neg": "n", "max": 0.5}})
    warm = _axes_by_name(_bare_substrate()._steer("/steer/axes", {}))["warm"]
    assert "library" not in warm
    assert "custom" not in warm


def test_steer_axes_library_dial_still_gets_calibration_merged(iso, tmp_path):
    """A real deployment has BOTH files: studio_library.json (this feature) and dial_calibration.json
    (the pre-existing calibration merge) -- they must compose, not conflict."""
    _write_library(tmp_path, {"editor": {"pos": "p", "neg": "n", "max": 0.5}})
    with open(os.path.join(str(tmp_path), "dial_calibration.json"), "w", encoding="utf-8") as f:
        json.dump({"editor": {"usable_max": 0.5, "usable_range": [0.25, 0.5],
                              "derail_point": None, "works": True}}, f)
    sub = _bare_substrate(custom={"editor": {"poles": ["editor", "neutral"], "max": 0.5}})
    editor = _axes_by_name(sub._steer("/steer/axes", {}))["editor"]
    assert editor["library"] is True
    assert "custom" not in editor
    assert editor["max"] == 0.5
    assert editor["works"] is True
    assert editor["calibrated"] is True
