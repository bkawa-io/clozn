"""Model-free tests for clozn.behavior.registry -- the unified corrective-action registry."""
from __future__ import annotations

import pytest

from clozn import schemas
from clozn.behavior import registry
from clozn.replay.corrective import CORRECTION_PRESETS


class FakeSteer:
    """Duck-typed EngineSteer stand-in: only `.ceiling_for(name)` is ever called by registry.py."""

    def __init__(self, table=None, *, raises=False):
        self._table = dict(table or {})
        self._raises = raises

    def ceiling_for(self, name):
        if self._raises:
            raise RuntimeError("boom")
        if name in self._table:
            return self._table[name]
        return (0.6, False)


def test_registry_covers_every_correction_preset_once():
    doc = registry.build_registry()
    ids = [a["id"] for a in doc["actions"]]
    assert ids == list(CORRECTION_PRESETS)
    assert len(set(ids)) == len(ids)


def test_registry_validates_against_its_own_schema():
    schemas.validate(registry.build_registry())
    schemas.validate(registry.build_registry(steer=FakeSteer({"concise": (1.1, True)})))


def test_every_action_has_a_prompt_policy_backend():
    for action in registry.build_registry()["actions"]:
        types = [b["type"] for b in action["backends"]]
        assert "prompt_policy" in types


def test_actions_with_no_matching_dial_declare_control_vector_unsupported():
    doc = registry.build_registry()
    by_id = {a["id"]: a for a in doc["actions"]}
    for action_id in ("use-context", "ask-before-guessing", "preserve-formatting", "stop-repeating"):
        control_vector = next(b for b in by_id[action_id]["backends"]
                              if b["type"] != "prompt_policy" and "dial" not in b.get("parameters", {}))
        assert control_vector["type"] == "unsupported"
        assert "reason" in control_vector


def test_actions_with_a_matching_dial_declare_control_vector_when_no_steer_given():
    doc = registry.build_registry()
    by_id = {a["id"]: a for a in doc["actions"]}
    entry = next(b for b in by_id["less-verbose"]["backends"] if b["type"] == "control_vector")
    assert entry["parameters"]["dial"] == "concise"
    assert entry["qualification"] == "model_build_exact"
    assert "qualified" not in entry            # no live model -- never guessed


def test_qualified_field_is_only_present_with_a_live_steer():
    calibrated = registry.build_registry(steer=FakeSteer({"concise": (1.1, True)}))
    by_id = {a["id"]: a for a in calibrated["actions"]}
    entry = next(b for b in by_id["less-verbose"]["backends"] if b["type"] == "control_vector")
    assert entry["qualified"] is True
    assert entry["safe_bounds"]["enforced_ceiling"] == 1.1
    assert "reason" not in entry

    uncalibrated = registry.build_registry(steer=FakeSteer({}))    # falls through to (0.6, False)
    by_id = {a["id"]: a for a in uncalibrated["actions"]}
    entry = next(b for b in by_id["less-verbose"]["backends"] if b["type"] == "control_vector")
    assert entry["qualified"] is False
    assert "no calibration for this exact model" in entry["reason"]


def test_a_raising_steer_degrades_that_one_backend_not_the_whole_registry():
    doc = registry.build_registry(steer=FakeSteer(raises=True))
    by_id = {a["id"]: a for a in doc["actions"]}
    entry = next(b for b in by_id["less-verbose"]["backends"] if b["type"] == "control_vector")
    assert "qualified" not in entry             # degraded to the static shape, never guessed
    assert doc["actions"][0]["id"] == "less-verbose"    # the rest of the registry is unaffected


def test_every_action_declares_sampling_policy_unsupported():
    for action in registry.build_registry()["actions"]:
        sampling = next(b for b in action["backends"] if b["type"] == "sampling_policy"
                        or (b["type"] == "unsupported" and "sampling-policy" in b.get("reason", "")))
        assert sampling["type"] == "unsupported"


def test_action_ids_matches_the_registry():
    assert registry.action_ids() == [a["id"] for a in registry.build_registry()["actions"]]
