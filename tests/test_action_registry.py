"""Model-free tests for clozn.behavior.registry -- the unified corrective-action registry."""
from __future__ import annotations

import pytest

from clozn import schemas
from clozn.behavior import registry
from clozn.replay.corrective import CORRECTION_PRESETS


def test_registry_covers_every_correction_preset_once():
    doc = registry.build_registry()
    ids = [a["id"] for a in doc["actions"]]
    assert ids == list(CORRECTION_PRESETS)
    assert len(set(ids)) == len(ids)
    for action in doc["actions"]:
        # Durable session/profile scoping was retired -- see docs/CAPABILITIES.md.
        assert action["scopes"] == ["once"]
        assert len(set(action["scopes"])) == len(action["scopes"])


def test_registry_validates_against_its_own_schema():
    schemas.validate(registry.build_registry())


def test_every_action_has_a_prompt_policy_backend():
    for action in registry.build_registry()["actions"]:
        types = [b["type"] for b in action["backends"]]
        assert "prompt_policy" in types


def test_every_action_declares_sampling_policy_unsupported():
    for action in registry.build_registry()["actions"]:
        sampling = next(b for b in action["backends"] if b["type"] == "sampling_policy"
                        or (b["type"] == "unsupported" and "sampling-policy" in b.get("reason", "")))
        assert sampling["type"] == "unsupported"


def test_action_ids_matches_the_registry():
    assert registry.action_ids() == [a["id"] for a in registry.build_registry()["actions"]]
