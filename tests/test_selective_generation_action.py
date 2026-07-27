"""tests/test_selective_generation_action.py -- generation_gateway.selective_generation_action /
selective_generation_enabled: the OPT-IN, DEFAULT-OFF action half of calibration backlog #10 (BK decision
2026-07-22: "abstain/ask may become an ACTION, but OPT-IN, DEFAULT OFF"). `policy_signal` (see
test_ask_band_signal.py) remains the always-on, metadata-only ANNOTATE half; this file covers the separate,
explicitly-gated call that can replace the reply text.

Model-free: no engine, no HTTP. Isolated per test via monkeypatching eval_store._PATH (mirrors
test_ask_band_signal.py's convention) -- never touches the real ~/.clozn.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clozn.eval import store as eval_store                     # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402

from clozn.server import generation_gateway as gw               # noqa: E402


MODEL = "fake-clozn-model"
SAVED = {"model": MODEL, "set": "arith", "score": "min",
         "policy": {"answer_at": 0.8, "ask_at": 0.4}}

ASK_STEPS = [{"piece": "The", "conf": 0.95}, {"piece": " answer", "conf": 0.6}]     # min 0.6 -> ask
ANSWER_STEPS = [{"piece": "The", "conf": 0.97}, {"piece": " answer", "conf": 0.9}]  # min 0.9 -> answer
ABSTAIN_STEPS = [{"piece": "The", "conf": 0.2}, {"piece": " answer", "conf": 0.1}]  # min 0.1 -> abstain

RAW_REPLY = "The answer is 42."


@pytest.fixture(autouse=True)
def legacy_loader(monkeypatch):
    monkeypatch.delattr(eval_store, "load_profile", raising=False)


def _save(tmp_path, monkeypatch, payload):
    path = str(tmp_path / "eval_report.json")
    monkeypatch.setattr(eval_store, "_PATH", path)
    eval_store.save(dict(payload), path)


# ================================================================================ selective_generation_enabled

def test_enabled_true_when_request_field_is_truthy():
    assert gw.selective_generation_enabled({"clozn_selective": True}) is True


def test_enabled_false_when_request_field_is_explicitly_false():
    assert gw.selective_generation_enabled({"clozn_selective": False}) is False


def test_enabled_false_when_field_and_setting_both_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    assert gw.selective_generation_enabled({}) is False
    assert gw.selective_generation_enabled(None) is False


def test_enabled_true_via_server_setting_when_field_absent(monkeypatch, tmp_path):
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    clozn_settings.set_setting(gw.SELECTIVE_SETTING, True)
    assert gw.selective_generation_enabled({}) is True


def test_explicit_false_field_wins_over_server_setting_on(monkeypatch, tmp_path):
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    clozn_settings.set_setting(gw.SELECTIVE_SETTING, True)
    assert gw.selective_generation_enabled({"clozn_selective": False}) is False


# ================================================================================ off = identical to baseline

def test_off_never_touches_the_reply_or_looks_at_calibration(tmp_path, monkeypatch):
    """opted_in=False must be a pure pass-through -- no eval_store lookup at all, and the caller's raw
    reply is reported back untouched, never replaced."""
    monkeypatch.setattr(eval_store, "_PATH", str(tmp_path / "nope.json"))  # no calibration exists either way
    out = gw.selective_generation_action(RAW_REPLY, ASK_STEPS, MODEL, opted_in=False)
    assert out == {"applied": False,
                   "reason": "selective-generation action is opt-in and was not requested",
                   "raw_reply": RAW_REPLY}


# ================================================================================ on + ask

def test_on_ask_band_replaces_reply_and_preserves_raw(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, SAVED)
    out = gw.selective_generation_action(RAW_REPLY, ASK_STEPS, MODEL, opted_in=True)
    assert out["applied"] is True
    assert out["band"] == "ask"
    assert out["raw_reply"] == RAW_REPLY          # never destroyed
    assert out["reply"] != RAW_REPLY
    assert "clarify" in out["reply"].lower() or "rephrase" in out["reply"].lower()
    assert out["verdict"]["band"] == "ask"
    assert out["verdict"]["score"] == 0.6
    assert out["verdict"]["calibration_model"] == MODEL
    assert out["verdict"]["calibration_task"] == "arith"


# ================================================================================ on + abstain

def test_on_abstain_band_replaces_reply_with_a_stronger_refusal(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, SAVED)
    out = gw.selective_generation_action(RAW_REPLY, ABSTAIN_STEPS, MODEL, opted_in=True)
    assert out["applied"] is True
    assert out["band"] == "abstain"
    assert out["raw_reply"] == RAW_REPLY
    assert out["reply"] != RAW_REPLY
    assert "won't" in out["reply"] or "can't" in out["reply"]
    assert out["verdict"]["band"] == "abstain"
    assert out["verdict"]["score"] == 0.1


# ================================================================================ on + answer band = no-op

def test_on_answer_band_does_not_apply(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, SAVED)
    out = gw.selective_generation_action(RAW_REPLY, ANSWER_STEPS, MODEL, opted_in=True)
    assert out["applied"] is False
    assert out["raw_reply"] == RAW_REPLY
    assert "answer" in out["reason"]


# ================================================================================ on, but no profile = fail-closed

def test_on_but_no_calibration_saved_fails_closed_annotate_only(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_store, "_PATH", str(tmp_path / "nope.json"))
    out = gw.selective_generation_action(RAW_REPLY, ASK_STEPS, MODEL, opted_in=True)
    assert out["applied"] is False
    assert out["raw_reply"] == RAW_REPLY
    assert "fail-closed" in out["reason"]
    assert "annotate-only" in out["reason"]


def test_on_but_model_mismatch_fails_closed(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, {**SAVED, "model": "a-completely-different-model"})
    out = gw.selective_generation_action(RAW_REPLY, ASK_STEPS, MODEL, opted_in=True)
    assert out["applied"] is False
    assert "fail-closed" in out["reason"]
    assert out["raw_reply"] == RAW_REPLY


def test_never_raises_on_malformed_saved_report(tmp_path, monkeypatch):
    path = str(tmp_path / "eval_report.json")
    monkeypatch.setattr(eval_store, "_PATH", path)
    eval_store.save({"policy": "not-a-dict"}, path)
    out = gw.selective_generation_action(RAW_REPLY, ASK_STEPS, MODEL, opted_in=True)
    assert out["applied"] is False
    assert out["raw_reply"] == RAW_REPLY


# ================================================================================ caveat text present

def test_caveat_text_present_on_every_fired_action(tmp_path, monkeypatch):
    """The token-probability + hard-tail honesty caveats (docs/RESEARCH_ROADMAP.md's Killed 'white-box
    risk controller advantage' entry) must ride the action output, both as its own field and inline in the
    replacement text, whenever the action actually fires."""
    _save(tmp_path, monkeypatch, SAVED)
    for steps, band in ((ASK_STEPS, "ask"), (ABSTAIN_STEPS, "abstain")):
        out = gw.selective_generation_action(RAW_REPLY, steps, MODEL, opted_in=True)
        assert out["applied"] is True and out["band"] == band
        assert "caveat" in out and "token-probability" in out["caveat"].lower()
        assert "hard tail" in out["caveat"].lower()
        assert "white-box" in out["caveat"].lower()
        assert out["caveat"] in out["reply"]


def test_indexed_profile_selection_is_exact_and_never_falls_back(monkeypatch):
    """selective_generation_action shares _policy_verdict's provenance gate with policy_signal -- the
    task-indexed store's exact-match-only selection applies here too."""
    profiles = {
        "arith": dict(SAVED),
        "medical": {**SAVED, "set": "medical", "policy": {"answer_at": 0.95, "ask_at": 0.7}},
    }

    def load_profile(model, task):
        return profiles.get(task)

    monkeypatch.setattr(eval_store, "load_profile", load_profile, raising=False)
    out = gw.selective_generation_action(RAW_REPLY, ASK_STEPS, MODEL, task="medical", opted_in=True)
    assert out["applied"] is True
    assert out["band"] == "abstain"          # min-conf 0.6 sits below medical's tighter ask_at=0.7
    assert out["verdict"]["calibration_task"] == "medical"

    missing = gw.selective_generation_action(RAW_REPLY, ASK_STEPS, MODEL, task="unknown", opted_in=True)
    assert missing["applied"] is False
    assert "fail-closed" in missing["reason"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
