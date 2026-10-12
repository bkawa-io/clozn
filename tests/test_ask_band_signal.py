"""generation_gateway.policy_signal turns a completed reply trace and an exact calibration profile into
just-completed reply's raw per-token trace + the saved calibration (clozn eval --save) into the metadata-
only 'ask band' signal (calibration backlog: "a retrieval/clarify action wired to the policy's ask band").
Both /v1/chat/completions (openai.py, non-streaming) and the SSE stream (sse.py) call this SAME function,
so covering it here covers the one piece of logic both routes share.

Pure-ish: the only I/O is the isolated calibration profile registry.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clozn.eval import store as eval_store                     # noqa: E402
from clozn.server import generation_gateway as gw              # noqa: E402


MODEL = "fake-clozn-model"
SAVED = {"model": MODEL, "set": "arith", "score": "min",
         "policy": {"answer_at": 0.8, "ask_at": 0.4}}

ASK_STEPS = [{"piece": "The", "conf": 0.95}, {"piece": " answer", "conf": 0.6}]     # min 0.6 -> ask
ANSWER_STEPS = [{"piece": "The", "conf": 0.97}, {"piece": " answer", "conf": 0.9}]  # min 0.9 -> answer
ABSTAIN_STEPS = [{"piece": "The", "conf": 0.2}, {"piece": " answer", "conf": 0.1}]  # min 0.1 -> abstain


def _save(tmp_path, monkeypatch, payload):
    path = str(tmp_path / "calibration_profiles.json")
    monkeypatch.setattr(eval_store, "_PATH", path)
    eval_store.save_profile(dict(payload), task=payload.get("task") or payload.get("set") or "general")


def test_ask_band_returns_the_signal(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, SAVED)
    out = gw.policy_signal(ASK_STEPS, MODEL)
    assert out["band"] == "ask"
    assert out["score"] == 0.6
    assert out["answer_at"] == 0.8 and out["ask_at"] == 0.4
    assert out["score_aggregate"] == "min"
    assert out["calibration_task"] == "arith"
    assert out["calibration_model"] == MODEL
    assert "note" in out and "ask" in out["note"]


def test_abstain_band_returns_a_stronger_signal(tmp_path, monkeypatch):
    """Below the ask band, the model is likely wrong -- the note must say so, more strongly than the ask
    note, and must still carry the same score/threshold fields."""
    _save(tmp_path, monkeypatch, SAVED)
    out = gw.policy_signal(ABSTAIN_STEPS, MODEL)
    assert out["band"] == "abstain"
    assert out["score"] == 0.1
    assert out["answer_at"] == 0.8 and out["ask_at"] == 0.4
    assert out["score_aggregate"] == "min"
    assert "note" in out and "abstain" in out["note"] and "likely wrong" in out["note"]


def test_answer_band_returns_none(tmp_path, monkeypatch):
    """A confident answer says nothing this endpoint needs to add."""
    _save(tmp_path, monkeypatch, SAVED)
    assert gw.policy_signal(ANSWER_STEPS, MODEL) is None


def test_no_calibration_saved_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_store, "_PATH", str(tmp_path / "nope.json"))
    assert gw.policy_signal(ASK_STEPS, MODEL) is None


def test_model_mismatch_returns_none(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, SAVED)
    assert gw.policy_signal(ASK_STEPS, "a-completely-different-model") is None


def test_no_trace_returns_none(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, SAVED)
    assert gw.policy_signal([], MODEL) is None
    assert gw.policy_signal(None, MODEL) is None


def test_never_raises_on_malformed_saved_report(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_store, "load_profile", lambda *_args, **_kwargs: {"policy": "not-a-dict"})
    assert gw.policy_signal(ASK_STEPS, MODEL) is None


def test_indexed_profile_selection_is_exact_and_never_falls_back(monkeypatch):
    calls = []
    profiles = {
        "arith": dict(SAVED),
        "medical": {**SAVED, "set": "medical", "policy": {"answer_at": 0.95, "ask_at": 0.7}},
    }

    def load_profile(model, task):
        calls.append((model, task))
        return profiles.get(task)

    monkeypatch.setattr(eval_store, "load_profile", load_profile, raising=False)

    out = gw.policy_signal(ASK_STEPS, MODEL, task="medical")
    assert out["band"] == "abstain"
    assert out["calibration_task"] == "medical"
    assert out["calibration_model"] == MODEL
    assert gw.policy_signal(ASK_STEPS, MODEL, task="unknown") is None
    assert calls == [(MODEL, "medical"), (MODEL, "unknown")]


def test_indexed_no_task_asks_store_for_newest_exact_model(monkeypatch):
    calls = []

    def load_profile(model, task):
        calls.append((model, task))
        return dict(SAVED)

    monkeypatch.setattr(eval_store, "load_profile", load_profile, raising=False)
    out = gw.policy_signal(ASK_STEPS, MODEL)
    assert out["band"] == "ask" and out["calibration_task"] == "arith"
    assert calls == [(MODEL, None)]


def test_explicit_task_requires_an_exact_profile(tmp_path, monkeypatch):
    _save(tmp_path, monkeypatch, SAVED)
    assert gw.policy_signal(ASK_STEPS, MODEL, task="arith")["band"] == "ask"
    assert gw.policy_signal(ASK_STEPS, MODEL, task="medical") is None
