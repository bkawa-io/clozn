"""tests/test_selective_generation_server.py -- the selective-generation ACTION half of calibration
backlog #10 wired into POST /v1/chat/completions (`clozn_selective`, BK decision 2026-07-22: opt-in,
default off). Companion to test_ask_band_server.py, which covers the always-on `clozn_policy` metadata;
this file covers the separate, explicitly-gated call that can replace the reply text
(generation_gateway.selective_generation_action).

Model-free: drives the REAL clozn_server do_POST handler with no socket (object.__new__(H)), isolated
runlog/cards/settings/eval stores, a fake substrate whose chat() fills a per-token trace. Mirrors
test_ask_band_server.py's conventions exactly.
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs                # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402
from clozn.eval import store as eval_store         # noqa: E402


import clozn.runs.store as runlog                  # noqa: E402


class FakeSteer:
    def __init__(self):
        self.strength = {}

    def active(self):
        return {}


class FakeMem:
    def __init__(self):
        self.memory_strength = 1.0
        self.rules = []
        self.prefix = None


MODEL = "fake-clozn-model"

ASK_STEPS = [{"piece": "The", "conf": 0.95}, {"piece": " answer", "conf": 0.6}]     # -> ask
ANSWER_STEPS = [{"piece": "The", "conf": 0.97}, {"piece": " answer", "conf": 0.9}]  # -> answer
ABSTAIN_STEPS = [{"piece": "The", "conf": 0.3}, {"piece": " answer", "conf": 0.1}]  # -> abstain

SAVED = {"model": MODEL, "set": "arith", "score": "min",
         "policy": {"answer_at": 0.8, "ask_at": 0.4}}


class TraceSub:
    """A qwen-shaped substrate whose chat() fills trace_out with a real per-token trace, mirroring
    test_ask_band_server.py's TraceSub."""
    name = "qwen"

    def __init__(self, steps=ASK_STEPS, reply="The answer."):
        self.memory = FakeMem()
        self._mem = self.memory
        self.steer = FakeSteer()
        self._steps = steps
        self._reply = reply
        self._run_meta = {"model_id": MODEL, "sampler_mode": "greedy",
                          "sampling": "greedy", "temperature": 0.0}

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        if trace_out is not None:
            trace_out.extend([dict(s) for s in self._steps])
        return self._reply

    def last_finish_reason(self):
        return "stop"

    def run_meta(self):
        return dict(self._run_meta)


def _dispatch(path, body_obj):
    raw = json.dumps(body_obj).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    return h.wfile.getvalue()


def _post(path, body_obj):
    _, _, payload = _dispatch(path, body_obj).partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(eval_store, "_PATH", str(tmp_path / "eval_report.json"))
    monkeypatch.delattr(eval_store, "load_profile", raising=False)
    monkeypatch.setattr(cs, "SUB", TraceSub())
    return tmp_path


def _body(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "what's the answer?"}], **extra}


# ============================================================================= OFF = byte-identical

def test_field_absent_is_byte_identical_to_annotate_only_baseline(iso):
    """With clozn_selective absent (default off) and a saved calibration that WOULD ask, the response
    must be identical to the always-on clozn_policy behavior: same keys, same reply, no new field at all."""
    eval_store.save(dict(SAVED))
    out = _post("/v1/chat/completions", _body())
    assert "clozn_selective_action" not in out
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "The answer."}
    assert out["clozn_policy"]["band"] == "ask"    # the always-on annotate signal is unaffected


def test_field_explicitly_false_is_also_byte_identical(iso):
    eval_store.save(dict(SAVED))
    out = _post("/v1/chat/completions", _body(clozn_selective=False))
    assert "clozn_selective_action" not in out
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "The answer."}


def test_server_setting_off_by_default_with_field_absent(iso):
    """No server-wide setting ever written -> selective_generation_enabled must read its documented
    default (False), not silently opt in."""
    eval_store.save(dict(SAVED))
    out = _post("/v1/chat/completions", _body())
    assert "clozn_selective_action" not in out


# ============================================================================= ON + ask

def test_on_ask_band_replaces_reply_and_preserves_raw(iso):
    eval_store.save(dict(SAVED))
    out = _post("/v1/chat/completions", _body(clozn_selective=True))
    action = out["clozn_selective_action"]
    assert action["applied"] is True
    assert action["band"] == "ask"
    assert action["raw_reply"] == "The answer."
    assert out["choices"][0]["message"]["content"] == action["reply"]
    assert out["choices"][0]["message"]["content"] != "The answer."


# ============================================================================= ON + abstain

def test_on_abstain_band_replaces_reply(iso, monkeypatch):
    eval_store.save(dict(SAVED))
    monkeypatch.setattr(cs, "SUB", TraceSub(steps=ABSTAIN_STEPS))
    out = _post("/v1/chat/completions", _body(clozn_selective=True))
    action = out["clozn_selective_action"]
    assert action["applied"] is True
    assert action["band"] == "abstain"
    assert action["raw_reply"] == "The answer."
    assert out["choices"][0]["message"]["content"] == action["reply"]


# ============================================================================= ON, no profile = fail closed

def test_on_but_no_calibration_fails_closed_and_leaves_reply_alone(iso):
    out = _post("/v1/chat/completions", _body(clozn_selective=True))
    action = out["clozn_selective_action"]
    assert action["applied"] is False
    assert "fail-closed" in action["reason"]
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "The answer."}


# ============================================================================= ON, answer band = no-op

def test_on_answer_band_leaves_reply_alone(iso, monkeypatch):
    eval_store.save(dict(SAVED))
    monkeypatch.setattr(cs, "SUB", TraceSub(steps=ANSWER_STEPS))
    out = _post("/v1/chat/completions", _body(clozn_selective=True))
    action = out["clozn_selective_action"]
    assert action["applied"] is False
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "The answer."}


# ============================================================================= validation

def test_clozn_selective_must_be_boolean(iso):
    out = _post("/v1/chat/completions", _body(clozn_selective="yes"))
    assert out["error"]["param"] == "clozn_selective"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
