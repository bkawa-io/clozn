"""Regression coverage for retiring broad assistant-behavior policy management (docs/CAPABILITIES.md):
Clozn is a causal debugger, not a production policy engine. "Measurement and explicit intervention stay.
Ambient policy goes."

Retired: the persisted server-wide guard default (`generation_guard.get/set_persisted_guard_spec`,
`GET/POST /guard/mode`), and the selective-generation ANSWER-REWRITING action
(`generation_gateway.selective_generation_action`/`selective_generation_enabled`, request field
`clozn_selective`, response field `clozn_selective_action`).

Kept (covered elsewhere, not re-tested here): the explicit, request-local `clozn_guard` intervention
(tests/test_generation_guard.py, tests/test_generation_guard_server.py) and the always-on
`clozn_policy` calibration-evidence annotation (tests/test_ask_band_signal.py,
tests/test_ask_band_server.py).
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from clozn.server import app as cs                       # noqa: E402
from clozn.server import generation_guard as gg           # noqa: E402
from clozn.server import generation_gateway as gw         # noqa: E402
import clozn.runs.store as runlog                          # noqa: E402
import clozn.settings as clozn_settings                    # noqa: E402


MODEL = "fake-clozn-model"


class TraceSub:
    """A qwen-shaped substrate that never touches the guard/policy machinery -- proves the ordinary
    chat() pipeline runs completely untouched (mirrors test_generation_guard_server.py's TraceSub)."""
    name = "qwen"

    def __init__(self, reply="An ordinary ungated reply."):
        class _Mem:
            memory_strength = 1.0
            rules: list = []
            prefix = None
        self.memory = _Mem()
        self._mem = self.memory
        self._reply = reply
        self._run_meta = {"model_id": MODEL, "sampler_mode": "greedy", "sampling": "greedy",
                          "temperature": 0.0}

        class _Steer:
            strength: dict = {}

            def active(self):
                return {}
        self.steer = _Steer()

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        if trace_out is not None:
            trace_out.extend([{"piece": "ordinary", "conf": 0.9}])
        return self._reply

    def last_finish_reason(self):
        return "stop"

    def run_meta(self):
        return dict(self._run_meta)


def _dispatch(method, path, body_obj=None):
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    if body_obj is None:
        h.rfile = io.BytesIO(b"")
        h.headers = {"Content-Length": "0", "User-Agent": "pytest"}
    else:
        raw = json.dumps(body_obj).encode("utf-8")
        h.rfile = io.BytesIO(raw)
        h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.wfile = io.BytesIO()
    h.requestline, h.request_version, h.command = f"{method} {path} HTTP/1.1", "HTTP/1.1", method
    getattr(h, f"do_{method}")()
    raw = h.wfile.getvalue()
    status = int(raw.split(b" ", 2)[1])
    _, _, payload = raw.partition(b"\r\n\r\n")
    body = json.loads(payload.decode("utf-8")) if payload.strip() else {}
    return status, body


def _post(path, body_obj):
    return _dispatch("POST", path, body_obj)


def _get(path):
    return _dispatch("GET", path)


def _body(**extra):
    return {"model": MODEL, "messages": [{"role": "user", "content": "tell me something"}], **extra}


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    return tmp_path


# ---------------------------------------------------------------------- /guard/mode route is fully gone

def test_guard_mode_route_no_longer_exists(iso):
    """The whole module was deleted, not stubbed to 410. GET falls through to the server's ordinary
    unknown-route 404. POST falls through to the same generic no-active-substrate 409 any other
    made-up write path gets (clozn/server/app.py's fallback tries the active substrate's own
    per-action dispatch before giving up) -- there is no guard-specific handling left at all."""
    status, _ = _get("/guard/mode")
    assert status == 404
    status, body = _post("/guard/mode", {"enabled": True, "concepts": ["violence"]})
    assert status == 409
    assert "/guard/mode" in body["error"]


def test_persisted_guard_spec_functions_no_longer_exist():
    assert not hasattr(gg, "get_persisted_guard_spec")
    assert not hasattr(gg, "set_persisted_guard_spec")
    assert not hasattr(gg, "GUARD_SETTING")


# -------------------------------------------------------------- stale persisted guard default is inert

def test_stale_generation_guard_setting_never_opts_a_request_in(iso, monkeypatch):
    """A settings file left over from an old Clozn build that still carries the retired
    `generation_guard` key must be completely inert end-to-end: a request that omits `clozn_guard`
    takes the ordinary, unguarded path through the real /v1/chat/completions handler."""
    clozn_settings.set_setting("generation_guard", {"enabled": True, "concepts": ["violence"]})
    monkeypatch.setattr(cs, "SUB", TraceSub())
    status, out = _post("/v1/chat/completions", _body())
    assert status == 200
    assert "clozn_guard_receipt" not in out
    assert out["choices"][0]["message"]["content"] == "An ordinary ungated reply."


# ------------------------------------------------------- selective-generation ACTION is fully removed

def test_selective_generation_functions_no_longer_exist():
    assert not hasattr(gw, "selective_generation_action")
    assert not hasattr(gw, "selective_generation_enabled")
    assert not hasattr(gw, "SELECTIVE_FIELD")
    assert not hasattr(gw, "SELECTIVE_SETTING")
    # The shared measurement/annotation surface must still be there.
    assert hasattr(gw, "policy_signal")
    assert hasattr(gw, "policy_meta_for_run")
    assert hasattr(gw, "policy_verdict_and_signal")


def test_clozn_selective_request_field_is_honestly_rejected_not_silently_ignored(iso, monkeypatch):
    """`clozn_selective: true` on the request used to be able to replace the model's actual reply text
    with a clarify/abstain message. It no longer does anything -- and, per this module's no-silent-pass
    policy (clozn/server/openai_compat.py: "raises ... for every behavior-bearing field the runtime
    cannot honor"), a caller that still sends it gets an honest 400 telling them so, rather than a 200
    that quietly drops the field and leaves them assuming it took effect."""
    monkeypatch.setattr(cs, "SUB", TraceSub())
    status, out = _post("/v1/chat/completions", _body(clozn_selective=True))
    assert status == 400
    assert out["error"]["param"] == "clozn_selective"
    assert out["error"]["code"] == "unsupported_parameter"


def test_stale_selective_generation_setting_never_rewrites_the_reply(iso, monkeypatch):
    """A settings file carrying the retired `selective_generation` key (an old server-wide opt-in
    default) must be completely inert -- the reply is never rewritten and no clozn_selective_action
    field is ever attached, regardless of what is on disk."""
    clozn_settings.set_setting("selective_generation", True)
    monkeypatch.setattr(cs, "SUB", TraceSub())
    status, out = _post("/v1/chat/completions", _body())
    assert status == 200
    assert out["choices"][0]["message"]["content"] == "An ordinary ungated reply."
    assert "clozn_selective_action" not in out


# -------------------------------------------------------------------- historical runs remain readable

def test_historical_run_with_retired_policy_fields_remains_readable(iso):
    """A run recorded before this retirement may carry meta.clozn_policy and meta.clozn_guard_receipt
    (and, from before THAT cut, meta.clozn_selective_action). Reading it back today must not crash and
    must preserve that historical evidence -- only new runs stop producing clozn_selective_action."""
    run_id = runlog.record(
        source="engine_chat", client="studio", model="alpha", substrate="engine",
        messages=[{"role": "user", "content": "hi"}], response="hello there",
        final_prompt="<user>hi</user>",
        meta={
            "clozn_policy": {"band": "ask", "score": 0.4, "score_aggregate": 0.4,
                             "answer_at": 0.8, "ask_at": 0.3},
            "clozn_guard_receipt": {"fired": True, "concepts": ["violence"], "cap_reached": False},
            "clozn_selective_action": {"applied": True, "band": "ask", "reply": "clarify?"},
        },
    )
    assert run_id
    run = runlog.get_run(run_id)
    assert run is not None
    assert run["meta"]["clozn_policy"]["band"] == "ask"
    assert run["meta"]["clozn_guard_receipt"]["fired"] is True
    assert run["meta"]["clozn_selective_action"]["applied"] is True
