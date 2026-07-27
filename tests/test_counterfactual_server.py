"""test_counterfactual_server -- POST /runs/<id>/counterfactual, the M3 endpoint wiring
(EXPLAIN_THIS_ANSWER_SPEC.md).

No model, no GPU: drives the REAL clozn_server do_POST handler (the object.__new__(H) no-socket trick used
(both-arms-greedy generation, the coherence axis, the unapplied-override guard, dose_sweep) is exhaustively
unit-tested in test_counterfactual.py against fixture dicts; this file only proves the THIN endpoint
wiring: the route matches, a missing run is a clean 404, no substrate is a clean 503 (both arms
regenerate, so -- like /runs/<id>/receipt -- it needs the live substrate), a missing/malformed
behavior_overrides body is a clean 400, and one real request's counterfactual comes back over HTTP with
its fields (and Python True -> JSON true) intact.
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

from clozn.server import app as cs   # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402


from clozn import receipts             # noqa: E402
import clozn.runs.store as runlog                 # noqa: E402


# --- a fake substrate: deterministic chat() keyed on the "warm" dial's exact value ----------------------

class FakeSteer:
    def __init__(self, strength=None):
        self.strength = dict(strength or {})

    def set(self, name, value):
        self.strength[str(name)] = float(value)

    def clear(self):
        self.strength = {}

    def active(self):
        return {k: v for k, v in self.strength.items() if v}


class FakeMem:
    def __init__(self, strength=1.0):
        self.memory_strength = float(strength)
        self.rules = []
        self.prefix = "PFX"


class FakeSub:
    name = "qwen"

    def __init__(self, mem=None, steer=None):
        self.memory = mem if mem is not None else FakeMem()
        self._mem = self.memory
        self.steer = steer if steer is not None else FakeSteer()
        self.calls = 0

    def chat(self, messages, max_new=256, sample=True):
        self.calls += 1
        warm = float(self.steer.strength.get("warm", 0.0) or 0.0)
        return f"A plain reply (warmth={warm:.2f})."


# --- driving the real handler without a socket (mirrors test_receipts_server / test_explain_server) ------

def _dispatch(method, path, body_obj=None):
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"{method} {path} HTTP/1.1", "HTTP/1.1", method
    getattr(h, f"do_{method}")()
    _, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


def _post(path, body_obj=None):
    return _dispatch("POST", path, body_obj)


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate the run/card/settings stores; SUB starts as a FakeSub (tests that want the 503 path
    override it to None explicitly)."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(cs, "SUB", FakeSub(mem=FakeMem(1.0), steer=FakeSteer({"warm": 0.2})))
    return tmp_path


def _seed_run():
    return runlog.record(source="studio_chat", client="studio", model="clozn-qwen", substrate="QwenSubstrate",
                         messages=[{"role": "user", "content": "hi there"}],
                         response="THE SAMPLED reply -- must never come back as a baseline",
                         behavior={"active_dials": {"warm": 0.2}})


def test_counterfactual_missing_run_is_a_clean_404(iso):
    out = _post("/runs/run_does_not_exist/counterfactual", {"behavior_overrides": {"warm": 0.9}})
    assert out == {"error": "run not found"}


def test_counterfactual_needs_the_substrate_503(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", None)
    rid = _seed_run()
    out = _post(f"/runs/{rid}/counterfactual", {"behavior_overrides": {"warm": 0.9}})
    assert out == {"error": "counterfactual requires a ready product model worker"}


def test_counterfactual_rejects_a_missing_behavior_overrides_with_400(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/counterfactual", {})
    assert "error" in out


def test_counterfactual_rejects_a_malformed_behavior_overrides_with_400(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/counterfactual", {"behavior_overrides": "not-a-dict"})
    assert "error" in out
    out2 = _post(f"/runs/{rid}/counterfactual", {"behavior_overrides": {}})
    assert "error" in out2


def test_counterfactual_happy_path_over_http(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/counterfactual", {"behavior_overrides": {"warm": 0.9}})
    assert "error" not in out
    assert out["causal_verified"] is True                  # JSON true, round-tripped from Python True
    assert out["has_effect"] is True
    assert out["overrides_applied"] == {"warm": 0.9}
    assert out["baseline_reply"] == "A plain reply (warmth=0.20)."         # the substrate's LIVE dial (0.2)
    assert out["counterfactual_reply"] == "A plain reply (warmth=0.90)."   # the override
    assert out["delta"] == receipts.receipt_metrics(out["baseline_reply"], out["counterfactual_reply"])
    assert out["coherence"] == {"degenerate": False, "reason": ""}
    # the stored sampled reply never shows up as either arm, and the receipt says so
    stored = runlog.get_run(rid)["response"]
    assert stored not in (out["baseline_reply"], out["counterfactual_reply"])
    assert "sampled" in out["note"].lower() and "baseline" in out["note"].lower()
    assert "cost_note" in out
