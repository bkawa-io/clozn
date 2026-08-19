"""test_rederive_server -- POST /runs/<id>/rederive, the endpoint wiring.

No model, no GPU: drives the REAL clozn_server do_POST handler (the object.__new__(H) no-socket trick --
mirrors test_receipts_server.py's own conventions) against an isolated runlog store, with a FAKE
substrate exposing only `.score_tokens` standing in for EngineSubstrate. rederive.py itself is
exhaustively unit-tested in test_rederive.py against fixture dicts; this file only proves the THIN
endpoint wiring: the route matches, a missing run is a clean 404, a substrate without score_tokens is a
clean 503 (unlike /receipt, this never needs .chat), and a real request's rederive comes back over HTTP
with the fields intact.
"""
from __future__ import annotations

import io
import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from clozn.server import app as cs   # noqa: E402
import clozn.runs.store as runlog                 # noqa: E402


class FakeScoreSub:
    """Exposes exactly `.score_tokens` -- what rederive.py needs from a substrate."""

    def __init__(self, tokens=None):
        self._tokens = tokens if tokens is not None else [
            {"id": 11, "piece": "Hello", "logprob": -0.1},
            {"id": 22, "piece": " there", "logprob": -0.2},
        ]
        self.calls = []

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        self.calls.append({"messages": messages, "continuation_ids": continuation_ids})
        return self._tokens


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


def _post_status(path, body_obj=None):
    """Like _post, but also returns the HTTP status (mirrors test_rewrite_route.py's / test_receipts_
    server.py's own _post_status) -- the engine-not-reachable-vs-bad-request tests below need the status
    code itself, not just the message."""
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    head, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(payload.decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", FakeScoreSub())
    return tmp_path


def _seed_run():
    return runlog.record(source="engine_chat", client="studio", model="clozn-engine", substrate="engine",
                         messages=[{"role": "user", "content": "hi"}], response="Hello there",
trace={"token_ids": [11, 22]})


def test_rederive_missing_run_is_a_clean_404(iso):
    out = _post("/runs/run_does_not_exist/rederive")
    assert out == {"error": "run not found"}


def test_rederive_needs_a_score_tokens_capable_substrate_503(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", None)
    rid = _seed_run()
    out = _post(f"/runs/{rid}/rederive")
    assert out == {"error": "rederive requires worker token scoring"}


def test_rederive_503_when_substrate_lacks_score_tokens(iso, monkeypatch):
    class NoScore:
        def chat(self, *a, **k):        # has .chat (like qwen) but no .score_tokens
            return "x"
    monkeypatch.setattr(cs, "SUB", NoScore())
    rid = _seed_run()
    out = _post(f"/runs/{rid}/rederive")
    assert "error" in out and "token scoring" in out["error"]


def test_rederive_happy_path_over_http(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/rederive")
    assert "error" not in out
    assert out["text"] == "Hello there"
    assert out["steps"] == [
        {"piece": "Hello", "token_id": 11, "logprob": -0.1, "conf": pytest.approx(math.exp(-0.1))},
        {"piece": " there", "token_id": 22, "logprob": -0.2, "conf": pytest.approx(math.exp(-0.2))},
    ]
    assert out["meta"]["retokenized"] is False


def test_rederive_scores_the_runs_own_messages_and_token_ids(iso, monkeypatch):
    fake = FakeScoreSub()
    monkeypatch.setattr(cs, "SUB", fake)
    rid = _seed_run()
    _post(f"/runs/{rid}/rederive")
    assert fake.calls[-1]["continuation_ids"] == [11, 22]
    assert fake.calls[-1]["messages"] == [{"role": "user", "content": "hi"}]


def test_rederive_failure_is_a_clean_500(iso, monkeypatch):
    class BoomSub:
        def score_tokens(self, *a, **k):
            raise RuntimeError("boom")
    monkeypatch.setattr(cs, "SUB", BoomSub())
    rid = _seed_run()
    out = _post(f"/runs/{rid}/rederive")
    assert "error" in out
    assert "rederive failed" in out["error"]


# ============================================================================== engine-not-reachable vs bad request
# (engine-down pressure test finding #3): rederive.rederive() (mirrors replay.py's contract) is documented
# to NEVER raise -- score_arm() catches whatever score_tokens() throws and returns ([], False), so
# BoomSub's plain RuntimeError above actually surfaces as the SAME ambiguous None a truly-unscoreable run
# would. The route now probes the substrate's own engine directly on that ambiguous path to tell "the
# engine is down" apart from "some other reason there was nothing to score".

class DownEngineScoreSub:
    """A substrate whose OWN engine is unreachable: .score_tokens() fails exactly like a live
    EngineSubstrate.score_tokens would against a dead C++ worker, and .engine.health() fails too (what
    ctx._engine_reachable() probes)."""

    def __init__(self):
        self.base = "http://127.0.0.1:8080"
        self.engine = self          # simplest double: this object IS its own "engine client"

    def health(self):
        raise OSError("connection refused")

    def score_tokens(self, *a, **k):
        raise OSError("connection refused")


def test_rederive_reports_engine_not_reachable_distinctly_from_a_bad_request(iso, monkeypatch):
    sub = DownEngineScoreSub()
    monkeypatch.setattr(cs, "SUB", sub)
    monkeypatch.setattr(cs, "ENGINE", sub)     # ctx._engine_unreachable_message() reads ENGINE.base
    rid = _seed_run()
    status, out = _post_status(f"/runs/{rid}/rederive")
    assert status == 502
    assert out == {"error": "engine not reachable at http://127.0.0.1:8080 -- is it running?"}


def test_rederive_bad_request_keeps_the_generic_500_when_the_engine_is_fine(iso, monkeypatch):
    """Control: the SAME ambiguous-None path (BoomSub above has no .engine at all), so
    ctx._engine_reachable() can't blame connectivity -- the original generic message survives unchanged."""
    class BoomSub:
        def score_tokens(self, *a, **k):
            raise RuntimeError("boom")
    monkeypatch.setattr(cs, "SUB", BoomSub())
    rid = _seed_run()
    status, out = _post_status(f"/runs/{rid}/rederive")
    assert status == 500
    assert out == {"error": "rederive failed (no continuation to score, or the "
                            "engine score call failed)"}
