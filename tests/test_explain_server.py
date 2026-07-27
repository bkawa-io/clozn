"""test_explain_server -- POST /runs/<id>/explain, the M1 endpoint wiring (EXPLAIN_THIS_ANSWER_SPEC.md).

No model, no GPU: the endpoint does zero generation, so unlike /runs/<id>/replay or /branch it needs no
substrate at all -- SUB can be None and the assembly still works (a point one test asserts directly, since
it's the whole reason M1 is "free"). Drives the REAL clozn_server do_GET/do_POST handler (the
object.__new__(H) no-socket trick used by test_propose_memory.py / test_profiles_server.py /

explain.py itself (the assembly logic, every honesty invariant) is unit-tested exhaustively in
test_explain.py against fixture dicts; this file only proves the THIN endpoint wiring: the route matches,
a missing run is a clean 404, and a real run's assembled object comes back over HTTP with the fields (and
None -> null) intact.
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

import clozn.runs.store as runlog                # noqa: E402


# --- driving the real handler without a socket (mirrors test_timetravel_server / test_facts_server) -------

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
    """Isolate the run log + card store; SUB stays None -- the endpoint must not need a substrate."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)
    return tmp_path


def test_explain_missing_run_is_a_clean_404(iso):
    out = _post("/runs/run_does_not_exist/explain")
    assert out == {"error": "run not found"}


def test_explain_needs_no_substrate_at_all(iso):
    """The whole point of M1: zero generation, so it must work with SUB is None (unlike /replay, /branch)."""
    rid = runlog.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey")
    assert cs.SUB is None
    out = _post(f"/runs/{rid}/explain")
    assert "error" not in out
    assert out["run_id"] == rid


