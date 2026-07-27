"""test_run_timeline_server -- GET /runs/<id>/timeline, the endpoint wiring for the RunEvent timeline.

Zero generation (run_timeline.timeline() is a pure reshape of what's already logged), so like /explain and
/export this route needs no substrate at all -- SUB can stay None. Drives the REAL clozn_server do_GET
handler with the no-socket object.__new__(H) trick (mirrors test_export_server.py's `_get()` harness).
run_timeline.py itself is unit-tested exhaustively against fixture dicts in test_run_timeline.py; this file
only proves the thin endpoint wiring: the route matches, a missing run is a clean 404, and a real run's
events come back over HTTP with their types (and ordering) intact.
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


def _get(path):
    """Drive do_GET without a socket; return (raw header block, raw body bytes) (mirrors test_export_server.py)."""
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(b"")
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": "0", "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"GET {path} HTTP/1.1", "HTTP/1.1", "GET"
    h.do_GET()
    head, _, body = h.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode("latin-1"), body


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate the run log; SUB stays None -- the endpoint must not need a substrate."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)
    return tmp_path


def test_timeline_missing_run_is_a_clean_404(iso):
    head, body = _get("/runs/run_nope/timeline")
    assert "404" in head
    assert json.loads(body) == {"error": "run not found"}


def test_timeline_needs_no_substrate_at_all(iso):
    """The whole point of a pure reshape: it must work with SUB is None (unlike /replay, /branch)."""
    rid = runlog.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey")
    assert cs.SUB is None
    head, body = _get(f"/runs/{rid}/timeline")
    assert "200" in head
    data = json.loads(body)
    assert data["run_id"] == rid
    assert "error" not in data
    assert [e["type"] for e in data["events"]] == ["run_started", "generation", "finished"]


def test_timeline_happy_path_returns_ordered_events_over_http(iso):
    rid = runlog.record(
        source="engine_chat", model="clozn-qwen",
        messages=[{"role": "user", "content": "explain gravity"}],
        response="Mass attracts mass.",
        trace={"tokens": ["Mass", " attracts", " mass", "."], "confidence": [0.95, 0.2, 0.9, 0.99],
               "alternatives": [[], [{"piece": " pulls", "prob": 0.4}], [], []]},
        behavior={"active_dials": {"concise": 0.5}},
        finish_reason="stop",
    )
    head, body = _get(f"/runs/{rid}/timeline")
    assert "200" in head
    data = json.loads(body)
    assert data["run_id"] == rid
    types = [e["type"] for e in data["events"]]
    assert types == ["run_started", "dials_applied", "generation", "hesitation", "finished"]
    # spot-check a couple of fields actually round-tripped over the wire (JSON null, floats intact)
    hes = data["events"][types.index("hesitation")]
    assert hes["token"] == " attracts"
    assert hes["alternatives"] == [{"piece": " pulls", "text": " pulls", "prob": 0.4, "logprob": -0.916291}]
    assert data["events"][-1]["finish_reason"] == "stop"
    assert data["events"][-1]["truncated"] is False


def test_timeline_errored_run_over_http_has_no_finished_event(iso):
    rid = runlog.record(source="cli", messages=[{"role": "user", "content": "q"}], response="",
                        error="boom")
    head, body = _get(f"/runs/{rid}/timeline")
    data = json.loads(body)
    types = [e["type"] for e in data["events"]]
    assert "error" in types
    assert "finished" not in types
