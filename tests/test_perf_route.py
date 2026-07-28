"""Focused HTTP wiring tests for GET /runs/<id>/performance (clozn/server/routes/performance.py).

Mirrors tests/test_run_diagnosis_server.py's fixture-handler pattern for the sibling /diagnosis route.
"""
from __future__ import annotations

import io
import json

import pytest

from clozn.server import app as cs
import clozn.runs.store as runlog


def _get(path):
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(b"")
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": "0", "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"GET {path} HTTP/1.1", "HTTP/1.1", "GET"
    h.do_GET()
    head, _, body = h.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode("latin-1"), json.loads(body)


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)


def test_route_module_registered_via_autoload_and_not_load_failures():
    from clozn.server.routes import _autoload, performance
    assert performance in cs._GET_ROUTES
    assert _autoload.LOAD_FAILURES == []


def test_performance_missing_run_is_a_clean_404(iso):
    head, body = _get("/runs/run_nope/performance")
    assert "404" in head
    assert body == {"error": "run not found"}


def test_performance_is_zero_generation_and_returns_a_valid_trace(iso):
    target = runlog.record(
        source="openai_api", client="pytest",
        messages=[{"role": "user", "content": "Explain gravity"}],
        response="Mass attracts mass.", finish_reason="stop",
        meta={"generation_duration_ms": 250.0, "generation_tokens_per_second": 30.0},
        started=1000.0, ended=1000.4,
    )
    assert cs.SUB is None
    head, body = _get(f"/runs/{target}/performance")

    assert "200" in head
    assert body["schema_version"] == "clozn.performance-trace.v1"
    assert body["run_id"] == target
    assert body["phases"] == [{"name": "decode", "owner": "clozn_worker", "duration_ns": 250_000_000}]
    assert len(body["diagnoses"]) == 7


def test_the_runs_id_fallback_still_wins_for_paths_performance_does_not_claim(iso):
    """/runs/<id>/performance is a specific suffix -- it must not shadow the generic /runs/<id> route for
    an id that merely happens to end differently."""
    target = runlog.record(source="openai_api", messages=[{"role": "user", "content": "hi"}],
                           response="hello", finish_reason="stop", started=1000.0, ended=1000.1)
    head, body = _get(f"/runs/{target}")
    assert "200" in head
    assert body.get("id") == target
