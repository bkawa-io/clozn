"""Route coverage for GET /runs/<id>/diagnosis-findings (clozn/server/routes/diagnosis_findings.py) --
D1 findings + D2 narrative served together. Mirrors tests/test_span_addresses_route.py's Handler stub and
autoload-registration pattern.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import clozn.runs.store as runlog  # noqa: E402
from clozn.server.routes import diagnosis_findings as route  # noqa: E402


class Handler:
    def __init__(self, path="/"):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _run(run_id="run_x", **over):
    out = {"id": run_id, "messages": [{"role": "user", "content": "hi"}], "response": "ok",
          "finish_reason": "stop"}
    out.update(over)
    return out


def test_route_returns_findings_and_narrative_together(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/diagnosis-findings")
    assert route.try_get(h, f"/runs/{run['id']}/diagnosis-findings") is True
    assert h.status == 200
    assert h.body["findings"]["schema_version"] == "clozn.diagnosis-findings.v1"
    assert h.body["narrative"]["schema_version"] == "clozn.diagnosis-narrative.v1"
    assert h.body["findings"]["run_id"] == run["id"]
    assert h.body["narrative"]["run_id"] == run["id"]
    assert len(h.body["findings"]["findings"]) == 12


def test_route_404_when_run_not_found(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: None)
    h = Handler("/runs/missing/diagnosis-findings")
    assert route.try_get(h, "/runs/missing/diagnosis-findings") is True
    assert h.status == 404


def test_route_with_comparison_run_via_query_param(monkeypatch):
    run_a = _run("run_a", meta={"temperature": 0.0})
    run_b = _run("run_b", meta={"temperature": 0.8})
    runs = {"run_a": run_a, "run_b": run_b}
    monkeypatch.setattr(runlog, "get_run", lambda rid: runs.get(rid))

    h = Handler(f"/runs/run_b/diagnosis-findings?compare=run_a")
    assert route.try_get(h, "/runs/run_b/diagnosis-findings") is True
    assert h.status == 200
    assert h.body["findings"]["comparison_run_id"] == "run_a"
    assert h.body["narrative"]["comparison_run_id"] == "run_a"
    assert h.body["narrative"]["comparison_available"] is True
    assert h.body["narrative"]["registers"]["observed_changes"]


def test_route_404_when_comparison_run_not_found(monkeypatch):
    run_b = _run("run_b")
    monkeypatch.setattr(runlog, "get_run", lambda rid: run_b if rid == "run_b" else None)

    h = Handler("/runs/run_b/diagnosis-findings?compare=nope")
    assert route.try_get(h, "/runs/run_b/diagnosis-findings") is True
    assert h.status == 404
    assert "nope" in h.body["error"]


def test_route_suppress_query_param_flips_status_without_evaluating(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/diagnosis-findings?suppress=R03,R07")
    route.try_get(h, f"/runs/{run['id']}/diagnosis-findings")
    findings_by_id = {f["rule_id"]: f for f in h.body["findings"]["findings"]}
    assert findings_by_id["R03"]["status"] == "suppressed"
    assert findings_by_id["R07"]["status"] == "suppressed"
    assert h.body["findings"]["suppressed_rule_ids"] == ["R03", "R07"]


def test_route_does_not_match_unrelated_paths():
    h = Handler("/runs/x/diagnosis")   # the OLD route -- must not be intercepted by this one
    assert route.try_get(h, "/runs/x/diagnosis") is False
    assert route.try_get(h, "/runs/x") is False
    assert route.try_get(h, "/other") is False


def test_route_registered_before_the_runs_fallback():
    from clozn.server import app as server
    assert route in server._GET_ROUTES
    assert server._GET_ROUTES.index(route) < server._GET_ROUTES.index(server._runs_fallback_routes)


def test_findings_and_narrative_agree_on_ranked_rule_ids(monkeypatch):
    """The route computes findings ONCE and hands the SAME document to narrate() -- every rule_id the
    narrative ranks must appear, with status=='finding', in the findings array returned alongside it."""
    run = _run(finish_reason="length")
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/diagnosis-findings")
    route.try_get(h, f"/runs/{run['id']}/diagnosis-findings")
    findings_by_id = {f["rule_id"]: f for f in h.body["findings"]["findings"]}
    for entry in h.body["narrative"]["registers"]["measured_effects"]:
        assert findings_by_id[entry["rule_id"]]["status"] == "finding"
