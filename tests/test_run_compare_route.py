"""test_run_compare_route.py -- model-free unit tests for clozn/server/routes/run_compare.py (GET
/runs/compare, agent roadmap feature 10). Mirrors tests/test_corrective_retry_route.py's Handler
duck-type and clozn/analysis/test_model_diff.py's FakeHandler: routes only ever touch `.path` and
`._json(...)`, so no real HTTP server/socket is needed.
"""
from __future__ import annotations

import clozn.runs.store as runlog
from clozn.server.routes import run_compare as route


class Handler:
    def __init__(self, path):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _run(rid, **kw):
    rec = {"id": rid, "identity": {}, "meta": {}, "messages": [], "response": "", "context_receipt": {},
          "output_contract": {}, "trace": {}}
    rec.update(kw)
    return rec


def test_autoload_marker_is_set():
    assert route.CLOZN_ROUTE_AUTOLOAD is True


def test_unrelated_path_is_not_handled():
    h = Handler("/runs/latest")
    assert route.try_get(h, "/runs/latest") is False
    assert h.status is None


def test_missing_query_params_yield_400():
    h = Handler("/runs/compare")
    assert route.try_get(h, "/runs/compare") is True
    assert h.status == 400
    assert "a=" in h.body["error"]


def test_only_one_param_still_400s():
    h = Handler("/runs/compare?a=run_a")
    assert route.try_get(h, "/runs/compare") is True
    assert h.status == 400


def test_missing_runs_yield_404_naming_both(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: None)
    h = Handler("/runs/compare?a=run_a&b=run_b")
    assert route.try_get(h, "/runs/compare") is True
    assert h.status == 404
    assert h.body["missing"] == ["run_a", "run_b"]


def test_one_missing_run_names_only_that_one(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: _run(rid) if rid == "run_a" else None)
    h = Handler("/runs/compare?a=run_a&b=run_b")
    route.try_get(h, "/runs/compare")
    assert h.status == 404
    assert h.body["missing"] == ["run_b"]


def test_success_returns_a_valid_run_diff_document(monkeypatch):
    a = _run("run_a", identity={"model_sha256": "a" * 64})
    b = _run("run_b", identity={"model_sha256": "b" * 64})
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"run_a": a, "run_b": b}[rid])
    h = Handler("/runs/compare?a=run_a&b=run_b")
    assert route.try_get(h, "/runs/compare") is True
    assert h.status == 200
    assert h.body["schema_version"] == "clozn.run-diff.v1"
    from clozn import schemas
    schemas.validate(h.body, "clozn.run-diff.v1")


def test_replay_query_param_adds_plan(monkeypatch):
    a = _run("run_a", meta={"temperature": 0.2})
    b = _run("run_b", meta={"temperature": 0.9})
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"run_a": a, "run_b": b}[rid])
    h = Handler("/runs/compare?a=run_a&b=run_b&replay=1")
    route.try_get(h, "/runs/compare")
    assert h.status == 200
    assert "replay_plan" in h.body
    assert h.body["replay_plan"]["runs_required"] >= 1


def test_no_replay_param_omits_plan(monkeypatch):
    a = _run("run_a")
    b = _run("run_b")
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"run_a": a, "run_b": b}[rid])
    h = Handler("/runs/compare?a=run_a&b=run_b")
    route.try_get(h, "/runs/compare")
    assert "replay_plan" not in h.body
