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
    assert "b=" in h.body["error"]


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


def test_automatic_get_selection_excludes_child_runs(monkeypatch):
    candidate = _run("candidate", identity={"model_sha256": "m" * 64}, recorded_ts=3)
    child = _run("child", identity={"model_sha256": "m" * 64}, recorded_ts=2,
                 parent_run_id="older", source="replay")
    older = _run("older", identity={"model_sha256": "m" * 64}, recorded_ts=1)
    monkeypatch.setattr(runlog, "get_run", lambda rid: candidate if rid == "candidate" else None)
    monkeypatch.setattr(runlog, "iter_runs", lambda: [candidate, child, older])
    h = Handler("/runs/compare?b=candidate&against=previous_compatible")
    route.try_get(h, "/runs/compare")
    assert h.status == 200
    assert h.body["run_a"] == "older"
    assert h.body["comparison_selection"]["mode"] == "previous_compatible"


def test_post_plan_is_model_free_and_schema_valid(monkeypatch):
    a = _run("run_a", response="good", messages=[{"role": "user", "content": "full"}])
    b = _run("run_b", response="bad", messages=[{"role": "user", "content": "short"}])
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"run_a": a, "run_b": b}.get(rid))
    h = Handler("/runs/compare/test")
    handled = route.try_post(h, "/runs/compare/test", {
        "a": "run_a", "b": "run_b", "tests": ["context"],
        "max_runs": 2, "max_seconds": 30, "plan": True,
    })
    assert handled is True and h.status == 200
    assert h.body["schema_version"] == "clozn.run-change-test.v1"
    assert h.body["budget"]["runs_used"] == 0


def test_post_execution_returns_real_runner_child_ids(monkeypatch):
    from clozn.replay import controlled
    from clozn.server import app

    meta = {"sampling": "greedy", "temperature": 0.0}
    a = _run("run_a", response="good", messages=[{"role": "user", "content": "full"}], meta=meta)
    b = _run("run_b", response="bad", messages=[{"role": "user", "content": "short"}], meta=meta)
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"run_a": a, "run_b": b}.get(rid))
    monkeypatch.setattr(app, "active_sub", lambda _h: type("Sub", (), {"chat": lambda *_a, **_k: ""})())

    class FakeLiveRunner:
        def __init__(self, _sub):
            self.calls = 0

        def qualify(self, *_args):
            return {"ok": True}

        def run_arm(self, _kind, arm, _a, _b, *, timeout_seconds):
            self.calls += 1
            return {"run": {"id": f"run_{arm}", "response": "bad" if arm == "control" else "good",
                            "context_receipt": {}, "trace": {}}}

    monkeypatch.setattr(controlled, "SubstrateReplayRunner", FakeLiveRunner)
    h = Handler("/runs/compare/test")
    route.try_post(h, "/runs/compare/test", {
        "a": "run_a", "b": "run_b", "tests": ["context"],
        "max_runs": 2, "max_seconds": 30,
    })
    assert h.status == 200
    assert h.body["tests"][0]["status"] == "causally_supported"
    assert [e["run_id"] for e in h.body["tests"][0]["evidence"]] == [
        "run_control", "run_treatment",
    ]
