"""HTTP routing tests for Test This planning and execution."""
from __future__ import annotations

import pytest

import clozn.runs.store as runlog
from clozn.server.routes import universal_test_this as route


class Handler:
    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def parent():
    return {
        "id": "run_route",
        "model": "parent-model",
        "trace": {"tokens": ["a", "b"], "token_ids": [1, 2],
                   "alternatives": [[], [{"piece": "x", "token_id": 9, "prob": 0.4}]]},
    }


def body():
    return {"selection": {"kind": "response_token", "position": 1},
            "test": {"kind": "try_alternative", "alternative_rank": 0}}


def test_plan_is_200_and_does_not_resolve_worker(monkeypatch):
    run = parent()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr("clozn.server.model_routing.select_control_model_for_run",
                        lambda *_a, **_k: pytest.fail("read-only plan selected a worker"))
    h = Handler()
    assert route.try_post(h, "/runs/run_route/test-this/plan", body()) is True
    assert h.status == 200
    assert h.body["execution"]["live_state"] == "not_checked"


def test_missing_run_is_404(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: None)
    h = Handler()
    route.try_post(h, "/runs/missing/test-this/plan", body())
    assert h.status == 404


@pytest.mark.parametrize("bad", [None, {}, {"selection": {}, "test": {}},
                                   {"selection": {"kind": "response_token", "position": 1}, "test": {"kind": "nope"}}])
def test_malformed_plan_is_400(monkeypatch, bad):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: parent())
    h = Handler()
    route.try_post(h, "/runs/run_route/test-this/plan", bad)
    assert h.status == 400


class Selection:
    runtime_key = None
    worker_identity = None
    engine = object()
    sub = object()


def test_execution_resolves_parent_model_and_returns_evidence(monkeypatch):
    run = parent()
    seen = {}
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)

    def select(_handler, model, *, route):
        seen.update({"model": model, "route": route})
        return Selection()

    monkeypatch.setattr("clozn.server.model_routing.select_control_model_for_run", select)
    monkeypatch.setattr("clozn.server.routes.execution_fork._identity_facts",
                        lambda _selection: (None, None, object()))
    monkeypatch.setattr("clozn.replay.test_this.execute_test_this", lambda *a, **k: {
        "schema_version": "clozn.test-this-result.v1", "run_id": run["id"],
        "selection": {"kind": "response_token", "position": 1},
        "test": {"kind": "try_alternative", "alternative_rank": 0},
        "operation": "force_token", "outcome": "completed",
        "result": {"observation_id": "observation-1"},
        "artifact": {"schema": "clozn.time-travel-result.v1"},
        "comparison": None,
    })
    h = Handler()
    route.try_post(h, "/runs/run_route/test-this", body())
    assert h.status == 201
    assert seen == {"model": "parent-model", "route": "/runs/<id>/test-this"}


def test_execution_does_not_accept_model_override(monkeypatch):
    run = parent()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    h = Handler()
    request = {**body(), "model": "wrong-model"}
    route.try_post(h, "/runs/run_route/test-this", request)
    assert h.status == 400


def test_route_is_autoloaded():
    from clozn.server import app
    assert route in app._POST_ROUTES
