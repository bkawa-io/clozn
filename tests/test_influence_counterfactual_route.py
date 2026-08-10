"""HTTP boundary tests for Influence -> Counterfactual Confirmation."""
from __future__ import annotations

import pytest

import clozn.runs.store as runlog
from clozn.server.routes import influence_counterfactual as route


class Handler:
    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _run():
    return {
        "id": "run-ict-route",
        "model": "parent-model",
        "messages": [{"role": "user", "content": "hello"}],
        "response": "reply",
        "trace": {"tokens": ["reply"]},
    }


def _body():
    return {
        "influence": {"source_span_id": "span-source", "answer_span_id": "span-answer"},
        "intervention": {"kind": "neutralize"},
    }


def test_plan_is_read_only_and_does_not_resolve_worker(monkeypatch):
    parent = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: parent)
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: pytest.fail("planning must not route to a worker"),
    )
    h = Handler()
    assert route.try_post(h, "/runs/run-ict-route/influence-counterfactual/plan", _body()) is True
    assert h.status == 200
    assert h.body["execution"]["live_runtime_state"] == "not_checked"
    assert h.body["execution"]["state"] == "unavailable"


def test_missing_parent_is_404(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: None)
    h = Handler()
    route.try_post(h, "/runs/missing/influence-counterfactual/plan", _body())
    assert h.status == 404


@pytest.mark.parametrize("body,code", [
    (None, "invalid_body"),
    ({}, "invalid_influence"),
    ({"influence": {"source_span_id": "s"}}, "invalid_influence"),
    ({"influence": {"source_span_id": "s", "answer_span_id": "a"},
      "intervention": {"kind": "replace"}}, "invalid_intervention"),
])
def test_malformed_request_is_400(monkeypatch, body, code):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: _run())
    h = Handler()
    route.try_post(h, "/runs/run-ict-route/influence-counterfactual/plan", body)
    assert h.status == 400
    assert h.body["code"] == code


def test_structurally_unavailable_execution_is_422_without_worker_selection(monkeypatch):
    parent = _run()
    seen = []
    monkeypatch.setattr(runlog, "get_run", lambda _rid: parent)
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: seen.append(True),
    )
    h = Handler()
    route.try_post(h, "/runs/run-ict-route/influence-counterfactual", _body())
    assert h.status == 422
    assert seen == []


def test_route_is_autoloaded():
    from clozn.server import app
    assert route in app._POST_ROUTES
