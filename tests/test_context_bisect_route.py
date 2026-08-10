"""HTTP contract coverage for Context Bisect plan/execute boundaries."""
from __future__ import annotations

import pytest

import clozn.runs.store as runlog
from clozn.server.routes import context_bisect as route


class Handler:
    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _run():
    return {
        "id": "run-context-bisect-route",
        "model": "parent-model",
        "messages": [{"role": "user", "content": "context"}],
        "response": "answer",
        "trace": {"tokens": ["answer"], "token_ids": [1]},
        "meta": {"decode": {"mode": "greedy", "temperature": 0}},
    }


def test_plan_is_read_only_and_returns_typed_unavailable_without_measurement(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: _run())
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_a, **_k: pytest.fail("Context Bisect planning selected a worker"),
    )
    h = Handler()
    assert route.try_post(
        h, "/runs/run-context-bisect-route/context-bisect/plan",
        {"influence": {"source_span_id": "span_source", "answer_span_id": "span_answer"}},
    ) is True
    assert h.status == 200
    assert h.body["execution"]["state"] == "unavailable"
    assert h.body["execution"]["reason"] == "no_influence_map"


@pytest.mark.parametrize("body,code", [
    ({}, "invalid_influence"),
    ({"influence": {"source_span_id": "s"}}, "invalid_influence"),
    ({"influence": {"source_span_id": "s", "answer_span_id": "a"}, "max_depth": 6}, "invalid_max_depth"),
    ({"influence": {"source_span_id": "s", "answer_span_id": "a"}, "max_runs": 1}, "invalid_max_runs"),
    ({"influence": {"source_span_id": "s", "answer_span_id": "a"}, "max_seconds": 0}, "invalid_max_seconds"),
    ({"influence": {"source_span_id": "s", "answer_span_id": "a"}, "min_region_chars": 7}, "invalid_min_region_chars"),
    ({"influence": {"source_span_id": "s", "answer_span_id": "a"}, "unexpected": True}, "invalid_body"),
])
def test_invalid_plan_is_400(monkeypatch, body, code):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: _run())
    h = Handler()
    route.try_post(h, "/runs/run-context-bisect-route/context-bisect/plan", body)
    assert h.status == 400
    assert h.body["code"] == code


def test_missing_parent_is_404(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: None)
    h = Handler()
    route.try_post(h, "/runs/missing/context-bisect/plan", {})
    assert h.status == 404


def test_execute_unavailable_does_not_route_worker(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: _run())
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_a, **_k: pytest.fail("unavailable Context Bisect selected a worker"),
    )
    h = Handler()
    route.try_post(
        h, "/runs/run-context-bisect-route/context-bisect",
        {"influence": {"source_span_id": "span_source", "answer_span_id": "span_answer"}},
    )
    assert h.status == 422
    assert h.body["execution"]["children_created"] == 0


def test_route_is_autoloaded():
    from clozn.server import app
    assert route in app._POST_ROUTES
