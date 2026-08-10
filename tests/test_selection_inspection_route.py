"""HTTP contract tests for the read-only selection inspection route."""
from __future__ import annotations

import pytest

import clozn.runs.store as runlog
from clozn.server.routes import selection_inspection as route


class Handler:
    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _run():
    return {
        "id": "run-selection-route",
        "response": "hello",
        "trace": {"tokens": ["he", "llo"], "token_ids": [1, 2],
                  "confidence": [0.8, 0.8], "alternatives": [[], []]},
        "identity": {"model_sha256": "a" * 64, "template_fingerprint": "t", "engine_build": "e"},
        "final_prompt": "prompt",
        "meta": {"decode": {"mode": "greedy", "temperature": 0}},
    }


def test_existing_run_returns_read_only_document(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr("clozn.server.model_routing.select_control_model_for_run",
                        lambda *_a, **_k: pytest.fail("selection inspection must not route a worker"))
    h = Handler()
    assert route.try_post(h, "/runs/run-selection-route/selection/inspect",
                          {"selection": {"kind": "response_token", "position": 0}}) is True
    assert h.status == 200
    assert h.body["selection"]["kind"] == "response_token"
    assert h.body["privacy"] == "metadata_only"


@pytest.mark.parametrize("body,code", [
    ({}, "invalid_selection"),
    ({"selection": {"kind": "unknown"}}, "invalid_selection_kind"),
    ({"selection": {"kind": "response_token", "position": -1}}, "invalid_position"),
    ({"selection": {"kind": "answer_span", "start": 0, "end": 99}}, "invalid_answer_span"),
    ({"selection": {"kind": "sampling", "position": True}}, "invalid_position"),
])
def test_malformed_selection_is_400(monkeypatch, body, code):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: _run())
    h = Handler()
    route.try_post(h, "/runs/run-selection-route/selection/inspect", body)
    assert h.status == 400
    assert h.body["code"] == code


def test_missing_run_is_404(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: None)
    h = Handler()
    route.try_post(h, "/runs/missing/selection/inspect", {"selection": {"kind": "sampling"}})
    assert h.status == 404


def test_contract_failure_is_sanitized(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: _run())
    monkeypatch.setattr("clozn.runs.selection_inspection.build_selection_inspection",
                        lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("PRIVATE PROMPT")))
    h = Handler()
    route.try_post(h, "/runs/run-selection-route/selection/inspect",
                   {"selection": {"kind": "sampling"}})
    assert h.status == 500
    assert h.body == {
        "error": "selection inspection could not be composed",
        "code": "selection_inspection_contract_invalid",
    }


def test_route_is_autoloaded():
    from clozn.server import app
    assert route in app._POST_ROUTES
