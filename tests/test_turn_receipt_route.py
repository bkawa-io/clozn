"""HTTP coverage for the read-only Turn Receipt route."""
from __future__ import annotations

import copy

import clozn.runs.store as runlog
from clozn import schemas
from clozn.replay import execution_fork_results
from clozn.server.routes import turn_receipt as route


class Handler:
    def __init__(self, path):
        self.path = path
        self.status = None
        self.body = None
        self.content_type = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body

    def _send(self, status, body, content_type, *_args, **_kwargs):
        self.status, self.body, self.content_type = status, body, content_type


def _run():
    return {
        "id": "run_route",
        "model": "llama-3.1-8b",
        "response": "hello",
        "finish_reason": "stop",
        "trace": {"tokens": ["hello"]},
        "context_receipt": {
            "schema_version": "clozn.context-receipt.v1", "run_id": "run_route",
            "privacy": "metadata_only", "limits": {"prompt_tokens": 100, "context_window_tokens": 1000,
                                                        "generated_tokens": 1},
            "delivered": [], "assembled": [], "omissions": [], "transformations": [],
            "termination": {"reason": "eos", "generated_tokens": 1},
        },
    }


def test_json_route_composes_from_recorded_evidence(monkeypatch):
    run = _run()
    before = copy.deepcopy(run)
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr(execution_fork_results, "list_for_parent", lambda _rid: [])

    h = Handler("/runs/run_route/turn-receipt")
    assert route.try_get(h, "/runs/run_route/turn-receipt") is True
    assert h.status == 200
    schemas.validate(h.body, "clozn.turn-receipt.v1")
    assert h.body["run_id"] == "run_route"
    assert h.body["what_mattered"]["measurement_state"] == "not_measured"
    assert run == before


def test_markdown_route_is_immediately_shareable(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr(execution_fork_results, "list_for_parent", lambda _rid: [])

    h = Handler("/runs/run_route/turn-receipt?format=md")
    assert route.try_get(h, "/runs/run_route/turn-receipt") is True
    assert h.status == 200
    assert h.content_type == "text/markdown; charset=utf-8"
    assert h.body.startswith("# Clozn Receipt")
    assert "hello" not in h.body


def test_missing_run_is_404(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: None)
    h = Handler("/runs/missing/turn-receipt")
    assert route.try_get(h, "/runs/missing/turn-receipt") is True
    assert h.status == 404


def test_route_does_not_start_any_expensive_analysis(monkeypatch):
    """The route succeeds even when every model/scoring/live-execution seam is booby-trapped."""
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr(execution_fork_results, "list_for_parent", lambda _rid: [])

    def explode(*_args, **_kwargs):
        raise AssertionError("Turn Receipt triggered expensive analysis")

    import clozn.receipts.context_answer_influence as influence_module
    import clozn.server.model_routing as routing
    from clozn.server import app as server_app

    monkeypatch.setattr(influence_module, "context_answer_influence", explode)
    monkeypatch.setattr(routing, "select_control_model_for_run", explode)
    monkeypatch.setattr(server_app.EngineSubstrate, "score_tokens", explode, raising=False)
    monkeypatch.setattr(server_app.EngineSubstrate, "generate", explode, raising=False)
    monkeypatch.setattr(server_app.EngineSubstrate, "execution_fork", explode, raising=False)
    monkeypatch.setattr(server_app.EngineSubstrate, "execution_fork_checkpoint", explode, raising=False)

    h = Handler("/runs/run_route/turn-receipt")
    assert route.try_get(h, "/runs/run_route/turn-receipt") is True
    assert h.status == 200


def test_route_is_registered_before_generic_run_fallback():
    from clozn.server import app
    assert route in app._GET_ROUTES
    assert app._GET_ROUTES.index(route) < app._GET_ROUTES.index(app._runs_fallback_routes)
