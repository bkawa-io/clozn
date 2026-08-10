"""HTTP coverage for the read-only Suggested Breakpoints route."""
from __future__ import annotations

import copy

import clozn.runs.store as runlog
from clozn import schemas
from clozn.runs import suggested_breakpoints
from clozn.server.routes import suggested_breakpoints as route


class Handler:
    def __init__(self, path):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _run():
    return {
        "id": "run_route_sb",
        "response": "answer",
        "trace": {
            "tokens": ["answer"],
            "confidence": [0.9],
            "alternatives": [[]],
        },
    }


def test_existing_run_returns_the_v1_document_with_default_limit(monkeypatch):
    run = _run()
    before = copy.deepcopy(run)
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    h = Handler("/runs/run_route_sb/suggested-breakpoints")
    assert route.try_get(h, "/runs/run_route_sb/suggested-breakpoints") is True
    assert h.status == 200
    schemas.validate(h.body, "clozn.suggested-breakpoints.v1")
    assert h.body["run_id"] == "run_route_sb"
    assert run == before


def test_custom_limit_is_passed_to_the_builder(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    seen = {}

    def fake_builder(value, *, limit, privacy="metadata_only"):
        seen.update({"run": value, "limit": limit, "privacy": privacy})
        return {
            "schema_version": "clozn.suggested-breakpoints.v1",
            "run_id": "run_route_sb",
            "privacy": "metadata_only",
            "coordinates": {"kind": "recorded_response_token_boundary", "index_base": 0},
            "analysis": {"state": "unavailable"},
            "evidence": {
                "close_calls": {"state": "unavailable", "thresholds": {"margin": 0.12, "min_runnerup": 0.3}},
                "context_tension": {"state": "not_measured"},
                "answer_alignment": {"state": "unavailable"},
            },
            "breakpoints": [],
            "summary": {
                "candidate_state": "unavailable",
                "suggested_breakpoints": 0,
                "returned_breakpoints": 0,
                "combined_breakpoints": 0,
                "meaningful_close_call_breakpoints": 0,
                "context_tension_breakpoints": 0,
                "ordinary_close_call_breakpoints": 0,
            },
        }

    monkeypatch.setattr(suggested_breakpoints, "build_suggested_breakpoints", fake_builder)
    h = Handler("/runs/run_route_sb/suggested-breakpoints?limit=5")
    assert route.try_get(h, "/runs/run_route_sb/suggested-breakpoints") is True
    assert h.status == 200
    assert seen == {"run": run, "limit": 5, "privacy": "metadata_only"}


def test_missing_run_is_404(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: None)
    h = Handler("/runs/missing/suggested-breakpoints")
    assert route.try_get(h, "/runs/missing/suggested-breakpoints") is True
    assert h.status == 404


def test_invalid_limits_are_typed_400s(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: _run())
    for raw in ("0", "-1", "51", "1.5", "true", ""):
        h = Handler(f"/runs/run_route_sb/suggested-breakpoints?limit={raw}")
        assert route.try_get(h, "/runs/run_route_sb/suggested-breakpoints") is True
        assert h.status == 400
        assert h.body["code"] == "invalid_limit"


def test_limit_50_is_the_inclusive_upper_bound(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: _run())
    h = Handler("/runs/run_route_sb/suggested-breakpoints?limit=50")
    assert route.try_get(h, "/runs/run_route_sb/suggested-breakpoints") is True
    assert h.status == 200


def test_builder_contract_failure_is_sanitized(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: _run())

    def explode(*_args, **_kwargs):
        raise RuntimeError("PRIVATE ANSWER TEXT")

    monkeypatch.setattr(suggested_breakpoints, "build_suggested_breakpoints", explode)
    h = Handler("/runs/run_route_sb/suggested-breakpoints")
    assert route.try_get(h, "/runs/run_route_sb/suggested-breakpoints") is True
    assert h.status == 500
    assert h.body == {
        "error": "run suggested breakpoints could not be composed",
        "code": "suggested_breakpoints_contract_invalid",
    }
    assert "PRIVATE ANSWER TEXT" not in str(h.body)


def test_route_does_not_invoke_expensive_analysis_or_execution(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)

    def explode(*_args, **_kwargs):
        raise AssertionError("Suggested Breakpoints triggered an execution seam")

    import clozn.receipts.context_answer_influence as influence_module
    import clozn.server.model_routing as routing
    from clozn.server import app as server_app

    monkeypatch.setattr(influence_module, "context_answer_influence", explode)
    monkeypatch.setattr(routing, "select_control_model_for_run", explode)
    for name in ("score_tokens", "generate", "execution_fork", "execution_fork_checkpoint"):
        monkeypatch.setattr(server_app.EngineSubstrate, name, explode, raising=False)

    h = Handler("/runs/run_route_sb/suggested-breakpoints")
    assert route.try_get(h, "/runs/run_route_sb/suggested-breakpoints") is True
    assert h.status == 200


def test_route_is_autoloaded_before_the_generic_run_fallback():
    from clozn.server import app

    assert route in app._GET_ROUTES
    assert app._GET_ROUTES.index(route) < app._GET_ROUTES.index(app._runs_fallback_routes)
