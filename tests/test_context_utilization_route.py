"""Route coverage for GET /runs/<id>/context-utilization (clozn/server/routes/context_utilization.py)
-- prompt-source measurement/effect coverage, served fresh on every request from an already-persisted
influence map. Mirrors tests/test_context_tension_route.py's Handler stub, autoload-registration, and
contract-failure patterns.

Model-free: nothing here touches clozn/engine, a worker, or a model file. `build_context_utilization` is
a pure function of already-recorded run data (see its own docstring), so this whole suite runs offline.
"""
from __future__ import annotations

import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import clozn.runs.store as runlog  # noqa: E402
from clozn import schemas  # noqa: E402
from clozn.server.routes import context_utilization as route  # noqa: E402


class Handler:
    def __init__(self, path="/"):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


_ANSWER = "Paris is the capital of France."


def _method() -> dict:
    return {
        "name": "teacher_forced_matched_context_replacement", "mode": "forced_score_intervention",
        "claim_limit": "no percentage claim", "caveat": "measured effect only, not correctness",
    }


def _influence(*, prompt_sources=(), prompt_spans=(), links=()) -> dict:
    selected_ids = [s["id"] for s in prompt_sources if s.get("selected")]
    omitted_ids = [s["id"] for s in prompt_sources if not s.get("selected")]
    return {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": _method(),
        "identity": {"model_sha256": "a" * 64},
        "thresholds": {"cell_abs_delta_nats": 0.5},
        "artifact_sha256": "c" * 64,
        "prompt_sources": list(prompt_sources),
        "prompt_spans": list(prompt_spans),
        "answer": {"scored_text": _ANSWER},
        "answer_spans": [{"id": "as-0", "start": 0, "end": len(_ANSWER), "text": _ANSWER}],
        "links": list(links),
        "matrix_complete": True,
        "selection": {
            "strategy": "earliest_policy_then_recent_sources_proportional_chunks_v1",
            "max_context_spans": 8, "selected_source_ids": selected_ids,
            "omitted_source_ids": omitted_ids, "measured_span_count": len(prompt_spans),
            "complete_for_selected_spans": True,
        },
    }


def _run(run_id="run_x", response=_ANSWER, **over) -> dict:
    out = {"id": run_id, "messages": [{"role": "user", "content": "hi"}], "response": response,
          "finish_reason": "stop"}
    out.update(over)
    return out


def _standard_run(run_id="run_x") -> dict:
    influence = _influence(
        prompt_sources=[
            {"id": "p.m000", "start": 0, "end": 20, "text": "PRIVATE SOURCE A LITERAL", "selected": True},
            {"id": "p.m001", "start": 0, "end": 15, "text": "PRIVATE SOURCE B LITERAL", "selected": False},
        ],
        prompt_spans=[{"id": "p.m000.c000", "parent_id": "p.m000", "level": "coarse", "start": 0,
                      "end": 20, "text": "PRIVATE SOURCE A LITERAL"}],
        links=[{"context_span_id": "p.m000.c000", "answer_span_id": "as-0", "context_index": 0,
                "answer_index": 0, "delta_nats": -1.5, "abs_delta_nats": 1.5, "effect": "supports",
                "clears_floor": True, "evidence_state": "causally_supported"}],
    )
    return _run(run_id, influence_map=influence)


def test_route_200_available_utilization(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-utilization")
    assert route.try_get(h, f"/runs/{run['id']}/context-utilization") is True
    assert h.status == 200
    schemas.validate(h.body, "clozn.context-utilization.v1")
    assert h.body["measurement"]["state"] == "available"
    assert len(h.body["sources"]) == 2


def test_route_200_valid_not_measured_state(monkeypatch):
    run = _run("run_no_map")
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-utilization")
    route.try_get(h, f"/runs/{run['id']}/context-utilization")
    assert h.status == 200
    schemas.validate(h.body, "clozn.context-utilization.v1")
    assert h.body["measurement"]["state"] == "not_measured"
    assert h.body["measurement"]["reason"] == "no_influence_map"
    assert h.body["sources"] == []


def test_route_200_valid_unavailable_state(monkeypatch):
    influence = {
        "schema": "clozn.context_answer_influence.v1", "status": "unavailable", "available": False,
        "method": _method(), "identity": {}, "error": {"code": "no_text_context", "message": "x"},
    }
    run = _run("run_unavailable", influence_map=influence)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-utilization")
    route.try_get(h, f"/runs/{run['id']}/context-utilization")
    assert h.status == 200
    schemas.validate(h.body, "clozn.context-utilization.v1")
    assert h.body["measurement"]["state"] == "unavailable"
    assert h.body["measurement"]["reason"] == "no_text_context"


def test_route_404_when_run_not_found():
    original = runlog.get_run
    try:
        runlog.get_run = lambda _rid: None
        h = Handler("/runs/missing/context-utilization")
        assert route.try_get(h, "/runs/missing/context-utilization") is True
        assert h.status == 404
    finally:
        runlog.get_run = original


def test_route_response_is_deterministic(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h1 = Handler(f"/runs/{run['id']}/context-utilization")
    route.try_get(h1, f"/runs/{run['id']}/context-utilization")
    h2 = Handler(f"/runs/{run['id']}/context-utilization")
    route.try_get(h2, f"/runs/{run['id']}/context-utilization")
    assert h1.body == h2.body


def test_route_response_is_metadata_only_and_never_leaks_source_or_answer_text(monkeypatch):
    private_answer = "SECRET PRIVATE ANSWER TEXT"
    influence = _influence(
        prompt_sources=[{"id": "p.m000", "start": 0, "end": 18, "text": "SECRET SOURCE LITERAL",
                        "selected": True}],
        prompt_spans=[{"id": "p.m000.c000", "parent_id": "p.m000", "level": "coarse", "start": 0,
                      "end": 18, "text": "SECRET SOURCE LITERAL"}],
        links=[{"context_span_id": "p.m000.c000", "answer_span_id": "as-0", "context_index": 0,
                "answer_index": 0, "delta_nats": -1.0, "abs_delta_nats": 1.0, "effect": "supports",
                "clears_floor": True, "evidence_state": "causally_supported"}],
    )
    influence["answer"]["scored_text"] = private_answer
    influence["answer_spans"] = [{"id": "as-0", "start": 0, "end": len(private_answer),
                                  "text": private_answer}]
    run = _run("run_private", response=private_answer, influence_map=influence)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-utilization")
    route.try_get(h, f"/runs/{run['id']}/context-utilization")
    assert h.status == 200
    assert h.body["privacy"] == "metadata_only"
    assert "SECRET" not in repr(h.body)


def test_route_does_not_match_unrelated_paths():
    h = Handler("/runs/x/context-tension")
    assert route.try_get(h, "/runs/x/context-tension") is False
    assert route.try_get(h, "/runs/x/influence-query") is False
    assert route.try_get(h, "/runs/x/claim-support") is False
    assert route.try_get(h, "/runs/x/span-addresses") is False
    assert route.try_get(h, "/runs/x") is False
    assert route.try_get(h, "/other") is False


def test_route_registered_before_the_runs_fallback():
    from clozn.server import app as server
    assert route in server._GET_ROUTES
    assert server._GET_ROUTES.index(route) < server._GET_ROUTES.index(server._runs_fallback_routes)


def test_route_contract_failure_does_not_echo_private_exception_text(monkeypatch):
    run = _standard_run("run_broken")
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr(
        "clozn.runs.context_utilization.build_context_utilization",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("PRIVATE MALFORMED SOURCE")),
    )

    h = Handler(f"/runs/{run['id']}/context-utilization")
    assert route.try_get(h, f"/runs/{run['id']}/context-utilization") is True

    assert h.status == 500
    assert h.body == {
        "error": "run context utilization could not be composed",
        "code": "context_utilization_contract_invalid",
    }
    assert "PRIVATE MALFORMED SOURCE" not in repr(h.body)


def test_route_selection_inconsistency_is_a_generic_contract_failure(monkeypatch):
    influence = _influence(
        prompt_sources=[{"id": "p.m000", "start": 0, "end": 10, "text": "x", "selected": True}],
        prompt_spans=[{"id": "p.m000.c000", "parent_id": "p.m000", "level": "coarse", "start": 0,
                      "end": 10, "text": "x"}],
        links=[{"context_span_id": "p.m000.c000", "answer_span_id": "as-0", "context_index": 0,
                "answer_index": 0, "delta_nats": -1.0, "abs_delta_nats": 1.0, "effect": "supports",
                "clears_floor": True, "evidence_state": "causally_supported"}],
    )
    influence["selection"]["selected_source_ids"] = []
    influence["selection"]["omitted_source_ids"] = ["p.m000"]
    run = _run("run_inconsistent", influence_map=influence)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-utilization")
    route.try_get(h, f"/runs/{run['id']}/context-utilization")
    assert h.status == 500
    assert h.body["code"] == "context_utilization_contract_invalid"


def test_route_run_not_mutated(monkeypatch):
    run = _standard_run("run_untouched")
    before = copy.deepcopy(run)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-utilization")
    route.try_get(h, f"/runs/{run['id']}/context-utilization")

    assert h.status == 200
    assert run == before
