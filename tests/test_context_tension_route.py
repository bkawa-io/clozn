"""Route coverage for GET /runs/<id>/context-tension (clozn/server/routes/context_tension.py) --
opposing measured context effects on the same recorded answer span, served fresh on every request from
an already-persisted influence map. Mirrors tests/test_influence_query_route.py's Handler stub,
autoload-registration, and contract-failure patterns.

Model-free: nothing here touches clozn/engine, a worker, or a model file. `build_context_tension` is a
pure function of already-recorded run data (see its own docstring), so this whole suite runs offline.
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
from clozn.server.routes import context_tension as route  # noqa: E402


class Handler:
    def __init__(self, path="/"):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


_ANSWER = "The launch date is June 12."


def _method() -> dict:
    return {
        "name": "teacher_forced_matched_context_replacement", "mode": "forced_score_intervention",
        "claim_limit": "no percentage claim", "caveat": "measured effect only, not correctness",
    }


def _influence(*, prompt_spans=(), answer_text=_ANSWER, answer_spans=(), links=()) -> dict:
    return {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": _method(),
        "identity": {"model_sha256": "a" * 64},
        "thresholds": {"cell_abs_delta_nats": 0.5},
        "artifact_sha256": "c" * 64,
        "prompt_spans": list(prompt_spans),
        "answer": {"scored_text": answer_text},
        "answer_spans": list(answer_spans),
        "links": list(links),
    }


def _run(run_id="run_x", response=_ANSWER, **over) -> dict:
    out = {"id": run_id, "messages": [{"role": "user", "content": "hi"}], "response": response,
          "finish_reason": "stop"}
    out.update(over)
    return out


def _standard_run(run_id="run_x") -> dict:
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 10, "text": "Source A."},
                     {"id": "ps-2", "start": 10, "end": 20, "text": "Source B."}],
        answer_spans=[{"id": "as-0", "start": 0, "end": len(_ANSWER), "text": _ANSWER}],
        links=[
            {"context_span_id": "ps-1", "answer_span_id": "as-0", "context_index": 0, "answer_index": 0,
             "delta_nats": -3.0, "abs_delta_nats": 3.0, "effect": "supports", "clears_floor": True,
             "evidence_state": "causally_supported"},
            {"context_span_id": "ps-2", "answer_span_id": "as-0", "context_index": 1, "answer_index": 0,
             "delta_nats": 2.0, "abs_delta_nats": 2.0, "effect": "suppresses", "clears_floor": True,
             "evidence_state": "causally_supported"},
        ],
    )
    return _run(run_id, influence_map=influence)


def test_route_200_whole_run_query(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-tension")
    assert route.try_get(h, f"/runs/{run['id']}/context-tension") is True
    assert h.status == 200
    schemas.validate(h.body, "clozn.context-tension.v1")
    assert h.body["target"]["scope"] == "whole_answer"
    assert len(h.body["tensions"]) == 1


def test_route_200_ranged_query(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-tension?start=0&end=10")
    route.try_get(h, f"/runs/{run['id']}/context-tension")
    assert h.status == 200
    schemas.validate(h.body, "clozn.context-tension.v1")
    assert h.body["target"]["scope"] == "answer_range"
    assert h.body["target"]["start"] == 0 and h.body["target"]["end"] == 10
    assert len(h.body["tensions"]) == 1


def test_route_404_when_run_not_found():
    original = runlog.get_run
    try:
        runlog.get_run = lambda _rid: None
        h = Handler("/runs/missing/context-tension")
        assert route.try_get(h, "/runs/missing/context-tension") is True
        assert h.status == 404
    finally:
        runlog.get_run = original


def test_route_400_start_only(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-tension?start=0")
    route.try_get(h, f"/runs/{run['id']}/context-tension")
    assert h.status == 400
    assert h.body["code"] == "incomplete_output_range"


def test_route_400_end_only(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-tension?end=10")
    route.try_get(h, f"/runs/{run['id']}/context-tension")
    assert h.status == 400
    assert h.body["code"] == "incomplete_output_range"


def test_route_400_malformed_range(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    for query in ("?start=-1&end=5", "?start=5&end=5", "?start=5&end=2", "?start=abc&end=5",
                  "?start=0&end=5.0", "?start=true&end=5"):
        h = Handler(f"/runs/{run['id']}/context-tension{query}")
        route.try_get(h, f"/runs/{run['id']}/context-tension")
        assert h.status == 400, query
        assert h.body["code"] == "invalid_output_range"


def test_route_400_out_of_bounds_range(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-tension?start=0&end=9999")
    route.try_get(h, f"/runs/{run['id']}/context-tension")
    assert h.status == 400
    assert h.body["code"] == "invalid_output_range"


def test_route_400_invalid_limit(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    for query in ("?limit=0", "?limit=101", "?limit=abc", "?limit=1.5"):
        h = Handler(f"/runs/{run['id']}/context-tension{query}")
        route.try_get(h, f"/runs/{run['id']}/context-tension")
        assert h.status == 400, query
        assert h.body["code"] == "invalid_limit"


def test_route_valid_not_measured_response(monkeypatch):
    run = _run("run_no_map")
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-tension")
    route.try_get(h, f"/runs/{run['id']}/context-tension")
    assert h.status == 200
    schemas.validate(h.body, "clozn.context-tension.v1")
    assert h.body["measurement"]["state"] == "not_measured"
    assert h.body["measurement"]["reason"] == "no_influence_map"
    assert h.body["tensions"] == []


def test_route_valid_empty_tension_response(monkeypatch):
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 10, "text": "Source A."}],
        answer_spans=[{"id": "as-0", "start": 0, "end": len(_ANSWER), "text": _ANSWER}],
        links=[{"context_span_id": "ps-1", "answer_span_id": "as-0", "context_index": 0,
                "answer_index": 0, "delta_nats": -3.0, "abs_delta_nats": 3.0, "effect": "supports",
                "clears_floor": True, "evidence_state": "causally_supported"}],
    )
    run = _run("run_no_tension", influence_map=influence)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-tension")
    route.try_get(h, f"/runs/{run['id']}/context-tension")
    assert h.status == 200
    assert h.body["measurement"]["state"] == "available"
    assert h.body["tensions"] == []


def test_route_contract_failure_does_not_echo_private_exception_text(monkeypatch):
    run = _standard_run("run_broken")
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr(
        "clozn.runs.context_tension.build_context_tension",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("PRIVATE MALFORMED SOURCE")),
    )

    h = Handler(f"/runs/{run['id']}/context-tension")
    assert route.try_get(h, f"/runs/{run['id']}/context-tension") is True

    assert h.status == 500
    assert h.body == {
        "error": "run context tension could not be composed",
        "code": "context_tension_contract_invalid",
    }
    assert "PRIVATE MALFORMED SOURCE" not in repr(h.body)


def test_route_does_not_match_unrelated_paths():
    h = Handler("/runs/x/influence-query")
    assert route.try_get(h, "/runs/x/influence-query") is False
    assert route.try_get(h, "/runs/x/claim-support") is False
    assert route.try_get(h, "/runs/x/span-addresses") is False
    assert route.try_get(h, "/runs/x") is False
    assert route.try_get(h, "/other") is False


def test_route_registered_before_the_runs_fallback():
    from clozn.server import app as server
    assert route in server._GET_ROUTES
    assert server._GET_ROUTES.index(route) < server._GET_ROUTES.index(server._runs_fallback_routes)


def test_route_response_never_leaks_answer_or_source_text(monkeypatch):
    private_answer = "SECRET PRIVATE ANSWER TEXT " + _ANSWER
    run = _run(
        "run_private", response=private_answer,
        influence_map=_influence(
            prompt_spans=[{"id": "ps-1", "start": 0, "end": 10, "text": "SECRET SOURCE A"},
                         {"id": "ps-2", "start": 10, "end": 20, "text": "SECRET SOURCE B"}],
            answer_text=private_answer,
            answer_spans=[{"id": "as-0", "start": 0, "end": len(private_answer), "text": private_answer}],
            links=[
                {"context_span_id": "ps-1", "answer_span_id": "as-0", "context_index": 0,
                 "answer_index": 0, "delta_nats": -1.0, "abs_delta_nats": 1.0, "effect": "supports",
                 "clears_floor": True, "evidence_state": "causally_supported"},
                {"context_span_id": "ps-2", "answer_span_id": "as-0", "context_index": 1,
                 "answer_index": 0, "delta_nats": 1.0, "abs_delta_nats": 1.0, "effect": "suppresses",
                 "clears_floor": True, "evidence_state": "causally_supported"},
            ],
        ),
    )
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-tension")
    route.try_get(h, f"/runs/{run['id']}/context-tension")
    assert h.status == 200
    assert h.body["privacy"] == "metadata_only"
    assert "SECRET" not in repr(h.body)


def test_route_400_error_message_never_echoes_selected_text(monkeypatch):
    run = _run("run_bounds", response="short")
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-tension?start=0&end=9999")
    route.try_get(h, f"/runs/{run['id']}/context-tension")
    assert h.status == 400
    assert "short" not in repr(h.body)


def test_route_run_not_mutated(monkeypatch):
    run = _standard_run("run_untouched")
    before = copy.deepcopy(run)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/context-tension")
    route.try_get(h, f"/runs/{run['id']}/context-tension")

    assert h.status == 200
    assert run == before
