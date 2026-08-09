"""Route coverage for GET /runs/<id>/influence-query (clozn/server/routes/influence_query.py) --
"Why this?": the measured context spans overlapping a caller-selected range of the recorded answer,
served fresh on every request from an already-persisted influence map. Mirrors
tests/test_claim_support_route.py and tests/test_span_addresses_route.py's Handler stub, autoload-
registration, and contract-failure patterns -- the closest analogs for a route that composes a single
pure derived artifact over one run.

Model-free: nothing here touches clozn/engine, a worker, or a model file. `build_influence_query` is a
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
from clozn.server.routes import influence_query as route  # noqa: E402


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
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 20, "text": "Paris is a city in France."}],
        answer_spans=[
            {"id": "as-0", "start": 0, "end": 6, "text": "Paris "},
            {"id": "as-1", "start": 6, "end": 31, "text": "is the capital of France."},
        ],
        links=[
            {"context_span_id": "ps-1", "answer_span_id": "as-0", "context_index": 0, "answer_index": 0,
             "delta_nats": -2.5, "abs_delta_nats": 2.5, "effect": "supports", "clears_floor": True,
             "evidence_state": "causally_supported"},
        ],
    )
    return _run(run_id, influence_map=influence)


def test_route_200_valid_query_returns_a_conformant_document(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/influence-query?start=0&end=6")
    assert route.try_get(h, f"/runs/{run['id']}/influence-query") is True
    assert h.status == 200
    schemas.validate(h.body, "clozn.influence-query.v1")
    assert h.body["run_id"] == run["id"]
    assert h.body["measurement"]["state"] == "available"
    assert len(h.body["links"]) == 1
    assert h.body["links"][0]["source_span_id"].startswith("span_")
    assert h.body["links"][0]["answer_span_id"].startswith("span_")


def test_route_404_when_run_not_found():
    handler_backup = runlog.get_run
    try:
        runlog.get_run = lambda _rid: None
        h = Handler("/runs/missing/influence-query?start=0&end=5")
        assert route.try_get(h, "/runs/missing/influence-query") is True
        assert h.status == 404
    finally:
        runlog.get_run = handler_backup


def test_route_400_missing_start_and_end(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    for query in ("", "?start=0", "?end=5"):
        h = Handler(f"/runs/{run['id']}/influence-query{query}")
        assert route.try_get(h, f"/runs/{run['id']}/influence-query") is True
        assert h.status == 400, query
        assert h.body["code"] == "invalid_output_range"


def test_route_400_invalid_range_shapes(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    for query in ("?start=-1&end=5", "?start=5&end=5", "?start=5&end=2", "?start=abc&end=5",
                  "?start=0&end=5.0", "?start=true&end=5"):
        h = Handler(f"/runs/{run['id']}/influence-query{query}")
        route.try_get(h, f"/runs/{run['id']}/influence-query")
        assert h.status == 400, query
        assert h.body["code"] == "invalid_output_range"


def test_route_400_out_of_bounds_range(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/influence-query?start=0&end=9999")
    route.try_get(h, f"/runs/{run['id']}/influence-query")
    assert h.status == 400
    assert h.body["code"] == "invalid_output_range"


def test_route_400_invalid_limit(monkeypatch):
    run = _standard_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    for query in ("?start=0&end=6&limit=0", "?start=0&end=6&limit=51", "?start=0&end=6&limit=abc",
                  "?start=0&end=6&limit=1.5"):
        h = Handler(f"/runs/{run['id']}/influence-query{query}")
        route.try_get(h, f"/runs/{run['id']}/influence-query")
        assert h.status == 400, query
        assert h.body["code"] == "invalid_limit"


def test_route_default_limit_is_twelve(monkeypatch):
    influence = _influence(
        prompt_spans=[{"id": f"ps-{i}", "start": i, "end": i + 1, "text": "x"} for i in range(20)],
        answer_spans=[{"id": "as-0", "start": 0, "end": 6, "text": "Paris "}],
        links=[
            {"context_span_id": f"ps-{i}", "answer_span_id": "as-0", "context_index": i, "answer_index": 0,
             "delta_nats": -float(i + 1), "abs_delta_nats": float(i + 1), "effect": "supports",
             "clears_floor": True, "evidence_state": "causally_supported"}
            for i in range(20)
        ],
    )
    run = _run(influence_map=influence)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/influence-query?start=0&end=6")
    route.try_get(h, f"/runs/{run['id']}/influence-query")
    assert h.status == 200
    assert len(h.body["links"]) == 12
    assert h.body["summary"]["measured_links"] == 20
    assert h.body["summary"]["returned_links"] == 12


def test_route_typed_not_measured_result_when_no_influence_map(monkeypatch):
    run = _run("run_no_map")
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/influence-query?start=0&end=5")
    route.try_get(h, f"/runs/{run['id']}/influence-query")
    assert h.status == 200
    assert h.body["measurement"]["state"] == "not_measured"
    assert h.body["measurement"]["reason"] == "no_influence_map"
    assert h.body["links"] == []


def test_route_response_is_metadata_only_and_never_leaks_answer_or_source_text(monkeypatch):
    run = _run(
        "run_private", response="SECRET PRIVATE ANSWER TEXT " + _ANSWER,
        influence_map=_influence(
            prompt_spans=[{"id": "ps-1", "start": 0, "end": 10, "text": "SECRET SOURCE TEXT"}],
            answer_text="SECRET PRIVATE ANSWER TEXT " + _ANSWER,
            answer_spans=[{"id": "as-0", "start": 0, "end": 27, "text": "SECRET PRIVATE ANSWER TEXT "}],
            links=[{"context_span_id": "ps-1", "answer_span_id": "as-0", "context_index": 0,
                    "answer_index": 0, "delta_nats": -1.0, "abs_delta_nats": 1.0, "effect": "supports",
                    "clears_floor": True, "evidence_state": "causally_supported"}],
        ),
    )
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/influence-query?start=0&end=10")
    route.try_get(h, f"/runs/{run['id']}/influence-query")
    assert h.status == 200
    assert h.body["privacy"] == "metadata_only"
    assert "SECRET" not in repr(h.body)


def test_route_does_not_match_unrelated_paths():
    h = Handler("/runs/x/claim-support")
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
        "clozn.runs.influence_query.build_influence_query",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("PRIVATE MALFORMED SOURCE")),
    )

    h = Handler(f"/runs/{run['id']}/influence-query?start=0&end=5")
    assert route.try_get(h, f"/runs/{run['id']}/influence-query") is True

    assert h.status == 500
    assert h.body == {
        "error": "run influence query could not be composed",
        "code": "influence_query_contract_invalid",
    }
    assert "PRIVATE MALFORMED SOURCE" not in repr(h.body)


def test_route_400_error_message_never_echoes_selected_text(monkeypatch):
    run = _run("run_bounds", response="short")
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/influence-query?start=0&end=9999")
    route.try_get(h, f"/runs/{run['id']}/influence-query")
    assert h.status == 400
    assert "short" not in repr(h.body)


def test_route_run_not_mutated(monkeypatch):
    run = _standard_run("run_untouched")
    before = copy.deepcopy(run)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/influence-query?start=0&end=6")
    route.try_get(h, f"/runs/{run['id']}/influence-query")

    assert h.status == 200
    assert run == before
