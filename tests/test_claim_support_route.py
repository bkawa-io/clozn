"""Route coverage for GET /runs/<id>/claim-support (clozn/server/routes/claim_support.py) -- E1's
`clozn.answer-claims.v1` claim segmentation and E2's `clozn.claim-support.v1` per-claim verification
status, served together, derived fresh on every request. Mirrors tests/test_diagnosis_findings_route.py
and tests/test_span_addresses_route.py's Handler stub, autoload-registration, and contract-failure
patterns -- the closest two analogs for a route that composes two pure derived artifacts over one run.

Model-free: nothing here touches clozn/engine, a worker, or a model file. Both `build_answer_claims` and
`build_claim_support` are pure functions of already-recorded run data (see their own docstrings), so this
whole suite runs offline.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import clozn.runs.store as runlog  # noqa: E402
from clozn import schemas  # noqa: E402
from clozn.server.routes import claim_support as route  # noqa: E402


class Handler:
    def __init__(self, path="/"):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _method_stub() -> dict:
    return {
        "name": "context_answer_influence", "mode": "forced_score_intervention",
        "claim_limit": "no percentage claim", "caveat": "measured effect only, not correctness",
    }


def _influence(*, prompt_spans=(), answer_text="", answer_spans=(), links=()) -> dict:
    return {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": _method_stub(),
        "identity": {"model_sha256": "a" * 64},
        "prompt_spans": list(prompt_spans),
        "answer": {"scored_text": answer_text},
        "answer_spans": list(answer_spans),
        "links": list(links),
    }


def _run(run_id="run_x", response="The bridge was built in 1920. You should double-check the date.",
         **over) -> dict:
    out = {"id": run_id, "messages": [{"role": "user", "content": "hi"}], "response": response,
          "finish_reason": "stop"}
    out.update(over)
    return out


def test_route_returns_claims_and_support_together(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler(f"/runs/{run['id']}/claim-support")
    assert route.try_get(h, f"/runs/{run['id']}/claim-support") is True
    assert h.status == 200
    assert h.body["claims"]["schema_version"] == "clozn.answer-claims.v1"
    assert h.body["support"]["schema_version"] == "clozn.claim-support.v1"
    assert h.body["claims"]["run_id"] == run["id"]
    assert h.body["support"]["run_id"] == run["id"]
    schemas.validate(h.body["claims"])
    schemas.validate(h.body["support"])

    # Two sentences -> two claims: a factual declarative and a recommendation. Neither run recorded an
    # influence map, so the factual claim's status is honestly "measurement_unavailable" (never demoted
    # to "unsupported"), and the recommendation is a category-rule "unverifiable_from_available_evidence".
    assert len(h.body["claims"]["claims"]) == 2
    statuses = {r["claim_index"]: r["status"] for r in h.body["support"]["results"]}
    categories = {c["index"]: c["category"] for c in h.body["claims"]["claims"]}
    assert categories[0] == "factual_claim"
    assert statuses[0] == "measurement_unavailable"
    assert h.body["support"]["results"][0]["method"]["name"] == "no_influence_map"
    assert categories[1] == "recommendation"
    assert statuses[1] == "unverifiable_from_available_evidence"
    assert h.body["support"]["results"][1]["method"] == {"name": "category_rule"}


def test_route_supported_claim_via_persisted_influence_map(monkeypatch):
    answer = "The Eiffel Tower was completed in 1889."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 49,
                       "text": "The Eiffel Tower was completed in 1889 in Paris."}],
        answer_text=answer,
        answer_spans=[{"id": "as-1", "start": 0, "end": len(answer), "text": answer}],
        links=[{"context_span_id": "ps-1", "answer_span_id": "as-1", "context_index": 0,
                "answer_index": 0, "delta_nats": -3.2, "abs_delta_nats": 3.2, "effect": "supports",
                "clears_floor": True, "evidence_state": "causally_supported"}],
    )
    run = _run("run_supported", response=answer, influence_map=influence)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler()
    route.try_get(h, f"/runs/{run['id']}/claim-support")

    assert h.status == 200
    result = h.body["support"]["results"][0]
    assert result["status"] == "supported"
    assert result["method"]["name"] == "forced_score_intervention"
    assert result["source_span_ids"][0].startswith("span_")


def test_route_metadata_only_never_leaks_claim_text(monkeypatch):
    run = _run("run_private", response="PRIVATE SENTENCE was true in 2019.")
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler()
    route.try_get(h, f"/runs/{run['id']}/claim-support")

    assert h.status == 200
    assert h.body["claims"]["privacy"] == "metadata_only"
    assert h.body["support"]["privacy"] == "metadata_only"
    assert "PRIVATE SENTENCE" not in repr(h.body)
    for claim in h.body["claims"]["claims"]:
        assert "text" not in (claim["text_span"].get("resolution") or {}).get("canonical", {})


def test_route_empty_response_produces_empty_claims_and_support(monkeypatch):
    run = _run("run_empty", response="")
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler()
    route.try_get(h, f"/runs/{run['id']}/claim-support")

    assert h.status == 200
    assert h.body["claims"]["segmentation"] == {"state": "empty", "reason": "answer_text_empty"}
    assert h.body["claims"]["claims"] == []
    assert h.body["support"]["results"] == []


def test_route_redacted_run_produces_unavailable_segmentation_no_leak(monkeypatch):
    run = _run("run_redacted", response=None)
    run["redaction"] = {"status": "redacted"}
    run["flags"] = ["redacted"]
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler()
    route.try_get(h, f"/runs/{run['id']}/claim-support")

    assert h.status == 200
    assert h.body["claims"]["segmentation"]["state"] == "unavailable"
    assert h.body["claims"]["segmentation"]["reason"] == "answer_text_redacted"
    assert h.body["claims"]["claims"] == []
    assert h.body["support"]["results"] == []


def test_route_404_when_run_not_found(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: None)
    h = Handler("/runs/missing/claim-support")
    assert route.try_get(h, "/runs/missing/claim-support") is True
    assert h.status == 404


def test_route_does_not_match_unrelated_paths():
    h = Handler("/runs/x/diagnosis-findings")
    assert route.try_get(h, "/runs/x/diagnosis-findings") is False
    assert route.try_get(h, "/runs/x/span-addresses") is False
    assert route.try_get(h, "/runs/x") is False
    assert route.try_get(h, "/other") is False


def test_route_registered_before_the_runs_fallback():
    from clozn.server import app as server
    assert route in server._GET_ROUTES
    assert server._GET_ROUTES.index(route) < server._GET_ROUTES.index(server._runs_fallback_routes)


def test_route_contract_failure_does_not_echo_private_exception_text(monkeypatch):
    run = _run("run_broken")
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr(
        "clozn.runs.claims.build_answer_claims",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ValueError("PRIVATE MALFORMED SOURCE")),
    )

    h = Handler()
    assert route.try_get(h, f"/runs/{run['id']}/claim-support") is True

    assert h.status == 500
    assert h.body == {
        "error": "run claim support could not be composed",
        "code": "claim_support_contract_invalid",
    }
    assert "PRIVATE MALFORMED SOURCE" not in repr(h.body)


def test_route_run_not_mutated(monkeypatch):
    import copy
    run = _run("run_untouched")
    before = copy.deepcopy(run)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler()
    route.try_get(h, f"/runs/{run['id']}/claim-support")

    assert h.status == 200
    assert run == before
