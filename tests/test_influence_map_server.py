from __future__ import annotations

import io
import json

import pytest

from clozn.server import app as cs
import clozn.runs.store as runlog


# Minimal but schema-valid `method` stub (Method requires name/mode/claim_limit/caveat -- see
# clozn/schemas/defs/clozn.context_answer_influence.v1.json) for tests that monkeypatch the backend
# with a hand-built dict rather than a real context_answer_influence() call.
_METHOD_STUB = {
    "name": "teacher_forced_matched_context_replacement",
    "mode": "forced_score_intervention",
    "claim_limit": "behavioral dependence under a controlled prompt intervention",
    "caveat": "Influence means this context changed the measured output under this intervention.",
}


class ScoreSub:
    def score_tokens(self, messages, ids, **kwargs):
        return [{"id": 41, "piece": "Answer", "logprob": -0.2}]


def _post(path, body=None):
    raw = json.dumps(body or {}).encode("utf-8")
    handler_type = cs.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    handler.requestline = f"POST {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = "POST"
    handler.do_POST()
    head, _, payload = handler.wfile.getvalue().partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(payload.decode("utf-8"))


def _get(path):
    handler_type = cs.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(b"")
    handler.wfile = io.BytesIO()
    handler.headers = {"Content-Length": "0", "User-Agent": "pytest"}
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = "GET"
    handler.do_GET()
    head, _, payload = handler.wfile.getvalue().partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(payload.decode("utf-8"))


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", ScoreSub())
    return tmp_path


def _seed():
    return runlog.record(
        source="studio_chat",
        client="studio",
        model="test-model",
        substrate="test",
        messages=[{"role": "user", "content": "Use this exact context."}],
        response="Answer",
        trace={"token_ids": [41]},
    )


def test_influence_map_computes_and_attaches_to_run(isolated, monkeypatch):
    rid = _seed()
    expected = {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": _METHOD_STUB,
        "identity": {"run_id": rid},
        "prompt_spans": [],
        "answer_spans": [],
        "links": [],
    }
    import clozn.receipts.context_answer_influence as backend
    monkeypatch.setattr(backend, "context_answer_influence", lambda run, sub, **opts: expected)

    status, out = _post(f"/runs/{rid}/influence-map")

    assert status == 200
    assert out == expected
    assert runlog.get_run(rid)["influence_map"] == expected


def test_influence_map_returns_attached_map_without_rescoring(isolated, monkeypatch):
    rid = _seed()
    run = runlog.get_run(rid)
    run["influence_map"] = {"schema": "clozn.context_answer_influence.v1", "available": True}
    assert runlog.replace_run(run)
    import clozn.receipts.context_answer_influence as backend
    monkeypatch.setattr(
        backend,
        "context_answer_influence",
        lambda *_args, **_kwargs: pytest.fail("cached maps must not be rescored"),
    )

    status, out = _post(f"/runs/{rid}/influence-map")

    assert status == 200
    assert out["schema"] == "clozn.context_answer_influence.v1"


def test_influence_map_validates_run_worker_and_cost_bound(isolated, monkeypatch):
    status, out = _post("/runs/missing/influence-map")
    assert status == 404 and out == {"error": "run not found"}

    rid = _seed()
    status, out = _post(f"/runs/{rid}/influence-map", {"max_context_spans": 9})
    assert status == 400 and "1 to 8" in out["error"]

    monkeypatch.setattr(cs, "SUB", None)
    status, out = _post(f"/runs/{rid}/influence-map")
    assert status == 503 and "token scoring" in out["error"]


def test_influence_map_failure_is_not_mistaken_for_a_saved_receipt(isolated, monkeypatch):
    rid = _seed()
    failed = {
        "schema": "clozn.context_answer_influence.v1",
        "status": "unavailable",
        "available": False,
        "method": _METHOD_STUB,
        "identity": {"run_id": rid},
        "error": {"code": "scoring_unavailable", "message": "not available"},
    }
    import clozn.receipts.context_answer_influence as backend
    monkeypatch.setattr(backend, "context_answer_influence", lambda *_args, **_kwargs: failed)

    status, out = _post(f"/runs/{rid}/influence-map")

    assert status == 422
    assert out == failed


def test_influence_map_error_status_maps_to_500(isolated, monkeypatch):
    """`status == "error"` (an intervention that should have worked did not complete) is a server-side
    500, distinct from `status == "unavailable"`'s 422 (a precondition was never met). Both are typed,
    non-silent refusals -- this locks in the one branch of that mapping the existing tests didn't cover."""
    rid = _seed()
    broken = {
        "schema": "clozn.context_answer_influence.v1",
        "status": "error",
        "available": False,
        "method": _METHOD_STUB,
        "identity": {"run_id": rid},
        "error": {"code": "intervention_score_failed", "message": "a controlled arm did not complete"},
    }
    import clozn.receipts.context_answer_influence as backend
    monkeypatch.setattr(backend, "context_answer_influence", lambda *_args, **_kwargs: broken)

    status, out = _post(f"/runs/{rid}/influence-map")

    assert status == 500
    assert out == broken
    assert "influence_map" not in runlog.get_run(rid)


def test_influence_map_schema_violation_is_a_loud_500_not_a_silent_pass(isolated, monkeypatch):
    """A backend that returns a shape its OWN schema rejects (here: an unknown error.code) must fail
    loudly at the write boundary, never persist, and never be handed to a caller as if it were trustworthy
    evidence (roadmap rule 3: no silent fallback)."""
    rid = _seed()
    malformed = {
        "schema": "clozn.context_answer_influence.v1",
        "status": "unavailable",
        "available": False,
        "method": _METHOD_STUB,
        "identity": {"run_id": rid},
        "error": {"code": "the_worker_felt_shy_today", "message": "not a real code"},
    }
    import clozn.receipts.context_answer_influence as backend
    monkeypatch.setattr(backend, "context_answer_influence", lambda *_args, **_kwargs: malformed)

    status, out = _post(f"/runs/{rid}/influence-map")

    assert status == 500
    assert "schema" in out["error"]
    assert "influence_map" not in runlog.get_run(rid)


# ------------------------------------------------------------ GET export path (Phase 3.7 persistence)

def test_get_influence_map_returns_the_persisted_artifact_without_a_worker(isolated, monkeypatch):
    """The export path is a pure journal read: it must work even with no substrate attached, and must
    never trigger a new scoring job -- only POST computes."""
    rid = _seed()
    stored = {
        "schema": "clozn.context_answer_influence.v1", "status": "ok", "available": True,
        "prompt_spans": [{"id": "p.m000.c000", "text": "x"}],
        "answer_spans": [{"id": "a.t0000", "text": "y"}],
        "matrix": [[0.3]],
    }
    run = runlog.get_run(rid)
    run["influence_map"] = stored
    assert runlog.replace_run(run)
    monkeypatch.setattr(cs, "SUB", None)     # no worker at all -- GET must still succeed

    status, out = _get(f"/runs/{rid}/influence-map")

    assert status == 200
    assert out == stored


def test_get_influence_map_is_honest_when_nothing_has_been_computed_yet(isolated):
    rid = _seed()
    status, out = _get(f"/runs/{rid}/influence-map")
    assert status == 404
    assert out["available"] is False
    assert out["schema"] == "clozn.context_answer_influence.v1"


def test_get_influence_map_missing_run_is_404(isolated):
    status, out = _get("/runs/missing/influence-map")
    assert status == 404 and out == {"error": "run not found"}


def test_get_influence_map_does_not_return_a_failed_unavailable_artifact(isolated):
    """A run whose POST attempt failed never had `influence_map` attached at all (see
    test_influence_map_failure_is_not_mistaken_for_a_saved_receipt) -- GET must report the same honest
    "nothing computed yet" rather than surfacing a stale/failed shape."""
    rid = _seed()
    status, out = _get(f"/runs/{rid}/influence-map")
    assert status == 404
    assert out["available"] is False
    assert "influence_map" not in runlog.get_run(rid)
