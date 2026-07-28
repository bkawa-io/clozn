from __future__ import annotations

import io
import json
import threading
import time

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
    from clozn.server.influence_jobs import JOBS

    JOBS.clear_for_tests()
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", ScoreSub())
    yield tmp_path
    JOBS.clear_for_tests()


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
    from clozn.receipts.context_answer_influence import cache_identity
    run["influence_map"] = {
        "schema": "clozn.context_answer_influence.v1",
        "available": True,
        "cache_identity": cache_identity(run),
    }
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


def test_influence_map_cache_invalidates_when_receipt_changes(isolated, monkeypatch):
    rid = _seed()
    run = runlog.get_run(rid)
    from clozn.receipts.context_answer_influence import cache_identity
    run["context_receipt"] = {"schema_version": "clozn.context-receipt.v1", "run_id": rid}
    run["influence_map"] = {
        "schema": "clozn.context_answer_influence.v1",
        "available": True,
        "cache_identity": cache_identity(run),
    }
    run["context_receipt"]["privacy"] = "metadata_only"
    assert runlog.replace_run(run)

    import clozn.receipts.context_answer_influence as backend
    calls = []

    def recompute(current, *_args, **_kwargs):
        calls.append(current["context_receipt"])
        return {
            "schema": "clozn.context_answer_influence.v1",
            "status": "ok",
            "available": True,
            "method": _METHOD_STUB,
            "identity": {"run_id": rid},
            "cache_identity": cache_identity(current),
        }

    monkeypatch.setattr(backend, "context_answer_influence", recompute)
    status, _out = _post(f"/runs/{rid}/influence-map")
    assert status == 200
    assert len(calls) == 1


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


def test_influence_map_portable_export_defaults_to_metadata_only_without_text(isolated):
    rid = _seed()
    secret_source = "private source sentence"
    secret_answer = "private answer"
    run = runlog.get_run(rid)
    run["influence_map"] = {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": _METHOD_STUB,
        "identity": {"run_id": rid},
        "cache_identity": {},
        "thresholds": {"cell_abs_delta_nats": 0.05},
        "prompt_sources": [{
            "id": "p.m000", "start": 0, "end": len(secret_source),
            "text": secret_source, "segment_id": "seg-1", "client_source_id": "doc-1",
            "selected": True,
        }, {
            "id": "p.m001", "start": 0, "end": 14,
            "text": "omitted secret", "segment_id": "seg-2", "client_source_id": "doc-2",
            "selected": False,
        }],
        "selection": {
            "strategy": "bounded_test",
            "max_context_spans": 1,
            "selected_source_ids": ["p.m000"],
            "omitted_source_ids": ["p.m001"],
            "measured_span_count": 1,
            "complete_for_selected_spans": True,
        },
        "prompt_spans": [{
            "id": "p.m000.c000", "start": 0, "end": len(secret_source),
            "text": secret_source, "segment_id": "seg-1", "client_source_id": "doc-1",
        }],
        "answer_spans": [{
            "id": "a.t0000", "start": 0, "end": len(secret_answer), "text": secret_answer,
        }],
        "answer": {"recorded_text": secret_answer},
        "links": [],
        "summary": {"no_clear_source": True},
    }
    assert runlog.replace_run(run)

    status, out = _get(f"/runs/{rid}/influence-map/export")
    encoded = json.dumps(out)

    assert status == 200
    assert out["privacy"] == "metadata_only"
    assert secret_source not in encoded
    assert secret_answer not in encoded
    assert "omitted secret" not in encoded
    assert [source["client_source_id"] for source in out["prompt_sources"]] == ["doc-1", "doc-2"]
    assert out["prompt_sources"][1]["selected"] is False
    assert out["selection"]["omitted_source_ids"] == ["p.m001"]
    assert len(out["prompt_sources"][1]["text_sha256"]) == 64
    assert out["prompt_spans"][0]["segment_id"] == "seg-1"
    assert out["prompt_spans"][0]["client_source_id"] == "doc-1"
    assert len(out["prompt_spans"][0]["text_sha256"]) == 64


def test_influence_map_full_text_export_requires_explicit_post_opt_in(isolated):
    rid = _seed()
    run = runlog.get_run(rid)
    run["influence_map"] = {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": _METHOD_STUB,
        "identity": {"run_id": rid},
        "cache_identity": {},
        "thresholds": {},
        "prompt_sources": [{
            "id": "p1", "start": 0, "end": 6, "text": "secret", "selected": True,
        }],
        "selection": {
            "strategy": "test",
            "max_context_spans": 1,
            "selected_source_ids": ["p1"],
            "omitted_source_ids": [],
            "measured_span_count": 1,
            "complete_for_selected_spans": True,
        },
        "prompt_spans": [{"id": "p1", "start": 0, "end": 6, "text": "secret"}],
        "answer_spans": [{"id": "a1", "start": 0, "end": 6, "text": "answer"}],
        "links": [],
        "summary": {},
    }
    assert runlog.replace_run(run)

    status, out = _post(f"/runs/{rid}/influence-map/export", {"privacy": "full"})

    assert status == 200
    assert out["privacy"] == "full"
    assert out["prompt_sources"][0]["text"] == "secret"
    assert out["prompt_spans"][0]["text"] == "secret"
    assert out["answer_spans"][0]["text"] == "answer"


# ----------------------------------------------------------- bounded async job lifecycle

def _wait_for_job(rid: str, job_id: str, states: set[str], timeout: float = 2.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        status, last = _get(f"/runs/{rid}/influence-map/jobs/{job_id}")
        assert status == 200
        if last["state"] in states:
            return last
        time.sleep(0.01)
    pytest.fail(f"job did not reach {states}; last={last}")


def _available_map(rid: str) -> dict:
    return {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": _METHOD_STUB,
        "identity": {"run_id": rid},
        "prompt_spans": [],
        "answer_spans": [],
        "links": [],
    }


def test_influence_job_reports_coarse_refine_progress_and_persists(isolated, monkeypatch):
    rid = _seed()
    refine_reached = threading.Event()
    release = threading.Event()
    import clozn.receipts.context_answer_influence as backend

    def measured(_run, _sub, **options):
        options["progress"](phase="coarse", completed=2, total=4)
        options["progress"](phase="refine", completed=1, total=3)
        refine_reached.set()
        assert release.wait(2)
        return _available_map(rid)

    monkeypatch.setattr(backend, "context_answer_influence", measured)
    status, started = _post(f"/runs/{rid}/influence-map/jobs")
    assert status == 202
    assert started["state"] in {"queued", "running"}
    assert refine_reached.wait(2)

    status, progress = _get(
        f"/runs/{rid}/influence-map/jobs/{started['job_id']}")
    assert status == 200
    assert progress["state"] == "running"
    assert progress["progress"] == {
        "phase": "refine",
        "completed_units": 1,
        "total_units": 3,
        "percent": 33.3,
    }

    release.set()
    done = _wait_for_job(rid, started["job_id"], {"completed"})
    assert done["progress"]["phase"] == "done"
    assert done["cancellable"] is False
    assert runlog.get_run(rid)["influence_map"]["available"] is True


def test_influence_job_cancel_checkpoint_prevents_persistence(isolated, monkeypatch):
    rid = _seed()
    scoring = threading.Event()
    import clozn.receipts.context_answer_influence as backend

    def measured(_run, _sub, **options):
        options["progress"](phase="coarse", completed=1, total=4)
        scoring.set()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            if options["cancel_requested"]():
                raise backend.InfluenceComputationCancelled("cancelled in test")
            time.sleep(0.005)
        pytest.fail("test job never observed cancellation")

    monkeypatch.setattr(backend, "context_answer_influence", measured)
    status, started = _post(f"/runs/{rid}/influence-map/jobs")
    assert status == 202
    assert scoring.wait(2)

    status, cancelled = _post(
        f"/runs/{rid}/influence-map/jobs/{started['job_id']}/cancel")
    assert status == 200
    assert cancelled["cancel_requested"] is True
    assert cancelled["cancel_accepted"] is True
    terminal = _wait_for_job(rid, started["job_id"], {"cancelled"})
    assert terminal["cancellable"] is False
    assert "influence_map" not in runlog.get_run(rid)


def test_influence_job_repeated_cancel_is_idempotent(isolated, monkeypatch):
    rid = _seed()
    scoring = threading.Event()
    import clozn.receipts.context_answer_influence as backend

    def measured(_run, _sub, **options):
        scoring.set()
        while not options["cancel_requested"]():
            time.sleep(0.005)
        raise backend.InfluenceComputationCancelled("cancelled")

    monkeypatch.setattr(backend, "context_answer_influence", measured)
    status, started = _post(f"/runs/{rid}/influence-map/jobs")
    assert status == 202 and scoring.wait(2)
    cancel_path = f"/runs/{rid}/influence-map/jobs/{started['job_id']}/cancel"
    status, first = _post(cancel_path)
    assert status == 200 and first["cancel_accepted"] is True
    status, second = _post(cancel_path)
    assert status == 200
    assert second["cancel_requested"] is True
    assert second["cancel_accepted"] is False
    terminal = _wait_for_job(rid, started["job_id"], {"cancelled"})

    status, third = _post(cancel_path)
    assert status == 200
    assert third["state"] == terminal["state"] == "cancelled"
    assert third["cancel_accepted"] is False
    assert "influence_map" not in runlog.get_run(rid)


def test_async_cached_job_and_synchronous_post_remain_compatible(isolated, monkeypatch):
    rid = _seed()
    run = runlog.get_run(rid)
    from clozn.receipts.context_answer_influence import cache_identity

    stored = {
        **_available_map(rid),
        "cache_identity": cache_identity(run),
    }
    run["influence_map"] = stored
    assert runlog.replace_run(run)
    import clozn.receipts.context_answer_influence as backend
    monkeypatch.setattr(
        backend,
        "context_answer_influence",
        lambda *_args, **_kwargs: pytest.fail("valid cached maps must not rescore"),
    )

    status, job = _post(f"/runs/{rid}/influence-map/jobs")
    assert status == 202
    assert job["state"] == "completed"
    assert job["cached"] is True
    status, sync = _post(f"/runs/{rid}/influence-map")
    assert status == 200
    assert sync == stored
