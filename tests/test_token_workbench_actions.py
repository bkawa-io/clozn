"""Milestone F: the token-workbench ACTION endpoints, model-free.

The engine and the job executor are both stubbed: FakeEngine/FakeSub never call a real model, and
clozn.analysis.tracer.trace is monkeypatched (mirrors tests/test_influence_map_server.py's own
monkeypatching of clozn.receipts.context_answer_influence.context_answer_influence). Jobs run for real
through clozn.server.influence_jobs.JOBS -- a real bounded ThreadPoolExecutor -- because that IS the
job system under test; `_wait_for_job` polls it the same way test_influence_map_server.py does.

Covers, per the coordinator's contract:
  * every action resolves to exactly one of the three shapes (cached / job / unavailable) and never a
    bare {"error": ...} for an "unavailable" determination -- see test_*_never_a_bare_error below.
  * a repeat request (same run + index + params) hits the cache rather than recomputing, for fork
    (an existing child run) and causal-trace (the new clozn.token-workbench-action.v1 entry).
  * a running job is genuinely cancellable: the cancel call returns immediately, and the job never
    reports "completed" nor persists a result once cancelled, even though the underlying computation
    (which has no cooperative cancellation hook) keeps running in its own thread until it returns.
  * mechanistic-diff refuses with pair_compatibility's own typed reason when it does not permit the
    comparison, and separately reports an honest "not yet wired" reason when it DOES permit one (see
    clozn.runs.token_workbench_actions's module docstring for why cross-model execution itself is out
    of scope this wave).
"""
from __future__ import annotations

import threading
import time

import pytest

import clozn.runs.store as runlog
from clozn.server.influence_jobs import JOBS
from clozn.server.routes import token_workbench_actions as route


RUNTIME = {
    "model_sha256": "a" * 64,
    "template_fingerprint": "b" * 16,
    "engine_build": "test-build",
    "context_size": 4096,
    "backend": "cpu",
    "adapter": {"present": False, "identity_sha256": None, "artifact_sha256": None, "scale": None},
    "white_box_flags": {},
}
WORKER = {
    "worker_id": "generation-a",
    "worker_generation_id": "generation-a",
    "protocol_version": "1.1",
}
_METHOD_STUB = {
    "name": "teacher_forced_matched_context_replacement",
    "mode": "forced_score_intervention",
    "claim_limit": "behavioral dependence under a controlled prompt intervention",
    "caveat": "Influence means this context changed the measured output under this intervention.",
}


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    JOBS.clear_for_tests()
    yield tmp_path
    JOBS.clear_for_tests()


def _organic_run(**overrides):
    values = {
        "source": "engine_chat",
        "client": "studio",
        "model": "fixture-model",
        "substrate": "engine",
        "messages": [{"role": "user", "content": "count"}],
        "response": "one two three",
        "final_prompt": "<prompt>",
        "trace": {
            "tokens": ["one", " two", " three"],
            "token_ids": [11, 22, 33],
            "alternatives": [[], [{"piece": " four", "token_id": 44, "prob": 0.02}], []],
        },
        "meta": {"n_ctx": RUNTIME["context_size"], "device": RUNTIME["backend"]},
        "identity": {
            "model_sha256": RUNTIME["model_sha256"],
            "template_fingerprint": RUNTIME["template_fingerprint"],
            "engine_build": RUNTIME["engine_build"],
            "white_box_flags": dict(RUNTIME["white_box_flags"]),
        },
    }
    values.update(overrides)
    run_id = runlog.record(**values)
    assert run_id
    return runlog.get_run(run_id)


class FakeEngine:
    def __init__(self, complete_text=" and beyond", complete_finish="stop"):
        self.complete_text = complete_text
        self.complete_finish = complete_finish
        self.calls = []

    def complete(self, prompt, **params):
        self.calls.append(dict(params, prompt=prompt))
        return {"choices": [{"text": self.complete_text, "finish_reason": self.complete_finish}]}


class FakeSub:
    def __init__(self, engine=None, *, runtime=RUNTIME, worker=WORKER, score_tokens=None):
        self.engine = engine if engine is not None else FakeEngine()
        self.steer = None
        self.memory = None
        self.runtime_identity = lambda: dict(runtime)
        self.worker_identity = lambda: dict(worker)
        if score_tokens is not None:
            self.score_tokens = score_tokens


class Handler:
    def __init__(self, sub=None, path=""):
        self._inj_sub = sub
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _post(sub, run_id, index, action, body):
    h = Handler(sub)
    claimed = route.try_post(h, f"/runs/{run_id}/tokens/{index}/{action}", body)
    return claimed, h


def _wait_for_job(run_id, job_id, states, timeout=2.0):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        last = JOBS.get(run_id, job_id)
        assert last is not None
        if last["state"] in states:
            return last
        time.sleep(0.01)
    pytest.fail(f"job did not reach {states}; last={last}")


# =================================================================================== fork
def test_fork_starts_a_job_that_completes_and_embeds_compat_forks_own_outcome(stores):
    run = _organic_run()
    sub = FakeSub()
    claimed, h = _post(sub, run["id"], 1, "fork", {"position": 1, "token_id": 44})

    assert claimed is True
    assert h.status == 202
    assert h.body["outcome"] == "job"
    assert h.body["job"]["kind"] == "fork"
    final = _wait_for_job(run["id"], h.body["job"]["job_id"], {"completed", "failed", "cancelled"})
    assert final["state"] == "completed"
    assert final["result"]["outcome"] == "reconstructed_replay"   # no exact checkpoint machinery here
    assert final["result"]["parent_run_id"] == run["id"]


def test_fork_repeat_request_hits_the_cache_not_a_new_job(stores):
    run = _organic_run()
    sub = FakeSub()
    _claimed, h = _post(sub, run["id"], 1, "fork", {"position": 1, "token_id": 44})
    first = _wait_for_job(run["id"], h.body["job"]["job_id"], {"completed"})
    child_id = first["result"]["id"]

    claimed2, h2 = _post(sub, run["id"], 1, "fork", {"position": 1, "token_id": 44})
    assert claimed2 is True
    assert h2.status == 200
    assert h2.body["outcome"] == "cached"
    assert h2.body["artifact"]["id"] == child_id


def test_fork_no_engine_is_a_hard_503_not_the_three_way_contract(stores):
    run = _organic_run()

    class NoEngine:
        engine = None

    claimed, h = _post(NoEngine(), run["id"], 1, "fork", {"position": 1, "token": "x"})
    assert claimed is True
    assert h.status == 503


def test_fork_bad_index_and_bad_token_are_400(stores):
    run = _organic_run()
    sub = FakeSub()
    claimed, h = _post(sub, run["id"], 99, "fork", {"position": 99, "token": "x"})
    assert claimed is True and h.status == 400
    assert "out of range" in h.body["error"]

    claimed2, h2 = _post(sub, run["id"], 1, "fork", {"position": 1})
    assert claimed2 is True and h2.status == 400


# =================================================================================== causal-trace
def test_causal_trace_starts_a_job_and_the_completed_job_carries_the_result(stores, monkeypatch):
    run = _organic_run()
    sub = FakeSub()
    import clozn.analysis.tracer as tracer

    def fake_trace(prompt, continuation, target_idx, **kwargs):
        return {"ok": True, "target_idx": target_idx, "verdict": "PASS", "nodes": []}

    monkeypatch.setattr(tracer, "trace", fake_trace)

    claimed, h = _post(sub, run["id"], 1, "causal-trace", {"seed": 0, "screen_mode": "ablate"})
    assert claimed is True
    assert h.status == 202
    assert h.body["outcome"] == "job"
    assert h.body["job"]["kind"] == "causal_trace"
    final = _wait_for_job(run["id"], h.body["job"]["job_id"], {"completed", "failed", "cancelled"})
    assert final["state"] == "completed"
    assert final["result"]["outcome"] == "ok"
    assert final["result"]["result"]["verdict"] == "PASS"

    # persisted onto the run for future cache lookups
    stored = runlog.get_run(run["id"])
    assert len(stored["token_workbench_actions"]) == 1
    assert stored["token_workbench_actions"][0]["cache_key"] == final["result"]["cache_key"]


def test_causal_trace_ok_false_blocked_still_completes_the_job_not_fails_it(stores, monkeypatch):
    """tracer.trace()'s own convention: a completed analysis that reports "couldn't" is a successful
    outcome, not a failed request -- the job mirrors that, exactly like the existing synchronous
    POST /runs/<id>/causal-trace route already does."""
    run = _organic_run()
    sub = FakeSub()
    import clozn.analysis.tracer as tracer

    monkeypatch.setattr(
        tracer, "trace", lambda *_a, **_k: {"ok": False, "blocked": "no sidecar available"})

    claimed, h = _post(sub, run["id"], 1, "causal-trace", {})
    final = _wait_for_job(run["id"], h.body["job"]["job_id"], {"completed", "failed", "cancelled"})
    assert final["state"] == "completed"
    assert final["result"]["outcome"] == "blocked"
    assert final["result"]["result"]["blocked"] == "no sidecar available"


def test_causal_trace_repeat_request_hits_the_cache(stores, monkeypatch):
    run = _organic_run()
    sub = FakeSub()
    import clozn.analysis.tracer as tracer

    calls = []

    def fake_trace(prompt, continuation, target_idx, **kwargs):
        calls.append(target_idx)
        return {"ok": True, "target_idx": target_idx, "verdict": "PASS", "nodes": []}

    monkeypatch.setattr(tracer, "trace", fake_trace)

    _claimed, h = _post(sub, run["id"], 1, "causal-trace", {"seed": 0})
    _wait_for_job(run["id"], h.body["job"]["job_id"], {"completed"})
    assert len(calls) == 1

    claimed2, h2 = _post(sub, run["id"], 1, "causal-trace", {"seed": 0})
    assert claimed2 is True
    assert h2.status == 200
    assert h2.body["outcome"] == "cached"
    assert len(calls) == 1  # never recomputed

    # a DIFFERENT seed is different evidence -- not a cache hit
    claimed3, h3 = _post(sub, run["id"], 1, "causal-trace", {"seed": 7})
    assert h3.status == 202 and h3.body["outcome"] == "job"
    _wait_for_job(run["id"], h3.body["job"]["job_id"], {"completed"})
    assert len(calls) == 2


def test_causal_trace_missing_final_prompt_is_typed_unavailable(stores):
    run = _organic_run(final_prompt=None)
    sub = FakeSub()
    claimed, h = _post(sub, run["id"], 1, "causal-trace", {})
    assert claimed is True
    assert h.status == 422
    assert h.body["outcome"] == "unavailable"
    assert h.body["reason"]["code"] == "missing_final_prompt"


def test_causal_trace_bad_screen_mode_is_400(stores):
    run = _organic_run()
    sub = FakeSub()
    claimed, h = _post(sub, run["id"], 1, "causal-trace", {"screen_mode": "not_a_mode"})
    assert claimed is True and h.status == 400


def test_causal_trace_no_engine_is_503(stores):
    run = _organic_run()

    class NoEngine:
        engine = None

    claimed, h = _post(NoEngine(), run["id"], 1, "causal-trace", {})
    assert claimed is True and h.status == 503


# =================================================================================== source-measure
def test_source_measure_delegates_to_the_influence_map_job_system(stores, monkeypatch):
    run = _organic_run()
    sub = FakeSub(score_tokens=lambda *_a, **_k: [])
    import clozn.receipts.context_answer_influence as backend

    def measured(_run, _sub, **options):
        return {
            "schema": "clozn.context_answer_influence.v1", "status": "ok", "available": True,
            "method": dict(_METHOD_STUB),
            "identity": {"run_id": run["id"]},
            "prompt_spans": [], "answer_spans": [], "links": [],
        }

    monkeypatch.setattr(backend, "context_answer_influence", measured)

    claimed, h = _post(sub, run["id"], 1, "source-measure", {})
    assert claimed is True
    assert h.status == 202
    assert h.body["outcome"] == "job"
    assert h.body["job"]["kind"] == "influence_map"  # the SAME kind influence-map jobs already use
    final = _wait_for_job(run["id"], h.body["job"]["job_id"], {"completed", "failed", "cancelled"})
    assert final["state"] == "completed"
    assert runlog.get_run(run["id"])["influence_map"]["available"] is True


def test_source_measure_cache_hit_reuses_influence_maps_own_cache(stores):
    run = _organic_run()
    stored_map = {
        "schema": "clozn.context_answer_influence.v1", "status": "ok", "available": True,
        "method": dict(_METHOD_STUB),
        "identity": {"run_id": run["id"]},
        "prompt_spans": [], "answer_spans": [], "links": [],
    }
    from clozn.receipts.context_answer_influence import cache_identity

    stored_map["cache_identity"] = cache_identity(run)
    updated = dict(run)
    updated["influence_map"] = stored_map
    assert runlog.replace_run(updated)
    run = runlog.get_run(run["id"])

    sub = FakeSub(score_tokens=lambda *_a, **_k: [])
    claimed, h = _post(sub, run["id"], 1, "source-measure", {})
    assert claimed is True
    assert h.status == 200
    assert h.body["outcome"] == "cached"
    assert h.body["artifact"]["available"] is True


def test_source_measure_no_scoring_worker_is_503(stores):
    run = _organic_run()

    class NoScoring:
        engine = FakeEngine()

    claimed, h = _post(NoScoring(), run["id"], 1, "source-measure", {})
    assert claimed is True and h.status == 503


# =================================================================================== mechanistic-diff
def test_mechanistic_diff_no_reference_is_typed_unavailable(stores):
    run = _organic_run()
    claimed, h = _post(FakeSub(), run["id"], 1, "mechanistic-diff", {})
    assert claimed is True
    assert h.status == 422
    assert h.body["outcome"] == "unavailable"
    assert h.body["reason"]["code"] == "reference_run_required"


def test_mechanistic_diff_reference_not_found_is_typed_unavailable(stores):
    run = _organic_run()
    claimed, h = _post(FakeSub(), run["id"], 1, "mechanistic-diff", {"reference_run_id": "run_missing"})
    assert claimed is True
    assert h.status == 422
    assert h.body["reason"]["code"] == "reference_run_not_found"


def test_mechanistic_diff_refuses_with_pair_compatibilitys_own_typed_reason(stores):
    """The REQUIRED scenario: mechanistic-diff refuses with a typed reason when pair compatibility does
    not permit the operation. Typical recorded runs carry only the lightweight reproduction identity
    (model_sha256/template_fingerprint/engine_build), not the full GGUF-header identity pair_
    compatibility needs -- so it honestly reports "unknown" and refuses, exactly as documented."""
    run = _organic_run()
    reference = _organic_run(model="other-model", identity={
        "model_sha256": "c" * 64, "template_fingerprint": "b" * 16,
        "engine_build": "test-build", "white_box_flags": {},
    })

    claimed, h = _post(FakeSub(), run["id"], 1, "mechanistic-diff", {"reference_run_id": reference["id"]})
    assert claimed is True
    assert h.status == 422
    assert h.body["outcome"] == "unavailable"
    assert h.body["reason"]["code"] == "pair_compatibility_refused"
    assert h.body["reason"]["message"]  # pair_compatibility's own operation reason text, non-empty
    assert h.body["pair_compatibility"]["schema_version"] == "clozn.pair-compatibility.v1"
    operations = h.body["pair_compatibility"]["verdict"]["operations"]
    assert operations["per_token_comparison"]["permitted"] is False
    assert operations["residual_transplant"]["permitted"] is False


def test_mechanistic_diff_permitted_pair_is_still_honestly_unavailable_this_milestone(stores):
    """A genuinely compatible pair (full GGUF-header identity, matching) is NOT refused by pair
    compatibility -- but actually executing a cross-model diff needs infrastructure under concurrent
    development this wave, so the honest outcome is still `unavailable`, with a DIFFERENT reason code
    that says so explicitly rather than silently reusing the refusal code."""
    gguf_identity = {
        "architecture": "qwen2", "layer_count": 28, "hidden_size": 1536, "vocab_size": 151936,
        "head_count": 12, "tokenizer_sha256": "e" * 64, "chat_template_sha256": "f" * 64,
    }
    run = _organic_run(identity=dict(gguf_identity))
    reference = _organic_run(model="other-model", identity=dict(gguf_identity))

    claimed, h = _post(FakeSub(), run["id"], 1, "mechanistic-diff", {"reference_run_id": reference["id"]})
    assert claimed is True
    assert h.status == 422
    assert h.body["outcome"] == "unavailable"
    assert h.body["reason"]["code"] == "cross_model_execution_not_wired"
    assert h.body["pair_compatibility"]["verdict"]["operations"]["per_token_comparison"]["permitted"] is True


# =================================================================================== job status/cancel
def test_job_status_and_unknown_job_404(stores, monkeypatch):
    run = _organic_run()
    sub = FakeSub()
    import clozn.analysis.tracer as tracer

    monkeypatch.setattr(
        tracer, "trace", lambda *_a, **_k: {"ok": True, "target_idx": 1, "verdict": "PASS", "nodes": []})
    _claimed, h = _post(sub, run["id"], 1, "causal-trace", {})
    job_id = h.body["job"]["job_id"]

    status_h = Handler()
    assert route.try_get(status_h, f"/runs/{run['id']}/tokens/1/jobs/{job_id}") is True
    assert status_h.status == 200
    assert status_h.body["job_id"] == job_id

    missing_h = Handler()
    assert route.try_get(missing_h, f"/runs/{run['id']}/tokens/1/jobs/nope") is True
    assert missing_h.status == 404


def test_running_job_is_genuinely_cancellable(stores, monkeypatch):
    """The REQUIRED scenario: a job is cancellable. Cancel is accepted and the response returns
    immediately (it does not block on the still-running background thread), and the job never reports
    "completed" nor persists a result once cancelled -- even though tracer.trace() itself has no
    cooperative cancellation hook and keeps running until it naturally returns."""
    run = _organic_run()
    sub = FakeSub()
    started = threading.Event()
    release = threading.Event()
    import clozn.analysis.tracer as tracer

    def blocking_trace(*_args, **_kwargs):
        started.set()
        assert release.wait(5), "test never released the blocked trace"
        return {"ok": True, "target_idx": 1, "verdict": "PASS", "nodes": []}

    monkeypatch.setattr(tracer, "trace", blocking_trace)

    claimed, h = _post(sub, run["id"], 1, "causal-trace", {})
    assert claimed is True
    job_id = h.body["job"]["job_id"]
    assert started.wait(2)

    cancel_h = Handler()
    t0 = time.monotonic()
    assert route.try_post(
        cancel_h, f"/runs/{run['id']}/tokens/1/jobs/{job_id}/cancel", {}) is True
    elapsed = time.monotonic() - t0
    assert elapsed < 1.0, "cancel must return immediately, not wait for the background thread"
    assert cancel_h.status == 200
    assert cancel_h.body["cancel_accepted"] is True
    assert cancel_h.body["cancellable"] is True  # still running, but a cancel is now in flight

    release.set()
    final = _wait_for_job(run["id"], job_id, {"cancelled", "completed", "failed"}, timeout=3.0)
    assert final["state"] == "cancelled"
    assert "result" not in final
    assert runlog.get_run(run["id"]).get("token_workbench_actions") in (None, [])


def test_repeated_cancel_is_idempotent(stores, monkeypatch):
    run = _organic_run()
    sub = FakeSub()
    started = threading.Event()
    release = threading.Event()
    import clozn.analysis.tracer as tracer

    def blocking_trace(*_args, **_kwargs):
        started.set()
        release.wait(5)
        return {"ok": True, "target_idx": 1, "verdict": "PASS", "nodes": []}

    monkeypatch.setattr(tracer, "trace", blocking_trace)
    _claimed, h = _post(sub, run["id"], 1, "causal-trace", {})
    job_id = h.body["job"]["job_id"]
    assert started.wait(2)

    first = Handler()
    route.try_post(first, f"/runs/{run['id']}/tokens/1/jobs/{job_id}/cancel", {})
    second = Handler()
    route.try_post(second, f"/runs/{run['id']}/tokens/1/jobs/{job_id}/cancel", {})
    assert first.body["cancel_accepted"] is True
    assert second.body["cancel_accepted"] is False
    release.set()
    _wait_for_job(run["id"], job_id, {"cancelled"})


# =================================================================================== never a bare error
@pytest.mark.parametrize(
    ("action", "body", "run_kwargs"),
    [
        ("causal-trace", {}, {"final_prompt": None}),
        ("mechanistic-diff", {}, {}),
        ("mechanistic-diff", {"reference_run_id": "run_missing"}, {}),
    ],
)
def test_unavailable_outcomes_are_never_a_bare_error(stores, action, body, run_kwargs):
    run = _organic_run(**run_kwargs)
    claimed, h = _post(FakeSub(), run["id"], 1, action, body)
    assert claimed is True
    assert h.status == 422
    assert h.body["outcome"] == "unavailable"
    assert isinstance(h.body["reason"], dict)
    assert h.body["reason"]["code"]
    assert h.body["reason"]["message"]
    assert "error" not in h.body  # never a bare {"error": ...} standing in for a typed outcome


def test_every_action_reaches_one_of_the_three_outcomes(stores, monkeypatch):
    """Cheap end-to-end sanity: for a fully eligible run, all four actions produce a top-level
    `outcome` in {"cached", "job", "unavailable"} -- never anything else."""
    run = _organic_run()
    sub = FakeSub(score_tokens=lambda *_a, **_k: [])
    import clozn.analysis.tracer as tracer

    monkeypatch.setattr(
        tracer, "trace", lambda *_a, **_k: {"ok": True, "target_idx": 1, "verdict": "PASS", "nodes": []})
    import clozn.receipts.context_answer_influence as backend

    monkeypatch.setattr(backend, "context_answer_influence", lambda *_a, **_k: {
        "schema": "clozn.context_answer_influence.v1", "status": "ok", "available": True,
        "method": dict(_METHOD_STUB),
        "identity": {"run_id": run["id"]}, "prompt_spans": [], "answer_spans": [], "links": [],
    })

    for action, body in (
        ("fork", {"position": 1, "token_id": 44}),
        ("causal-trace", {}),
        ("source-measure", {}),
        ("mechanistic-diff", {}),
    ):
        _claimed, h = _post(sub, run["id"], 1, action, body)
        assert h.body.get("outcome") in {"cached", "job", "unavailable"}, (action, h.body)


# =================================================================================== routing/registration
def test_unrelated_paths_are_not_claimed(stores):
    h = Handler()
    for path in (
        "/runs/run_x/tokens/0",            # no trailing action
        "/runs/run_x/workbench",           # missing /tokens/<index>
        "/runs/run_x/tokens/0/unknown-action",
        "/runs/run_x/investigation",
    ):
        assert route.try_post(h, path, {}) is False
    assert route.try_get(h, "/runs/run_x/tokens/0/workbench") is False


def test_autoload_registration():
    from clozn.server import app as server

    assert route in server._POST_ROUTES
    assert route in server._GET_ROUTES
    assert server._GET_ROUTES.index(route) < server._GET_ROUTES.index(server._runs_fallback_routes)
