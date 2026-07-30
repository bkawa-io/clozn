"""Model-free FORK-01 execution, persistence, and gateway coverage."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from clozn import schemas
import clozn.runs.store as runlog
from clozn.replay.execution_fork import plan_execution_fork
from clozn.replay.execution_fork_execute import execute_exact_fork
from clozn.replay import execution_fork_results
from clozn.server.routes import execution_fork as route


RUNTIME = {
    "model_sha256": "a" * 64,
    "template_fingerprint": "b" * 16,
    "engine_build": "test-build",
    "context_size": 4096,
    "backend": "cpu",
    "adapter": {
        "present": False,
        "identity_sha256": None,
        "artifact_sha256": None,
        "scale": None,
    },
    "white_box_flags": {},
}
WORKER = {
    "worker_id": "generation-a",
    "worker_generation_id": "generation-a",
    "protocol_version": "1.1",
}


def _sha(value):
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        execution_fork_results, "RESULTS_DIR", str(tmp_path / "execution-forks"))
    return tmp_path


def _parent():
    run_id = runlog.record(
        source="engine_chat",
        client="studio",
        model="fixture-model",
        substrate="engine",
        messages=[{"role": "user", "content": "count"}],
        assembled_messages=[{"role": "user", "content": "count"}],
        response="one two three",
        final_prompt="<prompt>",
        trace={"tokens": ["one", " two", " three"], "token_ids": [11, 22, 33]},
        meta={"n_ctx": 4096, "device": "cpu"},
        identity={
            "model_sha256": "a" * 64,
            "template_fingerprint": "b" * 16,
            "engine_build": "test-build",
            "white_box_flags": {},
        },
    )
    assert run_id
    return runlog.get_run(run_id)


def _plan(parent, *, position=1, change=None, worker=None):
    selected_worker = worker or WORKER
    return plan_execution_fork(
        parent,
        {
            "position": position,
            "change": change or {
                "type": "force_token", "token_id": 44, "token_piece": " four"},
        },
        checkpoint={
            "checkpoint_id": f"ckpt-{selected_worker['worker_generation_id']}-7",
            "worker_generation_id": selected_worker["worker_generation_id"],
            "state": "available",
            "parent_run_id": parent["id"],
            "prompt_tokens": 10,
            "n_past": 13,
        },
        runtime_identity=RUNTIME,
        worker_identity=selected_worker,
    )


def _reply(plan, *, tokens, text, intervention_type, generation="generation-a", applied=None):
    live = plan["exactness"]["regime"] == "generated_token_live_kv"
    return {
        "worker_generation_id": generation,
        "text": text,
        "tokens": tokens,
        "prompt_len": plan["checkpoint_reference"]["prompt_tokens"],
        "n_past_restored": plan["exactness"]["truncate_to"],
        "restore_mode": "live_kv_truncated" if live else "reprefill",
        "exactness": {
            "source": "live_kv" if live else "reprefill",
            "truncation_regime": "generated_token" if live else "prompt_boundary",
            "boundary_shape_true": True,
        },
        "sampler_source": "checkpoint",
        "steer_source": "none",
        "intervention_applied": applied or {"type": intervention_type},
        "finish_reason": "stop",
    }


class FakeEngine:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def execution_fork(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        return deepcopy(reply)

    def health(self):
        return {
            "worker_generation_id": "generation-a",
            "protocol_version": "1.1",
            "model_sha256": "a" * 64,
            "n_ctx": 4096,
            "device": "cpu",
            "capabilities": {},
        }


def _success_engine(plan):
    return FakeEngine([
        _reply(plan, tokens=[22, 33], text=" two three", intervention_type="none"),
        _reply(
            plan, tokens=[44, 55], text=" four five", intervention_type="force_token",
            applied={"type": "force_token", "token_id": 44}),
    ])


def test_control_runs_first_then_real_child_and_terminal_receipt_are_immutable(stores):
    parent = _parent()
    before = deepcopy(parent)
    plan = _plan(parent)
    engine = _success_engine(plan)

    result = execute_exact_fork(
        parent, plan, engine,
        runtime_identity=RUNTIME, worker_identity=WORKER,
        reload_parent=runlog.get_run,
    )

    receipt, child = result["receipt"], result["child"]
    schemas.validate(receipt)
    assert receipt["phase"] == "completed"
    assert receipt["execution"]["status"] == "succeeded"
    assert receipt["unchanged_control"]["result"]["exact_match"] is True
    assert child["response"] == "one four five"
    assert child["parent_run_id"] == parent["id"]
    assert child["source"] == "fork"
    assert child["execution_fork"] == receipt
    assert receipt["child_lineage"] == {
        "parent_run_id": parent["id"],
        "child_run_id": child["id"],
        "source": "fork",
        "change_sha256": plan["request"]["change_sha256"],
        "receipt_status": "created",
    }
    assert receipt["identity"] == plan["identity"]
    assert receipt["checkpoint_reference"] == plan["checkpoint_reference"]
    assert engine.calls[0]["intervention"] == {"type": "none"}
    assert engine.calls[1]["intervention"] == {"type": "force_token", "token_id": 44}
    assert runlog.get_run(parent["id"]) == before
    assert execution_fork_results.get(receipt["execution_id"]) == receipt


def _routing_artifact(*, worker_id="generation-a", worker_generation=1):
    key_facets = {
        "gguf_artifact_sha256": RUNTIME["model_sha256"],
        "template_fingerprint": RUNTIME["template_fingerprint"],
        "engine_build": RUNTIME["engine_build"],
        "context_size": RUNTIME["context_size"],
        "backend": RUNTIME["backend"],
        "adapter": deepcopy(RUNTIME["adapter"]),
        "white_box_flags": deepcopy(RUNTIME["white_box_flags"]),
    }
    runtime_key = {"key_sha256": _sha(key_facets), **key_facets}
    artifact = {
        "schema_version": "clozn.model-routing.v1",
        "protocol": {"surface": "openai", "route": "/v1/chat/completions"},
        "request": {
            "request_id": "parent-request",
            "requested_model": "fixture-model",
            "selection_source": "explicit",
            "load_policy": "wait",
        },
        "policy": {
            "default_model_id": "fixture-model",
            "max_loaded_workers": 1,
            "preload_model_ids": ["fixture-model"],
            "load_queue_limit": 1,
            "generation_queue_limit": 1,
            "load_timeout_ms": 180000,
            "queue_timeout_ms": 600000,
            "eviction_policy": "lru_idle",
            "active_worker_eviction": "forbidden",
            "cold_load_coalescing": True,
            "cancellation": "request_scoped_release_permits",
            "omitted_model_policy": "configured_default",
            "unknown_model_policy": "error_no_fallback",
            "alias_policy": "mutable_config_immutable_receipt",
        },
        "result": {
            "status": "routed",
            "lifecycle_state": "ready",
            "receipt": {
                "requested_model": "fixture-model",
                "selection_source": "explicit",
                "resolved_model_id": "fixture-model",
                "resolved_artifact": {
                    "model_id": "fixture-model",
                    "format": "gguf",
                    "artifact_sha256": RUNTIME["model_sha256"],
                },
                "runtime_key": runtime_key,
                "worker_identity": {
                    "worker_id": worker_id,
                    "worker_generation": worker_generation,
                    "runtime_key_sha256": runtime_key["key_sha256"],
                    "protocol_version": "1.1",
                    "engine_build": RUNTIME["engine_build"],
                    "backend": RUNTIME["backend"],
                },
                "adapter": deepcopy(RUNTIME["adapter"]),
                "load_event": {
                    "event_id": None,
                    "kind": "not_required",
                    "outcome": "already_ready",
                    "state_before": "ready",
                    "state_after": "ready",
                    "coalesced": False,
                    "wait_ms": 0,
                },
            },
        },
    }
    schemas.validate(artifact)
    return artifact


def test_child_journal_rebinds_restarted_worker_and_sampling_regime(stores):
    parent = _parent()
    parent = deepcopy(parent)
    parent["meta"]["model_routing"] = _routing_artifact()
    worker = {
        "worker_id": "generation-b",
        "worker_generation_id": "generation-b",
        "worker_generation": 2,
        "protocol_version": "1.1",
    }
    plan = _plan(
        parent,
        worker=worker,
        change={"type": "sampling", "temperature": 0.65, "seed": 19},
    )
    engine = FakeEngine([
        _reply(
            plan, tokens=[22, 33], text=" two three", intervention_type="none",
            generation="generation-b"),
        _reply(
            plan,
            tokens=[77, 88],
            text=" sampled",
            intervention_type="sampling",
            generation="generation-b",
            applied={
                "type": "sampling",
                "temperature": 0.65,
                "top_p": 0.9,
                "top_k": 40,
                "rep_penalty": 1.1,
                "seed": 19,
            },
        ),
    ])

    child = execute_exact_fork(
        parent,
        plan,
        engine,
        runtime_identity=RUNTIME,
        worker_identity=worker,
    )["child"]

    assert child["meta"]["decode"] == {
        "mode": "sample",
        "temperature": 0.65,
        "top_p": 0.9,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "seed": 19,
    }
    assert child["meta"]["sampler_mode"] == "sample"
    assert child["meta"]["execution_fork_worker_generation_id"] == "generation-b"
    routing = child["meta"]["model_routing"]
    assert routing["protocol"] == {
        "surface": "native",
        "route": "/runs/<id>/execution-fork",
    }
    assert routing["request"]["request_id"] == child["execution_fork"]["execution_id"]
    routed_worker = routing["result"]["receipt"]["worker_identity"]
    assert routed_worker["worker_id"] == "generation-b"
    assert routed_worker["worker_generation"] == 2
    schemas.validate(routing)


def test_raw_steer_child_clears_parent_dial_claim_and_records_exact_vector(stores):
    parent = _parent()
    parent = deepcopy(parent)
    parent["behavior"]["active_dials"] = {"warm": 0.7}
    parent["meta"]["execution_fork_steering"] = {
        "source": "recorded_raw_vector",
        "steer_vec": [9.0, 9.0],
        "steer_layer": 3,
        "steer_coef": 0.5,
        "active_dials_sha256": _sha(parent["behavior"]["active_dials"]),
    }
    change = {
        "type": "steer",
        "steer_vec": [0.25, -0.5],
        "steer_layer": 4,
        "steer_coef": 0.75,
    }
    plan = _plan(parent, change=change)
    engine = FakeEngine([
        _reply(plan, tokens=[22, 33], text=" two three", intervention_type="none"),
        _reply(
            plan,
            tokens=[66, 77],
            text=" steered",
            intervention_type="steer",
            applied={
                "type": "steer",
                "steer_layer_lo": 4,
                "steer_layer_hi": 4,
                "steer_coef": 0.75,
            },
        ),
    ])

    child = execute_exact_fork(
        parent,
        plan,
        engine,
        runtime_identity=RUNTIME,
        worker_identity=WORKER,
    )["child"]

    assert child["behavior"]["active_dials"] == {}
    assert child["meta"]["execution_fork_steering"] == {
        "source": "recorded_raw_vector",
        "steer_vec": [0.25, -0.5],
        "steer_layer": 4,
        "steer_coef": 0.75,
        "active_dials_sha256": _sha({}),
        "intervention_sha256": _sha(change),
    }


def test_diverged_control_is_terminal_evidence_and_never_runs_or_stores_intervention(stores):
    parent = _parent()
    plan = _plan(parent)
    engine = FakeEngine([
        _reply(plan, tokens=[999, 33], text=" wrong three", intervention_type="none"),
    ])
    before_ids = {run["id"] for run in runlog.iter_runs()}

    result = execute_exact_fork(
        parent, plan, engine,
        runtime_identity=RUNTIME, worker_identity=WORKER,
    )

    receipt = result["receipt"]
    assert result["child"] is None
    assert receipt["phase"] == "failed"
    assert receipt["execution"]["status"] == "control_diverged"
    assert receipt["unchanged_control"]["status"] == "diverged"
    assert len(engine.calls) == 1
    assert {run["id"] for run in runlog.iter_runs()} == before_ids
    assert execution_fork_results.get(receipt["execution_id"]) == receipt


def test_control_failure_is_persisted_without_fabricating_a_generation_run(stores):
    parent = _parent()
    plan = _plan(parent)
    engine = FakeEngine([RuntimeError("checkpoint evicted")])
    before_ids = {run["id"] for run in runlog.iter_runs()}

    result = execute_exact_fork(
        parent, plan, engine,
        runtime_identity=RUNTIME, worker_identity=WORKER,
    )

    receipt = result["receipt"]
    assert receipt["execution"]["status"] == "control_failed"
    assert receipt["execution"]["error"]["stage"] == "control"
    assert "checkpoint evicted" in receipt["execution"]["error"]["message"]
    assert {run["id"] for run in runlog.iter_runs()} == before_ids
    assert execution_fork_results.get(receipt["execution_id"]) == receipt


def test_stale_runtime_or_parent_precondition_persists_failure_before_worker_call(stores):
    parent = _parent()
    plan = _plan(parent)
    engine = _success_engine(plan)
    changed_runtime = {**RUNTIME, "backend": "cuda"}

    runtime_stale = execute_exact_fork(
        parent, plan, engine,
        runtime_identity=changed_runtime, worker_identity=WORKER,
    )
    assert runtime_stale["receipt"]["reasons"][0]["code"] == "stale_plan"
    assert engine.calls == []

    changed_parent = deepcopy(parent)
    changed_parent["trace"]["token_ids"][1] = 777
    parent_stale = execute_exact_fork(
        parent, plan, engine,
        runtime_identity=RUNTIME, worker_identity=WORKER,
        reload_parent=lambda _run_id: changed_parent,
    )
    assert parent_stale["receipt"]["reasons"][0]["code"] == "stale_plan"
    assert engine.calls == []

    stale_worker = {**WORKER, "worker_id": "generation-b",
                    "worker_generation_id": "generation-b"}
    worker_stale = execute_exact_fork(
        parent, plan, engine,
        runtime_identity=RUNTIME, worker_identity=stale_worker,
    )
    assert worker_stale["receipt"]["reasons"][0]["code"] == "stale_plan"
    assert engine.calls == []


def test_intervention_failure_and_cancellation_after_control_store_no_child(stores):
    parent = _parent()
    plan = _plan(parent)
    failure_engine = FakeEngine([
        _reply(plan, tokens=[22, 33], text=" two three", intervention_type="none"),
        RuntimeError("worker died"),
    ])
    failed = execute_exact_fork(
        parent, plan, failure_engine,
        runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    assert failed["child"] is None
    assert failed["receipt"]["execution"]["status"] == "intervention_failed"
    assert failed["receipt"]["unchanged_control"]["status"] == "matched"

    cancel_engine = FakeEngine([
        _reply(plan, tokens=[22, 33], text=" two three", intervention_type="none"),
    ])
    checks = iter([False, True])
    cancelled = execute_exact_fork(
        parent, plan, cancel_engine,
        runtime_identity=RUNTIME, worker_identity=WORKER,
        cancel_check=lambda: next(checks),
    )
    assert cancelled["child"] is None
    assert cancelled["receipt"]["phase"] == "cancelled"
    assert cancelled["receipt"]["unchanged_control"]["status"] == "matched"
    assert len(cancel_engine.calls) == 1


def test_worker_receipt_generation_or_intervention_drift_fails_closed(stores):
    parent = _parent()
    plan = _plan(parent)
    generation_drift = FakeEngine([
        _reply(
            plan, tokens=[22, 33], text=" two three", intervention_type="none",
            generation="generation-b"),
    ])
    failed_control = execute_exact_fork(
        parent, plan, generation_drift,
        runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    assert failed_control["receipt"]["execution"]["status"] == "control_failed"
    assert failed_control["child"] is None

    wrong_token = FakeEngine([
        _reply(plan, tokens=[22, 33], text=" two three", intervention_type="none"),
        _reply(
            plan, tokens=[45, 55], text=" other five", intervention_type="force_token",
            applied={"type": "force_token", "token_id": 45}),
    ])
    failed_child = execute_exact_fork(
        parent, plan, wrong_token,
        runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    assert failed_child["receipt"]["execution"]["status"] == "intervention_failed"
    assert failed_child["child"] is None


def test_persistence_failure_becomes_terminal_receipt_not_a_success(stores):
    parent = _parent()
    plan = _plan(parent)
    engine = _success_engine(plan)
    attempted = {}

    def fail_store(_parent, _plan, _reply, receipt, **_kwargs):
        attempted["execution_id"] = receipt["execution_id"]
        return None

    result = execute_exact_fork(
        parent, plan, engine,
        runtime_identity=RUNTIME, worker_identity=WORKER,
        record_child=fail_store,
    )
    assert result["child"] is None
    assert result["receipt"]["phase"] == "failed"
    assert result["receipt"]["execution"]["status"] == "persistence_failed"
    assert result["receipt"]["execution_id"] == attempted["execution_id"]
    assert execution_fork_results.get(result["receipt"]["execution_id"]) == result["receipt"]


class Handler:
    def __init__(self, sub):
        self._inj_sub = sub
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


class FakeSub:
    def __init__(self, engine, *, worker=None):
        self.engine = engine
        self.runtime_identity = lambda: deepcopy(RUNTIME)
        self.worker_identity = lambda: deepcopy(worker or WORKER)


def test_new_gateway_route_plans_and_executes_without_touching_legacy_route(stores):
    parent = _parent()
    planning_engine = FakeEngine([])
    sub = FakeSub(planning_engine)
    h = Handler(sub)
    checkpoint = {
        "checkpoint_id": "ckpt-generation-a-7",
        "worker_generation_id": "generation-a",
        "state": "available",
        "parent_run_id": parent["id"],
        "prompt_tokens": 10,
        "n_past": 13,
    }
    assert route.try_post(
        h,
        f"/runs/{parent['id']}/execution-fork/plan",
        {
            "request": {
                "position": 1,
                "change": {
                    "type": "force_token", "token_id": 44, "token_piece": " four"},
            },
            "checkpoint_reference": checkpoint,
        },
    )
    assert h.status == 200
    plan = h.body
    assert plan["classification"] == "exact_execution_fork"
    assert planning_engine.calls == []

    execution_engine = _success_engine(plan)
    h2 = Handler(FakeSub(execution_engine))
    assert route.try_post(
        h2, f"/runs/{parent['id']}/execution-fork", {"plan": plan})
    assert h2.status == 201
    execution_id = h2.body["receipt"]["execution_id"]

    h3 = Handler(FakeSub(execution_engine))
    assert route.try_get(h3, f"/execution-forks/{execution_id}")
    assert h3.status == 200
    assert h3.body == h2.body["receipt"]

    from clozn.server import app as server
    assert route in server._POST_ROUTES
    assert route in server._GET_ROUTES
    from clozn.server.routes import fork as legacy
    assert legacy in server._POST_ROUTES


def test_gateway_rejects_stale_planned_worker_and_persists_terminal_result(stores):
    parent = _parent()
    plan = _plan(parent)
    engine = _success_engine(plan)
    stale_worker = {
        "worker_id": "generation-b",
        "worker_generation_id": "generation-b",
        "protocol_version": "1.1",
    }
    h = Handler(FakeSub(engine, worker=stale_worker))
    assert route.try_post(
        h, f"/runs/{parent['id']}/execution-fork", {"plan": plan})
    assert h.status == 422
    assert h.body["child"] is None
    assert h.body["receipt"]["reasons"][0]["code"] == "stale_plan"
    assert engine.calls == []
    execution_id = h.body["receipt"]["execution_id"]
    assert execution_fork_results.get(execution_id) == h.body["receipt"]


def test_terminal_result_ids_are_immutable(stores):
    parent = _parent()
    plan = _plan(parent)
    engine = FakeEngine([RuntimeError("gone")])
    receipt = execute_exact_fork(
        parent, plan, engine,
        runtime_identity=RUNTIME, worker_identity=WORKER,
    )["receipt"]
    assert execution_fork_results.save(receipt) == receipt
    changed = deepcopy(receipt)
    changed["execution"]["error"]["message"] = "rewritten"
    with pytest.raises(
        execution_fork_results.ExecutionForkResultError,
        match="different immutable receipt",
    ):
        execution_fork_results.save(changed)
