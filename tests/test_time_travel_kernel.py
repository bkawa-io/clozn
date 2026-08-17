"""Batch 4 StateRef/Generate acceptance coverage."""
from __future__ import annotations

from copy import deepcopy
import io
import json

import pytest

from clozn.experiments.evaluators import Generate
from clozn.experiments.generation import GenerateExecutionAdapter
from clozn.experiments.interventions import ForceToken
from clozn.experiments.kernel import Experiment
from clozn.experiments.materialize import materialize_generated_observation
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.runner import run_experiment
from clozn.experiments.state import ExecutionState
from clozn.experiments.state_ref import (
    STOCHASTIC_EXECUTION_UNBOUND, StateRef, StateRefError, operation_readiness, resolve_state,
)
from clozn.recipes.time_travel import (
    enumerate_answer_boundaries, materialize_time_travel, resolve_time_travel,
    run_time_travel, time_travel_capabilities,
)
from clozn.runs import store as run_store
from clozn.replay.execution_fork import plan_execution_fork


RUNTIME = {
    "model_sha256": "a" * 64,
    "template_fingerprint": "b" * 16,
    "engine_build": "batch4-fixture",
    "context_size": 4096,
    "backend": "cpu",
    "adapter": {"present": False, "identity_sha256": None, "artifact_sha256": None, "scale": None},
    "white_box_flags": {},
}
WORKER = {"worker_id": "worker-a", "worker_generation_id": "generation-a", "protocol_version": "1.1"}


def _run():
    return {
        "id": "run_batch4_fixture",
        "model": "fixture-model",
        "substrate": "fixture",
        "messages": [{"role": "user", "content": "question"}],
        "assembled_messages": [{"role": "user", "content": "question"}],
        "final_prompt": "<prompt>",
        "response": "zero one two",
        "identity": deepcopy(RUNTIME),
        "meta": {"n_ctx": 4096, "device": "cpu"},
        "trace": {
            "tokens": ["zero", " one", " two"],
            "token_ids": [10, 11, 12],
            "steps": [
                {"token_id": 10, "piece": "zero"},
                {"token_id": 11, "piece": " one"},
                {"token_id": 12, "piece": " two"},
            ],
        },
    }


def _sampled_run():
    run = _run()
    run["generation_contract"] = {
        "decode_mode": "sample",
        "sampling": {
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.0,
            "seed": 7,
        },
        "max_new": 3,
        "stop": [],
        "expected_termination": {"reason": "stop", "reason_raw": "stop"},
    }
    return run


def _exact_checkpoint(run):
    return {
        "checkpoint_id": "checkpoint-fixture",
        "worker_generation_id": WORKER["worker_generation_id"],
        "state": "available",
        "parent_run_id": run["id"],
        "prompt_tokens": 8,
        "n_past": 10,
    }


class ReconstructedEngine:
    def __init__(self):
        self.calls = []

    def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"choices": [{"text": " tail", "finish_reason": "stop"}]}


class ReconstructedSubstrate:
    def __init__(self):
        self.engine = ReconstructedEngine()


class ExactEngine:
    def __init__(self, *, mismatch=False):
        self.calls = []
        self.mismatch = mismatch

    def execution_fork(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["intervention"]["type"] == "none":
            tokens, pieces, text = ([99], ["wrong"], "wrong") if self.mismatch else ([11, 12], [" one", " two"], " one two")
            applied = {"type": "none"}
        else:
            tokens, pieces, text, applied = [77, 78], ["ALT", " tail"], "ALT tail", {"type": "force_token", "token_id": 77}
        return {
            "worker_generation_id": WORKER["worker_generation_id"],
            "text": text,
            "tokens": tokens,
            "token_pieces": pieces,
            "steps": [{"token_id": token_id, "piece": piece} for token_id, piece in zip(tokens, pieces)],
            "restore_mode": "live_kv_truncated",
            "n_past_restored": 9,
            "exactness": {"source": "live_kv", "boundary_shape_true": True},
            "intervention_applied": applied,
            "finish_reason": "stop",
            "sampler_state_preserved": True,
        }


class ExactSubstrate:
    def __init__(self, *, mismatch=False):
        self.engine = ExactEngine(mismatch=mismatch)
        self.runtime_identity = deepcopy(RUNTIME)
        self.worker_identity = deepcopy(WORKER)


def test_state_ref_has_one_canonical_boundary_coordinate_and_stable_identity():
    run = _run()
    prompt = StateRef.prompt_boundary(run)
    before_zero = StateRef.before_answer_token(run, 0)
    before_one = StateRef.before_answer_token(run, 1)
    assert prompt == before_zero
    assert prompt.position.to_dict() == {"kind": "answer_token_boundary", "index": 0}
    assert prompt.state_fingerprint != before_one.state_fingerprint
    assert prompt.to_json() == StateRef.from_dict(prompt.to_dict()).to_json()
    with pytest.raises(StateRefError):
        StateRef.before_answer_token(run, 3)
    with pytest.raises(StateRefError):
        StateRef.before_answer_token({**run, "trace": {"tokens": ["bad"], "token_ids": [1, 2]}}, 0)


def test_time_travel_boundaries_are_model_free_and_trace_addressed():
    run = _run()
    boundaries = enumerate_answer_boundaries(run)
    assert [item.index for item in boundaries] == [0, 1, 2]
    assert boundaries[1].recorded_token_id == 11
    assert boundaries[1].recorded_token_piece == " one"
    assert boundaries[2].response_offset == len("zero one")
    assert boundaries[0].state_fingerprint == StateRef.before_answer_token(run, 0).state_fingerprint
    with pytest.raises(Exception, match="token history"):
        enumerate_answer_boundaries({**run, "trace": {"tokens": ["bad"], "token_ids": [1, 2]}})


def test_state_ref_rejects_stale_execution_and_exact_planning_is_not_confirmation():
    run = _run()
    ref = StateRef.before_answer_token(run, 1)
    with pytest.raises(StateRefError):
        ref.assert_current({**run, "response": "changed"})
    resolved = resolve_state(
        ref, run=run, policy="exact_required", checkpoint=_exact_checkpoint(run),
        runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    assert resolved.classification == "exact_execution_fork"
    assert resolved.proof_status == "planned"
    oracle = plan_execution_fork(
        run, {"position": 1, "change": {"type": "none"}}, checkpoint=_exact_checkpoint(run),
        runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    assert resolved.plan["classification"] == oracle["classification"]
    assert resolved.plan["exactness"] == oracle["exactness"]


def test_capabilities_are_model_free_pin_aware_and_do_not_claim_exact_proof(monkeypatch):
    run = _run()
    seen = []

    def resolve_pin(run_id):
        seen.append(run_id)
        reference = _exact_checkpoint(run)
        return {
            "ok": True,
            "manifest": {
                "pin_id": "pin_fixture", "run_id": run_id,
                "source": {
                    "checkpoint_id": reference["checkpoint_id"],
                    "worker_generation_id": reference["worker_generation_id"],
                },
            },
            "envelope": {"state": {"prompt_tokens": reference["prompt_tokens"], "n_past": reference["n_past"]}},
        }

    monkeypatch.setattr("clozn.replay.checkpoint_pin_store.resolve_pin", resolve_pin)
    capabilities = time_travel_capabilities(run)

    assert seen == [run["id"]]
    assert capabilities["answer_token_boundaries"]["available"] is True
    assert capabilities["exact_checkpoint_restore"]["state"] == "planned"
    assert capabilities["exact_checkpoint_restore"]["proof_status"] == "planned"
    assert capabilities["sampler_restore"]["available"] is False
    assert capabilities["sampler_restore"]["state"] == "not_required"
    assert capabilities["available_operations"]["continue"]["reason_code"] == "exact_control_required"
    assert capabilities["available_operations"]["force_token"]["reason_code"] == "force_token_id_required"


def test_operation_readiness_distinguishes_reconstructed_continue_and_force_input():
    run = _run()
    resolved = resolve_time_travel(
        run, position=1, policy="reconstructed_only", token_id=77,
    )
    operations = resolved.to_dict()["available_operations"]
    assert operations["continue"]["available"] is True
    assert operations["force_token"]["available"] is False
    assert operations["force_token"]["reason_code"] == "reconstruction_token_piece_unavailable"


def test_sampled_continue_and_force_token_share_fail_closed_readiness_without_worker_calls():
    run = _sampled_run()
    resolved_reconstructed = resolve_state(
        StateRef.before_answer_token(run, 1), run=run, policy="reconstructed_only",
        runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    resolved_exact = resolve_state(
        StateRef.before_answer_token(run, 1), run=run, policy="exact_required",
        checkpoint=_exact_checkpoint(run), runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    evaluator = Generate(max_new=2, decode_mode="sample", sampling={
        "temperature": 0.8, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.0, "seed": 7,
    })
    force = ForceToken(token_id=77, token_piece="ALT")

    reconstructed_substrate = ReconstructedSubstrate()
    reconstructed_adapter = GenerateExecutionAdapter(reconstructed_substrate, run=run)
    reconstructed_control = reconstructed_adapter.execute_control(resolved_reconstructed, evaluator=evaluator)
    reconstructed_force = reconstructed_adapter.execute(resolved_reconstructed, force, evaluator=evaluator)
    assert reconstructed_control.diagnostics["reason_code"] == STOCHASTIC_EXECUTION_UNBOUND
    assert reconstructed_force.diagnostics["reason_code"] == STOCHASTIC_EXECUTION_UNBOUND
    assert reconstructed_substrate.engine.calls == []

    exact_substrate = ExactSubstrate()
    exact_adapter = GenerateExecutionAdapter(exact_substrate, run=run)
    exact_control = exact_adapter.execute_control(resolved_exact, evaluator=evaluator)
    exact_force = exact_adapter.execute(resolved_exact, force, evaluator=evaluator)
    assert exact_control.diagnostics["reason_code"] == STOCHASTIC_EXECUTION_UNBOUND
    assert exact_force.diagnostics["reason_code"] == STOCHASTIC_EXECUTION_UNBOUND
    assert exact_substrate.engine.calls == []


def test_sampled_capabilities_and_recipe_reject_both_operations_without_model_calls():
    run = _sampled_run()
    ref = StateRef.before_answer_token(run, 1)
    reconstructed = resolve_state(ref, run=run, policy="reconstructed_only")
    readiness = operation_readiness(reconstructed, operation="continue", decode_mode="sample")
    assert readiness["plannable"] is False
    assert readiness["reason_code"] == STOCHASTIC_EXECUTION_UNBOUND

    capabilities = time_travel_capabilities(
        run, checkpoint=_exact_checkpoint(run), runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    for operation in ("continue", "force_token"):
        assert capabilities["available_operations"][operation]["plannable"] is False
        assert capabilities["available_operations"][operation]["reason_code"] == STOCHASTIC_EXECUTION_UNBOUND
    assert capabilities["sampler_restore"]["state"] == "unavailable"
    assert capabilities["sampler_restore"]["reason_code"] == STOCHASTIC_EXECUTION_UNBOUND


def test_incomplete_sampled_metadata_does_not_fall_back_to_greedy():
    run = _run()
    run["meta"]["decode"] = {"mode": "sample"}
    resolved = resolve_state(
        StateRef.before_answer_token(run, 1), run=run, policy="reconstructed_only",
        runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    readiness = operation_readiness(resolved, operation="continue")
    assert readiness["plannable"] is False
    assert readiness["reason_code"] == STOCHASTIC_EXECUTION_UNBOUND


def test_run_time_travel_rejects_sampled_parent_before_experiment_or_model_call():
    run = _sampled_run()
    substrate = ExactSubstrate()
    result = run_time_travel(
        run, position=1, max_new=2, policy="exact_required", checkpoint=_exact_checkpoint(run),
        runtime_identity=RUNTIME, worker_identity=WORKER, substrate=substrate,
    )
    assert result.status == "unavailable"
    assert result.diagnostics["operation_readiness"]["reason_code"] == STOCHASTIC_EXECUTION_UNBOUND
    assert substrate.engine.calls == []


def test_reopened_generated_evidence_materializes_without_touching_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run = _run()
    substrate = ReconstructedSubstrate()
    first_store = ObservationStore()
    result = run_time_travel(
        run, position=1, token_piece="ALT", max_new=2, policy="reconstructed_only", substrate=substrate,
        observation_store=first_store, runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    assert result.status == "completed"

    reopened = ObservationStore()
    calls = len(substrate.engine.calls)
    materialized = materialize_time_travel(
        run, result.experiment_id, arm_id=result.arm_id, observation_id=result.observation_id,
        observation_store=reopened,
    )
    assert materialized["state"] == "completed"
    assert len(substrate.engine.calls) == calls
    child = run_store.get_run(materialized["child_run_id"])
    lineage = child["changes_applied"]["experiment"]
    assert lineage["base_state"]["execution_fingerprint"]
    assert lineage["base_state"]["state_ref"]["position"]["index"] == 1
    assert lineage["base_state"]["resolved_classification"] == "reconstructed_replay"
    assert lineage["base_state"]["resolved_proof_status"] == "planned"
    assert lineage["operation"] == "force_token"
    assert lineage["intervention"] == {"kind": "force_token", "token_piece": "ALT"}


def test_force_token_and_generate_identity_include_state_token_and_budget():
    run = _run()
    first = resolve_state(StateRef.before_answer_token(run, 1), run=run, policy="reconstructed_only",
                          runtime_identity=RUNTIME, worker_identity=WORKER)
    second = resolve_state(StateRef.before_answer_token(run, 2), run=run, policy="reconstructed_only",
                           runtime_identity=RUNTIME, worker_identity=WORKER)
    assert Experiment(base=first, evaluator=Generate(max_new=2), arms=[ForceToken(token_piece="X")]).experiment_id != Experiment(
        base=first, evaluator=Generate(max_new=3), arms=[ForceToken(token_piece="X")]).experiment_id
    assert Experiment(base=first, evaluator=Generate(max_new=2), arms=[ForceToken(token_piece="X")]).experiment_id != Experiment(
        base=second, evaluator=Generate(max_new=2), arms=[ForceToken(token_piece="X")]).experiment_id
    assert ForceToken(token_id=3, token_piece="X").to_json() == ForceToken.from_dict(
        {"kind": "force_token", "token_id": 3, "token_piece": "X"}).to_json()


def test_reconstructed_generate_creates_no_runs_and_materialization_does_not_call_model(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run = _run()
    resolved = resolve_state(StateRef.before_answer_token(run, 1), run=run, policy="reconstructed_only",
                             runtime_identity=RUNTIME, worker_identity=WORKER)
    experiment = Experiment(base=resolved, evaluator=Generate(max_new=2), arms=[ForceToken(token_piece="ALT")])
    substrate = ReconstructedSubstrate()
    store = ObservationStore()
    result = run_experiment(experiment, GenerateExecutionAdapter(substrate, run=run), observation_store=store)
    assert result.state == "completed"
    assert result.arms[0].observation.fidelity_classification == "reconstructed_replay"
    assert substrate.engine.calls[-1][1]["max_tokens"] == 1  # the forced token consumes one max_new slot
    assert run_store.list_runs(20) == []
    calls = len(substrate.engine.calls)
    materialized = materialize_generated_observation(
        run, experiment.experiment_id, result.arms[0].arm_id, observation_store=store,
    )
    assert materialized["state"] == "completed"
    assert len(substrate.engine.calls) == calls
    child = run_store.get_run(materialized["child_run_id"])
    assert child["parent_run_id"] == run["id"]
    assert child["response"] == "zeroALT tail"
    assert child["changes_applied"]["experiment"]["base_state"]["position"]["index"] == 1
    assert child["changes_applied"]["experiment"]["base_state"]["realized_fidelity"] == "reconstructed_replay"
    second = materialize_generated_observation(
        run, experiment.experiment_id, result.arms[0].arm_id, observation_store=store,
    )
    assert second["child_run_id"] != materialized["child_run_id"]
    assert len(run_store.list_runs(20)) == 2


def test_reconstructed_continue_is_an_unchanged_generate_condition_and_reloads(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run = _run()
    substrate = ReconstructedSubstrate()
    store = ObservationStore()
    result = run_time_travel(
        run, position=1, max_new=2, policy="reconstructed_only", substrate=substrate,
        observation_store=store, runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    assert result.status == "completed"
    assert result.operation == {"kind": "continue"}
    assert result.fidelity == "RECONSTRUCTED"
    assert result.experiment_id and result.arm_id and result.observation_id
    assert len(substrate.engine.calls) == 1
    assert run_store.list_runs(20) == []

    reloaded = store.get_experiment(result.experiment_id)
    arm = reloaded.arm_for(result.arm_id)
    assert arm.intervention is None
    assert arm.observation_id == result.observation_id
    assert arm.observation.fidelity_classification == "reconstructed_replay"

    materialized = materialize_time_travel(run, result, observation_store=store)
    assert materialized["state"] == "completed"
    assert len(substrate.engine.calls) == 1
    child = run_store.get_run(materialized["child_run_id"])
    assert child["parent_run_id"] == run["id"]
    assert child["response"] == "zero tail"
    assert child["changes_applied"]["experiment"]["operation"] == "continue"


def test_exact_generate_confirms_control_creates_no_run_and_materializes_without_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run = _run()
    resolved = resolve_state(
        StateRef.before_answer_token(run, 1), run=run, policy="exact_required",
        checkpoint=_exact_checkpoint(run), runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    experiment = Experiment(
        base=resolved, evaluator=Generate(max_new=2),
        arms=[ForceToken(token_id=77, token_piece="ALT")],
    )
    substrate = ExactSubstrate(mismatch=False)
    store = ObservationStore()
    result = run_experiment(experiment, GenerateExecutionAdapter(substrate, run=run), observation_store=store)
    assert result.state == "completed"
    assert result.control.fidelity["proof_status"] == "confirmed"
    assert result.arms[0].observation.fidelity["exact_match"] is True
    assert result.arms[0].observation.fidelity_classification == "exact_execution_fork"
    assert substrate.engine.calls[-1]["max_tokens"] == 2  # exact worker budget includes the forced slot
    assert len(run_store.list_runs(20)) == 0
    calls = len(substrate.engine.calls)
    materialized = materialize_generated_observation(
        run, experiment.experiment_id, result.arms[0].arm_id, observation_store=store,
    )
    assert materialized["state"] == "completed"
    assert len(substrate.engine.calls) == calls
    child = run_store.get_run(materialized["child_run_id"])
    assert child["response"] == "zeroALT tail"
    assert child["changes_applied"]["experiment"]["base_state"]["realized_fidelity"] == "exact_execution_fork"


def test_exact_control_mismatch_blocks_force_arm_and_persists_no_transient_observation(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run = _run()
    resolved = resolve_state(
        StateRef.before_answer_token(run, 1), run=run, policy="exact_required",
        checkpoint=_exact_checkpoint(run), runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    experiment = Experiment(base=resolved, evaluator=Generate(max_new=2), arms=[ForceToken(token_id=77, token_piece="ALT")])
    substrate = ExactSubstrate(mismatch=True)
    result = run_experiment(experiment, GenerateExecutionAdapter(substrate, run=run), observation_store=ObservationStore())
    assert result.state == "blocked"
    assert result.arms[0].observation is None
    assert len(substrate.engine.calls) == 1
    assert run_store.list_runs(20) == []


def test_time_travel_v1_routes_continue_and_materialize_without_regeneration(tmp_path, monkeypatch):
    from clozn.server import app as server_app

    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()

    class RouteSubstrate:
        def __init__(self):
            self.engine = ReconstructedEngine()
            self.runtime_identity = deepcopy(RUNTIME)
            self.worker_identity = deepcopy(WORKER)

    substrate = RouteSubstrate()
    monkeypatch.setattr(server_app, "SUB", substrate)
    run_id = run_store.record(
        source="test", client="test", model="fixture-model", substrate="fixture",
        messages=[{"role": "user", "content": "question"}],
        assembled_messages=[{"role": "user", "content": "question"}],
        final_prompt="<prompt>", response="zero one two", identity=deepcopy(RUNTIME),
        trace={
            "tokens": ["zero", " one", " two"], "token_ids": [10, 11, 12],
            "steps": [
                {"token_id": 10, "piece": "zero"},
                {"token_id": 11, "piece": " one"},
                {"token_id": 12, "piece": " two"},
            ],
        },
    )

    def dispatch(method, path, payload=None):
        raw = json.dumps(payload or {}).encode("utf-8")
        handler_type = server_app.make_handler()
        handler = object.__new__(handler_type)
        handler.path = path
        handler.rfile = io.BytesIO(raw)
        handler.wfile = io.BytesIO()
        handler.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
        handler.requestline = f"{method} {path} HTTP/1.1"
        handler.request_version = "HTTP/1.1"
        handler.command = method
        getattr(handler, f"do_{method}")()
        return json.loads(handler.wfile.getvalue().partition(b"\r\n\r\n")[2])

    result = dispatch(
        "POST", f"/runs/{run_id}/time-travel/continue",
        {"boundary": 1, "policy": "reconstructed_only", "max_new": 2},
    )
    assert result["status"] == "completed"
    assert result["operation_kind"] == "continue"
    assert result["fidelity"] == "RECONSTRUCTED"
    calls = len(substrate.engine.calls)
    materialized = dispatch(
        "POST", f"/runs/{run_id}/time-travel/materialize",
        {
            "experiment_id": result["experiment_id"], "arm_id": result["arm_id"],
            "observation_id": result["observation_id"],
        },
    )
    assert materialized["state"] == "completed"
    assert len(substrate.engine.calls) == calls
    assert run_store.get_run(materialized["child_run_id"])["parent_run_id"] == run_id


def test_unbound_stochastic_reconstruction_is_unavailable():
    run = _run()
    resolved = resolve_state(StateRef.before_answer_token(run, 1), run=run, policy="reconstructed_only",
                             runtime_identity=RUNTIME, worker_identity=WORKER)
    experiment = Experiment(
        base=resolved,
        evaluator=Generate(max_new=2, decode_mode="sample", sampling={
            "temperature": 0.8, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.0, "seed": 7,
        }),
        arms=[ForceToken(token_piece="ALT")],
    )
    result = run_experiment(experiment, GenerateExecutionAdapter(ReconstructedSubstrate(), run=run))
    assert result.state == "blocked"
    assert result.control.status == "unavailable"
    assert result.control.diagnostics["reason_code"] == "stochastic_execution_unbound"
