"""Batch 4 StateRef/Generate acceptance coverage."""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn.experiments.evaluators import Generate
from clozn.experiments.generation import GenerateExecutionAdapter
from clozn.experiments.interventions import ForceToken
from clozn.experiments.kernel import Experiment
from clozn.experiments.materialize import materialize_generated_observation
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.runner import run_experiment
from clozn.experiments.state import ExecutionState
from clozn.experiments.state_ref import StateRef, StateRefError, resolve_state
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
