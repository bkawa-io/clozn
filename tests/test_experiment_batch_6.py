from __future__ import annotations

from copy import deepcopy

from clozn.experiments.batch import ArmExecutionOutcome, BatchExecutionResult
from clozn.experiments.evaluators import ExactReferenceMatch
from clozn.experiments.execution import DeleteSourceExactReferenceAdapter
from clozn.experiments.interventions import DeleteSource
from clozn.experiments.kernel import Experiment
from clozn.experiments.observations import Observation, execution_observation_identity
from clozn.experiments.runner import run_experiment
from clozn.experiments.selections import ContextSelection
from clozn.experiments.state import ExecutionState
from clozn.runs.context_receipt import build_context_receipt


def _run():
    messages = [
        {"role": "system", "content": "parent"},
        {"role": "user", "content": "source A"},
        {"role": "user", "content": "source B"},
    ]
    run = {
        "id": "batch-6-run", "model": "fixture", "substrate": "fixture",
        "messages": deepcopy(messages), "assembled_messages": deepcopy(messages),
        "final_prompt": "parent", "response": "answer",
        "generation_contract": {"decode_mode": "greedy", "sampling": None, "max_new": 2,
                                "stop": [], "expected_termination": {"reason": "eos", "reason_raw": "eos"}},
        "identity": {"model_sha256": "a" * 64, "template_fingerprint": "b" * 16,
                     "engine_build": "test", "context_size": 4096, "backend": "cpu",
                     "white_box_flags": {"sae": False, "jlens": False, "attn_knockout": False}},
        "meta": {"n_ctx": 4096, "device": "cpu"},
        "trace": {"tokens": ["an", "swer"], "token_ids": [1, 2],
                  "steps": [{"token_id": 1, "piece": "an"}, {"token_id": 2, "piece": "swer"}]},
    }
    run["context_receipt"] = build_context_receipt(
        messages=messages, assembled_messages=messages, final_prompt="parent", run_id=run["id"], privacy="full",
    )
    return run, [item["segment_id"] for item in run["context_receipt"]["assembled"]]


def _observation(state, evaluator, intervention, status="exact_preserved"):
    identity = execution_observation_identity(state, evaluator, intervention)
    return Observation(
        **identity, run_id=state.run_id, base_execution_fingerprint=state.execution_fingerprint,
        evaluator=identity["observation_key"]["evaluator"],
        condition=identity["observation_key"]["condition"], contract=identity["observation_key"]["contract"],
        status=status, matched_token_count=2 if status != "diverged" else 1,
        execution_provenance={"adapter": "test"}, proof_grade="trusted", trusted=True,
    )


class BatchAdapter:
    def __init__(self, *, statuses=None):
        self.statuses = dict(statuses or {})
        self.scalar_calls = []
        self.batch_requests = []

    def execute(self, state, intervention=None, *, evaluator=None, arm_id=None):
        self.scalar_calls.append(arm_id)
        return _observation(state, evaluator, intervention)

    def execute_many(self, requests, *, cancel=None):
        self.batch_requests.append(tuple(request.arm_id for request in requests))
        outcomes = []
        for request in requests:
            status = self.statuses.get(request.arm_id, "exact_preserved")
            if status in {"failed", "unavailable"}:
                outcomes.append(ArmExecutionOutcome(
                    arm_id=request.arm_id, execution_disposition="executed", state="failed",
                    diagnostics={"reason": status},
                ))
            else:
                observation = _observation(request.state, request.evaluator, request.intervention, status)
                outcomes.append(ArmExecutionOutcome(
                    arm_id=request.arm_id, observation=observation,
                    execution_disposition="executed", state="completed",
                ))
        return BatchExecutionResult(tuple(outcomes), {"execution_strategy": "test_batch", "batch_count": 1})


def _experiment(run, source_ids):
    state = ExecutionState.from_run(run)
    arm_a = DeleteSource(ContextSelection([source_ids[1]]))
    arm_b = DeleteSource(ContextSelection([source_ids[1]]))
    return Experiment(base=state, evaluator=ExactReferenceMatch(), arms=[arm_a, arm_b])


def test_duplicate_conditions_dispatch_once_and_alias_both_arms():
    run, source_ids = _run()
    adapter = BatchAdapter()
    result = run_experiment(_experiment(run, source_ids), adapter)
    assert adapter.batch_requests == [(result.arms[0].arm_id,)]
    assert result.arm_dispositions == ("executed", "reused")
    assert result.arms[0].observation_id == result.arms[1].observation_id


def test_batch_partial_outcomes_preserve_execution_disposition():
    run, source_ids = _run()
    experiment = Experiment(
        base=ExecutionState.from_run(run), evaluator=ExactReferenceMatch(),
        arms=[DeleteSource(ContextSelection([source_ids[0]])),
              DeleteSource(ContextSelection([source_ids[1]]))],
    )
    adapter = BatchAdapter(statuses={experiment.arms[1].arm_id: "failed"})
    result = run_experiment(experiment, adapter)
    assert result.arm_dispositions == ("executed", "executed")
    assert result.arms[0].status == "exact_preserved"
    assert result.arms[1].state == "failed"


def test_scalar_fallback_keeps_one_outcome_per_missing_condition():
    run, source_ids = _run()
    adapter = BatchAdapter()
    adapter.execute_many = None
    experiment = Experiment(
        base=ExecutionState.from_run(run), evaluator=ExactReferenceMatch(),
        arms=[DeleteSource(ContextSelection([source_ids[0]]))],
    )
    result = run_experiment(experiment, adapter)
    assert adapter.scalar_calls == [None, experiment.arms[0].arm_id]
    assert result.arms[0].status == "exact_preserved"


def test_scalar_and_proof_grade_native_many_have_identical_observation_artifacts():
    run, source_ids = _run()

    class Substrate:
        def identity_meta(self):
            return deepcopy(run["identity"])

        def run_meta(self):
            return deepcopy(run["meta"])

        def probe_reference_match(self, messages, reference_token_ids, *, generation_contract,
                                  explicit_conditions=None):
            return {"status": "matched", "matched_token_count": 2,
                    "first_divergence_index": None, "divergence_kind": None}

        probe_reference_match_many_proof_grade = True

        def probe_reference_match_many(self, arms, *, cancel=None):
            return [
                {"arm_index": index, "result": self.probe_reference_match(**arm)}
                for index, arm in reversed(list(enumerate(arms)))
            ]

    experiment = Experiment(
        base=ExecutionState.from_run(run), evaluator=ExactReferenceMatch(),
        arms=[DeleteSource(ContextSelection([source_ids[0]])),
              DeleteSource(ContextSelection([source_ids[1]]))],
    )
    scalar = run_experiment(
        experiment, DeleteSourceExactReferenceAdapter(Substrate(), run=run, execution_strategy="scalar"),
    )
    native = run_experiment(
        experiment, DeleteSourceExactReferenceAdapter(Substrate(), run=run, execution_strategy="native_many"),
    )
    assert [row.observation.to_json() for row in scalar.arms] == [row.observation.to_json() for row in native.arms]
    assert native.diagnostics["execution_strategy"] == "native_many"
