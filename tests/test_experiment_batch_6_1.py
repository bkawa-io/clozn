from __future__ import annotations

from copy import deepcopy

import pytest

from clozn.experiments.evaluators import ExactReferenceMatch, ScoreRecordedContinuation
from clozn.experiments.execution import DeleteSourceExactReferenceAdapter
from clozn.experiments.interventions import DeleteSource
from clozn.experiments.kernel import Experiment
from clozn.experiments.multi_arm import BatchCancelled, MultiArmError
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.runner import run_experiment
from clozn.experiments.scoring import DeleteSourceRecordedContinuationScoreAdapter
from clozn.experiments.selections import ContextSelection
from clozn.experiments.state import ExecutionState
from clozn.recipes.minimal_context import run_minimal_context
from clozn.runs.context_units import build_context_unit_manifest
from clozn.runs import store as run_store
from clozn.runs.context_receipt import build_context_receipt

from tests.test_experiment_kernel import ProbeSubstrate, _run
from tests.test_context_effects_kernel import ScoreSubstrate, _run as _score_run


def _experiment(run, source_ids):
    return Experiment(
        base=ExecutionState.from_run(run), evaluator=ExactReferenceMatch(),
        arms=[
            DeleteSource(ContextSelection([source_ids[0]])),
            DeleteSource(ContextSelection([source_ids[1]])),
            DeleteSource(ContextSelection([source_ids[2]])),
            DeleteSource(ContextSelection([source_ids[0], source_ids[1]])),
        ],
    )


def _four_source_run():
    run, _source_ids = _run()
    messages = [
        *deepcopy(run["messages"][:2]),
        {"role": "user", "content": "another removable source"},
        *deepcopy(run["messages"][2:]),
    ]
    run["messages"] = deepcopy(messages)
    run["assembled_messages"] = deepcopy(messages)
    run["context_receipt"] = build_context_receipt(
        messages=messages, assembled_messages=messages,
        final_prompt=run["final_prompt"], run_id=run["id"], privacy="full",
    )
    return run, [item["segment_id"] for item in run["context_receipt"]["assembled"]]


class _PartialNative(ProbeSubstrate):
    probe_reference_match_many_proof_grade = True

    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.native_calls = 0

    def probe_reference_match_many(self, arms, *, cancel=None):
        self.native_calls += 1
        rows = [
            {
                "status": "matched", "matched_token_count": 2,
                "first_divergence_index": None, "divergence_kind": None,
                "termination_match": True,
            }
            for _arm in arms
        ]
        if self.mode == "partial":
            raise MultiArmError(
                "worker failed on arm 2", arm_index=2, completed=rows[:2],
                completed_indices=(0, 1), dispatched_indices=(0, 1, 2),
            )
        if self.mode == "cancelled":
            raise BatchCancelled(
                completed=rows[:1], next_index=2, completed_indices=(0,),
                dispatched_indices=(0, 1),
            )
        raise MultiArmError(
            "connection lost after submission", completed=(),
            completed_indices=(), dispatched_indices=(0, 1, 2, 3),
        )


@pytest.mark.parametrize("mode", ["partial", "ambiguous"])
def test_partial_native_failure_preserves_completed_evidence_and_never_retries(
    mode, tmp_path, monkeypatch,
):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, source_ids = _four_source_run()
    substrate = _PartialNative(mode)
    result = run_experiment(
        _experiment(run, source_ids),
        DeleteSourceExactReferenceAdapter(substrate, run=run, execution_strategy="native_many"),
        observation_store=ObservationStore(),
    )

    assert substrate.native_calls == 1
    # The only scalar probe is the unchanged control.  No failed native batch
    # is silently retried arm-by-arm.
    assert len(substrate.calls) == 1
    assert result.state == "failed"
    if mode == "partial":
        assert [row.state for row in result.arms] == ["completed", "completed", "failed", "not_executed"]
        assert result.arm_dispositions == (
            "executed", "executed", "executed", "not_executed",
        )
        assert result.arms[0].observation_id is not None
        assert result.arms[1].observation_id is not None
        assert result.arms[2].observation_id is None
        assert result.arms[3].observation_id is None
    else:
        assert [row.state for row in result.arms] == ["failed"] * 4
        assert result.arm_dispositions == ("executed",) * 4


def test_mid_batch_cancellation_preserves_completed_and_started_statuses(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, source_ids = _four_source_run()
    substrate = _PartialNative("cancelled")
    result = run_experiment(
        _experiment(run, source_ids),
        DeleteSourceExactReferenceAdapter(substrate, run=run, execution_strategy="native_many"),
        observation_store=ObservationStore(),
    )

    assert result.state == "cancelled"
    assert result.arms[0].state == "completed"
    assert result.arms[0].observation_id is not None
    assert result.arms[1].state == "failed"
    assert result.arm_dispositions[1] == "executed"
    assert result.arms[2].state == "cancelled"
    assert result.arms[3].state == "cancelled"
    assert result.arms[2].observation_id is None
    assert result.arms[3].observation_id is None
    assert substrate.native_calls == 1


def test_score_adapter_translates_partial_batch_without_losing_completed_vector():
    run, source_ids = _score_run()

    class PartialScore(ScoreSubstrate):
        score_tokens_many_proof_grade = True

        def __init__(self):
            super().__init__()
            self.native_calls = 0

        def score_tokens_many(self, arms, *, cancel=None):
            self.native_calls += 1
            rows = [
                [
                    {"id": 10, "piece": "same", "logprob": -1.0},
                    {"id": 11, "piece": " answer", "logprob": -1.0},
                ]
                for _arm in arms
            ]
            raise MultiArmError(
                "score worker failed", arm_index=1, completed=rows[:1],
                completed_indices=(0,), dispatched_indices=(0, 1),
            )

    experiment = Experiment(
        base=ExecutionState.from_run(run), evaluator=ScoreRecordedContinuation(),
        arms=[
            DeleteSource(ContextSelection([source_ids[0]])),
            DeleteSource(ContextSelection([source_ids[1]])),
        ],
    )
    substrate = PartialScore()
    result = run_experiment(
        experiment, DeleteSourceRecordedContinuationScoreAdapter(substrate, run=run),
    )
    assert substrate.native_calls == 1
    assert len(substrate.calls) == 1  # control only
    assert result.state == "failed"
    assert result.arms[0].status == "completed"
    assert result.arms[0].observation_id is not None
    assert result.arms[1].state == "failed"
    assert result.arm_dispositions == ("executed", "executed")


class _SharedEngine:
    def __init__(self, *, fail_promotion: bool = False, fail_probe: bool = False,
                 partial_probe: bool = False, cancel_probe: bool = False):
        self.fail_promotion = fail_promotion
        self.fail_probe = fail_probe
        self.partial_probe = partial_probe
        self.cancel_probe = cancel_probe
        self.create_calls = []
        self.probe_rounds = []
        self.promotions = []
        self.closed = []
        self.parent_version = 0
        self.shared_work = 0
        self.naive_work = 0

    def apply_template(self, messages):
        return "\n".join(str(item.get("content", "")) for item in messages)

    def reference_match_persistent_create(self, prompt, *, reference_token_ids, generation_contract):
        self.create_calls.append((prompt, tuple(reference_token_ids), deepcopy(dict(generation_contract))))
        return {
            "session_id": "shared-session", "parent_version": 0,
            "parent_prompt_digest": "parent-digest",
            "runtime_identity": {"worker_generation_id": "worker-1"},
            "telemetry": {"parent_tokens_evaluated": 10, "prefix_tokens_reused": 0},
        }

    def reference_match_persistent_probe(self, session_id, *, expected_parent_version, children):
        if self.fail_probe:
            raise RuntimeError("shared worker disconnected")
        if self.partial_probe:
            raise MultiArmError(
                "shared worker failed after first child",
                completed=[{
                    "status": "matched", "matched_token_count": 2,
                    "first_divergence_index": None, "divergence_kind": None,
                }], completed_indices=(0,), dispatched_indices=(0, 1), arm_index=1,
            )
        if self.cancel_probe:
            raise BatchCancelled(
                "shared worker cancelled", completed=[{
                    "status": "matched", "matched_token_count": 2,
                    "first_divergence_index": None, "divergence_kind": None,
                }], completed_indices=(0,), dispatched_indices=(0,), next_index=1,
            )
        self.probe_rounds.append((expected_parent_version, deepcopy(list(children))))
        self.shared_work += 2 + (len(children) * 2)
        self.naive_work += sum(len(child["prompt"]) for child in children)
        return {
            "results": [
                {
                    "candidate_id": child["candidate_id"],
                    "result": {
                        "status": "matched" if "removable source" in child["prompt"] else "diverged",
                        "matched_token_count": 2 if "removable source" in child["prompt"] else 1,
                        "first_divergence_index": None if "removable source" in child["prompt"] else 1,
                        "divergence_kind": None if "removable source" in child["prompt"] else "token_mismatch",
                        "termination_match": True,
                        "generated_token_ids": [10, 11],
                    },
                }
                for child in children
            ],
            "telemetry": {
                "parent_tokens_evaluated": 2,
                "prefix_tokens_reused": 8,
                "child_tokens_evaluated": len(children) * 2,
            },
        }

    def reference_match_persistent_promote(self, session_id, *, expected_parent_version, candidate_id):
        if self.fail_promotion:
            raise RuntimeError("promotion rejected")
        assert expected_parent_version == self.parent_version
        self.promotions.append(candidate_id)
        self.parent_version += 1
        return {
            "parent_version": self.parent_version,
            "parent_prompt_digest": f"parent-{self.parent_version}",
            "telemetry": {"promoted_child_count": 1},
        }

    def reference_match_persistent_close(self, session_id):
        self.closed.append(session_id)
        return {"closed": True}


class _SharedSubstrate(ProbeSubstrate):
    shared_parent_exact_proof_grade = True


def _shared_experiment(run, source_ids):
    return Experiment(
        base=ExecutionState.from_run(run), evaluator=ExactReferenceMatch(),
        arms=[
            DeleteSource(ContextSelection([source_ids[0]])),
            DeleteSource(ContextSelection([source_ids[1]])),
        ],
    )


def test_shared_parent_adapter_creates_probes_promotes_and_reuses_parent_without_hidden_probe():
    run, source_ids = _run()
    run["behavior"]["active_dials"] = {}
    engine = _SharedEngine()
    adapter = DeleteSourceExactReferenceAdapter(
        _SharedSubstrate(), run=run, engine=engine, execution_strategy="shared_parent",
    )
    first = run_experiment(_shared_experiment(run, source_ids), adapter)
    assert first.diagnostics["execution_strategy"] == "shared_parent"
    assert len(engine.create_calls) == 1
    assert len(engine.probe_rounds) == 1
    assert first.arms[0].status == "exact_preserved"

    adapter.on_candidate_accepted(evidence={
        "disposition": "executed", "observation_status": "exact_preserved",
        "observation_id": first.arms[0].observation_id,
    })
    assert len(engine.promotions) == 1

    second = run_experiment(
        Experiment(
            base=ExecutionState.from_run(run), evaluator=ExactReferenceMatch(),
            arms=[DeleteSource(ContextSelection([source_ids[0]]))],
        ),
        adapter,
    )
    assert second.arms[0].status == "exact_preserved"
    assert engine.probe_rounds[-1][0] == 1

    probes_before = len(engine.probe_rounds)
    adapter.on_candidate_accepted(evidence={
        "disposition": "reused", "observation_status": "exact_preserved",
        "observation_id": first.arms[0].observation_id,
    })
    assert len(engine.probe_rounds) == probes_before
    assert len(engine.promotions) == 1
    adapter.close()
    assert engine.closed == ["shared-session"]


def test_shared_parent_promotion_failure_preserves_observation_and_closes_session():
    run, source_ids = _run()
    run["behavior"]["active_dials"] = {}
    engine = _SharedEngine(fail_promotion=True)
    adapter = DeleteSourceExactReferenceAdapter(
        _SharedSubstrate(), run=run, engine=engine, execution_strategy="shared_parent",
    )
    result = run_experiment(_shared_experiment(run, source_ids), adapter)
    observation_id = result.arms[0].observation_id
    adapter.on_candidate_accepted(evidence={
        "disposition": "executed", "observation_status": "exact_preserved",
        "observation_id": observation_id,
    })
    assert result.arms[0].observation_id == observation_id
    assert result.arms[0].status == "exact_preserved"
    assert engine.closed == ["shared-session"]


def test_shared_parent_probe_failure_is_executed_failure_without_scalar_retry():
    run, source_ids = _run()
    run["behavior"]["active_dials"] = {}
    engine = _SharedEngine(fail_probe=True)
    substrate = _SharedSubstrate()
    adapter = DeleteSourceExactReferenceAdapter(
        substrate, run=run, engine=engine, execution_strategy="shared_parent",
    )
    result = run_experiment(_shared_experiment(run, source_ids), adapter)
    assert result.state == "failed"
    assert result.arm_dispositions == ("executed", "executed")
    assert len(substrate.calls) == 1  # unchanged control only
    assert engine.closed == ["shared-session"]


def test_shared_parent_partial_failure_preserves_completed_child_evidence():
    run, source_ids = _run()
    run["behavior"]["active_dials"] = {}
    engine = _SharedEngine(partial_probe=True)
    substrate = _SharedSubstrate()
    adapter = DeleteSourceExactReferenceAdapter(
        substrate, run=run, engine=engine, execution_strategy="shared_parent",
    )
    result = run_experiment(_shared_experiment(run, source_ids), adapter)
    assert result.state == "failed"
    assert result.arms[0].status == "exact_preserved"
    assert result.arms[0].observation_id is not None
    assert result.arms[1].state == "failed"
    assert result.arm_dispositions == ("executed", "executed")
    assert len(substrate.calls) == 1
    assert engine.closed == ["shared-session"]


def test_shared_parent_cancellation_keeps_completed_child_and_closes_session():
    run, source_ids = _run()
    run["behavior"]["active_dials"] = {}
    engine = _SharedEngine(cancel_probe=True)
    adapter = DeleteSourceExactReferenceAdapter(
        _SharedSubstrate(), run=run, engine=engine, execution_strategy="shared_parent",
    )
    result = run_experiment(_shared_experiment(run, source_ids), adapter)
    assert result.state == "cancelled"
    assert result.arms[0].status == "exact_preserved"
    assert result.arms[0].observation_id is not None
    assert result.arms[1].state == "cancelled"
    assert result.arms[1].observation_id is None
    assert engine.closed == ["shared-session"]


def test_scalar_and_shared_parent_qualification_match_observations_and_search_result():
    run, _source_ids = _run()
    run["behavior"]["active_dials"] = {}
    run["context_units"] = build_context_unit_manifest(run)
    prompt_cost = lambda messages: sum(len(str(item.get("content", ""))) for item in messages)

    scalar = run_minimal_context(
        deepcopy(run), substrate=ProbeSubstrate(), prompt_token_counter=prompt_cost,
        max_new_counterfactual_observations=8, execution_strategy="scalar",
    )
    engine = _SharedEngine()
    shared = run_minimal_context(
        deepcopy(run), substrate=_SharedSubstrate(), engine=engine,
        prompt_token_counter=prompt_cost, max_new_counterfactual_observations=8,
        execution_strategy="shared_parent",
    )

    assert scalar.status == shared.status == "completed"
    assert scalar.search_id == shared.search_id
    assert scalar.certificate == shared.certificate
    assert scalar.best == shared.best
    assert scalar.trials == shared.trials
    assert scalar.trajectory == shared.trajectory
    assert scalar.budget.used_new_executions == shared.budget.used_new_executions
    assert engine.closed == ["shared-session"]
    assert engine.probe_rounds
    assert engine.shared_work < engine.naive_work
