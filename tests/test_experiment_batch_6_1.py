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
from clozn.experiments.scoring import (
    DeleteSourceRecordedContinuationScoreAdapter,
    ScoreTokensRefinementPolicy,
)
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
        score_tokens_many_execution_strategy = "native_prefix_reuse"

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
    assert result.arms[0].status == "unavailable"
    assert result.arms[0].observation.trusted is False
    assert result.arms[0].observation.diagnostics["reason"] == "unqualified_scoring_execution"
    assert result.arms[0].observation_id is None
    assert result.arms[1].state == "failed"
    assert result.arm_dispositions == ("executed", "executed")


def test_score_batch_reports_scalar_http_concurrency_and_preserves_observation_identity():
    run, source_ids = _score_run()

    class ConcurrentScalar(ScoreSubstrate):
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "scalar_http_concurrent"

        def score_tokens_many(self, arms, *, cancel=None):
            return [self.score_tokens(**arm) for arm in arms]

    experiment = Experiment(
        base=ExecutionState.from_run(run), evaluator=ScoreRecordedContinuation(),
        arms=[
            DeleteSource(ContextSelection([source_ids[0]])),
            DeleteSource(ContextSelection([source_ids[1]])),
        ],
    )
    serial = run_experiment(
        experiment, DeleteSourceRecordedContinuationScoreAdapter(ScoreSubstrate(), run=run),
    )
    concurrent = run_experiment(
        experiment, DeleteSourceRecordedContinuationScoreAdapter(ConcurrentScalar(), run=run),
    )

    assert concurrent.diagnostics["execution_strategy"] == "scalar_http_concurrent"
    assert concurrent.diagnostics["numerical_regime"] == "scalar_reference"
    assert concurrent.diagnostics["qualified"] is True
    assert [row.observation_id for row in concurrent.arms] == [row.observation_id for row in serial.arms]
    assert all(row.observation.trusted is True for row in concurrent.arms)
    assert all(
        row.observation.execution_provenance["execution_strategy"] == "scalar_http_concurrent"
        for row in concurrent.arms
    )


def test_successful_native_score_batch_without_metadata_is_quarantined():
    run, source_ids = _score_run()

    class UnqualifiedNative(ScoreSubstrate):
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "native_multisequence"

        def score_tokens_many(self, arms, *, cancel=None):
            return [self.score_tokens(**arm) for arm in arms]

    experiment = Experiment(
        base=ExecutionState.from_run(run), evaluator=ScoreRecordedContinuation(),
        arms=[DeleteSource(ContextSelection([source_ids[0]]))],
    )
    result = run_experiment(
        experiment, DeleteSourceRecordedContinuationScoreAdapter(UnqualifiedNative(), run=run),
    )

    assert result.state == "failed"
    assert result.diagnostics["execution_strategy"] == "native_multisequence"
    assert result.diagnostics["qualified"] is False
    assert result.arms[0].status == "unavailable"
    assert result.arms[0].observation_id is None
    assert result.arms[0].observation.trusted is False
    assert result.arms[0].observation.diagnostics["reason"] == "unqualified_scoring_execution"


def test_qualified_native_score_batch_is_reported_but_quarantined_from_scalar_observation_key():
    run, source_ids = _score_run()

    class QualifiedNative(ScoreSubstrate):
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "native_prefix_reuse"

        def score_tokens_many(self, arms, *, cancel=None):
            return {
                "results": [self.score_tokens(**arm) for arm in arms],
                "execution_metadata": {
                    "execution_strategy": "native_prefix_reuse",
                    "numerical_regime": "prefix_reuse_fp32_v1",
                    "evidence_use": "trusted",
                    "qualified": True,
                    "execution_identity": {"implementation": "fixture", "version": 1},
                },
                "metrics": {"shared_prefix_tokens": 9},
            }

    experiment = Experiment(
        base=ExecutionState.from_run(run), evaluator=ScoreRecordedContinuation(),
        arms=[DeleteSource(ContextSelection([source_ids[0]]))],
    )
    result = run_experiment(
        experiment, DeleteSourceRecordedContinuationScoreAdapter(QualifiedNative(), run=run),
    )

    assert result.state == "failed"
    assert result.diagnostics["execution_strategy"] == "native_prefix_reuse"
    assert result.diagnostics["numerical_regime"] == "prefix_reuse_fp32_v1"
    assert result.diagnostics["qualified"] is True
    assert result.diagnostics["execution_metadata"]["metrics"] == {"shared_prefix_tokens": 9}
    assert result.arms[0].status == "unavailable"
    assert result.arms[0].observation_id is None
    assert result.arms[0].observation.trusted is False
    assert result.arms[0].observation.proof_grade == "unqualified"
    assert result.arms[0].observation.diagnostics["reason"] == "unqualified_scoring_execution"


def test_native_screen_scalar_refinement_persists_only_explicit_scalar_arm_evidence(
        tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, source_ids = _score_run()

    class ScreeningNative(ScoreSubstrate):
        score_tokens_many_proof_grade = True
        # Capability and rollout policy are separate.  This fixture mirrors a
        # real EngineSubstrate with automatic native admission disabled.
        score_tokens_many_execution_strategy = "scalar_serial"
        native_score_variants_available = True

        def __init__(self):
            super().__init__()
            self.native_calls = 0

        def score_tokens_many(
            self, arms, *, cancel=None, require_native_screening=False,
        ):
            assert require_native_screening is True
            self.native_calls += 1
            return {
                "results": [
                    [
                        {"id": token_id, "logprob": -(arm_index + token_index + 1.0)}
                        for token_index, token_id in enumerate(arm["continuation_ids"])
                    ]
                    for arm_index, arm in enumerate(arms)
                ],
                "execution_metadata": {
                    "execution_strategy": "native_prefix_reuse_step_batch",
                    "numerical_regime": "screening_unqualified",
                    "evidence_use": "screening",
                    "qualified": False,
                    "execution_identity": {"implementation": "screen-fixture", "version": 1},
                },
            }

    experiment = Experiment(
        base=ExecutionState.from_run(run), evaluator=ScoreRecordedContinuation(),
        arms=[
            DeleteSource(ContextSelection([source_ids[0]])),
            DeleteSource(ContextSelection([source_ids[1]])),
        ],
    )
    substrate = ScreeningNative()
    policy = ScoreTokensRefinementPolicy(
        proof_critical_arm_ids=(experiment.arms[0].arm_id,),
        borderline_selector=lambda _rows: (),
    )
    store = ObservationStore()
    result = run_experiment(
        experiment,
        DeleteSourceRecordedContinuationScoreAdapter(
            substrate, run=run, refinement_policy=policy,
        ),
        observation_store=store,
    )

    assert substrate.native_calls == 1
    assert len(substrate.calls) == 2  # scalar control + one scalar refinement
    assert result.state == "failed"
    refined, screened = result.arms
    assert refined.status == "completed"
    assert refined.observation.trusted is True
    assert refined.observation.proof_grade == "trusted"
    assert refined.observation.execution_provenance["numerical_regime"] == "scalar_reference"
    assert refined.observation.execution_provenance["execution_identity"] == {
        "kind": "clozn.score-tokens-scalar-reference.v1",
    }
    assert "screening_execution" not in refined.observation.execution_provenance
    assert refined.observation_id is not None
    assert screened.status == "unavailable"
    assert screened.observation is None
    assert screened.diagnostics["observation_status"] == "unavailable"
    assert screened.diagnostics["diagnostics"]["reason"] == (
        "screening_not_selected_for_scalar_refinement"
    )
    assert screened.observation_id is None
    assert result.diagnostics["execution_strategy"] == "native_screen_then_scalar_refine"
    assert result.diagnostics["screening_execution"]["numerical_regime"] == (
        "screening_unqualified"
    )
    assert result.diagnostics["scalar_refinement_execution"]["numerical_regime"] == (
        "scalar_reference"
    )
    assert "numerical_regime" not in result.diagnostics


def test_native_screen_refinement_rejects_unknown_proof_arm_before_native_dispatch():
    run, source_ids = _score_run()

    class ScreeningNative(ScoreSubstrate):
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "native_prefix_reuse"

        def __init__(self):
            super().__init__()
            self.native_calls = 0

        def score_tokens_many(self, arms, *, cancel=None):
            self.native_calls += 1
            raise AssertionError("native screen must not be dispatched")

    experiment = Experiment(
        base=ExecutionState.from_run(run), evaluator=ScoreRecordedContinuation(),
        arms=[DeleteSource(ContextSelection([source_ids[0]]))],
    )
    substrate = ScreeningNative()
    result = run_experiment(
        experiment,
        DeleteSourceRecordedContinuationScoreAdapter(
            substrate,
            run=run,
            refinement_policy=ScoreTokensRefinementPolicy(
                proof_critical_arm_ids=("unknown-arm",),
            ),
        ),
    )

    assert substrate.native_calls == 0
    assert len(substrate.calls) == 1  # validated scalar control only
    assert result.state == "failed"
    assert result.arms[0].observation_id is None


def test_refinement_policy_does_not_change_default_scalar_execution():
    run, source_ids = _score_run()
    experiment = Experiment(
        base=ExecutionState.from_run(run), evaluator=ScoreRecordedContinuation(),
        arms=[
            DeleteSource(ContextSelection([source_ids[0]])),
            DeleteSource(ContextSelection([source_ids[1]])),
        ],
    )
    substrate = ScoreSubstrate()
    result = run_experiment(
        experiment,
        DeleteSourceRecordedContinuationScoreAdapter(
            substrate,
            run=run,
            refinement_policy=ScoreTokensRefinementPolicy(
                proof_critical_arm_ids=(experiment.arms[0].arm_id,),
            ),
        ),
    )

    assert len(substrate.calls) == 3  # scalar control + both scalar arms
    assert result.state == "completed"
    assert result.diagnostics["execution_strategy"] == "scalar_serial"
    assert all(row.observation.trusted is True for row in result.arms)


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
