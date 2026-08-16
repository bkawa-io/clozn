"""Batch 3 durability tests for standalone observations and arm associations."""
from __future__ import annotations

import pytest

from clozn.experiments.evaluators import ExactReferenceMatch, ScoreRecordedContinuation
from clozn.experiments.interventions import DeleteSource
from clozn.experiments.kernel import Experiment
from clozn.experiments.observations import TokenScoreObservation, execution_observation_identity
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.projection import project_answer_effects
from clozn.experiments.runner import run_experiment
from clozn.experiments.scoring import DeleteSourceRecordedContinuationScoreAdapter
from clozn.experiments.selections import AnswerSelection, ContextSelection
from clozn.experiments.state import ExecutionState
from clozn.runs import store as run_store

from tests.test_context_effects_kernel import ScoreSubstrate, _run


def _experiment(run, source_ids, *, evaluator=None, source_index=0):
    return Experiment(
        base=ExecutionState.from_run(run), evaluator=evaluator or ScoreRecordedContinuation(),
        arms=[DeleteSource(ContextSelection([source_ids[source_index]]))],
    )


def test_observation_identity_is_independent_of_arm_and_selection():
    run, source_ids = _run()
    state = ExecutionState.from_run(run)
    intervention = DeleteSource(ContextSelection([source_ids[0]]))
    first = execution_observation_identity(state, ScoreRecordedContinuation(), intervention)
    second = execution_observation_identity(state, ScoreRecordedContinuation(), intervention)
    assert first["observation_id"] == second["observation_id"]
    assert first["observation_key_sha256"] == second["observation_key_sha256"]
    assert "arm_id" not in first["observation_key"]
    assert first["observation_id"] != execution_observation_identity(
        state, ScoreRecordedContinuation(), DeleteSource(ContextSelection([source_ids[1]])),
    )["observation_id"]
    assert first["observation_id"] != execution_observation_identity(
        state, ExactReferenceMatch(), intervention,
    )["observation_id"]


def test_durable_score_observation_round_trip_and_cross_experiment_reuse(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, source_ids = _run()
    store = ObservationStore()
    first_substrate = ScoreSubstrate()
    first = run_experiment(
        _experiment(run, source_ids),
        DeleteSourceRecordedContinuationScoreAdapter(first_substrate, run=run),
        observation_store=store,
    )
    view = store.get_experiment(first.experiment_id)
    assert view.control.observation_id == first.control.observation_id
    assert view.arms[0].observation_id == first.arms[0].observation_id
    assert store.get_observation(view.arms[0].observation_id).to_json() == view.arms[0].observation.to_json()

    second_experiment = Experiment(
        base=ExecutionState.from_run(run), evaluator=ScoreRecordedContinuation(),
        arms=[
            DeleteSource(ContextSelection([source_ids[1]])),
            DeleteSource(ContextSelection([source_ids[0]])),
        ],
    )
    second_substrate = ScoreSubstrate()
    second = run_experiment(
        second_experiment,
        DeleteSourceRecordedContinuationScoreAdapter(second_substrate, run=run),
        observation_store=store,
    )
    assert len(second_substrate.calls) == 1
    assert second.arms[1].observation_id == first.arms[0].observation_id


def test_conflicting_evidence_under_one_key_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, source_ids = _run()
    store = ObservationStore()
    result = run_experiment(
        _experiment(run, source_ids),
        DeleteSourceRecordedContinuationScoreAdapter(ScoreSubstrate(), run=run),
        observation_store=store,
    )
    original = result.arms[0].observation
    conflicting = TokenScoreObservation(
        observation_id=original.observation_id,
        observation_key_sha256=original.observation_key_sha256,
        observation_key=original.observation_key,
        run_id=original.run_id,
        base_execution_fingerprint=original.base_execution_fingerprint,
        evaluator=original.evaluator,
        condition=original.condition,
        contract=original.contract,
        status="completed",
        recorded_token_ids=original.recorded_token_ids,
        token_pieces=original.token_pieces,
        token_spans=original.token_spans,
        token_logprobs=tuple(value + 1 for value in original.token_logprobs),
        total_continuation_logprob=original.total_continuation_logprob + len(original.token_logprobs),
        evaluator_provenance=original.evaluator_provenance,
        score_basis=original.score_basis,
        execution_provenance=original.execution_provenance,
        proof_grade=original.proof_grade,
        trusted=original.trusted,
        diagnostics=original.diagnostics,
    )
    with pytest.raises(Exception, match="conflicting|different evidence"):
        store.persist_observation(conflicting)


def test_cancelled_experiment_has_no_unexecuted_observations(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, source_ids = _run()
    experiment = Experiment(
        base=ExecutionState.from_run(run), evaluator=ScoreRecordedContinuation(),
        arms=[
            DeleteSource(ContextSelection([source_ids[0]])),
            DeleteSource(ContextSelection([source_ids[1]])),
        ],
    )
    calls = {"count": 0}

    def cancel():
        calls["count"] += 1
        return calls["count"] >= 3

    result = run_experiment(
        experiment,
        DeleteSourceRecordedContinuationScoreAdapter(ScoreSubstrate(), run=run),
        observation_store=ObservationStore(), cancel=cancel,
    )
    assert result.state == "cancelled"
    assert result.arms[0].state == "completed"
    assert result.arms[0].observation_id is not None
    assert result.arms[1].state == "cancelled"
    assert result.arms[1].observation_id is None
    assert result.arms[1].observation is None


def test_projection_references_observations_without_embedding_vectors(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, source_ids = _run()
    result = run_experiment(
        _experiment(run, source_ids),
        DeleteSourceRecordedContinuationScoreAdapter(ScoreSubstrate(), run=run),
        observation_store=ObservationStore(),
    )
    effect = project_answer_effects(result, AnswerSelection("answer"), ordering="source")[0]
    measurement = effect.provenance["measurement"]
    assert measurement["baseline_observation_id"] == result.control.observation_id
    assert measurement["intervention_observation_id"] == result.arms[0].observation_id
    assert "delta" not in measurement
    assert "token_logprobs" not in effect.provenance
