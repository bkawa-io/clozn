"""Focused Batch 2 tests: full score vectors and pure answer-span projections."""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn.experiments.evaluators import ScoreRecordedContinuation
from clozn.experiments.execution import ExecutionStateStaleError
from clozn.experiments.interventions import DeleteSource
from clozn.experiments.kernel import Experiment
from clozn.experiments.projection import project_answer_effects
from clozn.experiments.runner import ExperimentResult, run_experiment
from clozn.experiments.scoring import DeleteSourceRecordedContinuationScoreAdapter
from clozn.experiments.selections import (
    AnswerSelection, AnswerSelectionUnavailable, ContextSelection, resolve_answer_selection,
)
from clozn.experiments.state import ExecutionState
import clozn.recipes.context_effects as context_effects_recipe
from clozn.recipes.context_effects import measure_context_effects, project_context_effects
from clozn.runs.context_receipt import build_context_receipt


CONTRACT = {
    "decode_mode": "greedy", "sampling": None, "max_new": 4, "stop": [],
    "expected_termination": {"reason": "eos", "reason_raw": "eos"},
}


def _identity():
    return {
        "model_sha256": "a" * 64, "template_fingerprint": "b" * 16,
        "engine_build": "test-engine", "context_size": 4096, "backend": "cpu",
        "white_box_flags": {"sae": False, "jlens": False, "attn_knockout": False},
    }


def _run():
    messages = [
        {"role": "system", "content": "stable context"},
        {"role": "user", "content": "removable source"},
        {"role": "user", "content": "current question"},
    ]
    run = {
        "id": "run_context_effects_kernel", "model": "fixture-model", "substrate": "fixture",
        "messages": deepcopy(messages), "assembled_messages": deepcopy(messages),
        "final_prompt": "exact parent prompt", "response": "same answer",
        "generation_contract": deepcopy(CONTRACT), "identity": _identity(),
        "meta": {"n_ctx": 4096, "device": "cpu"}, "behavior": {"active_dials": {}},
        "trace": {"tokens": ["same", " answer"], "token_ids": [10, 11],
                  "steps": [{"token_id": 10, "piece": "same"}, {"token_id": 11, "piece": " answer"}]},
    }
    run["context_receipt"] = build_context_receipt(
        messages=messages, assembled_messages=messages, final_prompt=run["final_prompt"],
        run_id=run["id"], privacy="full",
    )
    return run, [segment["segment_id"] for segment in run["context_receipt"]["assembled"]]


class ScoreSubstrate:
    def __init__(self):
        self.calls = []

    def identity_meta(self):
        return _identity()

    def run_meta(self):
        return {"n_ctx": 4096, "device": "cpu"}

    def score_tokens(self, messages, continuation_ids=None, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        self.calls.append(deepcopy(messages))
        contents = [message.get("content") for message in messages]
        if "removable source" not in contents:
            values = [-1.0, -3.0]
        elif "stable context" not in contents:
            values = [-2.0, -0.5]
        else:
            values = [-1.0, -1.0]
        return [
            {"id": token_id, "piece": piece, "logprob": logprob}
            for token_id, piece, logprob in zip(continuation_ids, ["same", " answer"], values)
        ]


def _experiment(run, source_ids):
    return Experiment(
        base=ExecutionState.from_run(run), evaluator=ScoreRecordedContinuation(),
        arms=[
            DeleteSource(ContextSelection([source_ids[1]])),
            DeleteSource(ContextSelection([source_ids[0]])),
        ],
    )


def test_answer_selection_resolves_exact_tokens_and_partial_boundaries():
    run, _source_ids = _run()
    resolved = resolve_answer_selection(run, AnswerSelection.from_range(1, 8))
    assert resolved.selected_text == "ame ans"
    assert resolved.token_indices == (0, 1)
    assert resolved.token_ids == (10, 11)
    assert resolved.token_range == (0, 2)
    assert AnswerSelection("answer").to_dict()["text"] == "answer"


def test_answer_selection_rejects_empty_out_of_range_and_stale_requests():
    run, _source_ids = _run()
    with pytest.raises(Exception, match="non-empty"):
        AnswerSelection.from_range(2, 2)
    with pytest.raises(AnswerSelectionUnavailable, match="outside"):
        resolve_answer_selection(run, AnswerSelection.from_range(0, 99))
    with pytest.raises(AnswerSelectionUnavailable, match="does not match"):
        resolve_answer_selection(run, AnswerSelection.from_range(0, 4, selected_text="stale"))


def test_scoring_reuses_baseline_and_duplicate_arm_evidence():
    run, source_ids = _run()
    intervention = DeleteSource(ContextSelection([source_ids[1]]))
    experiment = Experiment(
        base=ExecutionState.from_run(run), evaluator=ScoreRecordedContinuation(),
        arms=[intervention, intervention],
    )
    substrate = ScoreSubstrate()
    result = run_experiment(
        experiment, DeleteSourceRecordedContinuationScoreAdapter(substrate, run=run),
    )
    assert result.control.status == "completed"
    assert [item.status for item in result.arms] == ["completed", "completed"]
    assert [item.arm_id for item in result.arms] == [arm.arm_id for arm in experiment.arms]
    assert len(substrate.calls) == 2
    assert result.arms[0].token_logprobs == result.arms[1].token_logprobs
    assert result.score_delta_for(result.arms[0].arm_id).deltas == (0.0, 2.0)


def test_score_and_evaluator_serialization_are_deterministic():
    run, source_ids = _run()
    substrate = ScoreSubstrate()
    result = run_experiment(
        _experiment(run, source_ids),
        DeleteSourceRecordedContinuationScoreAdapter(substrate, run=run),
    )
    assert ScoreRecordedContinuation().to_json() == ScoreRecordedContinuation().to_json()
    assert result.to_json() == ExperimentResult.from_dict(result.to_dict()).to_json()


def test_projection_is_signed_sum_and_resupports_without_calls():
    run, source_ids = _run()
    substrate = ScoreSubstrate()
    result = measure_context_effects(
        run, source_ids=[source_ids[1], source_ids[0]], substrate=substrate,
    )
    calls_after_measurement = len(substrate.calls)
    answer = project_context_effects(result, AnswerSelection("answer"), ordering="source")
    first_token = project_context_effects(result, AnswerSelection.from_range(0, 4), ordering="source")
    full = project_context_effects(result, AnswerSelection.from_range(0, len(run["response"])), ordering="source")
    overlapping = project_context_effects(result, AnswerSelection.from_range(3, 8), ordering="source")
    assert len(substrate.calls) == calls_after_measurement
    assert [effect.delta_nats for effect in answer] == [2.0, -0.5]
    assert [effect.delta_nats for effect in first_token] == [0.0, 1.0]
    assert [effect.delta_nats for effect in full] == [2.0, 0.5]
    assert [effect.delta_nats for effect in overlapping] == [2.0, 0.5]
    assert answer[0].provenance["experiment_id"] == result.experiment_id
    assert answer[0].provenance["selected_answer"]["token_ids"] == [11]


def test_selection_does_not_change_experiment_identity_and_abs_order_is_stable():
    run, source_ids = _run()
    first = measure_context_effects(run, source_ids=source_ids[:2], substrate=ScoreSubstrate(),
                                     answer_selection=AnswerSelection("same"))
    second = measure_context_effects(run, source_ids=source_ids[:2], substrate=ScoreSubstrate(),
                                     answer_selection=AnswerSelection("answer"))
    assert first.experiment_id == second.experiment_id
    effects = project_context_effects(first, AnswerSelection.from_range(0, len(run["response"])))
    assert abs(effects[0].delta_nats) >= abs(effects[1].delta_nats)


def test_measurement_forwards_cancellation_to_generic_runner(monkeypatch):
    run, source_ids = _run()
    cancel = object()
    observed = {}

    def fake_run_experiment(*args, **kwargs):
        observed["cancel"] = kwargs.get("cancel")
        return "result"

    monkeypatch.setattr(context_effects_recipe, "run_experiment", fake_run_experiment)
    result = measure_context_effects(
        run, source_ids=source_ids[:1], execution_adapter=object(), cancel=cancel,
    )

    assert result == "result"
    assert observed["cancel"] is cancel


def test_score_failure_is_typed_and_does_not_approximate_text():
    run, source_ids = _run()

    class BadScore(ScoreSubstrate):
        def score_tokens(self, *args, **kwargs):
            self.calls.append(True)
            return [{"id": 999, "piece": "wrong", "logprob": -1.0}]

    result = run_experiment(
        _experiment(run, source_ids),
        DeleteSourceRecordedContinuationScoreAdapter(BadScore(), run=run),
    )
    assert result.control.status == "unavailable"
    assert all(item.status == "unavailable" for item in result.arms)
