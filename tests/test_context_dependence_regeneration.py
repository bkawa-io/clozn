"""Focused model-free Task 8 regeneration tests.

The direct score fake creates the existing measured study.  The separate
generation fake has a tripwire ``score_tokens`` method, proving this optional
path makes one replay generation and never re-scores the Context Dependence
likelihood experiment.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

import clozn.runs.store as runlog
from clozn.receipts.context_dependence import measure_removal_effect
from clozn.runs.context_dependence_regeneration import (
    ContextDependenceRegenerationError,
    execute_context_dependence_regeneration,
    plan_context_dependence_regeneration,
)
from clozn.runs.context_receipt import build_context_receipt


class _ScoreSub:
    def score_tokens(self, messages, continuation_ids, **_kwargs):
        removed = 2 - len(messages)
        return [
            {"id": 11, "piece": "same", "logprob": -0.2 - removed},
            {"id": 12, "piece": " answer", "logprob": -0.3 - removed},
        ]


class _GenerationSub:
    def __init__(self):
        self.calls = 0
        self.seen_messages = None
        self.seen_sample = None
        self.seen_dials = None
        self.steer = _Steer({"unrelated_live_dial": 9.0})

    def score_tokens(self, *_args, **_kwargs):  # pragma: no cover - contract tripwire
        raise AssertionError("Task 8 regeneration must not call score_tokens")

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self.calls += 1
        self.seen_messages = deepcopy(messages)
        self.seen_sample = sample
        self.seen_dials = dict(self.steer.strength)
        if mem_out is not None:
            mem_out.update(assembled_messages=deepcopy(messages), final_prompt="child exact prompt")
        if trace_out is not None:
            trace_out.extend([
                {"id": 31, "piece": "changed", "conf": 0.8, "alts": []},
                {"id": 32, "piece": " answer", "conf": 0.7, "alts": []},
            ])
        return "changed answer"


class _Steer:
    def __init__(self, strength):
        self.strength = dict(strength)

    def clear(self):
        self.strength = {}

    def set(self, name, value):
        self.strength[name] = float(value)

    def active(self):
        return dict(self.strength)


def _parent() -> tuple[dict, list[str]]:
    messages = [
        {"role": "system", "content": "source A"},
        {"role": "user", "content": "source B"},
    ]
    receipt = build_context_receipt(
        messages=messages, assembled_messages=messages, final_prompt="parent exact prompt",
        run_id="run_cd_regen", privacy="full",
    )
    parent = {
        "id": "run_cd_regen",
        "model": "fixture-model",
        "substrate": "fixture",
        "messages": deepcopy(messages),
        "assembled_messages": deepcopy(messages),
        "context_receipt": receipt,
        "final_prompt": "parent exact prompt",
        "response": "same answer",
        "behavior": {"active_dials": {"careful": 0.5}},
        "trace": {
            "token_ids": [11, 12],
            "tokens": ["same", " answer"],
            "confidence": [0.9, 0.8],
            "alternatives": [[], []],
        },
    }
    return parent, [segment["segment_id"] for segment in receipt["assembled"]]


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return runlog


def _study(parent, source_ids):
    return measure_removal_effect(parent, _ScoreSub(), removed_source_ids=source_ids)


def test_plan_revalidates_the_existing_measured_experiment_against_strict_receipt_deletion():
    parent, source_ids = _parent()
    study = _study(parent, source_ids)
    experiment = study["experiments"][0]

    plan = plan_context_dependence_regeneration(parent, study, experiment["experiment_id"])

    assert plan["state"] == "ready"
    assert plan["teacher_forced_reference"]["measurement_mode"] == "teacher_forced"
    assert plan["teacher_forced_reference"]["experiment_id"] == experiment["experiment_id"]
    assert plan["intervention"]["removed_source_ids"] == sorted(source_ids)
    assert plan["intervention"]["exact_removed_ranges"] == experiment["exact_removed_ranges"]
    assert plan["execution"]["generation_calls"] == 1
    assert [message["content"] for message in plan["execution"]["messages_override"]] == []


def test_executor_generates_once_persists_provenance_and_returns_canonical_first_divergence(store):
    parent, source_ids = _parent()
    study = _study(parent, source_ids)
    experiment_id = study["experiments"][0]["experiment_id"]
    plan = plan_context_dependence_regeneration(parent, study, experiment_id)
    sub = _GenerationSub()

    result = execute_context_dependence_regeneration(parent, study, experiment_id, sub, plan=plan)

    assert sub.calls == 1
    assert sub.seen_messages == []
    assert sub.seen_sample is False
    assert sub.seen_dials == {"careful": 0.5}
    assert sub.steer.strength == {"unrelated_live_dial": 9.0}
    assert result["teacher_forced_reference"]["measurement_mode"] == "teacher_forced"
    regeneration = result["regeneration"]
    assert regeneration["measurement_mode"] == "free_generation"
    assert regeneration["causal_confirmation"] is False
    assert regeneration["comparison"]["first_divergence_view"]["state"] == "available"
    child = store.get_run(regeneration["child_run_id"])
    applied = child["changes_applied"]["context_dependence_regeneration"]
    assert child["parent_run_id"] == parent["id"]
    assert applied["parent_run_id"] == parent["id"]
    assert applied["experiment_id"] == experiment_id
    assert applied["intervention_operator"] == "delete_source"
    assert applied["removed_canonical_source_ids"] == sorted(source_ids)
    assert applied["exact_removed_ranges"] == study["experiments"][0]["exact_removed_ranges"]
    assert len(applied["intervened_context_digest"]) == 64


def test_planner_refuses_study_experiment_when_exact_ranges_no_longer_bind_current_receipt():
    parent, source_ids = _parent()
    study = _study(parent, source_ids)
    study["experiments"][0]["exact_removed_ranges"][0]["message_index"] = 99

    with pytest.raises(ContextDependenceRegenerationError, match="exact_removed_ranges"):
        plan_context_dependence_regeneration(parent, study, study["experiments"][0]["experiment_id"])
