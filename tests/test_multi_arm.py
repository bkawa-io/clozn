from __future__ import annotations

from threading import Event

import pytest

from clozn.runs.minimal_context import run_minimal_context_search
from clozn.runs.multi_arm import (
    BatchCancelled,
    MultiArmError,
    concurrent_many,
    probe_reference_match_many,
    score_tokens_many,
)


CONTRACT = {
    "decode_mode": "greedy",
    "max_new": 4,
    "stop": [],
    "expected_termination": {"reason": "stop", "reason_raw": "eos"},
}


class ScalarSubstrate:
    def __init__(self):
        self.score_calls = []
        self.probe_calls = []
        self.chat_calls = 0
        self.live_strengths = {"warm": 9.0}

    def score_tokens(self, messages, continuation_ids, *, block=None, steer_strengths=None,
                     steer_vec=None, topk=0, continuation=None):
        self.score_calls.append({
            "messages": messages,
            "continuation_ids": continuation_ids,
            "block": block,
            "steer_strengths": steer_strengths,
            "steer_vec": steer_vec,
            "topk": topk,
            "continuation": continuation,
        })
        value = float(len(str(block or ""))) + float((steer_strengths or {}).get("warm", 0.0))
        return [{"id": token, "piece": str(token), "logprob": -(value + index)}
                for index, token in enumerate(continuation_ids or [1, 2])]

    def probe_reference_match(self, messages, reference_token_ids, *, generation_contract,
                              explicit_conditions=None):
        conditions = dict(explicit_conditions or {})
        self.probe_calls.append({
            "messages": messages,
            "reference_token_ids": reference_token_ids,
            "generation_contract": generation_contract,
            "explicit_conditions": conditions,
        })
        block = str(conditions.get("block") or "")
        if block == "termination-mismatch":
            return {
                "status": "diverged",
                "matched_token_count": 2,
                "first_divergence_index": None,
                "divergence_kind": "termination_mismatch",
                "termination_match": False,
            }
        if block == "diverge-late":
            return {
                "status": "diverged",
                "matched_token_count": 3,
                "first_divergence_index": 3,
                "divergence_kind": "token_mismatch",
                "termination_match": True,
            }
        return {
            "status": "matched",
            "matched_token_count": len(reference_token_ids),
            "first_divergence_index": None,
            "divergence_kind": None,
            "termination_match": True,
        }


class NativeBatchSubstrate(ScalarSubstrate):
    """A deliberately completion-reordered native fake."""

    def score_tokens_many(self, arms, *, cancel=None):
        results = [{"arm_index": index, "result": self.score_tokens(**arm)}
                   for index, arm in enumerate(arms)]
        return list(reversed(results))

    def probe_reference_match_many(self, arms, *, cancel=None):
        results = [{"arm_index": index, "result": self.probe_reference_match(**arm)}
                   for index, arm in enumerate(arms)]
        return list(reversed(results))


def _score_arms():
    return [
        {"messages": [{"role": "user", "content": "q"}], "continuation_ids": [11, 12],
         "block": "a", "steer_strengths": {"warm": 1.0}},
        {"messages": [{"role": "user", "content": "q"}], "continuation_ids": [11, 12],
         "block": "b", "steer_strengths": {"warm": 1.0}},
        {"messages": [{"role": "user", "content": "q"}], "continuation_ids": [11, 12],
         "block": "c", "steer_strengths": {"warm": 1.0}},
    ]


def _probe_arms():
    return [
        {"messages": [{"role": "user", "content": "q"}], "reference_token_ids": [1, 2, 3, 4],
         "generation_contract": CONTRACT, "explicit_conditions": {"block": "matched"}},
        {"messages": [{"role": "user", "content": "q"}], "reference_token_ids": [1, 2, 3, 4],
         "generation_contract": CONTRACT, "explicit_conditions": {"block": "diverge-late"}},
        {"messages": [{"role": "user", "content": "q"}], "reference_token_ids": [1, 2, 3, 4],
         "generation_contract": CONTRACT, "explicit_conditions": {"block": "termination-mismatch"}},
    ]


def test_scalar_and_native_batch_score_results_are_observationally_equal_and_ordered():
    scalar = ScalarSubstrate()
    expected = [scalar.score_tokens(**arm) for arm in _score_arms()]
    native = NativeBatchSubstrate()
    actual = score_tokens_many(native, _score_arms())
    assert actual == expected
    assert [call["block"] for call in native.score_calls] == ["a", "b", "c"]
    assert native.live_strengths == {"warm": 9.0}
    assert native.chat_calls == 0


def test_scalar_and_native_batch_exact_results_preserve_mixed_divergence_and_termination():
    scalar = ScalarSubstrate()
    expected = [scalar.probe_reference_match(**arm) for arm in _probe_arms()]
    native = NativeBatchSubstrate()
    actual = probe_reference_match_many(native, _probe_arms())
    assert actual == expected
    assert [item["status"] for item in actual] == ["matched", "diverged", "diverged"]
    assert actual[1]["first_divergence_index"] == 3
    assert actual[2]["divergence_kind"] == "termination_mismatch"


def test_batch_cancellation_returns_completed_results_and_does_not_dispatch_queued_groups():
    substrate = ScalarSubstrate()
    cancel = Event()
    original = substrate.score_tokens

    def score_and_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        cancel.set()
        return result

    substrate.score_tokens = score_and_cancel
    with pytest.raises(BatchCancelled) as raised:
        score_tokens_many(substrate, _score_arms(), cancel=cancel)
    assert len(raised.value.completed) == 1
    assert len(substrate.score_calls) == 1


def test_different_messages_share_one_native_score_batch_when_contract_matches():
    class GroupingSubstrate(ScalarSubstrate):
        def __init__(self):
            super().__init__()
            self.batches = []

        def score_tokens_many(self, arms, *, cancel=None):
            self.batches.append(list(arms))
            return [self.score_tokens(**arm) for arm in arms]

    substrate = GroupingSubstrate()
    arms = [
        {"messages": [{"role": "user", "content": "minus A"}], "continuation_ids": [1, 2], "block": None},
        {"messages": [{"role": "user", "content": "minus B"}], "continuation_ids": [1, 2], "block": None},
        {"messages": [{"role": "user", "content": "minus C"}], "continuation_ids": [1, 2], "block": None},
    ]
    score_tokens_many(substrate, arms)
    assert len(substrate.batches) == 1
    assert [arm["messages"][0]["content"] for arm in substrate.batches[0]] == [
        "minus A", "minus B", "minus C"
    ]


def test_different_messages_share_exact_batch_but_contract_changes_split_groups():
    class GroupingSubstrate(ScalarSubstrate):
        def __init__(self):
            super().__init__()
            self.batches = []

        def probe_reference_match_many(self, arms, *, cancel=None):
            self.batches.append(list(arms))
            return [self.probe_reference_match(**arm) for arm in arms]

    substrate = GroupingSubstrate()
    arms = [
        {"messages": [{"role": "user", "content": "minus A"}], "reference_token_ids": [1, 2],
         "generation_contract": CONTRACT, "explicit_conditions": {"block": None}},
        {"messages": [{"role": "user", "content": "minus B"}], "reference_token_ids": [1, 2],
         "generation_contract": CONTRACT, "explicit_conditions": {"block": None}},
        {"messages": [{"role": "user", "content": "different contract"}], "reference_token_ids": [1, 2],
         "generation_contract": {**CONTRACT, "max_new": 5}, "explicit_conditions": {"block": None}},
    ]
    probe_reference_match_many(substrate, arms)
    assert [len(batch) for batch in substrate.batches] == [2, 1]


def test_different_messages_cancel_inside_one_scalar_group_without_unavailable_evidence():
    substrate = ScalarSubstrate()
    cancel = Event()
    original = substrate.score_tokens

    def score_and_cancel(*args, **kwargs):
        result = original(*args, **kwargs)
        if len(substrate.score_calls) == 3:
            cancel.set()
        return result

    substrate.score_tokens = score_and_cancel
    arms = [
        {"messages": [{"role": "user", "content": f"minus {index}"}], "continuation_ids": [1, 2], "block": None}
        for index in range(5)
    ]
    with pytest.raises(BatchCancelled) as raised:
        score_tokens_many(substrate, arms, cancel=cancel)
    assert len(substrate.score_calls) == 3
    assert len(raised.value.completed) == 3


def test_bounded_concurrent_many_preserves_scalar_result_order():
    calls = []

    def scalar(value):
        calls.append(value)
        return {"value": value}

    result = concurrent_many(scalar, [{"value": 0}, {"value": 1}, {"value": 2}], max_workers=2)
    assert result == [{"value": 0}, {"value": 1}, {"value": 2}]
    assert sorted(calls) == [0, 1, 2]


def test_malformed_arm_is_rejected_before_any_dispatch():
    substrate = ScalarSubstrate()
    with pytest.raises(MultiArmError) as raised:
        score_tokens_many(substrate, [_score_arms()[0], {"messages": "not-a-list"}])
    assert raised.value.arm_index == 1
    assert substrate.score_calls == []


def test_minimal_context_solver_consumes_batch_measurement_without_changing_certificate_logic():
    batches = []

    def measure_many(removed_sets):
        removed_sets = tuple(tuple(item) for item in removed_sets)
        batches.append(removed_sets)
        return [
            {
                "experiment_id": "exp_" + "_".join(removed),
                "removed_source_ids": list(removed),
                "delta_nats": 0.0 if len({"a", "b", "c", "d"}.difference(removed)) >= 2 else 1.0,
                "provenance": "measured",
            }
            for removed in removed_sets
        ]

    result = run_minimal_context_search(
        ["a", "b", "c", "d"], None, measure_removed_many=measure_many,
        tolerance_nats=0.1, search_probe_budget=20, certification_probe_budget=20,
        run_id="run_batch",
    )
    assert result["status"] == "found"
    assert result["certificate"]["kind"] == "exact_minimum"
    assert result["candidate"]["retained_source_count"] == 2
    assert batches
