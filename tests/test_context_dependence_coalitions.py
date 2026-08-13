"""Focused measured-coalition verification tests.

The synthetic worlds make the effect of every exact source set explicit, but
the verifier sees only a generic direct-measure callback.  That keeps these
tests about selection, pass accounting, and provenance rather than teaching
the production module a second scoring implementation.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn.receipts.context_dependence import ContextDependenceStudy
from clozn.runs.context_dependence_coalitions import (
    CoalitionVerificationError,
    verify_coalitions,
    verify_task1_coalitions,
)
from clozn.runs.context_receipt import build_context_receipt
from tests.context_dependence_cases import case_by_id


class CaseMeasure:
    """A generic measured-experiment callback backed by a benchmark scorer."""

    def __init__(self, case):
        self.case = case
        self.scorer = case.scorer()
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, source_ids: tuple[str, ...]) -> dict:
        measured = self.scorer.measure(source_ids)
        self.calls.append(measured.removed_source_ids)
        # This ID is created by this direct measurement callback, not by the
        # verifier or an estimated screen.  The direct result is intentionally
        # shaped like the Task 1 experiment record.
        return {
            "experiment_id": "measured_" + "__".join(measured.removed_source_ids),
            "removed_source_ids": list(measured.removed_source_ids),
            "provenance": "measured",
            "delta_nats": measured.delta_nats,
        }


def _verified_set(result: dict) -> tuple[str, ...]:
    assert len(result["verified_sets"]) == 1
    return tuple(result["verified_sets"][0]["source_ids"])


@pytest.mark.parametrize(
    ("case_id", "expected_effect"),
    [
        ("exact_duplicate_evidence", 7.30),
        ("paraphrased_substitutable_duplicate_evidence", 6.90),
    ],
)
def test_weak_singletons_do_not_block_direct_duplicate_or_paraphrase_coalition_testing(
    case_id, expected_effect,
):
    case = case_by_id(case_id)
    measure = CaseMeasure(case)

    result = verify_coalitions(
        measure, source_ids=case.source_ids, passes_requested=1,
    )

    # Neither singleton was measured or used as an eligibility floor.  The
    # bounded receipt-sibling pool selects the joint deletion directly.
    assert measure.calls == [case.source_ids]
    assert _verified_set(result) == case.source_ids
    assert result["measured_experiments"] == [{
        "source_ids": list(case.source_ids),
        "experiment_id": "measured_" + "__".join(case.source_ids),
        "provenance": "measured",
        "measurement_source": "new",
        "delta_nats": expected_effect,
    }]
    assert result["budget"] == {
        "passes_requested": 1, "passes_consumed": 1, "passes_remaining": 0, "state": "exhausted",
    }
    assert all(item["experiment_id"].startswith("measured_") for item in result["verified_sets"])


def test_complementarity_and_three_way_coalition_use_direct_joint_measurements():
    complementarity = case_by_id("a_and_b_complementarity")
    pair_measure = CaseMeasure(complementarity)
    pair = verify_coalitions(pair_measure, source_ids=complementarity.source_ids, passes_requested=1)
    assert pair_measure.calls == [complementarity.source_ids]
    assert pair["measured_experiments"][0]["delta_nats"] == pytest.approx(8.60)

    three_way = case_by_id("three_way_coalition")
    triple_measure = CaseMeasure(three_way)
    triple = verify_coalitions(triple_measure, source_ids=three_way.source_ids, passes_requested=1)
    # The full three-sibling group is scheduled before its bounded pairs, so a
    # one-arm budget can still test the actual three-way deletion.
    assert triple_measure.calls == [three_way.source_ids]
    assert _verified_set(triple) == three_way.source_ids
    assert triple["measured_experiments"][0]["delta_nats"] == pytest.approx(9.20)


def test_high_parent_with_weak_children_is_retained_through_its_existing_direct_experiment():
    case = case_by_id("exact_duplicate_evidence")
    parent_experiment = {
        "experiment_id": "measured_parent_ab",
        "removed_source_ids": list(case.source_ids),
        "provenance": "measured",
        "delta_nats": 7.30,
    }
    hierarchy = {
        "experiments": [parent_experiment],
        "hierarchy": {
            "nodes": [
                {"node_id": "root", "source_ids": list(case.source_ids)},
                {"node_id": "a", "parent_node_id": "root", "source_ids": [case.source_ids[0]]},
                {"node_id": "b", "parent_node_id": "root", "source_ids": [case.source_ids[1]]},
            ],
            "nonadditivity": [{"parent_node_id": "root", "derived_value_nats": 7.23}],
        },
    }
    measure = CaseMeasure(case)

    result = verify_coalitions(
        measure, source_ids=case.source_ids, hierarchy=hierarchy, passes_requested=0,
    )

    assert measure.calls == []
    assert _verified_set(result) == case.source_ids
    assert result["verified_sets"] == [{
        "source_ids": list(case.source_ids), "experiment_id": "measured_parent_ab",
    }]
    assert result["measured_experiments"][0]["measurement_source"] == "existing"
    assert result["measured_experiments"][0]["delta_nats"] == pytest.approx(7.30)


def test_too_small_remaining_budget_leaves_candidate_unverified_without_calling_measure():
    case = case_by_id("exact_duplicate_evidence")
    measure = CaseMeasure(case)
    result = verify_coalitions(
        measure, source_ids=case.source_ids, passes_requested=3, passes_consumed=3,
    )

    assert measure.calls == []
    assert result["verified_sets"] == []
    assert result["selection"]["unverified_candidate_sets"] == [{
        "source_ids": list(case.source_ids), "origins": ["receipt_sibling_pool"],
    }]
    assert result["budget"] == {
        "passes_requested": 3, "passes_consumed": 3, "passes_remaining": 0, "state": "exhausted",
    }


def test_false_estimated_candidate_is_still_only_a_directly_measured_set_not_an_estimate():
    case = case_by_id("irrelevant_filler")
    measure = CaseMeasure(case)
    estimated_screen = {
        "provenance": "estimated",
        "candidate_source_sets": [{
            "source_ids": list(case.source_ids), "estimated_delta_nats": 9.9,
        }],
    }

    result = verify_coalitions(
        measure, source_ids=case.source_ids, screen=estimated_screen, passes_requested=1,
    )

    assert measure.calls == [case.source_ids]
    assert _verified_set(result) == case.source_ids
    assert result["measured_experiments"][0]["delta_nats"] == 0.0
    assert result["measured_experiments"][0]["provenance"] == "measured"
    assert result["selection"]["candidate_sets"][0]["origins"] == [
        "receipt_sibling_pool", "screen:candidate_source_sets",
    ]
    # The estimated 9.9 is not copied into a verified set or a direct result.
    assert all("estimated" not in item for item in result["verified_sets"])
    assert all(item["provenance"] == "measured" for item in result["measured_experiments"])


def test_verification_rejects_missing_or_estimated_experiment_ids_instead_of_fabricating_references():
    case = case_by_id("exact_duplicate_evidence")
    with pytest.raises(CoalitionVerificationError, match="experiment_id"):
        verify_coalitions(
            lambda source_set: {"removed_source_ids": list(source_set), "provenance": "measured"},
            source_ids=case.source_ids, passes_requested=1,
        )
    with pytest.raises(CoalitionVerificationError, match="provenance='measured'"):
        verify_coalitions(
            lambda source_set: {
                "experiment_id": "estimated_only", "removed_source_ids": list(source_set),
                "provenance": "estimated",
            },
            source_ids=case.source_ids, passes_requested=1,
        )


class _Task1Sub:
    def __init__(self):
        self.calls: list[list[str]] = []

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        contents = [message["content"] for message in messages]
        self.calls.append(contents)
        penalty = 2.0 if not contents else 0.0
        return [
            {"id": 1, "piece": "A", "logprob": -1.0 - penalty},
            {"id": 2, "piece": ".", "logprob": -1.0 - penalty},
        ]


def _task1_study() -> ContextDependenceStudy:
    messages = [
        {"role": "user", "content": "same evidence A", "source_id": "fixture-a"},
        {"role": "user", "content": "same evidence B", "source_id": "fixture-b"},
    ]
    receipt = build_context_receipt(
        messages=messages, assembled_messages=messages, final_prompt="fixture prompt",
        run_id="coalition_task1", identity={"template_fingerprint": "0123456789abcdef"}, privacy="full",
    )
    run = {
        "id": "coalition_task1",
        "model": "fixture-model",
        "substrate": "Task1Sub",
        "identity": {"model_sha256": "fixture", "template_fingerprint": "0123456789abcdef"},
        "messages": deepcopy(messages),
        "assembled_messages": deepcopy(messages),
        "context_receipt": receipt,
        "final_prompt": "fixture prompt",
        "response": "A.",
        "trace": {"token_ids": [1, 2]},
    }
    return ContextDependenceStudy(run, _Task1Sub())


def test_task1_study_adapter_accounts_for_cached_baseline_and_uses_task1_experiment_id():
    study = _task1_study()
    result = verify_task1_coalitions(study, passes_requested=2)

    assert result["budget"] == {
        "passes_requested": 2, "passes_consumed": 2, "passes_remaining": 0, "state": "exhausted",
    }
    assert len(result["verified_sets"]) == 1
    experiment_id = result["verified_sets"][0]["experiment_id"]
    assert experiment_id.startswith("cdx_")
    document = study.document()
    assert document["budget"] == {"passes_requested": 2, "passes_consumed": 2}
    assert document["experiments"][0]["experiment_id"] == experiment_id
    assert len(study._sub.calls) == 2  # baseline plus the actual joint removal arm
