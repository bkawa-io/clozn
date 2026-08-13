"""Focused direct-verification tests for preserving Context Dependence sets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pytest

from clozn.runs.context_dependence_preserving import (
    COMPUTE_LEVELS,
    ContextDependencePreservingError,
    PreservingMeasurementError,
    compute_level_presets,
    get_compute_level_policy,
    run_preserving_subset_search,
    run_preserving_subset_search_for_study,
)
from tests.context_dependence_cases import case_by_id


class DirectCaseAdapter:
    """Expose a Task 2 synthetic world as auditable Task 1-shaped records."""

    def __init__(self, case_id: str):
        self.case = case_by_id(case_id)
        self.scorer = self.case.scorer()

    def measure(self, removed_source_ids: Iterable[str]):
        result = self.scorer.measure(removed_source_ids)
        # Synthetic observations are direct but do not need production-style
        # content-addressed IDs themselves, so add a deterministic bridge ID.
        return {
            "experiment_id": "synthetic_" + "__".join(result.removed_source_ids),
            "removed_source_ids": list(result.removed_source_ids),
            "delta_nats": result.delta_nats,
            "provenance": result.provenance,
        }


def test_irrelevant_source_is_removed_from_a_directly_verified_preserving_set():
    adapter = DirectCaseAdapter("one_necessary_source")
    necessary, filler = adapter.case.source_ids
    result = run_preserving_subset_search(
        adapter.case.source_ids, adapter.measure, tolerance_nats=0.1,
        passes_requested=3, candidate_retained_source_sets=[[], [necessary], [filler]],
        source_token_counts={necessary: 11, filler: 7}, max_candidates=3,
    )

    preserving = result["preserving_subsets"]
    assert [record["retained_source_ids"] for record in preserving] == [[necessary]]
    record = preserving[0]
    assert record["removed_source_ids"] == [filler]
    assert record["experiment_id"] == "synthetic_" + filler
    assert record["provenance"] == "measured"
    assert record["full_source_count"] == 2
    assert record["retained_source_count"] == 1
    assert record["full_token_count"] == 18
    assert record["retained_token_count"] == 11
    assert record["measured_difference_nats"] == 0.0
    assert record["absolute_difference_nats"] == 0.0
    assert all("useless" not in str(item).lower() for item in result.values())


def test_duplicate_evidence_retains_multiple_directly_verified_solutions():
    adapter = DirectCaseAdapter("exact_duplicate_evidence")
    source_a, source_b = adapter.case.source_ids
    result = run_preserving_subset_search(
        adapter.case.source_ids, adapter.measure, tolerance_nats=0.1,
        passes_requested=3, candidate_retained_source_sets=[[], [source_b], [source_a]],
        max_candidates=3, max_preserving_subsets=4,
    )

    assert [record["retained_source_ids"] for record in result["preserving_subsets"]] == [
        [source_a], [source_b],
    ]
    assert [record["experiment_id"] for record in result["preserving_subsets"]] == [
        "synthetic_" + source_b, "synthetic_" + source_a,
    ]
    assert {record["experiment_id"] for record in result["preserving_subsets"]}.issubset({
        record["experiment_id"] for record in result["tested_retained_subsets"]
    })


def test_no_smaller_subset_satisfies_the_tolerance():
    adapter = DirectCaseAdapter("one_necessary_source")
    necessary, filler = adapter.case.source_ids
    result = run_preserving_subset_search(
        adapter.case.source_ids, adapter.measure, tolerance_nats=0.0,
        passes_requested=2, candidate_retained_source_sets=[[], [filler]], max_candidates=2,
    )

    assert result["preserving_subsets"] == []
    assert all(record["within_tolerance"] is False for record in result["tested_retained_subsets"])
    assert all(record["experiment_id"] for record in result["tested_retained_subsets"])


def test_budget_exhaustion_leaves_later_retained_candidate_explicitly_unmeasured():
    adapter = DirectCaseAdapter("one_necessary_source")
    necessary, _filler = adapter.case.source_ids
    result = run_preserving_subset_search(
        adapter.case.source_ids, adapter.measure, tolerance_nats=0.1,
        passes_requested=1, candidate_retained_source_sets=[[], [necessary]], max_candidates=2,
    )

    assert len(adapter.scorer.calls) == 1
    assert len(result["tested_retained_subsets"]) == 1
    assert result["unmeasured_candidates"] == [{
        "retained_source_ids": [necessary],
        "removed_source_ids": [source_id for source_id in adapter.case.source_ids if source_id != necessary],
        "candidate_origin": "direct_search_candidate",
        "measurement_state": "unmeasured_budget_exhausted",
    }]
    assert result["budget"]["exhausted"] is True
    assert result["search"]["stopped_reason"] == "score_budget_exhausted"


def test_estimated_favorite_must_fail_or_pass_its_own_direct_experiment():
    adapter = DirectCaseAdapter("one_necessary_source")
    necessary, filler = adapter.case.source_ids
    result = run_preserving_subset_search(
        adapter.case.source_ids, adapter.measure, tolerance_nats=0.1,
        passes_requested=2,
        # The estimated nomination keeps only filler and fails when directly
        # measured.  The independently selected necessary-only set passes.
        candidate_retained_source_sets=[[necessary]],
        estimated_candidate_retained_source_sets=[[filler]],
        max_candidates=2,
    )

    by_retained = {tuple(item["retained_source_ids"]): item for item in result["tested_retained_subsets"]}
    assert by_retained[(filler,)]["candidate_origin"] == "estimated_screen_nomination"
    assert by_retained[(filler,)]["within_tolerance"] is False
    assert by_retained[(filler,)]["experiment_id"]
    assert [item["retained_source_ids"] for item in result["preserving_subsets"]] == [[necessary]]
    assert result["search"]["estimated_candidates_are_nominations_only"] is True


def test_determinism_and_smaller_subset_tie_breaking():
    source_ids = ("a", "b", "c")
    calls: list[tuple[str, ...]] = []

    def measure(removed):
        removed = tuple(source_id for source_id in source_ids if source_id in set(removed))
        calls.append(removed)
        # Every candidate preserves, so source-set size is the primary result
        # ordering and canonical receipt order supplies the deterministic tie.
        return {
            "experiment_id": "exp_" + "_".join(removed),
            "removed_source_ids": removed,
            "delta_nats": 0.0,
            "provenance": "measured",
        }

    input_a = [["c"], ["a", "b"], ["a"], ["b"]]
    input_b = list(reversed(input_a))
    first = run_preserving_subset_search(
        source_ids, measure, tolerance_nats=0.0, passes_requested=4,
        candidate_retained_source_sets=input_a, max_candidates=4, max_preserving_subsets=4,
    )
    calls.clear()
    second = run_preserving_subset_search(
        source_ids, measure, tolerance_nats=0.0, passes_requested=4,
        candidate_retained_source_sets=input_b, max_candidates=4, max_preserving_subsets=4,
    )

    assert first == second
    assert [item["retained_source_ids"] for item in first["preserving_subsets"]] == [
        ["a"], ["b"], ["c"], ["a", "b"],
    ]


def test_missing_experiment_id_cannot_certify_a_preserving_set():
    with pytest.raises(PreservingMeasurementError, match="experiment_id"):
        run_preserving_subset_search(
            ("source",),
            lambda removed: {
                "removed_source_ids": list(removed), "delta_nats": 0.0, "provenance": "measured",
            },
            tolerance_nats=0.0, passes_requested=1, candidate_retained_source_sets=[[]],
        )


def test_compute_presets_and_task1_study_wrapper_reserve_cached_baseline():
    assert list(COMPUTE_LEVELS) == ["quick", "standard", "deep"]
    assert get_compute_level_policy("Deep").adaptive_coalitions is True
    assert compute_level_presets()["quick"]["pass_budget"] == 8
    with pytest.raises(ContextDependencePreservingError, match="Quick, Standard, or Deep"):
        get_compute_level_policy("unbounded")

    @dataclass
    class Study:
        source_ids: tuple[str, ...] = ("a", "b")
        calls: int = 0

        def document(self):
            # First document call establishes the one cached full-context
            # baseline.  Each later direct arm increments exactly one pass.
            return {"study_id": "cds_test", "budget": {"passes_consumed": 1 + self.calls}}

        def measure_removal_effect(self, removed):
            self.calls += 1
            removed = tuple(source_id for source_id in self.source_ids if source_id in set(removed))
            return {
                "experiment_id": "real_" + "_".join(removed),
                "removed_source_ids": removed,
                "delta_nats": 0.0,
                "provenance": "measured",
            }

    study = Study()
    result = run_preserving_subset_search_for_study(
        study, tolerance_nats=0.0, passes_requested=2,
        candidate_retained_source_sets=[["a"]], max_candidates=1,
    )
    assert study.calls == 1
    assert result["budget"]["passes_consumed"] == 2
    assert result["search"]["measurement_study_id"] == "cds_test"
    assert result["preserving_subsets"][0]["experiment_id"] == "real_b"


def test_existing_direct_experiment_is_reused_without_charging_another_pass():
    def should_not_measure(_removed):
        raise AssertionError("an existing measured experiment must be reused")

    result = run_preserving_subset_search(
        ("a", "b"), should_not_measure, tolerance_nats=0.0,
        passes_requested=1, initial_passes_consumed=1,
        candidate_retained_source_sets=[["a"]],
        existing_experiments=[{
            "experiment_id": "exp_remove_b",
            "removed_source_ids": ["b"],
            "delta_nats": 0.0,
            "provenance": "measured",
        }],
    )

    assert result["budget"]["passes_consumed"] == 1
    assert result["search"]["direct_measurements_reused"] == 1
    assert result["preserving_subsets"][0]["experiment_id"] == "exp_remove_b"
