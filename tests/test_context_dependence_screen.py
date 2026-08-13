"""Focused model-free coverage for estimated Context Dependence subset screens.

The direct-score fixture remains the authority for every observed mask.  These
tests deliberately inspect the fit/holdout boundary so an additive surrogate
cannot quietly turn into a replacement for an experiment.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from clozn.runs.context_dependence_search import (
    ContextDependenceScreenError,
    SCREEN_PROVENANCE,
    ScreenMeasurementError,
    run_subset_screen,
    sample_subset_masks,
)
from tests.context_dependence_cases import SyntheticMeasurement, case_by_id


@dataclass
class AdditiveMeasuredScorer:
    """Tiny direct-score adapter for a surrogate that should qualify cleanly."""

    source_ids: tuple[str, ...] = ("alpha", "beta", "gamma")

    def __post_init__(self):
        self.calls: list[tuple[str, ...]] = []

    def measure(self, removed_source_ids):
        removed = tuple(source_id for source_id in self.source_ids if source_id in set(removed_source_ids))
        self.calls.append(removed)
        effects = {"alpha": 1.25, "beta": -0.50, "gamma": 0.75}
        delta = 0.2 + sum(effects[source_id] for source_id in removed)
        return {
            "experiment_id": f"direct_{len(self.calls)}",
            "removed_source_ids": list(removed),
            "delta_nats": delta,
            "provenance": "measured",
        }


def _all_nonempty_masks(source_ids):
    return (1 << len(tuple(source_ids))) - 1


def test_seeded_masks_are_deterministic_but_a_different_seed_changes_them():
    source_ids = ("source-a", "source-b", "source-c", "source-d")
    first = sample_subset_masks(source_ids, sampling_seed=41, mask_count=8)
    same = sample_subset_masks(source_ids, sampling_seed=41, mask_count=8)
    different = sample_subset_masks(source_ids, sampling_seed=42, mask_count=8)

    assert first == same
    assert [mask["mask_bits"] for mask in first] != [mask["mask_bits"] for mask in different]
    assert all(mask["removed_source_ids"] for mask in first)
    assert len({mask["mask_id"] for mask in first}) == len(first)


def test_qualified_screen_is_explicitly_estimated_and_keeps_scored_masks_separate():
    scorer = AdditiveMeasuredScorer()
    screen = run_subset_screen(
        scorer.source_ids,
        scorer.measure,
        sampling_seed=5,
        passes_requested=7,
        mask_count=7,
        holdout_fraction=2 / 7,
        min_holdout_observations=2,
        max_holdout_mae_nats=0.05,
    )

    assert screen["provenance"] == SCREEN_PROVENANCE == "estimated"
    assert screen["estimator"]["provenance"] == "estimated"
    assert screen["qualification"] == {
        "state": "qualified",
        "candidate_interpretation_available": True,
        "reasons": [],
    }
    assert len(screen["masks"]) == len(scorer.calls) == 7
    assert all(mask["measurement_provenance"] == "measured" for mask in screen["masks"])
    assert all("measurement_experiment_id" in mask for mask in screen["masks"])
    assert all(coefficient["provenance"] == "estimated" for coefficient in screen["coefficients"])
    assert all(coefficient["not_a_measured_effect"] is True for coefficient in screen["coefficients"])
    assert "experiments" not in screen
    assert screen["candidate_source_sets"] == []
    assert screen["budget"] == {
        "passes_requested": 7,
        "initial_passes_consumed": 0,
        "score_passes_per_mask": 1,
        "passes_consumed": 7,
        "passes_remaining": 0,
        "mask_passes_consumed": 7,
        "exhausted": True,
    }


def test_holdout_masks_are_chosen_before_fit_and_never_used_for_fitting():
    scorer = AdditiveMeasuredScorer()
    screen = run_subset_screen(
        scorer.source_ids, scorer.measure, sampling_seed=13, passes_requested=7,
        mask_count=7, holdout_fraction=2 / 7, min_holdout_observations=2,
        max_holdout_mae_nats=0.05,
    )
    fit_ids = set(screen["training_fit"]["fit_mask_ids"])
    holdout_ids = set(screen["holdout"]["holdout_mask_ids"])

    assert fit_ids and holdout_ids and fit_ids.isdisjoint(holdout_ids)
    assert fit_ids | holdout_ids == {mask["mask_id"] for mask in screen["masks"]}
    assert {mask["mask_id"] for mask in screen["masks"] if mask["split"] == "fit"} == fit_ids
    assert {mask["mask_id"] for mask in screen["masks"] if mask["split"] == "holdout"} == holdout_ids
    assert screen["holdout"]["used_for_fitting"] is False


def test_nonadditive_duplicate_fixture_fails_closed_and_does_not_claim_singleton_effects():
    case = case_by_id("exact_duplicate_evidence")
    scorer = case.scorer()
    # Three non-empty masks are enough to preserve every direct observation,
    # but not enough for a separate full-rank fit and held-out qualification.
    screen = run_subset_screen(
        case.source_ids, scorer.measure, sampling_seed=7, passes_requested=3,
        mask_count=3, holdout_fraction=1 / 3, min_holdout_observations=1,
        max_holdout_mae_nats=100.0,
    )

    assert screen["qualification"]["state"] == "unqualified"
    assert screen["qualification"]["candidate_interpretation_available"] is False
    assert screen["candidate_source_ids"] == []
    assert screen["candidate_source_sets"] == []
    assert len(screen["masks"]) == len(scorer.calls) == 3
    assert any(mask["observed_delta_nats"] == 7.30 for mask in screen["masks"])
    # The high joint score is an observed mask record, not a coefficient or a
    # direct experiment collection manufactured by the screen.
    assert all(coefficient["provenance"] == "estimated" for coefficient in screen["coefficients"])
    assert "experiments" not in screen


def test_bad_held_out_error_fails_closed_but_preserves_measurements_and_diagnostics():
    case = case_by_id("three_way_coalition")
    scorer = case.scorer()
    screen = run_subset_screen(
        case.source_ids, scorer.measure, sampling_seed=3, passes_requested=7,
        mask_count=7, holdout_fraction=2 / 7, min_holdout_observations=2,
        max_holdout_mae_nats=0.20,
    )

    assert screen["qualification"]["state"] == "unqualified"
    assert screen["qualification"]["candidate_interpretation_available"] is False
    assert any("held-out MAE" in reason for reason in screen["qualification"]["reasons"])
    assert len(screen["masks"]) == 7
    assert screen["training_fit"]["metrics"]["observation_count"] > 0
    assert screen["holdout"]["metrics"]["observation_count"] == 2
    assert screen["candidate_source_ids"] == []


def test_score_calls_and_budget_are_reproducible_and_never_overrun():
    case = case_by_id("multi_hop_evidence")
    first_scorer = case.scorer()
    second_scorer = case.scorer()
    first = run_subset_screen(
        case.source_ids, first_scorer.measure, sampling_seed=99, passes_requested=5,
        mask_count=20, holdout_fraction=0.4, min_holdout_observations=1,
        max_holdout_mae_nats=100.0,
    )
    second = run_subset_screen(
        case.source_ids, second_scorer.measure, sampling_seed=99, passes_requested=5,
        mask_count=20, holdout_fraction=0.4, min_holdout_observations=1,
        max_holdout_mae_nats=100.0,
    )

    assert first["masks"] == second["masks"]
    assert first_scorer.calls == second_scorer.calls
    assert len(first_scorer.calls) == 5
    assert first["budget"]["passes_consumed"] == 5
    assert first["budget"]["passes_remaining"] == 0
    assert first["budget"]["exhausted"] is True


def test_no_remaining_budget_returns_explicit_unavailable_screen_without_score_call():
    case = case_by_id("one_necessary_source")
    scorer = case.scorer()
    screen = run_subset_screen(
        case.source_ids, scorer.measure, sampling_seed=1, passes_requested=2,
        initial_passes_consumed=2,
    )

    assert scorer.calls == []
    assert screen["status"] == "unavailable"
    assert screen["qualification"]["state"] == "unavailable"
    assert screen["qualification"]["candidate_interpretation_available"] is False
    assert screen["budget"]["passes_consumed"] == 2
    assert screen["budget"]["passes_remaining"] == 0


def test_adapter_must_return_a_direct_measurement_for_exact_requested_mask():
    def estimated_adapter(_ids):
        return {"removed_source_ids": ["alpha"], "delta_nats": 1.0, "provenance": "estimated"}

    with pytest.raises(ScreenMeasurementError, match="only direct measured"):
        run_subset_screen(("alpha",), estimated_adapter, sampling_seed=1, passes_requested=1)

    with pytest.raises(ContextDependenceScreenError, match="cannot exceed"):
        run_subset_screen(("alpha",), estimated_adapter, sampling_seed=1, passes_requested=0,
                          initial_passes_consumed=1)


def test_mask_sampler_rejects_requests_beyond_finite_small_context_population():
    with pytest.raises(ContextDependenceScreenError, match="exceeds"):
        sample_subset_masks(("only",), sampling_seed=1, mask_count=2)


def test_existing_direct_mask_is_reused_without_another_pass():
    def should_not_measure(_removed):
        raise AssertionError("existing direct mask must be reused")

    screen = run_subset_screen(
        ("only",), should_not_measure, sampling_seed=3,
        passes_requested=1, initial_passes_consumed=1, mask_count=1,
        existing_measurements=[{
            "experiment_id": "exp_only",
            "removed_source_ids": ["only"],
            "delta_nats": 0.5,
            "provenance": "measured",
        }],
        min_holdout_observations=1,
    )

    assert screen["masks"][0]["measurement_reused"] is True
    assert screen["budget"]["passes_consumed"] == 1
    assert screen["budget"]["measurements_reused"] == 1
