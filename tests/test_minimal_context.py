"""Model-free adversarial tests for bounded minimal Context Units search."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pytest

from clozn.runs.minimal_context import (
    MinimalContextError,
    run_minimal_context_for_study,
    run_minimal_context_search,
)


class MeasuredWorld:
    def __init__(self, source_ids: Iterable[str], preserving: Iterable[Iterable[str]]):
        self.source_ids = tuple(source_ids)
        self.preserving = {frozenset(item) for item in preserving}
        self.calls: list[frozenset[str]] = []

    def measure(self, removed: Iterable[str]) -> dict:
        removed = tuple(source_id for source_id in self.source_ids if source_id in set(removed))
        key = frozenset(removed)
        self.calls.append(key)
        return {
            "experiment_id": "exp_" + "_".join(removed),
            "removed_source_ids": list(removed),
            "delta_nats": 0.0 if key in self.preserving else 1.0,
            "provenance": "measured",
        }


def test_unique_minimum_is_directly_certified():
    world = MeasuredWorld(("A", "B", "C"), [{"A"}, {"A", "C"}])
    result = run_minimal_context_search(
        world.source_ids, world.measure, tolerance_nats=0.0,
        search_probe_budget=20, certification_probe_budget=20,
    )

    assert result["status"] == "found"
    assert result["candidate"]["retained_source_ids"] == ["B"]
    assert result["certificate"]["kind"] == "exact_minimum"
    assert result["coverage"]["smaller_remaining_count"] == 0


def test_multiple_minima_do_not_claim_uniqueness():
    world = MeasuredWorld(("A", "B", "C"), [{"B", "C"}, {"A", "C"}])
    result = run_minimal_context_search(
        world.source_ids, world.measure, tolerance_nats=0.0,
        search_probe_budget=20, certification_probe_budget=20,
        candidate_retained_source_sets=[["A"], ["B"]],
    )

    assert result["certificate"]["kind"] == "exact_minimum"
    assert result["candidate"]["retained_source_ids"] == ["A"]
    assert "unique" not in result["certificate"]


def test_greedy_local_candidate_can_be_improved_by_a_direct_nomination():
    source_ids = ("A", "B", "C", "D", "E")
    world = MeasuredWorld(source_ids, [{"D"}, {"D", "E"}, {"A", "B", "C"}])
    greedy = run_minimal_context_search(
        source_ids, world.measure, tolerance_nats=0.0,
        search_probe_budget=30, certification_probe_budget=0,
    )
    assert greedy["candidate"]["retained_source_ids"] == ["A", "B", "C"]
    assert greedy["certificate"]["kind"] == "inclusion_minimum"

    improved = run_minimal_context_search(
        source_ids, world.measure, tolerance_nats=0.0,
        search_probe_budget=30, certification_probe_budget=30,
        candidate_retained_source_sets=[["D", "E"]],
    )
    assert improved["candidate"]["retained_source_ids"] == ["D", "E"]
    assert improved["certificate"]["kind"] == "exact_minimum"


def test_joint_failure_is_measured_and_never_inferred_from_singletons():
    world = MeasuredWorld(("A", "B", "C"), [{"A"}, {"B"}])
    result = run_minimal_context_search(
        world.source_ids, world.measure, tolerance_nats=0.0,
        search_probe_budget=20, certification_probe_budget=0,
    )

    assert frozenset({"A", "B"}) in world.calls
    assert result["certificate"]["kind"] == "inclusion_minimum"


def test_jointly_removable_sources_require_a_direct_coalition_experiment():
    world = MeasuredWorld(("A", "B", "C"), [{"A", "B"}])
    result = run_minimal_context_search(
        world.source_ids, world.measure, tolerance_nats=0.0,
        search_probe_budget=20, certification_probe_budget=0,
        candidate_retained_source_sets=[["C"]],
    )

    assert frozenset({"A", "B"}) in world.calls
    assert result["candidate"]["retained_source_ids"] == ["C"]
    assert result["certificate"]["kind"] == "best_verified"


def test_certification_budget_exhaustion_prevents_minimality_claim():
    world = MeasuredWorld(("A", "B", "C"), [{"A", "C"}])
    result = run_minimal_context_search(
        world.source_ids, world.measure, tolerance_nats=0.0,
        search_probe_budget=20, certification_probe_budget=0,
        candidate_retained_source_sets=[["B"]],
    )

    assert result["certificate"]["kind"] == "best_verified"
    assert result["coverage"]["smaller_remaining_count"] == 1


def test_certification_can_find_a_smaller_pair_without_testing_same_size_alternatives():
    source_ids = ("A", "B", "C", "D", "E")
    # Candidate {A,B,C,D} is preserved by deleting E.  Its local children
    # fail, while {A,B} is a preserving non-local smaller subset.
    world = MeasuredWorld(source_ids, [{"E"}, {"C", "D", "E"}])
    result = run_minimal_context_search(
        source_ids, world.measure, tolerance_nats=0.0,
        search_probe_budget=30, certification_probe_budget=30,
        candidate_retained_source_sets=[["A", "B", "C", "D"]],
    )

    assert result["candidate"]["retained_source_ids"] == ["A", "B"]
    assert result["certificate"]["kind"] == "exact_minimum"
    assert frozenset({"C", "D", "E"}) in world.calls
    lower = {row["retained_source_count"]: row for row in result["coverage"]["lower_cardinalities"]}
    assert lower[0]["complete"] is True
    assert lower[1]["complete"] is True


def test_existing_direct_evidence_is_reused_with_zero_new_budget():
    world = MeasuredWorld(("A", "B"), [{"A", "B"}])
    existing = [world.measure(("A", "B"))]
    world.calls.clear()
    result = run_minimal_context_search(
        world.source_ids, world.measure, tolerance_nats=0.0,
        search_probe_budget=0, certification_probe_budget=0,
        existing_experiments=existing,
    )

    assert world.calls == []
    assert result["candidate"]["retained_source_ids"] == []
    assert result["budget"]["total_new_probes"] == 0
    assert result["budget"]["reused_experiments"] == 1


def test_same_seed_and_world_are_deterministic():
    def run_once():
        world = MeasuredWorld(("A", "B", "C", "D"), [{"A"}, {"A", "D"}])
        result = run_minimal_context_search(
            world.source_ids, world.measure, tolerance_nats=0.0,
            search_probe_budget=40, certification_probe_budget=20, search_seed=17,
        )
        return result, list(world.calls)

    first, first_calls = run_once()
    second, second_calls = run_once()
    assert first == second
    assert first_calls == second_calls
    assert first["search"]["strategy"] == "forward_reverse_intersection.v1"
    assert first["search"]["greedy_orders"] == ["source_order", "reverse_source_order"]


def test_large_lower_cardinality_layers_are_consumed_in_budget_bounded_chunks():
    source_ids = tuple(f"s{index:03d}" for index in range(100))
    candidate = source_ids[:3]
    existing = {
        "experiment_id": "exp_candidate",
        "removed_source_ids": [source_id for source_id in source_ids if source_id not in candidate],
        "delta_nats": 0.0,
        "provenance": "measured",
    }
    batches: list[tuple[tuple[str, ...], ...]] = []

    def measure_many(removed_sets):
        batch = tuple(tuple(removed) for removed in removed_sets)
        assert len(batch) <= 17
        batches.append(batch)
        return [
            {
                "experiment_id": "exp_" + str(index),
                "removed_source_ids": list(removed),
                "delta_nats": 0.0 if tuple(
                    source_id for source_id in source_ids if source_id not in set(removed)
                ) == candidate else 1.0,
                "provenance": "measured",
            }
            for index, removed in enumerate(batch)
        ]

    result = run_minimal_context_search(
        source_ids,
        None,
        tolerance_nats=0.0,
        search_probe_budget=0,
        certification_probe_budget=17,
        existing_experiments=[existing],
        measure_removed_many=measure_many,
    )

    assert result["certificate"]["kind"] == "inclusion_minimum"
    assert result["budget"]["certification_new_probes"] <= 17
    assert result["coverage"]["smaller_remaining_count"] > 0
    assert batches and max(len(batch) for batch in batches) <= 17


def test_certification_progress_reports_actual_cardinality():
    phases: list[str] = []
    world = MeasuredWorld(("a", "b", "c", "d", "e"), [{"e"}])
    result = run_minimal_context_search(
        world.source_ids,
        world.measure,
        tolerance_nats=0.0,
        search_probe_budget=30,
        certification_probe_budget=30,
        candidate_retained_source_sets=[["a", "b", "c", "d"]],
        phase_callback=lambda phase, _completed, _total: phases.append(phase),
    )

    assert result["certificate"]["kind"] == "exact_minimum"
    assert "certifying_cardinality_3" in phases


def test_conflicting_direct_evidence_fails_closed():
    world = MeasuredWorld(("A", "B"), [{"A"}])
    first = world.measure(("B",))
    conflict = dict(first, delta_nats=0.0)
    with pytest.raises(MinimalContextError, match="conflicting direct evidence"):
        run_minimal_context_search(
            world.source_ids, world.measure, tolerance_nats=0.0,
            search_probe_budget=0, certification_probe_budget=0,
            existing_experiments=[first, conflict],
        )


def test_unknown_nomination_is_rejected_before_any_direct_measurement():
    world = MeasuredWorld(("A", "B"), [{"A"}])
    with pytest.raises(MinimalContextError, match="outside the source universe"):
        run_minimal_context_search(
            world.source_ids, world.measure, tolerance_nats=0.0,
            search_probe_budget=0, certification_probe_budget=0,
            candidate_retained_source_sets=[["protected"]],
        )
    assert world.calls == []


def _manifest(run_id: str) -> dict:
    units = [
        {
            "source_id": "seg_aaaaaaaaaaaaaaaa",
            "message_index": 0,
            "role": "user",
            "unicode_range": [0, 1],
            "byte_range": [0, 1],
            "source_kind": "whole_message",
            "derivation": "message_root",
        },
        {
            "source_id": "seg_bbbbbbbbbbbbbbbb",
            "message_index": 1,
            "role": "assistant",
            "unicode_range": [0, 1],
            "byte_range": [0, 1],
            "source_kind": "whole_message",
            "derivation": "message_root",
        },
    ]
    return {
        "schema_version": "clozn.context-units.v1",
        "run_id": run_id,
        "basis": "messages",
        "protected_message_indices": [1],
        "units": units,
        "default_source_ids": [unit["source_id"] for unit in units],
    }


@dataclass
class FakeStudy:
    source_ids: tuple[str, ...]
    preserving_removed: set[frozenset[str]]
    experiments: list[dict] = field(default_factory=list)
    documents: int = 0

    def document(self) -> dict:
        self.documents += 1
        return {
            "study_id": "cds_0123456789abcdef01234567",
            "baseline": {"scored_once": True},
            "experiments": list(self.experiments),
            "robustness_controls": [],
            "budget": {"passes_consumed": 1 + len(self.experiments)},
        }

    def measure_removal_effect(self, removed: Iterable[str]) -> dict:
        removed = tuple(source_id for source_id in self.source_ids if source_id in set(removed))
        experiment = {
            "experiment_id": "study_" + "_".join(removed),
            "removed_source_ids": list(removed),
            "delta_nats": 0.0 if frozenset(removed) in self.preserving_removed else 1.0,
            "provenance": "measured",
        }
        self.experiments.append(experiment)
        return experiment


def test_study_wrapper_binds_manifest_and_accounts_for_new_experiments():
    run = {"id": "run-1"}
    manifest = _manifest("run-1")
    study = FakeStudy(tuple(manifest["default_source_ids"]), {frozenset({manifest["default_source_ids"][0]}), frozenset(manifest["default_source_ids"])})
    result = run_minimal_context_for_study(
        run, manifest, study, tolerance_nats=0.0,
        search_probe_budget=10, certification_probe_budget=0,
    )

    assert result["status"] == "found"
    assert result["context_dependence_study_id"].startswith("cds_")
    assert result["budget"]["total_new_probes"] == 3
    assert result["budget"]["baseline_passes"] == 1
    assert result["budget"]["baseline_charged_as_deletion_probe"] is False


def test_study_wrapper_does_not_search_outside_default_units():
    run = {"id": "run-2"}
    manifest = _manifest("run-2")
    study = FakeStudy(tuple(manifest["default_source_ids"]) + ("protected",), set())
    with pytest.raises(MinimalContextError, match="outside the source universe"):
        run_minimal_context_for_study(
            run, manifest, study, tolerance_nats=0.0,
            search_probe_budget=0, certification_probe_budget=0,
            candidate_retained_source_sets=[["protected"]],
        )
