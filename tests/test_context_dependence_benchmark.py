"""Regression coverage for the model-free Context Dependence benchmark fixtures."""
from __future__ import annotations

from itertools import combinations

import pytest

from tests.context_dependence_cases import (
    ALL_CONTEXT_DEPENDENCE_CASES,
    MATERIAL_EFFECT_FLOOR_NATS,
    UnknownSyntheticSourceIDError,
    case_by_id,
    evaluate_direct_measurements,
)


EXPECTED_CASE_IDS = {
    "irrelevant_filler",
    "one_necessary_source",
    "exact_duplicate_evidence",
    "paraphrased_substitutable_duplicate_evidence",
    "either_a_or_b_sufficiency",
    "a_and_b_complementarity",
    "three_way_coalition",
    "multi_hop_evidence",
    "parametric_knowledge_overlap",
    "coreference_broken_by_deletion",
    "position_sensitive_long_context",
    "repeated_user_and_rag_evidence",
    "later_answer_depends_on_recorded_prefix",
}


def test_required_worlds_have_complete_explicit_direct_effect_oracles():
    assert {case.case_id for case in ALL_CONTEXT_DEPENDENCE_CASES} == EXPECTED_CASE_IDS
    for case in ALL_CONTEXT_DEPENDENCE_CASES:
        expected_subsets = 2 ** len(case.sources)
        assert len(case.expected_direct_effects) == expected_subsets
        assert case.material_source_sets.isdisjoint(case.non_material_source_sets)
        assert case.material_source_sets | case.non_material_source_sets == set(case.expected_direct_effects)
        assert all(delta >= MATERIAL_EFFECT_FLOOR_NATS
                   for source_set, delta in case.expected_direct_effects.items()
                   if source_set in case.material_source_sets)
        assert all(delta < MATERIAL_EFFECT_FLOOR_NATS
                   for source_set, delta in case.expected_direct_effects.items()
                   if source_set in case.non_material_source_sets)


@pytest.mark.parametrize("case_id", sorted(EXPECTED_CASE_IDS))
def test_fake_scorer_is_deterministic_model_free_and_preserves_delta_invariant(case_id):
    case = case_by_id(case_id)
    scorer = case.scorer()
    measured = scorer.measure(case.source_ids)
    same_measurement = case.scorer().measure(reversed(case.source_ids))
    assert measured == same_measurement
    assert measured.provenance == "measured"
    assert measured.delta_nats == case.expected_effect(case.source_ids)
    assert measured.target_logp == measured.baseline_target_logp - measured.delta_nats
    assert sum(measured.per_target_token_delta_nats) == measured.delta_nats
    assert scorer.passes_consumed == 1
    assert not hasattr(scorer, "generate")


def test_duplicate_coalition_defeats_singleton_floor_pair_policy():
    """The core regression: pairs must not require above-floor singleton effects."""
    case = case_by_id("exact_duplicate_evidence")
    scorer = case.scorer()
    singleton_measurements = [scorer.measure((source_id,)) for source_id in case.source_ids]
    above_floor = [
        measurement.removed_source_ids[0]
        for measurement in singleton_measurements
        if measurement.delta_nats >= MATERIAL_EFFECT_FLOOR_NATS
    ]
    # This is the historically unsafe policy: no pair is considered because
    # neither singleton clears the floor.
    for pair in combinations(above_floor, 2):
        scorer.measure(pair)

    evaluation = evaluate_direct_measurements(
        case,
        scorer.calls,
        passes_requested=2,
        declared_irrelevant_source_ids=case.source_ids,
    )
    joint = frozenset(("duplicate-a", "duplicate-b"))
    assert case.expected_direct_effects[joint] == 7.30
    assert joint in evaluation.missed_material_source_sets
    assert evaluation.directly_measured_material_sets == frozenset()
    assert evaluation.low_singleton_source_ids == frozenset(case.source_ids)
    assert evaluation.incorrectly_treated_low_singletons_as_irrelevant
    assert evaluation.source_ids_falsely_declared_irrelevant == frozenset(case.source_ids)
    assert evaluation.passes_consumed == 2
    assert evaluation.passes_remaining == 0


def test_directly_measuring_duplicate_coalition_is_the_only_evidence_that_discovers_it():
    case = case_by_id("exact_duplicate_evidence")
    scorer = case.scorer()
    scorer.measure(("duplicate-a",))
    joint_measurement = scorer.measure(("duplicate-b", "duplicate-a"))
    evaluation = evaluate_direct_measurements(case, scorer.calls, passes_requested=2)
    joint = frozenset(("duplicate-a", "duplicate-b"))
    assert joint_measurement.removed_source_ids == ("duplicate-a", "duplicate-b")
    assert joint_measurement.delta_nats == 7.30
    assert evaluation.directly_measured_material_sets == frozenset((joint,))
    assert evaluation.missed_material_source_sets == frozenset()
    assert evaluation.all_material_sets_discovered
    assert not evaluation.incorrectly_treated_low_singletons_as_irrelevant


def test_budget_evaluator_reports_consumption_and_does_not_hide_overrun():
    case = case_by_id("one_necessary_source")
    scorer = case.scorer()
    scorer.measure(())
    scorer.measure(("necessary",))
    scorer.measure(("necessary", "filler"))
    evaluation = evaluate_direct_measurements(case, scorer.calls, passes_requested=2)
    assert evaluation.passes_consumed == 3
    assert evaluation.passes_remaining == 0
    assert evaluation.exceeded_budget
    assert frozenset(("necessary",)) in evaluation.directly_measured_material_sets


def test_unknown_source_ids_fail_closed_in_scoring_and_evaluation():
    case = case_by_id("one_necessary_source")
    with pytest.raises(UnknownSyntheticSourceIDError, match="unknown source IDs"):
        case.scorer().measure(("not-in-context-receipt",))
    with pytest.raises(UnknownSyntheticSourceIDError, match="unknown declared source IDs"):
        evaluate_direct_measurements(case, (), passes_requested=0,
                                     declared_irrelevant_source_ids=("not-in-context-receipt",))


def test_case_specific_oracles_cover_the_required_nontrivial_conditions():
    duplicate = case_by_id("exact_duplicate_evidence")
    assert duplicate.expected_effect(("duplicate-a",)) == 0.03
    assert duplicate.expected_effect(("duplicate-b",)) == 0.04
    assert duplicate.expected_effect(("duplicate-a", "duplicate-b")) == 7.30

    complementarity = case_by_id("a_and_b_complementarity")
    assert complementarity.expected_effect(("component-a",)) == 4.20
    assert complementarity.expected_effect(("component-b",)) == 4.00
    assert complementarity.expected_effect(("component-a", "component-b")) == 8.60

    coalition = case_by_id("three_way_coalition")
    assert coalition.expected_effect(("coalition-a", "coalition-b")) == 0.05
    assert coalition.expected_effect(("coalition-a", "coalition-b", "coalition-c")) == 9.20

    parametric = case_by_id("parametric_knowledge_overlap")
    assert not parametric.material_source_sets
    assert parametric.metadata["recorded_answer_correct"] is True

    coreference = case_by_id("coreference_broken_by_deletion")
    assert coreference.expected_effect(("antecedent",)) == 5.40
    assert coreference.expected_effect(("coreferent-fact",)) == 0.35

    long_context = case_by_id("position_sensitive_long_context")
    assert len(long_context.sources) == 10
    assert long_context.expected_effect(("late-evidence",)) > long_context.expected_effect(("early-evidence",))
    assert long_context.metadata["position_sensitive"] is True

    repeated = case_by_id("repeated_user_and_rag_evidence")
    assert [source.origin for source in repeated.sources] == ["user", "rag"]
    assert repeated.expected_effect(repeated.source_ids) == 7.30

    prefix = case_by_id("later_answer_depends_on_recorded_prefix")
    assert prefix.target.recorded_prefix_range == (0, 4)
    assert prefix.metadata["conditioned_prefix_dependency_nats"] == 8.10
    assert not prefix.material_source_sets
