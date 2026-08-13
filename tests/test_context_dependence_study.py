"""Focused direct-hierarchy tests using the model-free benchmark worlds.

The bridge below intentionally presents every synthetic world through the
same recorded-run + ``score_tokens`` interface used by Task 1.  The study
therefore tests real Task 1 experiment records, rather than teaching the
orchestrator a second synthetic measurement API.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn import schemas
from clozn.runs.context_dependence import (
    ContextDependenceStudyError,
    DirectContextDependenceStudy,
    run_context_dependence_study,
)
from clozn.runs.context_receipt import build_context_receipt
from tests.context_dependence_cases import case_by_id


class StepClock:
    def __init__(self, *, start: float = 1.0, step: float = 0.01):
        self.value = start
        self.step = step

    def __call__(self) -> float:
        value = self.value
        self.value += self.step
        return value


class CaseScoreSub:
    """Teacher-forced bridge from a benchmark case to Task 1's score call."""

    def __init__(self, case):
        self.case = case
        self.calls: list[dict] = []

    def chat(self, *_args, **_kwargs):  # pragma: no cover - only a contract tripwire
        raise AssertionError("a direct Context Dependence study must not generate")

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        present = {
            message["fixture_source_id"] for message in messages
            if isinstance(message, dict) and isinstance(message.get("fixture_source_id"), str)
        }
        removed = tuple(source_id for source_id in self.case.source_ids if source_id not in present)
        delta = self.case.expected_effect(removed)
        self.calls.append({
            "removed_source_ids": removed,
            "continuation_ids": deepcopy(continuation_ids),
            "continuation": continuation,
        })
        # The sum of Task 1's per-token deltas will be exactly the fixture
        # effect.  A fixed three-token continuation keeps the bridge simple
        # while the case still supplies the source-set effect oracle.
        per_token = [delta / 3.0, delta / 3.0, delta - 2.0 * (delta / 3.0)]
        return [
            {"id": token_id, "piece": piece, "logprob": -1.0 - penalty}
            for token_id, piece, penalty in zip((101, 102, 103), ("The", " answer", "."), per_token)
        ]


def _run_for(case_id: str) -> tuple[dict, CaseScoreSub, tuple[str, ...]]:
    case = case_by_id(case_id)
    messages = [
        {
            "role": "user",
            "content": source.text,
            # Preserved through Task 1's deepcopy/filter operation and used
            # only by this test scorer to distinguish identical text sources.
            "fixture_source_id": source.source_id,
            "source_id": f"fixture-client-{source.source_id}",
        }
        for source in case.sources
    ]
    receipt = build_context_receipt(
        messages=messages,
        assembled_messages=messages,
        final_prompt=f"fixture prompt: {case.case_id}",
        run_id=f"fixture_{case.case_id}",
        identity={"template_fingerprint": "0123456789abcdef"},
        privacy="full",
    )
    source_ids = tuple(segment["segment_id"] for segment in receipt["assembled"])
    run = {
        "id": f"fixture_{case.case_id}",
        "model": "fixture-model",
        "substrate": "CaseScoreSub",
        "identity": {"model_sha256": "fixture-model", "template_fingerprint": "0123456789abcdef"},
        "messages": deepcopy(messages),
        "assembled_messages": deepcopy(messages),
        "context_receipt": receipt,
        "final_prompt": f"fixture prompt: {case.case_id}",
        "response": "The answer.",
        "trace": {"token_ids": [101, 102, 103]},
    }
    return run, CaseScoreSub(case), source_ids


def _nodes(document: dict, *, source_ids=None, kind=None):
    result = document["hierarchy"]["nodes"]
    if source_ids is not None:
        result = [node for node in result if node["source_ids"] == list(source_ids)]
    if kind is not None:
        result = [node for node in result if node["node_kind"] == kind]
    return result


def _experiment_by_id(document: dict) -> dict[str, dict]:
    return {experiment["experiment_id"]: experiment for experiment in document["experiments"]}


def test_discovers_a_necessary_source_with_a_direct_task1_experiment():
    run, sub, (necessary, filler) = _run_for("one_necessary_source")
    document = run_context_dependence_study(run, sub, passes_requested=4, clock=StepClock())

    schemas.validate(document)
    node = _nodes(document, source_ids=(necessary,), kind="source_unit")[0]
    experiments = _experiment_by_id(document)
    assert node["measurement_state"] == "measured"
    assert node["direct_effect"]["experiment_id"] in experiments
    assert node["direct_effect"] == {
        "experiment_id": node["direct_effect"]["experiment_id"],
        "delta_nats": pytest.approx(6.40),
        "provenance": "measured",
    }
    assert _nodes(document, source_ids=(filler,), kind="source_unit")[0]["direct_effect"]["delta_nats"] == 0.0
    assert document["budget"] == {
        "passes_requested": 4, "passes_consumed": 4, "passes_remaining": 0, "state": "exhausted",
    }
    assert all(call["continuation_ids"] == [101, 102, 103] for call in sub.calls)
    assert all(call["continuation"] is None for call in sub.calls)


def test_zero_effect_stays_measured_without_an_irrelevance_conclusion():
    run, sub, source_ids = _run_for("irrelevant_filler")
    document = run_context_dependence_study(run, sub, passes_requested=4, clock=StepClock())

    assert len(document["experiments"]) == 3
    assert all(experiment["delta_nats"] == 0.0 for experiment in document["experiments"])
    assert all(node["measurement_state"] == "measured" for node in document["hierarchy"]["nodes"])
    assert all("irrelevant" not in node for node in document["hierarchy"]["nodes"])
    assert "conclusions" not in document["hierarchy"]
    assert len(sub.calls) == 4
    assert _nodes(document, source_ids=source_ids)[0]["direct_effect"]["provenance"] == "measured"


def test_duplicate_case_retains_the_high_effect_parent_when_children_are_weak():
    run, _sub, source_ids = _run_for("exact_duplicate_evidence")
    document = run_context_dependence_study(run, _sub, passes_requested=4, clock=StepClock())

    root = _nodes(document, source_ids=source_ids, kind="requested_root_set")[0]
    children = _nodes(document, kind="source_unit")
    assert root["measurement_state"] == "measured"
    assert root["direct_effect"]["delta_nats"] == pytest.approx(7.30)
    assert [child["direct_effect"]["delta_nats"] for child in children] == pytest.approx([0.03, 0.04])
    # The discrepancy is declared derived search metadata, not a convenient
    # new effect measurement.  The root record remains backed by its own arm.
    metadata = document["hierarchy"]["nonadditivity"]
    assert metadata == [{
        "parent_node_id": root["node_id"],
        "parent_experiment_id": root["direct_effect"]["experiment_id"],
        "child_experiment_ids": [child["direct_effect"]["experiment_id"] for child in children],
        "derived_value_nats": pytest.approx(7.23),
        "provenance": "derived_search_metadata",
        "not_a_measured_effect": True,
    }]
    assert metadata[0]["parent_experiment_id"] != metadata[0]["child_experiment_ids"][0]
    assert root["direct_effect"]["experiment_id"] in _experiment_by_id(document)


def test_reuses_identical_source_set_instead_of_recomputing_it():
    run, sub, source_ids = _run_for("one_necessary_source")
    document = run_context_dependence_study(
        run,
        sub,
        # This explicit grouping is deliberately the root set itself: it is a
        # regression test for task orchestration caching, not a claim that the
        # receipt supplied a separate semantic unit.
        source_groups=[{
            "group_id": "same_as_requested_root", "source_ids": list(source_ids),
            "structure_origin": "caller_supplied",
        }],
        passes_requested=2,
        quick=True,
        clock=StepClock(),
    )

    root = _nodes(document, source_ids=source_ids, kind="requested_root_set")[0]
    group = _nodes(document, source_ids=source_ids, kind="structural_source_group")[0]
    assert len(document["experiments"]) == 1
    assert len(sub.calls) == 2  # baseline + one deletion arm
    assert root["measurement_state"] == "measured"
    assert group["measurement_state"] == "measured_reused"
    assert group["direct_effect"]["experiment_id"] == root["direct_effect"]["experiment_id"]
    assert document["budget"]["passes_consumed"] == 2


def test_budget_exhaustion_is_explicit_and_never_spends_an_extra_pass():
    run, sub, source_ids = _run_for("one_necessary_source")
    document = run_context_dependence_study(run, sub, passes_requested=2, clock=StepClock())

    root = _nodes(document, source_ids=source_ids, kind="requested_root_set")[0]
    leaves = _nodes(document, kind="source_unit")
    assert root["measurement_state"] == "measured"
    assert all(node["measurement_state"] == "unmeasured_budget_exhausted" for node in leaves)
    assert document["hierarchy"]["unmeasured_node_ids"] == [node["node_id"] for node in leaves]
    assert document["budget"] == {
        "passes_requested": 2, "passes_consumed": 2, "passes_remaining": 0, "state": "exhausted",
    }
    assert len(sub.calls) == 2


def test_quick_study_measures_root_and_top_level_structural_groups_only():
    run, sub, source_ids = _run_for("multi_hop_evidence")
    document = run_context_dependence_study(
        run, sub, passes_requested=3, quick=True,
        source_groups=[{
            "group_id": "first_two", "source_ids": list(source_ids[:2]),
            "structure_origin": "search_generated",
        }],
        clock=StepClock(),
    )

    assert [node["node_kind"] for node in document["hierarchy"]["nodes"]] == [
        "requested_root_set", "structural_source_group",
    ]
    assert all(node["measurement_state"] == "measured" for node in document["hierarchy"]["nodes"])
    assert document["hierarchy"]["nodes"][1]["structure_origin"] == "search_generated"
    assert len(sub.calls) == 3


def test_ordering_and_study_identity_are_deterministic_across_group_input_order():
    run_a, _sub_a, source_ids = _run_for("multi_hop_evidence")
    run_b, _sub_b, _ = _run_for("multi_hop_evidence")
    groups_a = {
        "zeta": {"source_ids": [source_ids[2]], "structure_origin": "caller_supplied"},
        "alpha": {"source_ids": list(source_ids[:2]), "structure_origin": "caller_supplied"},
    }
    groups_b = {
        "alpha": {"source_ids": list(reversed(source_ids[:2])), "structure_origin": "caller_supplied"},
        "zeta": {"source_ids": [source_ids[2]], "structure_origin": "caller_supplied"},
    }
    first = run_context_dependence_study(
        run_a, _sub_a, source_groups=groups_a, passes_requested=6, clock=StepClock(start=1),
    )
    second = run_context_dependence_study(
        run_b, _sub_b, source_groups=groups_b, passes_requested=6, clock=StepClock(start=200, step=0.9),
    )

    assert first["study_id"] == second["study_id"]
    assert [node["node_id"] for node in first["hierarchy"]["nodes"]] == [
        node["node_id"] for node in second["hierarchy"]["nodes"]
    ]
    assert [node["source_ids"] for node in first["hierarchy"]["nodes"]] == [
        node["source_ids"] for node in second["hierarchy"]["nodes"]
    ]
    assert [experiment["experiment_id"] for experiment in first["experiments"]] == [
        experiment["experiment_id"] for experiment in second["experiments"]
    ]


def test_every_hierarchy_effect_number_has_a_real_direct_experiment_reference():
    run, _sub, _source_ids = _run_for("a_and_b_complementarity")
    document = run_context_dependence_study(run, _sub, passes_requested=4, clock=StepClock())
    experiments = _experiment_by_id(document)

    for node in document["hierarchy"]["nodes"]:
        direct = node.get("direct_effect")
        if direct is None:
            continue
        experiment = experiments[direct["experiment_id"]]
        assert direct["provenance"] == experiment["provenance"] == "measured"
        assert direct["delta_nats"] == experiment["delta_nats"]
    for item in document["hierarchy"]["nonadditivity"]:
        assert item["provenance"] == "derived_search_metadata"
        assert item["not_a_measured_effect"] is True
        assert item["parent_experiment_id"] in experiments
        assert set(item["child_experiment_ids"]).issubset(experiments)


def test_invalid_budget_and_overlapping_group_structure_fail_closed():
    run, sub, source_ids = _run_for("one_necessary_source")
    with pytest.raises(ContextDependenceStudyError, match="passes_requested"):
        DirectContextDependenceStudy(run, sub, passes_requested=0)
    with pytest.raises(ContextDependenceStudyError, match="sibling groups overlap"):
        DirectContextDependenceStudy(
            run, sub,
            source_groups=[
                {"group_id": "one", "source_ids": list(source_ids)},
                {"group_id": "two", "source_ids": [source_ids[0]]},
            ],
        )
