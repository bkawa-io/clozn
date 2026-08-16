"""Focused Batch 1 tests for the new experimental kernel.

These tests keep the legacy experiment dispatcher out of the execution path.
The substrate fakes exercise only the trusted scalar probe and ordinary replay
seams.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn.experiments.evaluators import ExactReferenceMatch
from clozn.experiments.execution import (
    DeleteSourceExactReferenceAdapter,
    ExecutionStateStaleError,
    resolve_delete_source,
)
from clozn.experiments.interventions import DeleteSource
from clozn.experiments.kernel import Experiment
from clozn.experiments.materialize import materialize_arm
from clozn.experiments.runner import run_experiment
from clozn.experiments.selections import ContextSelection, SelectionError
from clozn.experiments.state import ExecutionState
import clozn.runs.store as runlog
from clozn.recipes.removability import can_remove, removability_message
from clozn.runs.context_receipt import build_context_receipt
from clozn.runs.answer_preservation import ExactAnswerPreservationStudy


CONTRACT = {
    "decode_mode": "greedy",
    "sampling": None,
    "max_new": 4,
    "stop": [],
    "expected_termination": {"reason": "eos", "reason_raw": "eos"},
}


def _identity():
    return {
        "model_sha256": "a" * 64,
        "template_fingerprint": "b" * 16,
        "engine_build": "test-engine",
        "context_size": 4096,
        "backend": "cpu",
        "white_box_flags": {"sae": False, "jlens": False, "attn_knockout": False},
    }


def _run():
    messages = [
        {"role": "system", "content": "stable context"},
        {"role": "user", "content": "removable source"},
        {"role": "user", "content": "current question"},
    ]
    run = {
        "id": "run_experimental_kernel",
        "model": "fixture-model",
        "substrate": "fixture",
        "messages": deepcopy(messages),
        "assembled_messages": deepcopy(messages),
        "final_prompt": "exact parent prompt",
        "response": "same answer",
        "generation_contract": deepcopy(CONTRACT),
        "identity": _identity(),
        "meta": {"n_ctx": 4096, "device": "cpu"},
        "behavior": {"active_dials": {"careful": 0.5}},
        "trace": {
            "tokens": ["same", " answer"],
            "token_ids": [10, 11],
            "steps": [
                {"token_id": 10, "piece": "same"},
                {"token_id": 11, "piece": " answer"},
            ],
        },
    }
    run["context_receipt"] = build_context_receipt(
        messages=messages,
        assembled_messages=messages,
        final_prompt=run["final_prompt"],
        run_id=run["id"],
        privacy="full",
    )
    source_ids = [segment["segment_id"] for segment in run["context_receipt"]["assembled"]]
    return run, source_ids


class ProbeSubstrate:
    def __init__(self):
        self.calls = []

    def identity_meta(self):
        return _identity()

    def run_meta(self):
        return {"n_ctx": 4096, "device": "cpu"}

    def probe_reference_match(self, messages, reference_token_ids, *, generation_contract,
                              explicit_conditions):
        self.calls.append((deepcopy(messages), list(reference_token_ids), deepcopy(generation_contract)))
        deleted = not any(message.get("content") == "removable source" for message in messages)
        if deleted:
            return {
                "status": "diverged",
                "matched_token_count": 1,
                "first_divergence_index": 1,
                "divergence_kind": "token_mismatch",
                "termination_match": True,
                "generated_token_ids": [10, 99],
                "termination": {"kind": "eos"},
            }
        return {
            "status": "matched",
            "matched_token_count": 2,
            "first_divergence_index": None,
            "divergence_kind": None,
            "termination_match": True,
            "generated_token_ids": [10, 11],
            "termination": {"kind": "eos"},
        }


class GenerationSubstrate(ProbeSubstrate):
    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        if mem_out is not None:
            mem_out.update(assembled_messages=deepcopy(messages), final_prompt="child prompt")
        if trace_out is not None:
            trace_out.extend([
                {"id": 20, "piece": "changed", "conf": 0.8, "alts": []},
                {"id": 21, "piece": " answer", "conf": 0.7, "alts": []},
            ])
        return "changed answer"


def test_execution_state_fingerprint_and_serialization_are_deterministic():
    run, _source_ids = _run()
    first = ExecutionState.from_run(run)
    second = ExecutionState.from_run(deepcopy(run))
    assert first.execution_fingerprint == second.execution_fingerprint
    assert first.to_json() == second.to_json()
    with pytest.raises(AttributeError):
        first.run_id = "different"


def test_context_selection_validates_canonical_ids_and_sorts_without_resolution():
    selection = ContextSelection(["src_b", "seg_a"])
    assert selection.source_ids == ("seg_a", "src_b")
    assert selection.to_dict() == {
        "kind": "context_selection", "source_ids": ["seg_a", "src_b"],
    }
    with pytest.raises(SelectionError, match="duplicates"):
        ContextSelection(["seg_a", "seg_a"])
    with pytest.raises(SelectionError, match="empty"):
        ContextSelection([])
    with pytest.raises(SelectionError, match="canonical"):
        ContextSelection(["document-1"])


def test_delete_and_evaluator_serialize_without_legacy_envelopes():
    intervention = DeleteSource(ContextSelection(["seg_a"]))
    assert intervention.to_dict() == {
        "kind": "delete_source",
        "target": {"kind": "context_selection", "source_ids": ["seg_a"]},
    }
    assert ExactReferenceMatch().to_dict() == {
        "kind": "exact_reference_match", "reference": "recorded_output",
    }


def test_experiment_arm_ids_are_stable_and_order_is_preserved():
    run, source_ids = _run()
    state = ExecutionState.from_run(run)
    arms = [
        DeleteSource(ContextSelection([source_ids[1]])),
        DeleteSource(ContextSelection([source_ids[0]])),
    ]
    first = Experiment(base=state, evaluator=ExactReferenceMatch(), arms=arms)
    second = Experiment(base=ExecutionState.from_run(deepcopy(run)), evaluator=ExactReferenceMatch(), arms=arms)
    assert [arm.intervention.source_ids for arm in first.arms] == [
        (source_ids[1],), (source_ids[0],),
    ]
    assert [arm.arm_id for arm in first.arms] == [arm.arm_id for arm in second.arms]
    assert first.experiment_id == second.experiment_id


def test_runner_executes_control_then_ordered_arms_without_run_persistence():
    run, source_ids = _run()
    state = ExecutionState.from_run(run)
    experiment = Experiment(
        base=state,
        evaluator=ExactReferenceMatch(),
        arms=[
            DeleteSource(ContextSelection([source_ids[1]])),
            DeleteSource(ContextSelection([source_ids[0]])),
        ],
    )
    substrate = ProbeSubstrate()
    result = run_experiment(
        experiment,
        DeleteSourceExactReferenceAdapter(substrate, run=run),
    )
    assert result.control.status == "exact_preserved"
    assert [item.arm_id for item in result.arms] == [arm.arm_id for arm in experiment.arms]
    assert [item.status for item in result.arms] == ["diverged", "exact_preserved"]
    assert result.to_json() == result.to_json()
    assert result.to_dict()["execution_provenance"]["arms_ephemeral"] is True


def test_control_failure_blocks_arms_honestly():
    run, source_ids = _run()
    state = ExecutionState.from_run(run)

    class Unavailable(ProbeSubstrate):
        def probe_reference_match(self, *args, **kwargs):
            return {"status": "unavailable", "reason": "worker_missing"}

    experiment = Experiment(
        base=state, evaluator=ExactReferenceMatch(),
        arms=[DeleteSource(ContextSelection([source_ids[1]]))],
    )
    substrate = Unavailable()
    result = run_experiment(experiment, DeleteSourceExactReferenceAdapter(substrate, run=run))
    assert result.state == "blocked"
    assert result.arms[0].status == "unavailable"
    assert len(substrate.calls) == 0


def test_strict_resolver_deletes_whole_message_and_protects_current_request():
    run, source_ids = _run()
    resolved = resolve_delete_source(
        run, DeleteSource(ContextSelection([source_ids[1]])),
    )
    assert [item["content"] for item in resolved["messages"]] == [
        "stable context", "current question",
    ]
    with pytest.raises(Exception, match="protected"):
        resolve_delete_source(run, DeleteSource(ContextSelection([source_ids[2]])))


def test_strict_resolver_handles_exact_spans_and_multiple_canonical_sources():
    messages = [
        {"role": "system", "content": "stable context"},
        {
            "role": "user", "content": "keep REMOVE and DROP",
            "_clozn_sources": [
                {"source_id": "remove", "unicode_range": [5, 11], "provenance_kind": "retrieved_document"},
                {"source_id": "drop", "unicode_range": [16, 20], "provenance_kind": "retrieved_document"},
            ],
        },
        {"role": "user", "content": "current question"},
    ]
    clean_messages = [{key: value for key, value in message.items() if key != "_clozn_sources"}
                      for message in messages]
    run = {
        "id": "run_experimental_spans",
        "messages": clean_messages,
        "assembled_messages": deepcopy(clean_messages),
        "context_receipt": build_context_receipt(
            messages=messages,
            assembled_messages=clean_messages,
            final_prompt="prompt",
            run_id="run_experimental_spans",
            privacy="full",
        ),
    }
    span_ids = [item["source_id"] for item in run["context_receipt"]["delivered"][1]["sources"]]
    resolved = resolve_delete_source(
        run, DeleteSource(ContextSelection(list(reversed(span_ids))))
    )
    assert resolved["canonical_source_ids"] == sorted(span_ids)
    assert resolved["messages"][1]["content"] == "keep  and "


def test_stale_parent_cannot_materialize():
    run, source_ids = _run()
    state = ExecutionState.from_run(run)
    experiment = Experiment(
        base=state, evaluator=ExactReferenceMatch(),
        arms=[DeleteSource(ContextSelection([source_ids[1]]))],
    )
    substrate = ProbeSubstrate()
    result = run_experiment(experiment, DeleteSourceExactReferenceAdapter(substrate, run=run))
    stale = deepcopy(run)
    stale["messages"][0]["content"] = "changed context"
    with pytest.raises(Exception, match="fingerprint|changed|stale"):
        materialize_arm(
            stale, result, result.arms[0].arm_id,
            substrate=substrate,
            replay_fn=lambda *args, **kwargs: None,
        )


def test_materialize_arm_persists_one_generic_child_and_returns_canonical_diff(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    run, source_ids = _run()
    state = ExecutionState.from_run(run)
    experiment = Experiment(
        base=state, evaluator=ExactReferenceMatch(),
        arms=[DeleteSource(ContextSelection([source_ids[1]]))],
    )
    substrate = GenerationSubstrate()
    result = run_experiment(experiment, DeleteSourceExactReferenceAdapter(substrate, run=run))
    outcome = materialize_arm(run, result, result.arms[0].arm_id, substrate=substrate)
    assert outcome["state"] == "completed"
    assert len(runlog.list_runs(20)) == 1
    child = runlog.get_run(outcome["child_run_id"])
    assert child["parent_run_id"] == run["id"]
    assert child["changes_applied"]["experiment"]["experiment_id"] == result.experiment_id
    assert child["changes_applied"]["experiment"]["intervention"] == {
        "kind": "delete_source", "source_ids": [source_ids[1]],
    }
    assert "minimal_context" not in str(child["changes_applied"])
    assert outcome["comparison"]["first_divergence_view"]["state"] == "available"


def test_failed_materialization_does_not_create_a_child(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    run, source_ids = _run()
    state = ExecutionState.from_run(run)
    experiment = Experiment(
        base=state, evaluator=ExactReferenceMatch(),
        arms=[DeleteSource(ContextSelection([source_ids[1]]))],
    )
    substrate = GenerationSubstrate()
    result = run_experiment(experiment, DeleteSourceExactReferenceAdapter(substrate, run=run))

    outcome = materialize_arm(
        run, result, result.arms[0].arm_id, substrate=substrate,
        replay_fn=lambda *args, **kwargs: None,
    )
    assert outcome["state"] == "failed"
    assert "child_run_id" not in outcome
    assert runlog.list_runs(20) == []


def test_removability_recipe_is_thin_and_uses_preservation_language():
    run, source_ids = _run()
    result = can_remove(
        run,
        [source_ids[1]],
        execution_adapter=DeleteSourceExactReferenceAdapter(ProbeSubstrate(), run=run),
    )
    arm_id = result.arms[0].arm_id
    assert removability_message(result, arm_id).startswith(
        "Deleting this source caused divergence at recorded answer token"
    )
    assert "irrelevant" not in removability_message(result, arm_id)


def test_new_removability_path_matches_legacy_direct_probe_oracle():
    run, source_ids = _run()
    substrate = ProbeSubstrate()
    legacy = ExactAnswerPreservationStudy(
        run, substrate, source_ids=[source_ids[1]],
    )
    legacy_probe = legacy.probe_removed_sources([source_ids[1]])
    new_result = can_remove(
        run,
        [source_ids[1]],
        execution_adapter=DeleteSourceExactReferenceAdapter(substrate, run=run),
    )
    new_observation = new_result.arms[0]
    expected_status = {
        "matched": "exact_preserved",
        "diverged": "diverged",
    }[legacy_probe["result"]["status"]]
    assert new_observation.status == expected_status
    assert new_observation.first_divergence_index == legacy_probe["result"].get("first_divergence_index")
