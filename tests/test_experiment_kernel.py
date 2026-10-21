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
from clozn.runs.context_receipt import build_context_receipt
from clozn import schemas


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
    }


def _run():
    messages = [
        {"role": "user", "content": "stable context"},
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


def test_strict_resolver_clears_whole_message_text_including_current_request():
    run, source_ids = _run()
    resolved = resolve_delete_source(
        run, DeleteSource(ContextSelection([source_ids[1]])),
    )
    assert [item["content"] for item in resolved["messages"]] == [
        "stable context", "", "current question",
    ]
    current_removed = resolve_delete_source(
        run, DeleteSource(ContextSelection([source_ids[2]])),
    )
    assert [item["content"] for item in current_removed["messages"]] == [
        "stable context", "removable source", "",
    ]


def test_strict_resolver_handles_exact_spans_and_multiple_canonical_sources():
    content = "## Keep\n" + ("keep " * 12) + "\n\n## Drop\n" + ("drop " * 12)
    messages = [
        {"role": "system", "content": "stable context"},
        {"role": "user", "content": content},
        {"role": "user", "content": "current question"},
    ]
    clean_messages = deepcopy(messages)
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
    assert [message["content"] for message in resolved["messages"]] == [
        "stable context", "", "current question",
    ]


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
    assert outcome["comparison"]["preservation"]["exact"] is False
    assert outcome["comparison"]["parent_run_id"] == run["id"]
    assert outcome["comparison"]["child_run_id"] == outcome["child_run_id"]
    schemas.validate(outcome, "clozn.experiment-materialization.v1")


def test_materialize_arm_replays_with_the_recorded_generation_budget():
    run, source_ids = _run()
    state = ExecutionState.from_run(run)
    experiment = Experiment(
        base=state, evaluator=ExactReferenceMatch(),
        arms=[DeleteSource(ContextSelection([source_ids[1]]))],
    )
    substrate = GenerationSubstrate()
    result = run_experiment(experiment, DeleteSourceExactReferenceAdapter(substrate, run=run))
    calls = []

    def replay(parent, changes, active_substrate, **kwargs):
        calls.append((parent, changes, active_substrate, kwargs))
        return {"id": "run_child", "response": "changed answer"}

    outcome = materialize_arm(
        run, result, result.arms[0].arm_id,
        substrate=substrate, replay_fn=replay,
    )

    assert outcome["state"] == "completed"
    assert len(calls) == 1
    assert calls[0][3]["max_new"] == CONTRACT["max_new"]
    assert calls[0][3]["sampling_override"] is False


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
    schemas.validate(outcome, "clozn.experiment-materialization.v1")
