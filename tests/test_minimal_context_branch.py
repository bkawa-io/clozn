from __future__ import annotations

from copy import deepcopy

import pytest

import clozn.runs.store as runlog
from clozn.runs.context_receipt import build_context_receipt
from clozn.runs.context_units import build_context_unit_manifest
from clozn.runs.minimal_context import run_minimal_context_search, EXACT_PRESERVATION_KIND, PRESERVATION_TARGET
from clozn.runs.minimal_context_branch import (
    MinimalContextBranchError,
    execute_minimal_context_branch,
    plan_minimal_context_branch,
)


class GenerationSub:
    def __init__(self):
        self.steer = Steer({"careful": 0.5})
        self.calls = 0

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self.calls += 1
        if mem_out is not None:
            mem_out.update(assembled_messages=deepcopy(messages), final_prompt="child prompt")
        if trace_out is not None:
            trace_out.extend([
                {"id": 31, "piece": "changed", "conf": 0.8, "alts": []},
                {"id": 32, "piece": " answer", "conf": 0.7, "alts": []},
            ])
        return "changed answer"


class Steer:
    def __init__(self, strength):
        self.strength = dict(strength)

    def clear(self):
        self.strength = {}

    def set(self, name, value):
        self.strength[name] = float(value)

    def active(self):
        return dict(self.strength)


@pytest.fixture
def parent_and_result(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    messages = [
        {"role": "system", "content": "source A"},
        {"role": "system", "content": "source B"},
        {"role": "user", "content": "current question"},
    ]
    run = {
        "id": "run_minimal_branch",
        "model": "fixture-model",
        "substrate": "fixture",
        "messages": deepcopy(messages),
        "assembled_messages": deepcopy(messages),
        "final_prompt": "parent prompt",
        "response": "same answer",
        "behavior": {"active_dials": {"careful": 0.5}},
        "trace": {"token_ids": [11, 12], "tokens": ["same", " answer"]},
    }
    run["context_receipt"] = build_context_receipt(
        messages=messages, assembled_messages=messages, final_prompt="parent prompt",
        run_id=run["id"], privacy="full",
    )
    run["context_units"] = build_context_unit_manifest(run)
    source_ids = list(run["context_units"]["default_source_ids"])

    def measure(removed):
        removed = tuple(removed)
        preserving = len(set(source_ids).difference(removed)) >= 1
        return {
            "experiment_id": "exp_" + "_".join(removed),
            "removed_source_ids": list(removed),
            "delta_nats": 0.0 if preserving else 1.0,
            "provenance": "measured",
        }

    result = run_minimal_context_search(
        source_ids, measure, tolerance_nats=0.1, search_probe_budget=20,
        certification_probe_budget=20, run_id=run["id"], context_unit_manifest=run["context_units"],
    )
    run["minimal_context_results"] = {result["result_id"]: result}
    return run, result, source_ids


def test_remove_and_add_back_plans_reuse_strict_receipt_and_bind_result(parent_and_result):
    parent, result, source_ids = parent_and_result
    retained = result["candidate"]["retained_source_ids"]
    omitted = [source_id for source_id in source_ids if source_id not in retained]

    remove_plan = plan_minimal_context_branch(
        parent, result, action="remove_and_branch", source_ids=[retained[0]],
    )
    assert remove_plan["intervention"]["removed_source_ids"] == source_ids
    assert remove_plan["intervention"]["action"] == "remove_and_branch"
    assert remove_plan["execution"]["messages_override"] == [parent["messages"][2]]

    add_plan = plan_minimal_context_branch(
        parent, result, action="add_back_and_branch", source_ids=omitted,
    )
    assert add_plan["intervention"]["target_retained_source_ids"] == source_ids
    assert add_plan["execution"]["messages_override"] == parent["messages"]


def test_branch_executes_one_normal_child_with_exact_intervention_and_canonical_diff(parent_and_result):
    parent, result, source_ids = parent_and_result
    retained = result["candidate"]["retained_source_ids"]
    plan = plan_minimal_context_branch(parent, result, action="remove_and_branch", source_ids=[retained[0]])
    sub = GenerationSub()

    outcome = execute_minimal_context_branch(
        parent, result, sub, action="remove_and_branch", source_ids=[retained[0]], plan=plan,
    )
    assert outcome["state"] == "completed"
    assert sub.calls == 1
    child = runlog.get_run(outcome["child_run_id"])
    assert child["parent_run_id"] == parent["id"]
    assert child["changes_applied"]["minimal_context_branch"]["result_id"] == result["result_id"]
    assert child["changes_applied"]["minimal_context_branch"]["removed_source_ids"] == source_ids
    assert "proof" not in child["changes_applied"]
    assert outcome["comparison"]["first_divergence"]["index"] == 0
    assert outcome["compare_path"].endswith(f"/{outcome['child_run_id']}")


def test_branch_from_exact_recorded_output_result_uses_the_same_source_binding(parent_and_result):
    parent, _likelihood_result, source_ids = parent_and_result

    def exact_measure(removed):
        removed = tuple(removed)
        matched = len(set(source_ids).difference(removed)) >= 1
        return {
            "probe_id": "probe_" + "_".join(removed),
            "removed_source_ids": list(removed),
            "result": {"status": "matched" if matched else "diverged"},
            "provenance": "direct_generation_probe",
        }

    exact = run_minimal_context_search(
        source_ids, exact_measure, tolerance_nats=0.0, search_probe_budget=20,
        certification_probe_budget=20, run_id=parent["id"], context_unit_manifest=parent["context_units"],
        preservation={"kind": EXACT_PRESERVATION_KIND, "target": PRESERVATION_TARGET},
    )
    parent["minimal_context_results"][exact["result_id"]] = exact
    retained = exact["candidate"]["retained_source_ids"]
    plan = plan_minimal_context_branch(parent, exact, action="remove_and_branch", source_ids=retained)
    assert plan["result_preservation"] == EXACT_PRESERVATION_KIND
    assert plan["result_id"] == exact["result_id"]


def test_drift_before_branch_refuses_without_calling_generation(parent_and_result):
    parent, result, source_ids = parent_and_result
    retained = result["candidate"]["retained_source_ids"]
    plan = plan_minimal_context_branch(parent, result, action="remove_and_branch", source_ids=[retained[0]])
    drifted = deepcopy(parent)
    drifted["messages"][0]["content"] = "changed source bytes"
    sub = GenerationSub()
    with pytest.raises(MinimalContextBranchError, match="Context Receipt|changed"):
        execute_minimal_context_branch(
            drifted, result, sub, action="remove_and_branch", source_ids=[retained[0]], plan=plan,
        )
    assert sub.calls == 0
