"""Pure planning and dispatcher tests for Universal Test This."""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn import schemas
from clozn.runs.test_this import TestThisInputError, build_test_this_plan
from clozn.replay import test_this as executor


def run(*, sampling=False):
    meta = {}
    if sampling:
        meta["decode"] = {
            "mode": "sample", "temperature": 0.7, "top_k": 40,
            "top_p": 0.95, "seed": 12, "repeat_penalty": 1.0,
        }
    return {
        "id": "run_parent",
        "model": "model-parent",
        "response": "abc",
        "trace": {
            "tokens": ["a", "b", "c"],
            "token_ids": [10, 11, 12],
            "alternatives": [
                [],
                [{"piece": "x", "token_id": 20, "prob": 0.39},
                 {"piece": "y", "token_id": 21, "probability": 0.12}],
                [],
            ],
        },
        "meta": meta,
    }


def token_request(test=None):
    return {
        "selection": {"kind": "response_token", "position": 1},
        "test": test or {"kind": "try_alternative", "alternative_rank": 0},
    }


def test_token_plan_resolves_recorded_candidate_and_fingerprint():
    parent = run()
    before = deepcopy(parent)
    plan = build_test_this_plan(parent, token_request())
    assert plan["resolution"] == {"state": "ready", "operation": "force_token"}
    assert plan["resolved_test"]["recorded_alternative"] == {
        "rank": 0, "token_id": 20, "probability": 0.39,
    }
    assert len(plan["parent_fingerprint_sha256"]) == 64
    assert "piece" not in str(plan)
    assert parent == before
    schemas.validate(plan, "clozn.test-this-plan.v1")


def test_sampling_plan_defaults_position_and_reuses_fork_validation():
    plan = build_test_this_plan(run(sampling=True), {
        "selection": {"kind": "sampling"},
        "test": {"kind": "change_sampling", "changes": {"temperature": 0, "top_p": 0.8}},
    })
    assert plan["selection"]["position"] == 0
    assert plan["resolution"]["operation"] == "sampling_fork"
    assert plan["execution"]["fidelity_policy"] == "exact_required"


def test_sampling_no_effective_change_is_unavailable():
    plan = build_test_this_plan(run(sampling=True), {
        "selection": {"kind": "sampling", "position": 0},
        "test": {"kind": "change_sampling", "changes": {"temperature": 0.7, "seed": 12}},
    })
    assert plan["resolution"] == {
        "state": "unavailable", "operation": "sampling_fork",
        "reason": {"code": "no_effective_change", "message": "the requested sampler change has no effective difference from the parent"},
    }


def test_sampler_sensitivity_is_an_additive_read_only_dispatch_plan():
    plan = build_test_this_plan(run(sampling=True), {
        "selection": {"kind": "sampling", "position": 1},
        "test": {"kind": "probe_sensitivity", "recipe": "nearby_v1", "seed_probes": 1},
    })
    assert plan["resolution"] == {"state": "ready", "operation": "sampler_sensitivity"}
    assert plan["execution"] == {
        "state": "ready", "backend": "sampler_sensitivity",
        "fidelity_policy": "exact_required", "live_state": "not_checked",
    }
    assert len(plan["resolved_test"]["sampler_sensitivity_plan"]["probes"]) == 5
    schemas.validate(plan, "clozn.test-this-plan.v1")


def test_sampler_sensitivity_test_this_dispatches_without_recipe_duplication(monkeypatch):
    parent = run(sampling=True)
    request = {
        "selection": {"kind": "sampling", "position": 1},
        "test": {"kind": "probe_sensitivity", "seed_probes": 0},
    }
    seen = {}

    def fake_execute(parent_arg, sub, plan, **kwargs):
        seen.update({"parent": parent_arg, "plan": plan, "kwargs": kwargs})
        return {
            "schema_version": "clozn.sampler-sensitivity.v1",
            "test_id": plan["test_id"], "parent_run_id": parent_arg["id"], "position": 1,
            "baseline_sampler": plan["baseline_sampler"], "recipe": plan["recipe"],
            "execution": {"state": "available", "fidelity": "exact_required", "order": "sequential",
                           "checkpoint_capture": {"state": "available", "reused_for_probes": True}},
            "probes": [], "parameter_sensitivity": {"state": "inconclusive"},
            "seed_sensitivity": {"state": "not_requested"},
            "summary": {"status": "completed", "planned_probes": 0, "completed_probes": 0,
                        "children_created": 0},
        }

    monkeypatch.setattr("clozn.replay.sampler_sensitivity.execute_sampler_sensitivity", fake_execute)
    result = executor.execute_test_this(parent, object(), request)
    assert result["operation"] == "sampler_sensitivity"
    assert result["outcome"] == "completed"
    assert result["artifact"]["schema"] == "clozn.sampler-sensitivity.v1"
    assert seen["plan"]["recipe"]["id"] == "nearby_v1"


@pytest.mark.parametrize("body_value,code", [
    ({"selection": {"kind": "unknown", "position": 1}, "test": {}}, "invalid_selection_kind"),
    ({"selection": {"kind": "response_token", "position": 1}, "test": {"kind": "change_sampling", "changes": {"temperature": 0}}}, "selection_test_mismatch"),
    ({"selection": {"kind": "sampling", "position": 1}, "test": {"kind": "try_alternative", "alternative_rank": 0}}, "selection_test_mismatch"),
    ({"selection": {"kind": "response_token", "position": 99}, "test": {"kind": "fan_alternatives"}}, "invalid_position"),
    ({"selection": {"kind": "response_token", "position": 1}, "test": {"kind": "try_alternative"}}, "invalid_selector"),
    ({"selection": {"kind": "response_token", "position": 1}, "test": {"kind": "try_alternative", "alternative_rank": 0, "token_id": 20}}, "invalid_selector"),
    ({"selection": {"kind": "sampling", "position": 0}, "test": {"kind": "change_sampling", "changes": {}}}, "invalid_intervention"),
    ({"selection": {"kind": "sampling", "position": 0}, "test": {"kind": "change_sampling", "changes": {"top_p": 2}}}, "invalid_intervention"),
])
def test_invalid_requests_are_typed(body_value, code):
    with pytest.raises(TestThisInputError) as exc:
        build_test_this_plan(run(), body_value)
    assert exc.value.code == code


def test_branch_fan_plan_is_a_dispatch_only_and_limit_is_bounded():
    plan = build_test_this_plan(run(), token_request({"kind": "fan_alternatives", "limit": 4}))
    assert plan["test"] == {"kind": "fan_alternatives", "limit": 4}
    assert plan["resolved_test"] == {"operation": "branch_fan"}
    assert "recorded_alternative" not in plan["resolved_test"]


def test_malformed_recorded_alternative_is_unavailable():
    parent = run()
    parent["trace"]["alternatives"][1] = [{"piece": "", "token_id": 20}]
    plan = build_test_this_plan(parent, token_request())
    assert plan["resolution"]["state"] == "unavailable"
    assert plan["resolution"]["reason"]["code"] == "alternative_unavailable"


def test_execution_dispatches_force_token_without_duplicating_fork(monkeypatch):
    parent = run()
    seen = {}
    child = {"id": "child", "outcome": "reconstructed_replay", "reasons": [],
             "trace": {"tokens": ["a", "x", "c"], "token_ids": [10, 20, 12]},
             "response": "axc"}

    def fake_compat(run_arg, sub, position, **kwargs):
        seen.update({"run": run_arg, "sub": sub, "position": position, **kwargs})
        return child

    monkeypatch.setattr("clozn.replay.fork.compat_fork", fake_compat)
    result = executor.execute_test_this(parent, object(), token_request(),
                                        runtime_identity=None, worker_identity=None)
    assert result["outcome"] == "completed"
    assert result["child_run_id"] == "child"
    assert seen["position"] == 1
    assert seen["token_id"] == 20
    assert "token" not in seen
    assert result["comparison"]["state"] in {"available", "trace_unavailable"}


def test_branch_fan_dispatches_directly_and_preserves_partial_result(monkeypatch):
    parent = run()
    seen = {}
    fan_result = {
        "schema_version": "clozn.branch-fan.v1",
        "parent_run_id": parent["id"], "position": 1,
        "selection": {"source": "recorded_alternatives", "state": "available", "requested_limit": 3,
                       "recorded_alternatives": 2, "selected_alternatives": 2},
        "execution": {"policy": "exact_first", "order": "sequential",
                       "checkpoint_capture": {"state": "available", "reused_for_exact_candidates": True},
                       "fidelity": "mixed"},
        "branches": [],
        "summary": {"status": "partial", "requested_branches": 2, "attempted_branches": 2,
                    "children_created": 1, "exact_children": 1, "reconstructed_children": 0,
                    "unavailable_branches": 1, "not_attempted_branches": 0},
    }

    def fake_fan(*args, **kwargs):
        seen.update({"args": args, "kwargs": kwargs})
        return fan_result

    monkeypatch.setattr("clozn.replay.branch_fan.branch_fan", fake_fan)
    request = token_request({"kind": "fan_alternatives", "limit": 2})
    result = executor.execute_test_this(parent, object(), request)
    assert result["outcome"] == "partial"
    assert result["artifact"]["schema"] == "clozn.branch-fan.v1"
    assert seen["kwargs"]["limit"] == 2
    assert seen["args"][2] == 1


def test_sampling_dispatch_requires_exact_and_never_uses_reconstructed_fork(monkeypatch):
    parent = run(sampling=True)
    request = {
        "selection": {"kind": "sampling", "position": 0},
        "test": {"kind": "change_sampling", "changes": {"temperature": 0.2}},
    }
    monkeypatch.setattr("clozn.replay.fork.capture_exact_fork_context", lambda *a, **k: {
        "status": "ineligible", "reason": {"code": "checkpoint_unavailable", "message": "no checkpoint"}
    })
    monkeypatch.setattr("clozn.replay.fork.compat_fork", lambda *a, **k: pytest.fail("reconstructed fallback"))
    result = executor.execute_test_this(parent, type("Sub", (), {"engine": object()})(), request,
                                        runtime_identity={"x": 1}, worker_identity={"x": 1})
    assert result["outcome"] == "unavailable"
    assert result["operation"] == "sampling_fork"


def test_result_does_not_embed_raw_child_text(monkeypatch):
    parent = run()
    child = {"id": "child", "outcome": "reconstructed_replay", "reasons": [],
             "response": "SECRET ANSWER", "trace": {"tokens": ["a", "x", "c"], "token_ids": [10, 20, 12]}}
    monkeypatch.setattr("clozn.replay.fork.compat_fork", lambda *a, **k: child)
    result = executor.execute_test_this(parent, object(), token_request())
    assert "SECRET ANSWER" not in str(result)
