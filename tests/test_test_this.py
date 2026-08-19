"""Pure planning and dispatcher tests for Universal Test This."""
from __future__ import annotations

from copy import deepcopy
import importlib

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


def test_execution_dispatches_force_token_through_time_travel_without_child_run(monkeypatch):
    parent = run()
    seen = {}

    class FakeTravel:
        status = "completed"
        experiment_id = "experiment-1"
        arm_id = "arm-1"
        observation_id = "observation-1"

        def to_dict(self):
            return {
                "schema_version": "clozn.time-travel-result.v1",
                "run_id": parent["id"], "status": "completed",
                "experiment_id": self.experiment_id, "arm_id": self.arm_id,
                "observation_id": self.observation_id, "diagnostics": {},
            }

    def fake_time_travel(run_arg, **kwargs):
        seen.update({"run": run_arg, **kwargs})
        return FakeTravel()

    monkeypatch.setattr(importlib.import_module("clozn.recipes.time_travel"), "run_time_travel", fake_time_travel)
    result = executor.execute_test_this(parent, object(), token_request(),
                                        runtime_identity=None, worker_identity=None)
    assert result["outcome"] == "completed"
    assert "child_run_id" not in result
    assert result["result"]["observation_id"] == "observation-1"
    assert seen["position"] == 1
    assert seen["token_id"] == 20
    assert seen["token_piece"] is None


def test_branch_fan_dispatches_directly_and_preserves_partial_result(monkeypatch):
    parent = run()
    seen = {}
    fan_result = {
        "schema_version": "clozn.branch-fan.v2",
        "parent_run_id": parent["id"], "position": 1,
        "selection": {"source": "recorded_alternatives", "state": "available", "requested_limit": 3,
                       "recorded_alternatives": 2, "selected_alternatives": 2},
        "execution": {"policy": "exact_first", "order": "sequential",
                       "materialization": "explicit_choice_only",
                       "checkpoint_capture": {"state": "available", "reused_for_exact_candidates": True},
                       "fidelity": "mixed"},
        "branches": [],
        "summary": {"status": "partial", "requested": 2, "attempted": 2,
                    "observations_completed": 1, "exact_observations": 1,
                    "reconstructed_observations": 0, "unavailable": 1, "not_attempted": 0},
    }

    def fake_fan(*args, **kwargs):
        seen.update({"args": args, "kwargs": kwargs})
        return fan_result

    monkeypatch.setattr("clozn.replay.branch_fan.branch_fan", fake_fan)
    request = token_request({"kind": "fan_alternatives", "limit": 2})
    result = executor.execute_test_this(parent, object(), request)
    assert result["outcome"] == "partial"
    assert result["artifact"]["schema"] == "clozn.branch-fan.v2"
    assert seen["kwargs"]["limit"] == 2
    assert seen["args"][2] == 1


def test_sampling_dispatch_requires_exact_and_never_uses_reconstructed_fork(monkeypatch):
    parent = run(sampling=True)
    request = {
        "selection": {"kind": "sampling", "position": 0},
        "test": {"kind": "change_sampling", "changes": {"temperature": 0.2}},
    }
    monkeypatch.setattr(
        "clozn.replay.checkpoint_capture.capture_parent_checkpoint",
        lambda *a, **k: {"status": "unavailable",
                          "reasons": [{"code": "checkpoint_unavailable", "message": "no checkpoint"}]})
    result = executor.execute_test_this(parent, type("Sub", (), {"engine": object()})(), request,
                                        runtime_identity={"x": 1}, worker_identity={"x": 1})
    assert result["outcome"] == "unavailable"
    assert result["operation"] == "sampling_fork"
    # No exact context means no probe at all; it is never quietly reconstructed.
    assert "child_run_id" not in result


def test_result_does_not_embed_raw_generated_text(monkeypatch):
    parent = run()
    class FakeTravel:
        status = "completed"
        experiment_id = arm_id = observation_id = "ref"

        def to_dict(self):
            return {"schema_version": "clozn.time-travel-result.v1", "status": "completed"}

    monkeypatch.setattr(importlib.import_module("clozn.recipes.time_travel"), "run_time_travel", lambda *a, **k: FakeTravel())
    result = executor.execute_test_this(parent, object(), token_request())
    assert "SECRET ANSWER" not in str(result)


# ------------------------------------------------------------- observation-first execution gates
SAMPLER_RUNTIME = {
    "model_sha256": "a" * 64,
    "template_fingerprint": "b" * 16,
    "engine_build": "test-this-fixture",
    "context_size": 4096,
    "backend": "cpu",
    "adapter": {"present": False, "identity_sha256": None, "artifact_sha256": None, "scale": None},
    "white_box_flags": {},
}
SAMPLER_WORKER = {"worker_id": "worker-a", "worker_generation_id": "generation-1", "protocol_version": "1.1"}


def exact_sampled_run():
    parent = run(sampling=True)
    parent.update({
        "substrate": "fixture",
        "messages": [{"role": "user", "content": "question"}],
        "assembled_messages": [{"role": "user", "content": "question"}],
        "final_prompt": "<prompt>",
        "identity": deepcopy(SAMPLER_RUNTIME),
    })
    parent["meta"].update({"n_ctx": 4096, "device": "cpu"})
    parent["trace"]["steps"] = [
        {"token_id": token_id, "piece": piece}
        for token_id, piece in zip(parent["trace"]["token_ids"], parent["trace"]["tokens"])
    ]
    return parent


class SamplerProbeEngine:
    def __init__(self):
        self.calls = []

    def execution_fork(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        intervention = kwargs["intervention"]
        recorded = {"temperature": 0.7, "top_p": 0.95, "top_k": 40, "rep_penalty": 1.0, "seed": 12}
        if intervention["type"] == "none":
            tokens, pieces, applied = [10, 11, 12], ["a", "b", "c"], {"type": "none"}
        else:
            applied = {"type": "sampling", **recorded,
                       **{k: v for k, v in intervention.items() if k != "type"}}
            tokens, pieces = [30, 11, 12], ["Z", "b", "c"]
        return {
            "worker_generation_id": SAMPLER_WORKER["worker_generation_id"],
            "text": "".join(pieces), "tokens": tokens, "token_pieces": pieces,
            "steps": [{"token_id": t, "piece": p} for t, p in zip(tokens, pieces)],
            "restore_mode": "reprefill", "n_past_restored": 4,
            "exactness": {"source": "reprefill", "boundary_shape_true": True},
            "intervention_applied": applied, "finish_reason": "stop",
            "sampler_state_preserved": True,
        }


class SamplerSub:
    def __init__(self):
        self.engine = SamplerProbeEngine()
        self.runtime_identity = deepcopy(SAMPLER_RUNTIME)
        self.worker_identity = deepcopy(SAMPLER_WORKER)


@pytest.fixture
def sampler_probe(tmp_path, monkeypatch):
    from clozn.runs import store as runlog

    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    runlog._schema_verified.clear()
    monkeypatch.setattr(
        "clozn.replay.checkpoint_capture.capture_parent_checkpoint",
        lambda parent, engine, **kwargs: {
            "status": "available",
            "checkpoint_reference": {
                "checkpoint_id": "checkpoint-1",
                "worker_generation_id": SAMPLER_WORKER["worker_generation_id"],
                "state": "available", "parent_run_id": parent["id"],
                "prompt_tokens": 4, "n_past": 7,
            },
        })
    return runlog


def _sampling_request():
    return {
        "selection": {"kind": "sampling", "position": 0},
        "test": {"kind": "change_sampling", "changes": {"temperature": 0.2}},
    }


def test_sampling_test_produces_an_observation_and_changes_no_run_count(sampler_probe):
    runlog = sampler_probe
    parent = exact_sampled_run()
    sub = SamplerSub()
    result = executor.execute_test_this(
        parent, sub, _sampling_request(),
        runtime_identity=SAMPLER_RUNTIME, worker_identity=SAMPLER_WORKER)

    assert result["operation"] == "sampling_fork"
    assert result["outcome"] == "completed"
    assert result["result"]["observation_id"]
    assert result["result"]["time_travel"]["operation"]["kind"] == "sample_with"
    assert "child_run_id" not in result
    # The hard gate: no Run is created by evaluating.
    assert runlog.list_runs(20) == []


def test_force_token_test_produces_an_observation_and_changes_no_run_count(sampler_probe):
    runlog = sampler_probe
    parent = exact_sampled_run()
    parent["meta"].pop("decode")            # a greedy parent, so ForceToken is plannable
    sub = SamplerSub()
    result = executor.execute_test_this(
        parent, sub, token_request({"kind": "try_alternative", "token_id": 20}),
        runtime_identity=SAMPLER_RUNTIME, worker_identity=SAMPLER_WORKER)

    assert result["operation"] == "force_token"
    assert result["result"]["observation_id"] or result["outcome"] != "completed"
    assert "child_run_id" not in result
    assert runlog.list_runs(20) == []


def test_materializing_a_sampling_observation_creates_exactly_one_child_run(sampler_probe):
    from clozn.experiments.materialize import materialize_generated_observation
    from clozn.experiments.persistence import ObservationStore

    runlog = sampler_probe
    parent = exact_sampled_run()
    sub = SamplerSub()
    result = executor.execute_test_this(
        parent, sub, _sampling_request(),
        runtime_identity=SAMPLER_RUNTIME, worker_identity=SAMPLER_WORKER)
    assert result["outcome"] == "completed"
    assert runlog.list_runs(20) == []
    calls = len(sub.engine.calls)

    materialized = materialize_generated_observation(
        parent, result["result"]["experiment_id"], result["result"]["arm_id"],
        observation_id=result["result"]["observation_id"],
        observation_store=ObservationStore(),
    )
    assert materialized["state"] == "completed"
    assert len(runlog.list_runs(20)) == 1
    assert len(sub.engine.calls) == calls
    child = runlog.get_run(materialized["child_run_id"])
    assert child["parent_run_id"] == parent["id"]
    assert child["changes_applied"]["experiment"]["operation"] == "sample_with"
