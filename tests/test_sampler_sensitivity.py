"""Planner and orchestration tests for the bounded Sampler Sensitivity Probe."""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn import schemas
from clozn.replay.controlled import _sampling_config, recorded_sampling_config
from clozn.replay.sampler_sensitivity import (
    SamplerSensitivityInputError,
    execute_sampler_sensitivity,
    plan_sampler_sensitivity,
)


def parent(*, sampled=True):
    meta = {}
    if sampled:
        meta["decode"] = {
            "mode": "sample", "temperature": 0.7, "top_p": 0.9, "top_k": 40,
            "repeat_penalty": 1.0, "seed": 1234,
        }
    else:
        meta["decode"] = {"mode": "greedy", "temperature": 0.0}
    return {
        "id": "run_sampler_parent",
        "model": "parent-model",
        "response": "abc",
        "trace": {"tokens": ["a", "b", "c"], "token_ids": [1, 2, 3]},
        "meta": meta,
    }


def test_public_sampler_reader_is_the_private_reader_and_handles_alias():
    run = parent()
    assert recorded_sampling_config(run) == _sampling_config(run)
    assert recorded_sampling_config(run)["repeat_penalty"] == 1.0
    run["meta"]["decode"].pop("repeat_penalty")
    run["meta"]["repetition_penalty"] = 1.1
    assert recorded_sampling_config(run)["repeat_penalty"] == 1.1


def test_nearby_v1_plan_is_explicit_ordered_and_deterministic():
    run = parent()
    before = deepcopy(run)
    plan = plan_sampler_sensitivity(run, position=1)
    again = plan_sampler_sensitivity(run, position=1)
    assert plan == again
    assert plan["execution"] == {
        "state": "ready", "fidelity": "exact_required", "live_state": "not_checked",
    }
    assert [(probe["axis"], probe["direction"], probe["change"]) for probe in plan["probes"]] == [
        ("temperature", "down", {"type": "sampling", "temperature": 0.56}),
        ("temperature", "up", {"type": "sampling", "temperature": 0.84}),
        ("top_p", "down", {"type": "sampling", "top_p": 0.85}),
        ("top_p", "up", {"type": "sampling", "top_p": 0.95}),
    ]
    assert plan["recipe"] == {
        "id": "nearby_v1",
        "temperature_multiplier_down": 0.8,
        "temperature_multiplier_up": 1.2,
        "top_p_delta": 0.05,
    }
    assert len({probe["probe_id"] for probe in plan["probes"]}) == 4
    assert parent() == before
    schemas.validate(plan, "clozn.sampler-sensitivity-plan.v1")


def test_seed_probes_are_bounded_deterministic_and_change_only_seed_in_plan():
    run = parent()
    plan = plan_sampler_sensitivity(run, seed_probes=2)
    seeds = [probe["change"]["seed"] for probe in plan["probes"] if probe["kind"] == "seed"]
    assert len(seeds) == 2
    assert len(set(seeds)) == 2
    assert all(seed != 1234 and 0 <= seed <= 2**32 - 1 for seed in seeds)
    assert all(set(probe["change"]) == {"type", "seed"} for probe in plan["probes"] if probe["kind"] == "seed")


@pytest.mark.parametrize("run,code", [
    (parent(sampled=False), "greedy_baseline_no_sampling_neighborhood"),
    ({"id": "r", "trace": {"tokens": ["a"]}, "meta": {"decode": {"mode": "sample"}}},
     "sampler_provenance_unavailable"),
    ({"id": "r", "trace": {"tokens": ["a"]}, "meta": {"decode": {
        "mode": "sample", "temperature": 0.7, "top_p": 0.9, "top_k": 40,
        "repeat_penalty": 1.0,
    }}}, "sampled_seed_unavailable"),
])
def test_missing_or_greedy_sampling_is_typed_unavailable(run, code):
    plan = plan_sampler_sensitivity(run)
    assert plan["execution"]["state"] == "unavailable"
    assert plan["execution"]["reason"] == code


def test_invalid_recipe_seed_count_and_position_are_typed():
    with pytest.raises(SamplerSensitivityInputError) as exc:
        plan_sampler_sensitivity(parent(), recipe="nearby_v2")
    assert exc.value.code == "invalid_recipe"
    with pytest.raises(SamplerSensitivityInputError) as exc:
        plan_sampler_sensitivity(parent(), seed_probes=3)
    assert exc.value.code == "invalid_seed_probes"
    with pytest.raises(SamplerSensitivityInputError) as exc:
        plan_sampler_sensitivity(parent(), position=3)
    assert exc.value.code == "invalid_position"


class Sub:
    engine = object()


def test_execution_captures_once_reuses_checkpoint_and_separates_probe_comparisons(monkeypatch):
    run = parent()
    plan = plan_sampler_sensitivity(run, seed_probes=1)
    counts = {"capture": 0, "plan": 0, "execute": 0}

    def capture(*args, **kwargs):
        counts["capture"] += 1
        return {"status": "available", "checkpoint_reference": {
            "checkpoint_id": "checkpoint-1", "worker_generation_id": "generation-1",
            "state": "available", "parent_run_id": run["id"], "prompt_tokens": 1,
            "n_past": 3,
        }}

    def exact_plan(parent_run, request, **kwargs):
        counts["plan"] += 1
        return {"classification": "exact_execution_fork"}

    def execute(parent_run, exact_plan, engine, **kwargs):
        counts["execute"] += 1
        # Use the request captured by the fake planner in a deterministic child for this test.
        change = seen_changes.pop(0)
        baseline = parent_run["meta"]["decode"]
        resolved = {
            "mode": "sample", "temperature": baseline["temperature"], "top_p": baseline["top_p"],
            "top_k": baseline["top_k"], "repeat_penalty": baseline["repeat_penalty"],
            "seed": baseline["seed"],
        }
        resolved.update({key: value for key, value in change.items() if key != "type"})
        child = deepcopy(parent_run)
        child["id"] = f"child-{counts['execute']}"
        child["meta"] = {"decode": resolved}
        child["response"] = "aXc" if counts["execute"] % 2 else "abc"
        child["trace"] = {"tokens": list(child["response"]), "token_ids": [1, 9, 3] if child["response"] != "abc" else [1, 2, 3]}
        return {"receipt": {
            "phase": "completed", "execution_id": f"fork-{counts['execute']}",
            "exactness": {"proof_status": "confirmed"},
            "unchanged_control": {"status": "matched"},
        }, "child": child}

    seen_changes = [probe["change"] for probe in plan["probes"]]
    monkeypatch.setattr("clozn.replay.fork.capture_exact_fork_context", capture)
    monkeypatch.setattr("clozn.replay.fork.plan_exact_force_token", exact_plan)
    monkeypatch.setattr("clozn.replay.fork.execute_exact_force_token", execute)
    result = execute_sampler_sensitivity(
        run, Sub(), plan,
        runtime_identity={"runtime": "same"}, worker_identity={"worker": "same"},
    )
    assert counts == {"capture": 1, "plan": 5, "execute": 5}
    assert result["execution"]["checkpoint_capture"]["reused_for_probes"] is True
    assert result["summary"]["children_created"] == 5
    assert result["parameter_sensitivity"]["diverged"] == 2
    assert result["seed_sensitivity"]["diverged"] == 1
    assert all("sensitivity_score" not in str(result) for _ in [0])
    schemas.validate(result, "clozn.sampler-sensitivity.v1")


def test_execution_does_not_fallback_to_reconstructed_replay(monkeypatch):
    run = parent()
    plan = plan_sampler_sensitivity(run)
    monkeypatch.setattr("clozn.replay.fork.capture_exact_fork_context", lambda *a, **k: {
        "status": "ineligible", "reason": {"code": "checkpoint_unavailable", "message": "no checkpoint"}
    })
    monkeypatch.setattr("clozn.replay.fork.fork", lambda *a, **k: pytest.fail("reconstructed sampler fork"))
    result = execute_sampler_sensitivity(
        run, Sub(), plan,
        runtime_identity={"runtime": "same"}, worker_identity={"worker": "same"},
    )
    assert result["summary"]["children_created"] == 0
    assert result["execution"]["checkpoint_capture"]["state"] == "unavailable"
