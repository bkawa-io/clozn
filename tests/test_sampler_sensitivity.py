"""Planner and orchestration tests for the bounded Sampler Sensitivity Probe."""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn import schemas
from clozn.experiments.persistence import ObservationStore
from clozn.replay import execution_fork_results
from clozn.replay.controlled import _sampling_config, recorded_sampling_config
from clozn.replay.sampler_sensitivity import (
    SamplerSensitivityInputError,
    execute_sampler_sensitivity,
    plan_sampler_sensitivity,
)
from clozn.runs import store as runlog


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


RUNTIME = {
    "model_sha256": "a" * 64,
    "template_fingerprint": "b" * 16,
    "engine_build": "sampler-fixture",
    "context_size": 4096,
    "backend": "cpu",
    "adapter": {"present": False, "identity_sha256": None, "artifact_sha256": None, "scale": None},
    "white_box_flags": {},
}
WORKER = {"worker_id": "worker-a", "worker_generation_id": "generation-1", "protocol_version": "1.1"}


def exact_parent(*, sampled=True):
    """A recorded run complete enough for canonical exact state resolution."""
    run = parent(sampled=sampled)
    run.update({
        "substrate": "fixture",
        "messages": [{"role": "user", "content": "question"}],
        "assembled_messages": [{"role": "user", "content": "question"}],
        "final_prompt": "<prompt>",
        "identity": deepcopy(RUNTIME),
    })
    run["meta"].update({"n_ctx": 4096, "device": "cpu"})
    run["trace"]["steps"] = [
        {"token_id": token_id, "piece": piece}
        for token_id, piece in zip(run["trace"]["token_ids"], run["trace"]["tokens"])
    ]
    return run


def checkpoint_for(run):
    return {
        "checkpoint_id": "checkpoint-1",
        "worker_generation_id": WORKER["worker_generation_id"],
        "state": "available",
        "parent_run_id": run["id"],
        "prompt_tokens": 4,
        "n_past": 7,
    }


class ProbeEngine:
    """A worker exposing only the low-level exact-resume RPC.

    It echoes the fully resolved sampler it applied, which is the evidence a sampler probe rests
    on now that no child Run is written.
    """

    def __init__(self, *, diverge_every=2, ignores_override=False):
        self.calls = []
        self.diverge_every = diverge_every
        self.ignores_override = ignores_override
        self.probe_count = 0

    def execution_fork(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        intervention = kwargs["intervention"]
        recorded = {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "rep_penalty": 1.0, "seed": 1234}
        if intervention["type"] == "none":
            return self._reply([2, 3], ["b", "c"], {"type": "none"})
        self.probe_count += 1
        applied = dict(recorded)
        if not self.ignores_override:
            applied.update({k: v for k, v in intervention.items() if k != "type"})
        diverged = self.diverge_every and self.probe_count % self.diverge_every == 1
        tokens, pieces = ([9, 3], ["X", "c"]) if diverged else ([2, 3], ["b", "c"])
        return self._reply(tokens, pieces, {"type": "sampling", **applied})

    def _reply(self, tokens, pieces, applied):
        return {
            "worker_generation_id": WORKER["worker_generation_id"],
            "text": "".join(pieces),
            "tokens": list(tokens),
            "token_pieces": list(pieces),
            "steps": [{"token_id": t, "piece": p} for t, p in zip(tokens, pieces)],
            "restore_mode": "live_kv_truncated",
            "n_past_restored": 5,
            "exactness": {"source": "live_kv", "boundary_shape_true": True},
            "intervention_applied": applied,
            "finish_reason": "stop",
            "sampler_state_preserved": True,
        }


class Sub:
    def __init__(self, engine=None):
        self.engine = engine if engine is not None else ProbeEngine()
        self.runtime_identity = deepcopy(RUNTIME)
        self.worker_identity = deepcopy(WORKER)


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(execution_fork_results, "RESULTS_DIR", str(tmp_path / "execution-forks"))
    runlog._schema_verified.clear()
    return tmp_path


def test_sweep_produces_observations_and_changes_no_run_count(isolated_store):
    run = exact_parent()
    plan = plan_sampler_sensitivity(run, position=1, seed_probes=1)
    sub = Sub()
    store = ObservationStore()
    result = execute_sampler_sensitivity(
        run, sub, plan, runtime_identity=RUNTIME, worker_identity=WORKER,
        checkpoint=checkpoint_for(run), observation_store=store,
    )

    schemas.validate(result, "clozn.sampler-sensitivity.v1")
    assert result["summary"]["completed_probes"] == len(plan["probes"])
    assert result["summary"]["observations_completed"] == len(plan["probes"])
    assert "children_created" not in result["summary"]
    observation_ids = [probe["observation_id"] for probe in result["probes"]]
    assert len(set(observation_ids)) == len(plan["probes"])
    for probe in result["probes"]:
        assert store.get_observation(probe["observation_id"]).status == "completed"
        assert "child_run_id" not in probe
    # The hard gate: evaluating sampler sensitivity creates nothing.
    assert runlog.list_runs(20) == []
    assert execution_fork_results.list_for_parent(run["id"]) == []


def test_checkpoint_is_captured_once_and_control_proven_once(isolated_store):
    run = exact_parent()
    plan = plan_sampler_sensitivity(run, position=1, seed_probes=1)
    sub = Sub()
    result = execute_sampler_sensitivity(
        run, sub, plan, runtime_identity=RUNTIME, worker_identity=WORKER,
        checkpoint=checkpoint_for(run),
    )
    assert result["execution"]["checkpoint_capture"]["reused_for_probes"] is True
    assert result["execution"]["materialization"] == "explicit_choice_only"
    control_calls = [c for c in sub.engine.calls if c["intervention"]["type"] == "none"]
    assert len(control_calls) == 1
    probe_calls = [c for c in sub.engine.calls if c["intervention"]["type"] == "sampling"]
    assert len(probe_calls) == len(plan["probes"])


def test_probe_divergence_is_projected_per_probe_from_its_own_observation(isolated_store):
    run = exact_parent()
    plan = plan_sampler_sensitivity(run, position=1, seed_probes=1)
    result = execute_sampler_sensitivity(
        run, Sub(), plan, runtime_identity=RUNTIME, worker_identity=WORKER,
        checkpoint=checkpoint_for(run),
    )
    states = [probe["comparison"]["state"] for probe in result["probes"]]
    assert set(states) <= {"diverged", "identical"}
    assert "diverged" in states and "identical" in states
    for probe in result["probes"]:
        if probe["comparison"]["state"] == "diverged":
            # Divergence is measured from the probe boundary, so it can never precede it.
            assert probe["comparison"]["divergence_offset_from_probe"] >= 0
            assert probe["comparison"]["first_divergence_position"] >= plan["position"]


def test_resolved_sampler_comes_from_the_worker_receipt_not_a_child_run(isolated_store):
    run = exact_parent()
    plan = plan_sampler_sensitivity(run, position=1)
    result = execute_sampler_sensitivity(
        run, Sub(), plan, runtime_identity=RUNTIME, worker_identity=WORKER,
        checkpoint=checkpoint_for(run),
    )
    first = result["probes"][0]
    assert first["state"] == "completed"
    assert first["resolved_sampler"]["temperature"] == first["requested_change"]["temperature"]
    assert first["resolved_sampler"]["seed"] == 1234
    assert runlog.list_runs(20) == []


def test_worker_that_ignores_the_override_fails_and_is_never_relabelled(isolated_store):
    run = exact_parent()
    plan = plan_sampler_sensitivity(run, position=1)
    result = execute_sampler_sensitivity(
        run, Sub(ProbeEngine(ignores_override=True)), plan,
        runtime_identity=RUNTIME, worker_identity=WORKER, checkpoint=checkpoint_for(run),
    )
    assert result["summary"]["completed_probes"] == 0
    assert all(probe["state"] != "completed" for probe in result["probes"])
    assert runlog.list_runs(20) == []


def test_stale_supplied_exact_state_is_refused_without_reconstructed_fallback(isolated_store):
    run = exact_parent()
    plan = plan_sampler_sensitivity(run, position=1)
    stale = {**checkpoint_for(run), "worker_generation_id": "another-generation"}
    sub = Sub()
    result = execute_sampler_sensitivity(
        run, sub, plan, runtime_identity=RUNTIME, worker_identity=WORKER, checkpoint=stale,
    )
    assert result["summary"]["completed_probes"] == 0
    assert result["probes"][0]["reasons"][0]["code"] == "stale_worker_generation"
    assert all(probe.get("execution", {}).get("outcome") != "reconstructed_replay"
               for probe in result["probes"])
    assert sub.engine.calls == []
    assert runlog.list_runs(20) == []


def test_execution_does_not_fallback_to_reconstructed_replay(monkeypatch, isolated_store):
    run = exact_parent()
    plan = plan_sampler_sensitivity(run)
    monkeypatch.setattr(
        "clozn.replay.checkpoint_capture.capture_parent_checkpoint",
        lambda *a, **k: {"status": "unavailable",
                          "reasons": [{"code": "checkpoint_unavailable", "message": "no checkpoint"}]})
    result = execute_sampler_sensitivity(
        run, Sub(), plan, runtime_identity=RUNTIME, worker_identity=WORKER,
    )
    assert result["summary"]["observations_completed"] == 0
    assert result["execution"]["checkpoint_capture"]["state"] == "unavailable"
    assert all(probe["state"] == "not_attempted" for probe in result["probes"])
    assert runlog.list_runs(20) == []


def test_materializing_one_probe_creates_exactly_one_child_run(isolated_store):
    from clozn.experiments.materialize import materialize_generated_observation

    run = exact_parent()
    plan = plan_sampler_sensitivity(run, position=1)
    sub = Sub()
    store = ObservationStore()
    result = execute_sampler_sensitivity(
        run, sub, plan, runtime_identity=RUNTIME, worker_identity=WORKER,
        checkpoint=checkpoint_for(run), observation_store=store,
    )
    assert runlog.list_runs(20) == []
    chosen = result["probes"][0]
    calls = len(sub.engine.calls)

    materialized = materialize_generated_observation(
        run, chosen["experiment_id"], chosen["arm_id"],
        observation_id=chosen["observation_id"], observation_store=store,
    )
    assert materialized["state"] == "completed"
    assert len(runlog.list_runs(20)) == 1
    assert len(sub.engine.calls) == calls
    child = runlog.get_run(materialized["child_run_id"])
    assert child["parent_run_id"] == run["id"]
    lineage = child["changes_applied"]["experiment"]
    assert lineage["operation"] == "sample_with"
    assert lineage["observation_id"] == chosen["observation_id"]
    assert lineage["base_state"]["realized_fidelity"] == "exact_execution_fork"


def test_sampler_probe_refuses_a_reconstructed_state_with_kernel_semantics():
    """A sampler resume cannot be honestly reconstructed from text, and says so in kernel terms."""
    from clozn.experiments.state_ref import StateRef, operation_readiness, resolve_state

    run = exact_parent()
    reconstructed = resolve_state(
        StateRef.before_answer_token(run, 1), run=run, policy="reconstructed_only")
    readiness = operation_readiness(reconstructed, operation="sample_with")
    assert readiness["plannable"] is False
    assert readiness["reason_code"] == "sampler_probe_requires_exact_state"
    assert readiness["sampler"]["status"] == "unbound"


def test_sampler_probe_on_an_exact_state_is_plannable_but_unconfirmed_until_execution():
    from clozn.experiments.state_ref import StateRef, operation_readiness, resolve_state

    run = exact_parent()
    exact = resolve_state(
        StateRef.before_answer_token(run, 1), run=run, policy="exact_required",
        checkpoint=checkpoint_for(run), runtime_identity=RUNTIME, worker_identity=WORKER)
    readiness = operation_readiness(exact, operation="sample_with")
    assert readiness["plannable"] is True
    # Planning an exact resume is never proof; the unchanged control still has to run.
    assert readiness["available"] is False
    assert readiness["state"] == "requires_verification"
    assert readiness["reason_code"] == "exact_control_required"
