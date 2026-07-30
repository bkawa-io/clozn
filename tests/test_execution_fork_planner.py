"""Model-free contract tests for the fail-closed execution-fork planner."""
from __future__ import annotations

import pytest

from clozn import schemas
from clozn.cli.worker_registry import AdapterRuntimeIdentity, RuntimeKey
from clozn.replay.execution_fork import CLASSIFICATIONS, plan_execution_fork


def _runtime(*, model="a", adapter=None):
    return {
        "model_sha256": model * 64,
        "template_fingerprint": "b" * 16,
        "engine_build": "clozn-engine-test",
        "context_size": 4096,
        "backend": "cpu",
        "adapter": adapter if adapter is not None else {
            "present": False,
            "identity_sha256": None,
            "artifact_sha256": None,
            "scale": None,
        },
        "white_box_flags": {},
    }


def _parent(*, model="a"):
    return {
        "id": "run_parent",
        "identity": {
            "model_sha256": model * 64,
            "template_fingerprint": "b" * 16,
            "engine_build": "clozn-engine-test",
            "white_box_flags": {},
        },
        "meta": {"n_ctx": 4096, "device": "cpu"},
        "final_prompt": "<prompt>",
        "trace": {
            "tokens": ["one", " two", " three"],
            "token_ids": [11, 22, 33],
        },
    }


def _worker(generation="generation-a"):
    return {
        "worker_id": "worker-a",
        "worker_generation_id": generation,
        "protocol_version": "1.1",
    }


def _checkpoint(**changes):
    out = {
        "checkpoint_id": "ckpt-generation-a-7",
        "worker_generation_id": "generation-a",
        "state": "available",
        "parent_run_id": "run_parent",
        "prompt_tokens": 10,
        "n_past": 13,
        "size_bytes": 8192,
    }
    out.update(changes)
    return out


def _plan(*, position=1, change=None, checkpoint_marker=True, parent=None,
          runtime=None, worker=None):
    kwargs = {}
    if checkpoint_marker is not False:
        kwargs["checkpoint"] = (
            _checkpoint() if checkpoint_marker is True else checkpoint_marker)
    return plan_execution_fork(
        parent or _parent(),
        {"position": position, "change": change or {"type": "force_token", "token_id": 44}},
        worker_identity=worker or _worker(),
        runtime_identity=runtime or _runtime(),
        **kwargs,
    )


def test_classification_vocabulary_is_closed_and_exact_plan_validates():
    assert CLASSIFICATIONS == (
        "exact_execution_fork", "reconstructed_replay", "unavailable")
    plan = _plan()
    schemas.validate(plan)
    assert plan["classification"] == "exact_execution_fork"
    assert plan["checkpoint_reference"] == _checkpoint()
    assert plan["exactness"] == {
        "regime": "generated_token_live_kv",
        "source": "live_kv",
        "proof_status": "planned",
        "truncate_to": 11,
        "boundary_shape_true": True,
    }
    assert plan["unavoidable_differences"] == []
    assert plan["unchanged_control"] == {
        "required": True, "status": "required_not_run"}
    assert plan["child_lineage"]["parent_run_id"] == "run_parent"
    assert plan["child_lineage"]["receipt_status"] == "not_created"


def test_fork_at_first_response_token_uses_prompt_boundary_reprefill_regime():
    plan = _plan(position=0)
    assert plan["classification"] == "exact_execution_fork"
    assert plan["exactness"]["regime"] == "prompt_boundary_reprefill"
    assert plan["exactness"]["source"] == "reprefill"
    assert plan["exactness"]["truncate_to"] == 10


@pytest.mark.parametrize(
    "change",
    [
        {"type": "none"},
        {"type": "force_token", "token_id": 44, "token_piece": " four"},
        {"type": "sampling", "temperature": 0.7, "top_k": 40, "top_p": 0.9,
         "seed": 7, "rep_penalty": 1.1},
        {"type": "steer", "clear": True},
        {"type": "steer", "steer_vec": [0.1, -0.2], "steer_layer": 4,
         "steer_coef": 0.75},
        {"type": "residual_write", "layer": 4, "position": 10, "values": [0.1, 0.2]},
    ],
)
def test_every_closed_engine_intervention_shape_can_be_planned_exactly(change):
    plan = _plan(change=change)
    schemas.validate(plan)
    assert plan["classification"] == "exact_execution_fork"
    assert plan["request"]["execution_change"] == change


def test_checkpoint_omission_explicitly_classifies_reconstructed_replay():
    plan = _plan(
        checkpoint_marker=False,
        change={"type": "force_token", "token_piece": " 2"},
    )
    schemas.validate(plan)
    assert plan["classification"] == "reconstructed_replay"
    assert "checkpoint_reference" not in plan
    assert plan["exactness"] == {
        "regime": "reconstructed_text",
        "source": "text_retokenization",
        "proof_status": "not_applicable",
    }
    assert plan["unavoidable_differences"] == [
        "kv_state_not_restored",
        "sampler_state_reinitialized",
        "prompt_prefix_retokenized",
        "batch_shape_not_preserved",
    ]
    assert plan["reasons"][0]["code"] == "checkpoint_not_supplied"


@pytest.mark.parametrize(
    ("checkpoint", "reason"),
    [
        (_checkpoint(state="missing"), "checkpoint_missing"),
        (_checkpoint(state="expired"), "checkpoint_expired"),
        (_checkpoint(worker_generation_id="old-generation"), "stale_worker_generation"),
        (_checkpoint(parent_run_id="run_someone_else"), "checkpoint_parent_mismatch"),
        (_checkpoint(prompt_tokens=0), "missing_prompt_boundary"),
        (_checkpoint(n_past=10), "checkpoint_range_mismatch"),
    ],
)
def test_unusable_supplied_checkpoint_is_unavailable_never_reconstructed(checkpoint, reason):
    plan = _plan(checkpoint_marker=checkpoint)
    assert plan["classification"] == "unavailable"
    assert plan["reasons"][0]["code"] == reason
    assert plan["exactness"]["regime"] == "unavailable"
    assert plan["unavoidable_differences"] == []


def test_malformed_supplied_checkpoint_fails_closed_without_becoming_reconstruction():
    plan = _plan(checkpoint_marker={"checkpoint_id": "only-one-field"})
    assert plan["classification"] == "unavailable"
    assert plan["reasons"][0]["code"] == "checkpoint_missing"
    assert "checkpoint_reference" not in plan


@pytest.mark.parametrize(
    "trace",
    [
        {},
        {"tokens": ["one"]},
        {"tokens": ["one"], "token_ids": []},
        {"tokens": ["one"], "token_ids": [None]},
        {"tokens": ["one", "two"], "token_ids": [1]},
    ],
)
def test_incomplete_response_token_boundaries_fail_before_generation(trace):
    parent = _parent()
    parent["trace"] = trace
    plan = _plan(parent=parent)
    assert plan["classification"] == "unavailable"
    assert plan["reasons"][0]["code"] == "missing_response_token_boundary"


def test_out_of_range_response_position_is_unavailable():
    plan = _plan(position=3)
    assert plan["classification"] == "unavailable"
    assert plan["reasons"][0]["code"] == "position_out_of_range"


def test_unknown_and_malformed_interventions_fail_before_boundary_or_worker_checks():
    unknown = _plan(change={"type": "rewrite_everything"})
    assert unknown["classification"] == "unavailable"
    assert unknown["reasons"][0]["code"] == "unsupported_intervention"
    assert "execution_change" not in unknown["request"]

    malformed = _plan(change={"type": "sampling"})
    assert malformed["classification"] == "unavailable"
    assert malformed["reasons"][0]["code"] == "invalid_intervention"


def test_reconstruction_rejects_interventions_only_supported_by_exact_worker():
    plan = _plan(
        checkpoint_marker=False,
        change={"type": "sampling", "temperature": 0.5},
    )
    assert plan["classification"] == "unavailable"
    assert plan["reasons"][0]["code"] == "reconstruction_unsupported_intervention"


def test_reconstruction_requires_rendered_prompt_and_forced_piece():
    parent = _parent()
    parent.pop("final_prompt")
    no_prompt = _plan(
        checkpoint_marker=False,
        parent=parent,
        change={"type": "force_token", "token_piece": "x"},
    )
    assert no_prompt["reasons"][0]["code"] == "reconstruction_prompt_unavailable"

    no_piece = _plan(
        checkpoint_marker=False,
        change={"type": "force_token", "token_id": 99},
    )
    assert no_piece["reasons"][0]["code"] == "reconstruction_token_piece_unavailable"


def test_exact_force_token_requires_id_and_does_not_downgrade_piece_to_text_replay():
    plan = _plan(change={"type": "force_token", "token_piece": "x"})
    assert plan["classification"] == "unavailable"
    assert plan["reasons"][0]["code"] == "invalid_intervention"


def test_runtime_identity_is_complete_exact_and_adapter_sensitive():
    mismatch = _plan(runtime=_runtime(model="c"))
    assert mismatch["classification"] == "unavailable"
    assert mismatch["reasons"][0]["code"] == "runtime_identity_mismatch"

    missing = _runtime()
    missing.pop("backend")
    unavailable = _plan(runtime=missing)
    assert unavailable["reasons"][0]["code"] == "runtime_identity_unavailable"

    parent = _parent()
    parent["identity"]["ext"] = {
        "adapter": {"path": "adapter.gguf", "scale": 0.0}}
    adapted = _plan(parent=parent)
    assert adapted["reasons"][0]["code"] == "runtime_identity_unavailable"

    parent["identity"]["ext"]["adapter"]["artifact_sha256"] = "d" * 64
    adapted = _plan(parent=parent)
    assert adapted["reasons"][0]["code"] == "runtime_identity_mismatch"

    zero_penalty = _plan(change={"type": "sampling", "rep_penalty": 0})
    assert zero_penalty["reasons"][0]["code"] == "invalid_intervention"


def test_registry_runtime_key_and_worker_identity_are_consumed_without_rehash_drift():
    flags = {"sae": False, "jlens": False, "attn_knockout": False}
    key = RuntimeKey(
        gguf_artifact_sha256="a" * 64,
        context_size=4096,
        backend="cpu",
        adapter=AdapterRuntimeIdentity.absent(),
        template_fingerprint="b" * 16,
        engine_build="clozn-engine-test",
        white_box_flags=flags,
    )
    parent = _parent()
    parent["identity"]["white_box_flags"] = flags
    worker = {
        "worker_id": "generation-a",
        "worker_generation_id": "generation-a",
        "worker_generation": 1,
        "runtime_key_sha256": key.key_sha256,
        "protocol_version": "1.1",
        "engine_build": "clozn-engine-test",
        "backend": "cpu",
    }
    plan = _plan(parent=parent, runtime=key.as_dict(), worker=worker)
    assert plan["classification"] == "exact_execution_fork"
    assert plan["identity"]["parent_runtime"]["runtime_key_sha256"] == key.key_sha256
    assert plan["identity"]["selected_runtime"]["runtime_key_sha256"] == key.key_sha256
    assert plan["identity"]["selected_worker"] == {
        "worker_id": "generation-a",
        "worker_generation_id": "generation-a",
        "protocol_version": "1.1",
    }

    tampered = key.as_dict()
    tampered["key_sha256"] = "f" * 64
    unavailable = _plan(parent=parent, runtime=tampered, worker=worker)
    assert unavailable["reasons"][0]["code"] == "runtime_identity_unavailable"


def test_worker_identity_is_required_even_for_reconstruction():
    plan = plan_execution_fork(
        _parent(),
        {"position": 1, "change": {"type": "force_token", "token_piece": "x"}},
        worker_identity=None,
        runtime_identity=_runtime(),
    )
    assert plan["classification"] == "unavailable"
    assert plan["reasons"][0]["code"] == "worker_identity_unavailable"


def test_plan_and_change_identity_are_deterministic_across_mapping_order():
    one = _plan(change={"type": "sampling", "temperature": 0.4, "seed": 7})
    two = _plan(change={"seed": 7, "temperature": 0.4, "type": "sampling"})
    assert one["plan_id"] == two["plan_id"]
    assert one["request"]["change_sha256"] == two["request"]["change_sha256"]
    assert one["child_lineage"]["change_sha256"] == one["request"]["change_sha256"]


@pytest.mark.parametrize(
    ("parent", "fork_request", "match"),
    [
        ({}, {"position": 0, "change": {"type": "none"}}, "parent_run.id"),
        (_parent(), {"position": -1, "change": {"type": "none"}}, "position"),
        (_parent(), {"position": 0, "change": {}}, "change.type"),
        (_parent(), {"position": 0, "change": {"type": "steer", "steer_vec": [float("nan")]}},
         "JSON-serializable"),
    ],
)
def test_malformed_non_artifact_inputs_raise_before_planning(parent, fork_request, match):
    with pytest.raises(ValueError, match=match):
        plan_execution_fork(parent, fork_request)
