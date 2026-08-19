"""Focused contract coverage for the canonical read-only Run diagnostics projection."""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn import schemas
from clozn.experiments.execution_facts import (
    parent_runtime_projection,
    recorded_execution_prerequisites as recorded_fork_prerequisites,
)
from clozn.recipes.context_effects import plan_context_effects
from clozn.recipes.time_travel import time_travel_capabilities
from clozn.runs.run_diagnostics import build_run_diagnostics
from clozn.runs import store as run_store
from clozn.server.routes import run_diagnostics as diagnostics_route


RUNTIME = {
    "model_sha256": "a" * 64,
    "template_fingerprint": "b" * 16,
    "engine_build": "diagnostics-fixture",
    "context_size": 4096,
    "backend": "cpu",
    "adapter": {"present": False, "identity_sha256": None, "artifact_sha256": None, "scale": None},
    "white_box_flags": {},
}


def _meta(*, sampled: bool = False, complete: bool = True) -> dict:
    decode = {"mode": "sample" if sampled else "greedy"}
    if sampled:
        decode.update({"temperature": 0.8, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.0, "seed": 7})
    result = {
        "n_ctx": 4096,
        "device": "cpu",
        "max_tokens": 3,
        "stop": [],
        "finish_reason": "stop",
        "decode": decode,
    }
    if not complete and sampled:
        result["decode"].pop("seed")
    return result


def _recorded_run(tmp_path, monkeypatch, *, sampled: bool = False, complete: bool = True):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run_id = run_store.record(
        source="test", client="diagnostics", model="fixture-model", substrate="fixture",
        messages=[
            {"role": "system", "content": "You are precise."},
            {"role": "user", "content": "Use the context."},
        ],
        assembled_messages=[
            {"role": "system", "content": "You are precise."},
            {"role": "user", "content": "Use the context."},
        ],
        final_prompt="<rendered prompt>", response="answer tail",
        identity=deepcopy(RUNTIME), meta=_meta(sampled=sampled, complete=complete),
        finish_reason="stop",
        trace={
            "tokens": ["answer", " tail"], "token_ids": [10, 11],
            "logprobs": [-0.1, -0.2],
            "alternatives": [[{"token_id": 12, "piece": "wrong", "prob": 0.2}], []],
            "steps": [
                {"token_id": 10, "piece": "answer", "dt_ms": 1.0},
                {"token_id": 11, "piece": " tail", "dt_ms": 1.2},
            ],
        },
    )
    assert run_id
    run = run_store.get_run(run_id)
    assert run
    return run


def test_complete_greedy_projection_is_schema_valid_and_reuses_authorities(tmp_path, monkeypatch):
    run = _recorded_run(tmp_path, monkeypatch)
    document = build_run_diagnostics(run)

    schemas.validate(document, "clozn.run-diagnostics.v1")
    assert document["generation_contract"]["state"] == "available"
    assert document["generation_contract"]["contract"]["decode_mode"] == "greedy"
    assert document["output"]["trace_completeness"]["recorded_fork_prerequisites"] == recorded_fork_prerequisites(run)
    assert document["output"]["pieces_reconstruct_response"]["value"] == "matched"
    assert document["output"]["ids_and_pieces_alignment"]["value"] == "matched"
    assert document["model_runtime"]["normalized"]["value"] == parent_runtime_projection(run)
    assert document["capabilities"]["context_effects"]["measurable_source_count"] == len(
        plan_context_effects(run).measurement_source_ids
    )
    assert document["capabilities"]["remove_and_test"]["state"] == "requires_input"
    assert document["capabilities"]["context_counterfactual_generation"]["state"] == "requires_input"
    assert document["capabilities"]["minimal_context"]["state"] == "requires_verification"


def test_complete_sampled_contract_is_preserved_and_incomplete_sample_never_becomes_greedy(
    tmp_path, monkeypatch,
):
    sampled = _recorded_run(tmp_path, monkeypatch, sampled=True)
    document = build_run_diagnostics(sampled)
    contract = document["generation_contract"]
    assert contract["state"] == "available"
    assert contract["decode_mode"] == "sample"
    assert contract["contract"]["sampling"] == {
        "temperature": 0.8, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.0, "seed": 7,
    }
    assert document["capabilities"]["time_travel"]["projection"]["available_operations"]["continue"]["reason_code"] == "stochastic_execution_unbound"
    assert document["capabilities"]["time_travel"]["projection"]["available_operations"]["force_token"]["reason_code"] == "stochastic_execution_unbound"

    incomplete = _recorded_run(tmp_path, monkeypatch, sampled=True, complete=False)
    incomplete_document = build_run_diagnostics(incomplete)
    assert incomplete_document["generation_contract"]["state"] == "incomplete"
    assert incomplete_document["generation_contract"]["decode_mode"] == "sample"
    assert incomplete_document["generation_contract"]["reason_code"] == "sampled_replay_not_proven"


@pytest.mark.parametrize(
    ("mutation", "expected_state"),
    [
        (lambda run: run.pop("context_receipt", None), "unavailable"),
        (lambda run: run.update(context_receipt={"schema": "clozn.context_receipt.v1", "delivered": {}}), "partial"),
        (lambda run: run.update(context_receipt={"schema_version": "unknown.receipt.v1"}), "malformed"),
    ],
)
def test_receipt_shapes_degrade_input_without_failing_the_run_document(
    tmp_path, monkeypatch, mutation, expected_state,
):
    run = _recorded_run(tmp_path, monkeypatch)
    mutation(run)
    document = build_run_diagnostics(run)
    assert document["input"]["context_receipt"]["state"] == expected_state
    assert document["run_id"] == run["id"]
    schemas.validate(document)


def test_mismatched_output_and_missing_optional_measurements_are_explicit(tmp_path, monkeypatch):
    run = _recorded_run(tmp_path, monkeypatch)
    run["response"] = "different"
    run["trace"].pop("logprobs", None)
    run["trace"].pop("alternatives", None)
    run.pop("timing", None)
    document = build_run_diagnostics(run)
    assert document["output"]["pieces_reconstruct_response"]["value"] == "mismatch"
    assert document["output"]["token_logprobs"]["state"] == "unavailable"
    assert document["output"]["token_alternatives"]["state"] == "unavailable"
    assert document["execution_health"]["state"] == "degraded"
    performance = document["execution_health"]["performance"]
    assert any(item.get("status") == "unavailable" for item in performance.get("diagnoses", []))


def test_malformed_and_contradictory_runtime_fails_closed(tmp_path, monkeypatch):
    run = _recorded_run(tmp_path, monkeypatch)
    run["meta"]["model_routing"] = {"not": "a model-routing receipt"}
    document = build_run_diagnostics(run)
    assert document["model_runtime"]["state"] == "contradictory"
    assert document["model_runtime"]["normalized"]["state"] == "unavailable"
    assert document["model_runtime"]["routing"]["state"] == "contradictory"
    assert document["evidence"]["runtime_identity"]["state"] == "contradictory"


def test_capabilities_are_read_only_and_time_travel_keeps_proof_distinctions(tmp_path, monkeypatch):
    run = _recorded_run(tmp_path, monkeypatch)
    before = len(run_store.list_runs(100))
    first = build_run_diagnostics(run)
    second = build_run_diagnostics(run)
    assert first == second
    assert len(run_store.list_runs(100)) == before

    tt = first["capabilities"]["time_travel"]
    assert tt["projection"]["exact_checkpoint_restore"]["proof_status"] in {"planned", "not_available"}
    assert tt["rewind_fidelity"]["live_execution"]["state"] == "not_checked"
    assert tt["rewind_fidelity"]["historical_proof"]["verified_boundaries"] == []
    assert first["evidence"]["checkpoint_pin"]["state"] == "unavailable"


def test_route_returns_projection_and_404_without_touching_a_worker(tmp_path, monkeypatch):
    run = _recorded_run(tmp_path, monkeypatch)

    class Handler:
        path = ""

        def __init__(self):
            self.response = None

        def _json(self, status, value):
            self.response = (status, value)

    handler = Handler()
    path = f"/runs/{run['id']}/diagnostics"
    assert diagnostics_route.try_get(handler, path) is True
    assert handler.response[0] == 200
    assert handler.response[1]["schema_version"] == "clozn.run-diagnostics.v1"

    missing = Handler()
    assert diagnostics_route.try_get(missing, "/runs/run_missing/diagnostics") is True
    assert missing.response == (404, {"error": "run not found"})


def test_time_travel_capability_projection_matches_the_existing_authority(tmp_path, monkeypatch):
    run = _recorded_run(tmp_path, monkeypatch)
    expected = time_travel_capabilities(run)
    actual = build_run_diagnostics(run)["capabilities"]["time_travel"]["projection"]
    assert actual == expected
