"""Tests for clozn.replay.rewind_fidelity -- the read-only Rewind Fidelity projection (E10).

No model, no network, no filesystem outside `tmp_path` (and this file touches none at all --
`build_rewind_fidelity` takes `historical_receipts` as a plain argument). Fixtures mirror
tests/test_execution_fork_planner.py's and tests/test_execution_fork_gateway.py's own conventions so a
reader familiar with those files recognizes these immediately.
"""
from __future__ import annotations

import copy

import pytest

from clozn import schemas
from clozn.replay import execution_fork, rewind_fidelity
from clozn.replay.rewind_fidelity import build_rewind_fidelity


def _identity(*, model="a"):
    return {
        "model_sha256": model * 64,
        "template_fingerprint": "b" * 16,
        "engine_build": "clozn-engine-test",
        "white_box_flags": {},
    }


def _run(*, run_id="run_parent", tokens=("one", " two", " three"), token_ids=(11, 22, 33),
         final_prompt="<prompt>", identity=True, response=None, **over):
    out = {
        "id": run_id,
        "meta": {"n_ctx": 4096, "device": "cpu"},
        "response": response if response is not None else "".join(tokens) if tokens else "",
    }
    if identity:
        out["identity"] = _identity()
    if final_prompt is not None:
        out["final_prompt"] = final_prompt
    if tokens is not None or token_ids is not None:
        trace = {}
        if tokens is not None:
            trace["tokens"] = list(tokens)
        if token_ids is not None:
            trace["token_ids"] = list(token_ids)
        out["trace"] = trace
    out.update(over)
    return out


_RUNTIME_KEY_SHA256 = None  # filled lazily below via execution_fork's own projection


def _receipt(run, *, execution_id, position=1, phase="completed", proof_status="confirmed",
            control_status="matched", exact_match=True, parent_run_id=None, fingerprint=None,
            regime="generated_token_live_kv", ended_ts=2.0):
    parent_run_id = parent_run_id if parent_run_id is not None else run["id"]
    fingerprint = fingerprint if fingerprint is not None else execution_fork.parent_execution_fingerprint(run)
    return {
        "schema_version": "clozn.execution-fork.v1",
        "plan_id": "fork_plan_" + ("0" * 20),
        "execution_id": execution_id,
        "phase": phase,
        "classification": "exact_execution_fork",
        "parent_run_id": parent_run_id,
        "parent_fingerprint_sha256": fingerprint,
        "request": {
            "position": position, "change": {"type": "none"},
            "execution_change": {"type": "none"}, "change_sha256": "b" * 64,
        },
        "identity": {
            "parent_runtime": {
                "runtime_key_sha256": "c" * 64, "model_sha256": "a" * 64,
                "template_fingerprint": "b" * 16, "engine_build": "x", "context_size": 4096,
                "backend": "cpu",
                "adapter": {"present": False, "identity_sha256": None, "artifact_sha256": None,
                           "scale": None},
                "white_box_flags": {},
            },
        },
        "exactness": {
            "regime": regime, "source": "live_kv" if regime == "generated_token_live_kv" else "reprefill",
            "proof_status": proof_status, "truncate_to": 11, "boundary_shape_true": True,
        },
        "unavoidable_differences": [],
        "unchanged_control": {
            "required": True, "status": control_status,
            "result": {"status": control_status if control_status != "required_not_run" else "failed",
                      "exact_match": exact_match},
        },
        "child_lineage": {"parent_run_id": parent_run_id, "source": "fork",
                          "change_sha256": "b" * 64, "receipt_status": "created"},
        "execution": {"status": "succeeded" if phase == "completed" else "control_failed",
                     "started_ts": 1.0, "ended_ts": ended_ts},
        "reasons": [{"code": "execution_succeeded", "message": "ok"}],
    }


def _validated(run, **kwargs) -> dict:
    doc = build_rewind_fidelity(run, **kwargs)
    schemas.validate(doc)
    return doc


# ======================================================================================================
# 1. Fully recorded run -> requires_live_plan, never exact_available
# ======================================================================================================

def test_fully_recorded_run_is_requires_live_plan_not_exact_available():
    doc = _validated(_run())
    exact = doc["recorded_capability"]["exact_rewind"]
    assert exact["state"] == "requires_live_plan"
    assert exact["state"] != "exact_available"
    assert doc["recorded_capability"]["state"] == "available"


# ======================================================================================================
# 2. The pure builder does not care whether a worker exists -- no worker param at all
# ======================================================================================================

def test_builder_has_no_worker_or_engine_parameter():
    import inspect
    params = set(inspect.signature(build_rewind_fidelity).parameters)
    assert params == {"run", "historical_receipts"}


# ======================================================================================================
# 3. Token trace unavailable
# ======================================================================================================

def test_token_trace_unavailable_fails_exact_static_prerequisites():
    doc = _validated(_run(tokens=None, token_ids=None))
    exact = doc["recorded_capability"]["exact_rewind"]
    assert exact["state"] == "static_prerequisites_unavailable"
    assert exact["static_prerequisites"]["token_alignment"] == "unavailable"
    assert "recorded_response_token_trace_unavailable" in exact["reasons"]


# ======================================================================================================
# 4. Token IDs unavailable (pieces exist) -- never retokenized
# ======================================================================================================

def test_token_ids_unavailable_fails_exact_static_prerequisites():
    doc = _validated(_run(tokens=("one", " two"), token_ids=None))
    static = doc["recorded_capability"]["exact_rewind"]["static_prerequisites"]
    assert static["recorded_token_pieces"] == "available"
    assert static["recorded_token_ids"] == "unavailable"
    assert static["token_alignment"] == "unavailable"
    assert doc["recorded_capability"]["exact_rewind"]["state"] == "static_prerequisites_unavailable"


# ======================================================================================================
# 5. Token pieces / IDs misaligned -- fail closed, distinct from #4
# ======================================================================================================

def test_misaligned_token_pieces_and_ids_fail_closed():
    doc = _validated(_run(tokens=("one", " two"), token_ids=(11,)))
    static = doc["recorded_capability"]["exact_rewind"]["static_prerequisites"]
    assert static["recorded_token_pieces"] == "available"
    assert static["recorded_token_ids"] == "available"
    assert static["token_alignment"] == "unavailable"
    assert doc["recorded_capability"]["exact_rewind"]["state"] == "static_prerequisites_unavailable"


# ======================================================================================================
# 6. Runtime identity unavailable
# ======================================================================================================

def test_runtime_identity_unavailable_fails_exact_static_prerequisites():
    doc = _validated(_run(identity=False))
    static = doc["recorded_capability"]["exact_rewind"]["static_prerequisites"]
    assert static["runtime_identity"] == "unavailable"
    assert doc["recorded_capability"]["exact_rewind"]["state"] == "static_prerequisites_unavailable"
    assert "parent_runtime_identity_unavailable" in doc["recorded_capability"]["exact_rewind"]["reasons"]
    # reconstruction does not need runtime identity -- still available
    assert doc["recorded_capability"]["reconstructed_replay"]["state"] == "available"
    assert doc["recorded_capability"]["state"] == "limited"


# ======================================================================================================
# 7. Reconstructed replay -- canonical unavoidable-difference vocabulary
# ======================================================================================================

def test_reconstructed_replay_available_with_canonical_differences():
    doc = _validated(_run())
    reconstructed = doc["recorded_capability"]["reconstructed_replay"]
    assert reconstructed["state"] == "available"
    assert reconstructed["unavoidable_differences"] == execution_fork.RECONSTRUCTION_DIFFERENCES


# ======================================================================================================
# 8. Reconstructed supported changes match the canonical planner definition
# ======================================================================================================

def test_reconstructed_supported_changes_match_canonical_planner_definition():
    doc = _validated(_run())
    reconstructed = doc["recorded_capability"]["reconstructed_replay"]
    assert set(reconstructed["supported_change_types"]) == execution_fork.RECONSTRUCTED_CHANGES


# ======================================================================================================
# 9. Exact conditional supported changes match the canonical planner definition
# ======================================================================================================

def test_exact_conditional_supported_changes_match_canonical_planner_definition():
    doc = _validated(_run())
    exact = doc["recorded_capability"]["exact_rewind"]
    assert set(exact["supported_change_types_if_live_plan_succeeds"]) == execution_fork.KNOWN_CHANGES


# ======================================================================================================
# 10. Planned exact fork is NOT proof
# ======================================================================================================

def test_planned_exact_fork_is_not_proof():
    run = _run()
    receipt = _receipt(run, execution_id="fork_exec_" + "a" * 20, phase="planned",
                       proof_status="planned", control_status="required_not_run", exact_match=False)
    doc = _validated(run, historical_receipts=[receipt])
    assert doc["historical_proof"]["verified_boundaries"] == []


# ======================================================================================================
# 11. Completed exact + matched control -> historically_verified_exact
# ======================================================================================================

def test_completed_exact_with_matched_control_is_verified():
    run = _run()
    receipt = _receipt(run, execution_id="fork_exec_" + "a" * 20, position=1)
    doc = _validated(run, historical_receipts=[receipt])
    boundaries = doc["historical_proof"]["verified_boundaries"]
    assert len(boundaries) == 1
    assert boundaries[0]["position"] == 1
    assert boundaries[0]["state"] == "historically_verified_exact"
    assert boundaries[0]["proof"] == {
        "proof_status": "confirmed", "unchanged_control_status": "matched", "exact_match": True,
    }


# ======================================================================================================
# 12. Completed exact + diverged control -> no verified boundary
# ======================================================================================================

def test_diverged_control_is_not_verified():
    run = _run()
    receipt = _receipt(run, execution_id="fork_exec_" + "a" * 20, phase="failed",
                       proof_status="failed", control_status="diverged", exact_match=False)
    doc = _validated(run, historical_receipts=[receipt])
    assert doc["historical_proof"]["verified_boundaries"] == []


# ======================================================================================================
# 13 / 14. Failed / cancelled executions -> no verified boundary
# ======================================================================================================

def test_failed_execution_is_not_verified():
    run = _run()
    receipt = _receipt(run, execution_id="fork_exec_" + "a" * 20, phase="failed",
                       proof_status="failed", control_status="failed", exact_match=False)
    doc = _validated(run, historical_receipts=[receipt])
    assert doc["historical_proof"]["verified_boundaries"] == []


def test_cancelled_execution_is_not_verified():
    run = _run()
    receipt = _receipt(run, execution_id="fork_exec_" + "a" * 20, phase="cancelled",
                       proof_status="failed", control_status="cancelled", exact_match=False)
    doc = _validated(run, historical_receipts=[receipt])
    assert doc["historical_proof"]["verified_boundaries"] == []


# ======================================================================================================
# 15. Wrong parent run -> ignored
# ======================================================================================================

def test_receipt_for_a_different_parent_is_ignored():
    run = _run()
    receipt = _receipt(run, execution_id="fork_exec_" + "a" * 20, parent_run_id="someone_else")
    doc = _validated(run, historical_receipts=[receipt])
    assert doc["historical_proof"]["verified_boundaries"] == []


# ======================================================================================================
# 16. Parent fingerprint mismatch -> stale proof not surfaced
# ======================================================================================================

def test_fingerprint_mismatch_is_not_surfaced_as_verified():
    run = _run()
    receipt = _receipt(run, execution_id="fork_exec_" + "a" * 20, fingerprint="f" * 64)
    doc = _validated(run, historical_receipts=[receipt])
    assert doc["historical_proof"]["verified_boundaries"] == []


# ======================================================================================================
# 17. Multiple verified executions at the same position aggregate
# ======================================================================================================

def test_multiple_verified_executions_at_one_position_aggregate():
    run = _run()
    r1 = _receipt(run, execution_id="fork_exec_" + "a" * 20, position=1, ended_ts=1.0)
    r2 = _receipt(run, execution_id="fork_exec_" + "b" * 20, position=1, ended_ts=2.0)
    r3 = _receipt(run, execution_id="fork_exec_" + "c" * 20, position=1, ended_ts=1.5)
    doc = _validated(run, historical_receipts=[r1, r2, r3])
    boundaries = doc["historical_proof"]["verified_boundaries"]
    assert len(boundaries) == 1
    assert boundaries[0]["verified_execution_count"] == 3
    assert boundaries[0]["latest_execution_id"] == "fork_exec_" + "b" * 20  # highest ended_ts


# ======================================================================================================
# 18. Verified executions at multiple positions -> sparse, deterministically ordered
# ======================================================================================================

def test_multiple_positions_produce_a_sparse_deterministically_ordered_list():
    run = _run(tokens=["t"] * 50, token_ids=list(range(50)))
    r_high = _receipt(run, execution_id="fork_exec_" + "a" * 20, position=37)
    r_low = _receipt(run, execution_id="fork_exec_" + "b" * 20, position=2)
    doc = _validated(run, historical_receipts=[r_high, r_low])
    positions = [b["position"] for b in doc["historical_proof"]["verified_boundaries"]]
    assert positions == [2, 37]


# ======================================================================================================
# 19. Historical verification never upgrades live state -- critical regression
# ======================================================================================================

def test_historical_verification_never_upgrades_live_state():
    run = _run()
    receipt = _receipt(run, execution_id="fork_exec_" + "a" * 20, position=1)
    doc = _validated(run, historical_receipts=[receipt])
    assert doc["historical_proof"]["verified_boundaries"], "fixture must actually produce proof"
    assert doc["recorded_capability"]["exact_rewind"]["state"] == "requires_live_plan"
    assert doc["live_execution"] == {
        "state": "not_checked", "reason": "read_only_projection", "authority": "execution_fork_plan",
    }


# ======================================================================================================
# 20. Coordinate range derived from the same recorded-token contract as the planner
# ======================================================================================================

def test_coordinate_range_matches_the_planners_own_token_boundary_contract():
    run = _run(tokens=("one", " two", " three"), token_ids=(11, 22, 33))
    doc = _validated(run)
    assert doc["coordinates"] == {
        "kind": "recorded_response_token_boundary", "index_base": 0,
        "start": 0, "end_exclusive": 3, "recorded_token_count": 3,
    }
    # cross-check against the planner's own boundary: position 3 must be out of range, 2 must be valid
    in_range = execution_fork.plan_execution_fork(
        run, {"position": 2, "change": {"type": "none"}},
        worker_identity={"worker_id": "w", "worker_generation_id": "g", "protocol_version": "1.1"},
        runtime_identity=_identity(),
    )
    assert in_range["reasons"][0]["code"] != "position_out_of_range"
    out_of_range = execution_fork.plan_execution_fork(
        run, {"position": 3, "change": {"type": "none"}},
        worker_identity={"worker_id": "w", "worker_generation_id": "g", "protocol_version": "1.1"},
        runtime_identity=_identity(),
    )
    assert out_of_range["reasons"][0]["code"] == "position_out_of_range"


def test_coordinates_omitted_when_no_valid_token_boundary_exists():
    doc = _validated(_run(tokens=None, token_ids=None))
    assert "coordinates" not in doc


# ======================================================================================================
# 21 / 22. Run and receipt immutability
# ======================================================================================================

def test_run_is_never_mutated():
    run = _run()
    before = copy.deepcopy(run)
    build_rewind_fidelity(run)
    assert run == before


def test_historical_receipts_are_never_mutated():
    run = _run()
    receipt = _receipt(run, execution_id="fork_exec_" + "a" * 20)
    before = copy.deepcopy(receipt)
    build_rewind_fidelity(run, historical_receipts=[receipt])
    assert receipt == before


# ======================================================================================================
# 23. Determinism
# ======================================================================================================

def test_deterministic_output():
    run = _run()
    receipt = _receipt(run, execution_id="fork_exec_" + "a" * 20)
    first = build_rewind_fidelity(copy.deepcopy(run), historical_receipts=[copy.deepcopy(receipt)])
    second = build_rewind_fidelity(copy.deepcopy(run), historical_receipts=[copy.deepcopy(receipt)])
    assert first == second


# ======================================================================================================
# 24. No engine/model/worker access
# ======================================================================================================

def test_no_engine_or_worker_access(monkeypatch):
    def _explode(*_a, **_kw):
        raise AssertionError("rewind_fidelity touched an engine/model/worker seam")

    from clozn.server import app as ctx
    monkeypatch.setattr(ctx, "active_engine", _explode, raising=False)
    monkeypatch.setattr(ctx, "ENGINE", None, raising=False)
    import clozn.server.model_routing as model_routing
    monkeypatch.setattr(model_routing, "select_control_model_for_run", _explode, raising=False)

    doc = _validated(_run())
    assert doc["recorded_capability"]["state"] == "available"


# ======================================================================================================
# Malformed historical receipts degrade gracefully rather than raising or fabricating proof
# ======================================================================================================

def test_malformed_historical_receipts_degrade_state_without_raising():
    run = _run()
    doc = _validated(run, historical_receipts=[{"not": "a receipt"}, "garbage", None, 42])
    assert doc["historical_proof"]["state"] == "partially_unavailable"
    assert doc["historical_proof"]["malformed_receipt_count"] == 4
    assert doc["historical_proof"]["verified_boundaries"] == []


# ======================================================================================================
# Input validation
# ======================================================================================================

def test_requires_a_run_id():
    with pytest.raises(ValueError):
        build_rewind_fidelity({"response": "x"})


def test_non_dict_run_degrades_rather_than_raising_unexpectedly():
    with pytest.raises(ValueError):
        build_rewind_fidelity(None)
