"""Tests for clozn.replay.rewind_fidelity -- the read-only Rewind Fidelity projection (E10).

No model, no network, no filesystem at all: `build_rewind_fidelity` takes already-loaded canonical
`GeneratedObservation` evidence as a plain argument. The fixtures construct real observations through
the kernel's own identity machinery, so an observation that would not be accepted by the store is not
accepted here either.
"""
from __future__ import annotations

import copy

import pytest

from clozn import schemas
from clozn.experiments.evaluators import Generate
from clozn.experiments.observations import GeneratedObservation, execution_observation_identity
from clozn.experiments.state_ref import ResolvedState, StateRef
from clozn.experiments import execution_facts
from clozn.replay import rewind_fidelity
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


def _observation(run, *, position=1, classification="exact_execution_fork",
                 proof_status="confirmed", control_status="matched", exact_match=True,
                 control_proof=True, status="completed", run_id=None, stale=False,
                 regime="generated_token_live_kv", max_new=2):
    """One canonical GeneratedObservation, built through the kernel's own identity machinery."""
    source = dict(run)
    if stale:
        # A different recorded continuation is a different execution fingerprint, which is exactly
        # what "evidence recorded against a run that has since changed" means. The alternate run is
        # itself well-formed, so this tests staleness rather than malformedness.
        pieces = ["one", " two", " four"]
        source = {**source, "response": "".join(pieces),
                  "trace": {"tokens": pieces, "token_ids": [11, 22, 44]}}
    ref = StateRef.before_answer_token(source, position)
    realization = {"regime": regime, "source": "live_kv" if regime == "generated_token_live_kv" else "reprefill",
                   "runtime_identity": {"runtime_key_sha256": "c" * 64}}
    resolved = ResolvedState(state_ref=ref, classification=classification, proof_status="planned",
                             realization=realization, diagnostics={})
    evaluator = Generate(max_new=max_new)
    identity = execution_observation_identity(resolved, evaluator, None)
    key = identity["observation_key"]
    fidelity = {"classification": classification, "proof_status": proof_status,
                "exact_match": exact_match, "unchanged_control": control_status}
    proof = ({"status": "matched", "result": {"status": "matched", "exact_match": True}}
             if control_proof else {})
    return GeneratedObservation(
        observation_id=identity["observation_id"],
        observation_key_sha256=identity["observation_key_sha256"], observation_key=key,
        run_id=run_id if run_id is not None else resolved.run_id,
        base_execution_fingerprint=resolved.execution_fingerprint,
        evaluator=key["evaluator"], condition=key["condition"], contract=key["contract"],
        status=status, state_ref=ref, realization=resolved.realization, fidelity=fidelity,
        intervention=None, generated_suffix_text=" two three", generated_token_ids=(22, 33),
        execution_provenance={"adapter": "generate"}, runtime_provenance={},
        generation_contract=evaluator.to_dict(), exact_control_proof=proof,
        proof_grade="trusted", trusted=True, diagnostics={})


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
    assert params == {"run", "historical_observations"}


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
    assert reconstructed["unavoidable_differences"] == execution_facts.RECONSTRUCTION_DIFFERENCES


# ======================================================================================================
# 8. Reconstructed supported changes match the canonical planner definition
# ======================================================================================================

def test_reconstructed_supported_changes_match_canonical_planner_definition():
    doc = _validated(_run())
    reconstructed = doc["recorded_capability"]["reconstructed_replay"]
    assert set(reconstructed["supported_change_types"]) == execution_facts.RECONSTRUCTED_CHANGES


# ======================================================================================================
# 9. Exact conditional supported changes match the canonical planner definition
# ======================================================================================================

def test_exact_conditional_supported_changes_match_canonical_planner_definition():
    doc = _validated(_run())
    exact = doc["recorded_capability"]["exact_rewind"]
    assert set(exact["supported_change_types_if_live_plan_succeeds"]) == execution_facts.KNOWN_CHANGES


# ======================================================================================================
# 10. Planned exact fork is NOT proof
# ======================================================================================================

def test_planned_exact_fork_is_not_proof():
    run = _run()
    observation = _observation(run, proof_status="planned", control_status="required_not_run",
                               exact_match=False, control_proof=False)
    doc = _validated(run, historical_observations=[observation])
    assert doc["historical_proof"]["verified_boundaries"] == []


# ======================================================================================================
# 11. Completed exact + matched control -> historically_verified_exact
# ======================================================================================================

def test_completed_exact_with_matched_control_is_verified():
    run = _run()
    observation = _observation(run, position=1)
    doc = _validated(run, historical_observations=[observation])
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
    observation = _observation(run, proof_status="failed", control_status="diverged",
                               exact_match=False, control_proof=False)
    doc = _validated(run, historical_observations=[observation])
    assert doc["historical_proof"]["verified_boundaries"] == []


# ======================================================================================================
# 13 / 14. Failed / cancelled executions -> no verified boundary
# ======================================================================================================

def test_failed_execution_is_not_verified():
    run = _run()
    observation = _observation(run, status="failed", proof_status="failed",
                               control_status="failed", exact_match=False, control_proof=False)
    doc = _validated(run, historical_observations=[observation])
    assert doc["historical_proof"]["verified_boundaries"] == []


def test_cancelled_execution_is_not_verified():
    run = _run()
    observation = _observation(run, classification="reconstructed_replay",
                               proof_status="not_applicable", control_status="not_required",
                               exact_match=False, control_proof=False)
    doc = _validated(run, historical_observations=[observation])
    assert doc["historical_proof"]["verified_boundaries"] == []


# ======================================================================================================
# 15. Wrong parent run -> ignored
# ======================================================================================================

def test_receipt_for_a_different_parent_is_ignored():
    run = _run()
    observation = _observation(run, run_id="someone_else")
    doc = _validated(run, historical_observations=[observation])
    assert doc["historical_proof"]["verified_boundaries"] == []


# ======================================================================================================
# 16. Parent fingerprint mismatch -> stale proof not surfaced
# ======================================================================================================

def test_fingerprint_mismatch_is_not_surfaced_as_verified():
    run = _run()
    observation = _observation(run, stale=True)
    doc = _validated(run, historical_observations=[observation])
    assert doc["historical_proof"]["verified_boundaries"] == []


# ======================================================================================================
# 17. Multiple verified executions at the same position aggregate
# ======================================================================================================

def test_multiple_verified_executions_at_one_position_aggregate():
    run = _run()
    # Distinct budgets are distinct conditions, so these are three separate observations at one
    # boundary. The loader returns newest-first, so the first entry is the latest evidence.
    newest = _observation(run, position=1, max_new=2)
    middle = _observation(run, position=1, max_new=3)
    oldest = _observation(run, position=1, max_new=4)
    doc = _validated(run, historical_observations=[newest, middle, oldest])
    boundaries = doc["historical_proof"]["verified_boundaries"]
    assert len(boundaries) == 1
    assert boundaries[0]["verified_observation_count"] == 3
    assert boundaries[0]["latest_observation_id"] == newest.observation_id


# ======================================================================================================
# 18. Verified executions at multiple positions -> sparse, deterministically ordered
# ======================================================================================================

def test_multiple_positions_produce_a_sparse_deterministically_ordered_list():
    run = _run(tokens=["t"] * 50, token_ids=list(range(50)))
    r_high = _observation(run, position=37)
    r_low = _observation(run, position=2)
    doc = _validated(run, historical_observations=[r_high, r_low])
    positions = [b["position"] for b in doc["historical_proof"]["verified_boundaries"]]
    assert positions == [2, 37]


# ======================================================================================================
# 19. Historical verification never upgrades live state -- critical regression
# ======================================================================================================

def test_historical_verification_never_upgrades_live_state():
    run = _run()
    observation = _observation(run, position=1)
    doc = _validated(run, historical_observations=[observation])
    assert doc["historical_proof"]["verified_boundaries"], "fixture must actually produce proof"
    assert doc["recorded_capability"]["exact_rewind"]["state"] == "requires_live_plan"
    assert doc["live_execution"] == {
        "state": "not_checked", "reason": "read_only_projection", "authority": "exact_state_resolution",
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
    # Cross-check against the canonical exact-resume resolver's own boundary: position 3 must be out
    # of range, 2 must be valid. That resolver, not a fork planner, is the boundary authority now.
    checkpoint = {"checkpoint_id": "c", "worker_generation_id": "g", "state": "available",
                  "parent_run_id": run["id"], "prompt_tokens": 4, "n_past": 12}
    worker = {"worker_id": "w", "worker_generation_id": "g", "protocol_version": "1.1"}
    _plan, in_range = execution_facts.resolve_exact_resume_facts(
        run, position=2, checkpoint=checkpoint, worker_identity=worker, runtime_identity=_identity())
    assert in_range["code"] != "position_out_of_range"
    _plan, out_of_range = execution_facts.resolve_exact_resume_facts(
        run, position=3, checkpoint=checkpoint, worker_identity=worker, runtime_identity=_identity())
    assert out_of_range["code"] == "position_out_of_range"


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


def test_historical_observations_are_never_mutated():
    run = _run()
    observation = _observation(run)
    before = observation.to_json()
    build_rewind_fidelity(run, historical_observations=[observation])
    assert observation.to_json() == before


# ======================================================================================================
# 23. Determinism
# ======================================================================================================

def test_deterministic_output():
    run = _run()
    observation = _observation(run)
    first = build_rewind_fidelity(copy.deepcopy(run), historical_observations=[observation])
    second = build_rewind_fidelity(copy.deepcopy(run), historical_observations=[observation])
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
    doc = _validated(run, historical_observations=[{"not": "a receipt"}, "garbage", None, 42])
    assert doc["historical_proof"]["state"] == "partially_unavailable"
    assert doc["historical_proof"]["rejected_evidence_count"] == 4
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
