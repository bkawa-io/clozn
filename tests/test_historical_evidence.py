"""Canonical historical exact-resume evidence over persisted GeneratedObservations.

The invariants defended here are the ones that make this a real convergence rather than a rename:
an observation counts WITHOUT materialization, only genuinely verified evidence counts, and the
three read-only surfaces that show historical proof all agree because they share one projection.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn.experiments.evaluators import Generate
from clozn.experiments.generation import GenerateExecutionAdapter
from clozn.experiments.historical_evidence import (
    is_verified_exact, load_exact_evidence, verified_exact_boundaries,
)
from clozn.experiments.kernel import Experiment
from clozn.experiments.observations import GeneratedObservation, execution_observation_identity
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.runner import run_experiment
from clozn.experiments.state import ExecutionState
from clozn.experiments.state_ref import ResolvedState, StateRef, resolve_state
from clozn.runs import store as runlog


RUNTIME = {
    "model_sha256": "a" * 64,
    "template_fingerprint": "b" * 16,
    "engine_build": "evidence-fixture",
    "context_size": 4096,
    "backend": "cpu",
    "adapter": {"present": False, "identity_sha256": None, "artifact_sha256": None, "scale": None},
    "white_box_flags": {},
}
WORKER = {"worker_id": "worker-a", "worker_generation_id": "generation-a", "protocol_version": "1.1"}


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    runlog._schema_verified.clear()
    return tmp_path


def _run(run_id="run_evidence", pieces=("zero", " one", " two"), token_ids=(10, 11, 12)):
    return {
        "id": run_id,
        "model": "fixture-model",
        "substrate": "fixture",
        "messages": [{"role": "user", "content": "question"}],
        "assembled_messages": [{"role": "user", "content": "question"}],
        "final_prompt": "<prompt>",
        "response": "".join(pieces),
        "identity": deepcopy(RUNTIME),
        "meta": {"n_ctx": 4096, "device": "cpu"},
        "trace": {
            "tokens": list(pieces),
            "token_ids": list(token_ids),
            "steps": [{"token_id": t, "piece": p} for t, p in zip(token_ids, pieces)],
        },
    }


def _checkpoint(run):
    return {
        "checkpoint_id": "checkpoint-1",
        "worker_generation_id": WORKER["worker_generation_id"],
        "state": "available",
        "parent_run_id": run["id"],
        "prompt_tokens": 8,
        "n_past": 11,
    }


class ExactEngine:
    """A worker exposing only the low-level exact-resume RPC."""

    def __init__(self):
        self.calls = []

    def execution_fork(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        tokens, pieces = [11, 12], [" one", " two"]
        return {
            "worker_generation_id": WORKER["worker_generation_id"],
            "text": "".join(pieces), "tokens": tokens, "token_pieces": pieces,
            "steps": [{"token_id": t, "piece": p} for t, p in zip(tokens, pieces)],
            "restore_mode": "live_kv_truncated", "n_past_restored": 9,
            "exactness": {"source": "live_kv", "boundary_shape_true": True},
            "intervention_applied": {"type": "none"}, "finish_reason": "stop",
        }


class Sub:
    def __init__(self, engine=None):
        self.engine = engine if engine is not None else ExactEngine()
        self.runtime_identity = deepcopy(RUNTIME)
        self.worker_identity = deepcopy(WORKER)


def _observation(run, *, position=1, classification="exact_execution_fork",
                 proof_status="confirmed", control_status="matched", exact_match=True,
                 control_proof=True, status="completed", run_id=None, max_new=2):
    """One canonical observation built through the kernel's own identity machinery."""
    ref = StateRef.before_answer_token(run, position)
    realization = {"regime": "generated_token_live_kv", "source": "live_kv",
                   "runtime_identity": {"runtime_key_sha256": "c" * 64}}
    resolved = ResolvedState(state_ref=ref, classification=classification, proof_status="planned",
                             realization=realization, diagnostics={})
    evaluator = Generate(max_new=max_new)
    identity = execution_observation_identity(resolved, evaluator, None)
    key = identity["observation_key"]
    return GeneratedObservation(
        observation_id=identity["observation_id"],
        observation_key_sha256=identity["observation_key_sha256"], observation_key=key,
        run_id=run_id if run_id is not None else resolved.run_id,
        base_execution_fingerprint=resolved.execution_fingerprint,
        evaluator=key["evaluator"], condition=key["condition"], contract=key["contract"],
        status=status, state_ref=ref, realization=resolved.realization,
        fidelity={"classification": classification, "proof_status": proof_status,
                  "exact_match": exact_match, "unchanged_control": control_status},
        intervention=None, generated_suffix_text=" one two", generated_token_ids=(11, 12),
        execution_provenance={"adapter": "generate"}, runtime_provenance={},
        generation_contract=evaluator.to_dict(),
        exact_control_proof=({"status": "matched", "result": {"status": "matched", "exact_match": True}}
                             if control_proof else {}),
        proof_grade="trusted", trusted=True, diagnostics={})


def _fingerprint(run):
    return ExecutionState.from_run(dict(run)).execution_fingerprint


# ------------------------------------------------------------------ the definition, fail-closed
def test_completed_exact_control_confirmed_observation_is_verified():
    run = _run()
    assert is_verified_exact(_observation(run), run=run, run_id=run["id"],
                             fingerprint=_fingerprint(run)) is True


@pytest.mark.parametrize("kwargs", [
    {"status": "failed"},
    {"classification": "reconstructed_replay"},
    {"proof_status": "not_applicable"},
    {"exact_match": False},
    {"control_status": "diverged"},
    {"control_proof": False},
    {"run_id": "a-different-run"},
])
def test_evidence_failing_any_single_condition_never_counts(kwargs):
    run = _run()
    observation = _observation(run, **kwargs)
    assert is_verified_exact(observation, run=run, run_id=run["id"],
                             fingerprint=_fingerprint(run)) is False
    assert verified_exact_boundaries(run, [observation])["verified_boundaries"] == []


def test_evidence_recorded_against_a_changed_run_is_stale_and_never_counts():
    recorded = _run()
    observation = _observation(recorded)
    changed = _run(pieces=("zero", " one", " different"), token_ids=(10, 11, 99))
    assert verified_exact_boundaries(changed, [observation])["verified_boundaries"] == []


def test_non_canonical_evidence_is_rejected_and_counted_not_reinterpreted():
    run = _run()
    projection = verified_exact_boundaries(run, [{"phase": "completed"}, "nonsense"])
    assert projection["state"] == "partially_unavailable"
    assert projection["rejected_evidence_count"] == 2
    assert projection["verified_boundaries"] == []


def test_verified_boundaries_group_by_position_and_are_sparsely_ordered():
    run = _run(pieces=tuple(f"t{i}" for i in range(6)), token_ids=tuple(range(6)))
    evidence = [_observation(run, position=4), _observation(run, position=1),
                _observation(run, position=1, max_new=3)]
    boundaries = verified_exact_boundaries(run, evidence)["verified_boundaries"]
    assert [item["position"] for item in boundaries] == [1, 4]
    assert boundaries[0]["verified_observation_count"] == 2
    assert boundaries[0]["latest_observation_id"] == evidence[1].observation_id
    assert boundaries[0]["proof"] == {
        "proof_status": "confirmed", "unchanged_control_status": "matched", "exact_match": True,
    }


# ------------------------------------------------------------- materialization is NOT required
def test_an_observation_without_materialization_is_historical_exact_evidence(isolated_store):
    """The load-bearing invariant: generation counts as proof without ever branching."""
    from clozn.replay.rewind_fidelity import build_rewind_fidelity

    run = _run()
    resolved = resolve_state(StateRef.before_answer_token(run, 1), run=run, policy="exact_required",
                             checkpoint=_checkpoint(run), runtime_identity=RUNTIME,
                             worker_identity=WORKER)
    sub = Sub()
    store = ObservationStore()
    result = run_experiment(Experiment(base=resolved, evaluator=Generate(max_new=2), arms=[]),
                            GenerateExecutionAdapter(sub, run=run), observation_store=store)
    assert result.control.fidelity["proof_status"] == "confirmed"
    assert runlog.list_runs(20) == []          # nothing was materialized

    evidence = load_exact_evidence(run["id"], observation_store=store)
    assert evidence, "the completed control must be readable as canonical evidence"
    document = build_rewind_fidelity(run, historical_observations=evidence)
    boundaries = document["historical_proof"]["verified_boundaries"]
    assert len(boundaries) == 1
    assert boundaries[0]["position"] == 1
    assert boundaries[0]["state"] == "historically_verified_exact"
    # Still no Run: proof did not require one, and reading it did not create one.
    assert runlog.list_runs(20) == []


def test_historical_proof_never_implies_live_exact_availability(isolated_store):
    from clozn.replay.rewind_fidelity import build_rewind_fidelity

    run = _run()
    document = build_rewind_fidelity(run, historical_observations=[_observation(run)])
    assert document["historical_proof"]["verified_boundaries"], "fixture must produce proof"
    assert document["recorded_capability"]["exact_rewind"]["state"] == "requires_live_plan"
    assert document["live_execution"] == {
        "state": "not_checked", "reason": "read_only_projection",
        "authority": "exact_state_resolution",
    }


# ------------------------------------------------------- the three surfaces agree, and stay read-only
def test_rewind_diagnostics_and_turn_receipt_agree_on_verified_boundaries(isolated_store):
    from clozn.replay.rewind_fidelity import build_rewind_fidelity
    from clozn.runs.run_diagnostics import build_run_diagnostics
    from clozn.runs.turn_receipt import build_turn_receipt

    run = _run()
    evidence = [_observation(run, position=1), _observation(run, position=2)]

    rewind = build_rewind_fidelity(run, historical_observations=evidence)
    diagnostics = build_run_diagnostics(run, historical_observations=evidence)
    receipt = build_turn_receipt(run, historical_observations=evidence)

    expected = [item["position"] for item in rewind["historical_proof"]["verified_boundaries"]]
    assert expected == [1, 2]
    diagnostics_rewind = diagnostics["capabilities"]["time_travel"]["rewind_fidelity"]
    assert [item["position"] for item in
            diagnostics_rewind["historical_proof"]["verified_boundaries"]] == expected
    # Diagnostics counts VERIFIED proofs, not whatever evidence happened to load.
    assert diagnostics["evidence"]["historical_exact_proofs"]["count"] == len(expected)
    assert receipt["rewind"]["historically_verified_boundaries"] == len(expected)


def test_diagnostics_does_not_count_unverified_evidence_as_proof(isolated_store):
    from clozn.runs.run_diagnostics import build_run_diagnostics

    run = _run()
    unverified = [_observation(run, position=1, control_status="diverged", exact_match=False,
                               control_proof=False)]
    diagnostics = build_run_diagnostics(run, historical_observations=unverified)
    proofs = diagnostics["evidence"]["historical_exact_proofs"]
    assert proofs["count"] == 0
    assert proofs["state"] == "unavailable"
    assert proofs["reason_code"] == "no_verified_historical_exact_proofs"


def test_reading_evidence_makes_no_worker_or_generation_calls(isolated_store, monkeypatch):
    """Every historical read is offline-safe: no worker, no model, no checkpoint, no write."""
    from clozn.replay.rewind_fidelity import build_rewind_fidelity
    from clozn.runs.run_diagnostics import build_run_diagnostics

    def explode(*_args, **_kwargs):
        raise AssertionError("a read-only historical projection reached an execution seam")

    import clozn.experiments.generation as generation
    import clozn.replay.checkpoint_capture as checkpoint_capture
    monkeypatch.setattr(generation, "prove_unchanged_control", explode, raising=False)
    monkeypatch.setattr(checkpoint_capture, "capture_parent_checkpoint", explode, raising=False)

    run = _run()
    evidence = [_observation(run)]
    before = runlog.list_runs(20)
    build_rewind_fidelity(run, historical_observations=evidence)
    build_run_diagnostics(run, historical_observations=evidence)
    load_exact_evidence(run["id"])
    assert runlog.list_runs(20) == before


def test_loading_evidence_for_an_unknown_run_is_empty_and_creates_nothing(isolated_store):
    assert load_exact_evidence("run_that_never_existed") == []
    assert load_exact_evidence("") == []
    assert runlog.list_runs(20) == []
