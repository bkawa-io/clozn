"""Branch Fan as an orchestration of canonical ForceToken time-travel experiments.

The invariant every test here defends: fanning N recorded alternatives produces N
GeneratedObservations and ZERO Runs.  A child Run exists only after an explicit
materialization choice, which is a separate operation.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn import schemas
from clozn.experiments.persistence import ObservationStore
from clozn.replay import branch_fan as fan
from clozn.replay import execution_fork_results
from clozn.runs import store as runlog


RUNTIME = {
    "model_sha256": "a" * 64,
    "template_fingerprint": "b" * 16,
    "engine_build": "test-build",
    "context_size": 4096,
    "backend": "cpu",
    "adapter": {"present": False, "identity_sha256": None, "artifact_sha256": None, "scale": None},
    "white_box_flags": {},
}
WORKER = {"worker_id": "worker-a", "worker_generation_id": "generation-a", "protocol_version": "1.1"}

ALTERNATIVES = [
    {"piece": " first", "token_id": 21, "prob": 0.39},
    {"piece": " second", "token_id": 22, "prob": 0.12},
    {"piece": " third", "token_id": 23, "prob": 0.04},
]


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Isolate the Run store (which the observation store shares) and the legacy receipt store.

    The legacy receipt directory is redirected too, so "the fan wrote no ExecutionForkResult" is a
    hermetic claim about this fan rather than a read of the developer's own ~/.clozn.
    """
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(execution_fork_results, "RESULTS_DIR", str(tmp_path / "execution-forks"))
    runlog._schema_verified.clear()
    return tmp_path


def _parent(*, alternatives=None, tokens=None):
    tokens = tokens or ["zero", " committed", " tail"]
    token_ids = [10, 11, 12][: len(tokens)]
    return {
        "id": "run_parent",
        "model": "fixture-model",
        "substrate": "fixture",
        "messages": [{"role": "user", "content": "question"}],
        "assembled_messages": [{"role": "user", "content": "question"}],
        "final_prompt": "<prompt>",
        "response": "".join(tokens),
        "identity": deepcopy(RUNTIME),
        "meta": {"n_ctx": 4096, "device": "cpu"},
        "trace": {
            "tokens": list(tokens),
            "token_ids": token_ids,
            "steps": [{"token_id": tid, "piece": piece} for tid, piece in zip(token_ids, tokens)],
            "alternatives": alternatives if alternatives is not None
            else [[], [deepcopy(item) for item in ALTERNATIVES], []],
        },
    }


def _checkpoint(parent, *, worker_generation_id=None, parent_run_id=None):
    return {
        "checkpoint_id": "checkpoint-1",
        "worker_generation_id": worker_generation_id or WORKER["worker_generation_id"],
        "state": "available",
        "parent_run_id": parent_run_id or parent["id"],
        "prompt_tokens": 8,
        "n_past": 11,
    }


class ReconstructedEngine:
    """A raw-prompt engine: the reconstructed StateRef path's only requirement."""

    def __init__(self):
        self.calls = []

    def complete(self, prompt, **kwargs):
        self.calls.append((prompt, kwargs))
        return {"choices": [{"text": " continuation", "finish_reason": "stop"}]}


PIECE_BY_ID = {item["token_id"]: item["piece"] for item in ALTERNATIVES}


class ExactEngine:
    """A worker exposing the low-level execution_fork RPC and nothing child-creating."""

    def __init__(self, *, control_diverges=False, pieces=None):
        self.calls = []
        self.control_diverges = control_diverges
        self.pieces = dict(pieces) if pieces is not None else dict(PIECE_BY_ID)

    def execution_fork(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        if kwargs["intervention"]["type"] == "none":
            if self.control_diverges:
                tokens, pieces, text = [99], ["wrong"], "wrong"
            else:
                tokens, pieces, text = [11, 12], [" committed", " tail"], " committed tail"
            applied = {"type": "none"}
        else:
            forced = kwargs["intervention"]["token_id"]
            piece = self.pieces[forced]
            tokens, pieces, text = [forced, 12], [piece, " tail"], piece + " tail"
            applied = {"type": "force_token", "token_id": forced}
        return {
            "worker_generation_id": WORKER["worker_generation_id"],
            "text": text,
            "tokens": tokens,
            "token_pieces": pieces,
            "steps": [{"token_id": tid, "piece": piece} for tid, piece in zip(tokens, pieces)],
            "restore_mode": "live_kv_truncated",
            "n_past_restored": 9,
            "exactness": {"source": "live_kv", "boundary_shape_true": True},
            "intervention_applied": applied,
            "finish_reason": "stop",
            "sampler_state_preserved": True,
        }


class Sub:
    def __init__(self, engine=None):
        self.engine = engine if engine is not None else ReconstructedEngine()
        self.runtime_identity = deepcopy(RUNTIME)
        self.worker_identity = deepcopy(WORKER)


# ------------------------------------------------------------------ the observation-first shape
def test_fanning_three_recorded_alternatives_creates_three_observations_and_no_runs(isolated_store):
    parent = _parent()
    store = ObservationStore()
    result = fan.branch_fan(parent, Sub(), 1, limit=3, runtime_identity=RUNTIME,
                            worker_identity=WORKER, observation_store=store)

    schemas.validate(result, "clozn.branch-fan.v2")
    assert result["summary"]["observations_completed"] == 3
    assert result["summary"]["status"] == "completed"
    observation_ids = [branch["observation_id"] for branch in result["branches"]]
    assert len(set(observation_ids)) == 3
    for branch in result["branches"]:
        assert branch["state"] == "completed"
        assert store.get_observation(branch["observation_id"]).status == "completed"
    # The whole point of the conversion: no Run is created by generating alternatives.
    assert runlog.list_runs(20) == []
    assert all("child_run_id" not in branch for branch in result["branches"])


def test_candidate_selection_preserves_recorded_order_and_filters_before_limit(isolated_store):
    alternatives = [[], [
        {"piece": " committed", "token_id": 99, "prob": 0.8},      # the committed token itself
        {"piece": " first", "token_id": 21, "prob": 0.39},
        {"piece": " first", "token_id": 21, "prob": 0.38},         # duplicate numeric id
        {"piece": "", "token_id": 24, "prob": 0.2},                # empty piece
        {"piece": " second", "prob": 0.12},
        {"piece": " second", "prob": 0.11},                        # duplicate piece
        "malformed",
        {"piece": " third", "token_id": 23, "prob": 0.04},
    ], []]
    parent = _parent(alternatives=alternatives)
    result = fan.branch_fan(parent, Sub(), 1, limit=3, runtime_identity=RUNTIME, worker_identity=WORKER)

    assert [branch["recorded_alternative"]["rank"] for branch in result["branches"]] == [1, 4, 7]
    assert result["selection"]["recorded_alternatives"] == 8
    assert result["summary"]["observations_completed"] == 3
    # A branch identifies its alternative by rank/id/probability, never by its piece text.
    assert all("piece" not in branch["recorded_alternative"] for branch in result["branches"])
    schemas.validate(result, "clozn.branch-fan.v2")


@pytest.mark.parametrize("limit", [1, 2, 3, 4])
def test_limit_is_bounded_and_applied_after_selection(isolated_store, limit):
    result = fan.branch_fan(_parent(), Sub(), 1, limit=limit,
                            runtime_identity=RUNTIME, worker_identity=WORKER)
    assert len(result["branches"]) == min(limit, 3)
    assert result["summary"]["requested"] == min(limit, 3)


@pytest.mark.parametrize("limit", [0, 5, True, "3"])
def test_invalid_limit_is_typed(limit):
    with pytest.raises(fan.BranchFanInputError) as exc:
        fan.branch_fan(_parent(), Sub(), 1, limit=limit)
    assert exc.value.code == "invalid_limit"


@pytest.mark.parametrize("position", [-1, 3, True, "1"])
def test_invalid_position_is_typed(position):
    with pytest.raises(fan.BranchFanInputError) as exc:
        fan.branch_fan(_parent(), Sub(), position)
    assert exc.value.code == "invalid_position"


def test_no_recorded_alternatives_is_unavailable_without_model_work(isolated_store):
    sub = Sub()
    result = fan.branch_fan(_parent(alternatives=[[], [], []]), sub, 1,
                            runtime_identity=RUNTIME, worker_identity=WORKER)
    assert result["summary"]["status"] == "unavailable"
    assert result["selection"]["reason"] == "no_recorded_alternatives"
    assert result["execution"]["checkpoint_capture"]["state"] == "not_attempted"
    assert sub.engine.calls == []
    assert runlog.list_runs(20) == []


# ------------------------------------------------------------------------- exact / reconstructed
def test_exact_alternative_completes_without_a_child_run_or_legacy_receipt(isolated_store):
    parent = _parent()
    sub = Sub(ExactEngine())
    store = ObservationStore()
    result = fan.branch_fan(parent, sub, 1, limit=2, runtime_identity=RUNTIME, worker_identity=WORKER,
                            checkpoint=_checkpoint(parent), observation_store=store)

    schemas.validate(result, "clozn.branch-fan.v2")
    assert [branch["outcome"] for branch in result["branches"]] == ["exact", "exact"]
    assert result["execution"]["fidelity"] == "all_exact"
    assert result["summary"]["exact_observations"] == 2
    for branch in result["branches"]:
        assert branch["resolution_policy"] == "exact_preferred"
        assert branch["fidelity"]["classification"] == "exact_execution_fork"
        assert branch["fidelity"]["proof_status"] == "confirmed"
        assert store.get_observation(branch["observation_id"]).status == "completed"
    assert runlog.list_runs(20) == []
    # The retired executor was the only writer of these receipts; the kernel writes none.
    assert execution_fork_results.list_for_parent(parent["id"]) == []
    # The unchanged control is proven once for the whole fan, not per candidate.
    control_calls = [call for call in sub.engine.calls if call["intervention"]["type"] == "none"]
    assert len(control_calls) == 1


def test_reconstructed_alternative_completes_without_a_child_run(isolated_store):
    parent = _parent()
    sub = Sub()
    result = fan.branch_fan(parent, sub, 1, limit=1, runtime_identity=RUNTIME, worker_identity=WORKER)

    branch = result["branches"][0]
    assert branch["state"] == "completed"
    assert branch["outcome"] == "reconstructed"
    assert branch["resolution_policy"] == "reconstructed_only"
    assert branch["fidelity"]["classification"] == "reconstructed_replay"
    assert result["execution"]["fidelity"] == "all_reconstructed"
    assert runlog.list_runs(20) == []


def test_alternative_without_a_recorded_id_reconstructs_beside_an_exact_one(isolated_store):
    parent = _parent(alternatives=[[], [
        {"piece": " exact", "token_id": 21, "prob": 0.4},
        {"piece": " no id", "prob": 0.2},
    ], []])

    class BothEngine(ExactEngine):
        def complete(self, prompt, **kwargs):
            return {"choices": [{"text": " continuation", "finish_reason": "stop"}]}

    engine = BothEngine(pieces={21: " exact"})
    result = fan.branch_fan(parent, Sub(engine), 1, limit=2, runtime_identity=RUNTIME,
                            worker_identity=WORKER, checkpoint=_checkpoint(parent))
    assert [branch["outcome"] for branch in result["branches"]] == ["exact", "reconstructed"]
    assert [branch["resolution_policy"] for branch in result["branches"]] == [
        "exact_preferred", "reconstructed_only"]
    assert result["execution"]["fidelity"] == "mixed"
    assert runlog.list_runs(20) == []


def test_unavailable_alternative_is_typed_and_creates_no_run(isolated_store):
    # No rendered prompt: reconstruction has nothing honest to replay from.
    parent = _parent()
    parent.pop("final_prompt")
    sub = Sub()
    result = fan.branch_fan(parent, sub, 1, limit=1, runtime_identity=RUNTIME, worker_identity=WORKER)

    branch = result["branches"][0]
    assert branch["state"] == "unavailable"
    assert branch["outcome"] == "unavailable"
    assert branch["reasons"][0]["code"] == "reconstruction_prompt_unavailable"
    assert branch["comparison"] is None
    assert result["summary"]["unavailable"] == 1
    assert result["summary"]["status"] == "unavailable"
    assert sub.engine.calls == []
    assert runlog.list_runs(20) == []


def test_stale_supplied_exact_state_is_refused_without_reconstructed_fallback(isolated_store):
    parent = _parent()
    sub = Sub(ExactEngine())
    stale = _checkpoint(parent, worker_generation_id="a-different-generation")
    result = fan.branch_fan(parent, sub, 1, limit=3, runtime_identity=RUNTIME,
                            worker_identity=WORKER, checkpoint=stale)

    first = result["branches"][0]
    assert first["state"] == "unavailable"
    assert first["outcome"] == "unavailable"
    assert first["resolution_policy"] == "exact_preferred"
    assert first["reasons"][0]["code"] == "stale_worker_generation"
    # The refusal is terminal: nothing quietly re-runs as reconstructed text.
    assert all(branch["outcome"] != "reconstructed" for branch in result["branches"])
    assert result["summary"]["reconstructed_observations"] == 0
    assert result["summary"]["observations_completed"] == 0
    assert [branch["state"] for branch in result["branches"][1:]] == ["not_attempted", "not_attempted"]
    assert sub.engine.calls == []
    assert runlog.list_runs(20) == []


def test_contradictory_supplied_checkpoint_parent_is_refused(isolated_store):
    parent = _parent()
    sub = Sub(ExactEngine())
    result = fan.branch_fan(parent, sub, 1, limit=1, runtime_identity=RUNTIME, worker_identity=WORKER,
                            checkpoint=_checkpoint(parent, parent_run_id="some-other-run"))
    assert result["branches"][0]["reasons"][0]["code"] == "checkpoint_parent_mismatch"
    assert result["summary"]["observations_completed"] == 0
    assert sub.engine.calls == []


def test_contradictory_recorded_alternative_evidence_is_refused(isolated_store):
    """The same recorded id carrying two different pieces is contradictory, not a what-if."""
    parent = _parent(alternatives=[[], [
        {"piece": " first", "token_id": 21, "prob": 0.39},
        {"piece": " conflicting", "token_id": 21, "prob": 0.38},
    ], []])
    result = fan.branch_fan(parent, Sub(), 1, limit=1, runtime_identity=RUNTIME, worker_identity=WORKER)
    branch = result["branches"][0]
    assert branch["state"] == "unavailable"
    assert branch["reasons"][0]["code"] == "force_token_mismatch"
    assert runlog.list_runs(20) == []


def test_diverged_exact_control_stops_scheduling_later_branches(isolated_store):
    parent = _parent()
    sub = Sub(ExactEngine(control_diverges=True))
    result = fan.branch_fan(parent, sub, 1, limit=3, runtime_identity=RUNTIME, worker_identity=WORKER,
                            checkpoint=_checkpoint(parent))
    # A refused control is not persisted, so the branch reports the fact it can prove -- the
    # unchanged exact control was unavailable -- rather than claiming a specific mismatch.
    assert result["branches"][0]["reasons"][0]["code"] == "exact_control_unavailable"
    assert [branch["state"] for branch in result["branches"][1:]] == ["not_attempted", "not_attempted"]
    assert result["summary"]["status"] == "unavailable"
    assert runlog.list_runs(20) == []


# ---------------------------------------------------------------------------------- cancellation
def test_cancellation_before_execution_attempts_nothing(isolated_store):
    sub = Sub()
    result = fan.branch_fan(_parent(), sub, 1, limit=3, runtime_identity=RUNTIME,
                            worker_identity=WORKER, cancel_check=lambda: True)
    assert result["summary"]["status"] == "cancelled"
    assert result["summary"]["not_attempted"] == 3
    assert all(branch["reasons"][0]["code"] == "branch_fan_cancelled" for branch in result["branches"])
    assert sub.engine.calls == []


def test_cancellation_preserves_completed_observations_and_marks_the_rest(isolated_store):
    parent = _parent()
    sub = Sub()
    # Cancel once the first branch has produced its control and its own observation.
    result = fan.branch_fan(parent, sub, 1, limit=3, runtime_identity=RUNTIME,
                            worker_identity=WORKER,
                            cancel_check=lambda: len(sub.engine.calls) >= 2)
    assert result["summary"]["status"] == "partial_cancelled"
    assert result["summary"]["observations_completed"] == 1
    assert result["summary"]["not_attempted"] == 2
    assert all(branch["reasons"][0]["code"] == "branch_fan_cancelled"
               for branch in result["branches"][1:])
    assert runlog.list_runs(20) == []


# --------------------------------------------------------------------------------- comparison
def test_comparison_is_projected_from_the_observation_not_from_a_temporary_run(isolated_store):
    parent = _parent()
    sub = Sub(ExactEngine())
    result = fan.branch_fan(parent, sub, 1, limit=1, runtime_identity=RUNTIME, worker_identity=WORKER,
                            checkpoint=_checkpoint(parent))
    comparison = result["branches"][0]["comparison"]
    assert comparison["basis"] == "recorded_suffix_vs_generated_suffix"
    assert comparison["state"] == "available"
    assert comparison["branch_point"]["answer_token_index"] == 1
    assert comparison["first_divergence"]["answer_token_index"] == 1
    assert comparison["first_divergence"]["recorded_piece"] == " committed"
    assert comparison["first_divergence"]["generated_piece"] == " first"
    assert comparison["identical_text"] is False
    # No Run was created to reach a two-Run diff.
    assert runlog.list_runs(20) == []


# ------------------------------------------------------------------------------ materialization
def test_materializing_one_fan_observation_creates_exactly_one_child_run(isolated_store):
    from clozn.experiments.materialize import materialize_generated_observation

    parent = _parent()
    sub = Sub(ExactEngine())
    store = ObservationStore()
    result = fan.branch_fan(parent, sub, 1, limit=3, runtime_identity=RUNTIME, worker_identity=WORKER,
                            checkpoint=_checkpoint(parent), observation_store=store)
    assert result["summary"]["observations_completed"] == 3
    assert runlog.list_runs(20) == []

    chosen = result["branches"][1]
    engine_calls = len(sub.engine.calls)
    materialized = materialize_generated_observation(
        parent, chosen["experiment_id"], chosen["arm_id"],
        observation_id=chosen["observation_id"], observation_store=store,
    )

    assert materialized["state"] == "completed"
    # Exactly one child Run, and generation was not re-run to produce it.
    assert len(runlog.list_runs(20)) == 1
    assert len(sub.engine.calls) == engine_calls
    child = runlog.get_run(materialized["child_run_id"])
    assert child["parent_run_id"] == parent["id"]
    lineage = child["changes_applied"]["experiment"]
    assert lineage["experiment_id"] == chosen["experiment_id"]
    assert lineage["arm_id"] == chosen["arm_id"]
    assert lineage["observation_id"] == chosen["observation_id"]
    assert lineage["operation"] == "force_token"
    assert lineage["intervention"]["token_id"] == chosen["recorded_alternative"]["token_id"]
    assert lineage["base_state"]["realized_fidelity"] == "exact_execution_fork"
    assert lineage["base_state"]["position"]["index"] == 1


# ------------------------------------------------------------------------------ safety envelope
def test_branch_execution_never_reaches_the_retired_child_creating_executor(isolated_store, monkeypatch):
    """Branch Fan must not reach the legacy planner, directly or through an adapter."""
    import clozn.replay.execution_fork as execution_fork

    def forbidden(name):
        def _explode(*_args, **_kwargs):
            raise AssertionError(f"Branch Fan called the retired planner seam {name}")
        return _explode

    for name in ("plan_execution_fork", "capture_exact_force_token_context",
                 "plan_exact_force_token", "execute_exact_force_token"):
        monkeypatch.setattr(execution_fork, name, forbidden(name), raising=False)

    parent = _parent()
    result = fan.branch_fan(parent, Sub(ExactEngine()), 1, limit=3, runtime_identity=RUNTIME,
                            worker_identity=WORKER, checkpoint=_checkpoint(parent))
    assert result["summary"]["observations_completed"] == 3
    assert runlog.list_runs(20) == []


def test_branch_fan_module_declares_no_legacy_fork_dependency():
    import clozn.replay.branch_fan as module

    source = open(module.__file__, encoding="utf-8").read()
    for name in ("execution_fork_execute", "plan_execution_fork", "capture_exact_force_token_context",
                 "plan_exact_force_token", "execute_exact_force_token", "runlog.record"):
        assert name not in source, f"branch_fan.py still references {name}"
