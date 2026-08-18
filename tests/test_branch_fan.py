"""Model-free Branch Fan orchestration and safety coverage."""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn import schemas
from clozn.replay import branch_fan as fan
from clozn.replay import execution_fork
from clozn.replay.execution_fork_execute import _exact_child_trace
import clozn.runs.store as runlog


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


class Engine:
    def __init__(self, *, exact=False):
        self.exact = exact
        self.calls = []

    def execution_fork(self, **kwargs):
        self.calls.append(("execution_fork", deepcopy(kwargs)))
        return {}

    def complete(self, *_args, **_kwargs):
        self.calls.append(("complete", {}))
        return {"choices": [{"text": " continuation", "finish_reason": "stop"}]}


class Sub:
    def __init__(self, *, exact=False):
        self.engine = Engine(exact=exact)


def _parent(*, alternatives=None, tokens=None):
    tokens = tokens or ["zero", " committed", " tail"]
    return {
        "id": "run_parent",
        "model": "fixture-model",
        "response": "".join(tokens),
        "final_prompt": "<prompt>",
        "trace": {
            "tokens": list(tokens),
            "token_ids": [10, 11, 12],
            "alternatives": alternatives if alternatives is not None else [[], [
                {"piece": " first", "token_id": 21, "prob": 0.39},
                {"piece": " second", "token_id": 22, "prob": 0.12},
                {"piece": " third", "token_id": 23, "prob": 0.04},
            ], []],
        },
    }


def _reconstructed_result(candidate, child_id=None):
    out = {
        "recorded_alternative": {"rank": candidate["recorded_rank"]},
        "state": "completed",
        "outcome": "reconstructed_replay",
        "child_run_id": child_id or f"child_{candidate['recorded_rank']}",
        "exactness": {"regime": "reconstructed_text", "proof_status": "not_applicable"},
        "unavoidable_differences": ["kv_state_not_restored"],
        "reasons": [],
        "comparison": {"state": "trace_unavailable", "first_divergence_view": {"state": "trace_unavailable"}},
    }
    if candidate.get("token_id") is not None:
        out["recorded_alternative"]["token_id"] = candidate["token_id"]
    if candidate.get("probability") is not None:
        out["recorded_alternative"]["probability"] = candidate["probability"]
    return out


def test_candidate_selection_preserves_recorded_order_and_filters_before_limit(monkeypatch):
    alternatives = [[], [
        {"piece": " committed", "token_id": 99, "prob": 0.8},
        {"piece": " first", "token_id": 21, "prob": 0.39},
        {"piece": " duplicate id", "token_id": 21, "prob": 0.38},
        {"piece": "" , "token_id": 24, "prob": 0.2},
        {"piece": " second", "prob": 0.12},
        {"piece": " second", "prob": 0.11},
        "malformed",
        {"piece": " third", "token_id": 23, "prob": 0.04},
    ], []]
    seen = []

    def fake_reconstruction(parent, sub, candidate, position, remaining, runtime, worker):
        seen.append((candidate["recorded_rank"], candidate["piece"], remaining))
        return _reconstructed_result(candidate)

    monkeypatch.setattr(fan, "_run_reconstructed", fake_reconstruction)
    result = fan.branch_fan(_parent(alternatives=alternatives), Sub(), 1, limit=3,
                            runtime_identity=RUNTIME, worker_identity=WORKER)
    assert [rank for rank, *_ in seen] == [1, 4, 7]
    assert [piece for _, piece, _ in seen] == [" first", " second", " third"]
    assert result["selection"]["recorded_alternatives"] == 8
    assert result["summary"]["children_created"] == 3
    assert all("piece" not in branch["recorded_alternative"] for branch in result["branches"])
    schemas.validate(result, "clozn.branch-fan.v1")


@pytest.mark.parametrize("limit", [1, 2, 3, 4])
def test_limit_is_bounded_and_applied_after_selection(monkeypatch, limit):
    monkeypatch.setattr(fan, "_run_reconstructed", lambda p, s, c, *args: _reconstructed_result(c))
    result = fan.branch_fan(_parent(), Sub(), 1, limit=limit,
                            runtime_identity=RUNTIME, worker_identity=WORKER)
    assert len(result["branches"]) == min(limit, 3)
    assert result["summary"]["requested_branches"] == min(limit, 3)


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


def test_committed_token_low_confidence_and_entropy_do_not_create_candidates(monkeypatch):
    alternatives = [[], [{"piece": " committed", "prob": 0.001}], []]
    parent = _parent(alternatives=alternatives)
    parent["trace"]["confidence"] = [0.9, 0.001, 0.9]
    parent["trace"]["topk_entropy"] = [0.1, 99.0, 0.1]
    sub = Sub()
    result = fan.branch_fan(parent, sub, 1, runtime_identity=RUNTIME, worker_identity=WORKER)
    assert result["selection"]["reason"] == "no_recorded_alternatives"
    assert result["branches"] == []
    assert sub.engine.calls == []


def test_no_recorded_alternatives_is_unavailable_without_model_work():
    sub = Sub()
    result = fan.branch_fan(_parent(alternatives=[[], [], []]), sub, 1,
                            runtime_identity=RUNTIME, worker_identity=WORKER)
    assert result["summary"]["status"] == "unavailable"
    assert result["execution"]["checkpoint_capture"]["state"] == "not_attempted"
    assert sub.engine.calls == []


def test_exact_capture_is_once_plans_and_controls_are_per_candidate(monkeypatch):
    parent = _parent()
    sub = Sub(exact=True)
    capture_calls, plan_calls, execute_calls = [], [], []

    monkeypatch.setattr(execution_fork, "capture_exact_force_token_context", lambda *args, **kwargs: (
        capture_calls.append((args, kwargs)) or {
            "status": "available",
            "checkpoint_reference": {
                "checkpoint_id": "checkpoint-1", "worker_generation_id": "generation-a",
                "state": "available", "parent_run_id": parent["id"],
            },
        }))

    def fake_plan(parent_run, request, **kwargs):
        plan_calls.append((request, kwargs))
        return {"classification": "exact_execution_fork", "request": {"change": request["change"]}}

    def fake_execute(parent_run, plan, engine, **kwargs):
        execute_calls.append((plan, kwargs))
        token_id = plan["request"]["change"]["token_id"]
        child = {
            "id": f"child_{token_id}",
            "response": "zero branch tail",
            "trace": {"tokens": ["zero", " branch", " tail"], "token_ids": [10, token_id, 12]},
        }
        receipt = {
            "phase": "completed", "execution_id": f"fork_exec_{token_id:02d}" + "a" * 18,
            "exactness": {"proof_status": "confirmed"},
            "unchanged_control": {"status": "matched"},
        }
        child["execution_fork"] = receipt
        return {"receipt": receipt, "child": child}

    monkeypatch.setattr(execution_fork, "plan_exact_force_token", fake_plan)
    monkeypatch.setattr(execution_fork, "execute_exact_force_token", fake_execute)
    monkeypatch.setattr(fan, "_comparison", lambda parent, child: {
        "state": "available", "common_prefix_len": 1,
        "first_divergence_view": {"state": "available"},
    })
    result = fan.branch_fan(parent, sub, 1, limit=3,
                            runtime_identity=RUNTIME, worker_identity=WORKER)
    assert len(capture_calls) == 1
    assert len(plan_calls) == len(execute_calls) == 3
    assert all(call[1]["checkpoint_reference"]["checkpoint_id"] == "checkpoint-1" for call in plan_calls)
    assert result["execution"]["fidelity"] == "all_exact"
    assert result["summary"]["children_created"] == 3
    assert all(branch["child_run_id"] != parent["id"] for branch in result["branches"])


def test_missing_id_can_reconstruct_alongside_exact_candidate(monkeypatch):
    parent = _parent(alternatives=[[], [
        {"piece": " exact", "token_id": 21, "prob": 0.4},
        {"piece": " no id", "prob": 0.2},
    ], []])
    sub = Sub(exact=True)
    monkeypatch.setattr(execution_fork, "capture_exact_force_token_context", lambda *args, **kwargs: {
        "status": "available",
        "checkpoint_reference": {
            "checkpoint_id": "checkpoint-1", "worker_generation_id": "generation-a",
            "state": "available", "parent_run_id": parent["id"],
        },
    })
    monkeypatch.setattr(execution_fork, "plan_exact_force_token", lambda parent_run, request, **kwargs: {
        "classification": "exact_execution_fork", "request": {"change": request["change"]},
    })
    monkeypatch.setattr(execution_fork, "execute_exact_force_token", lambda *args, **kwargs: {
        "receipt": {
            "phase": "completed", "execution_id": "fork_exec_" + "a" * 20,
            "exactness": {"proof_status": "confirmed"}, "unchanged_control": {"status": "matched"},
        },
        "child": {"id": "exact-child", "response": "x", "trace": {"tokens": ["x"], "token_ids": [21]}},
    })
    monkeypatch.setattr(fan, "_comparison", lambda *_: {
        "state": "trace_unavailable", "first_divergence_view": {"state": "trace_unavailable"},
    })
    monkeypatch.setattr(fan, "_run_reconstructed", lambda p, s, c, *args: _reconstructed_result(c, "recon-child"))
    # Re-run with the reconstruction seam patched; the first candidate remains exact and the second
    # candidate is allowed to degrade independently.
    result = fan.branch_fan(parent, sub, 1, limit=2,
                            runtime_identity=RUNTIME, worker_identity=WORKER)
    assert [branch["outcome"] for branch in result["branches"]] == [
        "exact_execution_fork", "reconstructed_replay"]
    assert result["execution"]["fidelity"] == "mixed"


def test_cancellation_preserves_completed_children_and_marks_remaining(monkeypatch):
    parent = _parent()
    sub = Sub(exact=True)
    monkeypatch.setattr(execution_fork, "capture_exact_force_token_context", lambda *args, **kwargs: {
        "status": "available",
        "checkpoint_reference": {
            "checkpoint_id": "checkpoint-1", "worker_generation_id": "generation-a",
            "state": "available", "parent_run_id": parent["id"],
        },
    })
    monkeypatch.setattr(execution_fork, "plan_exact_force_token", lambda parent_run, request, **kwargs: {
        "classification": "exact_execution_fork", "request": {"change": request["change"]},
    })
    monkeypatch.setattr(execution_fork, "execute_exact_force_token", lambda *args, **kwargs: {
        "receipt": {
            "phase": "completed", "execution_id": "fork_exec_" + "a" * 20,
            "exactness": {"proof_status": "confirmed"}, "unchanged_control": {"status": "matched"},
        },
        "child": {"id": "exact-child", "response": "x", "trace": {"tokens": ["x"], "token_ids": [21]}},
    })
    monkeypatch.setattr(fan, "_comparison", lambda *_: {
        "state": "trace_unavailable", "first_divergence_view": {"state": "trace_unavailable"},
    })
    calls = [0]

    def cancel():
        calls[0] += 1
        return calls[0] > 2

    result = fan.branch_fan(parent, sub, 1, limit=3,
                            runtime_identity=RUNTIME, worker_identity=WORKER, cancel_check=cancel)
    assert result["summary"]["status"] == "partial_cancelled"
    assert result["summary"]["children_created"] == 1
    assert result["summary"]["not_attempted_branches"] == 2
    assert all(branch["reasons"][0]["code"] == "branch_fan_cancelled"
               for branch in result["branches"][1:])


def test_exact_child_trace_uses_worker_steps_and_never_retokenizes():
    parent = {
        "id": "parent",
        "response": "one two five",
        "trace": {
            "tokens": ["one", " two", " five"],
            "token_ids": [11, 22, 55],
            "alternatives": [[], [{"piece": " four", "token_id": 44, "prob": 0.41}], []],
        },
    }
    plan = {"request": {"position": 1, "execution_change": {
        "type": "force_token", "token_piece": " four", "token_id": 44,
    }}}
    reply = {
        "text": " four five",
        "tokens": [44, 55],
        "steps": [
            {"piece": " four", "id": 44},
            {"piece": " five", "id": 55, "prob": 0.8},
        ],
    }
    trace = _exact_child_trace(parent, plan, reply)
    assert trace["tokens"] == ["one", " four", " five"]
    assert trace["token_ids"] == [11, 44, 55]
    assert trace["steps"][1]["prob"] == 0.41
    assert "four" in "".join(trace["tokens"])


def test_missing_worker_trace_evidence_does_not_falsify_exactness():
    parent = {"id": "parent", "trace": {"tokens": ["a"], "token_ids": [1]}}
    plan = {"request": {"position": 0, "execution_change": {
        "type": "force_token", "token_piece": " b", "token_id": 2,
    }}}
    assert _exact_child_trace(parent, plan, {"text": " b", "tokens": [2]}) is None


def test_reconstructed_final_boundary_uses_zero_continuation_without_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))

    class NoGenerationEngine:
        def __init__(self):
            self.calls = 0

        def complete(self, *_args, **_kwargs):
            self.calls += 1
            raise AssertionError("final-token reconstructed Branch Fan generated past the recorded horizon")

    engine = NoGenerationEngine()
    sub = type("Substrate", (), {"engine": engine})()
    parent = {
        "id": "parent-final",
        "response": "one two",
        "final_prompt": "<prompt>",
        "trace": {"tokens": ["one", " two"], "token_ids": [1, 2]},
    }
    child = fan.reconstruct_branch_child(parent, sub, 1, token=" three", max_new=0)
    assert child is not None
    assert child["response"] == "one three"
    assert engine.calls == 0
    stored = runlog.get_run(child["id"])
    assert stored["finish_reason"] == "branch_horizon_exhausted"
