"""Sequential execution of a bounded fan over a run's recorded token alternatives.

Branch Fan is deliberately an orchestration layer.  It selects only alternatives already recorded on
the immutable parent, delegates exactness to the existing execution-fork policy, delegates
reconstruction to the existing legacy fork, and delegates comparison to ``diff_runs``.  It does not
create a fan/experiment object and does not perform a new model-analysis operation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

from clozn import schemas

SCHEMA_VERSION = "clozn.branch-fan.v1"
DEFAULT_LIMIT = 3
MIN_LIMIT = 1
MAX_LIMIT = 4


class BranchFanInputError(ValueError):
    """A typed caller-input error suitable for the HTTP route's stable 400 response."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_limit(limit: int) -> int:
    if not _is_int(limit) or not MIN_LIMIT <= limit <= MAX_LIMIT:
        raise BranchFanInputError("invalid_limit", f"limit must be an integer from {MIN_LIMIT} to {MAX_LIMIT}")
    return limit


def _trace_tokens(parent: Mapping) -> list[str]:
    trace = parent.get("trace")
    tokens = trace.get("tokens") if isinstance(trace, Mapping) else None
    if not isinstance(tokens, list) or not tokens or not all(isinstance(piece, str) for piece in tokens):
        raise BranchFanInputError("invalid_position", "parent has no recorded response token pieces")
    return tokens


def _candidate_projection(candidate: Mapping) -> dict:
    out = {"rank": candidate["recorded_rank"]}
    if candidate.get("token_id") is not None:
        out["token_id"] = candidate["token_id"]
    if candidate.get("probability") is not None:
        out["probability"] = candidate["probability"]
    return out


def _recorded_candidates(parent: Mapping, position: int, limit: int) -> tuple[list[dict], int]:
    """Select usable recorded alternatives in their source-array order."""
    tokens = _trace_tokens(parent)
    if not _is_int(position) or position < 0 or position >= len(tokens):
        raise BranchFanInputError("invalid_position", "position is outside the recorded response token range")
    trace = parent["trace"]
    raw_alternatives = trace.get("alternatives")
    at_position = (
        raw_alternatives[position]
        if isinstance(raw_alternatives, list) and position < len(raw_alternatives)
        and isinstance(raw_alternatives[position], list)
        else []
    )
    candidates = []
    seen_ids = set()
    seen_pieces = set()
    committed_piece = tokens[position]
    token_ids = trace.get("token_ids")
    committed_id = (
        token_ids[position]
        if isinstance(token_ids, list) and position < len(token_ids)
        and _is_int(token_ids[position]) and token_ids[position] >= 0
        else None
    )
    for recorded_rank, raw in enumerate(at_position):
        if not isinstance(raw, Mapping):
            continue
        piece = raw.get("piece", raw.get("text"))
        if not isinstance(piece, str) or not piece or piece == committed_piece:
            continue

        token_id = raw.get("token_id", raw.get("id"))
        if token_id is not None:
            if not _is_int(token_id) or token_id < 0:
                continue
            if committed_id is not None and token_id == committed_id:
                continue
        probability = raw.get("prob", raw.get("probability", raw.get("confidence")))
        if probability is not None:
            if (
                not isinstance(probability, (int, float))
                or isinstance(probability, bool)
                or not math.isfinite(float(probability))
                or probability < 0
                or probability > 1
            ):
                continue
            probability = float(probability)

        if token_id is not None:
            if token_id in seen_ids:
                continue
            seen_ids.add(token_id)
        elif piece in seen_pieces:
            continue
        seen_pieces.add(piece)
        candidate = {
            "recorded_rank": recorded_rank,
            "piece": piece,
        }
        if token_id is not None:
            candidate["token_id"] = token_id
        if probability is not None:
            candidate["probability"] = probability
        candidates.append(candidate)
    return candidates[:limit], len(at_position)


def recorded_alternatives_available(parent: Mapping, position: int) -> bool:
    """Read-only availability check using Branch Fan's own candidate authority.

    Dispatchers may use this to avoid waking a worker for the typed no-candidate result.  It does
    not expose, sort, or duplicate candidate selection; the actual Branch Fan call still owns the
    complete filtering and ordering operation.
    """
    candidates, _ = _recorded_candidates(parent, position, 1)
    return bool(candidates)


def recorded_alternative_candidates(parent: Mapping, position: int, *, limit: int = MAX_LIMIT) -> list[dict]:
    """Return the Branch Fan candidate projection without executing anything.

    This is a small read-side seam for affordance builders.  Candidate filtering, deduplication, and
    recorded-array ordering remain owned by :func:`_recorded_candidates`; callers must not reproduce
    those rules merely to render a Test This or Selection Inspector descriptor.
    """
    if not _is_int(limit) or not MIN_LIMIT <= limit <= MAX_LIMIT:
        raise BranchFanInputError("invalid_limit", f"limit must be an integer from {MIN_LIMIT} to {MAX_LIMIT}")
    candidates, _ = _recorded_candidates(parent, position, limit)
    return [dict(candidate) for candidate in candidates]


def _cancelled(cancel_check) -> bool:
    if not callable(cancel_check):
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return False


def _reason(code: str, message: str) -> dict:
    return {"code": code, "message": message}


_SAFE_REASON_MESSAGES = {
    "control_diverged": "the unchanged exact control did not match",
    "stale_plan": "exact execution preconditions changed after planning",
    "checkpoint_expired": "the shared exact checkpoint expired",
    "stale_worker_generation": "the exact checkpoint belongs to another worker generation",
    "execution_cancelled": "exact execution was cancelled",
    "intervention_failed": "the exact intervention failed",
    "persistence_failed": "the completed child could not be persisted",
    "reconstructed_execution_failed": "reconstructed replay did not produce a child",
    "exact_execution_failed": "exact execution failed",
}


def _public_reasons(raw) -> list[dict]:
    out = []
    for item in raw or []:
        if not isinstance(item, Mapping):
            continue
        code = item.get("code")
        if isinstance(code, str) and code:
            message = _SAFE_REASON_MESSAGES.get(code, "branch execution was unavailable")
            out.append({"code": code, "message": message})
    return out


def comparison_projection_from_diff(parent: Mapping, child: Mapping, diff: Mapping) -> dict:
    """Project only values returned by an already-computed canonical run diff."""
    view = diff.get("first_divergence_view")
    if not isinstance(view, Mapping):
        view = {"schema_version": "clozn.first-divergence-view.v1", "state": "trace_unavailable"}
    if not diff.get("trace_available"):
        return {
            "state": "trace_unavailable",
            "first_divergence_view": deepcopy(dict(view)),
        }
    out = {"state": "available", "first_divergence_view": deepcopy(dict(view))}
    for key in ("common_prefix_len", "first_divergence"):
        if key in diff:
            out[key] = deepcopy(diff[key])
    summary = diff.get("summary")
    if isinstance(summary, Mapping):
        value = summary.get("char_similarity")
        label = summary.get("char_similarity_label")
        if value is not None or label is not None:
            surface = {}
            if value is not None:
                surface["value"] = deepcopy(value)
            if label is not None:
                surface["label"] = deepcopy(label)
            out["surface_similarity"] = surface
    return out


def comparison_projection(parent: Mapping, child: Mapping) -> dict:
    """Compute and project one canonical run diff."""
    from clozn.analysis.model_diff import diff_runs

    return comparison_projection_from_diff(parent, child, diff_runs(parent, child))


# Keep the original local name for older tests/callers while exposing one shared projection seam to
# Test This.  Both features still call the canonical ``diff_runs`` implementation exactly once per
# successful child; neither feature invents a second token diff.
_comparison = comparison_projection


def _branch_alternative(candidate: Mapping) -> dict:
    return {"recorded_alternative": _candidate_projection(candidate)}


def _not_attempted(candidate: Mapping, code: str, message: str) -> dict:
    out = _branch_alternative(candidate)
    out.update({
        "state": "not_attempted",
        "outcome": "unavailable",
        "reasons": [_reason(code, message)],
        "comparison": None,
    })
    return out


def _unavailable(candidate: Mapping, reasons, *, exactness=None, unchanged_control=None) -> dict:
    out = _branch_alternative(candidate)
    out.update({
        "state": "unavailable",
        "outcome": "unavailable",
        "reasons": _public_reasons(reasons) or [_reason("branch_unavailable", "branch could not be produced")],
        "comparison": None,
    })
    if isinstance(exactness, Mapping):
        out["exactness"] = deepcopy(dict(exactness))
    if isinstance(unchanged_control, Mapping):
        out["unchanged_control"] = {"status": unchanged_control.get("status", "unavailable")}
    return out


def _fidelity(branches: list[Mapping]) -> str:
    outcomes = [branch.get("outcome") for branch in branches if branch.get("state") == "completed"]
    if not outcomes:
        return "none_completed"
    exact = sum(outcome == "exact_execution_fork" for outcome in outcomes)
    reconstructed = sum(outcome == "reconstructed_replay" for outcome in outcomes)
    if exact and reconstructed:
        return "mixed"
    if exact == len(outcomes):
        return "all_exact"
    if reconstructed == len(outcomes):
        return "all_reconstructed"
    return "mixed"


def _summary(branches: list[Mapping], requested: int, *, status: str) -> dict:
    return {
        "status": status,
        "requested_branches": requested,
        "attempted_branches": sum(branch.get("state") != "not_attempted" for branch in branches),
        "children_created": sum(branch.get("state") == "completed" for branch in branches),
        "exact_children": sum(branch.get("outcome") == "exact_execution_fork" for branch in branches),
        "reconstructed_children": sum(branch.get("outcome") == "reconstructed_replay" for branch in branches),
        "unavailable_branches": sum(branch.get("state") == "unavailable" for branch in branches),
        "not_attempted_branches": sum(branch.get("state") == "not_attempted" for branch in branches),
    }


def _execution_base(capture_state: str, *, reused: bool, fidelity: str, reason=None) -> dict:
    capture = {"state": capture_state, "reused_for_exact_candidates": bool(reused)}
    if reason is not None:
        capture["reason"] = deepcopy(reason)
    return {
        "policy": "exact_first",
        "order": "sequential",
        "checkpoint_capture": capture,
        "fidelity": fidelity,
    }


def _completed_exact(candidate, child, receipt, parent) -> dict:
    out = _branch_alternative(candidate)
    out.update({
        "state": "completed",
        "outcome": "exact_execution_fork",
        "child_run_id": child.get("id"),
        "exactness": deepcopy(receipt.get("exactness") or {"proof_status": "confirmed"}),
        "unchanged_control": {"status": (receipt.get("unchanged_control") or {}).get("status", "matched")},
        "reasons": [],
        "comparison": _comparison(parent, child),
    })
    if receipt.get("execution_id"):
        out["execution_fork_execution_id"] = receipt["execution_id"]
    return out


def _completed_reconstructed(candidate, child, plan, parent) -> dict:
    out = _branch_alternative(candidate)
    out.update({
        "state": "completed",
        "outcome": "reconstructed_replay",
        "child_run_id": child.get("id"),
        "exactness": deepcopy(plan.get("exactness") or {
            "regime": "reconstructed_text",
            "proof_status": "not_applicable",
        }),
        "unavoidable_differences": deepcopy(plan.get("unavoidable_differences") or []),
        "reasons": [],
        "comparison": _comparison(parent, child),
    })
    return out


_SHARED_FAILURE_CODES = frozenset({
    "stale_plan", "checkpoint_expired", "stale_worker_generation", "runtime_identity_mismatch",
    "worker_generation_changed", "checkpoint_invalidated", "execution_cancelled",
})


def _run_reconstructed(parent, sub, candidate, position, remaining, runtime_identity, worker_identity):
    from clozn.replay.execution_fork import plan_execution_fork
    from clozn.replay.fork import fork

    request = {
        "position": position,
        "change": {"type": "force_token", "token_piece": candidate["piece"]},
    }
    if candidate.get("token_id") is not None:
        request["change"]["token_id"] = candidate["token_id"]
    plan = plan_execution_fork(
        parent, request, checkpoint=None,
        runtime_identity=runtime_identity, worker_identity=worker_identity,
    )
    if plan.get("classification") != "reconstructed_replay":
        return _unavailable(candidate, plan.get("reasons"), exactness=plan.get("exactness"))
    child = fork(parent, sub, position, token=candidate["piece"], max_new=remaining)
    if not isinstance(child, Mapping) or not child.get("id"):
        return _unavailable(candidate, [_reason("reconstructed_execution_failed", "reconstructed replay did not produce a child")])
    return _completed_reconstructed(candidate, child, plan, parent)


def _branch_failure_code(branch: Mapping) -> str | None:
    reasons = branch.get("reasons")
    if isinstance(reasons, list) and reasons and isinstance(reasons[0], Mapping):
        code = reasons[0].get("code")
        return code if isinstance(code, str) else None
    return None


def branch_fan(
    parent_run: dict,
    sub,
    position: int,
    *,
    limit: int = DEFAULT_LIMIT,
    runtime_identity: dict | None = None,
    worker_identity: dict | None = None,
    reload_parent=None,
    cancel_check=None,
) -> dict:
    """Create bounded child forks for the parent's already-recorded alternatives."""
    if not isinstance(parent_run, Mapping):
        raise BranchFanInputError("invalid_parent", "parent run must be an object")
    parent_id = parent_run.get("id")
    if not isinstance(parent_id, str) or not parent_id:
        raise BranchFanInputError("invalid_parent", "parent run id is unavailable")
    if not _is_int(position) or position < 0:
        raise BranchFanInputError("invalid_position", "position must be a non-negative integer")
    limit = _validate_limit(limit)
    tokens = _trace_tokens(parent_run)
    if position >= len(tokens):
        raise BranchFanInputError("invalid_position", "position is outside the recorded response token range")

    candidates, recorded_count = _recorded_candidates(parent_run, position, limit)
    selection = {
        "source": "recorded_alternatives",
        "recorded_alternatives": recorded_count,
        "selected_alternatives": len(candidates),
        "requested_limit": limit,
    }
    if not candidates:
        selection.update({"state": "unavailable", "reason": "no_recorded_alternatives"})
        result = {
            "schema_version": SCHEMA_VERSION,
            "parent_run_id": parent_id,
            "position": position,
            "selection": selection,
            "execution": _execution_base("not_attempted", reused=False, fidelity="none_completed"),
            "branches": [],
            "summary": _summary([], 0, status="unavailable"),
        }
        schemas.validate(result, SCHEMA_VERSION)
        return result
    selection["state"] = "available"

    branches = []
    capture_state = "not_attempted"
    capture_reason = None
    checkpoint_reference = None
    exact_candidate_exists = any(candidate.get("token_id") is not None for candidate in candidates)
    engine = getattr(sub, "engine", None) if sub is not None else None
    exact_possible = (
        exact_candidate_exists and callable(getattr(engine, "execution_fork", None))
        and isinstance(runtime_identity, Mapping) and isinstance(worker_identity, Mapping)
    )

    if _cancelled(cancel_check):
        branches = [_not_attempted(candidate, "branch_fan_cancelled", "branch fan cancelled before execution")
                    for candidate in candidates]
        result = {
            "schema_version": SCHEMA_VERSION, "parent_run_id": parent_id, "position": position,
            "selection": selection,
            "execution": _execution_base("not_attempted", reused=False, fidelity="none_completed"),
            "branches": branches,
            "summary": _summary(branches, len(candidates), status="cancelled"),
        }
        schemas.validate(result, SCHEMA_VERSION)
        return result

    if exact_possible:
        from clozn.replay.fork import capture_exact_fork_context
        try:
            capture = capture_exact_fork_context(
                parent_run, engine, runtime_identity=dict(runtime_identity), worker_identity=dict(worker_identity))
        except Exception:
            capture = {
                "status": "ineligible",
                "reason": _reason("checkpoint_capture_unavailable", "an exact checkpoint could not be captured"),
            }
        if capture.get("status") == "available":
            reference = capture.get("checkpoint_reference")
            if isinstance(reference, Mapping):
                capture_state = "available"
                checkpoint_reference = deepcopy(dict(reference))
            else:
                capture_state = "unavailable"
                capture_reason = _reason("checkpoint_capture_unavailable", "an exact checkpoint reference was unavailable")
        else:
            capture_state = "unavailable"
            capture_reason = (_public_reasons([capture.get("reason")]) or [
                _reason("checkpoint_capture_unavailable", "an exact checkpoint could not be captured")
            ])[0]
    elif exact_candidate_exists:
        capture_state = "unavailable"
        capture_reason = _reason("exact_execution_unavailable", "exact checkpoint prerequisites were unavailable")

    from clozn.replay.fork import execute_exact_force_token, plan_exact_force_token
    import clozn.runs.store as runlog
    reload_parent = reload_parent or runlog.get_run
    remaining = max(0, len(tokens) - position - 1)
    stop_scheduling = None
    cancelled = False

    for offset, candidate in enumerate(candidates):
        if stop_scheduling is not None:
            branches.append(_not_attempted(candidate, *stop_scheduling))
            continue
        if _cancelled(cancel_check):
            cancelled = True
            for rest in candidates[offset:]:
                branches.append(_not_attempted(rest, "branch_fan_cancelled", "branch fan cancelled between branches"))
            break

        exact_candidate = checkpoint_reference is not None and candidate.get("token_id") is not None
        if exact_candidate:
            request = {
                "position": position,
                "change": {
                    "type": "force_token",
                    "token_id": candidate["token_id"],
                    "token_piece": candidate["piece"],
                },
            }
            try:
                plan = plan_exact_force_token(
                    parent_run, request, checkpoint_reference=checkpoint_reference,
                    runtime_identity=dict(runtime_identity), worker_identity=dict(worker_identity),
                )
                if plan.get("classification") != "exact_execution_fork":
                    branch = _unavailable(candidate, plan.get("reasons"), exactness=plan.get("exactness"))
                else:
                    execution = execute_exact_force_token(
                        parent_run, plan, engine,
                        runtime_identity=dict(runtime_identity), worker_identity=dict(worker_identity),
                        reload_parent=reload_parent, cancel_check=cancel_check,
                    )
                    receipt = execution.get("receipt") or {}
                    child = execution.get("child")
                    if receipt.get("phase") == "completed" and isinstance(child, Mapping) and child.get("id"):
                        branch = _completed_exact(candidate, child, receipt, parent_run)
                    else:
                        branch = _unavailable(
                            candidate, receipt.get("reasons"), exactness=receipt.get("exactness"),
                            unchanged_control=receipt.get("unchanged_control"),
                        )
                        if receipt.get("execution_id"):
                            branch["execution_fork_execution_id"] = receipt["execution_id"]
                        if receipt.get("phase") == "cancelled":
                            cancelled = True
            except Exception:
                branch = _unavailable(candidate, [_reason("exact_execution_failed", "exact execution failed")])
            branches.append(branch)
            code = _branch_failure_code(branch)
            if code in _SHARED_FAILURE_CODES:
                stop_scheduling = ("shared_exact_precondition_failed", "later branches were not attempted after a shared exact precondition failed")
            if cancelled:
                for rest in candidates[offset + 1:]:
                    branches.append(_not_attempted(rest, "branch_fan_cancelled", "branch fan cancelled after a child attempt"))
                break
            continue

        # Missing numeric ids (or an unavailable shared checkpoint) take the same reconstructed
        # planner/fork path as compat_fork.  No candidate is invented or rescored here.
        try:
            branch = _run_reconstructed(
                parent_run, sub, candidate, position, remaining, runtime_identity, worker_identity)
        except Exception:
            branch = _unavailable(candidate, [_reason("reconstructed_execution_failed", "reconstructed replay failed")])
        branches.append(branch)

    if cancelled:
        status = "partial_cancelled" if any(branch.get("state") == "completed" for branch in branches) else "cancelled"
    elif any(branch.get("state") == "not_attempted" for branch in branches):
        status = "partial" if any(branch.get("state") == "completed" for branch in branches) else "unavailable"
    elif any(branch.get("state") == "completed" for branch in branches):
        status = "completed" if all(branch.get("state") == "completed" for branch in branches) else "partial"
    else:
        status = "unavailable"
    result = {
        "schema_version": SCHEMA_VERSION,
        "parent_run_id": parent_id,
        "position": position,
        "selection": selection,
        "execution": _execution_base(
            capture_state, reused=checkpoint_reference is not None,
            fidelity=_fidelity(branches), reason=capture_reason),
        "branches": branches,
        "summary": _summary(branches, len(candidates), status=status),
    }
    schemas.validate(result, SCHEMA_VERSION)
    return result


__all__ = [
    "DEFAULT_LIMIT", "MAX_LIMIT", "MIN_LIMIT", "BranchFanInputError", "branch_fan",
    "comparison_projection", "comparison_projection_from_diff", "recorded_alternatives_available",
    "recorded_alternative_candidates",
]
