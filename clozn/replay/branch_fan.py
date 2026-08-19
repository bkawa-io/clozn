"""Bounded orchestration of canonical ForceToken time-travel experiments.

Branch Fan is deliberately an orchestration layer and nothing else.  It owns candidate discovery
over the immutable parent's recorded alternatives, the bounded fan size, the sequential order,
cancellation, and the composed summary.  It does not own exactness, reconstruction, generation, or
child creation.

Every selected alternative executes as one canonical ForceToken through the same
StateRef -> resolve_state -> Generate -> GenerateExecutionAdapter path Time Travel already uses, and
stops at a :class:`GeneratedObservation`.  Fanning N alternatives therefore produces N observations
and zero Runs; a Run appears only when a caller explicitly materializes one selected observation.

The fan carries no bespoke reconstruction, no prompt splicing, and no child persistence.  It also
never asks for a comparison that would require a second Run to exist: comparison against the
recorded parent is projected directly from the observation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math

from clozn import schemas

SCHEMA_VERSION = "clozn.branch-fan.v2"
DEFAULT_LIMIT = 3
MIN_LIMIT = 1
MAX_LIMIT = 4


class BranchFanInputError(ValueError):
    """A typed caller-input error suitable for the HTTP route's stable 400 response."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


# ------------------------------------------------------------------ candidate discovery (pure)
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


# ------------------------------------------------------------------------------ typed reasons
def _cancelled(cancel_check) -> bool:
    if not callable(cancel_check):
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return False


def _reason(code: str, message: str) -> dict:
    return {"code": code, "message": message}


# Branch-level failures are reported with a stable code and a caller-safe message.  Kernel
# diagnostics may quote worker text, so a code with no entry here degrades to the generic message
# rather than forwarding whatever the substrate said.
_SAFE_REASON_MESSAGES = {
    "base_run_unavailable": "the parent run could not be re-read for execution",
    "branch_fan_cancelled": "branch fan was cancelled",
    "checkpoint_expired": "the shared exact checkpoint expired",
    "checkpoint_missing": "no exact checkpoint was available",
    "checkpoint_parent_mismatch": "the exact checkpoint belongs to another parent run",
    "checkpoint_range_mismatch": "the boundary is outside the exact checkpoint token history",
    "exact_control_mismatch": "the unchanged exact control did not match",
    "exact_control_unavailable": "the unchanged exact control could not be proven",
    "exact_generation_unavailable": "exact execution was unavailable",
    "exact_state_unavailable": "exact execution preconditions were unavailable",
    "force_token_id_required": "exact execution requires a recorded numeric token id",
    "force_token_mismatch": "the alternative disagrees with the recorded token evidence",
    "generation_failed": "generation did not produce a usable continuation",
    "generation_malformed": "generation returned no usable text",
    "generation_unsupported": "the recorded generation contract is unsupported",
    "malformed_worker_token_evidence": "the worker returned unusable token evidence",
    "missing_prompt_boundary": "the exact checkpoint has no usable prompt boundary",
    "position_out_of_range": "the boundary is outside the recorded token history",
    "recorded_token_history_unavailable": "the parent has no aligned recorded token history",
    "reconstruction_prompt_unavailable": "the parent has no exact rendered prompt to reconstruct from",
    "reconstruction_token_piece_unavailable": "reconstruction requires the alternative's piece text",
    "runtime_identity_mismatch": "the selected runtime does not match the recorded runtime",
    "runtime_identity_unavailable": "runtime identity was unavailable",
    "stale_parent_execution": "the parent execution changed after the fan began",
    "stale_worker_generation": "the exact checkpoint belongs to another worker generation",
    "state_unavailable": "the resolved execution state was unavailable",
    "stochastic_execution_unbound": "the recorded sampled decode cannot be reproduced honestly",
    "token_boundary_out_of_range": "the boundary is outside the recorded token history",
    "token_trace_unavailable": "the parent has no usable recorded token trace",
    "worker_identity_unavailable": "the selected worker identity was unavailable",
}

# Preconditions shared by every exact branch.  Once one of these fails, later branches would fail
# identically, so the fan stops scheduling instead of repeating the same refused work.
_SHARED_FAILURE_CODES = frozenset({
    "base_run_unavailable", "checkpoint_expired", "checkpoint_missing", "checkpoint_parent_mismatch",
    "checkpoint_range_mismatch", "exact_control_mismatch", "exact_control_unavailable",
    "missing_prompt_boundary", "recorded_token_history_unavailable", "runtime_identity_mismatch",
    "runtime_identity_unavailable", "stale_parent_execution", "stale_worker_generation",
    "worker_identity_unavailable",
})


def _safe_reason(code, fallback_code: str = "branch_unavailable") -> dict:
    if not isinstance(code, str) or not code:
        code = fallback_code
    return _reason(code, _SAFE_REASON_MESSAGES.get(code, "the branch could not be produced"))


# ------------------------------------------------------------------------- branch projections
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


def _observation_identity(travel) -> dict:
    identity = {}
    for name in ("experiment_id", "arm_id", "observation_id"):
        value = getattr(travel, name, None)
        if isinstance(value, str) and value:
            identity[name] = value
    state_ref = getattr(travel, "state_ref", None)
    if state_ref is not None:
        # The compact identity payload only -- never the full execution state, which carries
        # prompt material this response has no business republishing.
        identity["state_ref"] = deepcopy(state_ref.identity_payload())
    return identity


def _fidelity_projection(observation_fidelity: Mapping) -> dict:
    """Carry the observation's own fidelity claim without restating or upgrading it."""
    out = {}
    for name in ("classification", "proof_status", "exact_match", "unchanged_control",
                 "sampler_state", "retokenized_prefix"):
        if name in observation_fidelity:
            out[name] = deepcopy(observation_fidelity[name])
    differences = observation_fidelity.get("unavoidable_differences")
    if isinstance(differences, list):
        out["unavoidable_differences"] = [str(item) for item in differences]
    return out


def _completed(candidate: Mapping, travel, *, outcome: str, comparison, policy: str) -> dict:
    out = _branch_alternative(candidate)
    out.update(_observation_identity(travel))
    continuation = travel.continuation if isinstance(travel.continuation, Mapping) else {}
    fidelity = continuation.get("fidelity")
    out.update({
        "state": "completed",
        "outcome": outcome,
        "resolution_policy": policy,
        "fidelity": _fidelity_projection(fidelity if isinstance(fidelity, Mapping) else {}),
        "generated": {
            "text_chars": len(str(continuation.get("generated_suffix_text") or "")),
            "token_count": len(continuation.get("generated_token_ids") or ()),
            "finish_reason": continuation.get("finish_reason"),
        },
        "reasons": [],
        "comparison": comparison,
    })
    return out


def _diagnostic_reason_code(diagnostics: Mapping):
    """The arm's reason code, whether reported flat or nested under the observation's diagnostics.

    A durable store round-trips a refused arm's own diagnostics one level down, so both shapes are
    the same fact and neither is a guess.
    """
    code = diagnostics.get("reason_code")
    if isinstance(code, str) and code:
        return code
    nested = diagnostics.get("diagnostics")
    code = nested.get("reason_code") if isinstance(nested, Mapping) else None
    return code if isinstance(code, str) and code else None


def _unavailable(candidate: Mapping, travel, *, policy: str, code=None) -> dict:
    out = _branch_alternative(candidate)
    out.update(_observation_identity(travel) if travel is not None else {})
    diagnostics = travel.diagnostics if travel is not None and isinstance(travel.diagnostics, Mapping) else {}
    if code is None:
        code = _diagnostic_reason_code(diagnostics)
    if code is None and diagnostics.get("reason") == "control_observation_not_available":
        # The runner blocks an arm whose unchanged control did not hold.  A refused control is not
        # reusable evidence, so it is never persisted and its specific reason is not readable from
        # here.  Report the fact this branch can actually prove rather than claiming a mismatch.
        code = "exact_control_unavailable"
    out.update({
        "state": "unavailable",
        "outcome": "unavailable",
        "resolution_policy": policy,
        "reasons": [_safe_reason(code)],
        "comparison": None,
    })
    return out


def _outcome(travel) -> str:
    continuation = travel.continuation if isinstance(travel.continuation, Mapping) else {}
    fidelity = continuation.get("fidelity")
    classification = fidelity.get("classification") if isinstance(fidelity, Mapping) else None
    if classification == "exact_execution_fork":
        return "exact"
    if classification == "reconstructed_replay":
        return "reconstructed"
    return "unavailable"


def _travel_cancelled(travel) -> bool:
    diagnostics = travel.diagnostics if isinstance(travel.diagnostics, Mapping) else {}
    return diagnostics.get("cancelled") is True or diagnostics.get("reason") == "cancelled"


def _branch_failure_code(branch: Mapping):
    reasons = branch.get("reasons")
    if isinstance(reasons, list) and reasons and isinstance(reasons[0], Mapping):
        code = reasons[0].get("code")
        return code if isinstance(code, str) else None
    return None


def _fidelity(branches: list[Mapping]) -> str:
    outcomes = [branch.get("outcome") for branch in branches if branch.get("state") == "completed"]
    if not outcomes:
        return "none_completed"
    exact = sum(outcome == "exact" for outcome in outcomes)
    reconstructed = sum(outcome == "reconstructed" for outcome in outcomes)
    if exact == len(outcomes):
        return "all_exact"
    if reconstructed == len(outcomes):
        return "all_reconstructed"
    return "mixed"


def _summary(branches: list[Mapping], requested: int, *, status: str) -> dict:
    return {
        "status": status,
        "requested": requested,
        "attempted": sum(branch.get("state") != "not_attempted" for branch in branches),
        "observations_completed": sum(branch.get("state") == "completed" for branch in branches),
        "exact_observations": sum(branch.get("outcome") == "exact" for branch in branches),
        "reconstructed_observations": sum(branch.get("outcome") == "reconstructed" for branch in branches),
        "unavailable": sum(branch.get("state") == "unavailable" for branch in branches),
        "not_attempted": sum(branch.get("state") == "not_attempted" for branch in branches),
    }


def _execution_base(capture_state: str, *, reused: bool, fidelity: str, reason=None) -> dict:
    capture = {"state": capture_state, "reused_for_exact_candidates": bool(reused)}
    if reason is not None:
        capture["reason"] = deepcopy(reason)
    return {
        "policy": "exact_first",
        "order": "sequential",
        "materialization": "explicit_choice_only",
        "checkpoint_capture": capture,
        "fidelity": fidelity,
    }


def _result(parent_id: str, position: int, selection: Mapping, execution: Mapping,
            branches: list, summary: Mapping) -> dict:
    result = {
        "schema_version": SCHEMA_VERSION,
        "parent_run_id": parent_id,
        "position": position,
        "selection": deepcopy(dict(selection)),
        "execution": deepcopy(dict(execution)),
        "branches": branches,
        "summary": deepcopy(dict(summary)),
    }
    schemas.validate(result, SCHEMA_VERSION)
    return result


# ------------------------------------------------------------------------ shared exact context
def _capture_checkpoint(parent_run: Mapping, engine, *, runtime_identity, worker_identity):
    """Capture one exact checkpoint for the whole fan through the canonical capture seam.

    Capturing once and reusing it across every exact candidate is orchestration, which Branch Fan
    keeps.  Deciding what an exact execution then means is not, so nothing here plans or executes.
    """
    from clozn.replay.checkpoint_capture import capture_parent_checkpoint

    try:
        capture = capture_parent_checkpoint(
            parent_run, engine,
            runtime_identity=dict(runtime_identity), worker_identity=dict(worker_identity))
    except Exception:
        # A capture that cannot even be requested is an unavailable exact context, not a fan failure:
        # every candidate still gets its own typed reconstructed or refused result below.
        return None, _reason("checkpoint_capture_unavailable", "an exact checkpoint could not be captured")
    if not isinstance(capture, Mapping) or capture.get("status") != "available":
        reasons = capture.get("reasons") if isinstance(capture, Mapping) else None
        code = None
        if isinstance(reasons, list) and reasons and isinstance(reasons[0], Mapping):
            code = reasons[0].get("code")
        return None, _safe_reason(code, "checkpoint_capture_unavailable")
    reference = capture.get("checkpoint_reference")
    if not isinstance(reference, Mapping):
        return None, _reason("checkpoint_capture_unavailable", "an exact checkpoint reference was unavailable")
    return deepcopy(dict(reference)), None


# ------------------------------------------------------------------------------------ the fan
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
    checkpoint: Mapping | None = None,
    observation_store=None,
    execution_adapter=None,
) -> dict:
    """Fan the parent's recorded alternatives into canonical ForceToken observations.

    Creates no Runs.  Each branch reports the observation it produced; a caller materializes one
    selected observation afterwards through the generic materializer.
    """
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
        return _result(
            parent_id, position, selection,
            _execution_base("not_attempted", reused=False, fidelity="none_completed"),
            [], _summary([], 0, status="unavailable"),
        )
    selection["state"] = "available"

    if _cancelled(cancel_check):
        branches = [_not_attempted(candidate, "branch_fan_cancelled", "branch fan cancelled before execution")
                    for candidate in candidates]
        return _result(
            parent_id, position, selection,
            _execution_base("not_attempted", reused=False, fidelity="none_completed"),
            branches, _summary(branches, len(candidates), status="cancelled"),
        )

    engine = getattr(sub, "engine", None) if sub is not None else None
    exact_candidate_exists = any(candidate.get("token_id") is not None for candidate in candidates)
    capture_state = "not_attempted"
    capture_reason = None
    checkpoint_reference = deepcopy(dict(checkpoint)) if isinstance(checkpoint, Mapping) else None
    if checkpoint_reference is not None:
        capture_state = "available"
    elif exact_candidate_exists:
        exact_possible = (
            callable(getattr(engine, "execution_fork", None))
            and isinstance(runtime_identity, Mapping) and isinstance(worker_identity, Mapping)
        )
        if exact_possible:
            checkpoint_reference, capture_reason = _capture_checkpoint(
                parent_run, engine, runtime_identity=runtime_identity, worker_identity=worker_identity)
            capture_state = "available" if checkpoint_reference is not None else "unavailable"
        else:
            capture_state = "unavailable"
            capture_reason = _reason("exact_execution_unavailable", "exact checkpoint prerequisites were unavailable")

    from clozn.experiments.generation import GenerateExecutionAdapter
    from clozn.experiments.observation_comparison import observation_comparison
    from clozn.experiments.persistence import ObservationStore
    from clozn.recipes.time_travel import TimeTravelError, run_time_travel

    adapter = execution_adapter
    if adapter is None:
        if sub is None:
            raise BranchFanInputError("invalid_substrate", "branch fan requires a substrate to generate with")
        # One adapter for the whole fan: the unchanged exact control is proven once and reused by
        # every later exact branch instead of re-proving the same state per candidate.
        adapter = GenerateExecutionAdapter(
            sub, run=parent_run, run_loader=reload_parent,
            runtime_identity=runtime_identity, worker_identity=worker_identity,
        )
    durable = observation_store or ObservationStore()
    # The fan's horizon is the parent's remaining recorded horizon; the forced token consumes the
    # first generated slot, exactly as Generate defines it.
    max_new = len(tokens) - position

    branches: list[dict] = []
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

        # Branch Fan chooses the resolution policy; the resolver still owns what each policy means.
        # An alternative with no recorded numeric id can only be reconstructed, and a supplied or
        # captured checkpoint is never silently downgraded for one that does have an id.
        exact_candidate = checkpoint_reference is not None and candidate.get("token_id") is not None
        policy = "exact_preferred" if exact_candidate else "reconstructed_only"
        try:
            travel = run_time_travel(
                parent_run,
                position=position,
                token_id=candidate.get("token_id"),
                token_piece=candidate["piece"],
                max_new=max_new,
                policy=policy,
                checkpoint=checkpoint_reference if exact_candidate else None,
                runtime_identity=runtime_identity,
                worker_identity=worker_identity,
                execution_adapter=adapter,
                run_loader=reload_parent,
                observation_store=durable,
                cancel=cancel_check,
            )
        except TimeTravelError as exc:
            branches.append(_unavailable(candidate, None, policy=policy, code=exc.code))
            code = _branch_failure_code(branches[-1])
            if code in _SHARED_FAILURE_CODES:
                stop_scheduling = ("shared_exact_precondition_failed",
                                   "later branches were not attempted after a shared precondition failed")
            continue
        except Exception:
            branches.append(_unavailable(candidate, None, policy=policy, code="branch_execution_failed"))
            continue

        if travel.status == "completed":
            observation = _completed_observation(durable, travel)
            comparison = observation_comparison(parent_run, observation) if observation is not None else None
            branches.append(_completed(candidate, travel, outcome=_outcome(travel),
                                       comparison=comparison, policy=policy))
            continue
        if _travel_cancelled(travel):
            cancelled = True
            for rest in candidates[offset:]:
                branches.append(_not_attempted(rest, "branch_fan_cancelled", "branch fan cancelled during a branch"))
            break
        branches.append(_unavailable(candidate, travel, policy=policy))
        code = _branch_failure_code(branches[-1])
        if code in _SHARED_FAILURE_CODES:
            stop_scheduling = ("shared_exact_precondition_failed",
                               "later branches were not attempted after a shared precondition failed")

    completed = any(branch.get("state") == "completed" for branch in branches)
    if cancelled:
        status = "partial_cancelled" if completed else "cancelled"
    elif any(branch.get("state") == "not_attempted" for branch in branches):
        status = "partial" if completed else "unavailable"
    elif completed:
        status = "completed" if all(branch.get("state") == "completed" for branch in branches) else "partial"
    else:
        status = "unavailable"

    return _result(
        parent_id, position, selection,
        _execution_base(capture_state, reused=checkpoint_reference is not None,
                        fidelity=_fidelity(branches), reason=capture_reason),
        branches, _summary(branches, len(candidates), status=status),
    )


def _completed_observation(store, travel):
    """Re-read the persisted observation the branch just produced.

    Comparison is a read over durable evidence, not a second execution.  A store miss simply
    leaves the branch's comparison absent rather than inventing one.
    """
    observation_id = getattr(travel, "observation_id", None)
    if not isinstance(observation_id, str) or not observation_id:
        return None
    try:
        from clozn.experiments.observations import GeneratedObservation

        observation = store.get_observation(observation_id)
    except Exception:
        return None
    return observation if isinstance(observation, GeneratedObservation) else None


__all__ = [
    "DEFAULT_LIMIT", "MAX_LIMIT", "MIN_LIMIT", "SCHEMA_VERSION", "BranchFanInputError", "branch_fan",
    "recorded_alternatives_available", "recorded_alternative_candidates",
]
