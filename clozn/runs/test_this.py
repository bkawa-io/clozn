"""Pure planning for the backend's common ``Test This`` dispatcher.

This module deliberately stops at a read-only plan.  It resolves a user selection against recorded
run evidence, validates the request with the existing Execution Fork rules, and identifies the
existing primitive that an explicit execution route may invoke.  It never selects a worker, calls a
model, captures a checkpoint, creates a child, or writes a result.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from clozn import schemas
from clozn.replay.branch_fan import (
    DEFAULT_LIMIT,
    MAX_LIMIT,
    MIN_LIMIT,
    recorded_alternatives_available,
)
from clozn.replay.execution_fork import (
    normalize_intervention,
    parent_execution_fingerprint,
    recorded_sampling_state,
)

SCHEMA_VERSION = "clozn.test-this-plan.v1"
RESULT_SCHEMA_VERSION = "clozn.test-this-result.v1"


class TestThisInputError(ValueError):
    """Malformed caller input; routes expose only this stable code, not parser internals."""

    __test__ = False

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_numeric_id(value: Any) -> bool:
    return _is_int(value) and value >= 0


def _finite_probability(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and 0 <= value <= 1
    )


def _trace_tokens(run: Mapping) -> list[str]:
    trace = run.get("trace")
    tokens = trace.get("tokens") if isinstance(trace, Mapping) else None
    if not isinstance(tokens, list) or not tokens or not all(isinstance(piece, str) for piece in tokens):
        raise TestThisInputError(
            "invalid_position", "parent has no recorded response token boundaries")
    return tokens


def _position(run: Mapping, raw: Any, *, default: int | None = None) -> int:
    value = default if raw is None and default is not None else raw
    if not _is_int(value) or value < 0:
        raise TestThisInputError("invalid_position", "position must be a non-negative integer")
    tokens = _trace_tokens(run)
    if value >= len(tokens):
        raise TestThisInputError(
            "invalid_position", "position is outside the recorded response token range")
    return value


def _trace_alternatives(run: Mapping, position: int) -> list[Any]:
    trace = run.get("trace")
    alternatives = trace.get("alternatives") if isinstance(trace, Mapping) else None
    if not isinstance(alternatives, list) or position >= len(alternatives):
        return []
    at_position = alternatives[position]
    return at_position if isinstance(at_position, list) else []


def _committed_token(run: Mapping, position: int) -> tuple[str, int | None]:
    tokens = _trace_tokens(run)
    trace = run.get("trace")
    ids = trace.get("token_ids") if isinstance(trace, Mapping) else None
    committed_id = ids[position] if isinstance(ids, list) and position < len(ids) else None
    return tokens[position], committed_id if _is_numeric_id(committed_id) else None


def _candidate(raw: Any, rank: int, run: Mapping, position: int) -> dict | None:
    if not isinstance(raw, Mapping):
        return None
    piece = raw.get("piece", raw.get("text"))
    if not isinstance(piece, str) or not piece:
        return None
    token_id = raw.get("token_id", raw.get("id"))
    if token_id is not None and not _is_numeric_id(token_id):
        return None
    probability = raw.get("prob", raw.get("probability", raw.get("confidence")))
    if probability is not None:
        if not _finite_probability(probability):
            return None
        probability = float(probability)
    committed_piece, committed_id = _committed_token(run, position)
    if piece == committed_piece or (token_id is not None and token_id == committed_id):
        return None
    out = {"recorded_rank": rank, "piece": piece}
    if token_id is not None:
        out["token_id"] = token_id
    if probability is not None:
        out["probability"] = probability
    return out


def resolve_recorded_alternative(
    run: Mapping,
    position: int,
    *,
    alternative_rank: int | None = None,
    token_id: int | None = None,
) -> dict:
    """Resolve exactly one recorded alternative for execution.

    The returned ``piece`` is an internal dispatch value only.  It must never be copied into the
    metadata-only Test This plan/result documents.
    """
    alternatives = _trace_alternatives(run, position)
    candidates = []
    for rank, raw in enumerate(alternatives):
        candidate = _candidate(raw, rank, run, position)
        if candidate is not None:
            candidates.append(candidate)
    if alternative_rank is not None:
        if alternative_rank >= len(alternatives):
            raise TestThisInputError(
                "alternative_unavailable", "the requested recorded alternative is unavailable")
        candidate = _candidate(alternatives[alternative_rank], alternative_rank, run, position)
        if candidate is None:
            raise TestThisInputError(
                "alternative_unavailable", "the requested recorded alternative is malformed or committed")
        return candidate
    for candidate in candidates:
        if candidate.get("token_id") == token_id:
            return candidate
    raise TestThisInputError(
        "alternative_unavailable", "the requested token id is not a recorded alternative")


def _alternative_projection(candidate: Mapping) -> dict:
    out = {"rank": candidate["recorded_rank"]}
    if candidate.get("token_id") is not None:
        out["token_id"] = candidate["token_id"]
    if candidate.get("probability") is not None:
        out["probability"] = candidate["probability"]
    return out


def _body(request: Any) -> tuple[Mapping, Mapping]:
    if not isinstance(request, Mapping):
        raise TestThisInputError("invalid_body", "body must be an object")
    if set(request) != {"selection", "test"}:
        raise TestThisInputError("invalid_body", "body must contain only selection and test")
    selection = request.get("selection")
    test = request.get("test")
    if not isinstance(selection, Mapping):
        raise TestThisInputError("invalid_selection", "selection must be an object")
    if not isinstance(test, Mapping):
        raise TestThisInputError("invalid_test", "test must be an object")
    return selection, test


def _normalize_selection(run: Mapping, raw: Mapping) -> dict:
    kind = raw.get("kind")
    if kind == "context_span":
        if set(raw) != {"kind", "source_span_id", "answer_span_id"}:
            raise TestThisInputError(
                "invalid_selection",
                "context_span selection needs kind, source_span_id, and answer_span_id",
            )
        if not all(isinstance(raw.get(key), str) and raw.get(key)
                   for key in ("source_span_id", "answer_span_id")):
            raise TestThisInputError(
                "invalid_selection", "context span IDs must be non-empty strings")
        return {
            "kind": kind,
            "source_span_id": raw["source_span_id"],
            "answer_span_id": raw["answer_span_id"],
        }
    if kind not in {"response_token", "sampling"}:
        raise TestThisInputError("invalid_selection_kind", "selection kind is not supported")
    if kind == "response_token":
        if set(raw) != {"kind", "position"}:
            raise TestThisInputError("invalid_selection", "response_token selection needs kind and position")
        return {"kind": kind, "position": _position(run, raw.get("position"))}
    if set(raw) - {"kind", "position"}:
        raise TestThisInputError("invalid_selection", "sampling selection has unknown fields")
    return {
        "kind": kind,
        "position": _position(
            run,
            raw.get("position"),
            default=0 if "position" not in raw else None,
        ),
    }


def _sampling_test(run: Mapping, selection: Mapping, raw: Mapping) -> tuple[dict, dict, dict | None]:
    if selection["kind"] != "sampling":
        raise TestThisInputError("selection_test_mismatch", "change_sampling requires a sampling selection")
    if set(raw) != {"kind", "changes"}:
        raise TestThisInputError("invalid_test", "change_sampling needs only kind and changes")
    changes = raw.get("changes")
    if not isinstance(changes, Mapping) or not changes:
        raise TestThisInputError("invalid_intervention", "change_sampling needs at least one sampler change")
    normalized, reason = normalize_intervention({"type": "sampling", **dict(changes)})
    if reason is not None or normalized is None:
        raise TestThisInputError(
            str((reason or {}).get("code") or "invalid_intervention"),
            str((reason or {}).get("message") or "sampling change is invalid"),
        )
    normalized_changes = {key: value for key, value in normalized.items() if key != "type"}
    state = recorded_sampling_state(run)
    if isinstance(state, Mapping) and all(
        key in state and state[key] == value for key, value in normalized_changes.items()
    ):
        return (
            {"kind": "change_sampling", "changes": normalized_changes},
            {"operation": "sampling_fork", "change": {"type": "sampling", **normalized_changes}},
            {"code": "no_effective_change", "message": "the requested sampler change has no effective difference from the parent"},
        )
    return (
        {"kind": "change_sampling", "changes": normalized_changes},
        {"operation": "sampling_fork", "change": {"type": "sampling", **normalized_changes}},
        None,
    )


def _sampling_probe_test(run: Mapping, selection: Mapping, raw: Mapping) -> tuple[dict, dict, dict | None]:
    if selection["kind"] != "sampling":
        raise TestThisInputError(
            "selection_test_mismatch", "sampler sensitivity requires a sampling selection")
    if set(raw) - {"kind", "recipe", "seed_probes"}:
        raise TestThisInputError(
            "invalid_test", "probe_sensitivity supports only recipe and seed_probes")
    recipe = raw.get("recipe", "nearby_v1")
    seed_probes = raw.get("seed_probes", 0)
    from clozn.replay.sampler_sensitivity import (
        SamplerSensitivityInputError,
        plan_sampler_sensitivity,
    )
    try:
        sensitivity_plan = plan_sampler_sensitivity(
            run,
            position=selection["position"],
            recipe=recipe,
            seed_probes=seed_probes,
        )
    except SamplerSensitivityInputError as exc:
        raise TestThisInputError(exc.code, str(exc)) from None
    test = {"kind": "probe_sensitivity", "recipe": recipe, "seed_probes": seed_probes}
    resolved = {
        "operation": "sampler_sensitivity",
        "sampler_sensitivity_plan": sensitivity_plan,
    }
    if sensitivity_plan["execution"]["state"] != "ready":
        reason = (sensitivity_plan["execution"].get("reasons") or [{
            "code": sensitivity_plan["execution"].get("reason", "sampler_sensitivity_unavailable"),
            "message": "the recorded sampler cannot support this sensitivity probe",
        }])[0]
        return test, resolved, reason
    return test, resolved, None


def _token_test(run: Mapping, selection: Mapping, raw: Mapping) -> tuple[dict, dict, dict | None]:
    if selection["kind"] != "response_token":
        raise TestThisInputError("selection_test_mismatch", "token tests require a response_token selection")
    kind = raw.get("kind")
    if kind == "fan_alternatives":
        if set(raw) - {"kind", "limit"}:
            raise TestThisInputError("invalid_test", "fan_alternatives has unknown fields")
        limit = raw.get("limit", DEFAULT_LIMIT)
        if not _is_int(limit) or not MIN_LIMIT <= limit <= MAX_LIMIT:
            raise TestThisInputError("invalid_limit", "fan limit must be an integer from 1 to 4")
        # Candidate ordering and filtering belong exclusively to Branch Fan.  This shared authority
        # only answers whether the typed no-candidate result can be known without waking a worker.
        if not recorded_alternatives_available(run, selection["position"]):
            return (
                {"kind": kind, "limit": limit},
                {"operation": "branch_fan"},
                {"code": "no_recorded_alternatives", "message": "no usable recorded alternatives are available"},
            )
        return (
            {"kind": kind, "limit": limit},
            {"operation": "branch_fan"},
            None,
        )
    if kind != "try_alternative":
        raise TestThisInputError("invalid_test_kind", "test kind is not supported")
    selectors = [name for name in ("alternative_rank", "token_id") if name in raw]
    if set(raw) - {"kind", "alternative_rank", "token_id"} or len(selectors) != 1:
        raise TestThisInputError(
            "invalid_selector", "try_alternative needs exactly one alternative_rank or token_id")
    selector = raw[selectors[0]]
    if not _is_int(selector) or selector < 0:
        raise TestThisInputError(
            "invalid_selector", f"{selectors[0]} must be a non-negative integer")
    try:
        candidate = resolve_recorded_alternative(
            run,
            selection["position"],
            alternative_rank=selector if selectors[0] == "alternative_rank" else None,
            token_id=selector if selectors[0] == "token_id" else None,
        )
    except TestThisInputError as exc:
        if exc.code != "alternative_unavailable":
            raise
        return (
            {"kind": kind, selectors[0]: selector},
            {"operation": "force_token"},
            {"code": exc.code, "message": str(exc)},
        )
    return (
        {"kind": kind, selectors[0]: selector},
        {"operation": "force_token", "recorded_alternative": _alternative_projection(candidate)},
        None,
    )


def _context_test(run: Mapping, selection: Mapping, raw: Mapping) -> tuple[dict, dict, dict | None]:
    if selection["kind"] != "context_span":
        raise TestThisInputError(
            "selection_test_mismatch", "context tests require a context_span selection")
    kind = raw.get("kind")
    if kind not in {"neutralize", "remove"} or set(raw) != {"kind"}:
        raise TestThisInputError(
            "invalid_test", "context span tests support only neutralize or remove")

    from clozn.runs.influence_counterfactual import build_influence_counterfactual_plan

    request = {
        "influence": {
            "source_span_id": selection["source_span_id"],
            "answer_span_id": selection["answer_span_id"],
        },
        "intervention": {"kind": kind},
        "specificity_control": True,
    }
    counterfactual_plan = build_influence_counterfactual_plan(run, request)
    reason = None
    if counterfactual_plan["execution"]["state"] != "ready":
        reason = {
            "code": counterfactual_plan["execution"].get("reason", "counterfactual_unavailable"),
            "message": "the selected measured context link cannot currently be regenerated",
        }
    return (
        {"kind": kind},
        {
            "operation": "influence_counterfactual",
            "counterfactual_plan": counterfactual_plan,
        },
        reason,
    )


def _context_bisect_test(run: Mapping, selection: Mapping, raw: Mapping) -> tuple[dict, dict, dict | None]:
    if selection["kind"] != "context_span":
        raise TestThisInputError(
            "selection_test_mismatch", "context bisect requires a context_span selection")
    if raw.get("kind") != "bisect":
        raise TestThisInputError("invalid_test_kind", "test kind is not supported")
    allowed = {"kind", "recipe", "max_depth", "max_runs", "max_seconds", "min_region_chars"}
    if set(raw) - allowed:
        raise TestThisInputError("invalid_test", "bisect has unknown fields")
    option_keys = ("recipe", "max_depth", "max_runs", "max_seconds", "min_region_chars")
    request = {
        "influence": {
            "source_span_id": selection["source_span_id"],
            "answer_span_id": selection["answer_span_id"],
        },
        **{key: raw[key] for key in option_keys if key in raw},
    }
    from clozn.runs.context_bisect import ContextBisectInputError, plan_context_bisect
    try:
        bisect_plan = plan_context_bisect(run, request=request)
    except ContextBisectInputError as exc:
        raise TestThisInputError(exc.code, str(exc)) from None
    test = {"kind": "bisect", **{key: raw[key] for key in option_keys if key in raw}}
    resolved = {"operation": "context_bisect", "context_bisect_plan": bisect_plan}
    if bisect_plan["execution"]["state"] != "ready":
        reason = {
            "code": bisect_plan["execution"].get("reason", "context_bisect_unavailable"),
            "message": "the selected measured context relationship cannot currently be bisected",
        }
        return test, resolved, reason
    return test, resolved, None


def build_test_this_plan(run: Mapping, request: Any) -> dict:
    """Build and validate a deterministic, read-only Test This plan."""
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run.get("id"):
        raise TestThisInputError("invalid_parent", "parent run is unavailable")
    raw_selection, raw_test = _body(request)
    selection = _normalize_selection(run, raw_selection)
    if selection["kind"] == "context_span":
        if raw_test.get("kind") == "bisect":
            test, resolved, no_effect = _context_bisect_test(run, selection, raw_test)
        else:
            test, resolved, no_effect = _context_test(run, selection, raw_test)
    elif raw_test.get("kind") == "change_sampling":
        test, resolved, no_effect = _sampling_test(run, selection, raw_test)
    elif raw_test.get("kind") == "probe_sensitivity":
        test, resolved, no_effect = _sampling_probe_test(run, selection, raw_test)
    else:
        test, resolved, no_effect = _token_test(run, selection, raw_test)

    operation = resolved["operation"]
    fidelity = (
        "exact_required" if operation in {"sampling_fork", "sampler_sensitivity"}
        else "controlled_regeneration" if operation in {"influence_counterfactual", "context_bisect"}
        else "exact_first"
    )
    backend_by_operation = {
        "sampling_fork": "execution_fork",
        "branch_fan": "branch_fan",
        "influence_counterfactual": "influence_counterfactual",
        "sampler_sensitivity": "sampler_sensitivity",
        "context_bisect": "context_bisect",
        "force_token": "force_token_fork",
    }
    backend = backend_by_operation.get(operation, "force_token_fork")
    state = "unavailable" if no_effect is not None else "ready"
    resolution = {"state": state, "operation": operation}
    execution = {
        "state": state,
        "backend": backend,
        "fidelity_policy": fidelity,
        "live_state": "not_checked",
    }
    if no_effect:
        resolution["reason"] = no_effect
        execution["reason"] = no_effect["code"]
    plan = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run["id"],
        "parent_fingerprint_sha256": parent_execution_fingerprint(run),
        "selection": selection,
        "test": test,
        "resolution": resolution,
        "resolved_test": resolved,
        "execution": execution,
    }
    schemas.validate(plan, SCHEMA_VERSION)
    return plan


__all__ = [
    "RESULT_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "TestThisInputError",
    "build_test_this_plan",
    "resolve_recorded_alternative",
]
