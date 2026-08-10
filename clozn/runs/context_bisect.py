"""Pure planning for bounded Prompt / Context Bisect.

Context Bisect starts from one persisted influence link and plans a bounded search over a freshly
resolved message-backed source span.  It does not score, generate, mutate, or create a search session.
Execution lives in :mod:`clozn.replay.context_bisect` and delegates replay, span surgery, comparison,
and target-region classification to existing Clozn primitives.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from clozn import schemas
from clozn.replay import span_bridge
from clozn.replay.controlled import recorded_sampling_config
from clozn.replay.execution_fork import parent_execution_fingerprint, parent_runtime_projection
from clozn.runs import influence_geometry as geometry
from clozn.runs.influence_counterfactual import _measurement

PLAN_SCHEMA_VERSION = "clozn.context-bisect-plan.v1"
RESULT_SCHEMA_VERSION = "clozn.context-bisect.v1"
FILLER_RECIPE = "clozn.matched_length_neutral_filler.v1"
DEFAULT_RECIPE = "neutralize_bisect_v1"
DEFAULT_MAX_DEPTH = 3
DEFAULT_MAX_RUNS = 8
DEFAULT_MAX_SECONDS = 120.0
DEFAULT_MIN_REGION_CHARS = 32
MIN_DEPTH, MAX_DEPTH = 0, 5
MIN_RUNS, MAX_RUNS = 2, 12
MIN_SECONDS, MAX_SECONDS = 1.0, 300.0
MIN_REGION_CHARS, MAX_REGION_CHARS = 8, 256


class ContextBisectInputError(ValueError):
    """Malformed request data; routes expose only the stable ``code``."""

    __test__ = False

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                         allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reason(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def _request(request: Any) -> tuple[dict, dict]:
    if not isinstance(request, Mapping):
        raise ContextBisectInputError("invalid_body", "body must be an object")
    allowed = {"influence", "recipe", "max_depth", "max_runs", "max_seconds", "min_region_chars"}
    if set(request) - allowed:
        raise ContextBisectInputError("invalid_body", "body contains an unsupported field")
    influence = request.get("influence")
    if not isinstance(influence, Mapping) or set(influence) != {"source_span_id", "answer_span_id"}:
        raise ContextBisectInputError("invalid_influence", "influence needs source_span_id and answer_span_id")
    source_id, answer_id = influence.get("source_span_id"), influence.get("answer_span_id")
    if not all(isinstance(value, str) and value for value in (source_id, answer_id)):
        raise ContextBisectInputError("invalid_influence", "influence span IDs must be non-empty strings")

    recipe = request.get("recipe", DEFAULT_RECIPE)
    if recipe != DEFAULT_RECIPE:
        raise ContextBisectInputError("invalid_recipe", "recipe must be neutralize_bisect_v1")
    max_depth = request.get("max_depth", DEFAULT_MAX_DEPTH)
    if not _is_int(max_depth) or not MIN_DEPTH <= max_depth <= MAX_DEPTH:
        raise ContextBisectInputError("invalid_max_depth", "max_depth must be an integer from 0 to 5")
    max_runs = request.get("max_runs", DEFAULT_MAX_RUNS)
    if not _is_int(max_runs) or not MIN_RUNS <= max_runs <= MAX_RUNS:
        raise ContextBisectInputError("invalid_max_runs", "max_runs must be an integer from 2 to 12")
    max_seconds = request.get("max_seconds", DEFAULT_MAX_SECONDS)
    if (isinstance(max_seconds, bool) or not isinstance(max_seconds, (int, float))
            or not math.isfinite(float(max_seconds))
            or not MIN_SECONDS <= float(max_seconds) <= MAX_SECONDS):
        raise ContextBisectInputError("invalid_max_seconds", "max_seconds must be between 1 and 300")
    min_chars = request.get("min_region_chars", DEFAULT_MIN_REGION_CHARS)
    if not _is_int(min_chars) or not MIN_REGION_CHARS <= min_chars <= MAX_REGION_CHARS:
        raise ContextBisectInputError("invalid_min_region_chars", "min_region_chars must be an integer from 8 to 256")
    return {
        "source_span_id": source_id,
        "answer_span_id": answer_id,
    }, {
        "recipe": recipe,
        "max_depth": max_depth,
        "max_runs": max_runs,
        "max_seconds": float(max_seconds),
        "min_region_chars": min_chars,
    }


def split_region(text: str, start: int, end: int, min_region_chars: int) -> tuple[dict, dict] | None:
    """Return the nearest valid whitespace partition of ``[start, end)``.

    Coordinates are Unicode code-point offsets.  The returned children are exact, gap-free, non-
    overlapping partitions.  A word is never cut merely to spend more search budget.
    """
    if not isinstance(text, str) or not all(_is_int(value) for value in (start, end, min_region_chars)):
        return None
    if start < 0 or end <= start or end > len(text) or min_region_chars < 1:
        return None
    lower, upper = start + min_region_chars, end - min_region_chars
    if lower > upper:
        return None
    midpoint = start + (end - start) // 2
    boundaries = [boundary for boundary in range(lower, upper + 1)
                  if (boundary > start and text[boundary - 1].isspace())
                  or (boundary < end and text[boundary].isspace())]
    if not boundaries:
        return None
    boundary = min(boundaries, key=lambda item: (abs(item - midpoint), item))
    return ({"start": start, "end": boundary}, {"start": boundary, "end": end})


def _region_id(search_id: str, start: int, end: int) -> str:
    return "bisect_region_" + _sha256({
        "search_id": search_id,
        "root_relative_start": start,
        "root_relative_end": end,
    })[:24]


def _root_address(run: Mapping, source_span_id: str) -> tuple[dict | None, dict | None, str | None]:
    resolved = span_bridge.resolve_span_address(dict(run), source_span_id)
    if not isinstance(resolved, Mapping) or not resolved.get("ok"):
        reason = resolved.get("reason") if isinstance(resolved, Mapping) else None
        code = reason.get("code") if isinstance(reason, Mapping) else None
        return None, None, code or "span_address_not_found_or_drifted"
    span = resolved.get("span")
    if not isinstance(span, Mapping):
        return None, None, "span_address_not_found_or_drifted"
    try:
        from clozn.runs.text_span_addresses import build_persisted_text_span_addresses
        addresses = build_persisted_text_span_addresses(dict(run), privacy="metadata_only").get("addresses", [])
    except Exception:
        addresses = []
    address = next((item for item in addresses
                    if isinstance(item, Mapping) and item.get("address_id") == source_span_id), None)
    if not isinstance(address, Mapping):
        return None, None, "span_address_not_found_or_drifted"
    if address.get("kind") not in {"delivered_message", "attached_source_span"}:
        return None, None, "span_basis_unsupported"
    resolution = address.get("resolution") if isinstance(address.get("resolution"), Mapping) else {}
    canonical = resolution.get("canonical") if isinstance(resolution.get("canonical"), Mapping) else {}
    start, end, message_index = span.get("start"), span.get("end"), span.get("message_index")
    if not all(_is_int(value) for value in (start, end, message_index)) or end <= start:
        return None, None, "span_unavailable"
    return dict(span), {
        "kind": address.get("kind"),
        "basis_sha256": canonical.get("basis_sha256"),
        "span_sha256": canonical.get("span_sha256"),
        "length": end - start,
    }, None


def _region_projection(search_id: str, root_start: int, root_end: int, *, message_start: int,
                       message_index: int, message_text: str, parent_region_id: str | None,
                       depth: int) -> dict:
    absolute_start, absolute_end = message_start + root_start, message_start + root_end
    return {
        "region_id": _region_id(search_id, root_start, root_end),
        "parent_region_id": parent_region_id,
        "depth": depth,
        "root_interval": {
            "start": root_start, "end": root_end,
            "unit": "unicode_code_points", "interval": "half_open",
        },
        "message_interval": {
            "start": absolute_start, "end": absolute_end,
            "unit": "unicode_code_points", "interval": "half_open",
        },
        "code_points": root_end - root_start,
        "sha256": geometry.text_sha256(message_text[absolute_start:absolute_end]),
    }


def _decode(run: Mapping) -> tuple[dict | None, Any, str | None]:
    config = recorded_sampling_config(dict(run))
    if config is False:
        return {"source": "recorded_greedy", "matches_recorded_decode": True}, False, "recorded_greedy"
    if isinstance(config, Mapping) and _is_int(config.get("seed")):
        return {
            "source": "recorded_fixed_sampling",
            "matches_recorded_decode": True,
            "sampling": deepcopy(dict(config)),
        }, deepcopy(dict(config)), "recorded_fixed_sampling"
    return None, None, None


def _influence_projection(measured: Mapping) -> dict:
    return {
        key: deepcopy(measured[key])
        for key in ("source_span_id", "answer_span_id", "measurement_state", "measurement_reason",
                    "effect", "evidence_state", "clears_floor", "delta_nats", "abs_delta_nats")
        if key in measured
    }


def plan_context_bisect(
    run: dict,
    *,
    source_span_id: str | None = None,
    answer_span_id: str | None = None,
    recipe: str = DEFAULT_RECIPE,
    max_depth: int = DEFAULT_MAX_DEPTH,
    max_runs: int = DEFAULT_MAX_RUNS,
    max_seconds: float = DEFAULT_MAX_SECONDS,
    min_region_chars: int = DEFAULT_MIN_REGION_CHARS,
    request: Any = None,
) -> dict:
    """Build a deterministic zero-run Context Bisect plan."""
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run.get("id"):
        raise ContextBisectInputError("invalid_parent", "parent run is unavailable")
    if request is not None:
        influence, options = _request(request)
        source_span_id, answer_span_id = influence["source_span_id"], influence["answer_span_id"]
        recipe, max_depth = options["recipe"], options["max_depth"]
        max_runs, max_seconds = options["max_runs"], options["max_seconds"]
        min_region_chars = options["min_region_chars"]
    else:
        _, options = _request({
            "influence": {"source_span_id": source_span_id, "answer_span_id": answer_span_id},
            "recipe": recipe, "max_depth": max_depth, "max_runs": max_runs,
            "max_seconds": max_seconds, "min_region_chars": min_region_chars,
        })
    if not isinstance(source_span_id, str) or not isinstance(answer_span_id, str):
        raise ContextBisectInputError("invalid_influence", "source_span_id and answer_span_id are required")

    decode, _sampling_override, decode_source = _decode(run)
    fingerprint = parent_execution_fingerprint(run)
    search_id = "context_bisect_" + _sha256({
        "parent_execution_fingerprint": fingerprint,
        "source_span_id": source_span_id,
        "answer_span_id": answer_span_id,
        "recipe": recipe,
        "max_depth": max_depth,
        "min_region_chars": min_region_chars,
        "decode_regime": {
            "source": decode_source,
            "sampling": deepcopy(_sampling_override) if isinstance(_sampling_override, Mapping) else None,
        },
    })[:24]
    measured = _measurement(run, source_span_id, answer_span_id)
    response, answer_reason = geometry.resolve_answer_text(dict(run))
    source_span, source_address, source_reason = _root_address(run, source_span_id)
    target = None
    if isinstance(response, str) and isinstance(measured.get("answer_interval"), Mapping):
        interval = measured["answer_interval"]
        target = {
            "basis": "recorded_answer",
            "start": interval.get("start"), "end": interval.get("end"),
            "unit": "unicode_code_points", "interval": "half_open",
            "basis_sha256": geometry.text_sha256(response),
        }

    span_resolution = {"state": "unavailable", "reason": source_reason or "span_unavailable"}
    root_region = None
    messages = run.get("messages")
    if isinstance(source_span, Mapping) and isinstance(source_address, Mapping):
        message_index = source_span.get("message_index")
        start, end = source_span.get("start"), source_span.get("end")
        if (isinstance(messages, list) and _is_int(message_index) and 0 <= message_index < len(messages)
                and isinstance(messages[message_index], Mapping)
                and isinstance(messages[message_index].get("content"), str)):
            message_text = messages[message_index]["content"]
            if 0 <= start < end <= len(message_text):
                span_resolution = {
                    "state": "available", "basis": "message",
                    "message_index": message_index, "start": start, "end": end,
                    "length": end - start, "unit": "unicode_code_points", "interval": "half_open",
                    "basis_sha256": source_address.get("basis_sha256"),
                    "span_sha256": source_address.get("span_sha256"),
                }
                root_region = _region_projection(
                    search_id, 0, end - start, message_start=start, message_index=message_index,
                    message_text=message_text, parent_region_id=None, depth=0,
                )

    execution_state = "ready"
    execution_reason = None
    if measured.get("measurement_state") != "available" or measured.get("measurement_reason"):
        execution_state, execution_reason = "unavailable", measured.get("measurement_reason", "influence_link_not_found")
    elif target is None:
        execution_state, execution_reason = "unavailable", answer_reason or "answer_span_unavailable"
    elif span_resolution.get("state") != "available":
        execution_state, execution_reason = "unavailable", span_resolution.get("reason", "span_unavailable")
    elif decode is None:
        execution_state, execution_reason = "unavailable", "sampler_provenance_unavailable"
    elif parent_runtime_projection(run) is None:
        execution_state, execution_reason = "unavailable", "parent_runtime_identity_unavailable"
    elif not isinstance(messages, list):
        execution_state, execution_reason = "unavailable", "message_basis_unavailable"
    elif not isinstance(run.get("behavior", {}).get("active_dials", {}) if isinstance(run.get("behavior"), Mapping) else {}, Mapping):
        execution_state, execution_reason = "unavailable", "recorded_steering_malformed"

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "search_id": search_id,
        "parent_run_id": run["id"],
        "parent_execution_fingerprint": fingerprint,
        "influence": _influence_projection(measured),
        "intervention": {"kind": "neutralize", "recipe": FILLER_RECIPE},
        "target": target or {"basis": "recorded_answer", "state": "unavailable"},
        "span_resolution": span_resolution,
        "root_region": root_region or {"state": "unavailable"},
        "search": {
            "algorithm": "bounded_binary_partition_v1",
            "order": "breadth_first",
            "recipe": recipe,
            "max_depth": max_depth,
            "max_runs": max_runs,
            "max_seconds": float(max_seconds),
            "min_region_chars": min_region_chars,
            "criterion": "divergence_before_or_within_target",
            "initial_state": "terminal_root" if root_region is not None and not split_region(
                messages[source_span["message_index"]]["content"][source_span["start"]:source_span["end"]],
                0, root_region["code_points"], min_region_chars) else "ready",
        },
        "execution": {
            "state": execution_state,
            "requires_generation": True,
            "requires_full_reprefill": True,
            "fidelity": "controlled_regeneration",
            "live_state": "not_checked",
            "decode_regime": decode or {"state": "unavailable"},
        },
    }
    plan["root_basis_sha256"] = (span_resolution.get("basis_sha256")
                                  if isinstance(span_resolution, Mapping) else None)
    plan["root_span_sha256"] = (span_resolution.get("span_sha256")
                                 if isinstance(span_resolution, Mapping) else None)
    if execution_reason:
        plan["execution"]["reason"] = execution_reason
    schemas.validate(plan, PLAN_SCHEMA_VERSION)
    return plan


build_context_bisect_plan = plan_context_bisect


__all__ = [
    "DEFAULT_MAX_DEPTH", "DEFAULT_MAX_RUNS", "DEFAULT_MAX_SECONDS", "DEFAULT_MIN_REGION_CHARS",
    "FILLER_RECIPE", "MAX_DEPTH", "MAX_RUNS", "MAX_SECONDS", "MAX_REGION_CHARS",
    "MIN_DEPTH", "MIN_RUNS", "MIN_SECONDS", "MIN_REGION_CHARS", "PLAN_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION", "ContextBisectInputError", "build_context_bisect_plan",
    "plan_context_bisect", "split_region",
]
