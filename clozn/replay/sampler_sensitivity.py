"""Bounded exact-only sampler sensitivity probing.

This module has two deliberately separate surfaces.  ``plan_sampler_sensitivity`` reads only the
immutable parent evidence and deterministically describes a nearby sampler neighborhood.  The
executor then dispatches each planned change through the existing exact Execution Fork primitives.
There is no sampler application, model call, scoring pass, replay fallback, or durable probe object
here.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import hashlib
import json
import math

from clozn import schemas
from clozn.replay.controlled import recorded_sampling_config
from clozn.experiments.execution_facts import parent_execution_fingerprint


PLAN_SCHEMA_VERSION = "clozn.sampler-sensitivity-plan.v1"
SCHEMA_VERSION = "clozn.sampler-sensitivity.v1"
DEFAULT_POSITION = 0
DEFAULT_RECIPE = "nearby_v1"
DEFAULT_SEED_PROBES = 0
MAX_PARAMETER_PROBES = 4
MAX_SEED_PROBES = 2


class SamplerSensitivityInputError(ValueError):
    """Typed malformed-request error for the HTTP layer."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _is_int(value, minimum: int | None = None) -> bool:
    return (
        isinstance(value, int) and not isinstance(value, bool)
        and (minimum is None or value >= minimum)
    )


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _sha(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _reason(code: str, message: str) -> dict:
    return {"code": str(code), "message": str(message)}


def _trace_tokens(parent: Mapping) -> list[str]:
    trace = parent.get("trace")
    tokens = trace.get("tokens") if isinstance(trace, Mapping) else None
    if not isinstance(tokens, list) or not tokens or not all(isinstance(piece, str) for piece in tokens):
        raise SamplerSensitivityInputError(
            "invalid_position", "parent has no recorded response token boundaries")
    return tokens


def _position(parent: Mapping, value) -> int:
    if not _is_int(value, 0):
        raise SamplerSensitivityInputError(
            "invalid_position", "position must be a non-negative integer")
    tokens = _trace_tokens(parent)
    if value >= len(tokens):
        raise SamplerSensitivityInputError(
            "invalid_position", "position is outside the recorded response token range")
    return value


def _recipe(recipe: str) -> dict:
    if recipe != DEFAULT_RECIPE:
        raise SamplerSensitivityInputError(
            "invalid_recipe", "only the nearby_v1 sampler recipe is supported")
    return {
        "id": DEFAULT_RECIPE,
        "temperature_multiplier_down": 0.8,
        "temperature_multiplier_up": 1.2,
        "top_p_delta": 0.05,
    }


def _seed_count(seed_probes) -> int:
    if not _is_int(seed_probes, 0) or seed_probes > MAX_SEED_PROBES:
        raise SamplerSensitivityInputError(
            "invalid_seed_probes", "seed_probes must be an integer from 0 to 2")
    return seed_probes


def _canonical_number(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 6)
    return value


def _baseline_projection(config) -> dict:
    if config is False:
        return {"mode": "greedy"}
    if not isinstance(config, Mapping):
        return {"mode": "unavailable"}
    return {
        "mode": "sample",
        "temperature": _canonical_number(config.get("temperature")),
        "top_p": _canonical_number(config.get("top_p")),
        "top_k": config.get("top_k"),
        "rep_penalty": _canonical_number(config.get("repeat_penalty")),
        "seed": config.get("seed"),
    }


def _sampled_seed_missing(parent: Mapping) -> bool:
    """Distinguish an explicitly sampled run missing only its fixed seed from other provenance gaps."""
    meta = parent.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    decode = meta.get("decode")
    decode = decode if isinstance(decode, Mapping) else {}
    source = {**meta, **decode}
    mode = source.get("mode") or source.get("sampler_mode") or source.get("sampling")
    if mode not in {"sample", "sampling"} or source.get("seed") is not None:
        return False
    return (
        all(source.get(name) is not None for name in ("temperature", "top_p", "top_k"))
        and (source.get("repeat_penalty") is not None
             or source.get("repetition_penalty") is not None)
    )


def _valid_sampled_config(config: Mapping) -> bool:
    """Validate only the recorded fields needed to define an exact sampled neighborhood."""
    return (
        _finite(config.get("temperature")) and float(config["temperature"]) > 0
        and _finite(config.get("top_p")) and 0 <= float(config["top_p"]) <= 1
        and _is_int(config.get("top_k"), 0)
        and _finite(config.get("repeat_penalty")) and float(config["repeat_penalty"]) > 0
        and _is_int(config.get("seed"), 0)
    )


def _probe_id(fingerprint: str, position: int, change: Mapping) -> str:
    return "sampler_probe_" + _sha({
        "parent_fingerprint_sha256": fingerprint,
        "position": position,
        "change": dict(change),
    })[:24]


def _test_id(fingerprint: str, position: int, recipe: str, seed_probes: int) -> str:
    return "sampler_sensitivity_" + _sha({
        "parent_fingerprint_sha256": fingerprint,
        "position": position,
        "recipe": recipe,
        "seed_probes": seed_probes,
    })[:24]


def _seed_for(fingerprint: str, position: int, ordinal: int, recipe: str,
              parent_seed: int, prior: set[int]) -> int:
    subject = {
        "parent_fingerprint_sha256": fingerprint,
        "position": position,
        "kind": "seed_probe",
        "ordinal": ordinal,
        "recipe": recipe,
    }
    attempt = 0
    while True:
        candidate_subject = dict(subject)
        if attempt:
            candidate_subject["attempt"] = attempt
        digest = hashlib.sha256(
            json.dumps(candidate_subject, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).digest()
        candidate = int.from_bytes(digest[:4], "big", signed=False)
        if candidate != parent_seed and candidate not in prior:
            return candidate
        attempt += 1


def _parameter_probe(axis: str, direction: str, value) -> dict:
    change = {"type": "sampling", axis: _canonical_number(value)}
    return {
        "kind": "parameter",
        "axis": axis,
        "direction": direction,
        "change": change,
    }


def _seed_probe(ordinal: int, seed: int) -> dict:
    return {
        "kind": "seed",
        "axis": "seed",
        "direction": "alternate",
        "ordinal": ordinal,
        "change": {"type": "sampling", "seed": seed},
    }


def _variants(parent: Mapping, position: int, recipe: str, seed_probes: int,
              config: Mapping) -> list[dict]:
    baseline_t = float(config["temperature"])
    baseline_p = float(config["top_p"])
    values = [
        ("temperature", "down", round(baseline_t * 0.8, 6)),
        ("temperature", "up", round(baseline_t * 1.2, 6)),
        ("top_p", "down", round(max(0.0, baseline_p - 0.05), 6)),
        ("top_p", "up", round(min(1.0, baseline_p + 0.05), 6)),
    ]
    probes = []
    for axis, direction, value in values:
        baseline = config[axis]
        if not _finite(value) or value == round(float(baseline), 6) or (
            axis == "temperature" and value <= 0
        ):
            continue
        probes.append(_parameter_probe(axis, direction, value))

    parent_seed = config.get("seed")
    used = set()
    if seed_probes:
        for ordinal in range(seed_probes):
            seed = _seed_for(
                parent_execution_fingerprint(parent), position, ordinal, recipe,
                parent_seed, used)
            used.add(seed)
            probes.append(_seed_probe(ordinal, seed))
    return probes


def _unavailable_plan(parent: Mapping, fingerprint: str, position: int, recipe: dict,
                      seed_probes: int, config, code: str, message: str) -> dict:
    test_id = _test_id(fingerprint, position, recipe["id"], seed_probes)
    document = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "test_id": test_id,
        "run_id": parent["id"],
        "parent_fingerprint_sha256": fingerprint,
        "position": position,
        "seed_probes": seed_probes,
        "baseline_sampler": _baseline_projection(config),
        "recipe": recipe,
        "probes": [],
        "execution": {
            "state": "unavailable",
            "fidelity": "exact_required",
            "live_state": "not_checked",
            "reason": code,
            "reasons": [_reason(code, message)],
        },
    }
    schemas.validate(document, PLAN_SCHEMA_VERSION)
    return document


def plan_sampler_sensitivity(parent_run: dict, *, position: int = DEFAULT_POSITION,
                             recipe: str = DEFAULT_RECIPE,
                             seed_probes: int = DEFAULT_SEED_PROBES) -> dict:
    """Build a deterministic, read-only exact sampler probe plan."""
    if not isinstance(parent_run, Mapping) or not isinstance(parent_run.get("id"), str) \
            or not parent_run.get("id"):
        raise SamplerSensitivityInputError("invalid_parent", "parent run id is unavailable")
    position = _position(parent_run, position)
    recipe_projection = _recipe(recipe)
    seed_probes = _seed_count(seed_probes)
    fingerprint = parent_execution_fingerprint(parent_run)
    config = recorded_sampling_config(dict(parent_run))
    if config is False:
        return _unavailable_plan(
            parent_run, fingerprint, position, recipe_projection, seed_probes, config,
            "greedy_baseline_no_sampling_neighborhood",
            "greedy decoding has no nearby sampled parameter neighborhood",
        )
    if not isinstance(config, Mapping):
        return _unavailable_plan(
            parent_run, fingerprint, position, recipe_projection, seed_probes, config,
            "sampled_seed_unavailable" if _sampled_seed_missing(parent_run)
            else "sampler_provenance_unavailable",
            "the recorded sampler configuration is incomplete",
        )
    required = ("temperature", "top_p", "top_k", "repeat_penalty", "seed")
    if any(config.get(name) is None for name in required):
        return _unavailable_plan(
            parent_run, fingerprint, position, recipe_projection, seed_probes, config,
            "sampled_seed_unavailable" if config.get("seed") is None else "sampler_provenance_unavailable",
            "the sampled parent does not contain a complete fixed sampler configuration",
        )
    if not _valid_sampled_config(config):
        return _unavailable_plan(
            parent_run, fingerprint, position, recipe_projection, seed_probes, config,
            "sampler_provenance_unavailable",
            "the recorded sampler fields are outside the exact sampled configuration contract",
        )
    probes = _variants(parent_run, position, recipe, seed_probes, config)
    if not probes:
        return _unavailable_plan(
            parent_run, fingerprint, position, recipe_projection, seed_probes, config,
            "no_nearby_sampler_variants",
            "the recorded sampler has no distinct nearby_v1 variants",
        )
    out_probes = []
    for probe in probes:
        change = probe["change"]
        item = deepcopy(probe)
        item["probe_id"] = _probe_id(fingerprint, position, change)
        out_probes.append(item)
    document = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "test_id": _test_id(fingerprint, position, recipe, seed_probes),
        "run_id": parent_run["id"],
        "parent_fingerprint_sha256": fingerprint,
        "position": position,
        "seed_probes": seed_probes,
        "baseline_sampler": _baseline_projection(config),
        "recipe": recipe_projection,
        "probes": out_probes,
        "execution": {
            "state": "ready",
            "fidelity": "exact_required",
            "live_state": "not_checked",
        },
    }
    schemas.validate(document, PLAN_SCHEMA_VERSION)
    return document


def _safe_reasons(raw, default_code="probe_unavailable", default_message="sampler probe unavailable"):
    reasons = []
    for item in raw or []:
        if isinstance(item, Mapping) and isinstance(item.get("code"), str) and item["code"]:
            reasons.append(_reason(item["code"], item.get("message") or default_message))
    return reasons or [_reason(default_code, default_message)]


def _response(run: Mapping) -> str | None:
    value = run.get("response")
    if isinstance(value, str):
        return value
    trace = run.get("trace")
    tokens = trace.get("tokens") if isinstance(trace, Mapping) else None
    if isinstance(tokens, list) and all(isinstance(piece, str) for piece in tokens):
        return "".join(tokens)
    return None


def _comparison(parent: Mapping, child: Mapping, position: int) -> dict:
    from clozn.analysis.model_diff import diff_runs
    from clozn.analysis.comparison_projection import comparison_projection_from_diff

    diff = diff_runs(dict(parent), dict(child))
    if not diff.get("ok"):
        return {"state": "comparison_unavailable"}
    parent_text = _response(parent)
    child_text = _response(child)
    if parent_text is not None and child_text is not None and parent_text == child_text:
        return {
            "state": "identical",
            "first_divergence_view": deepcopy(diff.get("first_divergence_view") or {
                "schema_version": "clozn.first-divergence-view.v1", "state": "identical"
            }),
        }
    if not diff.get("trace_available"):
        return {
            "state": "trace_unavailable" if parent_text is not None and child_text is not None
            else "comparison_unavailable",
            "first_divergence_view": deepcopy(diff.get("first_divergence_view") or {
                "schema_version": "clozn.first-divergence-view.v1", "state": "trace_unavailable"
            }),
        }
    first = diff.get("first_divergence")
    if not isinstance(first, Mapping) or not _is_int(first.get("index"), 0):
        return {"state": "comparison_unavailable"}
    absolute = first["index"]
    if absolute < position:
        return {
            "state": "comparison_unavailable",
            "contract_failure": "divergence_before_probe_boundary",
            "first_divergence_position": absolute,
        }
    projected = comparison_projection_from_diff(parent, child, diff)
    projected.update({
        "state": "diverged",
        "first_divergence_position": absolute,
        "divergence_offset_from_probe": absolute - position,
    })
    return projected


def _sampler_state(run: Mapping) -> dict | None:
    config = recorded_sampling_config(dict(run))
    if not isinstance(config, Mapping):
        return None
    return {
        "temperature": config.get("temperature"),
        "top_p": config.get("top_p"),
        "top_k": config.get("top_k"),
        "rep_penalty": config.get("repeat_penalty"),
        "seed": config.get("seed"),
    }


def _verify_sampler(baseline: Mapping, child: Mapping, probe: Mapping) -> tuple[bool, dict | None]:
    actual = _sampler_state(child)
    if actual is None:
        return False, _reason("resolved_sampler_unavailable", "the child omitted resolved sampler evidence")
    change = probe["change"]
    axis = probe["axis"]
    for key in ("temperature", "top_p", "top_k", "rep_penalty", "seed"):
        if key == axis:
            if actual.get(key) == baseline.get(key):
                return False, _reason("resolved_sampler_mismatch", "the requested sampler field did not change")
            if actual.get(key) != change.get(key):
                return False, _reason("resolved_sampler_mismatch", "the child sampler did not apply the requested value")
        elif actual.get(key) != baseline.get(key):
            return False, _reason("resolved_sampler_mismatch", "an unrelated sampler field changed")
    return True, None


def _classification_summary(probes: list[Mapping], kind: str, planned: int) -> dict:
    selected = [probe for probe in probes if probe.get("kind") == kind]
    completed = [probe for probe in selected if probe.get("state") == "completed"]
    diverged = [probe for probe in completed if (probe.get("comparison") or {}).get("state") == "diverged"]
    identical = [probe for probe in completed if (probe.get("comparison") or {}).get("state") == "identical"]
    comparison_incomplete = [
        probe for probe in completed
        if (probe.get("comparison") or {}).get("state") not in {"diverged", "identical"}
    ]
    if not completed or comparison_incomplete:
        state = "inconclusive"
    elif len(diverged) == len(completed):
        state = "all_completed_probes_diverged"
    elif diverged:
        state = "some_divergence_observed"
    else:
        state = "no_divergence_observed"
    out = {
        "state": state,
        "planned": planned,
        "completed": len(completed),
        "diverged": len(diverged),
        "identical": len(identical),
        "unavailable": sum(probe.get("state") == "unavailable" for probe in selected),
        "not_attempted": sum(probe.get("state") == "not_attempted" for probe in selected),
    }
    divergences = [
        probe["comparison"] for probe in diverged
        if _is_int((probe.get("comparison") or {}).get("first_divergence_position"), 0)
    ]
    if divergences:
        earliest = min(divergences, key=lambda item: item["first_divergence_position"])
        out["earliest_divergence_position"] = earliest["first_divergence_position"]
        out["earliest_divergence_offset_from_probe"] = earliest["divergence_offset_from_probe"]
    return out


def _base_result(plan: Mapping) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "test_id": plan["test_id"],
        "parent_run_id": plan["run_id"],
        "position": plan["position"],
        "baseline_sampler": deepcopy(plan["baseline_sampler"]),
        "recipe": deepcopy(plan["recipe"]),
        "execution": {
            "fidelity": "exact_required",
            "order": "sequential",
            "checkpoint_reused": False,
            "checkpoint_capture": {"state": "not_attempted", "reused_for_probes": False},
        },
        "probes": [],
        "parameter_sensitivity": _classification_summary([], "parameter", 0),
        "seed_sensitivity": {"state": "not_requested", "requested": plan.get("seed_probes", 0)},
        "summary": {
            "status": "unavailable",
            "planned_probes": len(plan.get("probes") or []),
            "completed_probes": 0,
            "children_created": 0,
            "parameter_probes": sum(p.get("kind") == "parameter" for p in plan.get("probes") or []),
            "seed_probes": sum(p.get("kind") == "seed" for p in plan.get("probes") or []),
            "not_attempted_probes": 0,
        },
    }


def _not_attempted(probe: Mapping, code: str, message: str) -> dict:
    return {
        "probe_id": probe["probe_id"],
        "kind": probe["kind"],
        "axis": probe["axis"],
        **({"direction": probe["direction"]} if "direction" in probe else {}),
        "requested_change": {k: v for k, v in probe["change"].items() if k != "type"},
        "state": "not_attempted",
        "reasons": [_reason(code, message)],
        "comparison": None,
    }


def execute_sampler_sensitivity(parent_run: dict, sub, plan: Mapping, *, runtime_identity=None,
                               worker_identity=None, reload_parent=None,
                               cancel_check: Callable[[], bool] | None = None) -> dict:
    """Execute one exact fork per planned probe, sequentially, reusing one checkpoint."""
    if not isinstance(plan, Mapping) or plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError("invalid sampler sensitivity plan")
    schemas.validate(dict(plan), PLAN_SCHEMA_VERSION)
    result = _base_result(plan)
    if not isinstance(parent_run, Mapping) or parent_run.get("id") != plan.get("run_id"):
        result["execution"].update({"state": "unavailable", "reason": "stale_parent"})
        result["summary"]["status"] = "unavailable"
        result["summary"]["reasons"] = [_reason("stale_parent", "the planned parent does not match the requested run")]
        schemas.validate(result, SCHEMA_VERSION)
        return result
    try:
        rebuilt_plan = plan_sampler_sensitivity(
            parent_run,
            position=plan["position"],
            recipe=plan["recipe"]["id"],
            seed_probes=plan["seed_probes"],
        )
    except SamplerSensitivityInputError:
        rebuilt_plan = None
    if rebuilt_plan != dict(plan):
        result["execution"].update({"state": "unavailable", "reason": "stale_plan"})
        result["summary"]["status"] = "unavailable"
        result["summary"]["reasons"] = [_reason("stale_plan", "the sampler sensitivity plan no longer matches recorded parent evidence")]
        schemas.validate(result, SCHEMA_VERSION)
        return result
    if plan.get("execution", {}).get("state") != "ready":
        result["execution"].update({"state": "unavailable", "reason": plan.get("execution", {}).get("reason")})
        result["summary"]["status"] = "unavailable"
        result["summary"]["reasons"] = deepcopy(plan.get("execution", {}).get("reasons") or [])
        schemas.validate(result, SCHEMA_VERSION)
        return result
    if parent_execution_fingerprint(parent_run) != plan.get("parent_fingerprint_sha256"):
        result["execution"].update({"state": "unavailable", "reason": "stale_parent"})
        result["summary"]["status"] = "unavailable"
        result["summary"]["reasons"] = [_reason("stale_parent", "parent execution evidence changed after planning")]
        schemas.validate(result, SCHEMA_VERSION)
        return result

    from clozn.replay.execution_fork import (
        capture_exact_force_token_context, execute_exact_force_token, plan_exact_force_token,
    )

    engine = getattr(sub, "engine", None) if sub is not None else None
    if engine is None or not isinstance(runtime_identity, Mapping) or not isinstance(worker_identity, Mapping):
        result["execution"].update({"state": "unavailable", "reason": "exact_execution_unavailable"})
        result["summary"]["reasons"] = [_reason("exact_execution_unavailable", "exact sampler execution is unavailable")]
        schemas.validate(result, SCHEMA_VERSION)
        return result
    if callable(cancel_check) and cancel_check():
        result["execution"].update({"state": "cancelled", "reason": "execution_cancelled"})
        for probe in plan["probes"]:
            result["probes"].append(_not_attempted(probe, "sampler_sensitivity_cancelled", "sampler sensitivity was cancelled before execution"))
        result["summary"]["status"] = "cancelled"
        result["summary"]["not_attempted_probes"] = len(plan["probes"])
        result["parameter_sensitivity"] = _classification_summary(result["probes"], "parameter", sum(p.get("kind") == "parameter" for p in plan["probes"]))
        result["seed_sensitivity"] = _classification_summary(result["probes"], "seed", sum(p.get("kind") == "seed" for p in plan["probes"]))
        schemas.validate(result, SCHEMA_VERSION)
        return result

    try:
        capture = capture_exact_force_token_context(
            parent_run, engine, runtime_identity=dict(runtime_identity), worker_identity=dict(worker_identity))
    except Exception:
        capture = {"status": "ineligible", "reason": _reason("checkpoint_capture_unavailable", "an exact sampler checkpoint could not be captured")}
    checkpoint = capture.get("checkpoint_reference") if capture.get("status") == "available" else None
    if not isinstance(checkpoint, Mapping):
        reason = (capture.get("reason") if isinstance(capture, Mapping) else None) or _reason(
            "checkpoint_capture_unavailable", "an exact sampler checkpoint could not be captured")
        result["execution"].update({"state": "unavailable", "reason": reason.get("code")})
        result["execution"]["checkpoint_capture"] = {"state": "unavailable", "reused_for_probes": False}
        result["summary"]["reasons"] = [deepcopy(reason)]
        for probe in plan["probes"]:
            result["probes"].append(_not_attempted(probe, reason.get("code", "checkpoint_unavailable"), reason.get("message", "exact checkpoint unavailable")))
        result["summary"]["status"] = "unavailable"
        result["summary"]["not_attempted_probes"] = len(plan["probes"])
        result["parameter_sensitivity"] = _classification_summary(result["probes"], "parameter", sum(p.get("kind") == "parameter" for p in plan["probes"]))
        result["seed_sensitivity"] = _classification_summary(result["probes"], "seed", sum(p.get("kind") == "seed" for p in plan["probes"]))
        schemas.validate(result, SCHEMA_VERSION)
        return result

    result["execution"]["state"] = "available"
    result["execution"]["checkpoint_reused"] = True
    result["execution"]["checkpoint_capture"] = {"state": "available", "reused_for_probes": True}
    baseline = plan["baseline_sampler"]
    import clozn.runs.store as runlog
    reload_parent = reload_parent or runlog.get_run
    stop_reason = None
    cancelled = False
    for index, probe in enumerate(plan["probes"]):
        if stop_reason is not None:
            result["probes"].append(_not_attempted(probe, *stop_reason))
            continue
        if callable(cancel_check) and cancel_check():
            cancelled = True
            for remaining in plan["probes"][index:]:
                result["probes"].append(_not_attempted(remaining, "sampler_sensitivity_cancelled", "sampler sensitivity was cancelled between probes"))
            break
        request = {"position": plan["position"], "change": deepcopy(probe["change"])}
        try:
            exact_plan = plan_exact_force_token(
                parent_run, request, checkpoint_reference=dict(checkpoint),
                runtime_identity=dict(runtime_identity), worker_identity=dict(worker_identity))
            if exact_plan.get("classification") != "exact_execution_fork":
                item = _not_attempted(probe, "exact_execution_unavailable", "the sampler probe did not satisfy exact-fork prerequisites")
                item["state"] = "unavailable"
                item["reasons"] = _safe_reasons(exact_plan.get("reasons"), "exact_execution_unavailable", "the sampler probe did not satisfy exact-fork prerequisites")
                result["probes"].append(item)
                if item["reasons"][0]["code"] in {"checkpoint_expired", "stale_worker_generation", "runtime_identity_mismatch", "checkpoint_invalidated"}:
                    stop_reason = ("shared_exact_precondition_failed", "later sampler probes were not attempted after a shared exact precondition failed")
                continue
            execution = execute_exact_force_token(
                parent_run, exact_plan, engine,
                runtime_identity=dict(runtime_identity), worker_identity=dict(worker_identity),
                reload_parent=reload_parent, cancel_check=cancel_check)
            receipt = execution.get("receipt") or {}
            child = execution.get("child")
            if receipt.get("phase") != "completed" or not isinstance(child, Mapping) or not child.get("id"):
                item = _not_attempted(probe, "exact_execution_failed", "the exact sampler probe did not complete")
                item["state"] = "unavailable"
                item["reasons"] = _safe_reasons(receipt.get("reasons"), "exact_execution_failed", "the exact sampler probe did not complete")
                item["execution"] = {"outcome": "unavailable", "proof_status": (receipt.get("exactness") or {}).get("proof_status", "failed")}
                if receipt.get("execution_id"):
                    item["execution"]["execution_id"] = receipt["execution_id"]
                result["probes"].append(item)
                failure_code = item["reasons"][0]["code"]
                if failure_code in {"stale_plan", "checkpoint_expired", "stale_worker_generation", "worker_generation_changed", "checkpoint_invalidated", "execution_cancelled"}:
                    if failure_code == "execution_cancelled":
                        cancelled = True
                        stop_reason = ("sampler_sensitivity_cancelled", "later sampler probes were not attempted after cancellation")
                    else:
                        stop_reason = ("shared_exact_precondition_failed", "later sampler probes were not attempted after a shared exact precondition failed")
                continue
            child = dict(child)
            verification, verification_reason = _verify_sampler(baseline, child, probe)
            item = {
                "probe_id": probe["probe_id"],
                "kind": probe["kind"],
                "axis": probe["axis"],
                **({"direction": probe["direction"]} if "direction" in probe else {}),
                "requested_change": {k: v for k, v in probe["change"].items() if k != "type"},
                "state": "completed" if verification else "unavailable",
                "child_run_id": child["id"],
                "execution": {
                    "outcome": "exact_execution_fork",
                    "proof_status": (receipt.get("exactness") or {}).get("proof_status", "confirmed"),
                    "unchanged_control": (receipt.get("unchanged_control") or {}).get("status", "matched"),
                },
                "resolved_sampler": _sampler_state(child),
                "reasons": [] if verification else [verification_reason],
                "comparison": _comparison(parent_run, child, plan["position"]) if verification else None,
            }
            if receipt.get("execution_id"):
                item["execution"]["execution_id"] = receipt["execution_id"]
            result["probes"].append(item)
        except Exception:
            result["probes"].append(_not_attempted(probe, "exact_execution_failed", "the exact sampler probe failed"))

    if cancelled:
        status = "partial_cancelled" if any(p.get("state") == "completed" for p in result["probes"]) else "cancelled"
        result["execution"]["state"] = "cancelled"
    elif any(p.get("state") == "not_attempted" for p in result["probes"]):
        status = "partial" if any(p.get("state") == "completed" for p in result["probes"]) else "unavailable"
    elif any(p.get("state") == "completed" for p in result["probes"]):
        status = "completed" if all(p.get("state") == "completed" for p in result["probes"]) else "partial"
    else:
        status = "unavailable"
    result["parameter_sensitivity"] = _classification_summary(
        result["probes"], "parameter", sum(p.get("kind") == "parameter" for p in plan["probes"]))
    result["seed_sensitivity"] = (
        _classification_summary(result["probes"], "seed", sum(p.get("kind") == "seed" for p in plan["probes"]))
        if plan.get("seed_probes") else {"state": "not_requested", "requested": 0}
    )
    completed = sum(p.get("state") == "completed" for p in result["probes"])
    children = sum(bool(p.get("child_run_id")) for p in result["probes"])
    result["summary"].update({
        "status": status,
        "completed_probes": completed,
        "children_created": children,
        "not_attempted_probes": sum(p.get("state") == "not_attempted" for p in result["probes"]),
        "unavailable_probes": sum(p.get("state") == "unavailable" for p in result["probes"]),
    })
    schemas.validate(result, SCHEMA_VERSION)
    return result


__all__ = [
    "DEFAULT_POSITION", "DEFAULT_RECIPE", "DEFAULT_SEED_PROBES", "MAX_PARAMETER_PROBES",
    "MAX_SEED_PROBES", "PLAN_SCHEMA_VERSION", "SCHEMA_VERSION",
    "SamplerSensitivityInputError", "build_sampler_sensitivity_plan",
    "execute_sampler_sensitivity", "plan_sampler_sensitivity",
]

# Naming symmetry for callers that use the repository's ``build_*_plan`` convention.
build_sampler_sensitivity_plan = plan_sampler_sensitivity
