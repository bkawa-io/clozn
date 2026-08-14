"""Run-scoped planning, cache binding, and execution for Minimal Context jobs."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from collections.abc import Mapping
from typing import Any, Callable

from clozn import schemas
from clozn.receipts.context_dependence import ContextDependenceStudy
from clozn.runs.answer_preservation import (
    ExactAnswerPreservationError,
    ExactAnswerPreservationStudy,
    assess_exact_eligibility,
)
from clozn.runs.context_search_universe import (
    ContextSearchUniverseError,
    plan_context_search_universe,
)
from clozn.runs.minimal_context import (
    EXACT_PRESERVATION_KIND,
    PRESERVATION_KIND,
    PRESERVATION_TARGET,
    SEARCH_METHOD,
    MinimalContextError,
    run_minimal_context_search,
)
from clozn.receipts import rederive


SCHEMA = "clozn.minimal-context-result.v1"
DEFAULT_MAX_UNITS = 50
DEFAULT_SEARCH_PROBE_BUDGET = 128
DEFAULT_CERTIFICATION_PROBE_BUDGET = 2000
METHOD_VERSION = SEARCH_METHOD


class MinimalContextExecutionError(ValueError):
    """A typed request, capability, or run-binding failure."""

    def __init__(self, message: str, *, code: str = "minimal_context_execution_invalid", status: int = 409):
        super().__init__(message)
        self.code = code
        self.status = status


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str,
    ).encode("utf-8")).hexdigest()


def _int(value: Any, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or (value <= 0 if positive else value < 0):
        requirement = "a positive" if positive else "a non-negative"
        raise MinimalContextExecutionError(f"{name} must be {requirement} integer", status=400)
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MinimalContextExecutionError(f"{name} must be a finite number", status=400)
    if float(value) < 0:
        raise MinimalContextExecutionError(f"{name} must be non-negative", status=400)
    return float(value)


def normalize_request(body: Any) -> dict:
    """Validate only theorem-bearing request fields before worker selection."""
    if not isinstance(body, Mapping):
        raise MinimalContextExecutionError("request body must be an object", status=400)
    if "certificate" in body or "certificate_kind" in body:
        raise MinimalContextExecutionError(
            "certificate kind is backend-derived and cannot be requested", status=400
        )
    preservation = body.get("preservation")
    if preservation is None:
        preservation = {"kind": PRESERVATION_KIND, "tolerance_nats": 0.3}
    if not isinstance(preservation, Mapping):
        raise MinimalContextExecutionError("preservation must be an object", status=400)
    kind = preservation.get("kind")
    if kind not in {PRESERVATION_KIND, EXACT_PRESERVATION_KIND}:
        raise MinimalContextExecutionError(
            "preservation.kind must be teacher_forced_likelihood or exact_recorded_output", status=400
        )
    normalized_preservation = {"kind": kind, "target": PRESERVATION_TARGET}
    if kind == PRESERVATION_KIND:
        normalized_preservation["tolerance_nats"] = _finite(
            preservation.get("tolerance_nats", 0.3), "preservation.tolerance_nats"
        )
    elif "tolerance_nats" in preservation:
        raise MinimalContextExecutionError(
            "exact_recorded_output does not accept tolerance_nats", status=400
        )

    universe = body.get("universe")
    if universe is None:
        universe = {}
    if not isinstance(universe, Mapping):
        raise MinimalContextExecutionError("universe must be an object", status=400)
    max_units = _int(universe.get("max_units", DEFAULT_MAX_UNITS), "universe.max_units", positive=True)
    search_budget = _int(body.get("search_probe_budget", DEFAULT_SEARCH_PROBE_BUDGET), "search_probe_budget")
    certification_budget = _int(
        body.get("certification_probe_budget", DEFAULT_CERTIFICATION_PROBE_BUDGET),
        "certification_probe_budget",
    )
    seed = _int(body.get("search_seed", 0), "search_seed")
    refresh = body.get("refresh", False)
    if not isinstance(refresh, bool):
        raise MinimalContextExecutionError("refresh must be a boolean", status=400)
    return {
        "preservation": normalized_preservation,
        "universe": {"max_units": max_units},
        "search_probe_budget": search_budget,
        "certification_probe_budget": certification_budget,
        "search_seed": seed,
        "refresh": refresh,
    }


def planned_universe(run: Mapping[str, Any], request: Mapping[str, Any]) -> dict:
    try:
        manifest = run.get("context_units")
        return plan_context_search_universe(
            run, manifest,
            max_units=request["universe"]["max_units"],
        )
    except (ContextSearchUniverseError, TypeError, KeyError) as exc:
        raise MinimalContextExecutionError(
            f"context search universe is unavailable: {exc}",
            code="minimal_context_universe_unavailable", status=409,
        ) from exc


def _runtime_identity(run: Mapping[str, Any], sub: Any) -> dict:
    current: dict[str, Any] = {}
    for name in ("identity_meta", "run_meta"):
        fn = getattr(sub, name, None)
        if callable(fn):
            try:
                current[name] = deepcopy(dict(fn() or {}))
            except Exception:
                current[name] = {"unavailable": True}
    return {
        "recorded_model": run.get("model"),
        "recorded_substrate": run.get("substrate"),
        "recorded_identity": deepcopy(run.get("identity") if isinstance(run.get("identity"), Mapping) else {}),
        "current_worker": current,
    }


def cache_binding(run: Mapping[str, Any], request: Mapping[str, Any], universe: Mapping[str, Any], sub: Any) -> dict:
    conditions = rederive.with_arm_conditions(dict(run))
    continuation = {
        "continuation_ids": deepcopy(conditions.get("continuation_ids")),
        "response": conditions.get("response"),
        "trace_identity": _digest(run.get("trace") if isinstance(run.get("trace"), Mapping) else {}),
    }
    eligibility = assess_exact_eligibility(run, sub) if request["preservation"]["kind"] == EXACT_PRESERVATION_KIND else None
    return {
        "schema_version": "clozn.minimal-context-cache.v1",
        "run_identity": _digest({
            "run_id": run.get("id"),
            "model": run.get("model"),
            "substrate": run.get("substrate"),
            "identity": run.get("identity"),
            "context_receipt": run.get("context_receipt"),
        }),
        "context_units_manifest_sha256": universe.get("basis_context_units_digest"),
        "search_universe_id": universe.get("universe_id"),
        "continuation_identity": _digest(continuation),
        "runtime_generation_contract_identity": _digest({
            "runtime": (eligibility or {}).get("recorded_runtime") if eligibility else _runtime_identity(run, sub),
            "generation_contract": (eligibility or {}).get("generation_contract") if eligibility else None,
        }),
        "preservation": deepcopy(request["preservation"]),
        "method": METHOD_VERSION,
        "search_probe_budget": request["search_probe_budget"],
        "certification_probe_budget": request["certification_probe_budget"],
        "search_seed": request["search_seed"],
    }


def cache_matches(result: Any, binding: Mapping[str, Any]) -> bool:
    if not isinstance(result, Mapping) or result.get("schema_version") != SCHEMA:
        return False
    if result.get("status") not in {"found", "not_found_within_budget"}:
        return False
    stored = result.get("cache_binding")
    return isinstance(stored, Mapping) and dict(stored) == dict(binding)


def result_summary(result: Mapping[str, Any]) -> dict:
    candidate = result.get("candidate") if isinstance(result.get("candidate"), Mapping) else {}
    certificate = result.get("certificate") if isinstance(result.get("certificate"), Mapping) else {}
    source = result.get("source_universe") if isinstance(result.get("source_universe"), Mapping) else {}
    return {
        "result_id": result.get("result_id"),
        "preservation_kind": (result.get("preservation") or {}).get("kind"),
        "source_count": source.get("source_count"),
        "retained_source_count": candidate.get("retained_source_count"),
        "certificate_kind": certificate.get("kind"),
        "status": result.get("status"),
        "universe_id": source.get("search_universe_id"),
    }


def execute_minimal_context(
    run: Mapping[str, Any],
    sub: Any,
    request: Mapping[str, Any],
    universe: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    checkpoint: Callable[..., Any],
) -> tuple[dict, dict]:
    """Run one immutable study and return result plus separately persisted support evidence."""
    source_ids = list(universe["source_ids"])
    manifest = run.get("context_units")
    kind = request["preservation"]["kind"]
    exact = kind == EXACT_PRESERVATION_KIND
    if exact:
        eligibility = assess_exact_eligibility(run, sub)
        if not eligibility.get("eligible"):
            reason = eligibility.get("reason") or "exact_recorded_output_ineligible"
            raise MinimalContextExecutionError(
                reason, code="minimal_context_exact_unavailable", status=409
            )
        if not callable(getattr(sub, "probe_reference_match", None)):
            raise MinimalContextExecutionError(
                "exact_recorded_output requires direct generation probes",
                code="minimal_context_exact_capability_unavailable", status=503,
            )
        study = ExactAnswerPreservationStudy(run, sub, source_ids=source_ids)
        study_id = study.study_id
    else:
        if not callable(getattr(sub, "score_tokens", None)):
            raise MinimalContextExecutionError(
                "teacher_forced_likelihood requires token scoring",
                code="minimal_context_likelihood_capability_unavailable", status=503,
            )
        study = ContextDependenceStudy(dict(run), sub)
        study_id = None
        if set(source_ids).difference(study.source_ids):
            raise MinimalContextExecutionError(
                "planned search universe is not present in the strict scoring source catalog",
                code="minimal_context_universe_not_scoring_catalog", status=409,
            )

    checkpoint(phase="unchanged_control", completed=0, total=1)
    if exact:
        control = study.ensure_unchanged_control()
        if control.get("status") != "matched":
            raise MinimalContextExecutionError(
                control.get("reason", "unchanged_control_failed"),
                code="minimal_context_unchanged_control_failed", status=409,
            )
    else:
        try:
            study._ensure_baseline()  # the study owns the exact score contract and evidence validation
            study_id = study.document().get("study_id")
        except Exception as exc:
            raise MinimalContextExecutionError(
                f"unchanged teacher-forced control failed: {exc}",
                code="minimal_context_unchanged_control_failed", status=409,
            ) from exc
        if not isinstance(study_id, str):
            raise MinimalContextExecutionError("likelihood study has no stable study_id")
    checkpoint(phase="unchanged_control", completed=1, total=1)

    total = request["search_probe_budget"] + request["certification_probe_budget"]
    phase_name = "searching"
    probe_count = 0

    def phase_callback(name: str, completed: int, phase_total: int) -> None:
        nonlocal phase_name
        phase_name = name
        checkpoint(phase=name, completed=completed, total=phase_total)

    def measure(removed):
        nonlocal probe_count
        # This is the cancellation boundary immediately before every direct
        # deletion/generation probe.  Existing cached evidence never enters it.
        checkpoint(phase=phase_name, completed=probe_count, total=total)
        if exact:
            observation = study.probe_removed_sources(removed)
        else:
            observation = study.measure_removal_effect(removed)
        probe_count += 1
        return observation

    def measure_many(removed_sets):
        nonlocal probe_count
        requested = tuple(tuple(removed) for removed in removed_sets)
        # Cancellation is checked immediately before dispatching a batch. The
        # study's batch seam preserves per-arm evidence identity and uses the
        # serial adapter when the substrate has no native implementation.
        checkpoint(phase=phase_name, completed=probe_count, total=total)
        try:
            if exact:
                observations = study.probe_removed_sources_many(requested)
            else:
                observations = study.measure_removal_effect_many(requested)
        except AttributeError:
            observations = [measure(removed) for removed in requested]
        probe_count += len(observations)
        return observations

    preservation = deepcopy(request["preservation"])
    try:
        result = run_minimal_context_search(
            source_ids,
            measure,
            tolerance_nats=preservation.get("tolerance_nats", 0.0),
            search_probe_budget=request["search_probe_budget"],
            certification_probe_budget=request["certification_probe_budget"],
            search_seed=request["search_seed"],
            run_id=run["id"],
            context_unit_manifest=manifest,
            preservation=preservation,
            answer_preservation_study_id=study_id if exact else None,
            context_dependence_study_id=study_id if not exact else None,
            search_universe_id=universe["universe_id"],
            phase_callback=phase_callback,
            measure_removed_many=measure_many,
        )
    except MinimalContextError as exc:
        raise MinimalContextExecutionError(str(exc), code="minimal_context_search_failed", status=409) from exc
    if result.get("status") == "unavailable":
        raise MinimalContextExecutionError(
            result.get("error", "minimal-context evidence unavailable"),
            code="minimal_context_evidence_unavailable", status=409,
        )
    result["cache_binding"] = deepcopy(dict(binding))
    checkpoint(phase="validating", completed=0, total=1)
    schemas.validate(result, SCHEMA)
    checkpoint(phase="validating", completed=1, total=1)
    support = study.document()
    return result, support


__all__ = [
    "MinimalContextExecutionError",
    "cache_binding",
    "cache_matches",
    "execute_minimal_context",
    "normalize_request",
    "planned_universe",
    "result_summary",
]
