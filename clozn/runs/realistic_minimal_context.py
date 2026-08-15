"""Recorded-run integration and reporting for the realistic evaluation.

This module is deliberately an adapter/reporting layer.  Candidate search stays
in :mod:`clozn.runs.budgeted_reduce`; source identity and message surgery stay
with the ordinary Context Receipt resolver.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from collections.abc import Mapping, Sequence
import statistics
import time
from typing import Any

from clozn.receipts.rederive import with_arm_conditions
from clozn.replay.span_bridge import resolve_context_receipt_source_set
from clozn.runs.answer_preservation import assess_exact_eligibility
from clozn.runs.budgeted_reduce import (
    BEST_VERIFIED,
    CONTROL_FAILED,
    INCLUSION_MINIMUM,
    BudgetedReductionResult,
    Candidate,
    Trial,
)
from clozn.runs.budgeted_reduce_reference import (
    EngineReferenceMatchAdapter,
    run_engine_reference_match_reduction,
)
from clozn.runs.context_search_universe import plan_context_search_universe


SCHEMA = "clozn.minimal-context-eval.v1"
CHECKPOINT_TARGETS = (0, 25, 50, 100, 200)
GEOMETRIC_CHECKPOINT_TARGETS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 200)


@dataclass(frozen=True)
class RecordedRunBinding:
    run: Mapping[str, Any]
    universe: Mapping[str, Any]
    eligibility: Mapping[str, Any]
    adapter: Any


@dataclass(frozen=True)
class EvaluationOutcome:
    status: str
    eligibility: Mapping[str, Any]
    universe: Mapping[str, Any] | None = None
    result: BudgetedReductionResult | None = None
    reason: str | None = None
    metrics: Mapping[str, Any] = field(default_factory=dict)


class _TimedProbeAdapter:
    """Observe adapter dispatch cost without changing reducer behavior."""

    def __init__(self, delegate: Any):
        self._delegate = delegate
        self.substrate = getattr(delegate, "substrate", None)
        self.call_wall_seconds: list[float] = []

    def prepare_candidate(self, retained_ids: tuple[Any, ...]) -> Any:
        return self._delegate.prepare_candidate(retained_ids)

    def probe_many(self, prepared_candidates: Sequence[Any]) -> list[Any]:
        started = time.perf_counter()
        result = self._delegate.probe_many(prepared_candidates)
        self.call_wall_seconds.append(max(0.0, time.perf_counter() - started))
        return result

    def set_probe_context(self, *, stage: str, parent_retained_ids: tuple[Any, ...]) -> None:
        setter = getattr(self._delegate, "set_probe_context", None)
        if callable(setter):
            setter(stage=stage, parent_retained_ids=parent_retained_ids)

    def is_preserving(self, evidence: Any) -> bool:
        return bool(self._delegate.is_preserving(evidence))

    def is_failed(self, evidence: Any) -> bool:
        return bool(self._delegate.is_failed(evidence))


def _render_messages_for_retained(
    run: Mapping[str, Any], universe_ids: tuple[str, ...], retained_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    conditions = with_arm_conditions(dict(run))
    retained = set(retained_ids)
    removed = [source_id for source_id in universe_ids if source_id not in retained]
    if not removed:
        return deepcopy(list(conditions.get("messages") or run.get("messages") or []))
    resolved = resolve_context_receipt_source_set(run, removed)
    messages = resolved.get("messages")
    if not isinstance(messages, list):
        raise ValueError("strict Context Receipt resolver returned no messages")
    return deepcopy(messages)


def bind_engine_recorded_run(
    run: Mapping[str, Any], *, engine: Any, substrate: Any, max_units: int = 50,
    eligibility: Mapping[str, Any] | None = None,
) -> RecordedRunBinding:
    """Bind an existing run to exact probes without regenerating its baseline."""
    if not isinstance(run, Mapping):
        raise ValueError("recorded run must be a mapping")
    manifest = run.get("context_units")
    universe = plan_context_search_universe(run, manifest, max_units=max_units)
    if universe.get("status") != "planned":
        raise ValueError(universe.get("condition", {}).get("message", "search universe unavailable"))
    checked = dict(eligibility) if eligibility is not None else assess_exact_eligibility(run, substrate)
    if not checked.get("eligible"):
        return RecordedRunBinding(run, universe, checked, None)
    conditions = with_arm_conditions(dict(run))
    reference_ids = conditions.get("continuation_ids")
    contract = checked.get("generation_contract")
    if not isinstance(reference_ids, list) or not reference_ids:
        checked = {**checked, "eligible": False, "reason": "missing_exact_recorded_token_ids"}
        return RecordedRunBinding(run, universe, checked, None)
    if not isinstance(contract, Mapping):
        checked = {**checked, "eligible": False, "reason": "generation_contract_incomplete"}
        return RecordedRunBinding(run, universe, checked, None)
    universe_ids = tuple(universe["source_ids"])

    def render_messages(retained_ids: tuple[str, ...]) -> Sequence[Mapping[str, Any]]:
        return _render_messages_for_retained(run, universe_ids, retained_ids)

    explicit: dict[str, Any] = {
        "steer_strengths": deepcopy(conditions.get("steer_strengths") or {}),
    }
    if conditions.get("block") is not None:
        explicit["block"] = conditions["block"]
    adapter = EngineReferenceMatchAdapter(
        engine=engine,
        substrate=substrate,
        render_messages=render_messages,
        reference_token_ids=tuple(int(value) for value in reference_ids),
        generation_contract=deepcopy(dict(contract)),
        explicit_conditions=explicit,
    )
    return RecordedRunBinding(run, universe, checked, adapter)


def evaluate_recorded_run(
    run: Mapping[str, Any], *, adapter: Any, max_counterfactual_probes: int,
    max_units: int = 50, eligibility: Mapping[str, Any] | None = None,
    attempt_inclusion_check: bool = True,
) -> EvaluationOutcome:
    """Evaluate one saved ordinary run through the budgeted reducer.

    ``eligibility`` is injectable for model-free tests.  Production callers
    omit it, so strict exact replay eligibility is assessed against the live
    substrate before any counterfactual probe is dispatched.
    """
    checked = dict(eligibility) if eligibility is not None else assess_exact_eligibility(
        run, getattr(adapter, "substrate", None)
    )
    manifest = run.get("context_units") if isinstance(run, Mapping) else None
    universe = plan_context_search_universe(run, manifest, max_units=max_units)
    if universe.get("status") != "planned":
        return EvaluationOutcome(
            "universe_unavailable", checked, universe,
            reason=(universe.get("condition") or {}).get("message", "search universe unavailable"),
        )
    if not checked.get("eligible"):
        return EvaluationOutcome("exact_unavailable", checked, universe, reason=checked.get("reason"))
    if adapter is None:
        return EvaluationOutcome("exact_unavailable", checked, universe, reason="exact_adapter_unavailable")
    timed_adapter = _TimedProbeAdapter(adapter)
    started = time.perf_counter()
    try:
        result = run_engine_reference_match_reduction(
            timed_adapter,
            universe["source_ids"],
            max_counterfactual_probes,
            attempt_inclusion_check=attempt_inclusion_check,
        )
    except Exception as exc:
        return EvaluationOutcome(
            "exact_unavailable", checked, universe, reason=f"probe_failed: {exc}",
            metrics={"reduction_wall_seconds": max(0.0, time.perf_counter() - started)},
        )
    elapsed = max(0.0, time.perf_counter() - started)
    control_wall = timed_adapter.call_wall_seconds[0] if timed_adapter.call_wall_seconds else 0.0
    counterfactual_batch_wall = timed_adapter.call_wall_seconds[1:]
    metrics = {
        "reduction_wall_seconds": elapsed,
        "exact_control_wall_seconds": control_wall,
        "counterfactual_probe_batch_count": len(counterfactual_batch_wall),
        "counterfactual_probe_batch_wall_ms": [round(value * 1000.0, 6) for value in counterfactual_batch_wall],
        "total_counterfactual_candidate_prompt_tokens": sum(
            int(trial.cost) for trial in result.trials if trial.stage != "control"
        ),
    }
    native_delegate = getattr(timed_adapter, "_delegate", None)
    native_metrics = getattr(native_delegate, "native_parent_anchor_metrics", None)
    native_mismatches = getattr(native_delegate, "native_parent_anchor_parity_mismatches", None)
    if isinstance(native_metrics, list):
        metrics["native_parent_anchor_batch_count"] = len(native_metrics)
        metrics["native_parent_anchor_metrics"] = deepcopy(native_metrics)
        metrics["native_parent_anchor_parity_mismatch_count"] = (
            len(native_mismatches) if isinstance(native_mismatches, list) else 0
        )
    if result.status == CONTROL_FAILED:
        reason = None
        if isinstance(result.control_evidence, Mapping):
            reason = result.control_evidence.get("reason") or result.control_evidence.get("status")
            if result.control_evidence.get("status") == "unavailable":
                return EvaluationOutcome("exact_unavailable", checked, universe, result, reason=reason, metrics=metrics)
        return EvaluationOutcome("control_failed", checked, universe, result, reason=reason, metrics=metrics)
    return EvaluationOutcome("ok", checked, universe, result, metrics=metrics)


def _candidate_dict(candidate: Candidate, original_cost: int) -> dict[str, Any]:
    removed = original_cost - candidate.cost
    return {
        "retained_source_ids": list(candidate.retained_ids),
        "retained_unit_count": len(candidate.retained_ids),
        "cost": candidate.cost,
        "reduction_percent": round(100.0 * removed / original_cost, 6) if original_cost else None,
    }


def _trial_is_failed(trial: Trial) -> bool:
    if trial.preserves:
        return False
    evidence = trial.evidence
    if isinstance(evidence, Mapping):
        if evidence.get("status") == "diverged":
            return True
        if evidence.get("preserves") is False:
            return True
    return isinstance(evidence, bool) and evidence is False


def _best_at_prefix(result: BudgetedReductionResult, probe_count: int) -> Candidate:
    original = result.original_candidate
    candidates = [original]
    for trial in result.trials:
        if trial.stage == "control":
            continue
        # The reducer records one Trial per charged counterfactual arm, in
        # direct ordinal order.  ordinal 1 is the mandatory control.
        if trial.ordinal - 1 > probe_count or not trial.preserves:
            continue
        candidates.append(Candidate(tuple(trial.retained_ids), trial.cost))
    positions = {value: index for index, value in enumerate(original.retained_ids)}
    return min(candidates, key=lambda candidate: (
        candidate.cost,
        len(candidate.retained_ids),
        tuple(positions[value] for value in candidate.retained_ids),
    ))


def _certificate_at_prefix(result: BudgetedReductionResult, candidate: Candidate, probe_count: int) -> str:
    """Return only a certificate supported by direct evidence at this prefix."""
    children = [tuple(value for value in candidate.retained_ids if value != removed)
                for removed in candidate.retained_ids]
    if not children:
        return BEST_VERIFIED
    by_ids = {
        tuple(trial.retained_ids): trial
        for trial in result.trials
        if trial.stage != "control" and trial.ordinal - 1 <= probe_count
    }
    if all(child in by_ids and _trial_is_failed(by_ids[child]) for child in children):
        return INCLUSION_MINIMUM
    return BEST_VERIFIED


def reconstruct_checkpoint(result: BudgetedReductionResult, probe_count: int) -> dict[str, Any]:
    if isinstance(probe_count, bool) or not isinstance(probe_count, int) or probe_count < 0:
        raise ValueError("probe_count must be a non-negative integer")
    original = result.original_candidate
    actual_probe_count = result.budget.used_counterfactual_probes
    run_already_terminated = actual_probe_count < probe_count
    if result.status == CONTROL_FAILED:
        candidate = _candidate_dict(original, original.cost)
        return {
            "probe_count": probe_count,
            "run_already_terminated": run_already_terminated,
            "termination_probe": actual_probe_count if run_already_terminated else None,
            "status": CONTROL_FAILED,
            "best_verified": candidate,
            "cost": candidate["cost"],
            "retained_source_ids": candidate["retained_source_ids"],
            "retained_unit_count": candidate["retained_unit_count"],
            "best_prompt_tokens": candidate["cost"],
            "original_prompt_tokens": original.cost,
            "reduction_tokens": 0,
            "reduction_percent": 0.0,
            "certificate_level": None,
            "certificate_level_at_that_point": None,
        }
    best = _best_at_prefix(result, probe_count)
    candidate = _candidate_dict(best, original.cost)
    return {
        "probe_count": probe_count,
        "run_already_terminated": run_already_terminated,
        "termination_probe": actual_probe_count if run_already_terminated else None,
        "status": "ok",
        "best_verified": candidate,
        "cost": candidate["cost"],
        "retained_source_ids": candidate["retained_source_ids"],
        "retained_unit_count": candidate["retained_unit_count"],
        "best_prompt_tokens": candidate["cost"],
        "original_prompt_tokens": original.cost,
        "reduction_tokens": original.cost - best.cost,
        "reduction_percent": candidate["reduction_percent"],
        "certificate_level": _certificate_at_prefix(result, best, probe_count),
        "certificate_level_at_that_point": _certificate_at_prefix(result, best, probe_count),
    }


def reconstruct_checkpoints(
    result: BudgetedReductionResult, checkpoints: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    targets = tuple(checkpoints) if checkpoints is not None else CHECKPOINT_TARGETS
    if not targets or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in targets):
        raise ValueError("checkpoints must be non-empty non-negative integers")
    return [reconstruct_checkpoint(result, value) for value in targets]


def _accepted_trial(result: BudgetedReductionResult, trial: Trial) -> bool:
    return trial.preserves and any(
        entry.counterfactual_probe_count == trial.ordinal - 1
        and tuple(entry.retained_ids) == tuple(trial.retained_ids)
        for entry in result.trajectory
    )


def _termination_reason(outcome: EvaluationOutcome) -> str:
    if outcome.status == "exact_unavailable":
        return "exact_unavailable"
    if outcome.status == "control_failed":
        return "control_failed"
    if outcome.status != "ok" or outcome.result is None:
        return "fixture_invalid"
    result = outcome.result
    if result.certificate_level == INCLUSION_MINIMUM and result.inclusion_check.complete:
        return "inclusion_minimum"
    if result.budget.exhausted:
        return "budget_exhausted"
    return "search_exhausted_best_verified"


def _improvement_events(result: BudgetedReductionResult) -> list[dict[str, Any]]:
    """Return the control and only strict, directly verified objective changes."""
    if result.status == CONTROL_FAILED:
        return []
    original = result.original_candidate
    events: list[dict[str, Any]] = [{
        "probe_count": 0,
        "stage": "control",
        "retained_source_ids": list(original.retained_ids),
        "retained_unit_count": len(original.retained_ids),
        "cost": original.cost,
        "reduction_tokens": 0,
        "reduction_percent": 0.0,
    }]
    previous = (original.cost, len(original.retained_ids))
    for entry in result.trajectory:
        current = (int(entry.cost), int(entry.retained_unit_count))
        if current >= previous:
            continue
        events.append({
            "probe_count": int(entry.counterfactual_probe_count),
            "stage": entry.stage,
            "retained_source_ids": list(entry.retained_ids),
            "retained_unit_count": entry.retained_unit_count,
            "cost": entry.cost,
            "reduction_tokens": original.cost - entry.cost,
            "reduction_percent": round(
                100.0 * (original.cost - entry.cost) / original.cost, 6
            ) if original.cost else None,
        })
        previous = current
    return events


def _numeric_span_stats(manifest: Mapping[str, Any]) -> dict[str, int | None]:
    spans: list[int] = []
    for unit in manifest.get("units") if isinstance(manifest.get("units"), list) else []:
        value = unit.get("unicode_range") if isinstance(unit, Mapping) else None
        if isinstance(value, (list, tuple)) and len(value) == 2 and all(isinstance(part, int) for part in value):
            if value[1] >= value[0]:
                spans.append(int(value[1] - value[0]))
    if not spans:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(spans),
        "median": int(statistics.median(spans)),
        "max": max(spans),
    }


def _source_geometry(
    run: Mapping[str, Any], universe: Mapping[str, Any] | None,
) -> dict[str, Any]:
    manifest = run.get("context_units") if isinstance(run, Mapping) else {}
    manifest = manifest if isinstance(manifest, Mapping) else {}
    messages = run.get("messages") if isinstance(run, Mapping) else []
    raw_units = manifest.get("units") if isinstance(manifest.get("units"), list) else []
    coverage = universe.get("coverage") if isinstance(universe, Mapping) else {}
    coverage = coverage if isinstance(coverage, Mapping) else {}
    protected = list(manifest.get("protected_message_indices") or [])
    removable_indices = list(coverage.get("removable_message_indices") or sorted({
        unit.get("message_index") for unit in raw_units
        if isinstance(unit, Mapping) and isinstance(unit.get("message_index"), int)
        and unit.get("message_index") not in protected
    }))
    return {
        "message_count": len(messages) if isinstance(messages, list) else 0,
        "raw_context_unit_count": len(raw_units),
        "bounded_search_universe_count": universe.get("source_count") if isinstance(universe, Mapping) else None,
        "protected_message_indices": protected,
        "removable_message_indices": removable_indices,
        "raw_unit_span_chars": _numeric_span_stats(manifest),
        "unit_span_measure": "Unicode character length of each raw Context Unit range; not tokens.",
        "complete_catalog_source_count": coverage.get("complete_catalog_source_count"),
        "bounded_compression_ratio": round(
            len(raw_units) / int(universe["source_count"]), 6
        ) if isinstance(universe, Mapping) and universe.get("source_count") else None,
    }


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.999999)))
    return round(ordered[index], 6)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(float(value) for value in values) / len(values), 6)


def _milestones(report: Mapping[str, Any]) -> dict[str, float | int | None]:
    final = report.get("final") if isinstance(report.get("final"), Mapping) else {}
    eventual = final.get("reduction_percent")
    if not isinstance(eventual, (int, float)) or eventual <= 0:
        return {
            "eventual_reduction_percent": eventual if isinstance(eventual, (int, float)) else None,
            "probe_to_50_percent_eventual_reduction": None,
            "probe_to_90_percent_eventual_reduction": None,
        }
    points = report.get("geometric_checkpoints")
    points = points if isinstance(points, list) else []
    out: dict[str, float | int | None] = {
        "eventual_reduction_percent": round(float(eventual), 6),
        "probe_to_50_percent_eventual_reduction": None,
        "probe_to_90_percent_eventual_reduction": None,
    }
    for point in points:
        if not isinstance(point, Mapping) or point.get("status") != "ok":
            continue
        value = point.get("reduction_percent")
        probe = point.get("probe_count")
        if not isinstance(value, (int, float)) or not isinstance(probe, int):
            continue
        if out["probe_to_50_percent_eventual_reduction"] is None and value >= eventual * 0.5:
            out["probe_to_50_percent_eventual_reduction"] = probe
        if out["probe_to_90_percent_eventual_reduction"] is None and value >= eventual * 0.9:
            out["probe_to_90_percent_eventual_reduction"] = probe
    return out


def serialize_outcome(
    *, case_id: str, description: str, tags: Sequence[str], run: Mapping[str, Any],
    outcome: EvaluationOutcome, timing_seconds: float = 0.0, max_counterfactual_probes: int,
    checkpoints: Sequence[int] | None = None,
    suite: str | None = None,
    baseline_capture_wall_seconds: float | None = None,
) -> dict[str, Any]:
    """Create the stable JSON projection; prompts and evidence payloads stay out of it."""
    manifest = run.get("context_units") if isinstance(run, Mapping) else {}
    protected = manifest.get("protected_message_indices", []) if isinstance(manifest, Mapping) else []
    geometry = _source_geometry(run, outcome.universe)
    report: dict[str, Any] = {
        "case_id": case_id,
        "description": description,
        "case_tags": list(tags),
        "run_id": run.get("id"),
        "status": outcome.status,
        "exact": {
            "eligible": bool(outcome.eligibility.get("eligible")),
            "control_passed": None,
            "reason": outcome.eligibility.get("reason"),
        },
        "exact_replay_eligibility": {
            "eligible": bool(outcome.eligibility.get("eligible")),
            "reason": outcome.eligibility.get("reason"),
            "reasons": list(outcome.eligibility.get("reasons") or []),
            "reference_token_count": outcome.eligibility.get("reference_token_count"),
        },
        "universe": {
            "source_count": outcome.universe.get("source_count") if outcome.universe else None,
            "bounded_search_universe_count": outcome.universe.get("source_count") if outcome.universe else None,
            "universe_id": outcome.universe.get("universe_id") if outcome.universe else None,
            "removable_unit_count": len(outcome.universe.get("source_ids", [])) if outcome.universe else 0,
            "protected_message_indices": list(protected),
            "removable_message_indices": list(geometry["removable_message_indices"]),
        },
        "source_geometry": geometry,
        "raw_context_unit_count": geometry["raw_context_unit_count"],
        "bounded_search_universe_count": geometry["bounded_search_universe_count"],
        "timing": {
            "wall_seconds": round(float(timing_seconds), 6),
            "baseline_capture_wall_seconds": (
                round(float(baseline_capture_wall_seconds), 6)
                if baseline_capture_wall_seconds is not None else None
            ),
            **{
                key: value for key, value in outcome.metrics.items()
                if key in {"exact_control_wall_seconds", "reduction_wall_seconds"}
            },
        },
    }
    if suite is not None:
        report["suite"] = suite
    if outcome.reason:
        report["reason"] = outcome.reason
    report["termination"] = {
        "reason": _termination_reason(outcome),
        "probe": (
            outcome.result.budget.used_counterfactual_probes
            if outcome.result is not None else None
        ),
    }
    if outcome.result is None:
        report["budget"] = {
            "max_counterfactual_probes": max_counterfactual_probes,
            "used_counterfactual_probes": 0,
            "total_counterfactual_probes": 0,
        }
        report["checkpoints"] = []
        report["geometric_checkpoints"] = []
        report["improvement_events"] = []
        report["final"] = None
        return report

    result = outcome.result
    report["control"] = {
        "passed": bool(result.trials and result.trials[0].preserves),
        "status": (result.control_evidence.get("status")
                   if isinstance(result.control_evidence, Mapping) else None),
    }
    report["exact"]["control_passed"] = report["control"]["passed"]
    report["original"] = {
        "cost": result.original_candidate.cost,
        "rendered_prompt_tokens": result.original_candidate.cost,
        "unit_count": len(result.original_candidate.retained_ids),
        "bounded_search_universe_count": len(result.original_candidate.retained_ids),
        "raw_context_unit_count": geometry["raw_context_unit_count"],
    }
    report["budget"] = {
        "max_counterfactual_probes": max_counterfactual_probes,
        "used_counterfactual_probes": result.budget.used_counterfactual_probes,
        "total_counterfactual_probes": result.budget.used_counterfactual_probes,
    }
    report["checkpoints"] = reconstruct_checkpoints(result, checkpoints)
    report["geometric_checkpoints"] = reconstruct_checkpoints(result, GEOMETRIC_CHECKPOINT_TARGETS)
    final = reconstruct_checkpoint(result, result.budget.used_counterfactual_probes)
    report["final"] = final
    report["termination"]["probe"] = result.budget.used_counterfactual_probes
    report["improvement_events"] = _improvement_events(result)
    report["runtime_observability"] = {
        "total_counterfactual_candidate_prompt_tokens": outcome.metrics.get(
            "total_counterfactual_candidate_prompt_tokens", 0
        ),
        "counterfactual_probe_batch_count": outcome.metrics.get("counterfactual_probe_batch_count", 0),
        "counterfactual_probe_batch_wall_ms": outcome.metrics.get(
            "counterfactual_probe_batch_wall_ms", []
        ),
        "mean_probe_batch_wall_ms": _mean(
            outcome.metrics.get("counterfactual_probe_batch_wall_ms", [])
        ) if outcome.metrics.get("counterfactual_probe_batch_wall_ms") else None,
        "median_probe_batch_wall_ms": _percentile(
            outcome.metrics.get("counterfactual_probe_batch_wall_ms", []), 0.5
        ) if outcome.metrics.get("counterfactual_probe_batch_wall_ms") else None,
        "p95_probe_batch_wall_ms": _percentile(
            outcome.metrics.get("counterfactual_probe_batch_wall_ms", []), 0.95
        ) if outcome.metrics.get("counterfactual_probe_batch_wall_ms") else None,
        "native_parent_anchor_batch_count": outcome.metrics.get(
            "native_parent_anchor_batch_count", 0
        ),
        "native_parent_anchor_parity_mismatch_count": outcome.metrics.get(
            "native_parent_anchor_parity_mismatch_count", 0
        ),
        "native_parent_anchor_metrics": outcome.metrics.get(
            "native_parent_anchor_metrics", []
        ),
    }
    report["milestones"] = _milestones(report)
    report["probe_trajectory"] = [
        {
            "after_probe": entry.counterfactual_probe_count,
            "stage": entry.stage,
            "retained_source_ids": list(entry.retained_ids),
            "retained_unit_count": entry.retained_unit_count,
            "cost": entry.cost,
        }
        for entry in result.trajectory
    ]
    report["trial_ledger"] = [
        {
            "ordinal": trial.ordinal,
            "probe_count": max(0, trial.ordinal - 1),
            "stage": trial.stage,
            "retained_source_ids": list(trial.retained_ids),
            "cost": trial.cost,
            "preserves": trial.preserves,
            "accepted": _accepted_trial(result, trial),
            "rejected": not trial.preserves,
        }
        for trial in result.trials
    ]
    return report


def suite_summary(case_reports: Sequence[Mapping[str, Any]], *, max_counterfactual_probes: int) -> dict[str, Any]:
    def checkpoint_stats(target: int, field_name: str) -> dict[str, Any]:
        reductions: list[float] = []
        for report in case_reports:
            for checkpoint in report.get(field_name, []) if isinstance(report.get(field_name), list) else []:
                if checkpoint.get("probe_count") != target or checkpoint.get("status") != "ok":
                    continue
                value = checkpoint.get("reduction_percent")
                if isinstance(value, (int, float)):
                    reductions.append(float(value))
        if not reductions:
            return {
                "eligible_case_count": 0,
                "median_reduction_percent": None,
                "min_reduction_percent": None,
                "max_reduction_percent": None,
                "total_counterfactual_probes": 0,
            }
        ordered = sorted(reductions)
        return {
            "eligible_case_count": len(ordered),
            "median_reduction_percent": round(statistics.median(ordered), 6),
            "min_reduction_percent": round(min(ordered), 6),
            "max_reduction_percent": round(max(ordered), 6),
            "total_counterfactual_probes": sum(
                min(target, int((report.get("budget") or {}).get("used_counterfactual_probes") or 0))
                for report in case_reports if report.get("status") == "ok"
            ),
        }

    actual_by_case = {
        report.get("case_id"): int((report.get("budget") or {}).get("used_counterfactual_probes") or 0)
        for report in case_reports
    }
    termination_reasons = {
        str(report.get("case_id")): (report.get("termination") or {}).get("reason")
        for report in case_reports
    }
    summary: dict[str, Any] = {
        "case_count": len(case_reports),
        "eligible_case_count": sum(1 for report in case_reports if report.get("status") == "ok"),
        "exact_unavailable_cases": [
            report.get("case_id") for report in case_reports
            if report.get("status") == "exact_unavailable"
        ],
        "control_failed_cases": [
            report.get("case_id") for report in case_reports
            if report.get("status") == "control_failed"
        ],
        "total_counterfactual_probes": sum(actual_by_case.values()),
        "certificate_counts_final": {
            BEST_VERIFIED: sum(1 for report in case_reports if (report.get("final") or {}).get("certificate_level") == BEST_VERIFIED),
            INCLUSION_MINIMUM: sum(1 for report in case_reports if (report.get("final") or {}).get("certificate_level") == INCLUSION_MINIMUM),
            "NONE": sum(1 for report in case_reports if not (report.get("final") or {}).get("certificate_level")),
        },
        "termination_reason_counts": {
            reason: sum(1 for value in termination_reasons.values() if value == reason)
            for reason in sorted({value for value in termination_reasons.values() if value})
        },
        "checkpoints": {},
        "geometric_checkpoints": {},
        "case_probe_counts": actual_by_case,
        "case_termination_reasons": termination_reasons,
        "descriptive_questions": {
            "terminated_before_probe_16": {
                "count": sum(1 for report in case_reports if report.get("status") == "ok" and actual_by_case.get(report.get("case_id"), 0) < 16),
                "case_ids": [report.get("case_id") for report in case_reports if report.get("status") == "ok" and actual_by_case.get(report.get("case_id"), 0) < 16],
            },
            "terminated_before_probe_32": {
                "count": sum(1 for report in case_reports if report.get("status") == "ok" and actual_by_case.get(report.get("case_id"), 0) < 32),
                "case_ids": [report.get("case_id") for report in case_reports if report.get("status") == "ok" and actual_by_case.get(report.get("case_id"), 0) < 32],
            },
            "hit_probe_200": {
                "count": sum(1 for report in case_reports if report.get("status") == "ok" and actual_by_case.get(report.get("case_id"), 0) >= 200),
                "case_ids": [report.get("case_id") for report in case_reports if report.get("status") == "ok" and actual_by_case.get(report.get("case_id"), 0) >= 200],
            },
            "probe_to_50_percent_eventual_reduction": {
                report.get("case_id"): (report.get("milestones") or {}).get("probe_to_50_percent_eventual_reduction")
                for report in case_reports
            },
            "probe_to_90_percent_eventual_reduction": {
                report.get("case_id"): (report.get("milestones") or {}).get("probe_to_90_percent_eventual_reduction")
                for report in case_reports
            },
            "reduction_after_probe_32": {
                report.get("case_id"): next((point.get("reduction_percent") for point in report.get("geometric_checkpoints", []) if point.get("probe_count") == 32), None)
                for report in case_reports
            },
            "token_probe_relation": {
                report.get("case_id"): {
                    "original_prompt_tokens": (report.get("original") or {}).get("rendered_prompt_tokens"),
                    "counterfactual_probes": actual_by_case.get(report.get("case_id"), 0),
                }
                for report in case_reports
            },
            "token_wall_relation": {
                report.get("case_id"): {
                    "original_prompt_tokens": (report.get("original") or {}).get("rendered_prompt_tokens"),
                    "wall_seconds": (report.get("timing") or {}).get("wall_seconds"),
                    "reduction_wall_seconds": (report.get("timing") or {}).get("reduction_wall_seconds"),
                }
                for report in case_reports
            },
            "bounded_unit_compression": {
                report.get("case_id"): {
                    "raw_context_unit_count": report.get("raw_context_unit_count"),
                    "bounded_search_universe_count": report.get("bounded_search_universe_count"),
                    "ratio": (report.get("source_geometry") or {}).get("bounded_compression_ratio"),
                }
                for report in case_reports
            },
            "controls": {
                "passed_case_count": sum(1 for report in case_reports if (report.get("control") or {}).get("passed") is True),
                "failed_case_count": sum(1 for report in case_reports if (report.get("control") or {}).get("passed") is False),
                "exact_unavailable_case_count": sum(1 for report in case_reports if report.get("status") == "exact_unavailable"),
            },
        },
    }
    for target in (value for value in CHECKPOINT_TARGETS if value <= max_counterfactual_probes):
        summary["checkpoints"][str(target)] = checkpoint_stats(target, "checkpoints")
    for target in (value for value in GEOMETRIC_CHECKPOINT_TARGETS if value <= max_counterfactual_probes):
        summary["geometric_checkpoints"][str(target)] = checkpoint_stats(target, "geometric_checkpoints")
    return summary


__all__ = [
    "CHECKPOINT_TARGETS",
    "GEOMETRIC_CHECKPOINT_TARGETS",
    "EvaluationOutcome",
    "RecordedRunBinding",
    "SCHEMA",
    "bind_engine_recorded_run",
    "evaluate_recorded_run",
    "reconstruct_checkpoint",
    "reconstruct_checkpoints",
    "serialize_outcome",
    "suite_summary",
]
