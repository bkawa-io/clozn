"""Adaptive exact Minimal Context recipe over the generic experiment kernel."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from typing import Any

from clozn.experiments.context_search import ContextSearchDispatcher, ContextSearchUnavailable
from clozn.experiments.evaluators import ExactReferenceMatch
from clozn.experiments.search import (
    BEST_VERIFIED,
    EXACT_MINIMUM,
    INCLUSION_MINIMUM,
    canonical_search_policy,
    certify_exact_minimum,
    SearchBudget,
    SearchEvidenceRef,
    SearchTrial,
    SearchTrajectoryEntry,
    run_adaptive_search,
)
from clozn.experiments.state import ExecutionState, canonical_json
from clozn.replay.span_bridge import resolve_context_receipt_source_set
from clozn.runs.context_search_universe import plan_context_search_universe
from clozn.experiments.persistence import ObservationStore


SCHEMA_VERSION = "clozn.minimal-context-search-result.v2"
OBJECTIVE_VERSION = "rendered_prompt_tokens.v1"
STOPPING_REASONS = frozenset({
    "exact_minimum_proven", "inclusion_minimum_proven", "budget_exhausted",
    "search_policy_complete", "control_unavailable", "search_unavailable", "cancelled",
})


class MinimalContextUnavailable(ValueError):
    """The exact direct-observation search cannot be performed faithfully."""

    def __init__(self, message: str, *, reason: str = "minimal_context_unavailable"):
        super().__init__(message)
        self.reason = reason


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WinningCandidate:
    retained_source_ids: tuple[str, ...]
    removed_source_ids: tuple[str, ...]
    rendered_prompt_token_cost: int
    experiment_id: str | None
    arm_id: str | None
    observation_id: str | None
    observation_status: str | None = None

    @property
    def cost(self) -> int:
        return self.rendered_prompt_token_cost

    @property
    def retained_ids(self) -> tuple[str, ...]:
        return self.retained_source_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "retained_source_ids": list(self.retained_source_ids),
            "removed_source_ids": list(self.removed_source_ids),
            "rendered_prompt_token_cost": self.rendered_prompt_token_cost,
            "experiment_id": self.experiment_id,
            "arm_id": self.arm_id,
            "observation_id": self.observation_id,
            "observation_status": self.observation_status,
        }


@dataclass(frozen=True)
class MinimalContextResult:
    """Durable derived search result; ObservationStore remains evidence authority.

    The result contains references to direct observations, not their bodies.
    ``result_id`` is derived from the immutable result content and therefore
    remains stable across process restarts.  Timing and creation metadata are
    intentionally absent from that identity.
    """

    search_id: str
    status: str
    base_execution_fingerprint: str
    universe: Mapping[str, Any]
    objective: Mapping[str, Any]
    control_observation_id: str | None
    trials: tuple[SearchTrial, ...]
    trajectory: tuple[SearchTrajectoryEntry, ...]
    best: WinningCandidate | None
    certificate: str | None
    policy: Mapping[str, Any]
    budget: SearchBudget
    inclusion_check: Mapping[str, Any]
    reason: str | None = None
    reason_code: str | None = None
    search_status: str | None = None
    run_id: str | None = None
    result_id: str | None = None
    original: Mapping[str, Any] = field(default_factory=dict)
    reduction: Mapping[str, Any] = field(default_factory=dict)
    stopping_reason: str = "search_policy_complete"
    experiment_accounting: Mapping[str, Any] = field(default_factory=dict)
    source_inspection: tuple[Mapping[str, Any], ...] = ()
    proof: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        run_id = self.run_id or self.universe.get("run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("MinimalContextResult requires a non-empty run_id")
        object.__setattr__(self, "run_id", run_id)
        if self.stopping_reason not in STOPPING_REASONS:
            raise ValueError(f"unsupported Minimal Context stopping reason: {self.stopping_reason!r}")
        computed = "mcres_" + _digest(self._identity_document())[:24]
        if self.result_id is not None and self.result_id != computed:
            raise ValueError("MinimalContextResult result_id does not match its immutable content")
        object.__setattr__(self, "result_id", computed)

    def _identity_document(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "search_id": self.search_id,
            "run_id": self.run_id,
            "base_execution_fingerprint": self.base_execution_fingerprint,
            "universe": deepcopy(dict(self.universe)),
            "objective": deepcopy(dict(self.objective)),
            "policy": deepcopy(dict(self.policy)),
            "budget": self.budget.to_dict(),
            "trials": [trial.to_dict() for trial in self.trials],
            "trajectory": [item.to_dict() for item in self.trajectory],
            "control_observation_id": self.control_observation_id,
            "original": deepcopy(dict(self.original)),
            "best": self.best.to_dict() if self.best else None,
            "certificate": self.certificate,
            "stopping_reason": self.stopping_reason,
            "inclusion_check": deepcopy(dict(self.inclusion_check)),
            "experiment_accounting": deepcopy(dict(self.experiment_accounting)),
            "source_inspection": [deepcopy(dict(item)) for item in self.source_inspection],
            "proof": deepcopy(dict(self.proof)),
            "status": self.status,
            "search_status": self.search_status,
            "reason": self.reason,
            "reason_code": self.reason_code,
            "reduction": deepcopy(dict(self.reduction)),
        }

    @property
    def best_candidate(self) -> WinningCandidate | None:
        return self.best

    @property
    def schema_version(self) -> str:
        return SCHEMA_VERSION

    @property
    def certificate_level(self) -> str | None:
        return self.certificate

    @property
    def winning_experiment_id(self) -> str | None:
        return self.best.experiment_id if self.best else None

    @property
    def winning_arm_id(self) -> str | None:
        return self.best.arm_id if self.best else None

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": SCHEMA_VERSION,
            "result_id": self.result_id,
            "run_id": self.run_id,
            "search_id": self.search_id,
            "status": self.status,
            "search_status": self.search_status,
            "reason": self.reason,
            "reason_code": self.reason_code,
            "base_execution_fingerprint": self.base_execution_fingerprint,
            "universe": deepcopy(dict(self.universe)),
            "objective": deepcopy(dict(self.objective)),
            "control_observation_id": self.control_observation_id,
            "trials": [trial.to_dict() for trial in self.trials],
            "trajectory": [item.to_dict() for item in self.trajectory],
            "original": deepcopy(dict(self.original)),
            "best": self.best.to_dict() if self.best else None,
            "reduction": deepcopy(dict(self.reduction)),
            "certificate": self.certificate,
            "stopping_reason": self.stopping_reason,
            "experiment_accounting": deepcopy(dict(self.experiment_accounting)),
            "source_inspection": [deepcopy(dict(item)) for item in self.source_inspection],
            "proof": deepcopy(dict(self.proof)),
            "policy": deepcopy(dict(self.policy)),
            "budget": self.budget.to_dict(),
            "inclusion_check": dict(self.inclusion_check),
        }
        return value

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MinimalContextResult":
        if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("MinimalContextResult schema version is invalid")
        if not isinstance(value.get("result_id"), str) or not value["result_id"]:
            raise ValueError("MinimalContextResult result_id is required")
        if not isinstance(value.get("run_id"), str) or not value["run_id"]:
            raise ValueError("MinimalContextResult run_id is required")

        def candidate(raw: Any) -> WinningCandidate | None:
            if raw is None:
                return None
            if not isinstance(raw, Mapping):
                raise ValueError("MinimalContextResult candidate is malformed")
            cost = raw.get("rendered_prompt_token_cost")
            if isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
                raise ValueError("MinimalContextResult candidate cost is malformed")
            return WinningCandidate(
                tuple(raw.get("retained_source_ids") or []), tuple(raw.get("removed_source_ids") or []), cost,
                raw.get("experiment_id") if isinstance(raw.get("experiment_id"), str) else None,
                raw.get("arm_id") if isinstance(raw.get("arm_id"), str) else None,
                raw.get("observation_id") if isinstance(raw.get("observation_id"), str) else None,
                raw.get("observation_status") if isinstance(raw.get("observation_status"), str) else None,
            )

        def ref(raw: Any) -> SearchEvidenceRef | None:
            return SearchEvidenceRef.from_value(raw)

        raw_trials = value.get("trials")
        if not isinstance(raw_trials, list):
            raise ValueError("MinimalContextResult trials are malformed")
        trials: list[SearchTrial] = []
        for raw in raw_trials:
            if not isinstance(raw, Mapping) or raw.get("classification") not in {"preserves", "diverged", "unknown"}:
                raise ValueError("MinimalContextResult trial is malformed")
            trials.append(SearchTrial(
                ordinal=int(raw.get("ordinal")), stage=str(raw.get("stage")),
                retained_ids=tuple(raw.get("retained_ids") or []), cost=int(raw.get("cost")),
                classification=str(raw["classification"]), evidence_ref=ref(raw.get("evidence") or raw),
                disposition=str(raw.get("disposition", "executed")), batch_id=raw.get("batch_id"),
                parent_retained_ids=tuple(raw.get("parent_retained_ids") or []),
            ))
        budget_raw = value.get("budget")
        inclusion_raw = value.get("inclusion_check")
        if not isinstance(budget_raw, Mapping) or not isinstance(inclusion_raw, Mapping):
            raise ValueError("MinimalContextResult budget or inclusion check is malformed")
        budget = SearchBudget(
            int(budget_raw.get("max_new_executions")), int(budget_raw.get("used_new_executions")),
            int(budget_raw.get("reused_observation_count", 0)), bool(budget_raw.get("exhausted")),
            bool(budget_raw.get("blocked_by_budget", False)),
        )
        inclusion = {
            "attempted": bool(inclusion_raw.get("attempted")),
            "complete": bool(inclusion_raw.get("complete")),
            "tested_child_count": int(inclusion_raw.get("tested_child_count")),
            "total_child_count": int(inclusion_raw.get("total_child_count")),
            "all_children_failed": bool(inclusion_raw.get("all_children_failed")),
        }
        trajectory = tuple(SearchTrajectoryEntry(
            int(item.get("counterfactual_probe_count")), tuple(item.get("retained_ids") or []),
            int(item.get("cost")), int(item.get("retained_unit_count")), str(item.get("stage")),
        ) for item in value.get("trajectory") or [] if isinstance(item, Mapping))
        universe = value.get("universe")
        objective = value.get("objective")
        policy = value.get("policy")
        if not all(isinstance(item, Mapping) for item in (universe, objective, policy)):
            raise ValueError("MinimalContextResult bindings are malformed")
        return cls(
            result_id=value.get("result_id") if isinstance(value.get("result_id"), str) else None,
            run_id=value.get("run_id") if isinstance(value.get("run_id"), str) else None,
            search_id=str(value.get("search_id")), status=str(value.get("status")),
            search_status=value.get("search_status") if isinstance(value.get("search_status"), str) else None,
            reason=value.get("reason") if isinstance(value.get("reason"), str) else None,
            reason_code=value.get("reason_code") if isinstance(value.get("reason_code"), str) else None,
            base_execution_fingerprint=str(value.get("base_execution_fingerprint")),
            universe=universe, objective=objective,
            control_observation_id=value.get("control_observation_id") if isinstance(value.get("control_observation_id"), str) else None,
            trials=tuple(trials), trajectory=trajectory, best=candidate(value.get("best")),
            certificate=value.get("certificate") if isinstance(value.get("certificate"), str) else None,
            policy=policy, budget=budget, inclusion_check=inclusion,
            original=value.get("original") if isinstance(value.get("original"), Mapping) else {},
            reduction=value.get("reduction") if isinstance(value.get("reduction"), Mapping) else {},
            stopping_reason=str(value.get("stopping_reason")),
            experiment_accounting=value.get("experiment_accounting") if isinstance(value.get("experiment_accounting"), Mapping) else {},
            source_inspection=tuple(item for item in value.get("source_inspection") or [] if isinstance(item, Mapping)),
            proof=value.get("proof") if isinstance(value.get("proof"), Mapping) else {},
        )


def _empty_budget(max_new: int) -> SearchBudget:
    return SearchBudget(max_new_executions=max_new, used_new_executions=0, exhausted=max_new == 0)


def _source_inspection(run: Mapping[str, Any], universe: Mapping[str, Any], retained: Sequence[str]) -> tuple[dict[str, Any], ...]:
    """Materialize inspectable source metadata from the canonical receipt.

    This is a result projection only.  Source resolution remains owned by the
    receipt/span bridge, and the result stores the resolved metadata needed to
    explain the tested winner without copying the whole Run.
    """
    try:
        resolved = resolve_context_receipt_source_set(dict(run), list(universe.get("source_ids") or []))
    except Exception:
        return ()
    catalog = resolved.get("sources")
    basis = resolved.get("basis_messages")
    if not isinstance(catalog, list) or not isinstance(basis, list):
        return ()
    by_id = {item.get("source_id"): item for item in catalog if isinstance(item, Mapping)}
    roots = {
        item.get("message_index"): item for item in catalog
        if isinstance(item, Mapping) and item.get("source_kind") == "whole_message"
    }
    retained_set = set(retained)
    records: list[dict[str, Any]] = []
    for source_id in universe.get("source_ids") or []:
        source = by_id.get(source_id)
        if not isinstance(source, Mapping):
            continue
        message_index = source.get("message_index")
        content = basis[message_index].get("content") if (
            isinstance(message_index, int) and 0 <= message_index < len(basis)
            and isinstance(basis[message_index], Mapping)
        ) else None
        raw_range = source.get("unicode_range")
        if isinstance(raw_range, (list, tuple)) and len(raw_range) == 2:
            unicode_range = [int(raw_range[0]), int(raw_range[1])]
        elif isinstance(content, str):
            unicode_range = [0, len(content)]
        else:
            unicode_range = None
        raw_bytes = source.get("byte_range")
        if isinstance(raw_bytes, (list, tuple)) and len(raw_bytes) == 2:
            byte_range = [int(raw_bytes[0]), int(raw_bytes[1])]
        elif isinstance(content, str) and unicode_range is not None:
            byte_range = [
                len(content[:unicode_range[0]].encode("utf-8")),
                len(content[:unicode_range[1]].encode("utf-8")),
            ]
        else:
            byte_range = None
        text = None
        if isinstance(content, str) and unicode_range is not None:
            text = content[unicode_range[0]:unicode_range[1]]
        segment_id = source.get("segment_id")
        if not isinstance(segment_id, str):
            root = roots.get(message_index)
            segment_id = root.get("segment_id") if isinstance(root, Mapping) else None
        records.append({
            "source_id": source_id,
            "segment_id": segment_id,
            "message_index": message_index,
            "label": source.get("source_label"),
            "provenance_kind": source.get("provenance_kind"),
            "parent_source_id": source.get("parent_source_id"),
            "unicode_range": unicode_range,
            "byte_range": byte_range,
            "granularity": "exact_span" if source.get("source_kind") == "source_span" else "whole_segment",
            "text": text,
            "disposition": "retained" if source_id in retained_set else "removed",
        })
    return tuple(records)


def _inclusion_proof(best: WinningCandidate, trials: Sequence[SearchTrial], inclusion: Mapping[str, Any]) -> dict[str, Any]:
    children: list[dict[str, Any]] = []
    by_retained = {tuple(trial.retained_ids): trial for trial in trials}
    for source_id in best.retained_source_ids:
        retained = tuple(item for item in best.retained_source_ids if item != source_id)
        trial = by_retained.get(retained)
        children.append({
            "removed_source_id": source_id,
            "retained_source_ids": list(retained),
            "classification": trial.classification if trial else "unknown",
            "evidence": trial.evidence_ref.to_dict() if trial and trial.evidence_ref else None,
            "experiment_id": trial.experiment_id if trial else None,
            "arm_id": trial.arm_id if trial else None,
            "observation_id": trial.observation_id if trial else None,
            "observation_status": trial.observation_status if trial else None,
            "disposition": trial.disposition if trial else "not_executed",
        })
    return {
        "attempted": bool(inclusion.get("attempted")),
        "complete": bool(inclusion.get("complete")),
        "children": children,
    }


def _accounting(searched: Any) -> dict[str, Any]:
    candidate_trials = tuple(trial for trial in searched.trials if trial.stage != "control")
    direct = tuple(trial for trial in candidate_trials if trial.classification in {"preserves", "diverged"}
                   and trial.evidence_ref is not None and trial.evidence_ref.observation_id)
    return {
        "control": {
            "disposition": searched.control_evidence.disposition if searched.control_evidence else "not_executed",
            "observation_id": searched.control_evidence.observation_id if searched.control_evidence else None,
        },
        "candidate_trials": len(candidate_trials),
        "new_counterfactual_executions": searched.budget.used_new_executions,
        "reused_observations": searched.budget.reused_observation_count,
        "direct_candidates_observed": len(direct),
        "preserved_count": sum(trial.classification == "preserves" for trial in candidate_trials),
        "diverged_count": sum(trial.classification == "diverged" for trial in candidate_trials),
        "unknown_count": sum(trial.classification == "unknown" for trial in candidate_trials),
    }


def _reduction(original_cost: int, best_cost: int, universe_count: int, retained_count: int) -> dict[str, Any]:
    saved = original_cost - best_cost
    fraction = (saved / original_cost) if original_cost else 0.0
    return {
        "objective": OBJECTIVE_VERSION,
        "original_prompt_token_cost": original_cost,
        "retained_prompt_token_cost": best_cost,
        "removed_source_count": universe_count - retained_count,
        "retained_source_count": retained_count,
        "fraction": fraction,
        "percent": fraction * 100.0,
    }


def _stopping_reason(searched: Any, *, certificate: str | None, inclusion: Mapping[str, Any]) -> str:
    if searched.status == "cancelled":
        return "cancelled"
    if searched.status == "control_failed":
        return "control_unavailable"
    if searched.status not in {"ok", "completed"}:
        return "search_unavailable"
    if certificate == EXACT_MINIMUM:
        return "exact_minimum_proven"
    if certificate == INCLUSION_MINIMUM and inclusion.get("complete"):
        return "inclusion_minimum_proven"
    if searched.budget.blocked_by_budget:
        return "budget_exhausted"
    return "search_policy_complete"


def _identity(*, state: ExecutionState, universe: Mapping[str, Any], max_new: int,
              policy: Mapping[str, Any]) -> str:
    value = {
        "base_execution_fingerprint": state.execution_fingerprint,
        "universe_id": universe.get("universe_id"),
        "ordered_source_ids": list(universe.get("source_ids") or []),
        "policy": deepcopy(dict(policy)),
        "objective_version": OBJECTIVE_VERSION,
        "max_new_counterfactual_observations": max_new,
    }
    return "search_" + _digest(value)[:24]


def _unavailable(*, state: ExecutionState, universe: Mapping[str, Any], max_new: int,
                 search_id: str, reason: str, policy: Mapping[str, Any],
                 reason_code: str = "minimal_context_unavailable",
                 status: str = "unavailable", run: Mapping[str, Any] | None = None) -> MinimalContextResult:
    objective = {"kind": "rendered_prompt_tokens", "version": OBJECTIVE_VERSION}
    return MinimalContextResult(
        run_id=state.run_id,
        search_id=search_id,
        status=status,
        search_status="unavailable",
        base_execution_fingerprint=state.execution_fingerprint,
        universe=universe,
        objective=objective,
        control_observation_id=None,
        trials=(),
        trajectory=(),
        best=None,
        certificate=None,
        policy=policy,
        budget=_empty_budget(max_new),
        inclusion_check={"attempted": False, "complete": False, "tested_child_count": 0,
                         "total_child_count": 0, "all_children_failed": False},
        reason=reason,
        reason_code=reason_code,
        stopping_reason="cancelled" if status == "cancelled" else (
            "control_unavailable" if reason_code == "exact_control_unavailable" else "search_unavailable"
        ),
        source_inspection=_source_inspection(run or {}, universe, universe.get("source_ids") or []),
    )


def run_minimal_context(
    run: Mapping[str, Any], *,
    evaluator: ExactReferenceMatch | None = None,
    max_new_counterfactual_observations: int = 32,
    observation_store: ObservationStore | None = None,
    store: ObservationStore | None = None,
    substrate: Any = None,
    engine: Any = None,
    execution_adapter: Any = None,
    prompt_token_counter: Callable[[Sequence[Mapping[str, Any]]], int] | None = None,
    render_messages: Callable[[tuple[str, ...]], Sequence[Mapping[str, Any]]] | None = None,
    max_units: int = 50,
    attempt_inclusion_check: bool = True,
    cancel: Callable[[], bool] | None = None,
    execution_strategy: str = "auto",
) -> MinimalContextResult:
    """Search directly observed exact preservation over a canonical universe."""
    if observation_store is not None and store is not None and observation_store is not store:
        raise ValueError("pass only one observation store")
    durable = observation_store or store
    if durable is not None and not isinstance(durable, ObservationStore):
        raise TypeError("observation_store must be an ObservationStore")
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
        raise MinimalContextUnavailable("a recorded run with a non-empty id is required", reason="run_unavailable")
    if evaluator is not None and not isinstance(evaluator, ExactReferenceMatch):
        raise MinimalContextUnavailable(
            "Batch 5 Minimal Context supports exact recorded-answer preservation only",
            reason="unsupported_evaluator",
        )
    if (isinstance(max_new_counterfactual_observations, bool)
            or not isinstance(max_new_counterfactual_observations, int)
            or max_new_counterfactual_observations < 0):
        raise ValueError("max_new_counterfactual_observations must be a non-negative integer")
    try:
        policy = canonical_search_policy(attempt_inclusion_check=attempt_inclusion_check)
    except ValueError as exc:
        raise MinimalContextUnavailable(str(exc), reason="invalid_search_policy") from exc
    try:
        state = ExecutionState.from_run(run)
    except Exception as exc:
        raise MinimalContextUnavailable(
            f"recorded execution state unavailable: {exc}",
            reason="execution_state_unavailable",
        ) from exc
    manifest = run.get("context_units")
    try:
        universe = plan_context_search_universe(run, manifest, max_units=max_units)
    except Exception as exc:  # normalize planner failures at this recipe boundary
        raise MinimalContextUnavailable(f"context search universe unavailable: {exc}", reason="universe_unavailable") from exc
    search_id = _identity(
        state=state, universe=universe, max_new=max_new_counterfactual_observations,
        policy=policy,
    )
    if universe.get("status") != "planned":
        return _unavailable(
            state=state, universe=universe, max_new=max_new_counterfactual_observations,
            search_id=search_id, policy=policy,
            reason=(universe.get("condition") or {}).get("message", "context search universe unavailable"),
            reason_code=(universe.get("condition") or {}).get("code", "universe_unavailable"),
            run=run,
        )
    objective = {"kind": "rendered_prompt_tokens", "version": OBJECTIVE_VERSION}
    dispatcher: ContextSearchDispatcher | None = None
    try:
        dispatcher = ContextSearchDispatcher(
            run, universe, substrate=substrate, engine=engine,
            observation_store=durable, execution_adapter=execution_adapter,
            evaluator=ExactReferenceMatch(), prompt_token_counter=prompt_token_counter,
            render_messages=render_messages, cancel=cancel, execution_strategy=execution_strategy,
        )
        searched = run_adaptive_search(
            tuple(universe["source_ids"]),
            max_new_counterfactual_observations,
            dispatcher.prepare_candidate,
            dispatcher.probe_many,
            candidate_is_reusable=dispatcher.candidate_is_reusable,
            attempt_inclusion_check=attempt_inclusion_check,
            search_id=search_id,
            base_execution_fingerprint=state.execution_fingerprint,
            universe_id=universe.get("universe_id"),
            policy=policy,
            objective=objective,
        )
    except (ContextSearchUnavailable, ValueError, TypeError) as exc:
        return _unavailable(
            state=state, universe=universe, max_new=max_new_counterfactual_observations,
            search_id=search_id, policy=policy, reason=str(exc),
            reason_code=getattr(exc, "reason", "search_execution_unavailable"),
            run=run,
        )
    finally:
        if dispatcher is not None:
            try:
                dispatcher.close()
            except Exception:
                # Evidence already established remains valid; lifecycle cleanup
                # is operational metadata and cannot rewrite the result.
                pass

    control_ref = searched.control_evidence
    control_id = control_ref.observation_id if control_ref else None
    if searched.status == "control_failed":
        cancelled = control_ref is not None and control_ref.observation_status in {"cancelled", "cancelled_before_dispatch"}
        return MinimalContextResult(
            run_id=run["id"],
            search_id=search_id, status="cancelled" if cancelled else "unavailable", search_status=searched.status,
            base_execution_fingerprint=state.execution_fingerprint, universe=universe,
            objective=objective, control_observation_id=control_id,
            trials=searched.trials, trajectory=searched.trajectory, best=None,
            certificate=None, policy=policy, budget=searched.budget,
            inclusion_check=searched.inclusion_check.to_dict(),
            reason=(control_ref.observation_status if control_ref else "control_unavailable"),
            reason_code="cancelled" if cancelled else "exact_control_unavailable",
            stopping_reason="cancelled" if cancelled else "control_unavailable",
            experiment_accounting=_accounting(searched),
            source_inspection=_source_inspection(run, universe, universe.get("source_ids") or []),
        )

    best_trial = None
    for trial in searched.trials:
        if trial.retained_ids == searched.best_candidate.retained_ids and trial.preserves:
            best_trial = trial
            break
    if best_trial is None:
        best_ref = control_ref if searched.best_candidate.retained_ids == tuple(universe["source_ids"]) else None
    else:
        best_ref = best_trial.evidence_ref
    retained = tuple(str(item) for item in searched.best_candidate.retained_ids)
    removed = tuple(source_id for source_id in universe["source_ids"] if source_id not in set(retained))
    best = WinningCandidate(
        retained_source_ids=retained,
        removed_source_ids=removed,
        rendered_prompt_token_cost=searched.best_candidate.cost,
        experiment_id=best_ref.experiment_id if best_ref else None,
        arm_id=best_ref.arm_id if best_ref else None,
        observation_id=best_ref.observation_id if best_ref else None,
        observation_status=best_ref.observation_status if best_ref else None,
    )
    original_ref = control_ref
    original = WinningCandidate(
        retained_source_ids=tuple(str(item) for item in universe["source_ids"]),
        removed_source_ids=(),
        rendered_prompt_token_cost=searched.original_candidate.cost,
        experiment_id=original_ref.experiment_id if original_ref else None,
        arm_id=original_ref.arm_id if original_ref else None,
        observation_id=original_ref.observation_id if original_ref else None,
        observation_status=original_ref.observation_status if original_ref else None,
    )
    exact_proof = certify_exact_minimum(
        tuple(universe["source_ids"]), searched.trials,
        control_evidence=control_ref, original_candidate=searched.original_candidate,
        winner=searched.best_candidate,
    )
    certificate = EXACT_MINIMUM if exact_proof is not None else searched.certificate
    inclusion = searched.inclusion_check.to_dict()
    proof = {
        "certificate": certificate,
        "trajectory": [item.to_dict() for item in searched.trajectory],
        "inclusion": _inclusion_proof(best, searched.trials, inclusion),
        "exact_minimum": exact_proof,
    }
    cancelled = any(
        trial.observation_status in {"cancelled", "cancelled_before_dispatch"}
        for trial in searched.trials
    )
    result_status = "cancelled" if cancelled else "completed"
    stopping = "cancelled" if cancelled else _stopping_reason(searched, certificate=certificate, inclusion=inclusion)
    return MinimalContextResult(
        run_id=run["id"],
        search_id=search_id, status=result_status, search_status=searched.status,
        base_execution_fingerprint=state.execution_fingerprint, universe=universe,
        objective=objective, control_observation_id=control_id,
        trials=searched.trials, trajectory=searched.trajectory, best=best,
        certificate=certificate, policy=policy, budget=searched.budget,
        inclusion_check=inclusion, original=original.to_dict(),
        reduction=_reduction(
            original.cost, best.cost, len(universe["source_ids"]), len(best.retained_source_ids),
        ),
        stopping_reason=stopping,
        experiment_accounting=_accounting(searched),
        source_inspection=_source_inspection(run, universe, best.retained_source_ids),
        proof=proof,
    )


__all__ = [
    "BEST_VERIFIED", "EXACT_MINIMUM", "INCLUSION_MINIMUM", "OBJECTIVE_VERSION", "SCHEMA_VERSION",
    "STOPPING_REASONS", "MinimalContextError", "MinimalContextUnavailable",
    "MinimalContextResult", "WinningCandidate", "run_minimal_context",
]


MinimalContextError = MinimalContextUnavailable
