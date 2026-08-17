"""Model-free adaptive search contracts for the experimental kernel.

The search layer sees only an ordered universe, opaque candidate costs, and
direct-evidence references.  It deliberately does not know what a source,
prompt, model, or ObservationStore is.  The bounded reducer implementation is
shared with the historical reducer while this module owns the durable,
reference-only result shape used by new recipes.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any, TypeAlias

from clozn.runs.budgeted_reduce import (
    BEST_VERIFIED,
    CONTROL_FAILED,
    INCLUSION_MINIMUM,
    OK,
    BudgetedReductionResult,
    Candidate,
    PreparedCandidate,
    run_budgeted_reduction,
)


UnitID: TypeAlias = str | int
SEARCH_POLICY_VERSION = "adaptive_bounded_deletion.v1"
SEARCH_UNAVAILABLE = "search_unavailable"


@dataclass(frozen=True)
class SearchEvidenceRef:
    """A durable pointer to direct evidence, never the evidence body."""

    experiment_id: str | None = None
    arm_id: str | None = None
    observation_id: str | None = None
    observation_status: str | None = None
    disposition: str = "executed"

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "disposition": self.disposition,
        }
        for key in ("experiment_id", "arm_id", "observation_id", "observation_status"):
            item = getattr(self, key)
            if item is not None:
                value[key] = item
        return value

    @classmethod
    def from_value(cls, value: Any) -> "SearchEvidenceRef | None":
        if isinstance(value, SearchEvidenceRef):
            return value
        if not isinstance(value, Mapping):
            return None
        return cls(
            experiment_id=value.get("experiment_id") if isinstance(value.get("experiment_id"), str) else None,
            arm_id=value.get("arm_id") if isinstance(value.get("arm_id"), str) else None,
            observation_id=value.get("observation_id") if isinstance(value.get("observation_id"), str) else None,
            observation_status=(value.get("observation_status")
                                if isinstance(value.get("observation_status"), str) else value.get("status")
                                if isinstance(value.get("status"), str) else None),
            disposition=value.get("disposition") if value.get("disposition") in {
                "reused", "executed", "not_executed"
            } else "executed",
        )


@dataclass(frozen=True)
class SearchBudget:
    """Budget measured in new counterfactual model executions."""

    max_new_executions: int
    used_new_executions: int
    reused_observation_count: int = 0
    exhausted: bool = False

    @property
    def max_counterfactual_probes(self) -> int:
        return self.max_new_executions

    @property
    def used_counterfactual_probes(self) -> int:
        return self.used_new_executions

    @property
    def total_direct_experiments(self) -> int:
        return 1 + self.used_new_executions

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_new_executions": self.max_new_executions,
            "used_new_executions": self.used_new_executions,
            "reused_observation_count": self.reused_observation_count,
            "exhausted": self.exhausted,
        }


@dataclass(frozen=True)
class SearchTrial:
    """One directly tested candidate with a reference-only evidence ledger."""

    ordinal: int
    stage: str
    retained_ids: tuple[UnitID, ...]
    cost: int
    classification: str
    evidence_ref: SearchEvidenceRef | None
    disposition: str = "executed"
    batch_id: int | None = None
    parent_retained_ids: tuple[UnitID, ...] = ()

    @property
    def preserves(self) -> bool:
        return self.classification == "preserves"

    @property
    def diverged(self) -> bool:
        return self.classification == "diverged"

    @property
    def experiment_id(self) -> str | None:
        return self.evidence_ref.experiment_id if self.evidence_ref else None

    @property
    def arm_id(self) -> str | None:
        return self.evidence_ref.arm_id if self.evidence_ref else None

    @property
    def observation_id(self) -> str | None:
        return self.evidence_ref.observation_id if self.evidence_ref else None

    @property
    def observation_status(self) -> str | None:
        return self.evidence_ref.observation_status if self.evidence_ref else None

    def to_dict(self) -> dict[str, Any]:
        evidence = self.evidence_ref.to_dict() if self.evidence_ref else None
        return {
            "ordinal": self.ordinal,
            "stage": self.stage,
            "retained_ids": list(self.retained_ids),
            "cost": self.cost,
            "classification": self.classification,
            "disposition": self.disposition,
            "experiment_id": self.experiment_id,
            "arm_id": self.arm_id,
            "observation_id": self.observation_id,
            "observation_status": self.observation_status,
            "evidence": evidence,
            "batch_id": self.batch_id,
            "parent_retained_ids": list(self.parent_retained_ids),
        }


@dataclass(frozen=True)
class SearchTrajectoryEntry:
    counterfactual_probe_count: int
    retained_ids: tuple[UnitID, ...]
    cost: int
    retained_unit_count: int
    stage: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "counterfactual_probe_count": self.counterfactual_probe_count,
            "retained_ids": list(self.retained_ids),
            "cost": self.cost,
            "retained_unit_count": self.retained_unit_count,
            "stage": self.stage,
        }


@dataclass(frozen=True)
class InclusionCheck:
    attempted: bool
    complete: bool
    tested_child_count: int
    total_child_count: int
    all_children_failed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "complete": self.complete,
            "tested_child_count": self.tested_child_count,
            "total_child_count": self.total_child_count,
            "all_children_failed": self.all_children_failed,
        }


@dataclass(frozen=True)
class SearchResult:
    """Reference-only, deterministic search output."""

    status: str
    certificate: str | None
    original_candidate: Candidate
    best_candidate: Candidate
    control_evidence: SearchEvidenceRef | None
    trials: tuple[SearchTrial, ...]
    trajectory: tuple[SearchTrajectoryEntry, ...]
    budget: SearchBudget
    inclusion_check: InclusionCheck
    search_id: str | None = None
    base_execution_fingerprint: str | None = None
    universe_id: str | None = None
    policy_version: str = SEARCH_POLICY_VERSION
    objective: Mapping[str, Any] | None = None

    @property
    def certificate_level(self) -> str | None:
        return self.certificate

    @property
    def control_observation_id(self) -> str | None:
        return self.control_evidence.observation_id if self.control_evidence else None

    @property
    def best(self) -> Candidate:
        return self.best_candidate

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "clozn.experiment-search-result.v1",
            "search_id": self.search_id,
            "status": self.status,
            "certificate": self.certificate,
            "base_execution_fingerprint": self.base_execution_fingerprint,
            "universe_id": self.universe_id,
            "policy_version": self.policy_version,
            "objective": deepcopy(dict(self.objective or {})),
            "original_candidate": {
                "retained_ids": list(self.original_candidate.retained_ids),
                "cost": self.original_candidate.cost,
            },
            "best_candidate": {
                "retained_ids": list(self.best_candidate.retained_ids),
                "cost": self.best_candidate.cost,
            },
            "control_evidence": self.control_evidence.to_dict() if self.control_evidence else None,
            "trials": [trial.to_dict() for trial in self.trials],
            "trajectory": [entry.to_dict() for entry in self.trajectory],
            "budget": self.budget.to_dict(),
            "inclusion_check": self.inclusion_check.to_dict(),
        }

    def to_json(self) -> str:
        from .state import canonical_json
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SearchResult":
        if not isinstance(value, Mapping) or value.get("schema_version") != "clozn.experiment-search-result.v1":
            raise ValueError("SearchResult schema version is invalid")
        def candidate(raw: Any) -> Candidate:
            if not isinstance(raw, Mapping) or not isinstance(raw.get("retained_ids"), list):
                raise ValueError("SearchResult candidate is malformed")
            cost = raw.get("cost")
            if isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
                raise ValueError("SearchResult candidate cost is malformed")
            return Candidate(tuple(raw["retained_ids"]), cost)
        def ref(raw: Any) -> SearchEvidenceRef | None:
            return SearchEvidenceRef.from_value(raw)
        trials: list[SearchTrial] = []
        for raw in value.get("trials") or []:
            if not isinstance(raw, Mapping):
                raise ValueError("SearchResult trial is malformed")
            evidence = ref(raw.get("evidence") or raw)
            classification = raw.get("classification")
            if classification not in {"preserves", "diverged", "unknown"}:
                raise ValueError("SearchResult trial classification is invalid")
            trials.append(SearchTrial(
                ordinal=int(raw.get("ordinal")), stage=str(raw.get("stage")),
                retained_ids=tuple(raw.get("retained_ids") or []), cost=int(raw.get("cost")),
                classification=classification, evidence_ref=evidence,
                disposition=raw.get("disposition", "executed"),
                batch_id=raw.get("batch_id"), parent_retained_ids=tuple(raw.get("parent_retained_ids") or []),
            ))
        budget_raw = value.get("budget")
        inclusion_raw = value.get("inclusion_check")
        if not isinstance(budget_raw, Mapping) or not isinstance(inclusion_raw, Mapping):
            raise ValueError("SearchResult budget or inclusion check is malformed")
        budget = SearchBudget(
            max_new_executions=int(budget_raw.get("max_new_executions")),
            used_new_executions=int(budget_raw.get("used_new_executions")),
            reused_observation_count=int(budget_raw.get("reused_observation_count", 0)),
            exhausted=bool(budget_raw.get("exhausted")),
        )
        inclusion = InclusionCheck(
            bool(inclusion_raw.get("attempted")), bool(inclusion_raw.get("complete")),
            int(inclusion_raw.get("tested_child_count")), int(inclusion_raw.get("total_child_count")),
            bool(inclusion_raw.get("all_children_failed")),
        )
        trajectory = tuple(SearchTrajectoryEntry(
            int(raw.get("counterfactual_probe_count")), tuple(raw.get("retained_ids") or []),
            int(raw.get("cost")), int(raw.get("retained_unit_count")), str(raw.get("stage")),
        ) for raw in value.get("trajectory") or [])
        return cls(
            status=str(value.get("status")), certificate=value.get("certificate"),
            original_candidate=candidate(value.get("original_candidate")),
            best_candidate=candidate(value.get("best_candidate")),
            control_evidence=ref(value.get("control_evidence")), trials=tuple(trials),
            trajectory=trajectory, budget=budget, inclusion_check=inclusion,
            search_id=value.get("search_id"), base_execution_fingerprint=value.get("base_execution_fingerprint"),
            universe_id=value.get("universe_id"), policy_version=str(value.get("policy_version")),
            objective=value.get("objective") if isinstance(value.get("objective"), Mapping) else {},
        )

    @classmethod
    def from_reduction(cls, reduction: BudgetedReductionResult, *, search_id: str | None = None,
                       base_execution_fingerprint: str | None = None,
                       universe_id: str | None = None,
                       objective: Mapping[str, Any] | None = None) -> "SearchResult":
        def classify(evidence: Any, preserves: bool) -> str:
            if isinstance(evidence, Mapping):
                status = evidence.get("observation_status", evidence.get("status"))
                if status in {"exact_preserved", "matched", "preserves"}:
                    return "preserves"
                if status in {"diverged", "failed_direct", "not_preserved"}:
                    return "diverged"
                explicit = evidence.get("classification")
                if explicit in {"preserves", "diverged", "unknown"}:
                    return explicit
            return "preserves" if preserves else "unknown"

        trials = tuple(
            SearchTrial(
                ordinal=item.ordinal,
                stage=item.stage,
                retained_ids=tuple(item.retained_ids),
                cost=item.cost,
                classification=classify(item.evidence, item.preserves),
                evidence_ref=SearchEvidenceRef.from_value(item.evidence),
                disposition=(item.evidence.get("disposition")
                             if isinstance(item.evidence, Mapping)
                             and item.evidence.get("disposition") in {"reused", "executed", "not_executed"}
                             else "executed"),
                batch_id=item.batch_id,
                parent_retained_ids=tuple(item.parent_retained_ids),
            )
            for item in reduction.trials
        )
        budget = SearchBudget(
            max_new_executions=reduction.budget.max_counterfactual_probes,
            used_new_executions=reduction.budget.used_counterfactual_probes,
            reused_observation_count=sum(item.disposition == "reused" for item in trials),
            exhausted=reduction.budget.exhausted,
        )
        inclusion = reduction.inclusion_check
        return cls(
            status=reduction.status,
            certificate=reduction.certificate_level,
            original_candidate=reduction.original_candidate,
            best_candidate=reduction.best_candidate,
            control_evidence=SearchEvidenceRef.from_value(reduction.control_evidence),
            trials=trials,
            trajectory=tuple(SearchTrajectoryEntry(
                item.counterfactual_probe_count, tuple(item.retained_ids), item.cost,
                item.retained_unit_count, item.stage,
            ) for item in reduction.trajectory),
            budget=budget,
            inclusion_check=InclusionCheck(
                inclusion.attempted, inclusion.complete, inclusion.tested_child_count,
                inclusion.total_child_count, inclusion.all_children_failed,
            ),
            search_id=search_id,
            base_execution_fingerprint=base_execution_fingerprint,
            universe_id=universe_id,
            objective=objective,
        )


def classify_exact_observation(value: Any) -> str:
    """Classify only direct exact-reference statuses; everything else is unknown."""
    if isinstance(value, Mapping):
        status = value.get("observation_status", value.get("status"))
        if status in {"exact_preserved", "matched"}:
            return "preserves"
        if status == "diverged":
            return "diverged"
    return "unknown"


def run_adaptive_search(
    ordered_unit_ids: Iterable[UnitID],
    max_new_executions: int,
    prepare_candidate: Callable[[tuple[UnitID, ...]], Any],
    probe_many: Callable[[Sequence[PreparedCandidate]], Iterable[Any]],
    *,
    classify_evidence: Callable[[Any], str] = classify_exact_observation,
    candidate_is_reusable: Callable[[tuple[UnitID, ...]], bool] | None = None,
    attempt_inclusion_check: bool = True,
    search_id: str | None = None,
    base_execution_fingerprint: str | None = None,
    universe_id: str | None = None,
    objective: Mapping[str, Any] | None = None,
) -> SearchResult:
    """Run the one shared bounded reducer with explicit UNKNOWN semantics."""
    def preserving(value: Any) -> bool:
        return classify_evidence(value) == "preserves"

    def diverged(value: Any) -> bool:
        return classify_evidence(value) == "diverged"

    def charges(value: Any) -> bool:
        return not (isinstance(value, Mapping) and value.get("disposition") in {
            "reused", "not_executed",
        })

    reduction = run_budgeted_reduction(
        ordered_unit_ids,
        max_new_executions,
        prepare_candidate,
        probe_many,
        attempt_inclusion_check=attempt_inclusion_check,
        is_preserving=preserving,
        is_failed=diverged,
        candidate_is_reusable=candidate_is_reusable,
        probe_charges_budget=charges,
    )
    return SearchResult.from_reduction(
        reduction,
        search_id=search_id,
        base_execution_fingerprint=base_execution_fingerprint,
        universe_id=universe_id,
        objective=objective,
    )


# Explicit aliases make the contract names available to callers while the
# old reducer module remains a temporary parity import for existing reports.
SearchCandidate = Candidate


__all__ = [
    "BEST_VERIFIED", "CONTROL_FAILED", "INCLUSION_MINIMUM", "OK", "SEARCH_UNAVAILABLE",
    "SEARCH_POLICY_VERSION", "UnitID", "Candidate", "SearchCandidate", "PreparedCandidate",
    "SearchEvidenceRef", "SearchBudget", "SearchTrial", "SearchTrajectoryEntry", "InclusionCheck",
    "SearchResult", "classify_exact_observation", "run_adaptive_search",
]
