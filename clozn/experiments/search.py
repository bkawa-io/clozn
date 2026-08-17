"""Model-free adaptive search contracts and reducer for the experimental kernel.

The reducer searches an ordered universe of retained unit IDs.  It knows
nothing about prompts, models, or the meaning of probe evidence: callers
prepare candidates, dispatch direct probes, and provide the preservation
predicate.  Candidate cost is intentionally opaque to the search and is read
only from the prepared candidate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any, TypeAlias


UnitID: TypeAlias = str | int

BEST_VERIFIED = "BEST_VERIFIED"
INCLUSION_MINIMUM = "INCLUSION_MINIMUM"
EXACT_MINIMUM = "EXACT_MINIMUM"
OK = "ok"
CONTROL_FAILED = "control_failed"


@dataclass(frozen=True)
class Candidate:
    """A retained unit set and its prepared objective value."""

    retained_ids: tuple[UnitID, ...]
    cost: int


@dataclass(frozen=True)
class PreparedCandidate(Mapping[str, Any]):
    """The minimum preparation contract passed to ``probe_many``.

    ``probe_payload`` is deliberately unconstrained.  Mapping access keeps
    this object convenient for small test or integration callbacks while the
    explicit fields keep the reducer's contract typed and inspectable.
    """

    retained_ids: tuple[UnitID, ...]
    cost: int
    probe_payload: Any

    @property
    def payload(self) -> Any:
        """Short alias for integrations that call the payload ``payload``."""
        return self.probe_payload

    def __getitem__(self, key: str) -> Any:
        if key == "retained_ids":
            return self.retained_ids
        if key in {"cost", "candidate_cost"}:
            return self.cost
        if key in {"probe_payload", "payload", "probe"}:
            return self.probe_payload
        raise KeyError(key)

    def __iter__(self):
        return iter(("retained_ids", "cost", "probe_payload"))

    def __len__(self) -> int:
        return 3


@dataclass(frozen=True)
class Trial:
    """One direct experiment recorded by the reducer."""

    ordinal: int
    stage: str
    retained_ids: tuple[UnitID, ...]
    cost: int
    preserves: bool
    evidence: Any
    # Observability-only fields.  They identify the reducer search round and
    # semantic current-best parent; they do not participate in ordering,
    # budgeting, or preservation decisions.
    batch_id: int | None = None
    parent_retained_ids: tuple[UnitID, ...] = ()


@dataclass(frozen=True)
class TrajectoryEntry:
    """A directly evidenced change to the current best candidate."""

    counterfactual_probe_count: int
    retained_ids: tuple[UnitID, ...]
    cost: int
    retained_unit_count: int
    stage: str


@dataclass(frozen=True)
class Budget:
    max_counterfactual_probes: int
    used_counterfactual_probes: int
    exhausted: bool
    blocked_by_budget: bool = False

    @property
    def total_direct_experiments(self) -> int:
        """The mandatory control plus the altered-context experiments."""
        return 1 + self.used_counterfactual_probes


@dataclass(frozen=True)
class InclusionCheck:
    """Evidence state for the final one-unit-removal sweep."""

    attempted: bool
    complete: bool
    tested_child_count: int
    total_child_count: int
    all_children_failed: bool = False


@dataclass(frozen=True)
class BudgetedReductionResult:
    """The reducer result and its direct-evidence ledger."""

    status: str
    certificate_level: str | None
    original_candidate: Candidate
    best_candidate: Candidate
    control_evidence: Any
    trials: tuple[Trial, ...]
    trajectory: tuple[TrajectoryEntry, ...]
    budget: Budget
    inclusion_check: InclusionCheck

    @property
    def direct_experiments(self) -> dict[str, int]:
        return {
            "control": 1,
            "counterfactual": self.budget.used_counterfactual_probes,
            "total": self.budget.total_direct_experiments,
        }

    def to_dict(self, *, include_evidence: bool = True) -> dict[str, Any]:
        """Return a JSON-friendly projection for small reports."""

        def candidate(value: Candidate) -> dict[str, Any]:
            return {
                "retained_ids": list(value.retained_ids),
                "cost": value.cost,
            }

        result: dict[str, Any] = {
            "status": self.status,
            "certificate_level": self.certificate_level,
            "original_candidate": candidate(self.original_candidate),
            "best_candidate": candidate(self.best_candidate),
            "budget": {
                "max_counterfactual_probes": self.budget.max_counterfactual_probes,
                "used_counterfactual_probes": self.budget.used_counterfactual_probes,
                "exhausted": self.budget.exhausted,
            },
            "direct_experiments": self.direct_experiments,
            "trajectory": [
                {
                    "counterfactual_probe_count": item.counterfactual_probe_count,
                    "retained_ids": list(item.retained_ids),
                    "cost": item.cost,
                    "retained_unit_count": item.retained_unit_count,
                    "stage": item.stage,
                }
                for item in self.trajectory
            ],
            "inclusion_check": {
                "attempted": self.inclusion_check.attempted,
                "complete": self.inclusion_check.complete,
                "tested_child_count": self.inclusion_check.tested_child_count,
                "total_child_count": self.inclusion_check.total_child_count,
                "all_children_failed": self.inclusion_check.all_children_failed,
            },
        }
        if include_evidence:
            result["control_evidence"] = deepcopy(self.control_evidence)
            result["trials"] = [
                {
                    "ordinal": item.ordinal,
                    "stage": item.stage,
                    "retained_ids": list(item.retained_ids),
                    "cost": item.cost,
                    "preserves": item.preserves,
                    "evidence": deepcopy(item.evidence),
                    "batch_id": item.batch_id,
                    "parent_retained_ids": list(item.parent_retained_ids),
                }
                for item in self.trials
            ]
        return result


@dataclass
class _Observation:
    candidate: Candidate
    prepared: PreparedCandidate
    preserves: bool
    failed: bool
    evidence: Any


def _validate_universe(ordered_unit_ids: Iterable[UnitID]) -> tuple[UnitID, ...]:
    if isinstance(ordered_unit_ids, (str, bytes)):
        raise ValueError("ordered_unit_ids must be an iterable of unit IDs")
    try:
        values = tuple(ordered_unit_ids)
    except TypeError as exc:
        raise ValueError("ordered_unit_ids must be an iterable of unit IDs") from exc
    if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in values):
        raise ValueError("unit IDs must be strings or integers")
    if len(set(values)) != len(values):
        raise ValueError("ordered_unit_ids must not contain duplicates")
    return values


def _validate_subset(values: Iterable[UnitID], universe: tuple[UnitID, ...]) -> tuple[UnitID, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("retained_ids must be an iterable of unit IDs")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError("retained_ids must be an iterable of unit IDs") from exc
    if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in raw):
        raise ValueError("retained_ids must contain only string or integer IDs")
    if len(set(raw)) != len(raw):
        raise ValueError("retained_ids must not contain duplicates")
    unknown = set(raw).difference(universe)
    if unknown:
        raise ValueError(f"retained_ids contains IDs outside the ordered universe: {sorted(unknown, key=str)!r}")
    selected = set(raw)
    return tuple(value for value in universe if value in selected)


def _prepare_fields(raw: Any, retained_ids: tuple[UnitID, ...]) -> PreparedCandidate:
    missing = object()
    if isinstance(raw, PreparedCandidate):
        returned_ids = raw.retained_ids
        cost = raw.cost
        payload = raw.probe_payload
    elif isinstance(raw, Mapping):
        returned_ids = raw.get("retained_ids", raw.get("retained_unit_ids", missing))
        cost = raw.get("cost", missing)
        payload = raw.get("probe_payload", raw.get("payload", raw.get("probe", missing)))
        if payload is missing:
            payload = raw
    elif isinstance(raw, (tuple, list)) and len(raw) == 3:
        returned_ids, cost, payload = raw
    else:
        returned_ids = getattr(raw, "retained_ids", missing)
        cost = getattr(raw, "cost", missing)
        payload = getattr(raw, "probe_payload", getattr(raw, "payload", raw))
    if returned_ids is missing:
        raise ValueError("prepare_candidate must return retained_ids")
    if isinstance(returned_ids, (str, bytes)):
        raise ValueError("prepare_candidate returned invalid retained_ids")
    try:
        returned_tuple = tuple(returned_ids)
    except TypeError as exc:
        raise ValueError("prepare_candidate returned invalid retained_ids") from exc
    if any(isinstance(value, bool) or not isinstance(value, (str, int)) for value in returned_tuple):
        raise ValueError("prepare_candidate returned invalid retained_ids")
    if len(set(returned_tuple)) != len(returned_tuple):
        raise ValueError("prepare_candidate returned duplicate retained_ids")
    if set(returned_tuple) != set(retained_ids) or len(returned_tuple) != len(retained_ids):
        raise ValueError("prepare_candidate returned retained_ids for a different candidate")
    if isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
        raise ValueError("candidate cost must be a non-negative integer")
    return PreparedCandidate(retained_ids=retained_ids, cost=cost, probe_payload=payload)


def _default_preservation_predicate(evidence: Any) -> bool:
    """Handle the two deliberately small model-free fake-oracle shapes."""
    if isinstance(evidence, bool):
        return evidence
    if isinstance(evidence, Mapping):
        if "preserves" in evidence:
            return evidence["preserves"] is True
        if "status" in evidence:
            return evidence["status"] == "matched"
    raise ValueError(
        "probe evidence needs a preservation predicate or a boolean/'preserves'/'status' result"
    )


def _default_failure_predicate(evidence: Any) -> bool:
    """Recognize an observed negative result without treating unknown as failure."""
    if isinstance(evidence, bool):
        return not evidence
    if isinstance(evidence, Mapping):
        if "preserves" in evidence:
            return evidence["preserves"] is False
        if "status" in evidence:
            return evidence["status"] == "diverged"
    raise ValueError(
        "probe evidence needs a failure predicate or a boolean/'preserves'/'status' result"
    )


def _candidate_order_key(
    candidate: Candidate, positions: Mapping[UnitID, int]
) -> tuple[int, int, tuple[int, ...]]:
    return (
        candidate.cost,
        len(candidate.retained_ids),
        tuple(positions[value] for value in candidate.retained_ids),
    )


def _split_groups(values: tuple[UnitID, ...], granularity: int) -> list[tuple[UnitID, ...]]:
    count = min(max(1, granularity), len(values))
    return [values[index * len(values) // count:(index + 1) * len(values) // count]
            for index in range(count)]


def run_budgeted_reduction(
    ordered_unit_ids: Iterable[UnitID],
    max_counterfactual_probes: int,
    prepare_candidate: Callable[[tuple[UnitID, ...]], Any],
    probe_many: Callable[[Sequence[PreparedCandidate]], Iterable[Any]],
    *,
    attempt_inclusion_check: bool = True,
    is_preserving: Callable[[Any], bool] | None = None,
    preservation_predicate: Callable[[Any], bool] | None = None,
    is_failed: Callable[[Any], bool] | None = None,
    failure_predicate: Callable[[Any], bool] | None = None,
    candidate_is_reusable: Callable[[tuple[UnitID, ...]], bool] | None = None,
    probe_charges_budget: Callable[[Any], bool] | None = None,
) -> BudgetedReductionResult:
    """Run the deterministic v0 budgeted deletion search.

    The unchanged full candidate is always probed first and is not charged to
    the counterfactual budget.  Every other direct experiment is dispatched
    through ``probe_many`` and consumes one probe per returned candidate.
    """

    universe = _validate_universe(ordered_unit_ids)
    if isinstance(max_counterfactual_probes, bool) or not isinstance(max_counterfactual_probes, int):
        raise ValueError("max_counterfactual_probes must be a non-negative integer")
    if max_counterfactual_probes < 0:
        raise ValueError("max_counterfactual_probes must be a non-negative integer")
    if is_preserving is not None and preservation_predicate is not None:
        raise ValueError("provide only one of is_preserving or preservation_predicate")
    if is_failed is not None and failure_predicate is not None:
        raise ValueError("provide only one of is_failed or failure_predicate")
    predicate = is_preserving or preservation_predicate or _default_preservation_predicate
    failed_predicate = is_failed or failure_predicate
    if failed_predicate is None and (is_preserving is not None or preservation_predicate is not None):
        def inferred_failure(evidence: Any) -> bool:
            if isinstance(evidence, Mapping) and evidence.get("status") == "unavailable":
                return False
            return not bool(predicate(evidence))
        failed_predicate = inferred_failure
    if failed_predicate is None:
        failed_predicate = _default_failure_predicate

    positions = {value: index for index, value in enumerate(universe)}
    direct_cache: dict[tuple[UnitID, ...], _Observation] = {}
    prepared_cache: dict[tuple[UnitID, ...], PreparedCandidate] = {}
    trials: list[Trial] = []
    trajectory: list[TrajectoryEntry] = []
    used_probes = 0
    budget_blocked = False
    next_ordinal = 1
    next_batch_id = 1
    dispatcher_owner = getattr(probe_many, "__self__", None)
    on_control_accepted = getattr(dispatcher_owner, "on_control_accepted", None)
    on_candidate_accepted = getattr(dispatcher_owner, "on_candidate_accepted", None)

    def prepare(retained_ids: tuple[UnitID, ...]) -> PreparedCandidate:
        cached = prepared_cache.get(retained_ids)
        if cached is not None:
            return cached
        prepared = _prepare_fields(prepare_candidate(retained_ids), retained_ids)
        prepared_cache[retained_ids] = prepared
        return prepared

    def probe_direct(
        records: list[tuple[Candidate, PreparedCandidate]], stage: str,
        *, parent_retained_ids: tuple[UnitID, ...] = (),
    ) -> list[_Observation]:
        nonlocal used_probes, budget_blocked, next_ordinal, next_batch_id
        pending = [(candidate, prepared) for candidate, prepared in records
                   if candidate.retained_ids not in direct_cache]
        if not pending:
            return [direct_cache[candidate.retained_ids] for candidate, _prepared in records]
        remaining = max_counterfactual_probes - used_probes
        if candidate_is_reusable is None:
            reusable_pending: list[tuple[Candidate, PreparedCandidate]] = []
        else:
            reusable_pending = [
                item for item in pending if candidate_is_reusable(item[0].retained_ids)
            ]
        new_pending = [item for item in pending if item not in reusable_pending]
        if len(new_pending) > remaining:
            budget_blocked = True
            new_pending = new_pending[:remaining]
        pending = reusable_pending + new_pending
        if not pending:
            return [direct_cache[candidate.retained_ids] for candidate, _prepared in records
                    if candidate.retained_ids in direct_cache]
        batch_id = next_batch_id
        next_batch_id += 1
        # Optional integration hook for experimental dispatchers that need the
        # semantic parent to select a native execution regime. It is
        # observability/dispatch metadata only; ordinary callables are
        # unaffected and retain the historical probe_many signature.
        owner = getattr(probe_many, "__self__", None)
        set_probe_context = getattr(owner, "set_probe_context", None)
        if callable(set_probe_context):
            set_probe_context(stage=stage, parent_retained_ids=tuple(parent_retained_ids))
        raw_results = list(probe_many([prepared for _candidate, prepared in pending]))
        if len(raw_results) != len(pending):
            raise ValueError(
                "probe_many must return exactly one evidence result per prepared candidate"
            )
        if probe_charges_budget is None:
            used_probes += len(pending)
        else:
            used_probes += sum(1 for evidence in raw_results if probe_charges_budget(evidence))
        for (candidate, prepared), evidence in zip(pending, raw_results):
            preserves = bool(predicate(evidence))
            failed = bool(failed_predicate(evidence))
            observation = _Observation(candidate, prepared, preserves, failed, deepcopy(evidence))
            direct_cache[candidate.retained_ids] = observation
            trials.append(Trial(
                ordinal=next_ordinal,
                stage=stage,
                retained_ids=candidate.retained_ids,
                cost=candidate.cost,
                preserves=preserves,
                evidence=deepcopy(evidence),
                batch_id=batch_id,
                parent_retained_ids=tuple(parent_retained_ids),
            ))
            next_ordinal += 1
        return [direct_cache[candidate.retained_ids] for candidate, _prepared in records
                if candidate.retained_ids in direct_cache]

    full_ids = universe
    full_prepared = prepare(full_ids)
    full_candidate = Candidate(full_ids, full_prepared.cost)
    full_results = list(probe_many([full_prepared]))
    if len(full_results) != 1:
        raise ValueError("probe_many must return exactly one control result")
    control_evidence = deepcopy(full_results[0])
    control_preserves = bool(predicate(full_results[0]))
    direct_cache[full_ids] = _Observation(
        full_candidate, full_prepared, control_preserves, False, deepcopy(full_results[0])
    )
    trials.append(Trial(
        ordinal=next_ordinal,
        stage="control",
        retained_ids=full_ids,
        cost=full_candidate.cost,
        preserves=control_preserves,
        evidence=deepcopy(full_results[0]),
        batch_id=0,
        parent_retained_ids=full_ids,
    ))
    next_ordinal += 1

    inclusion = InclusionCheck(False, False, 0, 0)
    if not control_preserves:
        budget = Budget(max_counterfactual_probes, 0, max_counterfactual_probes == 0)
        return BudgetedReductionResult(
            status=CONTROL_FAILED,
            certificate_level=None,
            original_candidate=full_candidate,
            best_candidate=full_candidate,
            control_evidence=control_evidence,
            trials=tuple(trials),
            trajectory=tuple(trajectory),
            budget=budget,
            inclusion_check=inclusion,
        )

    if callable(on_control_accepted):
        on_control_accepted(full_candidate, full_prepared, control_evidence)

    best = full_candidate
    granularity = 2
    certificate = BEST_VERIFIED

    def best_preserving(observations: Iterable[_Observation]) -> _Observation | None:
        preserving = [item for item in observations if item.preserves]
        if not preserving:
            return None
        return min(preserving, key=lambda item: _candidate_order_key(item.candidate, positions))

    def maybe_adopt(observation: _Observation, stage: str) -> bool:
        nonlocal best
        candidate = observation.candidate
        if _candidate_order_key(candidate, positions) >= _candidate_order_key(best, positions):
            return False
        best = candidate
        trajectory.append(TrajectoryEntry(
            counterfactual_probe_count=used_probes,
            retained_ids=best.retained_ids,
            cost=best.cost,
            retained_unit_count=len(best.retained_ids),
            stage=stage,
        ))
        if callable(on_candidate_accepted):
            on_candidate_accepted(candidate, observation.prepared, observation.evidence)
        return True

    inclusion_complete = False
    while best.retained_ids:
        # One-unit deletions belong to the optional certification sweep.  Do
        # not spend the coarse-stage budget on the same candidates twice.
        if granularity >= len(best.retained_ids):
            break
        groups = _split_groups(best.retained_ids, granularity)
        records: list[tuple[Candidate, PreparedCandidate]] = []
        seen: set[tuple[UnitID, ...]] = set()
        for group in groups:
            group_set = set(group)
            retained = tuple(value for value in best.retained_ids if value not in group_set)
            if retained in seen:
                continue
            seen.add(retained)
            prepared = prepare(retained)
            records.append((Candidate(retained, prepared.cost), prepared))
        records.sort(key=lambda item: _candidate_order_key(item[0], positions))
        observations = probe_direct(
            records, "coarse", parent_retained_ids=best.retained_ids,
        )
        chosen = best_preserving(observations)
        if chosen is not None and maybe_adopt(chosen, "coarse"):
            granularity = 2
            continue
        if granularity < len(best.retained_ids):
            granularity = min(len(best.retained_ids), granularity * 2)
            continue
        break

    # Reusable observations remain eligible even when no new execution budget
    # remains. ``probe_direct`` dispatches only those cached candidates in
    # that case, so inclusion can still be derived for free.
    if attempt_inclusion_check:
        while True:
            children: list[tuple[Candidate, PreparedCandidate]] = []
            for unit_id in best.retained_ids:
                retained = tuple(value for value in best.retained_ids if value != unit_id)
                prepared = prepare(retained)
                children.append((Candidate(retained, prepared.cost), prepared))
            children.sort(key=lambda item: _candidate_order_key(item[0], positions))
            total_children = len(children)
            observations = probe_direct(
                children, "inclusion", parent_retained_ids=best.retained_ids,
            )
            chosen = best_preserving(observations)
            if chosen is not None and maybe_adopt(chosen, "inclusion"):
                # A new current candidate requires a fresh sweep.  Its exact
                # children may reuse direct evidence, but the completion state
                # is recomputed from this candidate's children.
                continue
            child_observations = [
                direct_cache[child.retained_ids]
                for child, _prepared in children
                if child.retained_ids in direct_cache
            ]
            tested = len(child_observations)
            complete = (
                tested == total_children
                and chosen is None
                and all(not observation.preserves and observation.failed
                        for observation in child_observations)
            )
            if complete:
                inclusion_complete = True
            inclusion = InclusionCheck(
                True,
                complete,
                tested,
                total_children,
                all_children_failed=complete,
            )
            break
        if inclusion_complete:
            certificate = INCLUSION_MINIMUM
    else:
        inclusion = InclusionCheck(
            False,
            False,
            0,
            0,
        )

    budget = Budget(
        max_counterfactual_probes=max_counterfactual_probes,
        used_counterfactual_probes=used_probes,
        exhausted=used_probes >= max_counterfactual_probes,
        blocked_by_budget=budget_blocked,
    )
    return BudgetedReductionResult(
        status=OK,
        certificate_level=certificate,
        original_candidate=full_candidate,
        best_candidate=best,
        control_evidence=control_evidence,
        trials=tuple(trials),
        trajectory=tuple(trajectory),
        budget=budget,
        inclusion_check=inclusion,
    )


def accepted_trial(result: BudgetedReductionResult, trial: Trial) -> bool:
    """Whether a preserving trial was adopted into the direct trajectory."""
    return trial.preserves and any(
        entry.counterfactual_probe_count == trial.ordinal - 1
        and tuple(entry.retained_ids) == tuple(trial.retained_ids)
        for entry in result.trajectory
    )


__all__ = [
    "BEST_VERIFIED",
    "INCLUSION_MINIMUM",
    "OK",
    "CONTROL_FAILED",
    "Budget",
    "BudgetedReductionResult",
    "Candidate",
    "InclusionCheck",
    "PreparedCandidate",
    "TrajectoryEntry",
    "Trial",
    "UnitID",
    "accepted_trial",
    "run_budgeted_reduction",
]


# Durable search result identity and unavailable-state constants.
SEARCH_POLICY_VERSION = "adaptive_bounded_deletion.v1"
SEARCH_UNAVAILABLE = "search_unavailable"
SEARCH_POLICY_KIND = "adaptive_bounded_deletion"


def canonical_search_policy(*, attempt_inclusion_check: bool = True) -> dict[str, Any]:
    """Return every behavior-bearing switch in the bounded search policy."""
    if not isinstance(attempt_inclusion_check, bool):
        raise ValueError("attempt_inclusion_check must be a boolean")
    return {
        "kind": SEARCH_POLICY_KIND,
        "version": SEARCH_POLICY_VERSION,
        "attempt_inclusion_check": attempt_inclusion_check,
    }

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
    blocked_by_budget: bool = False

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
            "blocked_by_budget": self.blocked_by_budget,
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
    policy: Mapping[str, Any] | None = None
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
            "policy": deepcopy(dict(self.policy or canonical_search_policy())),
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
            blocked_by_budget=bool(budget_raw.get("blocked_by_budget", False)),
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
            policy=value.get("policy") if isinstance(value.get("policy"), Mapping) else None,
            objective=value.get("objective") if isinstance(value.get("objective"), Mapping) else {},
        )


    @classmethod
    def from_reduction(cls, reduction: BudgetedReductionResult, *, search_id: str | None = None,
                       base_execution_fingerprint: str | None = None,
                       universe_id: str | None = None,
                       policy: Mapping[str, Any] | None = None,
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
            reused_observation_count=sum(item.disposition == "reused" for item in trials if item.stage != "control"),
            exhausted=reduction.budget.exhausted,
            blocked_by_budget=reduction.budget.blocked_by_budget,
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
            policy=policy,
            objective=objective,
        )


def certify_exact_minimum(
    ordered_unit_ids: Iterable[UnitID],
    trials: Iterable[SearchTrial],
    *,
    control_evidence: SearchEvidenceRef | None = None,
    original_candidate: Candidate | None = None,
    winner: Candidate | None = None,
) -> dict[str, Any] | None:
    """Certify a global minimum from an already-complete direct evidence ledger.

    This is deliberately a pure read-side check.  It never enumerates the
    candidate powerset and never calls a model; it only proves that the
    supplied ledger contains one trusted direct classification for every
    subset of the finite universe.  Missing, unavailable, failed, or
    not-executed candidates therefore make the certificate unavailable.
    """
    universe = tuple(ordered_unit_ids)
    if len(set(universe)) != len(universe):
        raise ValueError("EXACT_MINIMUM requires a universe without duplicate IDs")
    positions = {value: index for index, value in enumerate(universe)}

    def canonical(raw: Iterable[UnitID]) -> tuple[UnitID, ...] | None:
        values = tuple(raw)
        if len(set(values)) != len(values) or not set(values).issubset(positions):
            return None
        return tuple(value for value in universe if value in set(values))

    def direct(ref: SearchEvidenceRef | None, classification: str) -> bool:
        if classification not in {"preserves", "diverged"} or ref is None:
            return False
        if ref.disposition == "not_executed" or not ref.observation_id:
            return False
        if classification == "preserves":
            return ref.observation_status in {"exact_preserved", "matched"}
        return ref.observation_status == "diverged"

    ledger: dict[tuple[UnitID, ...], tuple[int, str, SearchEvidenceRef]] = {}
    for trial in trials:
        candidate = canonical(trial.retained_ids)
        if candidate is None:
            return None
        if not direct(trial.evidence_ref, trial.classification):
            continue
        value = (trial.cost, trial.classification, trial.evidence_ref)
        prior = ledger.get(candidate)
        if prior is not None and (
            (prior[0], prior[1], prior[2].observation_id)
            != (value[0], value[1], value[2].observation_id)
        ):
            return None
        ledger[candidate] = value

    # A control is normally represented by the reducer's control trial.  The
    # explicit ref is accepted as an additional guard for callers that keep
    # the control outside the candidate trial sequence.
    full = universe
    if control_evidence is not None and direct(control_evidence, "preserves"):
        if original_candidate is None:
            return None
        prior = ledger.get(full)
        value = (original_candidate.cost, "preserves", control_evidence)
        if prior is not None and (
            (prior[0], prior[1], prior[2].observation_id)
            != (value[0], value[1], value[2].observation_id)
        ):
            return None
        ledger[full] = value

    expected = 2 ** len(universe)
    if len(ledger) != expected:
        return None

    preserving = [
        (candidate, value) for candidate, value in ledger.items()
        if value[1] == "preserves"
    ]
    if not preserving or any(value[0] < 0 for _candidate, value in ledger.items()):
        return None
    best_candidate, best_value = min(
        preserving,
        key=lambda item: (item[1][0], len(item[0]), tuple(positions[value] for value in item[0])),
    )
    if winner is not None:
        canonical_winner = canonical(winner.retained_ids)
        if canonical_winner != best_candidate or winner.cost != best_value[0]:
            return None
    return {
        "certificate": EXACT_MINIMUM,
        "candidate_space_size": expected,
        "directly_classified_subset_count": len(ledger),
        "preserving_subset_count": len(preserving),
        "diverged_subset_count": expected - len(preserving),
        "globally_minimum_cost": best_value[0],
        "winner_retained_ids": list(best_candidate),
        "coverage_complete": True,
    }

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
    policy: Mapping[str, Any] | None = None,
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
    resolved_policy = canonical_search_policy(
        attempt_inclusion_check=attempt_inclusion_check,
    )
    if policy is not None and dict(policy) != resolved_policy:
        raise ValueError("search policy does not match the reducer options")
    return SearchResult.from_reduction(
        reduction,
        search_id=search_id,
        base_execution_fingerprint=base_execution_fingerprint,
        universe_id=universe_id,
        policy=resolved_policy,
        objective=objective,
    )


# Explicit aliases keep the contract names discoverable at the experiment-layer boundary.
SearchCandidate = Candidate


__all__ = [
    "BEST_VERIFIED", "CONTROL_FAILED", "EXACT_MINIMUM", "INCLUSION_MINIMUM", "OK", "SEARCH_UNAVAILABLE",
    "SEARCH_POLICY_VERSION", "SEARCH_POLICY_KIND", "canonical_search_policy",
    "UnitID", "Candidate", "SearchCandidate", "PreparedCandidate",
    "Budget", "BudgetedReductionResult", "Trial", "TrajectoryEntry",
    "accepted_trial", "run_budgeted_reduction",
    "SearchEvidenceRef", "SearchBudget", "SearchTrial", "SearchTrajectoryEntry", "InclusionCheck",
    "SearchResult", "classify_exact_observation", "certify_exact_minimum", "run_adaptive_search",
]
