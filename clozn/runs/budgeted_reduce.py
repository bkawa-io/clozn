"""A small, model-free reducer for directly verified context candidates.

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
        nonlocal used_probes, next_ordinal, next_batch_id
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

    budget_exhausted = False
    inclusion_complete = False
    while best.retained_ids:
        if used_probes >= max_counterfactual_probes:
            budget_exhausted = True
            break
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
        if used_probes >= max_counterfactual_probes:
            budget_exhausted = True
            break
        if granularity < len(best.retained_ids):
            granularity = min(len(best.retained_ids), granularity * 2)
            continue
        break

    if not budget_exhausted and used_probes < max_counterfactual_probes and attempt_inclusion_check:
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
                if used_probes >= max_counterfactual_probes:
                    tested = sum(
                        1 for child, _prepared in children
                        if child.retained_ids in direct_cache
                    )
                    budget_exhausted = True
                    inclusion = InclusionCheck(True, False, tested, total_children)
                    break
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
            elif used_probes >= max_counterfactual_probes:
                budget_exhausted = True
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
    )
    if budget_exhausted and not inclusion_complete:
        certificate = BEST_VERIFIED
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


def budgeted_reduce(*args: Any, **kwargs: Any) -> BudgetedReductionResult:
    """Compatibility spelling for :func:`run_budgeted_reduction`."""
    return run_budgeted_reduction(*args, **kwargs)


def reduce_budgeted(*args: Any, **kwargs: Any) -> BudgetedReductionResult:
    """Compatibility spelling for :func:`run_budgeted_reduction`."""
    return run_budgeted_reduction(*args, **kwargs)


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
    "budgeted_reduce",
    "reduce_budgeted",
    "run_budgeted_reduction",
]
