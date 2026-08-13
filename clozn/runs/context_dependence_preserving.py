"""Direct, budgeted search for Context Dependence preserving subsets.

This module answers a deliberately narrow question: which *directly tested*
smaller retained source sets keep the teacher-forced log likelihood of the
recorded target within a stated tolerance of the full supplied context?

It is a search layer, not an attribution layer.  In particular, a source that
is absent from a preserving result is not described as useless, redundant, or
unread.  A result only says that the exact retained set was tested under the
recorded continuation and passed the supplied likelihood tolerance.

The Task 1 primitive measures deleted sets.  This module makes the inverse
mapping explicit and auditable:

``retained R`` -> ``measure deletion of full_source_set - R``.

The generic entry point accepts a small direct-measurement adapter so the
bounded search is independently testable with the Task 2 synthetic worlds.
``run_preserving_subset_search_for_study`` is the production convenience
wrapper for a ``ContextDependenceStudy`` instance.  It accounts for that
primitive's one cached full-context baseline before spending deletion arms.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import combinations
import math
from typing import Any


MEASUREMENT_PROVENANCE = "measured"
PRESERVING_PROVENANCE = "verified_experiment"
SEARCH_VERSION = "deterministic_retained_subset_enumeration.v1"


class ContextDependencePreservingError(ValueError):
    """A preserving-subset request could not be tested faithfully."""


class PreservingMeasurementError(ContextDependencePreservingError):
    """A candidate adapter did not provide a direct measured experiment."""


@dataclass(frozen=True)
class ComputeLevelPolicy:
    """Central, deliberately modest defaults for Context Dependence compute.

    The policy is data, rather than a branch scattered through an orchestrator,
    so callers and tests can inspect the exact intended work profile.  The
    current module consumes the preserving-search fields; screen and coalition
    fields document the compatible Layer 3/4 envelope for a future composed
    study and do not cause those layers to run here.
    """

    name: str
    pass_budget: int
    max_direct_candidates: int
    max_preserving_subsets: int
    direct_source_work: bool
    subset_screen: bool
    heldout_qualification: bool
    adaptive_coalitions: bool
    subset_mask_count: int
    coalition_candidate_limit: int


COMPUTE_LEVELS: dict[str, ComputeLevelPolicy] = {
    # Root/top-level direct work plus only a few source-set candidates.  Its
    # total is intentionally within the brief's roughly 1--8 extra passes.
    "quick": ComputeLevelPolicy(
        name="Quick", pass_budget=8, max_direct_candidates=6,
        max_preserving_subsets=3, direct_source_work=False, subset_screen=False,
        heldout_qualification=False, adaptive_coalitions=False,
        subset_mask_count=0, coalition_candidate_limit=0,
    ),
    # Tens of passes: direct source work, an estimated screen with held-out
    # qualification, then direct verification/search selected by that work.
    "standard": ComputeLevelPolicy(
        name="Standard", pass_budget=32, max_direct_candidates=24,
        max_preserving_subsets=8, direct_source_work=True, subset_screen=True,
        heldout_qualification=True, adaptive_coalitions=False,
        subset_mask_count=16, coalition_candidate_limit=8,
    ),
    # A broader, still bounded envelope for additional masks, coalitions and
    # preserving-set direct checks.
    "deep": ComputeLevelPolicy(
        name="Deep", pass_budget=96, max_direct_candidates=72,
        max_preserving_subsets=16, direct_source_work=True, subset_screen=True,
        heldout_qualification=True, adaptive_coalitions=True,
        subset_mask_count=48, coalition_candidate_limit=24,
    ),
}


def get_compute_level_policy(level: str | ComputeLevelPolicy) -> ComputeLevelPolicy:
    """Return one immutable policy, accepting case-insensitive level names."""
    if isinstance(level, ComputeLevelPolicy):
        return level
    if not isinstance(level, str) or not level.strip():
        raise ContextDependencePreservingError(
            "compute_level must be Quick, Standard, Deep, or a ComputeLevelPolicy"
        )
    try:
        return COMPUTE_LEVELS[level.strip().lower()]
    except KeyError as exc:
        raise ContextDependencePreservingError(
            "compute_level must be Quick, Standard, or Deep"
        ) from exc


def compute_level_presets() -> dict[str, dict[str, Any]]:
    """Return serializable copies for configuration/UI callers and tests."""
    return {name: asdict(policy) for name, policy in COMPUTE_LEVELS.items()}


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ContextDependencePreservingError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContextDependencePreservingError(f"{name} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ContextDependencePreservingError(f"{name} must be >= {minimum}")
    return number


def _source_ids(value: Iterable[str], *, name: str = "source_ids") -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ContextDependencePreservingError(f"{name} must be an iterable of non-empty source ID strings")
    try:
        ids = tuple(value)
    except TypeError as exc:
        raise ContextDependencePreservingError(
            f"{name} must be an iterable of non-empty source ID strings"
        ) from exc
    if not ids:
        raise ContextDependencePreservingError("preserving-subset search requires at least one canonical source ID")
    if any(not isinstance(source_id, str) or not source_id for source_id in ids):
        raise ContextDependencePreservingError(f"{name} must contain only non-empty source ID strings")
    if len(ids) != len(set(ids)):
        raise ContextDependencePreservingError(f"{name} must contain unique canonical source IDs")
    return ids


def _canonical_retained_set(
    retained_source_ids: Iterable[str], *, source_ids: tuple[str, ...], name: str,
) -> tuple[str, ...]:
    if isinstance(retained_source_ids, (str, bytes)):
        raise ContextDependencePreservingError(f"{name} must be an iterable of source ID strings")
    try:
        supplied = tuple(retained_source_ids)
    except TypeError as exc:
        raise ContextDependencePreservingError(f"{name} must be an iterable of source ID strings") from exc
    if any(not isinstance(source_id, str) or not source_id for source_id in supplied):
        raise ContextDependencePreservingError(f"{name} must contain only non-empty source ID strings")
    if len(supplied) != len(set(supplied)):
        raise ContextDependencePreservingError(f"{name} must not contain duplicate source IDs")
    unknown = set(supplied).difference(source_ids)
    if unknown:
        raise ContextDependencePreservingError(
            f"{name} includes unknown canonical source IDs: {', '.join(sorted(unknown))}"
        )
    # Receipt order controls tie-breaking and the display.  It deliberately
    # does not infer a semantic order from labels or source text.
    supplied_set = set(supplied)
    return tuple(source_id for source_id in source_ids if source_id in supplied_set)


def _retained_candidates(
    candidate_sets: Iterable[Iterable[str]], *, source_ids: tuple[str, ...], name: str,
) -> list[tuple[str, ...]]:
    if isinstance(candidate_sets, (str, bytes)):
        raise ContextDependencePreservingError(f"{name} must be an iterable of retained source sets")
    try:
        values = list(candidate_sets)
    except TypeError as exc:
        raise ContextDependencePreservingError(f"{name} must be an iterable of retained source sets") from exc
    canonical: dict[frozenset[str], tuple[str, ...]] = {}
    for index, retained in enumerate(values):
        item = _canonical_retained_set(retained, source_ids=source_ids, name=f"{name}[{index}]")
        # The full set does not require an intervention and cannot carry the
        # mandatory experiment ID.  It is already represented by the baseline.
        if len(item) == len(source_ids):
            continue
        canonical.setdefault(frozenset(item), item)
    return sorted(canonical.values(), key=lambda item: (len(item), item))


def _default_candidates(source_ids: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
    """Yield smaller retained sets from smallest to largest, receipt-order ties.

    This is intentionally lazy: a large receipt never creates its exponential
    power set merely because the caller's direct-score budget is small.
    """
    for count in range(len(source_ids)):
        yield from combinations(source_ids, count)


def _token_counts(
    source_ids: tuple[str, ...], source_token_counts: Mapping[str, int] | None,
) -> dict[str, int] | None:
    if source_token_counts is None:
        return None
    if not isinstance(source_token_counts, Mapping):
        raise ContextDependencePreservingError("source_token_counts must map every canonical source ID to a count")
    supplied = set(source_token_counts)
    expected = set(source_ids)
    if supplied != expected:
        missing = expected.difference(supplied)
        unknown = supplied.difference(expected)
        details = []
        if missing:
            details.append(f"missing: {', '.join(sorted(missing))}")
        if unknown:
            details.append(f"unknown: {', '.join(sorted(map(str, unknown)))}")
        raise ContextDependencePreservingError(
            "source_token_counts must cover exactly the canonical source IDs" + (f" ({'; '.join(details)})" if details else "")
        )
    normalized: dict[str, int] = {}
    for source_id in source_ids:
        normalized[source_id] = _integer(
            source_token_counts[source_id], name=f"source_token_counts[{source_id!r}]"
        )
    return normalized


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _direct_measurement(
    raw: Any, *, requested_removed: tuple[str, ...], source_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Validate the audit minimum before a search result may cite a score."""
    returned_ids = _field(raw, "removed_source_ids")
    if isinstance(returned_ids, (str, bytes)):
        raise PreservingMeasurementError("direct scorer returned malformed removed_source_ids")
    try:
        returned = tuple(returned_ids)
    except TypeError as exc:
        raise PreservingMeasurementError("direct scorer returned malformed removed_source_ids") from exc
    if (
        len(returned) != len(set(returned))
        or set(returned) != set(requested_removed)
        or not set(returned).issubset(source_ids)
    ):
        raise PreservingMeasurementError(
            "direct scorer did not return a measurement for exactly the requested deleted source set"
        )
    if _field(raw, "provenance") != MEASUREMENT_PROVENANCE:
        raise PreservingMeasurementError(
            "preserving-subset search accepts only direct measured experiments (provenance='measured')"
        )
    experiment_id = _field(raw, "experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise PreservingMeasurementError(
            "a preserving subset requires a real non-empty direct experiment_id; estimates may only nominate candidates"
        )
    delta = _finite(_field(raw, "delta_nats"), name="direct experiment delta_nats")
    return {
        "experiment_id": experiment_id,
        "removed_source_ids": list(source_id for source_id in source_ids if source_id in set(returned)),
        "delta_nats": delta,
        "provenance": MEASUREMENT_PROVENANCE,
    }


def _candidate_record(
    *, retained: tuple[str, ...], source_ids: tuple[str, ...], measurement: dict[str, Any],
    tolerance_nats: float, token_counts: dict[str, int] | None, origin: str,
) -> dict[str, Any]:
    retained_set = set(retained)
    removed = tuple(source_id for source_id in source_ids if source_id not in retained_set)
    absolute_difference = abs(float(measurement["delta_nats"]))
    record = {
        "retained_source_ids": list(retained),
        "removed_source_ids": list(removed),
        "full_source_count": len(source_ids),
        "retained_source_count": len(retained),
        "removed_source_count": len(removed),
        # Delta keeps Task 1's sign convention; absolute_difference is the
        # actual two-sided preserving test against the stated tolerance.
        "measured_difference_nats": measurement["delta_nats"],
        "absolute_difference_nats": absolute_difference,
        "tolerance_nats": tolerance_nats,
        "within_tolerance": absolute_difference <= tolerance_nats,
        "experiment_id": measurement["experiment_id"],
        "provenance": MEASUREMENT_PROVENANCE,
        "candidate_origin": origin,
    }
    if token_counts is not None:
        full_tokens = sum(token_counts.values())
        retained_tokens = sum(token_counts[source_id] for source_id in retained)
        record.update({
            "full_token_count": full_tokens,
            "retained_token_count": retained_tokens,
            "removed_token_count": full_tokens - retained_tokens,
        })
    return record


def _origin_map(
    *, source_ids: tuple[str, ...], explicit: Iterable[Iterable[str]] | None,
    estimated: Iterable[Iterable[str]] | None,
) -> tuple[list[tuple[str, ...]] | None, dict[frozenset[str], str], list[list[str]]]:
    """Build an explicit candidate pool without turning a nomination into proof."""
    origin: dict[frozenset[str], str] = {}
    estimated_records: list[list[str]] = []
    explicit_candidates = (
        _retained_candidates(explicit, source_ids=source_ids, name="candidate_retained_source_sets")
        if explicit is not None else None
    )
    if explicit_candidates is not None:
        for candidate in explicit_candidates:
            origin[frozenset(candidate)] = "direct_search_candidate"
    if estimated is not None:
        estimates = _retained_candidates(
            estimated, source_ids=source_ids, name="estimated_candidate_retained_source_sets",
        )
        estimated_records = [list(candidate) for candidate in estimates]
        if explicit_candidates is None:
            explicit_candidates = []
        seen = {frozenset(candidate) for candidate in explicit_candidates}
        for candidate in estimates:
            key = frozenset(candidate)
            # Direct/search-supplied candidates retain that reason if both
            # paths nominated them; an estimate is never allowed to overwrite
            # the audit label of an independently selected direct candidate.
            origin.setdefault(key, "estimated_screen_nomination")
            if key not in seen:
                explicit_candidates.append(candidate)
                seen.add(key)
        if explicit_candidates is not None:
            explicit_candidates.sort(key=lambda item: (len(item), item))
    return explicit_candidates, origin, estimated_records


def run_preserving_subset_search(
    source_ids: Iterable[str],
    measure: Callable[[Iterable[str]], Any],
    *,
    tolerance_nats: float,
    passes_requested: int | None = None,
    compute_level: str | ComputeLevelPolicy = "standard",
    candidate_retained_source_sets: Iterable[Iterable[str]] | None = None,
    estimated_candidate_retained_source_sets: Iterable[Iterable[str]] | None = None,
    source_token_counts: Mapping[str, int] | None = None,
    initial_passes_consumed: int = 0,
    score_passes_per_measurement: int = 1,
    existing_experiments: Iterable[Any] = (),
    max_candidates: int | None = None,
    max_preserving_subsets: int | None = None,
) -> dict[str, Any]:
    """Directly test smaller retained sets within an explicit score-pass budget.

    ``measure`` receives the *removed* source IDs and must return a measured
    Task 1-style experiment (including ``experiment_id``).  Candidate input is
    expressed as retained sets because that is the user-facing question; this
    function converts each one to its exact complement before calling
    ``measure``.  Estimated candidates are accepted only as nominations and
    are still required to survive that direct call.

    The generic default is one declared pass per adapter call.  For an actual
    ``ContextDependenceStudy`` use :func:`run_preserving_subset_search_for_study`:
    it first accounts for the primitive's cached full-context baseline.
    """
    ids = _source_ids(source_ids)
    if not callable(measure):
        raise ContextDependencePreservingError("measure must be a direct measurement callable")
    tolerance = _finite(tolerance_nats, name="tolerance_nats", minimum=0.0)
    policy = get_compute_level_policy(compute_level)
    requested = policy.pass_budget if passes_requested is None else _integer(
        passes_requested, name="passes_requested", minimum=0,
    )
    initial = _integer(initial_passes_consumed, name="initial_passes_consumed", minimum=0)
    if initial > requested:
        raise ContextDependencePreservingError("initial_passes_consumed cannot exceed passes_requested")
    per_measurement = _integer(
        score_passes_per_measurement, name="score_passes_per_measurement", minimum=1,
    )
    candidate_limit = policy.max_direct_candidates if max_candidates is None else _integer(
        max_candidates, name="max_candidates", minimum=0,
    )
    solution_limit = policy.max_preserving_subsets if max_preserving_subsets is None else _integer(
        max_preserving_subsets, name="max_preserving_subsets", minimum=0,
    )
    token_counts = _token_counts(ids, source_token_counts)
    explicit_candidates, candidate_origins, estimated_records = _origin_map(
        source_ids=ids, explicit=candidate_retained_source_sets,
        estimated=estimated_candidate_retained_source_sets,
    )
    candidates: Iterable[tuple[str, ...]] = (
        explicit_candidates if explicit_candidates is not None else _default_candidates(ids)
    )

    consumed = initial
    tested: list[dict[str, Any]] = []
    preserving: list[dict[str, Any]] = []
    unmeasured: list[dict[str, Any]] = []
    experiment_cache: dict[frozenset[str], dict[str, Any]] = {}
    for index, raw in enumerate(existing_experiments):
        returned_ids = _field(raw, "removed_source_ids")
        if isinstance(returned_ids, (str, bytes)):
            raise PreservingMeasurementError(
                f"existing_experiments[{index}] has malformed removed_source_ids")
        try:
            returned_set = set(returned_ids)
        except TypeError as exc:
            raise PreservingMeasurementError(
                f"existing_experiments[{index}] has malformed removed_source_ids") from exc
        canonical_removed = tuple(source_id for source_id in ids if source_id in returned_set)
        if not canonical_removed or len(canonical_removed) != len(returned_set):
            raise PreservingMeasurementError(
                f"existing_experiments[{index}] is not a measured subset of this source set")
        validated = _direct_measurement(
            raw, requested_removed=canonical_removed, source_ids=ids,
        )
        key = frozenset(canonical_removed)
        previous = experiment_cache.get(key)
        if previous is not None and previous != validated:
            raise PreservingMeasurementError(
                "existing experiments conflict for the same deleted source set")
        experiment_cache[key] = validated
    reused_measurements = 0
    candidates_seen = 0
    stopped_reason: str | None = None

    for retained in candidates:
        if candidates_seen >= candidate_limit:
            stopped_reason = "candidate_limit_reached"
            break
        candidates_seen += 1
        retained_set = frozenset(retained)
        removed = tuple(source_id for source_id in ids if source_id not in retained_set)
        if not removed:  # defensive: candidate normalization has already excluded full context.
            continue
        origin = candidate_origins.get(retained_set, "deterministic_enumeration")
        cached = experiment_cache.get(frozenset(removed))
        if cached is None:
            if consumed + per_measurement > requested:
                unmeasured.append({
                    "retained_source_ids": list(retained),
                    "removed_source_ids": list(removed),
                    "candidate_origin": origin,
                    "measurement_state": "unmeasured_budget_exhausted",
                })
                stopped_reason = "score_budget_exhausted"
                break
            raw = measure(removed)
            cached = _direct_measurement(raw, requested_removed=removed, source_ids=ids)
            experiment_cache[frozenset(removed)] = cached
            consumed += per_measurement
        else:
            reused_measurements += 1
        record = _candidate_record(
            retained=retained, source_ids=ids, measurement=cached,
            tolerance_nats=tolerance, token_counts=token_counts, origin=origin,
        )
        tested.append(record)
        if record["within_tolerance"] and len(preserving) < solution_limit:
            preserving.append(deepcopy(record))

    # Candidate generation itself is in deterministic size/receipt order.  Do
    # the final sort as an invariant too, including after an explicit candidate
    # list was passed in a different order.
    preserving.sort(key=lambda item: (
        item["retained_source_count"], tuple(item["retained_source_ids"]), item["experiment_id"],
    ))
    budget_exhausted = consumed >= requested
    if stopped_reason is None and budget_exhausted:
        stopped_reason = "score_budget_exhausted"
    full_token_count = sum(token_counts.values()) if token_counts is not None else None
    result: dict[str, Any] = {
        "provenance": PRESERVING_PROVENANCE,
        "search_version": SEARCH_VERSION,
        "full_source_ids": list(ids),
        "full_source_count": len(ids),
        "tolerance_nats": tolerance,
        "preserving_subsets": preserving,
        "tested_retained_subsets": tested,
        "search": {
            "candidate_order": "ascending retained source count, then canonical Context Receipt source order",
            "estimated_candidates_are_nominations_only": True,
            "estimated_candidate_retained_source_sets": estimated_records,
            "candidate_limit": candidate_limit,
            "candidates_considered": candidates_seen,
            "direct_measurements_reused": reused_measurements,
            "stopped_reason": stopped_reason,
            "compute_level": policy.name,
            "policy": asdict(policy),
        },
        "budget": {
            "passes_requested": requested,
            "initial_passes_consumed": initial,
            "score_passes_per_measurement": per_measurement,
            "passes_consumed": consumed,
            "passes_remaining": requested - consumed,
            "measurement_passes_consumed": consumed - initial,
            "exhausted": budget_exhausted,
        },
        "unmeasured_candidates": unmeasured,
    }
    if full_token_count is not None:
        result["full_token_count"] = full_token_count
    return result


def run_preserving_subset_search_for_study(
    measurement_study: Any,
    *,
    tolerance_nats: float,
    passes_requested: int | None = None,
    compute_level: str | ComputeLevelPolicy = "standard",
    candidate_retained_source_sets: Iterable[Iterable[str]] | None = None,
    estimated_candidate_retained_source_sets: Iterable[Iterable[str]] | None = None,
    source_token_counts: Mapping[str, int] | None = None,
    max_candidates: int | None = None,
    max_preserving_subsets: int | None = None,
) -> dict[str, Any]:
    """Run a preserving search against Task 1's cached-baseline study object.

    ``ContextDependenceStudy.document()`` is intentionally used first because
    it is the primitive's public way to ensure and account for the exact
    full-context teacher-forced baseline.  Its baseline is not a preserving
    result (there is no smaller intervention); every returned preserving set
    nevertheless points to a later real deletion experiment ID.
    """
    source_ids = getattr(measurement_study, "source_ids", None)
    measure = getattr(measurement_study, "measure_removal_effect", None)
    document = getattr(measurement_study, "document", None)
    if source_ids is None or not callable(measure) or not callable(document):
        raise ContextDependencePreservingError(
            "measurement_study must expose source_ids, measure_removal_effect(), and document()"
        )
    baseline_document = document()
    budget = _field(baseline_document, "budget")
    baseline_consumed = _field(budget, "passes_consumed")
    initial = _integer(baseline_consumed, name="measurement_study baseline passes_consumed", minimum=1)
    result = run_preserving_subset_search(
        source_ids, measure, tolerance_nats=tolerance_nats,
        passes_requested=passes_requested, compute_level=compute_level,
        candidate_retained_source_sets=candidate_retained_source_sets,
        estimated_candidate_retained_source_sets=estimated_candidate_retained_source_sets,
        source_token_counts=source_token_counts, initial_passes_consumed=initial,
        score_passes_per_measurement=1,
        existing_experiments=_field(baseline_document, "experiments", ()),
        max_candidates=max_candidates,
        max_preserving_subsets=max_preserving_subsets,
    )
    # Fail closed if a future measurement primitive consumes a different
    # number of score passes than its public document reports.  Do not replace
    # budget accounting with a favorable local estimate.
    final_document = document()
    final_consumed = _integer(
        _field(_field(final_document, "budget"), "passes_consumed"),
        name="measurement_study final passes_consumed", minimum=1,
    )
    if final_consumed != result["budget"]["passes_consumed"]:
        raise ContextDependencePreservingError(
            "measurement study pass accounting disagrees with preserving-subset search"
        )
    result["search"]["measurement_study_id"] = _field(final_document, "study_id")
    return result


# Descriptive aliases make call sites read naturally without hiding that this
# is a bounded verified-experiment search rather than an estimator.
search_preserving_subsets = run_preserving_subset_search
search_preserving_subsets_for_study = run_preserving_subset_search_for_study


__all__ = [
    "COMPUTE_LEVELS",
    "ComputeLevelPolicy",
    "ContextDependencePreservingError",
    "MEASUREMENT_PROVENANCE",
    "PRESERVING_PROVENANCE",
    "PreservingMeasurementError",
    "SEARCH_VERSION",
    "compute_level_presets",
    "get_compute_level_policy",
    "run_preserving_subset_search",
    "run_preserving_subset_search_for_study",
    "search_preserving_subsets",
    "search_preserving_subsets_for_study",
]
