"""Bounded, measured verification of Context Dependence source coalitions.

This module is deliberately a *search policy* rather than another scoring
implementation.  It can use hierarchy structure or an estimated subset screen
to choose a small collection of source sets, but every item in
``verified_sets`` is backed by the result of a real measurement callback.

In particular, candidate generation never filters a pair because either of its
singletons had a small direct effect.  That is essential for substitutable
evidence, where removing A and B together can change the recorded answer even
though removing each separately does not.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from itertools import combinations
from typing import Any


DEFAULT_MAX_STRUCTURAL_CANDIDATE_SOURCES = 8
"""At most this many siblings receive automatic bounded pair enumeration."""

DEFAULT_MAX_CANDIDATES = 128
"""A defensive cap on heuristic candidates; the pass budget still governs work."""


class CoalitionVerificationError(ValueError):
    """Coalition selection or its measured evidence was not representable safely."""


@dataclass(frozen=True)
class _MeasuredExperiment:
    source_ids: tuple[str, ...]
    experiment_id: str
    provenance: str
    delta_nats: float | None
    raw: dict[str, Any]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if not _is_int(value) or value < 0:
        raise CoalitionVerificationError(f"{name} must be a non-negative integer")
    return value


def _canonical_source_order(source_ids: Iterable[str]) -> tuple[str, ...]:
    if isinstance(source_ids, (str, bytes)):
        raise CoalitionVerificationError("source_ids must be an iterable of non-empty source ID strings")
    try:
        ordered = tuple(source_ids)
    except TypeError as exc:
        raise CoalitionVerificationError(
            "source_ids must be an iterable of non-empty source ID strings") from exc
    if not ordered or any(not isinstance(source_id, str) or not source_id for source_id in ordered):
        raise CoalitionVerificationError("source_ids must contain at least one non-empty source ID")
    if len(set(ordered)) != len(ordered):
        raise CoalitionVerificationError("source_ids must not contain duplicates")
    return ordered


def _canonical_set(
    value: Iterable[str], *, source_order: tuple[str, ...], name: str,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise CoalitionVerificationError(f"{name} must be an iterable of source ID strings")
    try:
        supplied = tuple(value)
    except TypeError as exc:
        raise CoalitionVerificationError(f"{name} must be an iterable of source ID strings") from exc
    if not supplied or any(not isinstance(source_id, str) or not source_id for source_id in supplied):
        raise CoalitionVerificationError(f"{name} must contain at least one non-empty source ID")
    if len(set(supplied)) != len(supplied):
        raise CoalitionVerificationError(f"{name} must not contain duplicate source IDs")
    unknown = set(supplied).difference(source_order)
    if unknown:
        raise CoalitionVerificationError(
            f"{name} contains unknown Context Receipt source IDs: {', '.join(sorted(unknown))}")
    return tuple(source_id for source_id in source_order if source_id in supplied)


def _as_candidate_sets(value: Any, *, source_order: tuple[str, ...], origin: str) -> list[tuple[tuple[str, ...], str]]:
    """Accept compact candidate list forms without interpreting estimates as facts."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        # A single candidate object is the common screen representation.
        if "source_ids" in value:
            value = [value]
        else:
            # Mapping source-set labels to source-ID sequences is convenient
            # for callers that already have named search candidates.
            value = list(value.values())
    if isinstance(value, (str, bytes)):
        raise CoalitionVerificationError(f"{origin} candidates must be source-ID sets, not a string")
    try:
        entries = list(value)
    except TypeError as exc:
        raise CoalitionVerificationError(f"{origin} candidates must be iterable") from exc
    result: list[tuple[tuple[str, ...], str]] = []
    for index, entry in enumerate(entries):
        entry_origin = origin
        candidate = entry
        if isinstance(entry, Mapping):
            candidate = entry.get("source_ids")
            supplied_origin = entry.get("origin")
            if isinstance(supplied_origin, str) and supplied_origin:
                entry_origin = supplied_origin
        if candidate is None:
            raise CoalitionVerificationError(f"{origin} candidate {index} has no source_ids")
        source_set = _canonical_set(
            candidate, source_order=source_order, name=f"{origin} candidate {index}.source_ids",
        )
        # Coalition verification is specifically about a multi-source deletion.
        # Singleton direct experiments remain useful input, but are not emitted
        # as coalition candidates or used as an admission floor.
        if len(source_set) >= 2:
            result.append((source_set, entry_origin))
    return result


def _hierarchy_candidates(
    hierarchy: Any, *, source_order: tuple[str, ...],
    max_structural_candidate_sources: int = DEFAULT_MAX_STRUCTURAL_CANDIDATE_SOURCES,
) -> list[tuple[tuple[str, ...], str]]:
    """Propose parent and sibling coalitions from measured hierarchy structure.

    Nonadditivity is only a selection hint.  A parent that is already directly
    measured is still represented as a candidate so the result can reference
    that real parent experiment; no arithmetic is treated as an experiment.
    """
    if not isinstance(hierarchy, Mapping):
        return []
    hierarchy_value = hierarchy.get("hierarchy")
    tree = hierarchy_value if isinstance(hierarchy_value, Mapping) else hierarchy
    nodes = tree.get("nodes")
    if not isinstance(nodes, list):
        return []
    by_id: dict[str, tuple[str, ...]] = {}
    children_by_parent: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    result: list[tuple[tuple[str, ...], str]] = []
    for index, node in enumerate(nodes):
        if not isinstance(node, Mapping) or not isinstance(node.get("source_ids"), list):
            continue
        try:
            source_set = _canonical_set(
                node["source_ids"], source_order=source_order, name=f"hierarchy.nodes[{index}].source_ids",
            )
        except CoalitionVerificationError:
            # A hierarchy that includes source IDs outside this study is not a
            # safe source of candidates; unlike a caller's explicit candidate,
            # it is optional search metadata so it is ignored rather than
            # letting a stale hierarchy corrupt a new study.
            continue
        node_id = node.get("node_id")
        if isinstance(node_id, str) and node_id:
            by_id[node_id] = source_set
        parent_id = node.get("parent_node_id")
        if isinstance(parent_id, str) and parent_id:
            children_by_parent[parent_id].append(source_set)
        if len(source_set) >= 2:
            result.append((source_set, "hierarchy_parent_or_group"))

    # A parent whose individual leaves were weak must still be selected.  The
    # parent can have been absent from the current experiment cache (for
    # example, when hierarchy planning and measurement are separate stages),
    # in which case it will receive a direct arm below.
    for item in tree.get("nonadditivity", []):
        if not isinstance(item, Mapping):
            continue
        parent_id = item.get("parent_node_id")
        source_set = by_id.get(parent_id) if isinstance(parent_id, str) else None
        if source_set is not None and len(source_set) >= 2:
            result.append((source_set, "hierarchy_nonadditivity"))

    # Natural sibling leaves are a useful bounded pair pool.  This intentionally
    # happens without consulting their direct effect values.
    for children in children_by_parent.values():
        leaves = sorted({source_id for child in children if len(child) == 1 for source_id in child},
                       key=source_order.index)
        result.extend(_structural_group_candidates(
            leaves, source_order=source_order, origin="hierarchy_structural_siblings",
            max_structural_candidate_sources=max_structural_candidate_sources,
        ))
    return result


def _structural_group_candidates(
    group: Iterable[str], *, source_order: tuple[str, ...], origin: str,
    max_structural_candidate_sources: int = DEFAULT_MAX_STRUCTURAL_CANDIDATE_SOURCES,
) -> list[tuple[tuple[str, ...], str]]:
    source_set = _canonical_set(group, source_order=source_order, name=f"{origin}.source_ids")
    if len(source_set) < 2:
        return []
    result = [(source_set, origin)]
    # Eight siblings make 28 pairs, a bounded pool called out in the design
    # brief.  Larger contexts retain the group candidate and must rely on
    # hierarchy/screen hints instead of receiving an all-pairs explosion.
    if len(source_set) <= max_structural_candidate_sources:
        result.extend((tuple(pair), origin) for pair in combinations(source_set, 2))
    # For a three-source structural sibling group, the full-group candidate is
    # the needed three-way experiment.  We deliberately do not enumerate every
    # triple for larger sibling pools; screens or hierarchy can nominate those.
    return result


def _screen_candidates(screen: Any, *, source_order: tuple[str, ...]) -> list[tuple[tuple[str, ...], str]]:
    if not isinstance(screen, Mapping):
        return []
    result: list[tuple[tuple[str, ...], str]] = []
    for field in ("candidate_source_sets", "residual_candidate_source_sets", "strong_candidate_source_sets"):
        if field in screen:
            result.extend(_as_candidate_sets(
                screen[field], source_order=source_order, origin=f"screen:{field}",
            ))
    # Some screen implementations report a candidate *pool* as individual IDs.
    # It is still only a heuristic; bounded structural generation chooses the
    # actual groups to verify.
    candidate_ids = screen.get("candidate_source_ids")
    if candidate_ids:
        result.extend(_structural_group_candidates(
            candidate_ids, source_order=source_order, origin="screen:candidate_source_ids",
        ))
    return result


def _normalise_experiment(value: Any, *, source_order: tuple[str, ...], expected: tuple[str, ...] | None = None) -> _MeasuredExperiment:
    """Validate that a result is an actual direct experiment, never an estimate."""
    if isinstance(value, Mapping) and isinstance(value.get("experiments"), list):
        matching = []
        for experiment in value["experiments"]:
            try:
                normalised = _normalise_experiment(experiment, source_order=source_order, expected=expected)
            except CoalitionVerificationError:
                continue
            matching.append(normalised)
        if len(matching) != 1:
            wanted = list(expected) if expected is not None else "the requested source set"
            raise CoalitionVerificationError(
                f"measurement document must contain exactly one measured experiment for {wanted}")
        return matching[0]
    if isinstance(value, Mapping):
        raw = dict(value)
        removed = raw.get("removed_source_ids", raw.get("source_ids"))
        experiment_id = raw.get("experiment_id")
        provenance = raw.get("provenance")
        delta = raw.get("delta_nats")
    else:
        raw = {
            "experiment_id": getattr(value, "experiment_id", None),
            "removed_source_ids": getattr(value, "removed_source_ids", None),
            "provenance": getattr(value, "provenance", None),
        }
        removed = raw["removed_source_ids"]
        experiment_id = raw["experiment_id"]
        provenance = raw["provenance"]
        delta = getattr(value, "delta_nats", None)
        if delta is not None:
            raw["delta_nats"] = delta
    if not isinstance(experiment_id, str) or not experiment_id:
        raise CoalitionVerificationError("a direct coalition measurement must return a non-empty experiment_id")
    if provenance != "measured":
        raise CoalitionVerificationError(
            "a coalition can be verified only by a direct experiment with provenance='measured'")
    source_set = _canonical_set(
        removed, source_order=source_order, name="measured experiment.removed_source_ids",
    )
    if expected is not None and source_set != expected:
        raise CoalitionVerificationError(
            "measurement callback returned an experiment for a different source set")
    if delta is not None and (isinstance(delta, bool) or not isinstance(delta, (int, float))):
        raise CoalitionVerificationError("measured experiment.delta_nats must be a number when present")
    return _MeasuredExperiment(
        source_ids=source_set,
        experiment_id=experiment_id,
        provenance="measured",
        delta_nats=float(delta) if delta is not None else None,
        raw=deepcopy(raw),
    )


def _existing_experiments(
    values: Iterable[Any], *, source_order: tuple[str, ...],
) -> dict[tuple[str, ...], _MeasuredExperiment]:
    result: dict[tuple[str, ...], _MeasuredExperiment] = {}
    by_id: dict[str, tuple[str, ...]] = {}
    for value in values:
        experiment = _normalise_experiment(value, source_order=source_order)
        previous = result.get(experiment.source_ids)
        if previous is not None and previous.experiment_id != experiment.experiment_id:
            raise CoalitionVerificationError(
                "existing measured experiments contain conflicting experiment IDs for one source set")
        previous_set = by_id.get(experiment.experiment_id)
        if previous_set is not None and previous_set != experiment.source_ids:
            raise CoalitionVerificationError(
                "one experiment_id cannot verify two different source sets")
        result[experiment.source_ids] = experiment
        by_id[experiment.experiment_id] = experiment.source_ids
    return result


def _candidate_records(
    *,
    source_order: tuple[str, ...],
    candidate_source_sets: Any,
    hierarchy: Any,
    screen: Any,
    structural_sibling_groups: Any,
    max_structural_candidate_sources: int,
    max_candidates: int,
) -> list[dict[str, Any]]:
    candidates: list[tuple[tuple[str, ...], str]] = []
    candidates.extend(_as_candidate_sets(
        candidate_source_sets, source_order=source_order, origin="caller_candidate_source_sets",
    ))
    candidates.extend(_hierarchy_candidates(
        hierarchy, source_order=source_order,
        max_structural_candidate_sources=max_structural_candidate_sources,
    ))
    candidates.extend(_screen_candidates(screen, source_order=source_order))

    if structural_sibling_groups is not None:
        if isinstance(structural_sibling_groups, Mapping):
            groups = list(structural_sibling_groups.values())
        else:
            if isinstance(structural_sibling_groups, (str, bytes)):
                raise CoalitionVerificationError("structural_sibling_groups must be an iterable of source-ID groups")
            try:
                groups = list(structural_sibling_groups)
            except TypeError as exc:
                raise CoalitionVerificationError(
                    "structural_sibling_groups must be an iterable of source-ID groups") from exc
        for index, group in enumerate(groups):
            if isinstance(group, Mapping):
                group = group.get("source_ids")
            if group is None:
                raise CoalitionVerificationError(f"structural_sibling_groups[{index}] has no source_ids")
            candidates.extend(_structural_group_candidates(
                group, source_order=source_order, origin="structural_siblings",
                max_structural_candidate_sources=max_structural_candidate_sources,
            ))
    elif len(source_order) >= 2:
        # With no richer metadata, the receipt's small source list is itself a
        # structural sibling group.  This is what lets exact textual duplicates
        # be tested without text matching or singleton effect gating.
        candidates.extend(_structural_group_candidates(
            source_order, source_order=source_order, origin="receipt_sibling_pool",
            max_structural_candidate_sources=max_structural_candidate_sources,
        ))

    origins_by_set: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for source_set, origin in candidates:
        origins_by_set[source_set].add(origin)
    def priority(source_set: tuple[str, ...]) -> tuple[int, int, tuple[int, ...]]:
        origins = origins_by_set[source_set]
        # A residual/cohort screen set is a bounded interaction-search hint.
        # Give it a chance to receive its direct verification arm before the
        # fallback's enormous full-receipt deletion consumes a small budget.
        # This changes no epistemic status: every selected item below still
        # calls/uses only a measured delete-source experiment.
        screen_set = any(origin.startswith("screen:candidate_source_sets") for origin in origins)
        return (
            0 if screen_set else 1,
            -len(source_set),
            tuple(source_order.index(item) for item in source_set),
        )

    ordered = sorted(origins_by_set, key=priority)[:max_candidates]
    return [{"source_ids": list(source_set), "origins": sorted(origins_by_set[source_set])}
            for source_set in ordered]


def _result_experiment(experiment: _MeasuredExperiment, *, source: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_ids": list(experiment.source_ids),
        "experiment_id": experiment.experiment_id,
        "provenance": "measured",
        "measurement_source": source,
    }
    if experiment.delta_nats is not None:
        result["delta_nats"] = experiment.delta_nats
    return result


def verify_coalitions(
    measure: Callable[[tuple[str, ...]], Any],
    *,
    source_ids: Iterable[str],
    passes_requested: int,
    passes_consumed: int = 0,
    existing_experiments: Iterable[Any] = (),
    candidate_source_sets: Any = (),
    hierarchy: Any = None,
    screen: Any = None,
    structural_sibling_groups: Any = None,
    passes_per_measurement: int = 1,
    max_structural_candidate_sources: int = DEFAULT_MAX_STRUCTURAL_CANDIDATE_SOURCES,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> dict[str, Any]:
    """Select and directly verify a bounded set of source coalitions.

    ``measure`` is intentionally a narrow generic callback.  It receives one
    canonical tuple of source IDs and must return the real measured experiment
    (or a document containing exactly that experiment).  A Task 1
    ``ContextDependenceStudy.measure_removal_effect`` method fits directly.

    ``passes_consumed`` is the number already spent by the surrounding study;
    this makes the remaining budget explicit when a hierarchy or screen has
    already scored arms.  ``passes_per_measurement`` reserves each callback
    invocation before it happens, so the verifier cannot silently overrun the
    requested budget.  A callback with a different cost must expose that cost
    to its caller and use the corresponding reservation.
    """
    if not callable(measure):
        raise CoalitionVerificationError("measure must be a callable direct measurement callback")
    source_order = _canonical_source_order(source_ids)
    requested = _require_nonnegative_int(passes_requested, name="passes_requested")
    consumed = _require_nonnegative_int(passes_consumed, name="passes_consumed")
    if consumed > requested:
        raise CoalitionVerificationError("passes_consumed cannot exceed passes_requested")
    if not _is_int(passes_per_measurement) or passes_per_measurement < 1:
        raise CoalitionVerificationError("passes_per_measurement must be a positive integer")
    if not _is_int(max_structural_candidate_sources) or max_structural_candidate_sources < 2:
        raise CoalitionVerificationError("max_structural_candidate_sources must be an integer of at least 2")
    if not _is_int(max_candidates) or max_candidates < 1:
        raise CoalitionVerificationError("max_candidates must be a positive integer")

    existing_values = list(existing_experiments)
    # A Task 3 study document naturally supplies the direct experiment cache.
    if isinstance(hierarchy, Mapping) and isinstance(hierarchy.get("experiments"), list):
        existing_values.extend(hierarchy["experiments"])
    experiment_by_set = _existing_experiments(existing_values, source_order=source_order)
    candidates = _candidate_records(
        source_order=source_order,
        candidate_source_sets=candidate_source_sets,
        hierarchy=hierarchy,
        screen=screen,
        structural_sibling_groups=structural_sibling_groups,
        max_structural_candidate_sources=max_structural_candidate_sources,
        max_candidates=max_candidates,
    )

    verified_sets: list[dict[str, str | list[str]]] = []
    measured_experiments: list[dict[str, Any]] = []
    unverified: list[dict[str, Any]] = []
    verified_ids: set[str] = set()
    recorded_sets: set[tuple[str, ...]] = set()

    for candidate in candidates:
        source_set = tuple(candidate["source_ids"])
        experiment = experiment_by_set.get(source_set)
        measurement_source = "existing"
        if experiment is None:
            if consumed + passes_per_measurement > requested:
                unverified.append(deepcopy(candidate))
                continue
            # Reserve before calling the scorer.  A bad return fails closed and
            # never turns an estimated candidate into a verified source set.
            raw_result = measure(source_set)
            experiment = _normalise_experiment(raw_result, source_order=source_order, expected=source_set)
            previous = experiment_by_set.get(source_set)
            if previous is not None and previous.experiment_id != experiment.experiment_id:
                raise CoalitionVerificationError(
                    "measurement callback returned a conflicting experiment for an already measured set")
            experiment_by_set[source_set] = experiment
            consumed += passes_per_measurement
            measurement_source = "new"

        # Candidate selection may reach the same real parent/pair through
        # multiple hints.  Reference it once, preserving a compact audit trail.
        if experiment.experiment_id not in verified_ids:
            verified_sets.append({
                "source_ids": list(source_set),
                "experiment_id": experiment.experiment_id,
            })
            verified_ids.add(experiment.experiment_id)
        if source_set not in recorded_sets:
            measured_experiments.append(_result_experiment(experiment, source=measurement_source))
            recorded_sets.add(source_set)

    remaining = requested - consumed
    return {
        "provenance": "measured_set_verification",
        "selection": {
            "provenance": "search_heuristic",
            "candidate_sets": candidates,
            "unverified_candidate_sets": unverified,
        },
        # These references are deliberately the only notion of "verified" in
        # this module.  Estimated screen coefficients never appear here.
        "verified_sets": verified_sets,
        "measured_experiments": measured_experiments,
        "budget": {
            "passes_requested": requested,
            "passes_consumed": consumed,
            "passes_remaining": remaining,
            "state": "exhausted" if remaining == 0 else "complete",
        },
    }


def verify_task1_coalitions(
    measurement_study: Any,
    *,
    passes_requested: int,
    candidate_source_sets: Any = (),
    hierarchy: Any = None,
    screen: Any = None,
    structural_sibling_groups: Any = None,
    max_structural_candidate_sources: int = DEFAULT_MAX_STRUCTURAL_CANDIDATE_SOURCES,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> dict[str, Any]:
    """Adapt Task 1's reusable study to :func:`verify_coalitions`.

    Calling ``document()`` first exposes the exact baseline/pass count and any
    previously measured direct experiments.  A fresh Task 1 study therefore
    spends its baseline pass once, then reserves one pass per selected deletion
    arm.  This helper intentionally returns a separate verification payload;
    persistence/assembly of it into the study artifact belongs to the later
    orchestration layer.
    """
    requested = _require_nonnegative_int(passes_requested, name="passes_requested")
    if requested == 0:
        source_order = _canonical_source_order(getattr(measurement_study, "source_ids", ()))
        return verify_coalitions(
            lambda _source_set: (_ for _ in ()).throw(AssertionError("zero budget must not measure")),
            source_ids=source_order,
            passes_requested=0,
            candidate_source_sets=candidate_source_sets,
            hierarchy=hierarchy,
            screen=screen,
            structural_sibling_groups=structural_sibling_groups,
            max_structural_candidate_sources=max_structural_candidate_sources,
            max_candidates=max_candidates,
        )
    document_method = getattr(measurement_study, "document", None)
    measure_method = getattr(measurement_study, "measure_removal_effect", None)
    source_order = getattr(measurement_study, "source_ids", None)
    if not callable(document_method) or not callable(measure_method):
        raise CoalitionVerificationError(
            "measurement_study must provide document() and measure_removal_effect(source_ids)")
    baseline_document = document_method()
    if not isinstance(baseline_document, Mapping):
        raise CoalitionVerificationError("measurement_study.document() must return a study document")
    budget = baseline_document.get("budget")
    if not isinstance(budget, Mapping):
        raise CoalitionVerificationError("measurement study document has no budget")
    already_consumed = _require_nonnegative_int(budget.get("passes_consumed"), name="study budget.passes_consumed")
    if already_consumed > requested:
        raise CoalitionVerificationError(
            "the Task 1 study has already consumed more passes than the requested coalition budget")
    return verify_coalitions(
        measure_method,
        source_ids=_canonical_source_order(source_order),
        passes_requested=requested,
        passes_consumed=already_consumed,
        existing_experiments=baseline_document.get("experiments", ()),
        candidate_source_sets=candidate_source_sets,
        hierarchy=hierarchy,
        screen=screen,
        structural_sibling_groups=structural_sibling_groups,
        passes_per_measurement=1,
        max_structural_candidate_sources=max_structural_candidate_sources,
        max_candidates=max_candidates,
    )


# Clear descriptive aliases for callers that use either verb ordering.
run_coalition_verification = verify_coalitions
verify_context_dependence_coalitions = verify_coalitions


__all__ = [
    "DEFAULT_MAX_CANDIDATES",
    "DEFAULT_MAX_STRUCTURAL_CANDIDATE_SOURCES",
    "CoalitionVerificationError",
    "run_coalition_verification",
    "verify_coalitions",
    "verify_context_dependence_coalitions",
    "verify_task1_coalitions",
]
