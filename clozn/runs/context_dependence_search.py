"""Estimated, held-out-qualified Context Dependence subset screening.

This module implements only the *search* layer of Context Dependence.  It
selects deterministic source-deletion masks, asks a supplied direct-measurement
adapter to score every mask, and fits a deliberately small additive surrogate.
It never turns the surrogate into an experiment: mask observations remain
separate direct measurements and every coefficient is explicitly estimated.

The expected adapter is normally ``ContextDependenceStudy.measure_removal_effect``
from :mod:`clozn.receipts.context_dependence`.  That method uses the recorded
teacher-forced scoring primitive, so this module has no generation path and no
separate notion of an answer score.  The adapter shape is intentionally tiny to
also support deterministic model-free qualification tests:

.. code-block:: python

    screen = run_subset_screen(
        study.source_ids,
        study.measure_removal_effect,
        sampling_seed=17,
        passes_requested=24,
    )

``measure`` must return a direct measured record carrying
``removed_source_ids``, ``delta_nats``, and ``provenance == 'measured'``.  A
Task 1 experiment dictionary already has that shape; the test fixture's
``SyntheticMeasurement`` does too.

Budget note
-----------
The budget here is deliberately explicit and is enforced *before* a scorer is
called.  A caller declares the fixed direct-score cost of one mask and any
already-consumed shared-study passes (for example, a cached full-context
baseline).  This lets an orchestrator use one overall score-pass budget without
guessing after a call whether it has overrun it.  The default is one pass per
mask and zero pre-existing passes, which is appropriate for an adapter whose
baseline was already accounted for by the caller.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any


SCREEN_PROVENANCE = "estimated"
MEASUREMENT_PROVENANCE = "measured"
ESTIMATOR_NAME = "deterministic_additive_elastic_net"
ESTIMATOR_VERSION = "v1"
MASK_SAMPLER = "sha256_counter_subset_mask.v1"

# Interaction nominations are deliberately a *separate* estimated search
# product from additive coefficient qualification.  A poor additive fit is
# often precisely the signal that a jointly-deleted set deserves a direct
# experiment, so the nomination gate must not hide that information behind
# the additive model's "coefficient interpretation" gate.
INTERACTION_NOMINATOR = "residual_cooccurrence_enrichment.v1"
MAX_RESIDUAL_CANDIDATE_SOURCE_SETS = 8
MIN_RESIDUAL_OBSERVATIONS = 8
MIN_HIGH_RESIDUAL_PAIR_SUPPORT = 2
MIN_HIGH_RESIDUAL_HOLDOUT_SUPPORT = 1
MIN_HIGH_RESIDUAL_PAIR_RATE = 0.50
MIN_HIGH_RESIDUAL_RATE_ENRICHMENT = 0.25
RESIDUAL_ABS_FLOOR_NATS = 0.50
RESIDUAL_OUTLIER_QUANTILE = 0.75


class ContextDependenceScreenError(ValueError):
    """A subset screen cannot safely be sampled, scored, or qualified."""


class ScreenMeasurementError(ContextDependenceScreenError):
    """A score adapter did not return the requested direct measurement."""


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _source_ids(source_ids: Iterable[str]) -> tuple[str, ...]:
    """Validate receipt IDs while retaining their natural receipt order.

    Source order is not inferred from text.  Callers should pass the natural
    order from ``ContextDependenceStudy.source_ids`` (or the equivalent Context
    Receipt source order).  We retain that order for a readable mask vector;
    source identity itself remains the stable ID.
    """
    if isinstance(source_ids, (str, bytes)):
        raise ContextDependenceScreenError("source_ids must be an iterable of non-empty source ID strings")
    try:
        values = tuple(source_ids)
    except TypeError as exc:
        raise ContextDependenceScreenError(
            "source_ids must be an iterable of non-empty source ID strings"
        ) from exc
    if not values:
        raise ContextDependenceScreenError("subset screening requires at least one canonical source ID")
    if any(not isinstance(source_id, str) or not source_id for source_id in values):
        raise ContextDependenceScreenError("source_ids must contain only non-empty source ID strings")
    if len(set(values)) != len(values):
        raise ContextDependenceScreenError("source_ids must be unique canonical source IDs")
    return values


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        comparator = f">= {minimum}"
        raise ContextDependenceScreenError(f"{name} must be an integer {comparator}")
    return value


def _finite_number(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContextDependenceScreenError(f"{name} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ContextDependenceScreenError(f"{name} must be >= {minimum}")
    return number


def _seed(value: Any) -> int:
    return _integer(value, name="sampling_seed")


def _mask_bits_for_counter(source_count: int, sampling_seed: int, counter: int) -> tuple[int, ...]:
    """Derive a deterministic fair-looking bit vector without process RNG state."""
    bits: list[int] = []
    block = 0
    while len(bits) < source_count:
        material = f"clozn/context-dependence/mask/v1/{sampling_seed}/{counter}/{block}".encode("utf-8")
        digest = hashlib.sha256(material).digest()
        for byte in digest:
            for offset in range(8):
                bits.append((byte >> offset) & 1)
                if len(bits) == source_count:
                    return tuple(bits)
        block += 1
    return tuple(bits)  # defensive; the loop always returns above


def _mask_from_bits(
    source_ids: tuple[str, ...], bits: Sequence[int], *, sampling_seed: int, sample_index: int,
) -> dict[str, Any]:
    bit_values = tuple(int(bit) for bit in bits)
    removed = tuple(source_id for source_id, bit in zip(source_ids, bit_values) if bit)
    if not removed:
        raise ContextDependenceScreenError("internal error: subset mask must remove at least one source")
    mask_binding = {
        "sampler": MASK_SAMPLER,
        "sampling_seed": sampling_seed,
        "source_ids": list(source_ids),
        "sample_index": sample_index,
        "mask_bits": list(bit_values),
    }
    return {
        "mask_id": f"cdm_{_digest(mask_binding)[:24]}",
        "sample_index": sample_index,
        # 1 means this exact canonical Context Receipt source is deleted.
        "mask_bits": list(bit_values),
        "removed_source_ids": list(removed),
    }


def sample_subset_masks(
    source_ids: Iterable[str], *, sampling_seed: int, mask_count: int,
) -> list[dict[str, Any]]:
    """Return unique deterministic non-empty deletion masks.

    The returned masks are independent of wall-clock time and global RNG state.
    For small source populations we enumerate and seed-order the complete
    finite space, which avoids probabilistic retry behavior near exhaustion.
    For larger contexts, hash-counter sampling produces unique masks lazily.
    """
    ids = _source_ids(source_ids)
    seed = _seed(sampling_seed)
    count = _integer(mask_count, name="mask_count")
    if count == 0:
        return []

    source_count = len(ids)
    # Enumerating this bounded case also produces deterministic masks when a
    # caller requests every available non-empty subset.
    if source_count <= 16:
        available = (1 << source_count) - 1
        if count > available:
            raise ContextDependenceScreenError(
                f"mask_count={count} exceeds the {available} unique non-empty masks available"
            )
        bitmasks = list(range(1, available + 1))
        bitmasks.sort(key=lambda value: hashlib.sha256(
            f"clozn/context-dependence/mask-order/v1/{seed}/{value}".encode("utf-8")
        ).digest())
        return [
            _mask_from_bits(
                ids,
                tuple((value >> offset) & 1 for offset in range(source_count)),
                sampling_seed=seed,
                sample_index=index,
            )
            for index, value in enumerate(bitmasks[:count])
        ]

    masks: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    counter = 0
    # With more than 16 sources, a normal screen budget is vanishingly smaller
    # than the mask population.  Still cap retries so an accidental pathological
    # invocation cannot spin forever.
    max_attempts = max(1024, count * 128)
    while len(masks) < count and counter < max_attempts:
        bits = _mask_bits_for_counter(source_count, seed, counter)
        counter += 1
        if not any(bits) or bits in seen:
            continue
        seen.add(bits)
        masks.append(_mask_from_bits(ids, bits, sampling_seed=seed, sample_index=len(masks)))
    if len(masks) != count:
        raise ContextDependenceScreenError(
            "could not draw the requested number of unique subset masks within the deterministic retry limit"
        )
    return masks


def _partition_masks(masks: Sequence[dict[str, Any]], *, sampling_seed: int,
                     holdout_fraction: float) -> tuple[set[str], set[str]]:
    """Create an auditable deterministic fit/holdout split before fitting."""
    if not masks:
        return set(), set()
    if len(masks) == 1:
        return {str(masks[0]["mask_id"])}, set()
    holdout_count = max(1, int(round(len(masks) * holdout_fraction)))
    holdout_count = min(len(masks) - 1, holdout_count)
    ranked = sorted(
        (str(mask["mask_id"]) for mask in masks),
        key=lambda mask_id: hashlib.sha256(
            f"clozn/context-dependence/holdout/v1/{sampling_seed}/{mask_id}".encode("utf-8")
        ).digest(),
    )
    holdout = set(ranked[:holdout_count])
    return set(ranked[holdout_count:]), holdout


def _soft_threshold(value: float, penalty: float) -> float:
    if value > penalty:
        return value - penalty
    if value < -penalty:
        return value + penalty
    return 0.0


def _fit_additive_elastic_net(
    feature_rows: Sequence[Sequence[int]], observations: Sequence[float], *,
    l1_penalty: float, l2_penalty: float, max_iterations: int = 2_000,
    tolerance: float = 1e-10,
) -> tuple[float, list[float], dict[str, Any]]:
    """Fit a deterministic intercept + binary-source additive surrogate.

    This is intentionally compact stdlib coordinate descent, rather than a
    dependency-bearing statistical runtime.  Its only role is candidate
    selection; coefficients are never direct effects.
    """
    if len(feature_rows) != len(observations) or not feature_rows:
        raise ContextDependenceScreenError("fit data must contain aligned non-empty feature and observation rows")
    feature_count = len(feature_rows[0])
    if any(len(row) != feature_count for row in feature_rows):
        raise ContextDependenceScreenError("fit feature rows must have a consistent source dimension")
    if feature_count == 0:
        raise ContextDependenceScreenError("fit data must include at least one source feature")

    row_count = len(feature_rows)
    intercept = sum(observations) / row_count
    coefficients = [0.0] * feature_count
    residual = [value - intercept for value in observations]
    converged = False
    completed_iterations = 0

    for iteration in range(max_iterations):
        completed_iterations = iteration + 1
        largest_change = 0.0

        intercept_shift = sum(residual) / row_count
        intercept += intercept_shift
        if intercept_shift:
            residual = [value - intercept_shift for value in residual]
        largest_change = abs(intercept_shift)

        for feature_index in range(feature_count):
            old = coefficients[feature_index]
            # rho is the partial unpenalized covariance for this coordinate.
            rho = sum(
                row[feature_index] * (residual[index] + row[feature_index] * old)
                for index, row in enumerate(feature_rows)
            ) / row_count
            denominator = (
                sum(row[feature_index] * row[feature_index] for row in feature_rows) / row_count
                + l2_penalty
            )
            new = _soft_threshold(rho, l1_penalty) / denominator if denominator else 0.0
            shift = new - old
            if shift:
                coefficients[feature_index] = new
                residual = [
                    value - row[feature_index] * shift
                    for value, row in zip(residual, feature_rows)
                ]
            largest_change = max(largest_change, abs(shift))
        if largest_change <= tolerance:
            converged = True
            break

    diagnostics = {
        "algorithm": "cyclic_coordinate_descent",
        "max_iterations": max_iterations,
        "iterations": completed_iterations,
        "converged": converged,
        "l1_penalty": l1_penalty,
        "l2_penalty": l2_penalty,
        "tolerance": tolerance,
    }
    return intercept, coefficients, diagnostics


def _predict(intercept: float, coefficients: Sequence[float], features: Sequence[int]) -> float:
    return intercept + sum(coefficient * value for coefficient, value in zip(coefficients, features))


def _regression_metrics(observed: Sequence[float], predicted: Sequence[float]) -> dict[str, float | None]:
    if len(observed) != len(predicted) or not observed:
        raise ContextDependenceScreenError("metrics require aligned non-empty observed and predicted values")
    errors = [actual - estimate for actual, estimate in zip(observed, predicted)]
    squared_error = sum(error * error for error in errors)
    mean = sum(observed) / len(observed)
    total_sum_squares = sum((actual - mean) ** 2 for actual in observed)
    r_squared = None if total_sum_squares <= 1e-15 else 1.0 - squared_error / total_sum_squares
    return {
        "observation_count": len(observed),
        "mae_nats": sum(abs(error) for error in errors) / len(errors),
        "rmse_nats": math.sqrt(squared_error / len(errors)),
        "max_abs_error_nats": max(abs(error) for error in errors),
        "r_squared": r_squared,
    }


def _matrix_rank(rows: Sequence[Sequence[float]], *, tolerance: float = 1e-10) -> int:
    """Return a small dense matrix rank using deterministic Gaussian elimination."""
    if not rows:
        return 0
    matrix = [list(map(float, row)) for row in rows]
    column_count = len(matrix[0])
    if any(len(row) != column_count for row in matrix):
        raise ContextDependenceScreenError("rank rows must share a common width")
    rank = 0
    row_count = len(matrix)
    for column in range(column_count):
        pivot = max(range(rank, row_count), key=lambda index: abs(matrix[index][column]), default=rank)
        if pivot >= row_count or abs(matrix[pivot][column]) <= tolerance:
            continue
        matrix[rank], matrix[pivot] = matrix[pivot], matrix[rank]
        divisor = matrix[rank][column]
        matrix[rank] = [value / divisor for value in matrix[rank]]
        for index in range(row_count):
            if index == rank:
                continue
            factor = matrix[index][column]
            if abs(factor) > tolerance:
                matrix[index] = [
                    value - factor * pivot_value
                    for value, pivot_value in zip(matrix[index], matrix[rank])
                ]
        rank += 1
        if rank == row_count:
            break
    return rank


def _field(measurement: Any, name: str, default: Any = None) -> Any:
    if isinstance(measurement, Mapping):
        return measurement.get(name, default)
    return getattr(measurement, name, default)


def _record_measurement(mask: dict[str, Any], measurement: Any, *, source_ids: tuple[str, ...],
                        score_passes_per_mask: int) -> dict[str, Any]:
    """Validate a direct score response without copying it into experiments[]."""
    returned_ids = _field(measurement, "removed_source_ids")
    if isinstance(returned_ids, (str, bytes)):
        raise ScreenMeasurementError("direct mask scorer returned malformed removed_source_ids")
    try:
        returned_set = frozenset(returned_ids)
    except TypeError as exc:
        raise ScreenMeasurementError("direct mask scorer returned malformed removed_source_ids") from exc
    requested_ids = tuple(mask["removed_source_ids"])
    if returned_set != frozenset(requested_ids) or len(returned_set) != len(requested_ids):
        raise ScreenMeasurementError(
            "direct mask scorer did not return a measurement for exactly the requested source set"
        )
    if not returned_set.issubset(source_ids):  # also protects an unusual equality implementation
        raise ScreenMeasurementError("direct mask scorer returned a source outside the canonical receipt IDs")
    provenance = _field(measurement, "provenance")
    if provenance != MEASUREMENT_PROVENANCE:
        raise ScreenMeasurementError(
            "subset screening accepts only direct measured mask observations (provenance='measured')"
        )
    observed = _finite_number(_field(measurement, "delta_nats"), name="direct mask delta_nats")
    reported_cost = _field(measurement, "score_passes", None)
    if reported_cost is not None and reported_cost != score_passes_per_mask:
        raise ScreenMeasurementError(
            "direct mask scorer's score_passes disagrees with the predeclared score_passes_per_mask"
        )

    record = deepcopy(mask)
    record["observed_delta_nats"] = observed
    record["measurement_provenance"] = MEASUREMENT_PROVENANCE
    experiment_id = _field(measurement, "experiment_id", None)
    if experiment_id is not None:
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ScreenMeasurementError("direct mask scorer returned a malformed experiment_id")
        # A reference is audit linkage only.  The screen does not place it into
        # the study's direct experiments collection or relabel it as estimated.
        record["measurement_experiment_id"] = experiment_id
    return record


def _upper_quantile(values: Sequence[float], quantile: float) -> float:
    """Return a deterministic nearest-rank upper quantile for non-empty data."""
    if not values:
        raise ContextDependenceScreenError("quantile requires at least one value")
    ordered = sorted(float(value) for value in values)
    # 0.75 means the smallest value with at least 75% of observations at or
    # below it.  The no-interpolation rule keeps equal residuals and replay
    # order from changing a nomination threshold.
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _residual_nomination_thresholds(*, max_candidates: int) -> dict[str, int | float]:
    """The stable, persisted conservative search gate for interaction hints."""
    return {
        "minimum_observation_count": MIN_RESIDUAL_OBSERVATIONS,
        "absolute_residual_floor_nats": RESIDUAL_ABS_FLOOR_NATS,
        "outlier_quantile": RESIDUAL_OUTLIER_QUANTILE,
        "minimum_same_sign_pair_support": MIN_HIGH_RESIDUAL_PAIR_SUPPORT,
        "minimum_heldout_same_sign_pair_support": MIN_HIGH_RESIDUAL_HOLDOUT_SUPPORT,
        "minimum_pair_high_residual_rate": MIN_HIGH_RESIDUAL_PAIR_RATE,
        "minimum_rate_enrichment_over_population": MIN_HIGH_RESIDUAL_RATE_ENRICHMENT,
        "maximum_candidate_source_sets": max_candidates,
    }


def _residual_interaction_nominations(
    source_ids: tuple[str, ...], observations: Sequence[Mapping[str, Any]], *,
    max_candidates: int = MAX_RESIDUAL_CANDIDATE_SOURCE_SETS,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Nominate bounded pair deletions from recurring additive-residual outliers.

    Each residual is ``directly measured mask delta - additive prediction``.
    That makes a residual useful for *where to look next*, but never a source
    or set effect.  To avoid a one-mask coincidence, a pair must occur in at
    least two high-residual masks of the same sign, including one held-out
    mask, and be substantially enriched relative to that sign's overall mask
    prevalence.  Direct coalition verification is still the only path from
    this estimated hint to a measured set effect.
    """
    if isinstance(max_candidates, bool) or not isinstance(max_candidates, int) or max_candidates < 1:
        raise ContextDependenceScreenError("max residual candidate source sets must be a positive integer")
    thresholds = _residual_nomination_thresholds(max_candidates=max_candidates)
    base = {
        "provenance": SCREEN_PROVENANCE,
        "not_a_measured_effect": True,
        "method": INTERACTION_NOMINATOR,
        "residual_definition": "observed_delta_nats minus fitted_additive_prediction_nats",
        "candidate_set_size": 2,
        "thresholds": thresholds,
    }
    if len(observations) < MIN_RESIDUAL_OBSERVATIONS:
        return ({
            **base,
            "state": "unavailable",
            "reason": (
                f"requires at least {MIN_RESIDUAL_OBSERVATIONS} directly observed subset masks; "
                f"received {len(observations)}"
            ),
            "observation_count": len(observations),
            "candidate_count": 0,
        }, [])

    residuals: list[float] = []
    for observation in observations:
        residual = _finite_number(
            observation.get("additive_residual_nats"), name="additive residual",  # type: ignore[arg-type]
        )
        residuals.append(residual)
    residual_threshold = max(
        RESIDUAL_ABS_FLOOR_NATS,
        _upper_quantile([abs(value) for value in residuals], RESIDUAL_OUTLIER_QUANTILE),
    )

    # For each source pair, retain how frequently it was observed at all and
    # which direct masks gave a same-sign high residual.  We only enumerate
    # pairs already co-occurring in a bounded number of scored masks; no
    # all-pairs source-population expansion is performed for a large receipt.
    pair_stats: dict[tuple[str, str], dict[str, Any]] = {}
    high_masks_by_direction: dict[str, list[Mapping[str, Any]]] = {"positive": [], "negative": []}
    source_positions = {source_id: index for index, source_id in enumerate(source_ids)}
    for observation, residual in zip(observations, residuals):
        removed = observation.get("removed_source_ids")
        if not isinstance(removed, list):  # observations are internal, but fail closed if refactored
            raise ContextDependenceScreenError("screen observation is missing canonical removed_source_ids")
        removed_set = set(removed)
        if len(removed_set) != len(removed) or not removed_set.issubset(source_positions):
            raise ContextDependenceScreenError("screen observation has invalid canonical removed_source_ids")
        ordered_removed = tuple(source_id for source_id in source_ids if source_id in removed_set)
        is_high = abs(residual) >= residual_threshold
        direction = "positive" if residual > 0 else "negative" if residual < 0 else None
        if is_high and direction is not None:
            high_masks_by_direction[direction].append(observation)
        for left_index, left in enumerate(ordered_removed):
            for right in ordered_removed[left_index + 1:]:
                pair = (left, right)
                stats = pair_stats.setdefault(pair, {
                    "observation_count": 0,
                    "positive": [],
                    "negative": [],
                })
                stats["observation_count"] += 1
                if is_high and direction is not None:
                    stats[direction].append(observation)

    candidates: list[dict[str, Any]] = []
    population_rate = {
        direction: len(masks) / len(observations)
        for direction, masks in high_masks_by_direction.items()
    }
    for pair, stats in pair_stats.items():
        observed_count = int(stats["observation_count"])
        for direction in ("positive", "negative"):
            support = list(stats[direction])
            support_count = len(support)
            if support_count < MIN_HIGH_RESIDUAL_PAIR_SUPPORT:
                continue
            heldout_support = [item for item in support if item.get("split") == "holdout"]
            if len(heldout_support) < MIN_HIGH_RESIDUAL_HOLDOUT_SUPPORT:
                continue
            high_rate = support_count / observed_count
            if high_rate < MIN_HIGH_RESIDUAL_PAIR_RATE:
                continue
            if high_rate < population_rate[direction] + MIN_HIGH_RESIDUAL_RATE_ENRICHMENT:
                continue
            support_ids = [str(item["mask_id"]) for item in support]
            support_experiment_ids = [
                str(item["measurement_experiment_id"])
                for item in support if isinstance(item.get("measurement_experiment_id"), str)
            ]
            support_residuals = [float(item["additive_residual_nats"]) for item in support]
            candidates.append({
                "source_ids": list(pair),
                # Keep the verifier's origin as ``screen:candidate_source_sets``
                # (the common boundary for all estimated screen hints).  This
                # more specific provenance remains audit metadata rather than
                # replacing that route-level origin.
                "nomination_origin": "screen_residual_cooccurrence",
                "provenance": SCREEN_PROVENANCE,
                "not_a_measured_effect": True,
                "interaction_nominator": INTERACTION_NOMINATOR,
                "residual_direction": direction,
                "observed_pair_mask_count": observed_count,
                "high_residual_mask_count": support_count,
                "high_residual_holdout_mask_count": len(heldout_support),
                "high_residual_rate": high_rate,
                "population_high_residual_rate": population_rate[direction],
                "rate_enrichment": high_rate - population_rate[direction],
                "residual_threshold_abs_nats": residual_threshold,
                "aggregate_abs_residual_nats": sum(abs(value) for value in support_residuals),
                "minimum_abs_residual_nats": min(abs(value) for value in support_residuals),
                "supporting_mask_ids": support_ids,
                "supporting_measurement_experiment_ids": support_experiment_ids,
                "supporting_splits": [str(item.get("split")) for item in support],
            })

    candidates.sort(key=lambda item: (
        -float(item["aggregate_abs_residual_nats"]),
        -int(item["high_residual_mask_count"]),
        -float(item["rate_enrichment"]),
        tuple(source_positions[source_id] for source_id in item["source_ids"]),
    ))
    candidates = candidates[:max_candidates]
    high_masks = {
        direction: [str(item["mask_id"]) for item in masks]
        for direction, masks in high_masks_by_direction.items()
    }
    return ({
        **base,
        "state": "available",
        "observation_count": len(observations),
        "residual_threshold_abs_nats": residual_threshold,
        "high_residual_mask_ids": high_masks,
        "candidate_count": len(candidates),
    }, candidates)


def _empty_screen(
    source_ids: tuple[str, ...], *, sampling_seed: int, passes_requested: int,
    initial_passes_consumed: int, score_passes_per_mask: int, reason: str,
) -> dict[str, Any]:
    return {
        "provenance": SCREEN_PROVENANCE,
        "status": "unavailable",
        "sampling_seed": sampling_seed,
        "source_ids": list(source_ids),
        "mask_semantics": "1 means delete the source at the same source_ids index",
        "sampler": {"name": MASK_SAMPLER, "version": "v1"},
        "masks": [],
        "estimator": {
            "name": ESTIMATOR_NAME,
            "version": ESTIMATOR_VERSION,
            "provenance": SCREEN_PROVENANCE,
        },
        "coefficients": [],
        "training_fit": None,
        "holdout": None,
        "qualification": {
            "state": "unavailable",
            "candidate_interpretation_available": False,
            "reasons": [reason],
        },
        "candidate_source_ids": [],
        "candidate_source_sets": [],
        "interaction_nominations": {
            "provenance": SCREEN_PROVENANCE,
            "not_a_measured_effect": True,
            "method": INTERACTION_NOMINATOR,
            "residual_definition": "observed_delta_nats minus fitted_additive_prediction_nats",
            "candidate_set_size": 2,
            "thresholds": _residual_nomination_thresholds(
                max_candidates=MAX_RESIDUAL_CANDIDATE_SOURCE_SETS,
            ),
            "state": "unavailable",
            "reason": reason,
            "observation_count": 0,
            "candidate_count": 0,
        },
        "budget": {
            "passes_requested": passes_requested,
            "initial_passes_consumed": initial_passes_consumed,
            "score_passes_per_mask": score_passes_per_mask,
            "passes_consumed": initial_passes_consumed,
            "passes_remaining": passes_requested - initial_passes_consumed,
            "mask_passes_consumed": 0,
            "exhausted": initial_passes_consumed == passes_requested,
        },
    }


def run_subset_screen(
    source_ids: Iterable[str], measure: Callable[[Iterable[str]], Any], *,
    sampling_seed: int, passes_requested: int, mask_count: int | None = None,
    initial_passes_consumed: int = 0, score_passes_per_mask: int = 1,
    existing_measurements: Iterable[Any] = (),
    holdout_fraction: float = 0.25, l1_penalty: float = 0.01,
    l2_penalty: float = 1e-8, max_holdout_mae_nats: float = 0.25,
    min_holdout_observations: int = 2, max_candidate_source_ids: int = 8,
) -> dict[str, Any]:
    """Score deterministic deletion masks and return an *estimated* screen.

    ``measure`` is called only for masks that fit in the explicit remaining
    budget.  It must be a direct teacher-forced measurement adapter, not an
    estimated score.  ``initial_passes_consumed`` is for a shared baseline or
    other already-executed score passes; it is recorded, never inferred.

    A fitted model is considered qualified only when its fit design has full
    intercept-plus-source rank, it has genuine held-out observations, and its
    held-out MAE is within the caller-visible threshold.  Any failure empties
    coefficient-derived ``candidate_source_ids``.  Separately, recurring
    out-of-fit residual patterns can nominate bounded *sets* for direct
    testing even when an additive fit is unqualified; those hints remain
    explicitly estimated and are never reported as source effects.
    """
    ids = _source_ids(source_ids)
    if not callable(measure):
        raise ContextDependenceScreenError("measure must be a callable direct measurement adapter")
    seed = _seed(sampling_seed)
    requested = _integer(passes_requested, name="passes_requested")
    initial = _integer(initial_passes_consumed, name="initial_passes_consumed")
    per_mask = _integer(score_passes_per_mask, name="score_passes_per_mask", minimum=1)
    if initial > requested:
        raise ContextDependenceScreenError("initial_passes_consumed cannot exceed passes_requested")
    explicit_mask_count = mask_count is not None
    if explicit_mask_count:
        mask_limit = _integer(mask_count, name="mask_count")
    else:
        mask_limit = (requested - initial) // per_mask
    fraction = _finite_number(holdout_fraction, name="holdout_fraction")
    if not 0.0 < fraction < 1.0:
        raise ContextDependenceScreenError("holdout_fraction must be strictly between 0 and 1")
    l1 = _finite_number(l1_penalty, name="l1_penalty", minimum=0.0)
    l2 = _finite_number(l2_penalty, name="l2_penalty", minimum=0.0)
    max_holdout_mae = _finite_number(
        max_holdout_mae_nats, name="max_holdout_mae_nats", minimum=0.0,
    )
    minimum_holdout = _integer(min_holdout_observations, name="min_holdout_observations", minimum=1)
    candidate_limit = _integer(max_candidate_source_ids, name="max_candidate_source_ids")

    existing_by_set: dict[frozenset[str], Any] = {}
    for index, measurement in enumerate(existing_measurements):
        returned_ids = _field(measurement, "removed_source_ids")
        if isinstance(returned_ids, (str, bytes)):
            raise ScreenMeasurementError(
                f"existing_measurements[{index}] has malformed removed_source_ids")
        try:
            key = frozenset(returned_ids)
        except TypeError as exc:
            raise ScreenMeasurementError(
                f"existing_measurements[{index}] has malformed removed_source_ids") from exc
        if not key or not key.issubset(ids):
            raise ScreenMeasurementError(
                f"existing_measurements[{index}] is not a measured subset of this source set")
        if _field(measurement, "provenance") != MEASUREMENT_PROVENANCE:
            raise ScreenMeasurementError(
                "existing subset observations must have provenance='measured'")
        previous = existing_by_set.get(key)
        if previous is not None and _field(previous, "experiment_id") != _field(measurement, "experiment_id"):
            raise ScreenMeasurementError(
                "existing subset observations conflict for the same source set")
        existing_by_set[key] = measurement

    remaining = requested - initial
    allowed_masks = remaining // per_mask
    # Cached direct experiments cost no new score pass.  With a cache, draw
    # the requested policy population and reserve budget only for missing
    # masks; without one, retain the original strict pre-budget cap.
    target_mask_count = mask_limit if existing_by_set else min(mask_limit, allowed_masks)
    # A default screen samples as much of the available finite mask population
    # as its pass budget permits.  An explicit oversized request still reaches
    # ``sample_subset_masks`` and is rejected rather than silently changing a
    # caller's requested experimental design.
    if not explicit_mask_count and len(ids) <= 16:
        target_mask_count = min(target_mask_count, (1 << len(ids)) - 1)
    if target_mask_count == 0:
        return _empty_screen(
            ids, sampling_seed=seed, passes_requested=requested,
            initial_passes_consumed=initial, score_passes_per_mask=per_mask,
            reason="no score-pass budget remains for a direct subset-mask observation",
        )

    masks = sample_subset_masks(ids, sampling_seed=seed, mask_count=target_mask_count)
    fit_mask_ids, holdout_mask_ids = _partition_masks(masks, sampling_seed=seed, holdout_fraction=fraction)
    observations: list[dict[str, Any]] = []
    consumed = initial
    reused_observations = 0
    for mask in masks:
        raw_measurement = existing_by_set.get(frozenset(mask["removed_source_ids"]))
        reused = raw_measurement is not None
        # This condition is intentionally before ``measure``: a score adapter
        # never sees a mask it cannot afford under the declared fixed cost.
        if not reused and consumed + per_mask > requested:
            continue
        if not reused:
            raw_measurement = measure(tuple(mask["removed_source_ids"]))
        record = _record_measurement(
            mask, raw_measurement, source_ids=ids, score_passes_per_mask=per_mask,
        )
        record["measurement_reused"] = reused
        record["split"] = "fit" if record["mask_id"] in fit_mask_ids else "holdout"
        observations.append(record)
        if reused:
            reused_observations += 1
        else:
            consumed += per_mask

    # Keep the output ordered by sample index, not split, so replay sees exactly
    # the score call sequence.  The explicit split labels make leakage auditable.
    fit_rows = [record for record in observations if record["split"] == "fit"]
    holdout_rows = [record for record in observations if record["split"] == "holdout"]
    fit_features = [record["mask_bits"] for record in fit_rows]
    fit_effects = [record["observed_delta_nats"] for record in fit_rows]
    holdout_features = [record["mask_bits"] for record in holdout_rows]
    holdout_effects = [record["observed_delta_nats"] for record in holdout_rows]

    # There is always at least one fit row when there are observations.  Fit
    # even under an inadequate design so diagnostics are preserved; qualification
    # below still fails closed.
    intercept, weights, solver = _fit_additive_elastic_net(
        fit_features, fit_effects, l1_penalty=l1, l2_penalty=l2,
    )
    fit_predictions = [_predict(intercept, weights, row) for row in fit_features]
    holdout_predictions = [_predict(intercept, weights, row) for row in holdout_features]
    all_predictions = [_predict(intercept, weights, record["mask_bits"]) for record in observations]
    for record, prediction in zip(observations, all_predictions):
        record["fitted_additive_prediction_nats"] = prediction
        record["additive_residual_nats"] = record["observed_delta_nats"] - prediction
    fit_metrics = _regression_metrics(fit_effects, fit_predictions)
    holdout_metrics = (
        _regression_metrics(holdout_effects, holdout_predictions) if holdout_effects else None
    )
    design_rank = _matrix_rank([[1.0, *row] for row in fit_features])
    required_rank = len(ids) + 1
    training_fit = {
        "fit_mask_ids": [record["mask_id"] for record in fit_rows],
        "fit_observation_count": len(fit_rows),
        "design_rank": design_rank,
        "required_full_rank": required_rank,
        "intercept_nats": intercept,
        "metrics": fit_metrics,
        "solver": solver,
    }
    holdout = {
        "holdout_mask_ids": [record["mask_id"] for record in holdout_rows],
        "holdout_observation_count": len(holdout_rows),
        "metrics": holdout_metrics,
        "max_mae_nats_for_qualification": max_holdout_mae,
        "used_for_fitting": False,
    }

    reasons: list[str] = []
    if len(fit_rows) < required_rank:
        reasons.append(
            f"fit has {len(fit_rows)} observations but needs at least {required_rank} for intercept-plus-source estimation"
        )
    if design_rank < required_rank:
        reasons.append(f"fit design rank {design_rank} is below required full rank {required_rank}")
    if len(holdout_rows) < minimum_holdout:
        reasons.append(
            f"holdout has {len(holdout_rows)} observations but requires at least {minimum_holdout}"
        )
    if holdout_metrics is not None and holdout_metrics["mae_nats"] > max_holdout_mae:
        reasons.append(
            "held-out MAE "
            f"{holdout_metrics['mae_nats']:.12g} nats exceeds qualification maximum {max_holdout_mae:.12g}"
        )
    qualified = not reasons

    coefficients = [
        {
            "source_id": source_id,
            "estimated_removal_coefficient_nats": coefficient,
            "provenance": SCREEN_PROVENANCE,
            "not_a_measured_effect": True,
        }
        for source_id, coefficient in zip(ids, weights)
    ]
    ranked_coefficients = sorted(
        coefficients,
        key=lambda record: (-abs(float(record["estimated_removal_coefficient_nats"])), ids.index(record["source_id"])),
    )
    candidates = [record["source_id"] for record in ranked_coefficients[:candidate_limit]] if qualified else []
    interaction_nominations, candidate_source_sets = _residual_interaction_nominations(ids, observations)

    return {
        "provenance": SCREEN_PROVENANCE,
        "status": "completed",
        "sampling_seed": seed,
        "source_ids": list(ids),
        "mask_semantics": "1 means delete the source at the same source_ids index",
        "sampler": {"name": MASK_SAMPLER, "version": "v1"},
        "masks": observations,
        "estimator": {
            "name": ESTIMATOR_NAME,
            "version": ESTIMATOR_VERSION,
            "provenance": SCREEN_PROVENANCE,
            "kind": "additive binary deletion-mask surrogate",
            "coefficient_warning": "Coefficients are estimated search signals, never direct experiments or measured source effects.",
        },
        "coefficients": coefficients,
        "training_fit": training_fit,
        "holdout": holdout,
        "qualification": {
            "state": "qualified" if qualified else "unqualified",
            "candidate_interpretation_available": qualified,
            "reasons": reasons,
        },
        # These are *future direct-test nominations*.  No coefficient or
        # residual appears in a measured experiment.  The interaction
        # candidates are pair sets selected from recurring residual structure,
        # not inferred source effects; a coalition verifier must still score
        # their direct delete-source experiment before reporting a set effect.
        "candidate_source_ids": candidates,
        "candidate_source_sets": candidate_source_sets,
        "interaction_nominations": interaction_nominations,
        "budget": {
            "passes_requested": requested,
            "initial_passes_consumed": initial,
            "score_passes_per_mask": per_mask,
            "passes_consumed": consumed,
            "passes_remaining": requested - consumed,
            "mask_passes_consumed": consumed - initial,
            **({"measurements_reused": reused_observations} if reused_observations else {}),
            "exhausted": consumed == requested,
        },
    }


# Explicit aliases make the limited Layer 3 surface readable at call sites
# without implying that it is a study or coalition-verification implementation.
screen_context_dependence_subsets = run_subset_screen
run_context_dependence_screen = run_subset_screen


__all__ = [
    "ContextDependenceScreenError",
    "ESTIMATOR_NAME",
    "ESTIMATOR_VERSION",
    "MASK_SAMPLER",
    "MEASUREMENT_PROVENANCE",
    "SCREEN_PROVENANCE",
    "ScreenMeasurementError",
    "run_context_dependence_screen",
    "run_subset_screen",
    "sample_subset_masks",
    "screen_context_dependence_subsets",
]
