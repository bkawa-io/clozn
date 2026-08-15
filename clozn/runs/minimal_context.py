"""Bounded minimal-context search with directly checked proof certificates.

This module is deliberately separate from ``context_dependence_preserving``.
The older search answers a different operational question (its Quick/Standard/
Deep passes); this module answers the finite, evidence-backed question of how
small a preserving retained-source set was directly checked to be.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import combinations, islice
import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping
from collections.abc import Sequence
from typing import Any

from clozn import schemas
from clozn.runs.answer_preservation import is_reference_match_preserving


SCHEMA = "clozn.minimal-context-result.v1"
SEARCH_METHOD = "greedy_backward_elimination.v1"
SEARCH_STRATEGY = "forward_reverse_intersection.v1"
PRESERVATION_KIND = "teacher_forced_likelihood"
PRESERVATION_TARGET = "whole_recorded_continuation"
EXACT_PRESERVATION_KIND = "exact_recorded_output"
DEFAULT_CERTIFICATION_BATCH_SIZE = 32


class MinimalContextError(ValueError):
    """Raised when a minimal-context request or direct evidence is invalid."""


class MinimalContextUnavailable(MinimalContextError):
    """Raised by callers that cannot perform the requested direct measurement."""


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _finite_number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MinimalContextError(f"{field} must be a finite number")
    return float(value)


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MinimalContextError(f"{field} must be a non-negative integer")
    return value


def _ordered_ids(values: Iterable[Any], field: str = "source_ids") -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise MinimalContextError(f"{field} must be an iterable of source IDs, not a string")
    try:
        raw = list(values)
    except TypeError as exc:
        raise MinimalContextError(f"{field} must be an iterable of source IDs") from exc
    if not raw or any(not isinstance(item, str) or not item for item in raw):
        raise MinimalContextError(f"{field} must contain non-empty string IDs")
    if len(set(raw)) != len(raw):
        raise MinimalContextError(f"{field} must not contain duplicate IDs")
    return tuple(raw)


def _canonical_set(
    values: Iterable[Any], universe: tuple[str, ...], field: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise MinimalContextError(f"{field} must be an iterable of source IDs, not a string")
    try:
        raw = list(values)
    except TypeError as exc:
        raise MinimalContextError(f"{field} must be an iterable of source IDs") from exc
    if not raw and not allow_empty:
        raise MinimalContextError(f"{field} must contain at least one source ID")
    ids = _ordered_ids(raw, field) if raw else ()
    unknown = set(ids).difference(universe)
    if unknown:
        raise MinimalContextError(f"{field} contains IDs outside the source universe: {sorted(unknown)!r}")
    return tuple(source_id for source_id in universe if source_id in set(ids))


def _manifest_digest(manifest: Mapping[str, Any] | None, source_ids: tuple[str, ...]) -> str:
    payload: Any = manifest if manifest is not None else {"source_ids": list(source_ids)}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _experiment_signature(experiment: Mapping[str, Any]) -> str:
    # score_ms and similar observational metadata must not make two direct
    # measurements conflict.  The identity, intervention, delta, and all
    # other stable evidence fields do bind the cache entry.
    stable = {key: value for key, value in experiment.items() if key not in {"score_ms", "elapsed_ms"}}
    return json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _normalise_experiment(raw: Any, universe: tuple[str, ...]) -> dict[str, Any]:
    experiment_id = _value(raw, "experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id:
        raise MinimalContextError("direct evidence must carry a non-empty experiment_id")
    removed_raw = _value(raw, "removed_source_ids")
    if removed_raw is None:
        raise MinimalContextError(f"experiment {experiment_id!r} has no removed_source_ids")
    removed = _canonical_set(removed_raw, universe, "removed_source_ids")
    if not removed:
        raise MinimalContextError("the empty deletion is not a direct deletion experiment")
    provenance = _value(raw, "provenance")
    if provenance != "measured":
        raise MinimalContextError(
            f"experiment {experiment_id!r} is not direct measured evidence (provenance={provenance!r})"
        )
    delta = _finite_number(_value(raw, "delta_nats"), "delta_nats")
    result = {
        "experiment_id": experiment_id,
        "removed_source_ids": list(removed),
        "delta_nats": delta,
        "provenance": "measured",
    }
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if key not in result and key not in {"score_ms", "elapsed_ms"}:
                result[key] = deepcopy(value)
    return result


def _normalise_probe(raw: Any, universe: tuple[str, ...]) -> dict[str, Any]:
    probe_id = _value(raw, "probe_id")
    if not isinstance(probe_id, str) or not probe_id:
        raise MinimalContextError("direct exact evidence must carry a non-empty probe_id")
    removed_raw = _value(raw, "removed_source_ids")
    if removed_raw is None:
        raise MinimalContextError(f"probe {probe_id!r} has no removed_source_ids")
    removed = _canonical_set(removed_raw, universe, "removed_source_ids")
    if not removed:
        raise MinimalContextError("the empty deletion is not a direct exact probe")
    provenance = _value(raw, "provenance")
    if provenance != "direct_generation_probe":
        raise MinimalContextError(
            f"probe {probe_id!r} is not direct generation evidence (provenance={provenance!r})"
        )
    result = _value(raw, "result")
    if not isinstance(result, Mapping):
        raise MinimalContextError(f"probe {probe_id!r} has no result")
    status = result.get("status")
    if status not in {"matched", "diverged", "unavailable"}:
        raise MinimalContextError(f"probe {probe_id!r} has an invalid result status")
    normalised = {
        "probe_id": probe_id,
        "removed_source_ids": list(removed),
        "result_status": status,
        "provenance": "direct_generation_probe",
        "result": deepcopy(dict(result)),
    }
    if isinstance(raw, Mapping):
        for key, value in raw.items():
            if key not in normalised and key not in {"score_ms", "elapsed_ms"}:
                normalised[key] = deepcopy(value)
    return normalised


class _EvidenceEngine:
    def __init__(
        self,
        universe: tuple[str, ...],
        measure_removed: Callable[[tuple[str, ...]], Any] | None,
        tolerance_nats: float,
        search_budget: int,
        certification_budget: int,
        existing_experiments: Iterable[Any] | None,
        *,
        exact: bool = False,
        measure_removed_many: Callable[[tuple[tuple[str, ...], ...]], Iterable[Any]] | None = None,
    ) -> None:
        self.universe = universe
        self.measure_removed = measure_removed
        self.measure_removed_many = measure_removed_many
        self.tolerance_nats = tolerance_nats
        self.exact = exact
        self.budgets = {"search": search_budget, "certification": certification_budget}
        self.new_probes = {"search": 0, "certification": 0}
        self.exhausted = {"search": False, "certification": False}
        self.reused_experiments = 0
        self._reused_keys: set[frozenset[str]] = set()
        self.cache: dict[frozenset[str], dict[str, Any]] = {}
        self.unavailable_reasons: list[str] = []
        self.touched_cardinalities: set[int] = set()
        for raw in existing_experiments or ():
            experiment = _normalise_probe(raw, universe) if exact else _normalise_experiment(raw, universe)
            key = frozenset(experiment["removed_source_ids"])
            self._insert(experiment, reused=False)

    def note_reuse(self, key: frozenset[str]) -> None:
        if key not in self._reused_keys:
            self._reused_keys.add(key)
            self.reused_experiments = len(self._reused_keys)

    @staticmethod
    def is_preserving(observation: dict[str, Any]) -> bool:
        return observation["within_tolerance"] is True

    @staticmethod
    def is_proofable(observation: dict[str, Any]) -> bool:
        return observation.get("proofable") is True

    def _insert(self, experiment: dict[str, Any], *, reused: bool) -> dict[str, Any]:
        key = frozenset(experiment["removed_source_ids"])
        old = self.cache.get(key)
        if old is not None:
            if _experiment_signature(old["experiment"]) != _experiment_signature(experiment):
                raise MinimalContextError(
                    f"conflicting direct evidence for removed_source_ids={list(experiment['removed_source_ids'])!r}"
                )
            if reused:
                self.note_reuse(key)
            return old
        if self.exact:
            if experiment["result_status"] == "unavailable":
                reason = experiment.get("result", {}).get("reason")
                self.unavailable_reasons.append(reason if isinstance(reason, str) and reason else "exact_probe_unavailable")
            preserves = is_reference_match_preserving({"status": experiment["result_status"]})
            observation = {
                "experiment": experiment,
                "within_tolerance": preserves,
                "proofable": experiment["result_status"] in {"matched", "diverged"},
            }
        else:
            observation = {
                "experiment": experiment,
                "within_tolerance": abs(experiment["delta_nats"]) <= self.tolerance_nats,
                "proofable": True,
            }
        self.cache[key] = observation
        return observation

    def get(self, retained: tuple[str, ...], phase: str) -> dict[str, Any] | None:
        retained_set = frozenset(retained)
        removed = tuple(source_id for source_id in self.universe if source_id not in retained_set)
        if not removed:
            return None
        cached = self.cache.get(frozenset(removed))
        if cached is not None:
            self.note_reuse(frozenset(removed))
            self.touched_cardinalities.add(len(retained))
            return cached
        if self.new_probes[phase] >= self.budgets[phase]:
            self.exhausted[phase] = True
            return None
        if not callable(self.measure_removed):
            # A batch-only caller is still allowed to use the scalar-shaped
            # search path.  Route this one arm through the same validation and
            # accounting seam rather than inventing a second direct-measurement
            # implementation.
            return self.get_many([retained], phase)[0]
        raw = self.measure_removed(removed)
        experiment = _normalise_probe(raw, self.universe) if self.exact else _normalise_experiment(raw, self.universe)
        actual_removed = tuple(experiment["removed_source_ids"])
        if actual_removed != removed:
            raise MinimalContextError(
                "direct measurement returned a different deletion set: "
                f"requested={list(removed)!r}, returned={list(actual_removed)!r}"
            )
        self.new_probes[phase] += 1
        observation = self._insert(experiment, reused=False)
        self.touched_cardinalities.add(len(retained))
        return observation

    def get_many(self, retained_sets: Sequence[tuple[str, ...]], phase: str) -> list[dict[str, Any] | None]:
        """Resolve a fixed set of independent arms through the optional batch seam.

        The returned list is aligned with ``retained_sets``.  Cache lookup,
        budget charging, deletion-set validation, and evidence insertion are
        identical to :meth:`get`; only the direct measurement dispatch is
        grouped.  A batch adapter is therefore an execution optimization,
        not a second proof path.
        """
        if isinstance(retained_sets, (str, bytes)) or not isinstance(retained_sets, Sequence):
            raise MinimalContextError(
                "batched direct evidence requires a bounded sequence of retained source sets"
            )
        requested = [tuple(retained) for retained in retained_sets]
        results: list[dict[str, Any] | None] = [None] * len(requested)
        pending: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []
        for index, retained in enumerate(requested):
            retained_set = frozenset(retained)
            removed = tuple(source_id for source_id in self.universe if source_id not in retained_set)
            if not removed:
                continue
            cached = self.cache.get(frozenset(removed))
            if cached is not None:
                self.note_reuse(frozenset(removed))
                self.touched_cardinalities.add(len(retained))
                results[index] = cached
                continue
            if self.new_probes[phase] + len(pending) >= self.budgets[phase]:
                self.exhausted[phase] = True
                # Continue scanning this bounded chunk so cached evidence later
                # in the chunk is still reused.  Only uncached arms are held
                # back by the remaining new-probe budget.
                continue
            pending.append((index, tuple(retained), removed))

        if not pending:
            return results
        if self.measure_removed_many is None:
            for index, retained, _removed in pending:
                results[index] = self.get(retained, phase)
            return results

        removed_sets = tuple(removed for _index, _retained, removed in pending)
        raw_results = self.measure_removed_many(removed_sets)
        try:
            measured = list(raw_results)
        except TypeError as exc:
            raise MinimalContextError("batch direct measurement must return an iterable") from exc
        if len(measured) != len(pending):
            raise MinimalContextError(
                "batch direct measurement returned a different number of arms: "
                f"requested={len(pending)}, returned={len(measured)}"
            )
        for (index, retained, removed), raw in zip(pending, measured):
            experiment = _normalise_probe(raw, self.universe) if self.exact else _normalise_experiment(raw, self.universe)
            actual_removed = tuple(experiment["removed_source_ids"])
            if actual_removed != removed:
                raise MinimalContextError(
                    "batch direct measurement returned a different deletion set: "
                    f"requested={list(removed)!r}, returned={list(actual_removed)!r}"
                )
            self.new_probes[phase] += 1
            observation = self._insert(experiment, reused=False)
            self.touched_cardinalities.add(len(retained))
            results[index] = observation
        return results

    def record(self, retained: tuple[str, ...], observation: dict[str, Any]) -> dict[str, Any]:
        experiment = observation["experiment"]
        if self.exact:
            return {
                "retained_source_ids": list(retained),
                "removed_source_ids": list(experiment["removed_source_ids"]),
                "retained_source_count": len(retained),
                "probe_id": experiment["probe_id"],
                "result_status": experiment["result_status"],
                "within_tolerance": bool(observation["within_tolerance"]),
                "provenance": "direct_generation_probe",
            }
        delta = experiment["delta_nats"]
        return {
            "retained_source_ids": list(retained),
            "removed_source_ids": list(experiment["removed_source_ids"]),
            "retained_source_count": len(retained),
            "experiment_id": experiment["experiment_id"],
            "delta_nats": delta,
            "absolute_difference_nats": abs(delta),
            "within_tolerance": bool(observation["within_tolerance"]),
            "provenance": "measured",
        }

    def coverage(self, candidate_count: int) -> dict[str, Any]:
        lower: list[dict[str, Any]] = []
        for cardinality in sorted(self.touched_cardinalities):
            count = math.comb(len(self.universe), cardinality)
            if cardinality >= candidate_count:
                continue
            tested_sets = [
                tuple(source_id for source_id in self.universe if source_id not in removed)
                for removed in self.cache
                if len(self.universe) - len(removed) == cardinality
            ]
            observations = [
                self.cache[frozenset(source_id for source_id in self.universe if source_id not in set(retained))]
                for retained in tested_sets
            ]
            preserving = sum(1 for observation in observations if observation["within_tolerance"])
            proofable = all(observation["proofable"] for observation in observations)
            lower.append(
                {
                    "retained_source_count": cardinality,
                    "candidate_count": count,
                    "tested_count": len(tested_sets),
                    "preserving_count": preserving,
                    "complete": len(tested_sets) == count and proofable,
                }
            )
        smaller_total = sum(math.comb(len(self.universe), r) for r in range(candidate_count))
        smaller_tested = sum(
            1
            for removed in self.cache
            if len(self.universe) - len(removed) < candidate_count
        )
        return {
            "lower_cardinalities": lower,
            "smaller_candidate_count": smaller_total,
            "smaller_tested_count": smaller_tested,
            "smaller_remaining_count": max(0, smaller_total - smaller_tested),
        }


def _seeded_order(universe: tuple[str, ...], seed: int) -> tuple[str, ...]:
    return tuple(
        source_id
        for _digest, _index, source_id in sorted(
            (
                hashlib.sha256(f"{seed}\0{index}\0{source_id}".encode("utf-8")).hexdigest(),
                index,
                source_id,
            )
            for index, source_id in enumerate(universe)
        )
    )


def _canonical_nominations(
    nominations: Iterable[Iterable[str]] | None, universe: tuple[str, ...]
) -> list[tuple[str, ...]]:
    result: dict[tuple[str, ...], None] = {}
    for nomination in nominations or ():
        retained = _canonical_set(nomination, universe, "candidate_retained_source_set", allow_empty=True)
        result[retained] = None
    return sorted(result, key=lambda item: (len(item), item))


def _certificate(
    candidate: dict[str, Any],
    kind: str,
    coverage: dict[str, Any],
    *,
    local_checks: list[dict[str, Any]],
    local_complete: bool,
) -> dict[str, Any]:
    result = {
        "kind": kind,
        "candidate_retained_source_ids": list(candidate["retained_source_ids"]),
        "candidate_retained_source_count": candidate["retained_source_count"],
        "global_minimality": "proven" if kind == "exact_minimum" else "not_proven",
        "inclusion_minimality": "proven" if kind in {"exact_minimum", "inclusion_minimum"} else "not_proven",
        "local_checks": deepcopy(local_checks),
        "local_checks_complete": local_complete,
        "smaller_candidate_count": coverage["smaller_candidate_count"],
        "smaller_tested_count": coverage["smaller_tested_count"],
        "smaller_remaining_count": coverage["smaller_remaining_count"],
    }
    if "probe_id" in candidate:
        result["candidate_probe_id"] = candidate["probe_id"]
    else:
        result["candidate_experiment_id"] = candidate["experiment_id"]
    return result


def _result_id(identity: Mapping[str, Any]) -> str:
    encoded = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return "mc_" + hashlib.sha256(encoded).hexdigest()[:24]


def _base_result(
    *,
    run_id: str,
    source_ids: tuple[str, ...],
    manifest: Mapping[str, Any] | None,
    tolerance_nats: float,
    search_budget: int,
    certification_budget: int,
    search_seed: int,
    study_id: str | None,
    engine: _EvidenceEngine,
    status: str,
    search_stopped_reason: str,
    preservation: Mapping[str, Any] | None = None,
    answer_preservation_study_id: str | None = None,
    search_universe_id: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "result_id": "",
        "status": status,
        "source_universe": {
            "context_units_schema": "clozn.context-units.v1",
            "source_ids": list(source_ids),
            "source_count": len(source_ids),
            "cost_kind": "retained_source_count",
            "context_units_manifest_sha256": _manifest_digest(manifest, source_ids),
        },
        "preservation": dict(preservation) if preservation is not None else {
            "kind": PRESERVATION_KIND,
            "target": PRESERVATION_TARGET,
            "tolerance_nats": tolerance_nats,
        },
        "budget": {
            "search_probe_budget": search_budget,
            "search_new_probes": engine.new_probes["search"],
            "search_remaining": search_budget - engine.new_probes["search"],
            "certification_probe_budget": certification_budget,
            "certification_new_probes": engine.new_probes["certification"],
            "certification_remaining": certification_budget - engine.new_probes["certification"],
            "reused_experiments": engine.reused_experiments,
            "total_new_probes": sum(engine.new_probes.values()),
            "baseline_passes": 0,
            "baseline_charged_as_deletion_probe": False,
        },
        "search": {
            "method": SEARCH_METHOD,
            "strategy": SEARCH_STRATEGY,
            "greedy_orders": ["source_order", "reverse_source_order"],
            "search_seed": search_seed,
            "stopped_reason": search_stopped_reason,
        },
    }
    if study_id is not None:
        result["context_dependence_study_id"] = study_id
    if answer_preservation_study_id is not None:
        result["answer_preservation_study_id"] = answer_preservation_study_id
    if search_universe_id is not None:
        result["source_universe"]["search_universe_id"] = search_universe_id
    return result


def _bind_result_id(
    result: dict[str, Any],
    *,
    engine: _EvidenceEngine,
    study_id: str | None,
    search_seed: int,
    search_budget: int,
    certification_budget: int,
    answer_preservation_study_id: str | None = None,
    search_universe_id: str | None = None,
) -> None:
    source_order = tuple(result["source_universe"]["source_ids"])
    evidence = []
    for removed, observation in sorted(
        engine.cache.items(),
        key=lambda item: tuple(source_id for source_id in source_order if source_id in item[0]),
    ):
        experiment = observation["experiment"]
        row = {
            "removed_source_ids": [source_id for source_id in source_order if source_id in removed],
            "provenance": experiment["provenance"],
        }
        if engine.exact:
            row.update({"probe_id": experiment["probe_id"], "result_status": experiment["result_status"]})
        else:
            row.update({"experiment_id": experiment["experiment_id"], "delta_nats": experiment["delta_nats"]})
        evidence.append(row)
    identity = {
        "run_id": result["run_id"],
        "context_units_manifest_sha256": result["source_universe"]["context_units_manifest_sha256"],
        "ordered_default_source_ids": result["source_universe"]["source_ids"],
        "context_dependence_study_id": study_id,
        "answer_preservation_study_id": answer_preservation_study_id,
        "search_universe_id": search_universe_id,
        "preservation": result["preservation"],
        "search_method": SEARCH_METHOD,
        "search_strategy": SEARCH_STRATEGY,
        "search_seed": search_seed,
        "search_probe_budget": search_budget,
        "certification_probe_budget": certification_budget,
        "status": result["status"],
        "candidate": result.get("candidate"),
        "certificate": result.get("certificate"),
        "coverage": result.get("coverage"),
        "evidence": evidence,
    }
    result["result_id"] = _result_id(identity)


def _unavailable_result(
    *,
    run_id: str,
    source_ids: tuple[str, ...],
    manifest: Mapping[str, Any] | None,
    tolerance_nats: float,
    search_budget: int,
    certification_budget: int,
    search_seed: int,
    study_id: str | None,
    error: str,
    preservation: Mapping[str, Any] | None = None,
    exact: bool = False,
    answer_preservation_study_id: str | None = None,
    search_universe_id: str | None = None,
) -> dict[str, Any]:
    engine = _EvidenceEngine(
        source_ids,
        lambda _removed: (_ for _ in ()).throw(MinimalContextUnavailable(error)),
        tolerance_nats,
        search_budget,
        certification_budget,
        (),
        exact=exact,
    )
    result = _base_result(
        run_id=run_id,
        source_ids=source_ids,
        manifest=manifest,
        tolerance_nats=tolerance_nats,
        search_budget=search_budget,
        certification_budget=certification_budget,
        search_seed=search_seed,
        study_id=study_id,
        engine=engine,
        status="unavailable",
        search_stopped_reason="unavailable",
        preservation=preservation,
        answer_preservation_study_id=answer_preservation_study_id,
        search_universe_id=search_universe_id,
    )
    result["error"] = error
    _bind_result_id(
        result,
        engine=engine,
        study_id=study_id,
        search_seed=search_seed,
        search_budget=search_budget,
        certification_budget=certification_budget,
        answer_preservation_study_id=answer_preservation_study_id,
        search_universe_id=search_universe_id,
    )
    _validate_result_shape(result)
    return result


def _validate_result_shape(result: Mapping[str, Any]) -> None:
    schemas.validate(dict(result), SCHEMA)
    status = result["status"]
    if status == "found":
        if "candidate" not in result or "certificate" not in result or "coverage" not in result:
            raise MinimalContextError("found minimal-context results require candidate, certificate, and coverage")
        source_ids = result["source_universe"]["source_ids"]
        candidate = result["candidate"]
        certificate = result["certificate"]
        coverage = result["coverage"]
        exact = result["preservation"]["kind"] == EXACT_PRESERVATION_KIND
        if result["source_universe"]["source_count"] != len(source_ids):
            raise MinimalContextError("source_universe.source_count does not match source_ids")
        if len(set(source_ids)) != len(source_ids):
            raise MinimalContextError("source_universe.source_ids contains duplicates")
        if candidate["retained_source_count"] != len(candidate["retained_source_ids"]):
            raise MinimalContextError("candidate retained_source_count does not match retained_source_ids")
        if len(set(candidate["retained_source_ids"])) != len(candidate["retained_source_ids"]):
            raise MinimalContextError("candidate retained_source_ids contains duplicates")
        if len(set(candidate["removed_source_ids"])) != len(candidate["removed_source_ids"]):
            raise MinimalContextError("candidate removed_source_ids contains duplicates")
        if candidate["retained_source_count"] + len(candidate["removed_source_ids"]) != len(source_ids):
            raise MinimalContextError("candidate retained and removed sets do not cover the universe")
        if set(candidate["retained_source_ids"]).intersection(candidate["removed_source_ids"]):
            raise MinimalContextError("candidate retained and removed sets overlap")
        if set(candidate["retained_source_ids"]).union(candidate["removed_source_ids"]) != set(source_ids):
            raise MinimalContextError("candidate retained and removed sets are not the source universe")
        if exact:
            if candidate.get("result_status") != "matched" or candidate.get("provenance") != "direct_generation_probe":
                raise MinimalContextError("an exact candidate must be a directly matched generation probe")
            if "probe_id" not in candidate:
                raise MinimalContextError("an exact candidate must carry its direct probe ID")
        else:
            if candidate["absolute_difference_nats"] != abs(candidate["delta_nats"]):
                raise MinimalContextError("candidate absolute_difference_nats is not derived from delta_nats")
            _finite_number(candidate["delta_nats"], "candidate.delta_nats")
        if candidate["within_tolerance"] is not True:
            raise MinimalContextError("a found candidate must be directly preserving")
        candidate_evidence_id = candidate["probe_id"] if exact else candidate["experiment_id"]
        certificate_evidence_id = certificate.get("candidate_probe_id") if exact else certificate.get("candidate_experiment_id")
        if certificate_evidence_id != candidate_evidence_id:
            raise MinimalContextError("certificate is not bound to the direct candidate evidence")
        if certificate["candidate_retained_source_ids"] != candidate["retained_source_ids"]:
            raise MinimalContextError("certificate is not bound to the candidate retained set")
        if certificate["candidate_retained_source_count"] != candidate["retained_source_count"]:
            raise MinimalContextError("certificate candidate count does not match the candidate")
        local_checks = certificate["local_checks"]
        if certificate["local_checks_complete"] and len(local_checks) != candidate["retained_source_count"]:
            raise MinimalContextError("complete local proof does not cover every retained source")
        for check in local_checks:
            if exact:
                if check.get("result_status") != "diverged" or check.get("provenance") != "direct_generation_probe":
                    raise MinimalContextError("an exact local proof must be a directly diverged generation probe")
            else:
                if check["within_tolerance"] is not False or check["absolute_difference_nats"] != abs(check["delta_nats"]):
                    raise MinimalContextError("a local minimality check must be a directly measured failure")
                _finite_number(check["delta_nats"], "certificate.local_check.delta_nats")
        kind = certificate["kind"]
        if kind == "exact_minimum":
            if certificate["global_minimality"] != "proven" or certificate["smaller_remaining_count"] != 0:
                raise MinimalContextError("exact_minimum lacks complete lower-cardinality proof")
            rows = {
                row["retained_source_count"]: row for row in coverage["lower_cardinalities"]
            }
            for cardinality in range(candidate["retained_source_count"]):
                row = rows.get(cardinality)
                if row is None or not row["complete"] or row["preserving_count"] != 0:
                    raise MinimalContextError("exact_minimum is missing a complete failing lower cardinality")
            for row in coverage["lower_cardinalities"]:
                if row["retained_source_count"] < candidate["retained_source_count"] and (
                    not row["complete"] or row["preserving_count"] != 0
                ):
                    raise MinimalContextError("exact_minimum has an incomplete or preserving lower cardinality")
        elif kind == "inclusion_minimum":
            if certificate["inclusion_minimality"] != "proven" or not certificate["local_checks_complete"]:
                raise MinimalContextError("inclusion_minimum lacks complete local proof")
        elif kind == "best_verified":
            if certificate["global_minimality"] == "proven" or certificate["inclusion_minimality"] == "proven":
                raise MinimalContextError("best_verified cannot claim minimality")
    elif any(key in result for key in ("candidate", "certificate", "coverage")):
        raise MinimalContextError(f"{status} minimal-context results cannot carry a candidate certificate")


def run_minimal_context_search(
    source_ids: Iterable[str],
    measure_removed: Callable[[tuple[str, ...]], Any] | None,
    *,
    tolerance_nats: float,
    search_probe_budget: int,
    certification_probe_budget: int,
    existing_experiments: Iterable[Any] | None = None,
    candidate_retained_source_sets: Iterable[Iterable[str]] | None = None,
    search_seed: int = 0,
    run_id: str = "synthetic",
    context_unit_manifest: Mapping[str, Any] | None = None,
    context_dependence_study_id: str | None = None,
    preservation: Mapping[str, Any] | None = None,
    answer_preservation_study_id: str | None = None,
    search_universe_id: str | None = None,
    phase_callback: Callable[[str, int, int], Any] | None = None,
    measure_removed_many: Callable[[tuple[tuple[str, ...], ...]], Iterable[Any]] | None = None,
) -> dict[str, Any]:
    """Run bounded search and derive the strongest certificate supported by evidence.

    ``measure_removed`` is the only experiment adapter.  It must return one
    direct measured experiment for exactly the requested non-empty deletion
    set.  Existing experiments are accepted as reusable cache entries.
    """
    universe = _ordered_ids(source_ids)
    if not callable(measure_removed) and not callable(measure_removed_many):
        raise MinimalContextError("measure_removed or measure_removed_many must be callable")
    tolerance = _finite_number(tolerance_nats, "tolerance_nats")
    if tolerance < 0:
        raise MinimalContextError("tolerance_nats must be non-negative")
    search_budget = _nonnegative_int(search_probe_budget, "search_probe_budget")
    certification_budget = _nonnegative_int(certification_probe_budget, "certification_probe_budget")
    seed = _nonnegative_int(search_seed, "search_seed")
    if not isinstance(run_id, str) or not run_id:
        raise MinimalContextError("run_id must be a non-empty string")
    if context_dependence_study_id is not None and (
        not isinstance(context_dependence_study_id, str) or not context_dependence_study_id
    ):
        raise MinimalContextError("context_dependence_study_id must be a non-empty string when supplied")
    exact = isinstance(preservation, Mapping) and preservation.get("kind") == EXACT_PRESERVATION_KIND
    def phase(name: str, completed: int, total: int) -> None:
        if callable(phase_callback):
            phase_callback(name, completed, total)
    if exact and answer_preservation_study_id is None and isinstance(context_dependence_study_id, str) \
            and context_dependence_study_id.startswith("aps_"):
        answer_preservation_study_id = context_dependence_study_id
        context_dependence_study_id = None
    if preservation is not None:
        if not isinstance(preservation, Mapping) or preservation.get("target") != PRESERVATION_TARGET:
            raise MinimalContextError("preservation must name a supported whole-continuation contract")
        preservation = dict(preservation)
        if exact:
            preservation.pop("tolerance_nats", None)
        elif preservation.get("kind") != PRESERVATION_KIND:
            raise MinimalContextError("unsupported preservation kind")
    nominations = _canonical_nominations(candidate_retained_source_sets, universe)

    phase("searching", 0, search_budget)

    engine = _EvidenceEngine(
        universe,
        measure_removed,
        tolerance,
        search_budget,
        certification_budget,
        existing_experiments,
        exact=exact,
        measure_removed_many=measure_removed_many,
    )
    preserving: dict[tuple[str, ...], dict[str, Any]] = {}

    def offer(retained: tuple[str, ...], observation: dict[str, Any]) -> None:
        if engine.is_preserving(observation):
            preserving[retained] = observation

    # Existing direct evidence is useful even when the new search budget is 0.
    for removed, observation in engine.cache.items():
        retained = tuple(source_id for source_id in universe if source_id not in removed)
        if engine.is_preserving(observation) and retained != universe:
            engine.note_reuse(removed)
            offer(retained, observation)

    orders: list[tuple[str, ...]] = [universe, tuple(reversed(universe))]
    if seed:
        orders.append(_seeded_order(universe, seed))
    search_exhausted = False
    preserving_greedy_candidates: list[tuple[str, ...]] = []
    for order in orders:
        current = universe
        current_observation: dict[str, Any] | None = None
        made_progress = True
        while made_progress:
            made_progress = False
            for source_id in order:
                if source_id not in current:
                    continue
                retained = tuple(item for item in current if item != source_id)
                observation = engine.get(retained, "search")
                if observation is None:
                    search_exhausted = engine.exhausted["search"]
                    break
                if engine.is_preserving(observation):
                    current = retained
                    current_observation = observation
                    offer(current, observation)
                    made_progress = True
            if search_exhausted:
                break
        if search_exhausted:
            break
        if current_observation is not None:
            offer(current, current_observation)
            preserving_greedy_candidates.append(current)
    if not search_exhausted:
        # The intersection is only a nomination. It still goes through the
        # direct experiment adapter below; set arithmetic is never evidence.
        if len(preserving_greedy_candidates) >= 2:
            first, second = preserving_greedy_candidates[0], preserving_greedy_candidates[1]
            nominations = list(nominations) + [
                tuple(source_id for source_id in universe
                      if source_id in set(first) and source_id in set(second))
            ]
        for retained in _canonical_nominations(nominations, universe):
            observation = engine.get(retained, "search")
            if observation is None:
                search_exhausted = engine.exhausted["search"]
                break
            if engine.is_preserving(observation):
                offer(retained, observation)

    if not preserving:
        if exact and engine.unavailable_reasons:
            result = _base_result(
                run_id=run_id,
                source_ids=universe,
                manifest=context_unit_manifest,
                tolerance_nats=tolerance,
                search_budget=search_budget,
                certification_budget=certification_budget,
                search_seed=seed,
                study_id=context_dependence_study_id,
                engine=engine,
                status="unavailable",
                search_stopped_reason="exact_probe_unavailable",
                preservation=preservation,
                answer_preservation_study_id=answer_preservation_study_id,
                search_universe_id=search_universe_id,
            )
            result["error"] = engine.unavailable_reasons[0]
            _bind_result_id(
                result,
                engine=engine,
                study_id=context_dependence_study_id,
                search_seed=seed,
                search_budget=search_budget,
                certification_budget=certification_budget,
                answer_preservation_study_id=answer_preservation_study_id,
                search_universe_id=search_universe_id,
            )
            _validate_result_shape(result)
            return result
        result = _base_result(
            run_id=run_id,
            source_ids=universe,
            manifest=context_unit_manifest,
            tolerance_nats=tolerance,
            search_budget=search_budget,
            certification_budget=certification_budget,
            search_seed=seed,
            study_id=context_dependence_study_id,
            engine=engine,
            status="not_found_within_budget",
            search_stopped_reason="search_probe_budget_exhausted" if search_exhausted else "search_complete",
            preservation=preservation,
            answer_preservation_study_id=answer_preservation_study_id,
            search_universe_id=search_universe_id,
        )
        _bind_result_id(
            result,
            engine=engine,
            study_id=context_dependence_study_id,
            search_seed=seed,
            search_budget=search_budget,
            certification_budget=certification_budget,
            answer_preservation_study_id=answer_preservation_study_id,
            search_universe_id=search_universe_id,
        )
        _validate_result_shape(result)
        return result

    retained = min(preserving, key=lambda item: (len(item), item))
    observation = preserving[retained]
    local_proven = False
    exact_proven = False
    certification_stopped = False
    local_checks: list[dict[str, Any]] = []

    phase("verifying_candidate", 0, certification_budget)

    # Direct inclusion checks are explicit.  A preserving child can replace a
    # nominated/search candidate, but no monotonic inference is made.
    while True:
        children: list[tuple[tuple[str, ...], dict[str, Any]]] = []
        checks: list[dict[str, Any]] = []
        local_complete = True
        child_sets = [tuple(item for item in retained if item != source_id) for source_id in retained]
        child_observations = engine.get_many(child_sets, "certification")
        for child, child_observation in zip(child_sets, child_observations):
            if child_observation is None:
                local_complete = False
                certification_stopped = engine.exhausted["certification"]
                break
            checks.append(engine.record(child, child_observation))
            if not engine.is_proofable(child_observation):
                local_complete = False
                certification_stopped = True
                break
            if engine.is_preserving(child_observation):
                children.append((child, child_observation))
        local_checks = checks
        if not local_complete:
            break
        if not children:
            local_proven = True
            phase("inclusion_minimal", engine.new_probes["certification"], certification_budget)
            break
        retained, observation = min(children, key=lambda item: (len(item[0]), item[0]))
        offer(retained, observation)

    if local_proven:
        # Enumerate lower cardinalities lazily.  The first preserving subset
        # is exact once every smaller cardinality has already completed with no
        # preserving result; alternatives of the same cardinality are not needed.
        for cardinality in range(len(retained)):
            phase(
                f"certifying_cardinality_{cardinality}",
                engine.new_probes["certification"],
                certification_budget,
            )
            engine.touched_cardinalities.add(cardinality)
            cardinality_complete = True
            found_smaller: tuple[tuple[str, ...], dict[str, Any]] | None = None
            candidates = combinations(universe, cardinality)
            layer_complete = False
            while True:
                remaining_budget = certification_budget - engine.new_probes["certification"]
                chunk_size = (
                    min(DEFAULT_CERTIFICATION_BATCH_SIZE, remaining_budget)
                    if remaining_budget > 0
                    else DEFAULT_CERTIFICATION_BATCH_SIZE
                )
                chunk = tuple(islice(candidates, chunk_size))
                if not chunk:
                    layer_complete = True
                    break
                candidate_observations = engine.get_many(chunk, "certification")
                chunk_missing = False
                for candidate, candidate_observation in zip(chunk, candidate_observations):
                    if candidate_observation is None:
                        chunk_missing = True
                        certification_stopped = engine.exhausted["certification"]
                        continue
                    if not engine.is_proofable(candidate_observation):
                        cardinality_complete = False
                        certification_stopped = True
                        break
                    if engine.is_preserving(candidate_observation):
                        found_smaller = (candidate, candidate_observation)
                        break
                if found_smaller is not None or not cardinality_complete:
                    break
                if chunk_missing:
                    cardinality_complete = False
                    break
                if len(chunk) < chunk_size:
                    layer_complete = True
                    break
            if found_smaller is None and (not cardinality_complete or not layer_complete):
                cardinality_complete = False
            if not cardinality_complete:
                break
            if found_smaller is not None:
                retained, observation = found_smaller
                offer(retained, observation)
                exact_proven = True
                break
        else:
            exact_proven = True
        if exact_proven:
            phase("exact_certificate", engine.new_probes["certification"], certification_budget)

    if exact_proven:
        # Certification may have replaced the search candidate with a lower
        # cardinality subset.  Rebind the persisted local proof rows to that
        # final candidate using the already-complete lower-cardinality cache.
        final_checks: list[dict[str, Any]] = []
        final_child_sets = [tuple(item for item in retained if item != source_id) for source_id in retained]
        final_child_observations = engine.get_many(final_child_sets, "certification")
        for child, child_observation in zip(final_child_sets, final_child_observations):
            if child_observation is None or not engine.is_proofable(child_observation) or engine.is_preserving(child_observation):
                raise MinimalContextError("exact certification lost its final candidate local proof")
            final_checks.append(engine.record(child, child_observation))
        local_checks = final_checks
        local_proven = True

    candidate = engine.record(retained, observation)
    coverage = engine.coverage(len(retained))
    if exact_proven:
        kind = "exact_minimum"
    elif local_proven:
        kind = "inclusion_minimum"
    else:
        kind = "best_verified"
    certificate = _certificate(
        candidate,
        kind,
        coverage,
        local_checks=local_checks,
        local_complete=local_proven,
    )
    result = _base_result(
        run_id=run_id,
        source_ids=universe,
        manifest=context_unit_manifest,
        tolerance_nats=tolerance,
        search_budget=search_budget,
        certification_budget=certification_budget,
        search_seed=seed,
        study_id=context_dependence_study_id,
        engine=engine,
        status="found",
        search_stopped_reason="search_probe_budget_exhausted" if search_exhausted else "search_complete",
        preservation=preservation,
        answer_preservation_study_id=answer_preservation_study_id,
        search_universe_id=search_universe_id,
    )
    result["candidate"] = candidate
    result["certificate"] = certificate
    result["coverage"] = coverage
    result["search"]["certification_lower_cardinality_candidate_count"] = coverage[
        "smaller_candidate_count"
    ]
    result["budget"]["certification_stopped_reason"] = (
        "certification_probe_budget_exhausted" if certification_stopped else "certification_complete"
    )
    _bind_result_id(
        result,
        engine=engine,
        study_id=context_dependence_study_id,
        search_seed=seed,
        search_budget=search_budget,
        certification_budget=certification_budget,
        answer_preservation_study_id=answer_preservation_study_id,
        search_universe_id=search_universe_id,
    )
    _validate_result_shape(result)
    return result


def validate_minimal_context_result(result: Mapping[str, Any]) -> None:
    """Validate the versioned result and its structural status invariants."""
    _validate_result_shape(result)


def run_minimal_context_for_study(
    run: Mapping[str, Any],
    context_unit_manifest: Mapping[str, Any],
    measurement_study: Any,
    *,
    tolerance_nats: float,
    search_probe_budget: int,
    certification_probe_budget: int,
    search_seed: int = 0,
    candidate_retained_source_sets: Iterable[Iterable[str]] | None = None,
    preservation: Mapping[str, Any] | None = None,
    exact_answer_study: Any | None = None,
) -> dict[str, Any]:
    """Run bounded minimal-context analysis against a live study."""
    if not isinstance(run, Mapping):
        raise MinimalContextError("run must be a mapping")
    if not isinstance(context_unit_manifest, Mapping):
        raise MinimalContextError("context_unit_manifest must be a mapping")
    run_id = run.get("id")
    if not isinstance(run_id, str) or not run_id:
        raise MinimalContextError("run must carry a non-empty id")
    if context_unit_manifest.get("schema_version") != "clozn.context-units.v1":
        raise MinimalContextError("context_unit_manifest has the wrong schema_version")
    if context_unit_manifest.get("run_id") != run_id:
        raise MinimalContextError("context_unit_manifest belongs to a different run")
    schemas.validate(dict(context_unit_manifest), "clozn.context-units.v1")
    default_raw = context_unit_manifest.get("default_source_ids")
    if not isinstance(default_raw, list):
        raise MinimalContextError("context_unit_manifest.default_source_ids must be a list")
    stored_manifest = run.get("context_units")
    if isinstance(stored_manifest, Mapping) and _manifest_digest(stored_manifest, ()) != _manifest_digest(
        context_unit_manifest, ()
    ):
        raise MinimalContextError("context_unit_manifest does not match the run's stored manifest")

    if default_raw:
        universe = _ordered_ids(default_raw, "default_source_ids")
    else:
        universe = ()
    if not universe:
        return _unavailable_result(
            run_id=run_id,
            source_ids=universe,
            manifest=context_unit_manifest,
            tolerance_nats=_finite_number(tolerance_nats, "tolerance_nats"),
            search_budget=_nonnegative_int(search_probe_budget, "search_probe_budget"),
            certification_budget=_nonnegative_int(certification_probe_budget, "certification_probe_budget"),
            search_seed=_nonnegative_int(search_seed, "search_seed"),
            study_id=None,
            error="the Context Units manifest has no default source IDs",
        )

    exact = isinstance(preservation, Mapping) and preservation.get("kind") == EXACT_PRESERVATION_KIND
    if exact:
        if exact_answer_study is None:
            raise MinimalContextError("exact_recorded_output requires an ExactAnswerPreservationStudy")
        study_source_ids = _ordered_ids(exact_answer_study.source_ids, "exact_answer_study.source_ids")
        if study_source_ids != universe:
            raise MinimalContextError("exact preservation study source universe does not match Context Units")
        document = exact_answer_study.document()
        study_id = document.get("study_id") if isinstance(document, Mapping) else None
        if not isinstance(study_id, str) or not study_id.startswith("aps_"):
            raise MinimalContextError("exact preservation study has no stable study_id")
        control = document.get("unchanged_control") if isinstance(document, Mapping) else None
        if not isinstance(control, Mapping) or control.get("status") != "matched":
            return _unavailable_result(
                run_id=run_id,
                source_ids=universe,
                manifest=context_unit_manifest,
                tolerance_nats=0.0,
                search_budget=_nonnegative_int(search_probe_budget, "search_probe_budget"),
                certification_budget=_nonnegative_int(certification_probe_budget, "certification_probe_budget"),
                search_seed=_nonnegative_int(search_seed, "search_seed"),
                study_id=None,
                error=(control.get("reason") if isinstance(control, Mapping) else "unchanged_control_unavailable"),
                preservation={"kind": EXACT_PRESERVATION_KIND, "target": PRESERVATION_TARGET},
                exact=True,
                answer_preservation_study_id=study_id,
            )
        existing = document.get("probes")
        if not isinstance(existing, list):
            raise MinimalContextError("exact preservation study has no probes list")
        initial_probe_ids = {
            probe.get("probe_id") for probe in existing if isinstance(probe, Mapping)
        }
        result = run_minimal_context_search(
            universe,
            exact_answer_study.probe_removed_sources,
            tolerance_nats=0.0,
            search_probe_budget=search_probe_budget,
            certification_probe_budget=certification_probe_budget,
            existing_experiments=existing,
            candidate_retained_source_sets=candidate_retained_source_sets,
            search_seed=search_seed,
            run_id=run_id,
            context_unit_manifest=context_unit_manifest,
            preservation={"kind": EXACT_PRESERVATION_KIND, "target": PRESERVATION_TARGET},
            answer_preservation_study_id=study_id,
        )
        final_document = exact_answer_study.document()
        if final_document.get("study_id") != study_id:
            raise MinimalContextError("exact preservation study identity drifted during search")
        final_probes = final_document.get("probes")
        if not isinstance(final_probes, list):
            raise MinimalContextError("exact preservation study lost its probes list")
        final_probe_ids = {
            probe.get("probe_id") for probe in final_probes if isinstance(probe, Mapping)
        }
        expected_new = result["budget"]["total_new_probes"]
        if len(final_probe_ids - initial_probe_ids) != expected_new:
            raise MinimalContextError("exact preservation study probe accounting drifted")
        result["budget"]["baseline_passes"] = 1
        result["budget"]["baseline_charged_as_deletion_probe"] = False
        schemas.validate(result, SCHEMA)
        return result

    study_source_ids = _ordered_ids(measurement_study.source_ids, "measurement_study.source_ids")
    missing = [source_id for source_id in universe if source_id not in study_source_ids]
    if missing:
        raise MinimalContextError(
            "context unit default_source_ids are not all in measurement_study.source_ids: "
            f"{missing!r}"
        )

    from clozn.receipts.context_dependence import ContextDependenceError

    initial_study_id: str | None = None
    try:
        initial_document = measurement_study.document()
        if not isinstance(initial_document, Mapping) or not initial_document.get("baseline"):
            raise MinimalContextError("measurement study has no baseline evidence")
        initial_study_id = initial_document.get("study_id")
        if not isinstance(initial_study_id, str) or not initial_study_id:
            raise MinimalContextError("measurement study document has no study_id")
        existing = initial_document.get("experiments")
        if not isinstance(existing, list):
            raise MinimalContextError("measurement study document has no experiments list")
        initial_experiment_ids = {
            experiment.get("experiment_id")
            for experiment in existing
            if isinstance(experiment, Mapping)
        }
        initial_controls = initial_document.get("robustness_controls")
        initial_control_ids = {
            control.get("control_id")
            for control in initial_controls or ()
            if isinstance(control, Mapping)
        }
        initial_experiment_count = len(existing)
        initial_passes_consumed = (
            initial_document.get("budget", {}).get("passes_consumed")
            if isinstance(initial_document.get("budget"), Mapping)
            else None
        )
        result = run_minimal_context_search(
            universe,
            measurement_study.measure_removal_effect,
            tolerance_nats=tolerance_nats,
            search_probe_budget=search_probe_budget,
            certification_probe_budget=certification_probe_budget,
            existing_experiments=existing,
            candidate_retained_source_sets=candidate_retained_source_sets,
            search_seed=search_seed,
            run_id=run_id,
            context_unit_manifest=context_unit_manifest,
            context_dependence_study_id=initial_study_id,
        )
        final_document = measurement_study.document()
        if final_document.get("study_id") != initial_study_id:
            raise MinimalContextError("measurement study identity drifted during minimal-context search")
        final_experiments = final_document.get("experiments")
        if not isinstance(final_experiments, list):
            raise MinimalContextError("measurement study lost its experiments list")
        final_experiment_ids = {
            experiment.get("experiment_id")
            for experiment in final_experiments
            if isinstance(experiment, Mapping)
        }
        final_controls = final_document.get("robustness_controls")
        final_control_ids = {
            control.get("control_id")
            for control in final_controls or ()
            if isinstance(control, Mapping)
        }
        expected_new = result["budget"]["total_new_probes"]
        actual_new = len(final_experiment_ids - initial_experiment_ids)
        final_passes_consumed = (
            final_document.get("budget", {}).get("passes_consumed")
            if isinstance(final_document.get("budget"), Mapping)
            else None
        )
        if (
            actual_new != expected_new
            or len(final_experiments) - initial_experiment_count != expected_new
            or final_control_ids != initial_control_ids
            or (
                isinstance(initial_passes_consumed, int)
                and isinstance(final_passes_consumed, int)
                and final_passes_consumed - initial_passes_consumed != expected_new
            )
        ):
            raise MinimalContextError("measurement study pass/experiment accounting drifted")
        if result.get("context_dependence_study_id") != final_document.get("study_id"):
            raise MinimalContextError("minimal-context result is not bound to the final study identity")
        result["budget"]["baseline_passes"] = 1
        result["budget"]["baseline_charged_as_deletion_probe"] = False
        schemas.validate(result, SCHEMA)
        return result
    except (ContextDependenceError, MinimalContextUnavailable) as exc:
        return _unavailable_result(
            run_id=run_id,
            source_ids=universe,
            manifest=context_unit_manifest,
            tolerance_nats=_finite_number(tolerance_nats, "tolerance_nats"),
            search_budget=_nonnegative_int(search_probe_budget, "search_probe_budget"),
            certification_budget=_nonnegative_int(certification_probe_budget, "certification_probe_budget"),
            search_seed=_nonnegative_int(search_seed, "search_seed"),
            study_id=initial_study_id,
            error=str(exc),
        )
