"""Pure linked Context ⇄ Answer projections over durable Q2 evidence.

This module is deliberately read-side only. It consumes a model-free
ContextEffectsPlan, a recorded Run/Context Receipt, and the durable
ExperimentView/ObservationStore representation. It never selects a worker,
starts a job, or scores a continuation.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
from typing import Any

from clozn.replay.span_bridge import (
    ContextReceiptSourceResolutionError,
    resolve_context_receipt_source_set,
)

from .evaluators import ScoreRecordedContinuation
from .interventions import DeleteSource
from .observations import ObservationIntegrityError, TokenScoreDelta, TokenScoreObservation
from .persistence import (
    ExperimentArmView,
    ExperimentPersistenceError,
    ExperimentView,
    ObservationNotFound,
    ObservationStore,
)
from .selections import (
    AnswerSelection,
    AnswerSelectionUnavailable,
    ResolvedAnswerSelection,
    resolve_answer_selection_from_observation,
)
from .state import ExecutionState, canonical_json


SCHEMA_VERSION = "clozn.context-investigation-reader.v1"
QUERY_SCHEMA_VERSION = "clozn.context-investigation-query.v1"
DISPLAY_COORDINATE_BASIS = "context_display_unicode.v1"
DISPLAY_SEPARATOR = "\n\n"
LOCUS_PROJECTION_VERSION = "context_investigation_loci.v1"
DEFAULT_MEASUREMENT_FLOOR_NATS = 0.1


class ContextInvestigationError(ValueError):
    """The linked read model cannot be composed faithfully."""


class ContextInvestigationUnavailable(ContextInvestigationError):
    """Required recorded context or answer evidence is unavailable."""


class ContextInvestigationStale(ContextInvestigationError):
    """Persisted Q2 evidence no longer binds the current Run."""


class AnswerSelectionProjectionUnavailable(ContextInvestigationError):
    """An answer selection cannot be mapped to the recorded token evidence."""


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _floor(value: Any) -> float:
    if value is None:
        return DEFAULT_MEASUREMENT_FLOOR_NATS
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ContextInvestigationError("measurement_floor_nats must be a finite non-negative number")
    value = float(value)
    if value < 0:
        raise ContextInvestigationError("measurement_floor_nats must be non-negative")
    return value


def _plan_field(plan: Any, name: str, default: Any = None) -> Any:
    if isinstance(plan, Mapping):
        return plan.get(name, default)
    return getattr(plan, name, default)


def _plan_experiment(plan: Any):
    experiment = _plan_field(plan, "experiment")
    if experiment is None:
        raise ContextInvestigationError("Context Effects plan has no Experiment")
    return experiment


def _plan_ids(plan: Any, name: str) -> tuple[str, ...]:
    values = _plan_field(plan, name, ())
    if not isinstance(values, (list, tuple)) or any(not isinstance(item, str) or not item for item in values):
        raise ContextInvestigationError(f"Context Effects plan has malformed {name}")
    return tuple(values)


def _receipt_context(run: Mapping[str, Any], plan: Any) -> dict[str, Any]:
    display_ids = _plan_ids(plan, "display_source_ids")
    if not display_ids:
        raise ContextInvestigationUnavailable(
            "the recorded Run has no current canonical Context Receipt source catalogue"
        )
    try:
        resolved = resolve_context_receipt_source_set(dict(run), display_ids)
    except (ContextReceiptSourceResolutionError, TypeError, ValueError) as exc:
        raise ContextInvestigationUnavailable(str(exc)) from exc
    basis_messages = resolved.get("basis_messages")
    catalog = resolved.get("sources")
    if not isinstance(basis_messages, list) or not isinstance(catalog, list):
        raise ContextInvestigationUnavailable("the Context Receipt resolver returned incomplete context evidence")
    by_id = {
        item.get("source_id"): item for item in catalog
        if isinstance(item, Mapping) and isinstance(item.get("source_id"), str)
    }
    missing = [source_id for source_id in display_ids if source_id not in by_id]
    if missing:
        raise ContextInvestigationStale(
            "the current Context Receipt no longer contains display source(s): " + ", ".join(missing)
        )

    roots = {
        item.get("message_index"): item for item in catalog
        if item.get("source_kind") == "whole_message"
    }
    blocks: list[dict[str, Any]] = []
    source_display_ranges: dict[str, list[int]] = {}
    cursor = 0
    for message_index, message in enumerate(basis_messages):
        if not isinstance(message, Mapping) or not isinstance(message.get("role"), str) or not isinstance(message.get("content"), str):
            raise ContextInvestigationUnavailable("the Context Receipt basis contains malformed message content")
        if message_index:
            cursor += len(DISPLAY_SEPARATOR)
        text = message["content"]
        block_start = cursor
        block_end = block_start + len(text)
        root = roots.get(message_index)
        segment_id = root.get("segment_id") if isinstance(root, Mapping) else None
        if not isinstance(segment_id, str) or not segment_id:
            raise ContextInvestigationStale("the Context Receipt has no verified segment for a display message")
        message_sources = [
            item for item in catalog
            if isinstance(item, Mapping) and item.get("message_index") == message_index
        ]
        source_ids = [str(item["source_id"]) for item in message_sources]
        blocks.append({
            "message_index": message_index,
            "role": message["role"],
            "text": text,
            "segment_id": segment_id,
            "display_unicode_range": [block_start, block_end],
            "source_ids": source_ids,
        })
        for item in message_sources:
            start, end = item["unicode_range"]
            source_display_ranges[str(item["source_id"])] = [block_start + start, block_start + end]
        cursor = block_end
    return {
        "coordinate_basis": DISPLAY_COORDINATE_BASIS,
        "separator": DISPLAY_SEPARATOR,
        "blocks": blocks,
        "catalog": catalog,
        "source_display_ranges": source_display_ranges,
    }


def _validate_answer_tokens(*, response: str, token_ids: list[int], token_pieces: list[str],
                            token_spans: list[tuple[int, int]], token_logprobs: list[float] | None = None) -> None:
    if not isinstance(response, str):
        raise ContextInvestigationUnavailable("the recorded answer text is unavailable")
    if not (len(token_ids) == len(token_pieces) == len(token_spans)):
        raise ContextInvestigationStale("recorded answer token evidence has inconsistent lengths")
    if token_logprobs is not None and len(token_logprobs) != len(token_ids):
        raise ContextInvestigationStale("baseline token score evidence has an incomplete token vector")
    if "".join(token_pieces) != response:
        raise ContextInvestigationStale("recorded answer token pieces do not reconstruct the current response")
    cursor = 0
    for index, (piece, span) in enumerate(zip(token_pieces, token_spans)):
        if not isinstance(piece, str) or not isinstance(span, (list, tuple)) or len(span) != 2:
            raise ContextInvestigationStale(f"recorded answer token {index} is malformed")
        start, end = span
        if start != cursor or end != cursor + len(piece) or start < 0 or end < start:
            raise ContextInvestigationStale("recorded answer token spans do not agree with response text")
        cursor = end
    if cursor != len(response):
        raise ContextInvestigationStale("recorded answer token spans do not cover response text")


def _trace_answer(run: Mapping[str, Any]) -> dict[str, Any]:
    response = run.get("response")
    if not isinstance(response, str):
        raise ContextInvestigationUnavailable("the recorded answer text is unavailable")
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
    raw_steps = trace.get("steps")
    records: list[Mapping[str, Any]] = []
    if isinstance(raw_steps, list) and raw_steps:
        records = [item for item in raw_steps if isinstance(item, Mapping)]
        if len(records) != len(raw_steps):
            raise ContextInvestigationStale("the recorded answer token trace is malformed")
    elif isinstance(trace.get("tokens"), list) and isinstance(trace.get("token_ids"), list):
        if len(trace["tokens"]) != len(trace["token_ids"]):
            raise ContextInvestigationStale("the recorded answer token trace has inconsistent lengths")
        records = [
            {"piece": piece, "token_id": token_id}
            for piece, token_id in zip(trace["tokens"], trace["token_ids"])
        ]
    if not records:
        raise ContextInvestigationUnavailable("the recorded answer token trace is unavailable")
    pieces = [item.get("piece") for item in records]
    ids = [item.get("token_id", item.get("id")) for item in records]
    if any(not isinstance(piece, str) for piece in pieces) or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) for token_id in ids
    ):
        raise ContextInvestigationStale("the recorded answer token trace is malformed")
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        spans.append((cursor, cursor + len(piece)))
        cursor += len(piece)
    _validate_answer_tokens(response=response, token_ids=ids, token_pieces=pieces, token_spans=spans)
    return {
        "response": response,
        "token_ids": ids,
        "token_pieces": pieces,
        "token_spans": spans,
        "token_logprobs": None,
        "observation_id": None,
        "fidelity": "recorded_run_trace",
    }


def _answer_from_baseline(run: Mapping[str, Any], baseline: TokenScoreObservation | None,
                          *, execution_fingerprint: str) -> dict[str, Any]:
    if baseline is None or not baseline.completed:
        return _trace_answer(run)
    if baseline.run_id != run.get("id") or baseline.base_execution_fingerprint != execution_fingerprint:
        raise ContextInvestigationStale("baseline score observation belongs to a different Run execution")
    response = run.get("response")
    token_ids = list(baseline.recorded_token_ids)
    pieces = list(baseline.token_pieces)
    spans = [tuple(span) for span in baseline.token_spans]
    logprobs = list(baseline.token_logprobs)
    _validate_answer_tokens(
        response=response, token_ids=token_ids, token_pieces=pieces,
        token_spans=spans, token_logprobs=logprobs,
    )
    return {
        "response": response,
        "token_ids": token_ids,
        "token_pieces": pieces,
        "token_spans": spans,
        "token_logprobs": logprobs,
        "observation_id": baseline.observation_id,
        "fidelity": "baseline_token_score_observation",
    }


def _answer_document(answer: dict[str, Any]) -> dict[str, Any]:
    tokens = []
    for index, (token_id, piece, span) in enumerate(zip(
        answer["token_ids"], answer["token_pieces"], answer["token_spans"]
    )):
        item = {
            "index": index,
            "token_id": token_id,
            "piece": piece,
            "unicode_range": list(span),
        }
        if answer.get("token_logprobs") is not None:
            item["logprob"] = answer["token_logprobs"][index]
        tokens.append(item)
    return {
        "status": "available",
        "text": answer["response"],
        "sha256": _sha256_text(answer["response"]),
        "tokens": tokens,
        "fidelity": answer.get("fidelity"),
        "observation_id": answer.get("observation_id"),
    }


def _source_record(item: Mapping[str, Any], *, display_range: list[int]) -> dict[str, Any]:
    source_id = item.get("source_id")
    return {
        "source_id": source_id,
        "segment_id": item.get("segment_id"),
        "message_index": item.get("message_index"),
        "label": item.get("source_label"),
        "provenance_kind": item.get("provenance_kind"),
        "parent_source_id": item.get("parent_source_id"),
        "unicode_range": list(item.get("unicode_range")),
        "byte_range": list(item.get("byte_range")),
        "display_unicode_range": list(display_range),
        "granularity": "exact_span" if item.get("source_kind") == "source_span" else "whole_segment",
        "measurement": {
            "status": "not_measured",
            "experiment_id": None,
            "arm_id": None,
            "observation_id": None,
        },
        "summary": {
            "max_abs_token_delta_nats": None,
            "total_continuation_delta_nats": None,
            "above_floor_locus_count": 0,
        },
        "loci": [],
    }


def _arm_status(row: ExperimentArmView) -> str:
    if isinstance(row.observation, TokenScoreObservation):
        if row.observation.completed:
            return "measured"
        return row.observation.status
    observation_status = row.diagnostics.get("observation_status")
    if observation_status in {"unavailable", "failed"}:
        return str(observation_status)
    if row.state == "failed":
        return "failed"
    return "not_measured"


def _validate_view(view: ExperimentView, run: Mapping[str, Any], plan: Any) -> None:
    experiment = _plan_experiment(plan)
    state = _plan_field(plan, "execution_state")
    if not isinstance(state, ExecutionState):
        state = ExecutionState.from_run(run)
    if view.experiment_id != experiment.experiment_id:
        raise ContextInvestigationStale("persisted Q2 Experiment does not match the current plan")
    if view.base.run_id != run.get("id") or view.base.execution_fingerprint != state.execution_fingerprint:
        raise ContextInvestigationStale("persisted Q2 Experiment is bound to a stale Run execution")
    if not isinstance(view.evaluator, ScoreRecordedContinuation):
        raise ContextInvestigationStale("persisted Experiment is not a Q2 score measurement")
    if view.control is not None and not isinstance(view.control, TokenScoreObservation):
        raise ContextInvestigationStale("persisted Q2 baseline is not a TokenScoreObservation")
    expected_arms = tuple(experiment.arms)
    if len(view.arms) != len(expected_arms):
        raise ContextInvestigationStale("persisted Q2 arm set does not match the current plan")
    display_ids = set(_plan_ids(plan, "display_source_ids"))
    measurement_ids = set(_plan_ids(plan, "measurement_source_ids"))
    if not measurement_ids.issubset(display_ids):
        raise ContextInvestigationStale("the Q2 measurement universe is not contained in the current receipt catalogue")
    for expected, row in zip(expected_arms, view.arms):
        if row.arm_id != expected.arm_id or not isinstance(row.intervention, DeleteSource):
            raise ContextInvestigationStale("persisted Q2 arm is not a canonical DeleteSource arm")
        source_ids = tuple(row.intervention.source_ids)
        if len(source_ids) != 1 or source_ids != tuple(expected.intervention.source_ids):
            raise ContextInvestigationStale("persisted Q2 arm is not a one-source deletion")
        if row.observation is not None and not isinstance(row.observation, TokenScoreObservation):
            raise ContextInvestigationStale("persisted Q2 deletion evidence is not a TokenScoreObservation")
        if isinstance(row.observation, TokenScoreObservation):
            if (
                row.observation_id != row.observation.observation_id
                or
                row.observation.run_id != run.get("id")
                or row.observation.base_execution_fingerprint != state.execution_fingerprint
                or row.observation.condition != row.condition
            ):
                raise ContextInvestigationStale("persisted Q2 observation is bound to stale or mismatched evidence")


def _locus_id(*, experiment_id: str, source_id: str, token_range: tuple[int, int], floor_nats: float) -> str:
    return "locus_" + _digest({
        "projection_version": LOCUS_PROJECTION_VERSION,
        "experiment_id": experiment_id,
        "source_id": source_id,
        "answer_token_range": list(token_range),
        "floor_nats": floor_nats,
    })[:24]


def project_source_loci(
    baseline: TokenScoreObservation,
    intervention: TokenScoreObservation,
    *,
    source_id: str,
    floor_nats: float,
    experiment_id: str,
    arm_id: str,
) -> list[dict[str, Any]]:
    """Project one direct score vector into contiguous signed answer loci."""
    floor_nats = _floor(floor_nats)
    if not isinstance(baseline, TokenScoreObservation) or not isinstance(intervention, TokenScoreObservation):
        return []
    delta = TokenScoreDelta.from_observations(baseline, intervention)
    if delta.status != "completed":
        return []
    values = list(delta.deltas)
    spans = list(delta.token_spans)
    loci: list[dict[str, Any]] = []
    index = 0
    while index < len(values):
        value = float(values[index])
        if abs(value) <= floor_nats:
            index += 1
            continue
        direction = "support" if value > floor_nats else "suppression"
        start = index
        index += 1
        while index < len(values):
            current = float(values[index])
            same_sign = current > floor_nats if direction == "support" else current < -floor_nats
            if not same_sign:
                break
            index += 1
        end = index
        token_range = (start, end)
        answer_start = spans[start][0]
        answer_end = spans[end - 1][1]
        signed = sum(values[start:end])
        loci.append({
            "locus_id": _locus_id(
                experiment_id=experiment_id, source_id=source_id,
                token_range=token_range, floor_nats=floor_nats,
            ),
            "source_id": source_id,
            "direction": direction,
            "classification": "measured_support" if direction == "support" else "measured_suppression",
            "answer_token_range": [start, end],
            "answer_unicode_range": [answer_start, answer_end],
            "answer_text": "".join(baseline.token_pieces[start:end]),
            "delta_nats": signed,
            "absolute_delta_nats": abs(signed),
            "baseline_selected_logp": sum(baseline.token_logprobs[start:end]),
            "intervened_selected_logp": sum(intervention.token_logprobs[start:end]),
            "experiment_id": experiment_id,
            "arm_id": arm_id,
            "baseline_observation_id": baseline.observation_id,
            "intervention_observation_id": intervention.observation_id,
            "per_token_delta_nats": values[start:end],
        })
    return loci


def _effect_status(row: ExperimentArmView) -> str:
    status = _arm_status(row)
    return "measured" if status == "measured" else status


def _base_document(*, run: Mapping[str, Any], plan: Any, context: dict[str, Any],
                   answer: dict[str, Any] | None, floor_nats: float, status: str,
                   measurement_status: str, reason: str | None = None,
                   experiment_id: str | None = None) -> dict[str, Any]:
    measurement_ids = list(_plan_ids(plan, "measurement_source_ids"))
    measurement = {
        "status": measurement_status,
        "experiment_id": experiment_id or getattr(_plan_experiment(plan), "experiment_id", None),
        "evaluator": "score_recorded_continuation",
        "intervention": "delete_source",
        "source_universe": measurement_ids,
        "source_universe_basis": _plan_field(plan, "source_universe_basis"),
        "completed_source_count": 0,
        "unavailable_source_count": 0,
        "failed_source_count": 0,
        "not_measured_source_count": len(measurement_ids),
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run.get("id"),
        "status": status,
        "measurement": measurement,
        "context": {
            "coordinate_basis": context.get("coordinate_basis"),
            "separator": context.get("separator"),
            "blocks": context.get("blocks", []),
        },
        "answer": answer or {"status": "unavailable"},
        "sources": [],
        "loci": [],
        "presentation": {"measurement_floor_nats": floor_nats},
        "capabilities": {
            "context_effects": "measurable",
            "measurement": measurement_status,
            "answer_selection_projection": "available" if answer else "unavailable",
        },
    }
    if reason:
        document["reason"] = reason
        document["reason_code"] = "context_investigation_unavailable" if status == "unavailable" else "context_investigation_stale"
    return document


def build_context_investigation_reader(
    run: Mapping[str, Any],
    plan: Any,
    *,
    observation_store: ObservationStore,
    floor_nats: float | None = None,
) -> dict[str, Any]:
    """Build the linked reader from current Run facts and persisted Q2 evidence."""
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
        raise ContextInvestigationError("a recorded Run with a non-empty id is required")
    if not isinstance(observation_store, ObservationStore):
        raise TypeError("observation_store must be an ObservationStore")
    floor = _floor(floor_nats)
    try:
        context = _receipt_context(run, plan)
    except ContextInvestigationError as exc:
        return _base_document(
            run=run, plan=plan, context={"coordinate_basis": DISPLAY_COORDINATE_BASIS, "separator": DISPLAY_SEPARATOR},
            answer=None, floor_nats=floor, status="unavailable", measurement_status="unavailable", reason=str(exc),
        )

    experiment_id = _plan_experiment(plan).experiment_id
    try:
        view = observation_store.get_experiment(experiment_id)
    except ObservationIntegrityError as exc:
        return _base_document(
            run=run, plan=plan, context=context, answer=None, floor_nats=floor,
            status="stale", measurement_status="stale", reason=str(exc), experiment_id=experiment_id,
        )
    except (ObservationNotFound, ExperimentPersistenceError, KeyError):
        answer = None
        try:
            state = _plan_field(plan, "execution_state")
            answer = _answer_from_baseline(run, None, execution_fingerprint=state.execution_fingerprint)
        except ContextInvestigationError:
            pass
        document = _base_document(
            run=run, plan=plan, context=context, answer=_answer_document(answer) if answer else None,
            floor_nats=floor, status="not_measured", measurement_status="not_measured", experiment_id=experiment_id,
        )
        display_by_id = {item.get("source_id"): item for item in context["catalog"]}
        document["sources"] = [
            _source_record(display_by_id[source_id], display_range=context["source_display_ranges"][source_id])
            for source_id in _plan_ids(plan, "display_source_ids")
        ]
        return document

    try:
        _validate_view(view, run, plan)
        baseline = view.control if isinstance(view.control, TokenScoreObservation) else None
        state = _plan_field(plan, "execution_state")
        answer_evidence = _answer_from_baseline(
            run, baseline, execution_fingerprint=state.execution_fingerprint,
        )
    except ContextInvestigationStale as exc:
        return _base_document(
            run=run, plan=plan, context=context, answer=None, floor_nats=floor,
            status="stale", measurement_status="stale", reason=str(exc), experiment_id=experiment_id,
        )
    except ContextInvestigationUnavailable as exc:
        return _base_document(
            run=run, plan=plan, context=context, answer=None, floor_nats=floor,
            status="unavailable", measurement_status="unavailable", reason=str(exc), experiment_id=experiment_id,
        )

    display_ids = _plan_ids(plan, "display_source_ids")
    display_id_set = set(display_ids)
    source_records = {
        source_id: _source_record(item, display_range=context["source_display_ranges"][source_id])
        for source_id, item in (
            (item.get("source_id"), item) for item in context["catalog"]
            if isinstance(item, Mapping) and isinstance(item.get("source_id"), str)
        )
        if source_id in display_id_set
    }
    baseline_status = "measured" if baseline is not None and baseline.completed else "not_measured"
    loci: list[dict[str, Any]] = []
    statuses: dict[str, str] = {}
    measurement_sources = set(_plan_ids(plan, "measurement_source_ids"))
    for row in view.arms:
        if not isinstance(row, ExperimentArmView) or not isinstance(row.intervention, DeleteSource):
            continue
        source_ids = tuple(row.intervention.source_ids)
        if len(source_ids) != 1 or source_ids[0] not in measurement_sources:
            continue
        source_id = source_ids[0]
        status = _effect_status(row)
        statuses[source_id] = status
        record = source_records.get(source_id)
        if record is None:
            continue
        measurement = record["measurement"]
        measurement.update({
            "status": status,
            "experiment_id": view.experiment_id,
            "arm_id": row.arm_id,
            "observation_id": row.observation_id,
            "intervention": row.intervention.to_dict(),
        })
        if status != "measured" or not isinstance(row.observation, TokenScoreObservation):
            continue
        delta = TokenScoreDelta.from_observations(baseline, row.observation) if baseline else None
        if delta is None or delta.status != "completed":
            return _base_document(
                run=run, plan=plan, context=context, answer=None, floor_nats=floor,
                status="stale", measurement_status="stale",
                reason="a completed Q2 arm does not align with the baseline token vector",
                experiment_id=view.experiment_id,
            )
        values = list(delta.deltas)
        record["summary"] = {
            "max_abs_token_delta_nats": max((abs(float(value)) for value in values), default=0.0),
            "total_continuation_delta_nats": delta.total_delta_nats,
            "above_floor_locus_count": 0,
        }
        row_loci = project_source_loci(
            baseline, row.observation, source_id=source_id, floor_nats=floor,
            experiment_id=view.experiment_id, arm_id=row.arm_id,
        )
        record["loci"] = [item["locus_id"] for item in row_loci]
        record["summary"]["above_floor_locus_count"] = len(row_loci)
        loci.extend(row_loci)

    for source_id in measurement_sources:
        statuses.setdefault(source_id, "not_measured")
    completed_count = sum(status == "measured" for status in statuses.values())
    unavailable_count = sum(status == "unavailable" for status in statuses.values())
    failed_count = sum(status == "failed" for status in statuses.values())
    not_measured_count = len(measurement_sources) - completed_count - unavailable_count - failed_count
    if completed_count == len(measurement_sources) and baseline_status == "measured":
        measurement_status = "completed"
    elif completed_count or unavailable_count or failed_count:
        measurement_status = "partial"
    else:
        measurement_status = "not_measured"
    document = _base_document(
        run=run, plan=plan, context=context, answer=_answer_document(answer_evidence),
        floor_nats=floor, status=measurement_status, measurement_status=measurement_status,
        experiment_id=view.experiment_id,
    )
    document["measurement"].update({
        "completed_source_count": completed_count,
        "unavailable_source_count": unavailable_count,
        "failed_source_count": failed_count,
        "not_measured_source_count": max(0, not_measured_count),
        "baseline_observation_id": baseline.observation_id if baseline else None,
    })
    document["sources"] = [
        source_records[source_id] for source_id in display_ids if source_id in source_records
    ]
    display_order = {source_id: index for index, source_id in enumerate(display_ids)}
    document["loci"] = sorted(loci, key=lambda item: (
        display_order.get(item["source_id"], 2**31 - 1),
        item["answer_token_range"][0], item["locus_id"],
    ))
    return document


def _rows_for_single_source(view: ExperimentView) -> list[tuple[int, ExperimentArmView, str]]:
    result = []
    for ordinal, row in enumerate(view.arms):
        if not isinstance(row.intervention, DeleteSource):
            continue
        source_ids = tuple(row.intervention.source_ids)
        if len(source_ids) == 1:
            result.append((ordinal, row, source_ids[0]))
    return result


def _selection_document(resolved: ResolvedAnswerSelection) -> dict[str, Any]:
    return {
        "unicode_range": list(resolved.character_range),
        "token_range": list(resolved.token_range),
        "text": resolved.selected_text,
        "token_ids": list(resolved.token_ids),
        "token_pieces": list(resolved.token_pieces),
    }


def query_answer_effects(
    experiment_view: ExperimentView,
    answer_selection: AnswerSelection,
    *,
    floor_nats: float = DEFAULT_MEASUREMENT_FLOOR_NATS,
) -> dict[str, Any]:
    """Rank measured one-source effects for an arbitrary recorded answer span."""
    floor_nats = _floor(floor_nats)
    if not isinstance(experiment_view, ExperimentView):
        raise TypeError("query_answer_effects requires an ExperimentView")
    if not isinstance(answer_selection, AnswerSelection):
        raise TypeError("answer_selection must be an AnswerSelection")
    baseline = experiment_view.control
    if not isinstance(baseline, TokenScoreObservation) or not baseline.completed:
        return {
            "schema_version": QUERY_SCHEMA_VERSION,
            "status": "not_measured",
            "selection": None,
            "effects": [],
            "measurement_floor_nats": floor_nats,
        }
    try:
        resolved = resolve_answer_selection_from_observation(
            baseline, answer_selection, run_id=experiment_view.base.run_id,
        )
    except AnswerSelectionUnavailable as exc:
        raise AnswerSelectionProjectionUnavailable(str(exc)) from exc
    effects: list[dict[str, Any]] = []
    for ordinal, row, source_id in _rows_for_single_source(experiment_view):
        status = _effect_status(row)
        effect: dict[str, Any] = {
            "source_id": source_id,
            "delta_nats": None,
            "direction": "below_floor" if status == "measured" else status,
            "classification": "below_floor" if status == "measured" else None,
            "measurement_status": status,
            "experiment_id": experiment_view.experiment_id,
            "arm_id": row.arm_id,
            "baseline_observation_id": baseline.observation_id,
            "intervention_observation_id": row.observation_id,
        }
        if status == "measured" and isinstance(row.observation, TokenScoreObservation):
            delta = TokenScoreDelta.from_observations(baseline, row.observation)
            if delta.status != "completed":
                effect["measurement_status"] = "unavailable"
                effect["direction"] = "unavailable"
                effect["classification"] = None
            else:
                selected = sum(delta.deltas[index] for index in resolved.token_indices)
                effect["delta_nats"] = selected
                if selected > floor_nats:
                    effect["direction"] = "support"
                    effect["classification"] = "measured_support"
                elif selected < -floor_nats:
                    effect["direction"] = "suppression"
                    effect["classification"] = "measured_suppression"
                else:
                    effect["direction"] = "below_floor"
                    effect["classification"] = "below_floor"
        effect["_order"] = ordinal
        effects.append(effect)
    effects.sort(key=lambda item: (
        0 if item["delta_nats"] is not None else 1,
        -abs(item["delta_nats"]) if item["delta_nats"] is not None else 0.0,
        item["_order"],
    ))
    for item in effects:
        item.pop("_order", None)
    return {
        "schema_version": QUERY_SCHEMA_VERSION,
        "status": "completed" if all(item["measurement_status"] == "measured" for item in effects) else "partial",
        "selection": _selection_document(resolved),
        "effects": effects,
        "measurement_floor_nats": floor_nats,
    }


def project_locus_details(reader: Mapping[str, Any], locus_id: str) -> dict[str, Any]:
    """Return exact evidence references for one read-side locus."""
    if not isinstance(reader, Mapping) or not isinstance(locus_id, str) or not locus_id:
        raise ContextInvestigationError("reader and locus_id are required")
    locus = next((item for item in reader.get("loci", []) if item.get("locus_id") == locus_id), None)
    if not isinstance(locus, Mapping):
        raise ContextInvestigationError("locus was not found")
    source = next((item for item in reader.get("sources", []) if item.get("source_id") == locus.get("source_id")), None)
    if not isinstance(source, Mapping):
        raise ContextInvestigationStale("locus source record is unavailable")
    start, end = locus["answer_token_range"]
    return {
        "source": {
            "source_id": source.get("source_id"),
            "label": source.get("label"),
            "unicode_range": source.get("unicode_range"),
            "byte_range": source.get("byte_range"),
            "granularity": source.get("granularity"),
        },
        "answer": {
            "text": locus.get("answer_text"),
            "unicode_range": locus.get("answer_unicode_range"),
            "token_range": [start, end],
        },
        "effect": {
            "direction": locus.get("direction"),
            "classification": locus.get("classification"),
            "delta_nats": locus.get("delta_nats"),
            "baseline_selected_logp": locus.get("baseline_selected_logp"),
            "intervened_selected_logp": locus.get("intervened_selected_logp"),
        },
        "experiment": {"experiment_id": locus.get("experiment_id"), "arm_id": locus.get("arm_id")},
        "evidence": {
            "baseline_observation_id": locus.get("baseline_observation_id"),
            "intervention_observation_id": locus.get("intervention_observation_id"),
        },
        "intervention": source.get("measurement", {}).get(
            "intervention", {"kind": "delete_source", "target": {"kind": "context_selection", "source_ids": [source.get("source_id")]}}
        ),
        "evaluator": "score_recorded_continuation",
    }


__all__ = [
    "AnswerSelectionProjectionUnavailable", "ContextInvestigationError", "ContextInvestigationStale",
    "ContextInvestigationUnavailable", "DEFAULT_MEASUREMENT_FLOOR_NATS", "DISPLAY_COORDINATE_BASIS",
    "DISPLAY_SEPARATOR", "LOCUS_PROJECTION_VERSION", "QUERY_SCHEMA_VERSION", "SCHEMA_VERSION",
    "build_context_investigation_reader", "project_locus_details", "project_source_loci",
    "query_answer_effects",
]
