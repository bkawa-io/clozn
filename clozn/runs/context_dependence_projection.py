"""Read-only answer-selection projections over a v2 Context Dependence study.

The expensive Context Dependence measurement is deliberately run-level: every
direct deletion arm stores its vector over the complete recorded continuation.
This module is the small, strict read-side counterpart.  It turns a Unicode
selection in the recorded response into the *already scored* token interval and
sums the persisted vector for each direct experiment.  It never imports a
model, scorer, router, or persistence layer.

Coordinates are Unicode code-point offsets (Python ``str`` indices), half-open
``[start, end)``.  We accept only selections whose two endpoints are recorded
token boundaries.  Guessing a partial-token correspondence would make a
selection look exact when it is not, so malformed or stale evidence fails
closed with :class:`ContextDependenceProjectionError`.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Mapping


STUDY_SCHEMA = "clozn.context-dependence-study.v2"
QUERY_SCHEMA = "clozn.context-dependence-query.v1"


class ContextDependenceProjectionError(ValueError):
    """A query or its persisted study cannot be reconciled faithfully.

    ``status`` is intentionally part of the typed error so the HTTP route can
    distinguish a caller's invalid coordinates (400) from an artifact that is
    absent, legacy, malformed, or no longer describes the recorded run (409).
    """

    def __init__(self, message: str, *, code: str, status: int = 409):
        super().__init__(message)
        self.code = code
        self.status = status


def _bad_selection(message: str) -> ContextDependenceProjectionError:
    return ContextDependenceProjectionError(message, code="invalid_output_range", status=400)


def _stale(message: str, *, code: str = "context_dependence_projection_stale") -> ContextDependenceProjectionError:
    return ContextDependenceProjectionError(message, code=code, status=409)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise _stale(f"persisted Context Dependence {field} must be a finite number")
    return float(value)


def _same_number(actual: float, expected: float, *, field: str) -> None:
    # Values are persisted sums of the same vector.  This deliberately permits
    # harmless IEEE accumulation order differences but not a different target
    # or a hand-edited aggregate.
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise _stale(f"persisted Context Dependence {field} disagrees with its full token evidence")


def _range(
    value: Any, *, field: str, upper: int | None = None, allow_empty: bool = False,
) -> tuple[int, int]:
    if not (
        isinstance(value, (list, tuple)) and len(value) == 2
        and _is_int(value[0]) and _is_int(value[1])
    ):
        raise _stale(f"persisted Context Dependence {field} must be a two-integer half-open range")
    start, end = int(value[0]), int(value[1])
    invalid_order = end < start if allow_empty else end <= start
    if start < 0 or invalid_order or (upper is not None and end > upper):
        qualifier = "ordered" if allow_empty else "non-empty"
        raise _stale(f"persisted Context Dependence {field} is not an in-bounds {qualifier} range")
    return start, end


def _safe_source(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return source identity/labels, never opaque additional properties.

    A receipt source can be extended with transport-specific fields, including
    fields that contain source text.  The query needs addressing metadata and
    human labels, not those opaque fields, so copying a small allow-list avoids
    accidentally turning the projection into a prompt-text endpoint.
    """
    fields = (
        "source_id", "segment_id", "message_index", "unicode_range", "byte_range", "role",
        "client_source_id", "source_label", "provenance_kind", "source_kind", "parent_source_id",
        "content_sha256",
    )
    return {field: deepcopy(source[field]) for field in fields if field in source}


def _safe_removed_range(value: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("source_id", "message_index", "unicode_range", "byte_range")
    return {field: deepcopy(value[field]) for field in fields if field in value}


def _safe_effective_neutralized_range(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return neutralization evidence without carrying any prompt text."""
    fields = (
        "message_index", "unicode_range", "original_content_sha256", "replacement_content_sha256",
        "original_unicode_code_points", "replacement_unicode_code_points",
        "original_utf8_bytes", "replacement_utf8_bytes",
    )
    return {field: deepcopy(value[field]) for field in fields if field in value}


def _validate_selection(output_start: Any, output_end: Any, *, response: str) -> tuple[int, int]:
    if not _is_int(output_start) or not _is_int(output_end):
        raise _bad_selection("output_start and output_end must be integers")
    start, end = int(output_start), int(output_end)
    if start < 0 or end <= start:
        raise _bad_selection("output_start and output_end must be non-negative with output_end > output_start")
    if end > len(response):
        raise _bad_selection("output_end is beyond the recorded response Unicode length")
    return start, end


def _validated_token_records(study: Mapping[str, Any], *, response: str) -> list[dict[str, Any]]:
    baseline = study.get("baseline")
    if not isinstance(baseline, Mapping):
        raise _stale("persisted Context Dependence study has no baseline token records")
    if baseline.get("provenance") != "measured" or baseline.get("scored_once") is not True:
        raise _stale("persisted Context Dependence baseline is not a measured one-pass baseline")
    raw_tokens = baseline.get("tokens")
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise _stale("persisted Context Dependence baseline.tokens must be a non-empty array")

    records: list[dict[str, Any]] = []
    cursor = 0
    for index, raw in enumerate(raw_tokens):
        if not isinstance(raw, Mapping):
            raise _stale("persisted Context Dependence baseline.tokens has a malformed token record")
        if raw.get("index") != index:
            raise _stale("persisted Context Dependence baseline token indices are not contiguous")
        piece = raw.get("piece")
        if not isinstance(piece, str) or not piece:
            raise _stale("persisted Context Dependence baseline token pieces must be non-empty strings")
        start, end = _range(raw.get("unicode_range"), field="baseline token unicode_range", upper=len(response))
        if (start, end) != (cursor, cursor + len(piece)):
            raise _stale("persisted Context Dependence token coordinates do not reconstruct the recorded response")
        token_id = raw.get("token_id")
        if token_id is not None and not _is_int(token_id):
            raise _stale("persisted Context Dependence baseline token_id must be an integer when supplied")
        records.append({
            "index": index,
            "token_id": token_id,
            "piece": piece,
            "unicode_range": [start, end],
            "logprob": _finite(raw.get("logprob"), field="baseline token logprob"),
        })
        cursor = end

    if cursor != len(response) or "".join(item["piece"] for item in records) != response:
        raise _stale("persisted Context Dependence token pieces do not reconstruct the recorded response")
    _same_number(
        _finite(baseline.get("teacher_forced_logp"), field="baseline teacher_forced_logp"),
        sum(item["logprob"] for item in records),
        field="baseline teacher_forced_logp",
    )
    return records


def _validated_continuation(study: Mapping[str, Any], *, response: str, records: list[dict[str, Any]]) -> tuple[str, str]:
    continuation = study.get("continuation")
    if not isinstance(continuation, Mapping):
        raise _stale("persisted Context Dependence study has no continuation identity")
    fidelity = continuation.get("fidelity")
    if fidelity not in {"exact_recorded_token_ids", "recomputed_from_recorded_response_text"}:
        raise _stale("persisted Context Dependence continuation fidelity is not supported")
    if continuation.get("unicode_offset_basis") != "recorded_response_unicode":
        raise _stale("persisted Context Dependence continuation uses a different Unicode coordinate basis")
    if continuation.get("recorded_text") != response or continuation.get("scored_text") != response:
        raise _stale("persisted Context Dependence continuation text is stale for this recorded response")

    exact = fidelity == "exact_recorded_token_ids"
    if continuation.get("token_ids_exact") is not exact:
        raise _stale("persisted Context Dependence continuation fidelity disagrees with token_ids_exact")
    expected_kind = "recorded_token_ids" if exact else "recorded_response_text_retokenized"
    if continuation.get("kind") != expected_kind:
        raise _stale("persisted Context Dependence continuation kind disagrees with fidelity")
    if continuation.get("retokenized") is not (not exact):
        raise _stale("persisted Context Dependence continuation retokenized state disagrees with fidelity")
    if exact:
        token_ids = continuation.get("token_ids")
        if not isinstance(token_ids, list) or len(token_ids) != len(records):
            raise _stale("exact Context Dependence continuation is missing recorded token IDs")
        if any(not _is_int(token_id) for token_id in token_ids):
            raise _stale("exact Context Dependence continuation token IDs must be integers")
        if [item["token_id"] for item in records] != token_ids:
            raise _stale("persisted Context Dependence baseline token IDs disagree with continuation IDs")

    return ("exact" if exact else "recomputed"), str(fidelity)


def _validated_sources(study: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    source_identity = study.get("source_identity")
    if not isinstance(source_identity, Mapping) or not isinstance(source_identity.get("sources"), list):
        raise _stale("persisted Context Dependence study has no canonical source identity")
    result: dict[str, dict[str, Any]] = {}
    for source in source_identity["sources"]:
        if not isinstance(source, Mapping):
            raise _stale("persisted Context Dependence source identity has a malformed source")
        source_id = source.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in result:
            raise _stale("persisted Context Dependence source IDs must be unique non-empty strings")
        # Ranges are not used for response selection, but source addressing is
        # still evidence returned by this endpoint and must itself be sane.
        # A whole empty chat message is still a real structural deletion
        # source: removing it changes roles/template separators even though
        # its content range is [0, 0). Exact source *spans* remain non-empty at
        # receipt capture, but the read side must not reject an honest empty
        # whole-message root that was not selected by this answer query.
        _range(source.get("unicode_range"), field="source unicode_range", allow_empty=True)
        _range(source.get("byte_range"), field="source byte_range", allow_empty=True)
        if not _is_int(source.get("message_index")) or source["message_index"] < 0:
            raise _stale("persisted Context Dependence source message_index must be non-negative")
        result[source_id] = dict(source)
    return result


def _selection_bounds(records: list[dict[str, Any]], *, start: int, end: int) -> tuple[int, int]:
    starts = {item["unicode_range"][0]: item["index"] for item in records}
    ends = {item["unicode_range"][1]: item["index"] + 1 for item in records}
    token_start = starts.get(start)
    token_end = ends.get(end)
    if token_start is None or token_end is None:
        raise _bad_selection("output selection must begin and end on persisted recorded-token boundaries")
    if token_end <= token_start:
        # This should be impossible once input and records are validated, but
        # preserves a clear user-facing range error if an exotic token stream
        # violates the expected coordinate ordering.
        raise _bad_selection("output selection does not cover a non-empty recorded token range")
    return token_start, token_end


def _effect_rows(
    study: Mapping[str, Any], *, source_by_id: Mapping[str, dict[str, Any]],
    token_count: int, token_start: int, token_end: int, baseline_logp: float,
) -> list[dict[str, Any]]:
    experiments = study.get("experiments")
    if not isinstance(experiments, list):
        raise _stale("persisted Context Dependence experiments must be an array")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    expected_indices = list(range(token_count))
    for experiment in experiments:
        if not isinstance(experiment, Mapping):
            raise _stale("persisted Context Dependence experiment is malformed")
        experiment_id = experiment.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id or experiment_id in seen_ids:
            raise _stale("persisted Context Dependence experiment IDs must be unique non-empty strings")
        seen_ids.add(experiment_id)
        if experiment.get("intervention_operator") != "delete_source" or experiment.get("provenance") != "measured":
            raise _stale("Context Dependence query only accepts directly measured delete_source experiments")
        removed_ids = experiment.get("removed_source_ids")
        if (
            not isinstance(removed_ids, list) or not removed_ids
            or any(not isinstance(item, str) or not item for item in removed_ids)
            or len(set(removed_ids)) != len(removed_ids)
            or any(item not in source_by_id for item in removed_ids)
        ):
            raise _stale("persisted Context Dependence experiment has invalid canonical removed_source_ids")
        deltas = experiment.get("per_token_delta_nats")
        if not isinstance(deltas, list) or len(deltas) != token_count:
            raise _stale("persisted Context Dependence experiment does not contain a full per-token delta vector")
        deltas = [_finite(value, field="experiment per_token_delta_nats") for value in deltas]
        if experiment.get("token_indices") != expected_indices:
            raise _stale("persisted Context Dependence experiment token_indices do not cover the recorded continuation")

        full_delta = sum(deltas)
        persisted_delta = _finite(experiment.get("delta_nats"), field="experiment delta_nats")
        _same_number(persisted_delta, full_delta, field="experiment delta_nats")
        exp_baseline = _finite(experiment.get("baseline_logp"), field="experiment baseline_logp")
        intervened = _finite(experiment.get("intervened_logp"), field="experiment intervened_logp")
        _same_number(exp_baseline, baseline_logp, field="experiment baseline_logp")
        _same_number(exp_baseline - intervened, full_delta, field="experiment delta_nats")

        raw_ranges = experiment.get("exact_removed_ranges")
        if not isinstance(raw_ranges, list) or len(raw_ranges) != len(removed_ids):
            raise _stale("persisted Context Dependence experiment exact_removed_ranges are malformed")
        ranges_by_source: dict[str, Mapping[str, Any]] = {}
        for removed in raw_ranges:
            if not isinstance(removed, Mapping):
                raise _stale("persisted Context Dependence experiment has a malformed removed range")
            source_id = removed.get("source_id")
            if not isinstance(source_id, str) or source_id not in source_by_id or source_id in ranges_by_source:
                raise _stale("persisted Context Dependence experiment removed ranges do not identify its sources")
            _range(
                removed.get("unicode_range"), field="experiment removed unicode_range",
                allow_empty=True,
            )
            _range(
                removed.get("byte_range"), field="experiment removed byte_range",
                allow_empty=True,
            )
            if not _is_int(removed.get("message_index")) or removed["message_index"] < 0:
                raise _stale("persisted Context Dependence experiment removed message_index is malformed")
            source = source_by_id[source_id]
            for field in ("message_index", "unicode_range", "byte_range"):
                if removed.get(field) != source.get(field):
                    raise _stale("persisted Context Dependence experiment removed ranges disagree with canonical source identity")
            ranges_by_source[source_id] = removed
        if set(ranges_by_source) != set(removed_ids):
            raise _stale("persisted Context Dependence experiment removed ranges disagree with removed_source_ids")

        selection_delta = sum(deltas[token_start:token_end])
        rows.append({
            "experiment_id": experiment_id,
            "intervention_operator": "delete_source",
            "provenance": "measured",
            "removed_source_ids": list(removed_ids),
            "sources": [_safe_source(source_by_id[source_id]) for source_id in removed_ids],
            "exact_removed_ranges": [_safe_removed_range(ranges_by_source[source_id]) for source_id in removed_ids],
            # This is the answer-selection projection: always recomputed from
            # the persisted full vector, never reused from a target-specific
            # aggregate.
            "delta_nats": selection_delta,
            "full_continuation_delta_nats": full_delta,
        })
    return rows


def _merged_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Union source intervals so nested control evidence has one effective span."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _neutralization_control_rows(
    study: Mapping[str, Any], *, source_by_id: Mapping[str, dict[str, Any]],
    token_count: int, token_start: int, token_end: int, baseline_logp: float,
) -> list[dict[str, Any]]:
    """Project separately persisted robustness controls, never deletion effects.

    The v2 study schema permits this collection to be absent for artifacts
    computed before a control was requested.  If it exists, every row must
    prove its own operator, full vector, source binding, merged exact ranges,
    and honest Unicode/UTF-8 replacement contract before it becomes public.
    """
    raw_controls = study.get("robustness_controls")
    if raw_controls is None:
        return []
    if not isinstance(raw_controls, list):
        raise _stale("persisted Context Dependence robustness_controls must be an array")
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    expected_indices = list(range(token_count))
    expected_neutralization = {
        "operator": "neutralize_source",
        "strategy": "matched_length_neutral_filler",
        "recipe": "clozn.matched_length_neutral_filler.v1",
        "length_contract": "unicode_code_points_exact",
        "utf8_byte_length_contract": "not_guaranteed",
        "message_structure": "preserved",
    }
    for control in raw_controls:
        if not isinstance(control, Mapping):
            raise _stale("persisted Context Dependence robustness control is malformed")
        control_id = control.get("control_id")
        if not isinstance(control_id, str) or not control_id or control_id in seen_ids:
            raise _stale("persisted Context Dependence robustness control IDs must be unique non-empty strings")
        seen_ids.add(control_id)
        if (
            control.get("intervention_operator") != "neutralize_source"
            or control.get("provenance") != "measured_matched_length_neutralization_control"
        ):
            raise _stale("Context Dependence robustness controls must be explicitly measured neutralizations")
        source_ids = control.get("neutralized_source_ids")
        if (
            not isinstance(source_ids, list) or not source_ids
            or any(not isinstance(item, str) or not item for item in source_ids)
            or len(set(source_ids)) != len(source_ids)
            or any(item not in source_by_id for item in source_ids)
        ):
            raise _stale("persisted Context Dependence robustness control has invalid canonical source IDs")
        if control.get("neutralization") != expected_neutralization:
            raise _stale("persisted Context Dependence robustness control has an unsupported neutralization recipe")

        deltas = control.get("per_token_delta_nats")
        if not isinstance(deltas, list) or len(deltas) != token_count:
            raise _stale("persisted Context Dependence robustness control lacks a full per-token delta vector")
        deltas = [_finite(value, field="robustness control per_token_delta_nats") for value in deltas]
        if control.get("token_indices") != expected_indices:
            raise _stale("persisted Context Dependence robustness control token_indices do not cover the recorded continuation")
        full_delta = sum(deltas)
        _same_number(
            _finite(control.get("delta_nats"), field="robustness control delta_nats"),
            full_delta, field="robustness control delta_nats",
        )
        control_baseline = _finite(control.get("baseline_logp"), field="robustness control baseline_logp")
        intervened = _finite(control.get("intervened_logp"), field="robustness control intervened_logp")
        _same_number(control_baseline, baseline_logp, field="robustness control baseline_logp")
        _same_number(control_baseline - intervened, full_delta, field="robustness control delta_nats")

        raw_ranges = control.get("exact_neutralized_ranges")
        if not isinstance(raw_ranges, list) or len(raw_ranges) != len(source_ids):
            raise _stale("persisted Context Dependence robustness control exact_neutralized_ranges are malformed")
        ranges_by_source: dict[str, Mapping[str, Any]] = {}
        grouped: dict[int, list[tuple[int, int]]] = {}
        for neutralized in raw_ranges:
            if not isinstance(neutralized, Mapping):
                raise _stale("persisted Context Dependence robustness control has a malformed exact range")
            source_id = neutralized.get("source_id")
            if not isinstance(source_id, str) or source_id not in source_by_id or source_id in ranges_by_source:
                raise _stale("persisted Context Dependence robustness control ranges do not identify its sources")
            start, end = _range(neutralized.get("unicode_range"), field="robustness control unicode_range")
            _range(neutralized.get("byte_range"), field="robustness control byte_range")
            message_index = neutralized.get("message_index")
            if not _is_int(message_index) or message_index < 0:
                raise _stale("persisted Context Dependence robustness control message_index is malformed")
            source = source_by_id[source_id]
            for field in ("message_index", "unicode_range", "byte_range", "content_sha256", "source_kind"):
                if neutralized.get(field) != source.get(field):
                    raise _stale("persisted Context Dependence robustness ranges disagree with canonical source identity")
            original_chars = neutralized.get("original_unicode_code_points")
            replacement_chars = neutralized.get("replacement_unicode_code_points")
            original_bytes = neutralized.get("original_utf8_bytes")
            replacement_bytes = neutralized.get("replacement_utf8_bytes")
            if (
                not all(_is_int(item) and item > 0 for item in (
                    original_chars, replacement_chars, original_bytes, replacement_bytes,
                ))
                or original_chars != end - start or replacement_chars != original_chars
                or not isinstance(neutralized.get("replacement_content_sha256"), str)
                or len(neutralized["replacement_content_sha256"]) != 64
            ):
                raise _stale("persisted Context Dependence robustness range violates its length evidence")
            ranges_by_source[source_id] = neutralized
            grouped.setdefault(message_index, []).append((start, end))
        if set(ranges_by_source) != set(source_ids):
            raise _stale("persisted Context Dependence robustness ranges disagree with neutralized_source_ids")

        expected_effective = [
            (message_index, offsets)
            for message_index, ranges in sorted(grouped.items())
            for offsets in _merged_ranges(ranges)
        ]
        effective = control.get("effective_neutralized_ranges")
        if not isinstance(effective, list) or len(effective) != len(expected_effective):
            raise _stale("persisted Context Dependence robustness control effective ranges are malformed")
        safe_effective: list[dict[str, Any]] = []
        for item, (expected_index, (expected_start, expected_end)) in zip(effective, expected_effective):
            if not isinstance(item, Mapping):
                raise _stale("persisted Context Dependence robustness control effective range is malformed")
            start, end = _range(item.get("unicode_range"), field="robustness control effective unicode_range")
            if item.get("message_index") != expected_index or (start, end) != (expected_start, expected_end):
                raise _stale("persisted Context Dependence robustness effective ranges do not equal the source union")
            original_chars = item.get("original_unicode_code_points")
            replacement_chars = item.get("replacement_unicode_code_points")
            if not (_is_int(original_chars) and _is_int(replacement_chars)
                    and original_chars == replacement_chars == end - start):
                raise _stale("persisted Context Dependence robustness effective range violates Unicode length preservation")
            if not all(
                isinstance(item.get(field), str) and len(item[field]) == 64
                for field in ("original_content_sha256", "replacement_content_sha256")
            ) or not all(
                _is_int(item.get(field)) and item[field] > 0
                for field in ("original_utf8_bytes", "replacement_utf8_bytes")
            ):
                raise _stale("persisted Context Dependence robustness effective range is missing byte/hash evidence")
            safe_effective.append(_safe_effective_neutralized_range(item))

        rows.append({
            "control_id": control_id,
            "intervention_operator": "neutralize_source",
            "provenance": "measured_matched_length_neutralization_control",
            "neutralized_source_ids": list(source_ids),
            "sources": [_safe_source(source_by_id[source_id]) for source_id in source_ids],
            "exact_neutralized_ranges": [_safe_removed_range(ranges_by_source[source_id]) for source_id in source_ids],
            "effective_neutralized_ranges": safe_effective,
            "neutralization": deepcopy(expected_neutralization),
            "delta_nats": sum(deltas[token_start:token_end]),
            "full_continuation_delta_nats": full_delta,
        })
    return rows


def build_context_dependence_query(run: Mapping[str, Any], *, output_start: Any, output_end: Any) -> dict[str, Any]:
    """Project one exact answer selection over persisted v2 direct experiments.

    The function is pure: callers pass an already-loaded run and this function
    neither writes it nor accesses an engine.  ``ContextDependenceProjectionError``
    identifies the HTTP status/code appropriate for invalid coordinates versus
    stale/unprojectable persisted evidence.
    """
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
        raise _stale("run identity is unavailable for Context Dependence projection", code="context_dependence_projection_invalid_run")
    response = run.get("response")
    if not isinstance(response, str):
        raise _stale("recorded response text is unavailable for Context Dependence projection")
    start, end = _validate_selection(output_start, output_end, response=response)

    study = run.get("context_dependence_study")
    if not isinstance(study, Mapping):
        raise _stale("no Context Dependence study has been computed for this run", code="context_dependence_projection_unavailable")
    schema_version = study.get("schema_version")
    if schema_version != STUDY_SCHEMA:
        code = "context_dependence_projection_legacy_artifact" if schema_version == "clozn.context-dependence-study.v1" else "context_dependence_projection_unavailable"
        raise _stale("Context Dependence selection projection requires a persisted v2 run-level study", code=code)
    if study.get("run_id") != run["id"]:
        raise _stale("persisted Context Dependence study belongs to a different recorded run")

    # The JSON schema catches version-level omissions/types; the local checks
    # below establish the stronger cross-field invariants a generic schema
    # cannot express (token reconstruction, vector sums, and run freshness).
    from clozn import schemas
    try:
        schemas.validate(study, STUDY_SCHEMA)
    except schemas.ValidationError as exc:
        raise _stale("persisted Context Dependence study does not satisfy its v2 evidence contract",
                     code="context_dependence_projection_invalid_artifact") from exc

    records = _validated_token_records(study, response=response)
    registration, fidelity = _validated_continuation(study, response=response, records=records)
    sources = _validated_sources(study)
    token_start, token_end = _selection_bounds(records, start=start, end=end)
    baseline_logp = sum(item["logprob"] for item in records)
    effects = _effect_rows(
        study, source_by_id=sources, token_count=len(records), token_start=token_start,
        token_end=token_end, baseline_logp=baseline_logp,
    )
    controls = _neutralization_control_rows(
        study, source_by_id=sources, token_count=len(records), token_start=token_start,
        token_end=token_end, baseline_logp=baseline_logp,
    )

    document = {
        "schema_version": QUERY_SCHEMA,
        "run_id": run["id"],
        "selection": {
            "unicode_range": [start, end],
            "text": response[start:end],
            "registration": registration,
            "fidelity": fidelity,
            "recorded_token_range": [token_start, token_end],
            "conditioned_prefix": {
                "unicode_range": [0, start],
                "recorded_token_range": [0, token_start],
                "text": response[:start],
            },
        },
        "measurement": {
            "state": "available",
            "study_schema_version": STUDY_SCHEMA,
            "intervention_operator": "delete_source",
            "provenance": "measured",
        },
        "measured_removal_effects": effects,
    }
    if controls:
        document["matched_length_neutralization_controls"] = controls
    # This projection is a public artifact in its own right.  Validate it at
    # the producer boundary so a future response-shape edit cannot silently
    # make the HTTP route emit an uncontracted document.
    schemas.validate(document, QUERY_SCHEMA)
    return document


__all__ = [
    "ContextDependenceProjectionError",
    "QUERY_SCHEMA",
    "STUDY_SCHEMA",
    "build_context_dependence_query",
]
