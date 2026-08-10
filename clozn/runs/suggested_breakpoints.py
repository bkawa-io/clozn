"""Suggested Breakpoints -- pure, metadata-only locations worth considering for a controlled test.

This module deliberately stops before investigation.  It consumes the audited Close Calls detector and
the complete, pre-limit Context Tension evidence, then joins both evidence families at the recorded
response-token coordinate used by Execution Fork.  A returned breakpoint is a suggested test location,
never a diagnosis, error point, fragility claim, or automatic action.

No function here starts measurement, scores tokens, calls a model, starts a worker, creates/checks a
checkpoint, invokes Execution Fork, or plans a live rewind.  Missing evidence stays explicit and the
artifact contains no prompt, answer, source, or token text.
"""
from __future__ import annotations

import hashlib
import json
import math

from clozn.runs import close_calls
from clozn.runs import context_tension
from clozn.runs import influence_geometry as geometry


SCHEMA_VERSION = "clozn.suggested-breakpoints.v1"
DEFAULT_LIMIT = 12
MIN_LIMIT = 1
MAX_LIMIT = 50

_RANK_CLASS_ORDER = {
    "combined": 0,
    "meaningful_close_call": 1,
    "context_tension": 2,
    "close_call": 3,
}


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _int(value, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return int(value)


def _number(value, *, minimum: float = 0.0, maximum: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        return None
    if maximum is not None and result > maximum:
        return None
    return result


def _run_id(run: dict) -> str:
    value = str(run.get("id") or "") if isinstance(run, dict) else ""
    if not value:
        raise ValueError("run.id must be a non-empty string")
    return value


def _trace(run: dict) -> dict:
    return _dict(run.get("trace"))


def _coordinates(trace: dict) -> dict:
    out = {"kind": "recorded_response_token_boundary", "index_base": 0}
    tokens = trace.get("tokens")
    if isinstance(tokens, list):
        out.update({
            "start": 0,
            "end_exclusive": len(tokens),
            "recorded_token_count": len(tokens),
        })
    return out


def _coverage(trace: dict) -> dict:
    """Describe exactly how much of the parallel trace the existing detector could inspect."""
    tokens = trace.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        out = {"state": "unavailable", "reason": "no_trace_tokens"}
        if isinstance(tokens, list):
            out.update({"recorded_tokens": len(tokens), "analyzed_tokens": 0})
        return out

    recorded_tokens = len(tokens)
    confidence = trace.get("confidence")
    alternatives = trace.get("alternatives")
    if not isinstance(confidence, list):
        return {
            "state": "unavailable", "reason": "confidence_unavailable",
            "recorded_tokens": recorded_tokens, "analyzed_tokens": 0,
        }
    if not isinstance(alternatives, list):
        return {
            "state": "unavailable", "reason": "alternatives_unavailable",
            "recorded_tokens": recorded_tokens, "analyzed_tokens": 0,
        }

    parallel_count = min(recorded_tokens, len(confidence), len(alternatives))
    analyzed_tokens = 0
    for index in range(parallel_count):
        probability = _number(confidence[index], minimum=0.0, maximum=1.0)
        if (isinstance(tokens[index], str) and probability is not None
                and isinstance(alternatives[index], list)):
            analyzed_tokens += 1

    lengths_complete = len(confidence) == recorded_tokens and len(alternatives) == recorded_tokens
    entries_complete = analyzed_tokens == recorded_tokens
    if lengths_complete and entries_complete:
        state = "available"
    elif analyzed_tokens > 0:
        state = "partial"
    else:
        state = "unavailable"
    reason = None
    if not lengths_complete:
        reason = "parallel_trace_arrays_incomplete"
    elif not entries_complete:
        reason = "trace_entries_incomplete"
    out = {
        "state": state,
        "recorded_tokens": recorded_tokens,
        "analyzed_tokens": analyzed_tokens,
    }
    if reason is not None:
        out["reason"] = reason
    return out


def _answer_alignment(run: dict, trace: dict) -> tuple[dict, dict | None]:
    """Return the public alignment state and private exact token intervals."""
    response, response_reason = geometry.resolve_answer_text(run)
    if response is None:
        return {"state": "unavailable", "reason": response_reason}, None

    tokens = trace.get("tokens")
    if not isinstance(tokens, list):
        return {"state": "unavailable", "reason": "no_trace_tokens"}, None
    if any(not isinstance(piece, str) for piece in tokens):
        return {"state": "unavailable", "reason": "trace_response_mismatch"}, None
    if "".join(tokens) != response:
        return {"state": "unavailable", "reason": "trace_response_mismatch"}, None

    intervals: dict[int, tuple[int, int]] = {}
    cursor = 0
    for index, piece in enumerate(tokens):
        end = cursor + len(piece)
        intervals[index] = (cursor, end)
        cursor = end
    return {
        "state": "available",
        "basis": "recorded_answer",
        "unit": "unicode_code_points",
    }, {"intervals": intervals, "answer_length": cursor}


def _token_id(value) -> int | None:
    return _int(value, minimum=0)


def _piece(value) -> str:
    return str(value or "").strip()


def _alternative_metadata(alternatives, rival_piece: str) -> tuple[float | None, int | None]:
    if not isinstance(alternatives, list):
        return None, None
    for item in alternatives:
        if not isinstance(item, dict):
            continue
        piece = _piece(item.get("piece", item.get("text", "")))
        if piece != rival_piece:
            continue
        probability = _number(
            item.get("prob", item.get("confidence", item.get("conf"))),
            minimum=0.0, maximum=1.0,
        )
        token_id = _token_id(item.get("token_id", item.get("id")))
        return probability, token_id
    return None, None


def _emitted_token_id(trace: dict, index: int) -> int | None:
    token_ids = trace.get("token_ids")
    if isinstance(token_ids, list) and index < len(token_ids):
        token_id = _token_id(token_ids[index])
        if token_id is not None:
            return token_id
    steps = trace.get("steps")
    if isinstance(steps, list) and index < len(steps) and isinstance(steps[index], dict):
        return _token_id(steps[index].get("token_id", steps[index].get("id")))
    return None


def _close_call_reason(call: dict, trace: dict) -> dict:
    """Enrich an already-proven close call without reimplementing its truth condition."""
    index = _int(call.get("index"), minimum=0)
    top_piece = _piece(call.get("top"))
    alt_piece = _piece(call.get("alt"))
    emitted_piece = _piece(call.get("emitted"))
    top_probability = _number(call.get("top_prob"), minimum=0.0, maximum=1.0)
    alt_probability = _number(call.get("alt_prob"), minimum=0.0, maximum=1.0)

    if emitted_piece == top_piece:
        emitted_probability, rival_probability, rival_piece = (
            top_probability, alt_probability, alt_piece)
    elif emitted_piece == alt_piece:
        emitted_probability, rival_probability, rival_piece = (
            alt_probability, top_probability, top_piece)
    else:
        # The audited detector normally guarantees one of the two matches.  If a legacy trace has a
        # representation mismatch, keep the detector's recorded pair and omit only the ambiguous
        # emitted/rival probability fields rather than inventing a relationship.
        emitted_probability = None
        rival_probability = None
        rival_piece = alt_piece

    reason = {"type": "close_call"}
    if emitted_probability is not None:
        reason["emitted_probability"] = emitted_probability
    if rival_probability is not None:
        reason["rival_probability"] = rival_probability
    margin = _number(call.get("margin"), minimum=0.0)
    if margin is not None:
        reason["margin"] = margin
    reason["meaningful_heuristic"] = call.get("meaningful") is True

    if index is not None:
        emitted_id = _emitted_token_id(trace, index)
        if emitted_id is not None:
            reason["emitted_token_id"] = emitted_id
        alternatives = trace.get("alternatives")
        rival_probability_from_trace, rival_id = _alternative_metadata(alternatives[index], rival_piece) \
            if isinstance(alternatives, list) and index < len(alternatives) else (None, None)
        if "rival_probability" not in reason and rival_probability_from_trace is not None:
            reason["rival_probability"] = rival_probability_from_trace
        if rival_id is not None:
            reason["rival_token_id"] = rival_id
    return reason


def _tension_reason(groups: list[dict], span_by_id: dict[str, dict], tension_order: dict[str, int]) -> tuple[dict, int]:
    """Aggregate every pair attached to one breakpoint and retain Context Tension's pair order."""
    spans = []
    pairs = []
    seen_pairs = set()
    for group in groups:
        span_id = group.get("answer_span_id")
        if isinstance(span_id, str) and span_id not in spans:
            spans.append(span_id)
        for pair in group.get("tensions", []):
            tension_id = pair.get("tension_id") if isinstance(pair, dict) else None
            if not isinstance(tension_id, str) or tension_id in seen_pairs:
                continue
            seen_pairs.add(tension_id)
            pairs.append(pair)
    pairs.sort(key=lambda pair: tension_order.get(pair.get("tension_id"), 10**9))
    reason = {
        "type": "context_tension",
        "answer_span_ids": spans,
        "tension_pair_count": len(pairs),
        "distinct_source_span_count": len({
            side.get("source_span_id")
            for pair in pairs
            if isinstance(pair, dict)
            for side in (pair.get("supporting"), pair.get("suppressing"))
            if isinstance(side, dict) and isinstance(side.get("source_span_id"), str)
        }),
    }
    if len(spans) == 1:
        reason["answer_span_id"] = spans[0]
        interval = span_by_id.get(spans[0])
        if isinstance(interval, dict):
            reason["answer_interval"] = {
                "start": interval["start"], "end": interval["end"],
                "unit": "unicode_code_points", "interval": "half_open",
            }
    if pairs:
        strongest = pairs[0]
        supporting = _dict(strongest.get("supporting"))
        suppressing = _dict(strongest.get("suppressing"))
        strongest_pair = {
            "tension_id": strongest.get("tension_id"),
            "supporting_source_span_id": supporting.get("source_span_id"),
            "suppressing_source_span_id": suppressing.get("source_span_id"),
            "supporting_abs_delta_nats": supporting.get("abs_delta_nats"),
            "suppressing_abs_delta_nats": suppressing.get("abs_delta_nats"),
        }
        reason["strongest_pair"] = strongest_pair
    return reason, tension_order.get(pairs[0].get("tension_id"), 10**9) if pairs else 10**9


def _breakpoint_id(run_id: str, position: int) -> str:
    identity = {"run_id": run_id, "position": position}
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "breakpoint_" + hashlib.sha256(encoded).hexdigest()[:24]


def _rank_key(candidate: dict) -> tuple:
    rank_class = candidate["rank_class"]
    position = candidate["position"]
    margin = candidate.get("close_margin")
    margin = margin if isinstance(margin, (int, float)) else 1.0
    rival_probability = candidate.get("rival_probability")
    rival_probability = rival_probability if isinstance(rival_probability, (int, float)) else 0.0
    tension_order = candidate.get("tension_order", 10**9)
    if rank_class == "combined":
        return (
            _RANK_CLASS_ORDER[rank_class],
            0 if candidate.get("meaningful_close_call") else 1,
            margin,
            -rival_probability,
            tension_order,
            position,
        )
    if rank_class in {"meaningful_close_call", "close_call"}:
        return (_RANK_CLASS_ORDER[rank_class], margin, -rival_probability, position)
    return (_RANK_CLASS_ORDER[rank_class], tension_order, position)


def _analysis_state(close_evidence: dict, tension_evidence: dict, alignment: dict,
                    projection_failed: bool) -> str:
    close_state = close_evidence.get("state")
    close_usable = close_state in {"available", "partial"}
    close_partial = close_state == "partial"
    tension_state = tension_evidence.get("state")
    tension_summary = _dict(tension_evidence.get("summary"))
    tension_pairs = _int(tension_summary.get("tension_pairs"), minimum=0) or 0
    alignment_available = alignment.get("state") == "available"

    tension_usable = tension_state == "available" and (alignment_available or tension_pairs == 0)
    tension_failure = tension_state in {"unavailable", "error"} or (
        tension_state == "available" and tension_pairs > 0 and not alignment_available
    ) or projection_failed

    if not close_usable and not tension_usable:
        return "unavailable"
    if close_partial or tension_failure:
        return "partially_available"
    return "available"


def _public_tension_evidence(evidence: dict) -> dict:
    state = evidence.get("state")
    out = {"state": state if state in {"available", "not_measured", "unavailable", "error"}
            else "error"}
    if out["state"] == "available":
        summary = _dict(evidence.get("summary"))
        for key in ("answer_spans_examined", "answer_spans_with_tension", "tension_pairs"):
            value = _int(summary.get(key), minimum=0)
            if value is not None:
                out[key] = value
    else:
        reason = evidence.get("reason")
        if isinstance(reason, str) and reason:
            out["reason"] = reason
    return out


def build_suggested_breakpoints(run: dict, *, limit: int = DEFAULT_LIMIT,
                                privacy: str = "metadata_only") -> dict:
    """Build deterministic, metadata-only Suggested Breakpoints over one recorded run."""
    if privacy != "metadata_only":
        raise ValueError("privacy must be metadata_only")
    if not isinstance(limit, int) or isinstance(limit, bool) or not (MIN_LIMIT <= limit <= MAX_LIMIT):
        raise ValueError("limit must be between 1 and 50")

    run = run if isinstance(run, dict) else {}
    run_id = _run_id(run)
    trace = _trace(run)
    coordinates = _coordinates(trace)
    close_evidence = _coverage(trace)
    alignment, private_alignment = _answer_alignment(run, trace)

    # The existing detector owns the close-call truth condition.  This call is intentionally made once;
    # everything below only enriches and joins its returned indices.
    close_call_results = close_calls.close_calls(run)

    try:
        complete_tension = context_tension.collect_context_tension_evidence(run)
    except Exception:
        complete_tension = {
            "state": "error", "reason": "context_tension_contract_invalid",
            "answer_spans": [], "tensions": [], "summary": {},
        }
    public_tension = _public_tension_evidence(complete_tension)

    candidates: dict[int, dict] = {}

    def candidate_for(position: int) -> dict:
        candidate = candidates.get(position)
        if candidate is None:
            candidate = {
                "position": position, "reasons": [], "close_margin": None,
                "rival_probability": None, "meaningful_close_call": False,
                "tension_groups": [], "tension_order": 10**9,
            }
            candidates[position] = candidate
        return candidate

    intervals = private_alignment.get("intervals", {}) if private_alignment else {}
    for call in close_call_results:
        position = _int(call.get("index"), minimum=0)
        if position is None:
            continue
        candidate = candidate_for(position)
        reason = _close_call_reason(call, trace)
        candidate["reasons"].append(reason)
        candidate["meaningful_close_call"] = reason.get("meaningful_heuristic") is True
        candidate["close_margin"] = reason.get("margin")
        candidate["rival_probability"] = reason.get("rival_probability")
        interval = intervals.get(position)
        if isinstance(interval, tuple) and len(interval) == 2 and interval[1] > interval[0]:
            candidate["token_interval"] = interval

    tension_order = {
        pair.get("tension_id"): index
        for index, pair in enumerate(complete_tension.get("tensions", []))
        if isinstance(pair, dict) and isinstance(pair.get("tension_id"), str)
    }
    span_by_id = {
        item.get("answer_span_id"): item
        for item in complete_tension.get("answer_spans", [])
        if isinstance(item, dict) and isinstance(item.get("answer_span_id"), str)
    }
    tensions_by_span: dict[str, list[dict]] = {}
    for pair in complete_tension.get("tensions", []):
        if isinstance(pair, dict) and isinstance(pair.get("answer_span_id"), str):
            tensions_by_span.setdefault(pair["answer_span_id"], []).append(pair)

    projection_failed = False
    if complete_tension.get("state") == "available" and tensions_by_span:
        for answer_span_id, tensions in tensions_by_span.items():
            span = span_by_id.get(answer_span_id)
            if not isinstance(span, dict):
                projection_failed = True
                continue
            start = _int(span.get("start"), minimum=0)
            end = _int(span.get("end"), minimum=0)
            if start is None or end is None or end <= start:
                projection_failed = True
                continue

            overlap_positions = [
                position for position, interval in intervals.items()
                if isinstance(interval, tuple) and len(interval) == 2
                and interval[1] > interval[0] and interval[0] < end and interval[1] > start
                and position in candidates
            ]
            if overlap_positions:
                target_positions = sorted(overlap_positions)
            else:
                target_positions = [
                    position for position, interval in sorted(intervals.items())
                    if isinstance(interval, tuple) and len(interval) == 2
                    and interval[1] > interval[0] and interval[0] < end and interval[1] > start
                ][:1]
            if not target_positions:
                projection_failed = True
                continue
            group = {"answer_span_id": answer_span_id, "tensions": tensions}
            for position in target_positions:
                candidate_for(position)["tension_groups"].append(group)

    for candidate in candidates.values():
        if candidate["tension_groups"]:
            tension_reason, tension_order_value = _tension_reason(
                candidate["tension_groups"], span_by_id, tension_order,
            )
            candidate["reasons"].append(tension_reason)
            candidate["tension_order"] = tension_order_value
        has_close = any(reason.get("type") == "close_call" for reason in candidate["reasons"])
        has_tension = any(reason.get("type") == "context_tension" for reason in candidate["reasons"])
        if has_close and has_tension:
            candidate["rank_class"] = "combined"
        elif has_tension:
            candidate["rank_class"] = "context_tension"
        elif candidate["meaningful_close_call"]:
            candidate["rank_class"] = "meaningful_close_call"
        else:
            candidate["rank_class"] = "close_call"

    ordered = sorted(candidates.values(), key=_rank_key)
    analysis_state = _analysis_state(close_evidence, complete_tension, alignment, projection_failed)

    breakpoints = []
    for candidate in ordered[:limit]:
        position = candidate["position"]
        breakpoint = {
            "breakpoint_id": _breakpoint_id(run_id, position),
            "position": position,
            "placement": "exact_token_decision"
            if any(reason.get("type") == "close_call" for reason in candidate["reasons"])
            else "answer_span_entry_proxy",
            "rank_class": candidate["rank_class"],
            "reasons": candidate["reasons"],
        }
        interval = candidate.get("token_interval")
        if interval is None:
            interval = intervals.get(position)
        if isinstance(interval, tuple) and len(interval) == 2 and interval[1] > interval[0]:
            breakpoint["token_interval"] = {
                "start": interval[0], "end": interval[1],
                "unit": "unicode_code_points", "interval": "half_open",
            }
        breakpoints.append(breakpoint)

    class_counts = {name: 0 for name in _RANK_CLASS_ORDER}
    for candidate in ordered:
        class_counts[candidate["rank_class"]] += 1
    summary = {
        "candidate_state": (
            "detected" if ordered else "unavailable" if analysis_state == "unavailable" else "none_detected"
        ),
        "suggested_breakpoints": len(ordered),
        "returned_breakpoints": len(breakpoints),
        "combined_breakpoints": class_counts["combined"],
        "meaningful_close_call_breakpoints": class_counts["meaningful_close_call"],
        "context_tension_breakpoints": class_counts["context_tension"],
        "ordinary_close_call_breakpoints": class_counts["close_call"],
    }
    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "privacy": "metadata_only",
        "coordinates": coordinates,
        "analysis": {"state": analysis_state},
        "evidence": {
            "close_calls": {
                **close_evidence,
                **{
                    "thresholds": {
                        "margin": close_calls.MARGIN,
                        "min_runnerup": close_calls.MIN_RUNNERUP,
                    }
                },
            },
            "context_tension": public_tension,
            "answer_alignment": alignment,
        },
        "breakpoints": breakpoints,
        "summary": summary,
    }
    from clozn import schemas
    schemas.validate(document)
    return document


__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "MIN_LIMIT",
    "SCHEMA_VERSION",
    "build_suggested_breakpoints",
]
