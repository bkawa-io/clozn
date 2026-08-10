"""Attach exact Selection Reference targets to already-visible Turn Receipt findings.

This module is deliberately a decorator, not an evidence provider.  It never ranks findings, starts
analysis, or broadens the Receipt.  A target is emitted only when the Receipt item already identifies
enough canonical coordinates for the shared Selection Contract to validate them against the run.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from clozn.runs.selection_contract import normalize_selection, public_selection
from clozn.runs.selection_reference import encode_selection_reference


def _mapping(value) -> dict:
    return value if isinstance(value, dict) else {}


def _int(value, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return value


def _target_from_normalized(run: dict, normalized: dict) -> dict | None:
    """Build one compact target from the shared canonical selection and reference APIs."""
    try:
        selection = public_selection(normalized)
        encoded = encode_selection_reference(run, selection)
    except Exception:
        # A Receipt finding is still useful if its optional navigation target cannot be resolved.  The
        # caller intentionally receives no exception text: old artifacts can contain private literals.
        return None
    if encoded.get("state") != "resolved":
        return None
    return {
        "selection_ref": encoded["reference"],
        "selection": selection,
        "api_href": encoded["api_href"],
        "deep_link": encoded["deep_link"],
    }


def build_inspect_target(run: dict, selection: dict) -> dict | None:
    """Normalize and encode one optional Receipt inspection target, failing closed on any limitation."""
    try:
        normalized = normalize_selection(run, selection)
    except Exception:
        return None
    return _target_from_normalized(run, normalized)


def _attach_item_target(run: dict, item: dict, selection: dict) -> None:
    target = build_inspect_target(run, selection)
    if target is not None:
        item["inspect"] = target


def _source_selection(item: Mapping) -> dict | None:
    source_id = item.get("source_span_id")
    if not isinstance(source_id, str):
        return None
    answer_id = item.get("answer_span_id")
    if isinstance(answer_id, str):
        return {
            "kind": "context_span",
            "source_span_id": source_id,
            "answer_span_id": answer_id,
        }
    return {"kind": "context_span", "source_span_id": source_id}


def _answer_selection(value: Mapping | None) -> dict | None:
    value = value if isinstance(value, Mapping) else {}
    nested = value.get("answer_interval")
    nested = nested if isinstance(nested, Mapping) else value.get("interval")
    nested = nested if isinstance(nested, Mapping) else value.get("target")
    nested = nested if isinstance(nested, Mapping) else value
    start = _int(nested.get("start"))
    end = _int(nested.get("end"), minimum=1)
    if start is None or end is None or end <= start:
        return None
    return {"kind": "answer_span", "start": start, "end": end}


def _position_selection(value: Mapping | None) -> dict | None:
    value = value if isinstance(value, Mapping) else {}
    nested = value.get("breakpoint")
    nested = nested if isinstance(nested, Mapping) else value.get("evidence")
    nested = nested if isinstance(nested, Mapping) else value
    position = _int(nested.get("position"), minimum=0)
    if position is None:
        position = _int(nested.get("index"), minimum=0)
    if position is None:
        return None
    return {"kind": "response_token", "position": position}


def _tension_answer_selection(
    run: dict,
    receipt: Mapping,
    signal: Mapping,
    tension_artifact: Mapping | None = None,
) -> dict | None:
    """Resolve a common tension answer region from explicit metadata-only evidence.

    The preferred sources are already-projected Receipt metadata and the persisted tension artifact.
    We never select a supporting or suppressing source, and we only choose an answer region when the
    visible finding identifies exactly one unambiguous region.
    """
    for candidate in (
        signal,
        _mapping(receipt.get("context_tension")),
        tension_artifact if isinstance(tension_artifact, Mapping) else _mapping(run.get("context_tension")),
    ):
        selection = _answer_selection(candidate)
        if selection is not None:
            return selection

    tension_artifact = (
        tension_artifact if isinstance(tension_artifact, Mapping)
        else run.get("context_tension")
    )
    if not isinstance(tension_artifact, Mapping):
        return None
    tensions = tension_artifact.get("tensions")
    if not isinstance(tensions, list):
        return None
    answer_ids = sorted({
        item.get("answer_span_id") for item in tensions
        if isinstance(item, Mapping) and isinstance(item.get("answer_span_id"), str)
    })
    if len(answer_ids) != 1:
        return None
    answer_id = answer_ids[0]

    # Context Tension already carries the exact answer intervals it used.  Prefer those persisted
    # offsets over any derived fallback; this is a projection of native evidence, not new geometry.
    answer_spans = tension_artifact.get("answer_spans")
    if isinstance(answer_spans, list):
        matching = [
            item for item in answer_spans
            if isinstance(item, Mapping) and item.get("answer_span_id") == answer_id
        ]
        if len(matching) == 1:
            selection = _answer_selection(matching[0])
            if selection is not None:
                return selection

    # Persisted public Context Tension keeps the stable answer id but not its offsets.  Resolve those
    # offsets through the existing influence geometry projection; never search the response text.
    response = run.get("response")
    influence_map = run.get("influence_map")
    run_id = run.get("id")
    if not isinstance(response, str) or not isinstance(influence_map, Mapping) or not isinstance(run_id, str):
        return None
    try:
        from clozn.runs import influence_geometry as geometry
        resolved, _reason = geometry.resolve_geometry(run_id, dict(influence_map), response)
    except Exception:
        return None
    if resolved is None:
        return None
    native_id = next(
        (native for native, public in resolved.answer_address_by_id.items() if public == answer_id),
        None,
    )
    interval = resolved.answer_offsets.get(native_id) if native_id is not None else None
    if not isinstance(interval, tuple) or len(interval) != 2:
        return None
    start, end = interval
    if not isinstance(start, int) or not isinstance(end, int) or end <= start:
        return None
    return {"kind": "answer_span", "start": start, "end": end}


def _decorate_signals(run: dict, receipt: dict, *, tension_artifact: Mapping | None = None) -> None:
    signals = receipt.get("signals")
    if not isinstance(signals, list):
        return
    comparison = _mapping(receipt.get("comparison"))
    for signal in signals:
        if not isinstance(signal, dict) or "inspect" in signal:
            continue
        code = signal.get("code")
        selection = None
        if code == "context_tension_detected":
            selection = _tension_answer_selection(run, receipt, signal, tension_artifact)
        elif code == "first_divergence_available":
            selection = _position_selection(comparison.get("first_divergence"))
        elif code in {"suggested_breakpoint", "close_call", "breakpoint", "context_tension_breakpoint"}:
            selection = _position_selection(signal)
        elif "position" in signal or "breakpoint" in signal:
            # Supports future Receipt signal entries without creating a new ranking or signal rule.
            selection = _position_selection(signal)
        if selection is not None:
            _attach_item_target(run, signal, selection)


def _decorate_comparison(run: dict, receipt: dict) -> None:
    comparison = receipt.get("comparison")
    if not isinstance(comparison, dict) or comparison.get("state") != "available" or "inspect" in comparison:
        return
    selection = _position_selection(_mapping(comparison.get("first_divergence")))
    if selection is not None:
        _attach_item_target(run, comparison, selection)


def _decorate_rewind(run: dict, receipt: dict) -> None:
    rewind = receipt.get("rewind")
    if not isinstance(rewind, dict) or "inspect" in rewind:
        return
    # A generic rewind summary has no selection.  Only an already-specific boundary is eligible.
    selection = _position_selection(rewind)
    if selection is not None:
        _attach_item_target(run, rewind, selection)


def attach_inspection_targets(
    run: dict,
    receipt: dict,
    *,
    context_tension_artifact: Mapping | None = None,
) -> dict:
    """Return a detached Receipt with optional exact inspect targets on visible findings."""
    result = deepcopy(receipt) if isinstance(receipt, dict) else {}
    if not isinstance(run, dict) or not isinstance(receipt, dict):
        return result

    mattered = _mapping(result.get("what_mattered"))
    notable = mattered.get("notable_sources")
    if isinstance(notable, list):
        for item in notable:
            if not isinstance(item, dict) or "inspect" in item:
                continue
            selection = _source_selection(item)
            if selection is not None:
                _attach_item_target(run, item, selection)

    _decorate_signals(run, result, tension_artifact=context_tension_artifact)
    _decorate_comparison(run, result)
    _decorate_rewind(run, result)
    return result


__all__ = ["attach_inspection_targets", "build_inspect_target"]
