"""Exact-token parent/child geometry for the parent-anchored KV reuse study.

This module is deliberately model-free.  The reducer supplies the semantic
parent recorded at dispatch time; this layer only compares the worker's exact
template token IDs and computes structural row models.  The row models are
ceilings, not measurements of wall-clock speed or native KV behavior.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any


SCHEMA = "clozn.parent-anchor-geometry.v0"
STAGES = ("coarse", "refine", "inclusion")


def exact_lcp(parent_token_ids: Sequence[int], child_token_ids: Sequence[int]) -> int:
    """Return the exact token-prefix length shared by parent and child."""
    limit = min(len(parent_token_ids), len(child_token_ids))
    index = 0
    while index < limit and parent_token_ids[index] == child_token_ids[index]:
        index += 1
    return index


def _percent(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(100.0 * float(numerator) / float(denominator), 6)


def _distribution(values: Iterable[int]) -> dict[str, int | None]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"min": None, "median": None, "p90": None, "max": None}
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    p90 = ordered[max(0, min(len(ordered) - 1, int((len(ordered) - 1) * 0.9 + 0.999999)))]
    return {
        "min": ordered[0],
        "median": median,
        "p90": p90,
        "max": ordered[-1],
    }


def _fraction_distribution(values: Iterable[float]) -> dict[str, float | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"min": None, "median": None, "p90": None, "max": None}
    middle = len(ordered) // 2
    median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
    p90 = ordered[max(0, min(len(ordered) - 1, int((len(ordered) - 1) * 0.9 + 0.999999)))]
    return {
        "min": round(ordered[0], 6),
        "median": round(median, 6),
        "p90": round(p90, 6),
        "max": round(ordered[-1], 6),
    }


def _ids(value: Iterable[Any], *, name: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of IDs")
    try:
        result = tuple(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of IDs") from exc
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def build_probe_row(
    *,
    ordinal: int,
    stage: str,
    batch_id: int,
    parent_source_ids: Iterable[Any],
    child_source_ids: Iterable[Any],
    parent_token_ids: Sequence[int],
    child_token_ids: Sequence[int],
    preserved: bool,
    accepted_as_best: bool,
) -> dict[str, Any]:
    """Build one prompt-geometry row without retaining prompt text."""
    parent_ids = _ids(parent_source_ids, name="parent_source_ids")
    child_ids = _ids(child_source_ids, name="child_source_ids")
    if not set(child_ids).issubset(parent_ids):
        raise ValueError("child_source_ids must be a direct retained-source subset of parent_source_ids")
    parent_tokens = tuple(int(value) for value in parent_token_ids)
    child_tokens = tuple(int(value) for value in child_token_ids)
    lcp = exact_lcp(parent_tokens, child_tokens)
    first_changed = lcp if lcp < min(len(parent_tokens), len(child_tokens)) else None
    suffix = len(child_tokens) - lcp
    return {
        "probe_ordinal": int(ordinal),
        "stage": str(stage),
        "batch_id": int(batch_id),
        "parent_source_ids": list(parent_ids),
        "child_source_ids": list(child_ids),
        "parent_prompt_tokens": len(parent_tokens),
        "child_prompt_tokens": len(child_tokens),
        "exact_lcp_tokens": lcp,
        "reusable_prefix_rows": lcp,
        "required_child_suffix_rows": suffix,
        "lcp_fraction_of_child": round(lcp / len(child_tokens), 6) if child_tokens else None,
        "first_changed_token_index": first_changed,
        "preserved": bool(preserved),
        "accepted_as_best": bool(accepted_as_best),
    }


def _row_model(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "child_count": 0,
            "parent_prompt_tokens": None,
            "naive_logical_child_rows": 0,
            "request_local_parent_ideal_rows": 0,
            "persistent_parent_ideal_rows": 0,
            "total_lcp_rows": 0,
            "total_suffix_rows": 0,
            "request_local_reduction_percent": None,
            "persistent_reduction_percent": None,
        }
    parent_ids = tuple(rows[0]["parent_source_ids"])
    parent_tokens = int(rows[0]["parent_prompt_tokens"])
    if any(tuple(row["parent_source_ids"]) != parent_ids for row in rows):
        raise ValueError("all children in a search batch must share one semantic parent")
    if any(int(row["parent_prompt_tokens"]) != parent_tokens for row in rows):
        raise ValueError("all children in a search batch must share one parent prompt length")
    logical = sum(int(row["child_prompt_tokens"]) for row in rows)
    lcp = sum(int(row["reusable_prefix_rows"]) for row in rows)
    suffix = sum(int(row["required_child_suffix_rows"]) for row in rows)
    request_local = parent_tokens + suffix
    persistent = suffix
    return {
        "child_count": len(rows),
        "parent_prompt_tokens": parent_tokens,
        "naive_logical_child_rows": logical,
        "request_local_parent_ideal_rows": request_local,
        "persistent_parent_ideal_rows": persistent,
        "total_lcp_rows": lcp,
        "total_suffix_rows": suffix,
        "request_local_reduction_percent": _percent(logical - request_local, logical),
        "persistent_reduction_percent": _percent(logical - persistent, logical),
    }


def aggregate_batches(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the three structural row models for each reducer batch."""
    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(int(row["batch_id"]), []).append(row)
    output = []
    for batch_id in sorted(grouped):
        batch_rows = grouped[batch_id]
        model = _row_model(batch_rows)
        output.append({
            "batch_id": batch_id,
            "stage": batch_rows[0]["stage"],
            "parent_source_ids": list(batch_rows[0]["parent_source_ids"]),
            "probe_ordinals": [int(row["probe_ordinal"]) for row in batch_rows],
            "row_models": model,
        })
    return output


def aggregate_case(
    rows: Sequence[Mapping[str, Any]],
    *,
    native_physical_rows: int | None = None,
    native_metrics_status: str = "not_measured",
) -> dict[str, Any]:
    """Compute case totals and stage-separated LCP distributions."""
    batches = aggregate_batches(rows)
    logical = sum(int(row["child_prompt_tokens"]) for row in rows)
    lcp = sum(int(row["reusable_prefix_rows"]) for row in rows)
    suffix = sum(int(row["required_child_suffix_rows"]) for row in rows)
    request_local = sum(int(batch["row_models"]["request_local_parent_ideal_rows"]) for batch in batches)
    persistent = sum(int(batch["row_models"]["persistent_parent_ideal_rows"]) for batch in batches)
    by_stage: dict[str, Any] = {}
    for stage in STAGES:
        stage_rows = [row for row in rows if row["stage"] == stage]
        by_stage[stage] = {
            "probe_count": len(stage_rows),
            "lcp_distribution_tokens": _distribution(
                int(row["exact_lcp_tokens"]) for row in stage_rows
            ),
            "lcp_fraction_distribution": _fraction_distribution(
                float(row["lcp_fraction_of_child"])
                for row in stage_rows
                if row["lcp_fraction_of_child"] is not None
            ),
        }
    return {
        "search_batch_count": len(batches),
        "probe_count": len(rows),
        "logical_child_prompt_rows": logical,
        "total_lcp_rows": lcp,
        "total_suffix_rows": suffix,
        "request_local_parent_ideal_rows": request_local,
        "persistent_parent_ideal_rows": persistent,
        "request_local_reduction_percent": _percent(logical - request_local, logical),
        "persistent_reduction_percent": _percent(logical - persistent, logical),
        "lcp_distribution_tokens": _distribution(
            int(row["exact_lcp_tokens"]) for row in rows
        ),
        "lcp_fraction_distribution": _fraction_distribution(
            float(row["lcp_fraction_of_child"])
            for row in rows
            if row["lcp_fraction_of_child"] is not None
        ),
        "by_stage": by_stage,
        "current_native_physical_rows": native_physical_rows,
        "current_native_metrics_status": native_metrics_status,
    }


__all__ = [
    "SCHEMA",
    "STAGES",
    "aggregate_batches",
    "aggregate_case",
    "build_probe_row",
    "exact_lcp",
]
