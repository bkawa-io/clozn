"""Rebuildable trend views over authoritative experiment-result artifacts."""
from __future__ import annotations

from collections import defaultdict
import copy
from typing import Iterable

from clozn.experiments import suite


def _identity_values(result: dict) -> dict:
    fields = ("model_sha256", "engine_build", "template_fingerprint", "adapter_sha256")
    values: dict[str, set[str]] = {field: set() for field in fields}
    for cell in result.get("cells") or []:
        run = cell.get("run")
        identity = run.get("identity") if isinstance(run, dict) else None
        if not isinstance(identity, dict):
            continue
        ext = identity.get("ext") if isinstance(identity.get("ext"), dict) else {}
        for field in fields:
            value = identity.get(field)
            if value is None:
                value = ext.get(field)
            if isinstance(value, str) and value:
                values[field].add(value)
    return {field: sorted(items) for field, items in values.items() if items}


def _instability(result: dict) -> dict:
    observed: dict[tuple[str, str, str], set[str]] = defaultdict(set)
    for cell in result.get("cells") or []:
        key = (cell.get("suite"), cell.get("case"), cell.get("variant"))
        observed[key].add(cell.get("status"))
    unstable = [
        {"suite": key[0], "case": key[1], "variant": key[2], "statuses": sorted(statuses)}
        for key, statuses in observed.items() if len(statuses) > 1
    ]
    unstable.sort(key=lambda row: (row["suite"], row["case"], row["variant"]))
    return {"coordinate_count": len(unstable), "coordinates": unstable}


def _comparison_counts(result: dict) -> dict:
    out = {}
    for comparison in result.get("summary", {}).get("comparisons") or []:
        out[comparison.get("variant")] = {
            field: len(comparison.get(field) or [])
            for field in (
                "target_gains", "target_regressions", "guard_regressions", "guard_fixes",
                "changed_unscored",
            )
        }
    return out


def trend_point(result: dict) -> dict:
    """Reduce one validated artifact without inventing absent identity or provenance."""
    point = {
        "experiment_id": result.get("experiment_id"),
        "name": result.get("name"),
        "created_at": result.get("created_at"),
        "suite_fingerprint": suite.result_fingerprint(result),
        # Identity precedes outcome fields deliberately: consumers can surface drift before metrics.
        "identity": _identity_values(result),
        "baseline_variant": result.get("summary", {}).get("baseline_variant"),
        "aggregates": copy.deepcopy(result.get("summary", {}).get("aggregates") or {}),
        "comparison_counts": _comparison_counts(result),
        "error_cells": sum(cell.get("status") == "error" for cell in result.get("cells") or []),
        "replicate_instability": _instability(result),
    }
    for field in ("vcs", "artifact_provenance"):
        if field in result:
            point[field] = copy.deepcopy(result[field])
    return point


def build_trend_index(results: Iterable[dict]) -> dict:
    """Group points by exact (algorithm, digest); incompatible suites never share a group."""
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for result in results:
        fingerprint = suite.result_fingerprint(result)
        groups[(fingerprint["algorithm"], fingerprint["sha256"])].append(trend_point(result))
    payload = []
    for (algorithm, digest), points in sorted(groups.items()):
        points.sort(key=lambda point: (point.get("created_at") or "", point.get("experiment_id") or ""))
        payload.append({
            "suite_fingerprint": {"algorithm": algorithm, "sha256": digest},
            "points": points,
        })
    return {"schema_version": "clozn.experiment-trends.v1", "groups": payload}


def select_compatible(index: dict, fingerprint: dict) -> dict:
    matches = [
        group for group in index.get("groups") or []
        if group.get("suite_fingerprint") == fingerprint
    ]
    return {
        "schema_version": index.get("schema_version"),
        "suite_fingerprint": copy.deepcopy(fingerprint),
        "points": matches[0]["points"] if matches else [],
    }


__all__ = ["build_trend_index", "select_compatible", "trend_point"]
