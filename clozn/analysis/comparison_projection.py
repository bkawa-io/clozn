"""Compact wire projections over one already-computed canonical run diff.

Features that materialize two Runs and want a small comparison surface share this
projection instead of inventing a second token diff.  It never computes a diff of
its own beyond calling :func:`clozn.analysis.model_diff.diff_runs` exactly once.

Observation-first features (Branch Fan) do not use this module: they have no
second Run to compare and must not create one merely to reach ``diff_runs``.  See
:mod:`clozn.experiments.observation_comparison` for the Run-free projection.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy


def comparison_projection_from_diff(parent: Mapping, child: Mapping, diff: Mapping) -> dict:
    """Project only values returned by an already-computed canonical run diff."""
    view = diff.get("first_divergence_view")
    if not isinstance(view, Mapping):
        view = {"schema_version": "clozn.first-divergence-view.v1", "state": "trace_unavailable"}
    if not diff.get("trace_available"):
        return {
            "state": "trace_unavailable",
            "first_divergence_view": deepcopy(dict(view)),
        }
    out = {"state": "available", "first_divergence_view": deepcopy(dict(view))}
    for key in ("common_prefix_len", "first_divergence"):
        if key in diff:
            out[key] = deepcopy(diff[key])
    summary = diff.get("summary")
    if isinstance(summary, Mapping):
        value = summary.get("char_similarity")
        label = summary.get("char_similarity_label")
        if value is not None or label is not None:
            surface = {}
            if value is not None:
                surface["value"] = deepcopy(value)
            if label is not None:
                surface["label"] = deepcopy(label)
            out["surface_similarity"] = surface
    return out


def comparison_projection(parent: Mapping, child: Mapping) -> dict:
    """Compute and project one canonical run diff."""
    from clozn.analysis.model_diff import diff_runs

    return comparison_projection_from_diff(parent, child, diff_runs(parent, child))


__all__ = ["comparison_projection", "comparison_projection_from_diff"]
