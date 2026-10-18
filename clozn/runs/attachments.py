"""Atomic updates to fields attached to an existing SQLite run document."""
from __future__ import annotations

from . import store


def attach_performance_phase(rid: str, phase: dict) -> bool:
    """Append one measured gateway phase after a run was initially persisted.

    JSON response serialization necessarily happens after ``record()`` assigns the run id. This narrow
    attachment keeps that measurement real without moving run persistence behind network delivery.
    """
    if not isinstance(phase, dict) or not isinstance(phase.get("name"), str):
        return False
    duration_ns = phase.get("duration_ns")
    if not isinstance(duration_ns, int) or isinstance(duration_ns, bool) or duration_ns < 0:
        return False
    try:
        rec = store.get_run(rid)
        if rec is None:
            return False
        meta = dict(rec.get("meta") or {})
        from .perf_spans import merge_timing_documents, timing_document
        meta["gateway_timing"] = merge_timing_documents(
            meta.get("gateway_timing"), timing_document([phase])
        )
        rec["meta"] = meta
        return store.replace_run(rec)
    except Exception:
        return False
