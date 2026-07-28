"""Atomic updates to fields attached to an existing SQLite run document."""
from __future__ import annotations

from . import store


def update_tiny_tests(rid: str, tiny_tests: list) -> bool:
    rec = store.get_run(rid)
    if rec is None:
        return False
    rec["tiny_tests"] = list(tiny_tests) if isinstance(tiny_tests, list) else []
    return store.replace_run(rec)


def attach_policy_verdict(rid: str, policy: dict) -> bool:
    """Attach the selective-generation policy's full verdict (band/score/thresholds/calibration
    provenance -- clozn.server.generation_gateway.policy_meta_for_run) onto an already-logged run's
    `meta.clozn_policy`, after the fact.

    Why after the fact: the verdict is derived from THIS reply's own trace, but the run must already be
    logged (id assigned, trace durably stored) before that trace can be scored against the saved
    calibration -- so it can never ride the same `record()` call the reply's own text does. Same
    read-modify-atomically-replace shape as `update_tiny_tests` above.

    Returns False (never raises) on a missing run, a malformed `policy`, or any store hiccup -- callers
    must treat that exactly like "never called": the run's meta simply carries no clozn_policy key,
    which reads honestly as "not computed for this run," never a fabricated 'answer'."""
    if not isinstance(policy, dict) or not policy:
        return False
    try:
        rec = store.get_run(rid)
        if rec is None:
            return False
        meta = dict(rec.get("meta") or {})
        meta["clozn_policy"] = dict(policy)
        rec["meta"] = meta
        return store.replace_run(rec)
    except Exception:
        return False


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
