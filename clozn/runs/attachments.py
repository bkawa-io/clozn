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
