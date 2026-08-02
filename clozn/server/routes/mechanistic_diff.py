"""HTTP product surface for cross-model mechanistic divergence artifacts.

The token workbench action remains the primary Studio entry point.  This route adds the explicit
comparison contract from the roadmap:

    POST /runs/compare/mechanistic   {"a": run_id, "b": reference_run_id, "index": N, ...}
    GET  /mechanistic-diffs/<id>      where <id> is the returned cache/artifact identity

Execution and caching are delegated to the workbench action implementation.  The GET endpoint is a
small index over persisted run action entries, so it never starts a model or recomputes a capture.
"""
from __future__ import annotations

import re

CLOZN_ROUTE_AUTOLOAD = True
_COMPARE_PATH = "/runs/compare/mechanistic"
_DIFF_PREFIX = "/mechanistic-diffs/"
_CACHE_ID = re.compile(r"^[0-9a-f]{64}$")


def try_get(h, p):
    if not p.startswith(_DIFF_PREFIX):
        return False
    artifact_id = p[len(_DIFF_PREFIX):]
    if not _CACHE_ID.fullmatch(artifact_id):
        h._json(404, {"error": "mechanistic diff not found"})
        return True

    import clozn.runs.store as runlog

    for run in runlog.iter_runs(limit=2000):
        for entry in run.get("token_workbench_actions") or []:
            if not isinstance(entry, dict) or entry.get("action") != "mechanistic_diff":
                continue
            if entry.get("cache_key") != artifact_id:
                continue
            artifact = entry.get("result")
            if not isinstance(artifact, dict) or artifact.get("schema_version") != "clozn.mechanistic-diff.v1":
                h._json(500, {"error": "stored mechanistic diff failed its artifact contract",
                              "code": "mechanistic_diff_contract_invalid"})
                return True
            h._json(200, {
                "id": artifact_id,
                "run_id": run.get("id"),
                "index": entry.get("index"),
                "artifact": artifact,
            })
            return True
    h._json(404, {"error": "mechanistic diff not found"})
    return True


def try_post(h, p, body):
    if p != _COMPARE_PATH:
        return False
    body = body if isinstance(body, dict) else {}
    anchor_id = body.get("a")
    reference_id = body.get("b")
    if not isinstance(anchor_id, str) or not anchor_id or not isinstance(reference_id, str) or not reference_id:
        h._json(400, {"error": "body.a and body.b must name two recorded run IDs"})
        return True
    index = body.get("index")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        h._json(400, {"error": "body.index must be a non-negative integer"})
        return True

    import clozn.runs.store as runlog
    from clozn.server.routes.token_workbench_actions import _mechanistic_diff_action

    anchor = runlog.get_run(anchor_id)
    reference = runlog.get_run(reference_id)
    missing = [rid for rid, run in ((anchor_id, anchor), (reference_id, reference)) if run is None]
    if missing:
        h._json(404, {"error": "run(s) not found: " + ", ".join(missing), "missing": missing})
        return True
    action_body = dict(body)
    action_body["reference_run_id"] = reference_id
    # The action route performs the authoritative pair gate, exact-evidence checks, managed-router
    # selection, cache lookup, and job admission. Its response now includes mechanistic_diff_id.
    return _mechanistic_diff_action(h, anchor, index, action_body)
