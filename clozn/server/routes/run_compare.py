"""GET-only comparison for an original run and its saved reduced child."""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True


def try_get(h, p):
    if p != "/runs/compare":
        return False

    from urllib.parse import parse_qs, urlparse

    import clozn.runs.store as runlog
    from clozn.analysis.reduction_compare import compare_reduction

    q = parse_qs(urlparse(h.path).query)
    rid_a, rid_b = (q.get("a") or [""])[0], (q.get("b") or [""])[0]
    if not rid_a or not rid_b:
        h._json(400, {"error": "need ?a=<parent_run_id>&b=<reduced_child_run_id>",
                      "code": "comparison_pair_required"})
        return True

    run_a = runlog.get_run(rid_a)
    run_b = runlog.get_run(rid_b)
    missing = [rid for rid, run in ((rid_a, run_a), (rid_b, run_b)) if run is None]
    if missing:
        h._json(404, {"error": "run(s) not found: " + ", ".join(missing), "missing": missing})
        return True

    result = compare_reduction(run_a, run_b)
    if not result.get("ok"):
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        h._json(409, {"error": error.get("message") or "comparison is not a saved reduction",
                      "code": error.get("code") or "not_reduction_child"})
        return True

    h._json(200, result)
    return True
