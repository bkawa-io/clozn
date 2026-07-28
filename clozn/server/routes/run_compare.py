"""server/routes/run_compare.py -- GET /runs/compare (agent roadmap feature 10, "What changed"): the
HTTP surface over clozn.analysis.run_diff.compare_runs(), mirroring clozn/cli/commands/compare_runs.py's
CLI and clozn/server/routes/diff.py's own POST /diff/runs lookup pattern (clozn.runs.store.get_run).

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4) -- no edit to clozn/server/app.py. This path
lives under the shared "/runs/" prefix that GET /runs/<id> (clozn/server/routes/runs.py's fallback) also
matches; the autoloader splices GET modules like this one BEFORE that fallback specifically so
"/runs/compare" is never swallowed as a wrong-shaped 200 from a run lookup for run id "compare" -- see
clozn/server/routes/_autoload.py's own docstring for why that ordering is semantic, not cosmetic.

Wire shape:
  GET /runs/compare?a=<run_id>&b=<run_id>[&replay=1]
      -> 200 run_diff.compare_runs(run_a, run_b) (a clozn.run-diff.v1 document), plus a "replay_plan" key
         (run_diff.plan_replay()'s MODEL-FREE proposal -- never executes anything) when ?replay=1 is set.
      -> 400 missing ?a=/?b=, or the comparison engine reported a non-ok result (malformed run content)
      -> 404 run(s) not found

POST /runs/compare/replay (the spec's named execution endpoint) is deliberately NOT implemented here:
actually running a proposed swap needs a live substrate/GPU (clozn.replay.replay.replay()) and is a
separately-scoped, deferred slice -- this module only ever reads already-recorded runs.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True


def try_get(h, p):
    if p != "/runs/compare":
        return False

    from urllib.parse import parse_qs, urlparse

    import clozn.runs.store as runlog
    from clozn.analysis import run_diff

    q = parse_qs(urlparse(h.path).query)
    rid_a, rid_b = (q.get("a") or [""])[0], (q.get("b") or [""])[0]
    if not rid_a or not rid_b:
        h._json(400, {"error": "need ?a=<run_id>&b=<run_id> -- two recorded run ids"})
        return True

    run_a, run_b = runlog.get_run(rid_a), runlog.get_run(rid_b)
    missing = [rid for rid, run in ((rid_a, run_a), (rid_b, run_b)) if run is None]
    if missing:
        h._json(404, {"error": "run(s) not found: " + ", ".join(missing), "missing": missing})
        return True

    result = run_diff.compare_runs(run_a, run_b)
    if not result.get("ok"):
        h._json(400, {"error": result.get("error") or "comparison failed for an unknown reason"})
        return True

    if (q.get("replay") or [""])[0] in ("1", "true", "yes"):
        result = dict(result)
        result["replay_plan"] = run_diff.plan_replay(run_a, run_b, result)

    h._json(200, result)
    return True
