"""GET /runs/<id>/performance -- the clozn.performance-trace.v1 report (phases/metrics/rule-engine
diagnoses) for one recorded run. Mirrors clozn/server/routes/runs.py's existing GET /runs/<id>/diagnosis
branch exactly (same 404-on-missing-run shape, same "fetch the run, hand it to a pure function, return
the JSON" body) -- registered here instead of editing that shared dispatch file, via the route-autoload
seam (docs/SEAMS.md Seam 4) that landed after routes/runs.py's own diagnosis branch was written by hand.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True


def try_get(h, p):
    if p.startswith("/runs/") and p.endswith("/performance"):
        import clozn.runs.store as runlog

        rid = p[len("/runs/"):-len("/performance")]
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.runs.perf_diagnosis import build_performance_report

        h._json(200, build_performance_report(run, related_runs=runlog.iter_runs(limit=200)))
        return True
    return False
