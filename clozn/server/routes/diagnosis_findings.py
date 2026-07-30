"""GET /runs/<id>/diagnosis-findings -- D1's `clozn.diagnosis-findings.v1` rule-engine findings AND D2's
`clozn.diagnosis-narrative.v1` plain-language narrative, together in one response: "the API returns both
structured findings and rendered prose" (D2's own spec, verbatim).

DISTINCT FROM, NEVER A REPLACEMENT FOR, `GET /runs/<id>/diagnosis`
------------------------------------------------------------------------
That existing route (clozn/server/routes/runs.py, hand-wired, untouched by this file) serves
`clozn.run_diagnosis.v1` -- the OLD why-slow/why-cut-off vocabulary (`clozn/runs/diagnosis.py`). This
route serves a DIFFERENT artifact family entirely (`clozn.diagnosis-findings.v1` / `clozn.diagnosis-
narrative.v1`, D1/D2) under a DIFFERENT path (`/diagnosis-findings`, not `/diagnosis` -- the two suffixes
do not collide: `"...run_id/diagnosis-findings".endswith("/diagnosis")` is false). Neither route reads or
supersedes the other.

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4) -- no edit to clozn/server/app.py. This path
lives under the shared "/runs/" prefix GET /runs/<id> (clozn/server/routes/runs.py's own fallback) also
matches; the autoloader splices GET modules like this one BEFORE that fallback specifically so this route
is never swallowed as a wrong-shaped 200 from a run lookup for run id "<id>/diagnosis-findings" -- see
clozn/server/routes/_autoload.py's own docstring for why that ordering is semantic, not cosmetic.

SELF-CONTAINED: wiring this into `clozn.runs.investigation` is a later slice's job, not this route's --
this module only ever imports `clozn.runs.store`, `clozn.runs.diagnosis_rules`, and
`clozn.runs.diagnosis_narratives`.

Wire shape:
  GET /runs/<id>/diagnosis-findings[?compare=<run_id>][&suppress=R03,R07]
      -> 200 {"findings": <clozn.diagnosis-findings.v1>, "narrative": <clozn.diagnosis-narrative.v1>}
      -> 404 the primary run (or, when supplied, the comparison run) was not found
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/diagnosis-findings"


def try_get(h, p):
    if not (p.startswith("/runs/") and p.endswith(_SUFFIX)):
        return False

    from urllib.parse import parse_qs, urlparse

    import clozn.runs.store as runlog
    from clozn.runs import diagnosis_narratives, diagnosis_rules

    run_id = p[len("/runs/"):-len(_SUFFIX)]
    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    query = parse_qs(urlparse(h.path).query)
    comparison_run_id = (query.get("compare") or [""])[0]
    comparison_run = None
    if comparison_run_id:
        comparison_run = runlog.get_run(comparison_run_id)
        if comparison_run is None:
            h._json(404, {"error": f"comparison run not found: {comparison_run_id}"})
            return True

    suppressed_raw = (query.get("suppress") or [""])[0]
    suppressed_rule_ids = [item.strip() for item in suppressed_raw.split(",") if item.strip()]

    findings = diagnosis_rules.evaluate(run, comparison_run=comparison_run,
                                        suppressed_rule_ids=suppressed_rule_ids)
    narrative = diagnosis_narratives.narrate(run, comparison_run=comparison_run, findings=findings)

    h._json(200, {"findings": findings, "narrative": narrative})
    return True
