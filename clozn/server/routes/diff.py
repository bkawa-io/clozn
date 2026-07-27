"""The model-diff surface: POST /diff/runs (two recorded runs, token by token --
clozn.analysis.model_diff, where ALL the comparison logic and its honesty caveats live) and
GET /diff/suites (two saved CI suite results -- a thin wrapper over clozn.testkit.ci.diff_suites,
no new diff logic here).

Wire shapes:
  POST /diff/runs   {"a": "<run_id>", "b": "<run_id>"}
      -> 200 model_diff.diff_runs(run_a, run_b)  (see that module's docstring for the full shape:
         prompts_match/warn, common_prefix_len, first_divergence, positions[<=200], summary with
         b_was_alternative_in_a + char_similarity labeled "surface similarity — wording, not meaning",
         a/b model+quant identity, caveat)
      -> 400 missing ids in the body; 404 run(s) not found (with "missing": [...])
  GET  /diff/suites?a=<name>&b=<name>[&dir=<path>]
      -> 200 {"a", "b", "order", regressions, fixed, drift, receipt_changes, new_cases,
              removed_cases, prev_timestamp, curr_timestamp, "text": Diff.render()}
         `a`/`b` name files ci.save_result wrote (".json" optional; basenames only -- a name is never
         allowed to path-escape the directory). `dir` overrides the default ~/.clozn/ci/ location
         (ci.DEFAULT_CI_DIR) -- the injectable-directory knob save_result itself has, exposed on the
         wire so tests (and a caller with a non-default CI dir) can point the diff anywhere.
         `a` is diffed as prev/baseline, `b` as curr -- ci.diff_suites(prev, curr)'s own order.
      -> 400 missing/unloadable params; 404 suite file(s) not found

Registered in clozn/server/app.py: imported as `_diff_routes` and placed in BOTH `_GET_ROUTES` (before
`_runs_fallback_routes`) and `_POST_ROUTES` (POST falls through to the generic SUB.handle after every
registered family, and none of them claims "/diff/..."). Live surface: Studio's Atlas panel
(studio-frontend/src/features/compare/Compare.tsx) drives POST /diff/runs; GET /diff/suites
has no Studio caller yet -- it's exercised via clozn/testkit/ci.py and its own tests.

Unlike health.py, this module deliberately does NOT `from clozn.server import app as ctx`: it reads no
shared server state (no SUB/ENGINE/SUBNAME) -- only the run journal and saved CI results on disk -- so
it can be imported and unit-tested (clozn/analysis/test_model_diff.py) without touching the app module.
"""
from __future__ import annotations

import os


def _resolve_suite(directory: str, name: str) -> str | None:
    """`name` -> the saved-suite path inside `directory`, or None. Basename-only (a name can never
    traverse out of the CI directory); tries the name verbatim, then with ".json" appended, so callers
    may pass save_result's filename with or without its extension."""
    base = os.path.basename(str(name or ""))
    if not base:
        return None
    for candidate in (base, base + ".json"):
        path = os.path.join(directory, candidate)
        if os.path.isfile(path):
            return path
    return None


def try_get(h, p):
    if p == "/diff/suites":
        from dataclasses import asdict
        from urllib.parse import urlparse, parse_qs
        from clozn.testkit import ci

        q = parse_qs(urlparse(h.path).query)
        name_a, name_b = (q.get("a") or [""])[0], (q.get("b") or [""])[0]
        if not name_a or not name_b:
            h._json(400, {"error": "need ?a=<saved suite>&b=<saved suite> -- filenames ci.save_result "
                                   "wrote under the CI dir ('.json' optional); a=prev/baseline, b=curr"})
            return True
        directory = (q.get("dir") or [ci.DEFAULT_CI_DIR])[0]
        path_a, path_b = _resolve_suite(directory, name_a), _resolve_suite(directory, name_b)
        missing = [n for n, path in ((name_a, path_a), (name_b, path_b)) if path is None]
        if missing:
            h._json(404, {"error": "saved suite(s) not found in " + directory + ": " + ", ".join(missing),
                          "missing": missing})
            return True
        try:
            prev, curr = ci.load_result(path_a), ci.load_result(path_b)
        except Exception as e:
            h._json(400, {"error": f"could not load a saved suite: {type(e).__name__}: {e}"})
            return True
        d = ci.diff_suites(prev, curr)                    # the diff logic, reused verbatim -- none here
        h._json(200, {"a": name_a, "b": name_b, "order": "a = prev/baseline, b = curr",
                      **asdict(d), "text": d.render()})
        return True
    return False


def try_post(h, p, body):
    if p == "/diff/runs":
        rid_a, rid_b = body.get("a"), body.get("b")
        if not rid_a or not rid_b:
            h._json(400, {"error": 'need {"a": "<run_id>", "b": "<run_id>"} -- two recorded run ids '
                                   "(same prompt for a controlled comparison; a mismatch is warned, "
                                   "not refused)"})
            return True
        import clozn.runs.store as runlog
        run_a, run_b = runlog.get_run(str(rid_a)), runlog.get_run(str(rid_b))
        missing = [str(rid) for rid, run in ((rid_a, run_a), (rid_b, run_b)) if run is None]
        if missing:
            h._json(404, {"error": "run(s) not found: " + ", ".join(missing), "missing": missing})
            return True
        from clozn.analysis import model_diff
        h._json(200, model_diff.diff_runs(run_a, run_b))
        return True
    return False
