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

  POST /runs/compare/test
      -> a bounded clozn.run-change-test.v1 artifact. `plan`/`dry_run` is model-free; live execution
         requires the gateway's active substrate and persists each control/treatment arm as a child run.
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
    against = (q.get("against") or [""])[0]
    pinned = (q.get("reference") or [""])[0]
    include_children = (q.get("include_child_runs") or [""])[0] in ("1", "true", "yes")
    if not rid_b:
        h._json(400, {"error": "need ?b=<candidate_run_id> and either ?a=<reference_run_id> "
                               "or ?against=previous_compatible|same_session|same_client|same_task"})
        return True

    run_b = runlog.get_run(rid_b)
    selection = None
    if run_b is not None and not rid_a:
        mode = "pinned" if pinned else against
        selected = run_diff.select_reference_run(
            run_b, runlog.iter_runs(), mode=mode, reference_run_id=pinned or None,
            include_child_runs=include_children,
        )
        if not selected.get("ok"):
            h._json(400, {"error": selected.get("error") or "comparison selection failed"})
            return True
        run_a, selection = selected["run"], selected["selection"]
        rid_a = run_a["id"]
    else:
        run_a = runlog.get_run(rid_a)
    missing = [rid for rid, run in ((rid_a, run_a), (rid_b, run_b)) if run is None]
    if missing:
        h._json(404, {"error": "run(s) not found: " + ", ".join(missing), "missing": missing})
        return True

    result = run_diff.compare_runs(run_a, run_b, selection=selection)
    if not result.get("ok"):
        h._json(400, {"error": result.get("error") or "comparison failed for an unknown reason"})
        return True

    if (q.get("replay") or [""])[0] in ("1", "true", "yes"):
        result = dict(result)
        result["replay_plan"] = run_diff.plan_replay(run_a, run_b, result)

    h._json(200, result)
    return True


def try_post(h, p, body):
    if p != "/runs/compare/test":
        return False

    import clozn.runs.store as runlog
    from clozn.replay import controlled

    body = body if isinstance(body, dict) else {}
    rid_a, rid_b = str(body.get("a") or ""), str(body.get("b") or "")
    if not rid_a or not rid_b:
        h._json(400, {"error": "need body fields a and b with two recorded run ids"})
        return True
    run_a, run_b = runlog.get_run(rid_a), runlog.get_run(rid_b)
    missing = [rid for rid, run in ((rid_a, run_a), (rid_b, run_b)) if run is None]
    if missing:
        h._json(404, {"error": "run(s) not found: " + ", ".join(missing), "missing": missing})
        return True

    tests = body.get("tests")
    if isinstance(tests, str):
        tests = [item.strip() for item in tests.split(",") if item.strip()]
    dry_run = bool(body.get("dry_run") or body.get("plan"))
    try:
        max_runs = body.get("max_runs", controlled.DEFAULT_MAX_RUNS)
        max_seconds = body.get("max_seconds", controlled.DEFAULT_MAX_SECONDS)
        match_criterion = body.get("match_criterion", "exact_output")
        if dry_run:
            document = controlled.plan_change_tests(
                run_a, run_b, tests=tests, max_runs=max_runs, max_seconds=max_seconds,
                match_criterion=match_criterion,
            )
        else:
            # NOT converted to per-run model selection (clozn.server.model_routing.
            # select_control_model_for_run): this route compares TWO runs (a and b), which under a
            # managed multi-model gateway may legitimately belong to two DIFFERENT models -- there is
            # no single "the run's model" to read a worker from the way execution-fork/receipts/etc.
            # do for a single immutable run. Picking one of the two (or requiring they match) would be
            # a real product decision this task's brief did not ask for; left as ctx.active_sub(h) --
            # i.e. it keeps today's exact (already fail-closed-under-a-router) behavior -- rather than
            # guess. Reported as genuinely ambiguous.
            from clozn.server import app as ctx
            substrate = ctx.active_sub(h)
            if not (substrate and callable(getattr(substrate, "chat", None))):
                h._json(503, {"error": "controlled tests require a ready product model worker"})
                return True
            document = controlled.execute_change_tests(
                run_a, run_b, runner=controlled.SubstrateReplayRunner(substrate),
                tests=tests, max_runs=max_runs, max_seconds=max_seconds,
                match_criterion=match_criterion,
            )
    except controlled.ControlledTestError as exc:
        h._json(400, {"error": str(exc)})
        return True
    except Exception as exc:
        h._json(500, {"error": f"controlled tests failed: {type(exc).__name__}: {exc}"})
        return True
    h._json(200, document)
    return True
