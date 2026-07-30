"""Milestone F: the token-workbench ACTION endpoints -- where expensive work is actually requested.
Milestone E's GET /runs/<id>/tokens/<index>/workbench is deliberately read-only and computes nothing;
these four POST routes are its action surface, plus a generic job status/cancel pair.

    POST /runs/<id>/tokens/<index>/fork
    POST /runs/<id>/tokens/<index>/causal-trace
    POST /runs/<id>/tokens/<index>/source-measure
    POST /runs/<id>/tokens/<index>/mechanistic-diff
    GET  /runs/<id>/tokens/<index>/jobs/<job_id>
    POST /runs/<id>/tokens/<index>/jobs/<job_id>/cancel

Every action responds with exactly one of three shapes (see clozn/runs/token_workbench_actions.py's
own docstring for the full contract):
    200 {"outcome": "cached", "artifact": ...}
    202 {"outcome": "job", "job": ...}
    422 {"outcome": "unavailable", "reason": {"code", "message"}}
plus the ordinary HTTP failure modes: 404 (no such run/job), 400 (bad index/body), 503 (no worker
reachable at all -- a harder failure than "this action is unavailable", mirroring
POST /runs/<id>/fork's and POST /runs/<id>/influence-map's own established 503 convention), 429 (job
capacity full).

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4) -- no edit to clozn/server/app.py. All
logic lives in clozn.runs.token_workbench_actions; this file only parses the path, validates the
request shape, and picks a status code.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True
_MARKER = "/tokens/"


def _parse(p: str):
    """(run_id, index_str, trailing) for /runs/<id>/tokens/<index>/<trailing...>, or None when `p`
    does not match this route family at all (never claims a path it cannot fully parse)."""
    if not p.startswith("/runs/"):
        return None
    middle = p[len("/runs/"):]
    if _MARKER not in middle:
        return None
    run_id, _, rest = middle.partition(_MARKER)
    index_str, _, trailing = rest.partition("/")
    if not run_id or not index_str or not trailing:
        return None
    return run_id, index_str, trailing


def _parse_index(h, index_str: str) -> int | None:
    try:
        return int(index_str)
    except ValueError:
        h._json(400, {"error": "token index must be an integer"})
        return None


def try_get(h, p):
    parsed = _parse(p)
    if parsed is None:
        return False
    run_id, index_str, trailing = parsed
    if not trailing.startswith("jobs/") or trailing.endswith("/cancel"):
        return False
    job_id = trailing[len("jobs/"):]
    if not job_id or "/" in job_id:
        return False
    if _parse_index(h, index_str) is None:
        return True

    from clozn.server.influence_jobs import JOBS

    job = JOBS.get(run_id, job_id)
    if job is None:
        h._json(404, {"error": "job not found"})
    else:
        h._json(200, job)
    return True


def try_post(h, p, body):
    parsed = _parse(p)
    if parsed is None:
        return False
    run_id, index_str, trailing = parsed
    body = body if isinstance(body, dict) else {}

    if trailing.startswith("jobs/") and trailing.endswith("/cancel"):
        job_id = trailing[len("jobs/"):-len("/cancel")]
        if not job_id or "/" in job_id:
            return False
        if _parse_index(h, index_str) is None:
            return True
        from clozn.server.influence_jobs import JOBS

        job = JOBS.cancel(run_id, job_id)
        if job is None:
            h._json(404, {"error": "job not found"})
        else:
            h._json(200, job)
        return True

    handlers = {
        "fork": _fork_action,
        "causal-trace": _causal_trace_action,
        "source-measure": _source_measure_action,
        "mechanistic-diff": _mechanistic_diff_action,
    }
    handler = handlers.get(trailing)
    if handler is None:
        return False

    index = _parse_index(h, index_str)
    if index is None:
        return True

    import clozn.runs.store as runlog

    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    return handler(h, run, index, body)


# ================================================================================================ fork
def _fork_action(h, run, index, body):
    # The parent run's OWN model selects the worker -- never a client-supplied one -- through the
    # same shared, router-aware helper POST /runs/<id>/fork uses (clozn.server.routes.fork).
    from clozn.server.model_routing import select_control_model_for_run
    selection = select_control_model_for_run(
        h, run.get("model"), route="/runs/<id>/tokens/<index>/fork")
    if selection is None:
        return True   # typed clozn.model-routing.v1 refusal already written
    sub = selection.sub
    if not (sub and getattr(sub, "engine", None)):
        h._json(503, {"error": "fork requires a ready product model worker"})
        return True

    import clozn.replay.fork as fork_mod

    trace = run.get("trace") or {}
    pieces = trace.get("tokens")
    if not isinstance(pieces, list) or not pieces:
        h._json(400, {"error": "run has no trace to fork from"})
        return True
    if index < 0 or index >= len(pieces):
        h._json(400, {
            "error": f"token index {index} out of range (the reply has {len(pieces)} trace tokens)"})
        return True
    try:
        forced_piece, _was_recorded = fork_mod.resolve_forced_token(
            trace, index, token=body.get("token"), token_id=body.get("token_id"))
    except ValueError as exc:
        h._json(400, {"error": str(exc)})
        return True

    import clozn.runs.store as runlog
    from clozn.runs.token_workbench_actions import find_cached_fork_child, fork_worker

    related = list(runlog.iter_runs(limit=200))
    cached = find_cached_fork_child(related, run["id"], index, forced_piece)
    if cached is not None:
        h._json(200, {"outcome": "cached", "artifact": cached})
        return True

    from clozn.server.routes.execution_fork import _identity_facts

    runtime_identity, worker_identity, _engine = _identity_facts(selection)
    worker = fork_worker(
        run, sub, index, token=body.get("token"), token_id=body.get("token_id"),
        runtime_identity=runtime_identity, worker_identity=worker_identity)

    from clozn.server.influence_jobs import JOBS, JobCapacityError

    try:
        job = JOBS.start(run["id"], worker, kind="fork")
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc)})
        return True
    h._json(202, {"outcome": "job", "job": job})
    return True


# ======================================================================================== causal-trace
def _causal_trace_action(h, run, index, body):
    final_prompt = run.get("final_prompt")
    if not isinstance(final_prompt, str) or not final_prompt:
        h._json(422, {"outcome": "unavailable", "reason": {
            "code": "missing_final_prompt",
            "message": "run has no recorded final_prompt (the exact rendered prompt) to trace",
        }})
        return True
    response = run.get("response")
    if not isinstance(response, str) or not response:
        h._json(422, {"outcome": "unavailable", "reason": {
            "code": "missing_response", "message": "run has no recorded response text to trace",
        }})
        return True
    trace = run.get("trace") or {}
    pieces = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
    if index < 0 or index >= len(pieces):
        h._json(400, {
            "error": f"token index {index} out of range (the reply has {len(pieces)} trace tokens)"})
        return True

    seed = body.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        h._json(400, {"error": "'seed' must be an integer"})
        return True
    screen_mode = body.get("screen_mode", "ablate")
    if screen_mode not in ("auto", "jlens", "ablate"):
        h._json(400, {"error": "'screen_mode' must be one of auto|jlens|ablate"})
        return True
    contrast = body["contrast"] if "contrast" in body else "auto"
    if not (contrast is None or isinstance(contrast, (str, int))):
        h._json(400, {"error": "'contrast' must be a string, an integer token id, null, or omitted"})
        return True

    # The run's OWN model selects the worker -- never a client-supplied one -- through the same
    # shared, router-aware helper POST /runs/<id>/causal-trace uses (clozn.server.routes.causal_trace).
    from clozn.server.model_routing import select_control_model_for_run
    selection = select_control_model_for_run(
        h, run.get("model"), route="/runs/<id>/tokens/<index>/causal-trace")
    if selection is None:
        return True   # typed clozn.model-routing.v1 refusal already written
    sub = selection.sub
    if not (sub and getattr(sub, "engine", None)):
        h._json(503, {"error": "causal-trace requires a ready product model worker"})
        return True

    from clozn.server.routes.causal_trace import _engine_base
    from clozn.server.routes.execution_fork import _identity_facts
    from clozn.runs.token_workbench_actions import (
        CAUSAL_TRACE_METHOD_VERSION, action_cache_key, causal_trace_worker, find_cached_action)

    runtime_identity, _worker_identity, _engine = _identity_facts(selection)
    params = {"seed": seed, "screen_mode": screen_mode, "contrast": contrast}
    cache_key = action_cache_key(
        run, index, "causal_trace", CAUSAL_TRACE_METHOD_VERSION, params,
        runtime_identity=runtime_identity)
    if not body.get("refresh"):
        cached = find_cached_action(run, cache_key)
        if cached is not None:
            h._json(200, {"outcome": "cached", "artifact": cached})
            return True

    engine_url = _engine_base(selection.engine)
    worker = causal_trace_worker(
        run, index, seed=seed, screen_mode=screen_mode, contrast=contrast, engine_url=engine_url,
        cache_key=cache_key)

    from clozn.server.influence_jobs import JOBS, JobCapacityError

    try:
        job = JOBS.start(run["id"], worker, kind="causal_trace")
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc)})
        return True
    h._json(202, {"outcome": "job", "job": job})
    return True


# ======================================================================================= source-measure
def _source_measure_action(h, run, index, body):
    trace = run.get("trace") or {}
    pieces = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
    if index < 0 or index >= len(pieces):
        h._json(400, {
            "error": f"token index {index} out of range (the reply has {len(pieces)} trace tokens)"})
        return True

    from clozn.receipts.context_answer_influence import cache_matches

    cached = run.get("influence_map")
    if cache_matches(run, cached) and not body.get("refresh"):
        h._json(200, {"outcome": "cached", "artifact": cached})
        return True

    # The run's OWN model selects the worker -- never a client-supplied one -- through the same
    # shared, router-aware helper POST /runs/<id>/influence-map uses (clozn.server.routes.influence_map).
    from clozn.server.model_routing import select_control_model_for_run
    selection = select_control_model_for_run(
        h, run.get("model"), route="/runs/<id>/tokens/<index>/source-measure")
    if selection is None:
        return True   # typed clozn.model-routing.v1 refusal already written
    sub = selection.sub
    if not (sub and callable(getattr(sub, "score_tokens", None))):
        h._json(503, {"error": "source-measure requires worker token scoring"})
        return True

    from clozn.server.routes.influence_map import _max_spans

    try:
        max_spans = _max_spans(body)
    except ValueError as exc:
        h._json(400, {"error": str(exc)})
        return True

    from clozn.runs.token_workbench_actions import source_measure_job_worker

    worker = source_measure_job_worker(run, sub, max_spans)

    from clozn.server.influence_jobs import JOBS, JobCapacityError

    try:
        job = JOBS.start(run["id"], worker, kind="influence_map")
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc)})
        return True
    h._json(202, {"outcome": "job", "job": job})
    return True


# ====================================================================================== mechanistic-diff
def _mechanistic_diff_action(h, run, index, body):
    reference_run_id = body.get("reference_run_id")
    if not isinstance(reference_run_id, str) or not reference_run_id:
        h._json(422, {"outcome": "unavailable", "reason": {
            "code": "reference_run_required",
            "message": (
                "mechanistic diff needs body.reference_run_id naming a run from a DIFFERENT model"),
        }})
        return True

    import clozn.runs.store as runlog

    reference_run = runlog.get_run(reference_run_id)
    if reference_run is None:
        h._json(422, {"outcome": "unavailable", "reason": {
            "code": "reference_run_not_found",
            "message": f"reference run {reference_run_id!r} was not found",
        }})
        return True

    from clozn.runs.token_workbench_actions import mechanistic_diff_gate

    gate = mechanistic_diff_gate(run, reference_run)
    if not gate["permitted"]:
        h._json(422, {
            "outcome": "unavailable",
            "reason": {"code": "pair_compatibility_refused", "message": gate["reason"]},
            "pair_compatibility": gate["report"],
        })
        return True

    # Pair compatibility permits a cross-model comparison in principle. Actually EXECUTING one needs
    # two GGUFs loaded sequentially through worker_registry/model_routing's cold-load-and-evict
    # machinery -- owned by a different agent and under concurrent development this same wave. Wiring
    # a job against infrastructure that could change shape mid-task would risk a job that silently
    # cannot run; an honest, typed "not yet" is the more correct outcome (see clozn.runs.
    # token_workbench_actions's module docstring).
    h._json(422, {
        "outcome": "unavailable",
        "reason": {
            "code": "cross_model_execution_not_wired",
            "message": (
                "pair compatibility permits this comparison, but sequential cross-model loading is "
                "not yet wired to an HTTP job in this milestone; run `clozn diff-model` at the CLI, "
                "which loads both GGUFs directly"),
        },
        "pair_compatibility": gate["report"],
    })
    return True
