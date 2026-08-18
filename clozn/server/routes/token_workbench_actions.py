"""Milestone F: the token-workbench ACTION endpoints -- where expensive work is actually requested.
Milestone E's GET /runs/<id>/tokens/<index>/workbench is deliberately read-only and computes nothing;
these four POST routes are its action surface, plus a generic job status/cancel pair.

    POST /runs/<id>/tokens/<index>/force-token
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
reachable at all -- a harder failure than "this action is unavailable", using the shared run-scoped
model-action convention), 429 (job
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
        "force-token": _force_token_action,
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


# ========================================================================================== force-token
def _force_token_action(h, run, index, body):
    # The parent run's OWN model selects the worker -- never a client-supplied one -- through the
    # shared, router-aware run-scoped model resolver.
    from clozn.server.model_routing import select_control_model_for_run
    selection = select_control_model_for_run(
        h, run.get("model"), route="/runs/<id>/tokens/<index>/force-token")
    if selection is None:
        return True   # typed clozn.model-routing.v1 refusal already written
    sub = selection.sub
    if not (sub and getattr(sub, "engine", None)):
        h._json(503, {"error": "force-token requires a ready product model worker"})
        return True

    trace = run.get("trace") or {}
    pieces = trace.get("tokens")
    if not isinstance(pieces, list) or not pieces:
        h._json(400, {"error": "run has no recorded token trace"})
        return True
    if index < 0 or index >= len(pieces):
        h._json(400, {
            "error": f"token index {index} out of range (the reply has {len(pieces)} trace tokens)"})
        return True
    token_piece = body.get("token_piece", body.get("token"))
    if token_piece is not None and (not isinstance(token_piece, str) or not token_piece):
        h._json(400, {"error": "token_piece must be a non-empty string"})
        return True
    token_id = body.get("token_id")
    if token_id is not None and (isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0):
        h._json(400, {"error": "token_id must be a non-negative integer"})
        return True
    if token_piece is None and token_id is None:
        h._json(400, {"error": "force-token requires token_id or token_piece"})
        return True
    alternatives = trace.get("alternatives") if isinstance(trace.get("alternatives"), list) else []
    pairs = alternatives[index] if index < len(alternatives) and isinstance(alternatives[index], list) else []
    recorded_id = trace.get("token_ids")[index] if isinstance(trace.get("token_ids"), list) and index < len(trace.get("token_ids")) else None
    recorded_piece = pieces[index]
    matched_piece = None
    if token_id is not None:
        for item in pairs:
            if not isinstance(item, dict):
                continue
            item_id = item.get("token_id", item.get("id"))
            item_piece = item.get("piece", item.get("text"))
            if item_id == token_id:
                matched_piece = item_piece
                break
        if matched_piece is None and token_id == recorded_id:
            matched_piece = recorded_piece
        if matched_piece is None and token_piece is None:
            h._json(400, {"error": f"token_id {token_id} is not recorded at position {index}"})
            return True
        if matched_piece is not None and token_piece is not None and token_piece != matched_piece:
            h._json(400, {"error": "token_id and token_piece identify different recorded tokens"})
            return True
        if token_piece is None:
            token_piece = matched_piece
    max_new = body.get("max_new", 32)
    if isinstance(max_new, bool) or not isinstance(max_new, int) or max_new < 0:
        h._json(400, {"error": "max_new must be a non-negative integer"})
        return True

    if selection.runtime_key is not None:
        runtime_identity = dict(selection.runtime_key)
        worker_identity = dict(selection.worker_identity) if selection.worker_identity is not None else None
    else:
        from clozn.experiments.execution_facts import selection_identity_facts
        runtime_identity, worker_identity, _engine = selection_identity_facts(selection)
    from clozn.runs.token_workbench_actions import force_token_worker
    worker = force_token_worker(
        run, sub, index, token_piece=token_piece, token_id=token_id,
        max_new=max_new, runtime_identity=runtime_identity, worker_identity=worker_identity)

    from clozn.server.influence_jobs import JOBS, JobCapacityError

    try:
        job = JOBS.start(run["id"], worker, kind="force_token")
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
    from clozn.runs.token_workbench_actions import (
        CAUSAL_TRACE_METHOD_VERSION, action_cache_key, causal_trace_worker, find_cached_action)

    if selection.runtime_key is not None:
        runtime_identity = dict(selection.runtime_key)
    else:
        from clozn.experiments.execution_facts import selection_identity_facts
        runtime_identity, _worker_identity, _engine = selection_identity_facts(selection)
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

    # A successful mechanistic comparison needs the exact prompt and continuation token IDs already
    # recorded by the anchor run. Refuse before creating a job when that evidence is absent; this keeps
    # a missing trace distinct from an engine-side capture failure.
    trace = run.get("trace") if isinstance(run.get("trace"), dict) else {}
    token_ids = trace.get("token_ids") if isinstance(trace.get("token_ids"), list) else []
    if not isinstance(run.get("final_prompt"), str) or not run.get("final_prompt"):
        h._json(422, {"outcome": "unavailable", "reason": {
            "code": "mechanistic_diff_missing_prompt",
            "message": "mechanistic diff needs the anchor run's exact final_prompt",
        }, "pair_compatibility": gate["report"]})
        return True
    if index < 0 or index >= len(token_ids):
        h._json(400, {"error": f"token index {index} has no exact recorded token id to compare"})
        return True

    from clozn.server import app as ctx
    router = getattr(ctx, "MODEL_ROUTER", None)
    if router is None:
        # Preserve the legacy single-worker behavior: a compatible pair is still honestly unavailable
        # when this process has no managed model registry capable of selecting two model identities.
        h._json(422, {
            "outcome": "unavailable",
            "reason": {
                "code": "cross_model_execution_not_wired",
                "message": (
                    "pair compatibility permits this comparison, but this gateway is not serving a "
                    "managed multi-model registry; configure two models or use `clozn diff-model` "
                    "at the CLI"),
            },
            "pair_compatibility": gate["report"],
        })
        return True

    layers = body.get("layers")
    if layers is None:
        layer_info = gate["report"].get("layer_count") or {}
        layer_count = layer_info.get("value_a") if isinstance(layer_info, dict) else None
        if not isinstance(layer_count, int) or layer_count < 3:
            h._json(422, {"outcome": "unavailable", "reason": {
                "code": "mechanistic_diff_layer_grid_unavailable",
                "message": "the pair report has no usable layer_count; supply body.layers explicitly",
            }, "pair_compatibility": gate["report"]})
            return True
        # A small, deterministic default keeps a click bounded while still sampling early, middle, and
        # late capturable layers. Callers can request a full grid explicitly.
        layers = sorted({1, max(1, layer_count // 2), layer_count - 2})
    if (isinstance(layers, (str, bytes)) or not isinstance(layers, list)
            or not layers or any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 1 for layer in layers)):
        h._json(400, {"error": "body.layers must be a non-empty list of positive integers"})
        return True
    layers = sorted(set(layers))

    topk = body.get("topk", 8)
    if isinstance(topk, bool) or not isinstance(topk, int) or not 0 <= topk <= 128:
        h._json(400, {"error": "body.topk must be an integer in [0, 128]"})
        return True
    store_tensors = body.get("store_tensors", False)
    if not isinstance(store_tensors, bool):
        h._json(400, {"error": "body.store_tensors must be a boolean"})
        return True

    from clozn.runs.token_workbench_actions import (
        mechanistic_diff_cache_key, mechanistic_diff_worker, find_cached_action)

    cache_key = mechanistic_diff_cache_key(
        run, reference_run, index, layers=layers, topk=topk, store_tensors=store_tensors)
    if not body.get("refresh"):
        cached = find_cached_action(run, cache_key)
        if cached is not None:
            h._json(200, {"outcome": "cached", "mechanistic_diff_id": cache_key, "artifact": cached})
            return True

    worker = mechanistic_diff_worker(
        run, reference_run, index, pair_compatibility=gate["report"], router=router,
        layers=layers, topk=topk, store_tensors=store_tensors, cache_key=cache_key)
    from clozn.server.influence_jobs import JOBS, JobCapacityError

    try:
        job = JOBS.start(run["id"], worker, kind="mechanistic_diff")
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc)})
        return True
    h._json(202, {"outcome": "job", "mechanistic_diff_id": cache_key, "job": job})
    return True
