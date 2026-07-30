"""POST /runs/<id>/provenance -- did a STORED run's recorded answer come from the context or the
model's own weights? Runs clozn.analysis.provenance.trace_provenance over the run's own final_prompt
(the exact rendered prompt) + its recorded response -- never a fresh generation, since the question is
about the answer that actually happened, not whatever the model would say greedily today.

NOT self-registering: wire it in clozn/server/app.py by importing this module alongside the other route
families and appending it to _POST_ROUTES (POST-only -- it has no try_get, mirroring span_receipts.py).

CAPABILITY-UNAVAILABLE CHOICE (documented per the task's Gate-0 ask): trace_provenance() itself probes
GET <engine>/health and returns a typed, non-raising {"ok": False, "blocked": "..."} dict -- never an
exception -- when the engine lacks attn_knockout (started without --no-flash-attn), when a requested
focus region is degenerate, or on any other runtime failure (e.g. an unreachable engine). This route
ships that dict straight through as a 200 body (ok: false) rather than mapping it to 503.

Reasoning: 503 in this route family (span_receipts.py, influence_map.py) is reserved for a precondition
the ROUTE itself checks BEFORE ever calling the analysis backend -- "there is no substrate/worker here
at all to even attempt this" (e.g. influence_map.py's `if not (sub and score_tokens): 503`). That is not
this endpoint's situation: the route never inspects engine capabilities itself, it hands the prompt
straight to trace_provenance and that function does its OWN /health probe as part of a normal, completed
call. A completed call that answers "the engine can't do this, and here specifically is why" is a
successful analysis outcome, not a failed HTTP request -- the same distinction influence_map.py draws
between its own 503 ("no worker to try") and 422 ("the backend tried and reported unavailable"). Since
the task's two offered shapes for this endpoint are 200/ok:false or 503, and 503 belongs to the
route-level "nothing to try" precondition this route does not have, 200/ok:false is the one that is
honestly correct here -- it also preserves trace_provenance's own explicit design contract verbatim ("the
product-facing call never raises -- it returns a labeled dict"), and folds three heterogeneous blocked
reasons (missing engine capability, a degenerate/empty focus region, any other runtime exception) under
one status without having to guess which of several non-200 codes best fits all three.

Which worker: the run's OWN `model` field resolves the worker, through
clozn.server.model_routing.select_control_model_for_run -- never a client-supplied model, and never the
bare module-level ctx.ENGINE fallback this route used before (a real bug under a managed multi-model
gateway: ctx.ENGINE there is the ROUTER'S CONTROL-PAIR engine, i.e. some OTHER model's worker, not this
run's -- silently tracing provenance against the wrong model's weights while returning a normal-looking
200. See clozn.server.routes.causal_trace's identical fix and ctx.active_engine's own docstring for why
it, too, refuses rather than falls back to ctx.ENGINE under a managed router). An unknown/not-ready
parent model now refuses with a typed `clozn.model-routing.v1` error instead -- this is the ONE non-200
outcome this route can produce, and only when a worker cannot be resolved at all, never when the
resolved engine merely lacks a capability (that stays 200/ok:false, per the reasoning above). Legacy
one-worker serving is unaffected: `engine_url` stays None (trace_provenance's own DEFAULT_ENGINE
applies), exactly as before.
"""


def _engine_base(engine):
    """The live engine client's base URL ('http://host:port') off the resolved run-scoped
    ModelSelection's own `.engine` -- in legacy (no-router) mode this is
    `ctx.active_engine(h)`, which already falls back to the module-level `ctx.ENGINE` when the
    substrate itself has none (there is only one engine either way, so that fallback is safe);
    under a managed router it is the run's OWN resolved worker engine, never any other model's.
    None (trace_provenance's own DEFAULT_ENGINE applies) when there truly is none."""
    return getattr(engine, "base", None)


def try_post(h, p, body):
    if not (p.startswith("/runs/") and p.endswith("/provenance")):
        return False

    rid = p[len("/runs/"):-len("/provenance")]
    import clozn.runs.store as runlog

    run = runlog.get_run(rid)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    prompt = run.get("final_prompt")
    if not isinstance(prompt, str) or not prompt:
        h._json(200, {"ok": False, "blocked": "run has no recorded final_prompt (the exact rendered "
                                              "prompt) to trace"})
        return True

    answer = run.get("response")
    if not isinstance(answer, str) or not answer:
        h._json(200, {"ok": False,
                     "blocked": "run has no recorded response text to trace provenance for"})
        return True

    body = body if isinstance(body, dict) else {}

    focus = body.get("focus")
    if focus is not None:
        if (not isinstance(focus, list) or len(focus) != 2
                or any(isinstance(x, bool) or not isinstance(x, int) for x in focus)):
            h._json(400, {"error": "'focus' must be a [start, end] integer array (prompt token indices)"})
            return True
        focus = tuple(focus)

    seed = body.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        h._json(400, {"error": "'seed' must be an integer"})
        return True

    from clozn.server.model_routing import select_control_model_for_run
    selection = select_control_model_for_run(h, run.get("model"), route="/runs/<id>/provenance")
    if selection is None:
        return True   # typed clozn.model-routing.v1 refusal already written

    from clozn.analysis.provenance import trace_provenance

    kwargs = {"focus": focus, "seed": seed}
    engine_url = _engine_base(selection.engine)
    if engine_url:
        kwargs["engine_url"] = engine_url

    result = trace_provenance(prompt, answer, **kwargs)
    h._json(200, result)
    return True
