"""POST /runs/<id>/causal-trace -- the on-demand causal trace behind ONE token of a stored run's
recorded answer. Runs clozn.analysis.tracer.trace over the run's own final_prompt + recorded
response at continuation-token `position` (0-based; default 0 = the first answer token). This is
what the Studio click-a-token panel calls: click token i -> trace token i, honestly.

Unlike provenance (which severs input attention to ask "did the answer USE this context span"),
this is the residual-site tracer: which (layer, position) sites causally support the token, with
matched controls and an honest verdict (PASS / NO_CAUSAL_NODES / FAILED_CONTROLS). Defaults chosen
so it works on ANY engine and answers the SELECTIVE question:
  * screen_mode "ablate" -- the any-GGUF mean-ablation screen; no jlens sidecar required (the
    default so this never 400s on an engine whose sidecar can't be served).
  * contrast "auto" -- answer-SELECTIVE scoring against the runner-up foil (the screen-null fix),
    so a node counts only if it prefers THIS token over its top competitor.
  * The honest caveat rides in the receipt's config: individual sites rarely carry the answer on
    these models (the distributed-function result) -- this is the causal skeleton, not a full
    explanation.

Registered in clozn/server/app.py's _POST_ROUTES (POST-only, no try_get -- mirrors provenance.py /
span_receipts.py). CAPABILITY-UNAVAILABLE: tracer.trace() returns a typed {"ok": False,
"blocked": ...} dict rather than raising, and this route ships it as a 200 body (ok: false) -- the
same reasoning provenance.py documents (a completed analysis that reports "couldn't" is a
successful outcome, not a failed request).

Which worker: the run's OWN `model` field resolves the worker, through
clozn.server.model_routing.select_control_model_for_run -- never a client-supplied model, and never
the bare module-level ctx.ENGINE fallback this route used before (a real bug under a managed
multi-model gateway: ctx.ENGINE there is the ROUTER'S CONTROL-PAIR engine, i.e. some OTHER model's
worker, not this run's -- silently tracing the wrong model's weights while returning a normal-looking
200. See ctx.active_engine's own docstring for why it, too, refuses rather than falls back to
ctx.ENGINE under a managed router). An unknown/not-ready parent model now refuses with a typed
`clozn.model-routing.v1` error instead. Legacy one-worker serving is unaffected: `engine_url` stays
None (tracer.trace's own DEFAULT_ENGINE applies), exactly as before.
"""


def _engine_base(engine):
    return getattr(engine, "base", None)


def try_post(h, p, body):
    if not (p.startswith("/runs/") and p.endswith("/causal-trace")):
        return False

    rid = p[len("/runs/"):-len("/causal-trace")]
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
        h._json(200, {"ok": False, "blocked": "run has no recorded response text to trace"})
        return True

    body = body if isinstance(body, dict) else {}

    position = body.get("position", 0)
    if isinstance(position, bool) or not isinstance(position, int) or position < 0:
        h._json(400, {"error": "'position' must be a non-negative integer (continuation token index)"})
        return True

    seed = body.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        h._json(400, {"error": "'seed' must be an integer"})
        return True

    screen_mode = body.get("screen_mode", "ablate")
    if screen_mode not in ("auto", "jlens", "ablate"):
        h._json(400, {"error": "'screen_mode' must be one of auto|jlens|ablate"})
        return True

    # contrast: default "auto" (answer-selective). Accept a foil string/token, or explicit null to
    # disable (absolute scoring). Missing => "auto".
    contrast = body["contrast"] if "contrast" in body else "auto"
    if not (contrast is None or isinstance(contrast, (str, int))):
        h._json(400, {"error": "'contrast' must be a string, an integer token id, null, or omitted"})
        return True

    from clozn.server.model_routing import select_control_model_for_run
    selection = select_control_model_for_run(h, run.get("model"), route="/runs/<id>/causal-trace")
    if selection is None:
        return True   # typed clozn.model-routing.v1 refusal already written

    from clozn.analysis import tracer

    kwargs = {"seed": seed, "screen_mode": screen_mode, "contrast": contrast}
    engine_url = _engine_base(selection.engine)
    if engine_url:
        kwargs["engine_url"] = engine_url

    result = tracer.trace(prompt, answer, position, **kwargs)
    h._json(200, result)
    return True
