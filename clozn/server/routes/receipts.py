"""The receipts surface: a run's downloadable bundle (GET /runs/<id>/export), the free M1 explain read,
the M2 leave-one-out/redundancy receipts, the S3 teacher-forced rederivation, the J3 J-lens feed (general
+ per-run), and the M4 accountable-self narration. All read a past run and either reshape free signals
or regenerate arms to prove a causal claim about it. Mechanical extraction of the matching
`if p.startswith("/runs/") and p.endswith(...)` / `if p == "/jlens"` branches out of clozn.server.app's
do_GET/do_POST; behavior unchanged. -> clozn.receipts (+ the app's own J-lens envelope helpers).
"""
from clozn.server import app as ctx


def try_get(h, p):
    if p == "/experiments/types":   # the experiment-type registry, for a UI "Experiment drawer" catalog
        from clozn.experiments import experiment as clozn_experiment
        h._json(200, {"types": clozn_experiment.catalog()})
        return True
    if (p.startswith("/runs/") and p.endswith("/export")
            and "/" not in p[len("/runs/"):-len("/export")]):
        # one-call download: run + M1 explain + trace.  Do not greedily
        # shadow nested, typed exports such as /runs/<id>/influence-map/export.
        import clozn.runs.store as runlog
        import clozn.receipts.explain as explain
        import clozn.receipts.bundle as receipt_bundle
        from urllib.parse import urlparse, parse_qs
        rid = p[len("/runs/"):-len("/export")]
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True
        try:
            xr = explain.explain(run)                # M1: pure read/reshape, no generation
        except Exception:
            xr = None
        bundle = receipt_bundle.build(run, explain=xr)
        fmt = (parse_qs(urlparse(h.path).query).get("format") or ["json"])[0]
        if fmt == "md":
            h._send(200, receipt_bundle.to_markdown(bundle), "text/markdown; charset=utf-8",
                   extra_headers={"Content-Disposition": 'attachment; filename="' + rid + '.md"'})
            return True
        h._json(200, bundle,
               extra_headers={"Content-Disposition": 'attachment; filename="' + rid + '.json"'})
        return True
    return False



def _jlens_int(h, body, key, default):
    """Strict int parse for /jlens params -> the value, or None after answering 400.
    int() on user input crashed the connection ("deep" -> ValueError up through the dispatcher), and
    bool/float coercion silently truncated (layer 14.7 read as 14 with no complaint)."""
    v = body.get(key, default)
    if v is None:
        v = default
    if isinstance(v, bool) or not isinstance(v, int):
        h._json(400, {"error": f"{key} must be an integer (got {type(v).__name__}: {v!r})"})
        return None
    return v

def try_post(h, p, body):
    if p.startswith("/runs/") and p.endswith("/explain"):   # M1: assemble the FREE signals -- zero generation
        rid = p[len("/runs/"):-len("/explain")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        import clozn.receipts.explain as explain
        h._json(200, explain.explain(run))
        return True
    if p.startswith("/runs/") and p.endswith("/receipts"):   # M2: prove-all -- leave-one-out + redundancy guard
        rid = p[len("/runs/"):-len("/receipts")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        mode = str(body.get("mode") or "regen")
        if mode not in ("regen", "forced", "both"):
            h._json(400, {"error": "mode must be one of regen|forced|both"})
            return True
        coalitions_batch = str(body.get("coalitions_batch") or "auto")
        if coalitions_batch not in ("auto", "off", "approximate"):
            h._json(400, {"error": "coalitions_batch must be one of auto|off|approximate"})
            return True
        # regen/both regenerate both arms through the product model; forced-only never generates
        # (S3: teacher-forced /score on the worker) -- no chat needed.
        if mode in ("regen", "both") and not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "chat", None)):
            h._json(503, {"error": "receipts require a ready product model worker"})
            return True
        from clozn import receipts
        try:
            # `coalitions` (opt-in, default False -- see docs/PRODUCT_ROADMAP.md §8 tail): pairwise
            # coalition deltas + a Shapley approximation + the interaction gap, alongside the default
            # leave-one-out receipts above. Never changes the default response shape when omitted.
            h._json(200, receipts.prove_all(run, ctx.active_sub(h), mode=mode,
                                            coalitions=bool(body.get("coalitions", False)),
                                            coalitions_batch=coalitions_batch))
        except Exception as e:
            h._json(500, {"error": f"receipts failed: {type(e).__name__}: {e}"})
        return True
    if p.startswith("/runs/") and p.endswith("/receipt"):   # M2: one rigorous both-arms-greedy causal receipt
        rid = p[len("/runs/"):-len("/receipt")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        mode = str(body.get("mode") or "regen")
        if mode not in ("regen", "forced", "both"):
            h._json(400, {"error": "mode must be one of regen|forced|both"})
            return True
        if mode in ("regen", "both") and not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "chat", None)):
            h._json(503, {"error": "receipt requires a ready product model worker"})
            return True
        influence = body.get("influence")
        if not isinstance(influence, dict) or not influence:
            h._json(400, {"error": "need an influence spec: one of "
                         "{card_id|dial|memory_off|behavior_off}"})
            return True
        from clozn import receipts
        try:
            out = receipts.receipt(run, influence, ctx.active_sub(h), mode=mode)
        except Exception as e:
            h._json(500, {"error": f"receipt failed: {type(e).__name__}: {e}"})
            return True
        if out is None:
            # receipts.receipt (and everything it calls -- replay.py/rederive.py) is documented to NEVER
            # raise, swallowing a dead-engine URLError into this same ambiguous None a bad influence spec
            # also produces. Probe the engine directly (only on this already-failed path, never on the
            # happy path) so the two don't read as the same problem to a caller debugging blind.
            if not ctx._engine_reachable(ctx.active_sub(h)):
                h._json(502, {"error": ctx._engine_unreachable_message()})
            else:
                h._json(500, {"error": "receipt failed (bad influence spec, or the replay "
                             "could not be generated)"})
            return True
        h._json(200, out)
        return True
    if p.startswith("/runs/") and p.endswith("/swap_receipt"):   # Tier-1 #3: read-disposition, write-a-different-concept, diff
        rid = p[len("/runs/"):-len("/swap_receipt")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        if not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "engine", None) and getattr(ctx.active_sub(h), "jlens", None)):
            h._json(503, {"error": "swap_receipt requires the product worker with J-lens enabled"})
            return True
        to_concept = str(body.get("to_concept", "")).strip()
        if not to_concept:
            h._json(400, {"error": "need a 'to_concept' to swap in"})
            return True
        from clozn.receipts.swap_receipt import swap_receipt
        try:
            out = swap_receipt(run, body.get("from_hint"), to_concept, ctx.active_sub(h))
        except Exception as e:
            h._json(500, {"error": f"swap_receipt failed: {type(e).__name__}: {e}"})
            return True
        if not out.get("causal_verified"):
            # swap_receipt() never raises -- a total failure (no engine, template render, generation, ...)
            # degrades to causal_verified:false + blocked/note explaining why, all other fields null. That
            # used to still ship as HTTP 200, so a caller checking response.ok alone would read a complete
            # failure as a success (unlike the sibling /receipt and /rederive routes, both non-2xx on the
            # equivalent condition). Body is left intact -- unlike those siblings, swap_receipt's own
            # blocked/note already self-describe the failure; only the status code was wrong.
            status = 502 if not ctx._engine_reachable(ctx.active_sub(h)) else 500
            h._json(status, out)
            return True
        h._json(200, out)
        return True
    if p.startswith("/runs/") and p.endswith("/rederive"):   # S3: deterministic teacher-forced re-derivation
        rid = p[len("/runs/"):-len("/rederive")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        if not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "score_tokens", None)):
            h._json(503, {"error": "rederive requires worker token scoring"})
            return True
        import clozn.receipts.rederive as rederive
        try:
            out = rederive.rederive(run, ctx.active_sub(h))
        except Exception as e:
            h._json(500, {"error": f"rederive failed: {type(e).__name__}: {e}"})
            return True
        if out is None:
            # rederive() is documented to NEVER raise (mirrors replay.py's contract), so a dead-engine
            # URLError from score_tokens() collapses into this same ambiguous None a genuinely-missing
            # continuation also produces. Probe directly (only here, on the already-failed path) so the
            # two don't read as the same problem to a caller debugging blind.
            if not ctx._engine_reachable(ctx.active_sub(h)):
                h._json(502, {"error": ctx._engine_unreachable_message()})
            else:
                h._json(500, {"error": "rederive failed (no continuation to score, or the "
                             "engine score call failed)"})
            return True
        h._json(200, out)
        return True
    if p == "/jlens":   # J3: general J-lens passthrough (brain/lab page) -- per-token "disposed to say"
        text = str(body.get("text", ""))
        if not text:
            h._json(400, {"error": "need a 'text' to read"})
            return True
        layer = body.get("layer")
        if layer is not None and (isinstance(layer, bool) or not isinstance(layer, int)):
            h._json(400, {"error": f"layer must be an integer (got {type(layer).__name__}: {layer!r})"})
            return True
        topk = _jlens_int(h, body, "topk", 5)
        if topk is None:
            return True
        want_protocol = bool(body.get("protocol", False))
        if not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "jlens", None)):
            h._json(200, {"available": False, "run_id": None,
                         "reason": "the product model worker has no J-lens"})
            return True
        res = ctx.active_sub(h).jlens(text, layer=layer, topk=topk)
        h._json(200, ctx._jlens_envelope(res, run_id=None, text_source="input", want_protocol=want_protocol))
        return True
    if p.startswith("/runs/") and p.endswith("/jlens"):   # J3: the Run Inspector J-lens feed for a run
        rid = p[len("/runs/"):-len("/jlens")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        if not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "jlens", None)):
            h._json(200, {"available": False, "run_id": rid,
                         "reason": "the product model worker has no J-lens"})
            return True
        text, text_source = ctx._jlens_run_text(run)          # prefer the stored response; note the source
        if not text:
            h._json(200, {"available": False, "run_id": rid,
                         "reason": "this run has no response/text to read"})
            return True
        layer = body.get("layer")
        if layer is not None and (isinstance(layer, bool) or not isinstance(layer, int)):
            h._json(400, {"error": f"layer must be an integer (got {type(layer).__name__}: {layer!r})"})
            return True
        topk = _jlens_int(h, body, "topk", 5)
        if topk is None:
            return True
        want_protocol = bool(body.get("protocol", False))
        res = ctx.active_sub(h).jlens(text, layer=layer, topk=topk)
        h._json(200, ctx._jlens_envelope(res, run_id=rid, text_source=text_source, want_protocol=want_protocol))
        return True
    if p.startswith("/runs/") and p.endswith("/narrate"):   # M4: accountable-self narration + confabulation-diff
        rid = p[len("/runs/"):-len("/narrate")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        if not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "chat", None)):   # constrained + unconstrained both generate
            h._json(503, {"error": "narration requires a ready product model worker"})
            return True
        import clozn.receipts.narrate as narrate
        matcher = narrate.lexical_default            # the weak keyword proxy -- the LABELED fallback judge
        try:
            import clozn.receipts.semantic_matcher as semantic_matcher                  # the real, INDEPENDENT cross-encoder judge, if present
            if semantic_matcher.available():
                matcher = semantic_matcher.nli_support_matcher
        except Exception:
            pass
        try:
            # returns the receipt-constrained narration + confabulation flags; the raw unconstrained
            # "why" is NEVER a field in the result (narrate.py's structural trap guard). narrate()'s
            # own `note` states which matcher ran, so the response is self-describing about its honesty.
            out = narrate.narrate(run, ctx.active_sub(h), support_matcher=matcher)
        except Exception as e:
            h._json(500, {"error": f"narrate failed: {type(e).__name__}: {e}"})
            return True
        h._json(200, out)
        return True
    if p.startswith("/runs/") and p.endswith("/experiment"):   # ONE experiment primitive over replay/
        # counterfactual/receipt/branch/swap_receipt -- dispatches on change.type, returns ONE envelope.
        rid = p[len("/runs/"):-len("/experiment")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.experiments import experiment as clozn_experiment
        change = body.get("change")
        ctype = change.get("type") if isinstance(change, dict) else None
        if not isinstance(change, dict) or not ctype:
            h._json(400, {"error": "need a change spec: {type, ...}"})
            return True
        if ctype not in clozn_experiment.REGISTRY:
            h._json(400, {"error": f"unknown change.type: {ctype!r} (know: "
                         f"{sorted(clozn_experiment.REGISTRY)})"})
            return True
        if not clozn_experiment.substrate_ok(ctype, ctx.active_sub(h)):
            needs = clozn_experiment.REGISTRY[ctype]["substrate"]
            msg = ("experiment requires a ready product model worker" if needs == "chat" else
                   "experiment requires the product worker with J-lens enabled")
            h._json(503, {"error": msg})
            return True
        method = body.get("method")
        try:
            out = clozn_experiment.run_experiment(run, change, method, ctx.active_sub(h))
        except ValueError as e:
            h._json(400, {"error": str(e)})
            return True
        except Exception as e:
            h._json(500, {"error": f"experiment failed: {type(e).__name__}: {e}"})
            return True
        if out is None:
            h._json(500, {"error": "experiment failed (bad change spec, or the underlying "
                         "op could not be generated)"})
            return True
        h._json(200, out)
        return True
    return False
