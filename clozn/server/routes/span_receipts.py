"""The span-receipt surface ("injection forensics"): POST /runs/<id>/span_receipt -- causally attribute
one SPAN of the run's own prompt/context by ablation (regen arm + forced arm, both from
clozn.receipts.span_receipt). Mirrors routes/receipts.py's try_post(h, p, body) pattern exactly: read the
run, gate on the substrate, map SpanSpecError -> 400, exceptions -> 500, None -> 500.

Registered in clozn/server/app.py: imported as `_span_receipt_routes` and placed in `_POST_ROUTES`
(POST-only -- it has no try_get). Live surface: Studio's span-forensics UI in Replay (`api.spanReceipt`
in the previous shell, studio/heavn -- no longer served from "/"). NO CALLER in the current
studio/app UI: still reachable over HTTP, but nothing links to it -- rewire or retire is an open call.
"""
from clozn.server import app as ctx


def try_post(h, p, body):
    if p.startswith("/runs/") and p.endswith("/span_receipt"):   # span receipt: ablate one context span, both arms
        rid = p[len("/runs/"):-len("/span_receipt")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        # Both regen arms use the product model (the same gate /receipt applies for regen/both); the
        # forced arm degrades honestly INSIDE the receipt when score_tokens is absent.
        if not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "chat", None)):
            h._json(503, {"error": "span_receipt requires a ready product model worker"})
            return True
        from clozn.receipts.span_receipt import SpanSpecError, span_receipt
        try:
            out = span_receipt(run, body, ctx.active_sub(h))
        except SpanSpecError as e:
            h._json(400, {"error": str(e)})
            return True
        except Exception as e:
            h._json(500, {"error": f"span_receipt failed: {type(e).__name__}: {e}"})
            return True
        if out is None:
            h._json(500, {"error": "span_receipt failed (the baseline or ablated replay could not be "
                                   "generated)"})
            return True
        h._json(200, out)
        return True
    return False
