"""GET /runs/<id>/card — the shareable receipt card: ONE self-contained HTML rendering of the SAME
bundle GET /runs/<id>/export returns. Replicates the export handler's exact assembly (routes/receipts.py:
get_run -> explain (M1, pure read) -> receipt_bundle.build) and adds only runlog.lineage(rid) — another
pure read of already-recorded runs, zero generation — so the card can draw its family tree. The card is a
RENDERING of that bundle, never a new computation: receipts/lens data appear only if the record already
carries them; otherwise the card states their honest absence.

Registered in clozn/server/app.py before the generic `/runs/<id>` fallback. The route is intentionally
available over HTTP even though Studio does not link to it.
"""


def try_get(h, p):
    if p.startswith("/runs/") and p.endswith("/card"):
        import clozn.runs.store as runlog
        import clozn.receipts.explain as explain
        import clozn.receipts.bundle as receipt_bundle
        from clozn.server.card_html import render_card
        rid = p[len("/runs/"):-len("/card")]
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True
        try:
            xr = explain.explain(run)                # M1: pure read/reshape, no generation
        except Exception:
            xr = None
        bundle = receipt_bundle.build(run, explain=xr)
        try:
            bundle["lineage"] = runlog.lineage(rid)  # pure journal read (parent/children ids)
        except Exception:
            bundle["lineage"] = None
        safe_rid = "".join(c for c in str(rid) if c.isalnum() or c in "_-")   # header-safe (defense in depth)
        h._send(200, render_card(bundle), "text/html; charset=utf-8",
               extra_headers={"Content-Disposition": 'inline; filename="' + safe_rid + '.card.html"'})
        return True
    return False
