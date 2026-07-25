"""The Run Log: list runs, fetch one, and its structural GET-only views (timeline / lineage / family /
confidence spans) -- all zero-generation reads over clozn.runs.store. Mechanical extraction of the
matching `if p == ...` / `if p.startswith("/runs/") and p.endswith(...)` branches out of
clozn.server.app's do_GET; behavior unchanged.

NOTE on the generic "/runs/<id>" fallback: it is deliberately NOT here yet. GET /runs/<id>/export
(receipts.py) is still checked in app.py's legacy dispatch until that family is split out, and a generic
prefix match here would shadow it (both start with "/runs/"). The fallback moves into this module only
once every more-specific /runs/<id>/<suffix> GET has its own registered family ahead of this one in the
dispatch order -- see app.py's `_GET_ROUTES` comment.
"""
from urllib.parse import parse_qs, urlsplit


def _one(query: dict, name: str):
    values = query.get(name) or []
    return values[-1] if values else None


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _association_error(h, exc):
    # Flat {"error": str} -- the codebase's dominant convention (~160 sites). The nested OpenAI shape
    # lives ONLY in routes/openai.py, where it is wire-compat, not house style.
    h._json(400, {"error": f"{exc} (field: {getattr(exc, 'field', 'association')})"})


def _selectors(h, query: dict) -> dict:
    """Validated raw selectors. Custom headers are opt-in; ordinary User-Agent is not a lookup token."""
    from clozn.runs.association import validate_selector
    client_id = _one(query, "client_id")
    session_id = _one(query, "session")
    if client_id is None:
        client_id = h.headers.get("X-Clozn-Client-Id")
    if session_id is None:
        session_id = h.headers.get("X-Clozn-Session-Id")
    if client_id is not None:
        client_id = validate_selector(client_id, "client_id")
    if session_id is not None:
        session_id = validate_selector(session_id, "session")
    client = _one(query, "client")
    if client is not None:
        client = validate_selector(client, "client")
    model = _one(query, "model")
    if model is not None:
        model = validate_selector(model, "model")
    return {"client": client, "client_id": client_id, "session_id": session_id, "model": model}


def try_get(h, p):
    if p == "/runs/latest":
        import clozn.runs.store as runlog
        query = parse_qs(urlsplit(h.path).query, keep_blank_values=True)
        try:
            selectors = _selectors(h, query)
        except Exception as exc:
            _association_error(h, exc)
            return True
        exact = bool(selectors["client_id"] or selectors["session_id"])
        if not exact and not selectors["client"]:
            h._json(400, {"error": {"message": "choose client_id, session, or client; global newest is /runs",
                                     "type": "invalid_request_error", "param": "association",
                                     "code": "association_selector_required"}})
            return True
        run = runlog.latest_run(**selectors, include_derived=_truthy(_one(query, "include_derived")))
        h._json(200, {
            "available": run is not None,
            "run": run,
            "association": {
                "exact": exact,
                "ambiguous": not exact,
                "selector": "session" if selectors["session_id"] else (
                    "client_id" if selectors["client_id"] else "client"
                ),
            },
        })
        return True
    if p == "/runs/watch":
        import clozn.runs.store as runlog
        query = parse_qs(urlsplit(h.path).query, keep_blank_values=True)
        try:
            selectors = _selectors(h, query)
            raw_limit = _one(query, "limit")
            limit = 100 if raw_limit is None else int(raw_limit)
            if not 1 <= limit <= 1000:
                raise ValueError("limit must be between 1 and 1000")
            page = runlog.runs_after(
                _one(query, "after"), limit=limit, **selectors,
                include_derived=_truthy(_one(query, "include_derived")),
            )
        except Exception as exc:
            if getattr(exc, "field", None):
                _association_error(h, exc)
            else:
                h._json(400, {"error": {"message": str(exc), "type": "invalid_request_error",
                                         "param": "after" if "cursor" in str(exc) else "limit",
                                         "code": "invalid_watch_query"}})
            return True
        h._json(200, page)
        return True
    if p == "/runs":                 # the Run Log -- every interaction, newest first (the Studio Runs page)
        import clozn.runs.store as runlog
        h._json(200, {"runs": runlog.list_runs(80)})
        return True
    if p.startswith("/runs/") and p.endswith("/timeline"):   # ordered RunEvent list -- zero generation
        import clozn.runs.store as runlog
        rid = p[len("/runs/"):-len("/timeline")]
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.runs import timeline as run_timeline
        h._json(200, {"run_id": rid, "events": run_timeline.timeline(run)})
        return True
    if p.startswith("/runs/") and p.endswith("/diagnosis"):  # evidence-only slow/cutoff explanation
        import clozn.runs.store as runlog
        rid = p[len("/runs/"):-len("/diagnosis")]
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.runs.diagnosis import diagnose
        h._json(200, diagnose(run, related_runs=runlog.iter_runs(limit=200)))
        return True
    if p.startswith("/runs/") and p.endswith("/lineage"):   # branch/replay ancestry + child tree
        import clozn.runs.store as runlog
        rid = p[len("/runs/"):-len("/lineage")]
        out = runlog.lineage(rid)
        if not out:
            h._json(404, {"error": "run not found"})
            return True
        h._json(200, out)
        return True
    if p.startswith("/runs/") and p.endswith("/family"):   # the WHOLE branch family as GET /runs-shaped
        # summaries -- the full lineage past the /runs 80-window, so the client's buildLineageFromRuns
        # builds the complete tree instead of the recent-runs slice. (Distinct from /lineage, which
        # returns the server-built ancestors/children/tree object.)
        import clozn.runs.store as runlog
        rid = p[len("/runs/"):-len("/family")]
        fam = runlog.lineage_family(rid)
        if fam is None:
            h._json(404, {"error": "run not found"})
            return True
        h._json(200, {"runs": fam})
        return True
    if p.startswith("/runs/") and p.endswith("/spans"):   # confidence spans -- the shape of the reply's certainty
        import clozn.runs.store as runlog
        rid = p[len("/runs/"):-len("/spans")]
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.runs import confidence_spans
        sp = confidence_spans.spans(run)
        h._json(200, {"run_id": rid, "spans": sp, "summary": confidence_spans.summarize(sp)})
        return True
    return False


def try_get_fallback(h, p):
    """The generic GET /runs/<id> catch-all -- registered LAST (see app.py's _GET_ROUTES), after every
    more-specific /runs/<id>/<suffix> family has had first refusal."""
    if p.startswith("/runs/"):
        import clozn.runs.store as runlog
        r = runlog.get_run(p.split("/runs/", 1)[1])
        h._json(200, r) if r else h._json(404, {"error": "run not found"})
        return True
    return False
