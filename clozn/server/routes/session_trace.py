"""GET /sessions/<id>/trace -- F2: clozn.runs.session_trace.build_trace()'s HTTP surface.

A NEW file, not an extension of clozn/server/routes/sessions.py (F1's own route family), even though both
live under the shared `/sessions/` prefix: F1's route module already ends in a broad `/sessions/<id>`
fallback (its own session-detail GET) inside ITS `try_get`, and trace composition (paginate, diff every
adjacent turn pair, run the diagnostic rule engine per turn, scan the whole session for "first went wrong"
candidates) is substantial enough machinery to deserve its own module -- matching this codebase's existing
convention of one file per feature-scoped route family even when several share a URL prefix (e.g.
clozn/server/routes/context_receipt.py and clozn/server/routes/runs.py both live under `/runs/`).

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4); no edit to clozn/server/app.py or to
clozn/server/routes/sessions.py. Autoload dispatch order is alphabetical by module name
(clozn/server/routes/_autoload.py's own documented contract) -- "session_trace.py" sorts before
"sessions.py" (`_` < `s` in ASCII), so THIS module's `try_get` always gets first refusal on
`/sessions/<id>/trace` before F1's own `/sessions/<id>` catch-all in sessions.py ever sees it.
tests/test_session_trace.py proves this explicitly rather than leaving it an incidental, easily-broken
property of filename sort order.

Wire shape:
  GET /sessions/<id>/trace[?cursor=<opaque>][&limit=N]
      -> 200 <clozn.session-trace.v1>
      -> 400 a malformed cursor, or a non-integer limit
      -> 404 the session honestly does not exist (see clozn.runs.sessions.get_session's contract --
         no explicit row AND no member run for this id)

Never touches a live substrate/engine (clozn.runs.session_trace.build_trace is engine-free by
construction -- see that module's own docstring); this route accordingly never imports or calls
`clozn.server.app.active_sub` or anything resembling it. tests/test_session_trace.py proves this with a
substrate double whose `__getattr__` raises on any access, the same trick
tests/test_token_workbench.py's `test_route_never_calls_engine_or_scorer` already uses.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

from clozn.runs import session_trace

CLOZN_ROUTE_AUTOLOAD = True

_PREFIX = "/sessions/"
_SUFFIX = "/trace"


def _one(query: dict, name: str):
    values = query.get(name) or []
    return values[-1] if values else None


def try_get(h, p):
    if not (p.startswith(_PREFIX) and p.endswith(_SUFFIX)):
        return False
    session_id = p[len(_PREFIX):-len(_SUFFIX)]

    query = parse_qs(urlsplit(h.path).query, keep_blank_values=True)
    limit_raw = _one(query, "limit")
    try:
        limit = session_trace.DEFAULT_LIMIT if limit_raw is None else int(limit_raw)
    except ValueError:
        h._json(400, {"error": "limit must be an integer"})
        return True

    try:
        document = session_trace.build_trace(session_id, cursor=_one(query, "cursor"), limit=limit)
    except ValueError as exc:
        h._json(400, {"error": str(exc)})
        return True

    if document is None:
        h._json(404, {"error": "session not found"})
        return True
    h._json(200, document)
    return True
