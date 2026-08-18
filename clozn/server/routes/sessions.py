"""F1: HTTP surface over clozn.runs.sessions -- session records as a first-class conversation entity.

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4); no edit to clozn/server/app.py. Spliced in
before the generic `GET /runs/<id>` fallback like every other autoloaded family, though nothing here
shares the `/runs/` prefix -- this module's own paths are all under `/sessions`, so ordering relative to
that fallback is not actually load-bearing for THIS module (only for the `/runs/...` families); it is
still spliced the same way the autoloader always splices GET modules, for one uniform rule.

Wire shape
----------
  GET  /sessions                       -> list_sessions() (?include_hidden=1, ?limit=N)
  GET  /sessions/<id>                  -> get_session(id, materialize=True); 404 if it does not exist
  GET  /sessions/<id>/runs             -> list_session_runs(id) (?cursor=..., ?limit=N, ?include_derived=1)
  POST /sessions                       -> create_session(body) -- idempotent; see sessions.py's
                                           concurrency contract for what "the same id twice" does
  POST /sessions/<id>                  -> update_session(id, title=..., visibility=...); 404 if absent
  POST /sessions/<id>/delete           -> delete_session(id) -- entity only, member runs untouched

There is no DELETE/PATCH verb here because clozn/server/app.py's handler only ever dispatches do_GET/
do_POST (confirmed by grep -- no do_DELETE/do_PATCH exists anywhere in the module); update and delete are
both POST actions, matching the rest of this codebase's route surface (e.g. this module's own
POST /sessions/<id>/delete above).

A caller's raw `session_id` -- from the JSON body, the path segment, or the `X-Clozn-Session-Id` header
-- is normalized identically regardless of source: clozn.runs.sessions' own `accept_key=True` rule
(create_session/get_session/etc. all route through association.session_key with the default), NOT
association.request_session's always-digest header shim (that shim exists for the cross-protocol
"treat any header value as raw text" case on the run-recording hot path; this route's whole job is
managing the entity BY id, so accepting an already-opaque key as-is is the correct, and simpler,
behavior here).
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlsplit

from clozn.runs import sessions

CLOZN_ROUTE_AUTOLOAD = True

_PREFIX = "/sessions"


def _one(query: dict, name: str):
    values = query.get(name) or []
    return values[-1] if values else None


def _truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _query(h) -> dict:
    return parse_qs(urlsplit(h.path).query, keep_blank_values=True)


def _session_id_from(h, query: dict) -> "str | None":
    """The raw candidate session id from wherever a GET caller supplied one: query, then header. POST
    handlers read the JSON body instead (see try_post)."""
    value = _one(query, "session_id")
    if value is None:
        try:
            value = h.headers.get("X-Clozn-Session-Id")
        except Exception:
            value = None
    return value


def try_get(h, p):
    if p == _PREFIX:
        query = _query(h)
        limit_raw = _one(query, "limit")
        try:
            limit = 50 if limit_raw is None else int(limit_raw)
        except ValueError:
            h._json(400, {"error": "limit must be an integer"})
            return True
        h._json(200, {"sessions": sessions.list_sessions(
            limit=limit, include_hidden=_truthy(_one(query, "include_hidden")))})
        return True

    if not p.startswith(_PREFIX + "/"):
        return False
    rest = p[len(_PREFIX) + 1:]

    if rest.endswith("/runs"):
        session_id = rest[:-len("/runs")]
        query = _query(h)
        limit_raw = _one(query, "limit")
        try:
            limit = 50 if limit_raw is None else int(limit_raw)
        except ValueError:
            h._json(400, {"error": "limit must be an integer"})
            return True
        try:
            page = sessions.list_session_runs(
                session_id, cursor=_one(query, "cursor"), limit=limit,
                include_derived=_truthy(_one(query, "include_derived")),
            )
        except ValueError as exc:
            h._json(400, {"error": str(exc)})
            return True
        if page["session_id"] is None:
            h._json(400, {"error": "session_id must be a non-empty value"})
            return True
        h._json(200, page)
        return True

    # GET /sessions/<id> -- the session detail view. Lazily materializes a legacy session on first
    # look-up (see sessions.get_session's `materialize` docstring) so the SAME id resolves to a real row
    # on the next request instead of re-deriving it every time.
    session_id = rest
    doc = sessions.get_session(session_id, materialize=True)
    if doc is None:
        h._json(404, {"error": "session not found"})
        return True
    h._json(200, doc)
    return True


def try_post(h, p, body):
    body = body if isinstance(body, dict) else {}

    if p == _PREFIX:
        session_id = body.get("session_id")
        if session_id is None:
            try:
                session_id = h.headers.get("X-Clozn-Session-Id")
            except Exception:
                session_id = None
        client_id = body.get("client_id")
        if client_id is None:
            try:
                client_id = h.headers.get("X-Clozn-Client-Id")
            except Exception:
                client_id = None
        try:
            doc = sessions.create_session(
                session_id, client_id=client_id, title=body.get("title"),
                visibility=body.get("visibility", "visible"),
            )
        except sessions.SessionValueError as exc:
            h._json(400, {"error": str(exc)})
            return True
        h._json(200, {"ok": True, "session": doc})
        return True

    if not p.startswith(_PREFIX + "/"):
        return False
    rest = p[len(_PREFIX) + 1:]

    if rest.endswith("/delete"):
        session_id = rest[:-len("/delete")]
        try:
            result = sessions.delete_session(session_id)
        except sessions.SessionValueError as exc:
            h._json(400, {"error": str(exc)})
            return True
        h._json(200, result)
        return True

    session_id = rest
    try:
        doc = sessions.update_session(
            session_id,
            title=body.get("title"),
            visibility=body.get("visibility"),
        )
    except sessions.SessionValueError as exc:
        h._json(400, {"error": str(exc)})
        return True
    if doc is None:
        h._json(404, {"error": "session not found"})
        return True
    h._json(200, {"ok": True, "session": doc})
    return True
