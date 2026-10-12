"""Persisted conversation sessions.

A session row stores identity and user-managed metadata. Runs refer to it through
``runs.session_key = sessions.id``; the session never embeds run content or a list
of run IDs. Recording the first run for a session creates the session row in the
same SQLite transaction. Activity timestamps and counts remain derived from runs,
so they cannot become stale.

Session IDs are opaque, install-local keys. The same caller token resolves to the
same key across clients, and a run belongs to at most one session. Runs without a
session remain valid. Deleting session metadata does not delete its run evidence.
"""
from __future__ import annotations

import json
import secrets
import time
from contextlib import closing

from clozn import schemas

from . import association, store
from .summaries import _summary

SCHEMA_NAME = "clozn.session.v1"
_VISIBILITIES = ("visible", "hidden")
_DERIVED_SOURCES = ("replay", "branch", "fork")


class SessionValueError(ValueError):
    """A caller-supplied session argument (id, title, visibility, ...) is malformed. Mirrors
    clozn.runs.association.AssociationValueError / clozn.runs.mutations.MutationError's role: a typed,
    catchable failure a route layer turns into a 400, never a silent no-op."""


# ---------------------------------------------------------------------------------------------- identity

def resolve_session_id(session_id) -> str:
    """Normalize a caller-supplied session identifier, or mint a fresh Studio-generated one when
    `session_id` is None. See the module docstring's "SESSION IDENTITY, COLLISIONS, AND CROSS-CLIENT
    REUSE" section for the full contract. Raises SessionValueError for a supplied-but-empty value --
    unlike `association.session_key()`, which returns None for that case (fine for an OPTIONAL filter
    argument in store.find_runs/runs_after, wrong here where an explicit id was clearly intended)."""
    if session_id is None:
        return association.SESSION_PREFIX + secrets.token_hex(12)
    key = association.session_key(session_id)
    if key is None:
        raise SessionValueError("session_id must be a non-empty value")
    return key


def _normalize_lookup(session_id) -> "str | None":
    """Same normalization as resolve_session_id, but for LOOKUP paths (get/list/delete) where an
    empty/None input honestly means 'no such session' rather than 'mint me a new one'."""
    if session_id is None:
        return None
    return association.session_key(session_id)


# ------------------------------------------------------------------------------------------ row <-> document

def _new_document(session_id: str, *, client_key, title, visibility: str) -> dict:
    now = time.time()
    doc = {
        "schema_version": SCHEMA_NAME,
        "id": session_id,
        "created_ts": now,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "privacy": {"visibility": visibility},
    }
    if client_key:
        doc["client_key"] = client_key
    if title:
        doc["title"] = title
    schemas.validate(doc, SCHEMA_NAME)
    return doc


def _row_to_document(row) -> dict:
    doc = {
        "schema_version": SCHEMA_NAME,
        "id": row["id"],
        "created_ts": float(row["created_ts"]),
        "created_at": row["created_at"],
        "privacy": json.loads(row["privacy_json"]),
    }
    if row["client_key"]:
        doc["client_key"] = row["client_key"]
    if row["title"]:
        doc["title"] = row["title"]
    return doc


# -------------------------------------------------------------------------------------- derived run activity

def _activity_aggregate(session_key_value: str) -> "dict | None":
    """`{first_activity_ts, last_activity_ts, run_count}` over every run tagged with this session_key --
    None (not zeros) when the session has no member runs at all, so callers can tell 'exists, zero runs
    yet' apart from 'no such session'. Includes replay/branch/fork-derived runs deliberately: a session's
    OWN existence must not depend on which run kinds happen to compose it, even though
    `list_session_runs`'s default view excludes them for readability."""
    store._ensure()
    with closing(store._connect()) as db:
        row = db.execute(
            "SELECT MIN(recorded_ts) AS first_ts, MAX(recorded_ts) AS last_ts, COUNT(*) AS n "
            "FROM runs WHERE session_key = ?",
            (session_key_value,),
        ).fetchone()
    if row is None or not row["n"]:
        return None
    return {
        "first_activity_ts": float(row["first_ts"]),
        "last_activity_ts": float(row["last_ts"]),
        "run_count": int(row["n"]),
    }


# --------------------------------------------------------------------------------------------- write helpers

def _insert_or_fetch(doc: dict) -> dict:
    """Race-safe create-or-fetch: INSERT OR IGNORE then read back whatever row now exists. See the module
    database's primary-key constraint makes concurrent creates converge on one row."""
    store._ensure()
    privacy_json = json.dumps(doc["privacy"], sort_keys=True, separators=(",", ":"))
    with closing(store._connect()) as db, db:
        db.execute(
            "INSERT OR IGNORE INTO sessions"
            "(id, created_ts, created_at, client_key, title, privacy_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (doc["id"], doc["created_ts"], doc["created_at"], doc.get("client_key"),
             doc.get("title"), privacy_json),
        )
        row = db.execute("SELECT * FROM sessions WHERE id = ?", (doc["id"],)).fetchone()
    return _row_to_document(row)


# ------------------------------------------------------------------------------------------------- public API

def create_session(session_id=None, *, client_id=None, title=None, visibility: str = "visible") -> dict:
    """Idempotently create (or, if the id already exists, simply fetch) a session and return its stored
    document -- WITHOUT the derived activity fields (call get_session for those; a brand-new session has
    none to report anyway). `client_id` is a raw client identifier OR an already-opaque `client_...` key
    (clozn.runs.association.client_key()'s own accept-both rule); omit it for a session with no resolvable
    client identity yet."""
    if visibility not in _VISIBILITIES:
        raise SessionValueError(f"visibility must be one of {_VISIBILITIES}")
    if title is not None and not isinstance(title, str):
        raise SessionValueError("title must be a string")
    cleaned_title = title.strip() if isinstance(title, str) else None
    key = resolve_session_id(session_id)
    resolved_client = association.client_key(client_id) if client_id is not None else None
    doc = _new_document(key, client_key=resolved_client, title=cleaned_title or None,
                        visibility=visibility)
    return _insert_or_fetch(doc)


def get_session(session_id) -> "dict | None":
    """The session's document, with derived `first_activity_ts`/`last_activity_ts`/`run_count` merged in
    when it has at least one member run (omitted otherwise). Returns None when the
    session row does not exist.
    """
    key = _normalize_lookup(session_id)
    if key is None:
        return None
    store._ensure()
    with closing(store._connect()) as db:
        row = db.execute("SELECT * FROM sessions WHERE id = ?", (key,)).fetchone()
    if row is None:
        return None
    doc = _row_to_document(row)
    agg = _activity_aggregate(key)
    if agg is not None:
        doc.update(agg)
    return doc


def update_session(session_id, *, title=None, visibility=None) -> "dict | None":
    """Explicit metadata mutation (rename and/or change visibility). Returns None if no `sessions` row
    exists. `title=""` (or an all-whitespace
    string) clears the title back to unset, matching "omit, never null-pad" -- the stored column becomes
    NULL, not an empty string."""
    key = _normalize_lookup(session_id)
    if key is None:
        raise SessionValueError("session_id must be given")
    sets: list = []
    params: list = []
    if title is not None:
        if not isinstance(title, str):
            raise SessionValueError("title must be a string")
        sets.append("title = ?")
        params.append(title.strip() or None)
    if visibility is not None:
        if visibility not in _VISIBILITIES:
            raise SessionValueError(f"visibility must be one of {_VISIBILITIES}")
        sets.append("privacy_json = ?")
        params.append(json.dumps({"visibility": visibility}, sort_keys=True, separators=(",", ":")))
    if not sets:
        return get_session(key)
    store._ensure()
    with closing(store._connect()) as db, db:
        db.execute(f"UPDATE sessions SET {', '.join(sets)} WHERE id = ?", (*params, key))
        row = db.execute("SELECT * FROM sessions WHERE id = ?", (key,)).fetchone()
    if row is None:
        return None
    doc = _row_to_document(row)
    agg = _activity_aggregate(key)
    if agg is not None:
        doc.update(agg)
    return doc


def delete_session(session_id) -> dict:
    """Remove only the session metadata row. Member runs keep their own lifecycle."""
    key = _normalize_lookup(session_id)
    if key is None:
        raise SessionValueError("session_id must be given")
    store._ensure()
    with closing(store._connect()) as db, db:
        before = db.total_changes
        db.execute("DELETE FROM sessions WHERE id = ?", (key,))
        deleted = db.total_changes > before
    return {"ok": True, "action": "delete_session", "session_id": key, "deleted_row": deleted}


def list_sessions(*, limit: int = 50, include_hidden: bool = False) -> list:
    """Persisted sessions, newest-activity-first.

    `run_count` includes every member run while `turn_count` excludes derived
    replay/branch/fork runs. The result is bounded because the local run store is
    itself bounded.
    """
    store._ensure()
    with closing(store._connect()) as db:
        session_rows = db.execute("SELECT * FROM sessions").fetchall()
        agg_rows = db.execute(
            "SELECT session_key, MIN(recorded_ts) AS first_ts, MAX(recorded_ts) AS last_ts, "
            "COUNT(*) AS n, "
            "SUM(CASE WHEN source NOT IN (?, ?, ?) THEN 1 ELSE 0 END) AS turn_n "
            "FROM runs WHERE session_key IS NOT NULL GROUP BY session_key",
            _DERIVED_SOURCES,
        ).fetchall()
        preview_rows = db.execute(
            "SELECT session_key, payload_json FROM runs "
            "WHERE session_key IS NOT NULL AND source NOT IN (?, ?, ?) "
            "ORDER BY recorded_ts ASC, id ASC",
            _DERIVED_SOURCES,
        ).fetchall()
    previews = {}
    for row in preview_rows:
        key = row["session_key"]
        if key in previews:
            continue
        try:
            summary = _summary(json.loads(row["payload_json"]))
        except Exception:
            continue
        run_id = summary.get("id")
        prompt_summary = summary.get("prompt_summary")
        response_summary = summary.get("response_summary")
        if not all(isinstance(value, str) for value in (run_id, prompt_summary, response_summary)):
            continue
        previews[key] = {
            "run_id": run_id,
            "prompt_summary": prompt_summary,
            "response_summary": response_summary,
        }
    agg_by_key = {r["session_key"]: r for r in agg_rows}
    out = []
    for row in session_rows:
        doc = _row_to_document(row)
        agg = agg_by_key.get(doc["id"])
        if agg is not None:
            doc["first_activity_ts"] = float(agg["first_ts"])
            doc["last_activity_ts"] = float(agg["last_ts"])
            doc["run_count"] = int(agg["n"])
            doc["turn_count"] = int(agg["turn_n"] or 0)
        else:
            doc["turn_count"] = 0
        if doc["id"] in previews:
            doc["preview"] = previews[doc["id"]]
        out.append(doc)
    if not include_hidden:
        out = [d for d in out if d.get("privacy", {}).get("visibility") != "hidden"]
    out.sort(key=lambda d: d.get("last_activity_ts", d["created_ts"]), reverse=True)
    return out[:max(0, int(limit))]


def list_session_runs(session_id, *, cursor=None, limit: int = 50,
                      include_derived: bool = False) -> dict:
    """Ordered (oldest-first, conversational reading order), PAGINATED run query scoped to exactly one
    session -- the substrate F2/F3 read from instead of re-deriving every run in a long agent session on
    every page. Uses the SAME opaque `(recorded_ts, id)` cursor as `store.encode_cursor`/`decode_cursor`
    (already relied on by `store.runs_after`), and `runs_session_latest_idx`, so this is a
    bounded index range scan even for a session with thousands of turns -- deliberately NOT built on top
    of `store.runs_after`, whose SQL has no `WHERE session_key = ...` at all (it is a watcher-oriented
    full-table scan that filters in Python after fetching every row past the cursor; fine for `clozn
    watch` tailing new events across the WHOLE store, wrong for paging deep into one specific session).

    Returns `{"session_id", "runs", "next_cursor", "count"}`. `next_cursor` is None once fewer than
    `limit` rows were returned (the session is exhausted); otherwise it is set even though there might
    turn out to be zero further rows -- the same tolerant convention `store.runs_after` already uses,
    self-terminating on the next call rather than requiring an extra existence probe now.
    """
    key = _normalize_lookup(session_id)
    if key is None:
        return {"session_id": None, "runs": [], "next_cursor": None, "count": 0}
    wanted = max(1, min(1000, int(limit)))
    after_ts, after_id = store.decode_cursor(cursor) if cursor else (float("-inf"), "")
    clauses = ["session_key = ?", "(recorded_ts > ? OR (recorded_ts = ? AND id > ?))"]
    params: list = [key, after_ts, after_ts, after_id]
    if not include_derived:
        clauses.append("source NOT IN (" + ", ".join("?" for _ in _DERIVED_SOURCES) + ")")
        params.extend(_DERIVED_SOURCES)
    where = " AND ".join(clauses)
    store._ensure()
    with closing(store._connect()) as db:
        rows = db.execute(
            f"SELECT recorded_ts, id, payload_json FROM runs WHERE {where} "
            "ORDER BY recorded_ts ASC, id ASC LIMIT ?",
            (*params, wanted),
        ).fetchall()
    out = []
    last_ts, last_id = after_ts, after_id
    for row in rows:
        try:
            run = json.loads(row["payload_json"])
        except Exception:
            continue
        run.setdefault("recorded_ts", float(row["recorded_ts"]))
        out.append(_summary(run))
        last_ts, last_id = float(row["recorded_ts"]), str(row["id"])
    next_cursor = store.encode_cursor(last_ts, last_id) if len(rows) >= wanted and rows else None
    return {"session_id": key, "runs": out, "next_cursor": next_cursor, "count": len(out)}


__all__ = [
    "SCHEMA_NAME",
    "SessionValueError",
    "create_session",
    "delete_session",
    "get_session",
    "list_session_runs",
    "list_sessions",
    "resolve_session_id",
    "update_session",
]
