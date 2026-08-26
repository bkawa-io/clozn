"""F1: first-class session records.

Promotes `X-Clozn-Session-Id` from a lookup column (`runs.session_key`, migration 2 in
clozn/runs/migrations.py) into a persisted conversation ENTITY -- the substrate for session traces (F2)
and the conversation investigation view (F3). See migrations.py's `_migration_0004_sessions` docstring
for the schema rationale; this module is the read/write API over that table.

RUN IMMUTABILITY IS PRESERVED
------------------------------
A session is a new entity that REFERENCES runs by `runs.session_key == sessions.id` equality -- it never
embeds run content or even a list of run ids, and nothing here ever rewrites an existing run row. A run's
own document, once written by `clozn.runs.store.record()`, is exactly as immutable after this module
exists as before it.

WHAT'S STORED VS. WHAT'S DERIVED
---------------------------------
Only session-level identity/metadata is a stored column (id, created_ts, created_at, client_key, title,
privacy, materialized_from -- see clozn.session.v1). `first_activity_ts`, `last_activity_ts`, and
`run_count` are NEVER stored: every read recomputes them fresh via `MIN/MAX(recorded_ts)`/`COUNT(*)` over
`runs WHERE session_key = ?` (indexed by migration 2's `runs_session_latest_idx`). This is deliberate,
not an oversight -- see migrations.py's docstring for the two reasons (record()'s hot-path blast radius,
and staleness under clozn.runs.mutations.redact_run/delete_run, which this module has NO hook into and
therefore can never drift out of sync with).

SESSION IDENTITY, COLLISIONS, AND CROSS-CLIENT REUSE (read this before wiring a new caller)
---------------------------------------------------------------------------------------------
A session's id IS `clozn.runs.association.session_key()`'s output -- the SAME normalization already
applied to every run's `session_key` column. Concretely:

  * A caller-supplied raw token (e.g. a wrapper script's own conversation id) is HMAC-SHA256-digested
    against the install-local secret (`store.association_secret()`), truncated to 24 hex chars, and
    prefixed `session_`. Two DIFFERENT clients supplying the IDENTICAL raw token therefore land on the
    IDENTICAL session id -- this is intentional cross-client reuse (e.g. a CLI wrapper and a browser tab
    both pointed at conversation "thread-42" are treated as one continuous logical session), not an
    accidental collision. It exactly mirrors behavior `runs.session_key` already had before this module
    existed (`find_runs`/`latest_run` already merged such runs) -- this module gives that merge point a
    first-class row, it does not change what merges.
  * A caller-supplied value that ALREADY matches the opaque `session_[0-9a-f]{24}` shape is accepted
    as-is (not re-digested) -- same `accept_key=True` rule `association.session_key()` already applies
    everywhere else. Studio can therefore hand a session id it minted itself (see below) straight back
    on a later call and land on the same row.
  * `session_id=None` mints a FRESH Studio-generated id: `secrets.token_hex(12)` (96 bits) in the same
    `session_<24hex>` shape, NOT HMAC-digested -- there is no caller-supplied raw text to protect, so
    digesting a random value would be theatre. A caller cannot tell by inspection whether an id was
    digested from raw text or freshly minted, which is the point: both are equally opaque.
  * A genuine cryptographic collision between two DIFFERENT raw tokens (or between a minted id and a
    digest) is the same negligible probability as any 96-bit random-oracle output colliding -- not a
    behavior this module special-cases beyond what SQLite's PRIMARY KEY constraint already guarantees
    (see the concurrency contract below).

CONCURRENCY: TWO THREADS CREATING THE SAME SESSION ID YIELD ONE ROW
-----------------------------------------------------------------------
`create_session()` (and the lazy-backfill path, `materialize_session()`) both funnel through
`_insert_or_fetch()`, which does `INSERT OR IGNORE INTO sessions(id, ...) VALUES (...)` followed by a
`SELECT` of whatever row now exists -- inside `with closing(store._connect()) as db, db:` (the same
auto-commit-on-clean-exit pattern `store._put()` uses). SQLite's PRIMARY KEY constraint plus its own
writer serialization (busy_timeout retries, exactly like every other write in this package) means a
losing racer's `INSERT OR IGNORE` is a silent no-op, and its subsequent `SELECT` reads the WINNER's row --
never a duplicate, never an exception, never a lost create. First-writer-wins: a racer's own
title/client_key/visibility are discarded if another thread's create landed first; call `update_session()`
afterward for an explicit, unambiguous metadata change. See tests/test_sessions.py for the flake-hunted
proof (25 trials x 20 threads racing one id, house standard per tests/test_runs_store_concurrency.py).

A RUN BELONGS TO AT MOST ONE SESSION
--------------------------------------
Already true structurally: `runs.session_key` is a single scalar column (migration 2), not a set. This
module does not change that. It deliberately does NOT introduce a run<->session join table -- doing so
would be exactly the "import" mechanism the F1 spec says NOT to build yet. Because a session here is
IDENTITY/metadata only (never a stored run-id list), adding a future join table for an explicit multi-
session import is additive and structurally unblocked: it would sit beside `sessions`/`runs`, needing no
change to either.

SESSIONLESS RUNS
------------------
A run with `session_key IS NULL` is untouched by anything in this module -- it was already a valid,
first-class state before F1 (BACKLOG's "state, not an error" framing), and nothing here requires a
session to exist for a run to be recorded, queried, or read.

DELETION
---------
`delete_session()` removes ONLY the `sessions` row. Member runs are not touched, not requeued, not
re-tagged -- they keep their `session_key` value and remain exactly as queryable via `list_session_runs`/
`get_session`'s derived-aggregate path as any other session (this module never distinguishes "has an
explicit row" from "derived only" in what a run can do). This mirrors the run/trace separation already in
this package: `clozn.runs.mutations.delete_run` never deletes a shared trace blob another run still
references; `redact_run` never deletes the trace blob at all when scoped to specific literals. A session
row is a LABEL on a group of runs, not a container that owns them -- removing the label never removes the
labeled evidence. A caller that wants the runs themselves gone still reaches for
`clozn.runs.mutations.delete_run`/`redact_run` per run; this module's deletion contract stops at the
session entity, exactly as scoped by the F1 spec ("runs keep their own lifecycle").

BACKFILL: LAZY, NOT MIGRATION-TIME (see migrations.py for the full rationale)
---------------------------------------------------------------------------------
`materialize_session()` persists an identity row for a legacy `session_key` that already has member runs
but no `sessions` row yet -- `created_ts` becomes an honest approximation (earliest member run's
`recorded_ts`), `client_key` is included only if every member run agrees on one value (never guessed at
when ambiguous), and `materialized_from` is stamped `"lazy_backfill"` so a reader can tell the difference
from a `"explicit"` row whose `created_ts` is the true creation time. `get_session(..., materialize=True)`
is the usual trigger (e.g. GET /sessions/<id> in clozn/server/routes/sessions.py); plain reads never write.
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

def _new_document(session_id: str, *, client_key, title, visibility: str,
                   materialized_from: str) -> dict:
    now = time.time()
    doc = {
        "schema_version": SCHEMA_NAME,
        "id": session_id,
        "created_ts": now,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
        "privacy": {"visibility": visibility},
        "materialized_from": materialized_from,
    }
    if client_key:
        doc["client_key"] = client_key
    if title:
        doc["title"] = title
    schemas.validate(doc, SCHEMA_NAME)
    return doc


def _lazy_document(session_id: str, agg: dict) -> dict:
    """The document `materialize_session` would persist for a legacy session_key -- built but not
    written; `get_session(materialize=False)` uses this directly for a pure read."""
    created_ts = agg["first_activity_ts"]
    client_key = _lazy_client_key(session_id)
    doc = {
        "schema_version": SCHEMA_NAME,
        "id": session_id,
        "created_ts": created_ts,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(created_ts)),
        "privacy": {"visibility": "visible"},
        "materialized_from": "lazy_backfill",
    }
    if client_key:
        doc["client_key"] = client_key
    schemas.validate(doc, SCHEMA_NAME)
    return doc


def _row_to_document(row) -> dict:
    doc = {
        "schema_version": SCHEMA_NAME,
        "id": row["id"],
        "created_ts": float(row["created_ts"]),
        "created_at": row["created_at"],
        "privacy": json.loads(row["privacy_json"]),
        "materialized_from": row["materialized_from"],
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


def _lazy_client_key(session_key_value: str) -> "str | None":
    """The member runs' client_key IF every run that has one agrees on the same value -- else None. An
    ambiguous legacy session (runs from more than one client_key) never has a client_key fabricated for
    it; see the module docstring."""
    with closing(store._connect()) as db:
        rows = db.execute(
            "SELECT DISTINCT client_key FROM runs WHERE session_key = ? AND client_key IS NOT NULL",
            (session_key_value,),
        ).fetchall()
    keys = {r["client_key"] for r in rows}
    return next(iter(keys)) if len(keys) == 1 else None


# --------------------------------------------------------------------------------------------- write helpers

def _insert_or_fetch(doc: dict) -> dict:
    """Race-safe create-or-fetch: INSERT OR IGNORE then read back whatever row now exists. See the module
    docstring's concurrency section -- this is the ONE write path both create_session() and
    materialize_session() funnel through, so the "one row" guarantee has exactly one implementation."""
    store._ensure()
    privacy_json = json.dumps(doc["privacy"], sort_keys=True, separators=(",", ":"))
    with closing(store._connect()) as db, db:
        db.execute(
            "INSERT OR IGNORE INTO sessions"
            "(id, created_ts, created_at, client_key, title, privacy_json, materialized_from) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (doc["id"], doc["created_ts"], doc["created_at"], doc.get("client_key"),
             doc.get("title"), privacy_json, doc["materialized_from"]),
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
                        visibility=visibility, materialized_from="explicit")
    return _insert_or_fetch(doc)


def materialize_session(session_id) -> "dict | None":
    """Persist the lazily-derived identity row for a session that already has member runs but no
    `sessions` row -- the F1 spec's chosen "lazy" backfill path (see migrations.py for why migration-time
    backfill was rejected). Returns None if the session does not exist at all (no explicit row AND no
    member runs). Idempotent and race-safe via the same `_insert_or_fetch` path `create_session` uses --
    a second call, or a concurrent one, is a harmless no-op that returns the same row."""
    key = _normalize_lookup(session_id)
    if key is None:
        return None
    store._ensure()
    with closing(store._connect()) as db:
        existing = db.execute("SELECT 1 FROM sessions WHERE id = ?", (key,)).fetchone()
    if existing is not None:
        return get_session(key)
    agg = _activity_aggregate(key)
    if agg is None:
        return None
    persisted = _insert_or_fetch(_lazy_document(key, agg))
    persisted.update(agg)
    return persisted


def get_session(session_id, *, materialize: bool = False) -> "dict | None":
    """The session's document, with derived `first_activity_ts`/`last_activity_ts`/`run_count` merged in
    when it has at least one member run (omitted, never zero-padded, otherwise). Returns None only when
    NEITHER an explicit row NOR any member run exists for this id -- a session identifier nobody has ever
    used is honestly absent, not an empty shell.

    `materialize=False` (default) is a pure read: a legacy session_key with runs but no row gets a
    document built on the fly (`materialized_from: "lazy_backfill"`) and NOTHING is written. Pass
    `materialize=True` (e.g. from the session detail route) to persist that document so the next lookup
    finds a real row -- see materialize_session().
    """
    key = _normalize_lookup(session_id)
    if key is None:
        return None
    store._ensure()
    with closing(store._connect()) as db:
        row = db.execute("SELECT * FROM sessions WHERE id = ?", (key,)).fetchone()
    if row is None:
        agg = _activity_aggregate(key)
        if agg is None:
            return None
        if materialize:
            return materialize_session(key)
        doc = _lazy_document(key, agg)
        doc.update(agg)
        return doc
    doc = _row_to_document(row)
    agg = _activity_aggregate(key)
    if agg is not None:
        doc.update(agg)
    return doc


def update_session(session_id, *, title=None, visibility=None) -> "dict | None":
    """Explicit metadata mutation (rename and/or change visibility). Returns None if no `sessions` row
    exists yet -- update never fabricates a row (materialize_session/create_session do that on purpose;
    conflating the two would make an update silently double as a create). `title=""` (or an all-whitespace
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
    """Remove ONLY the session entity row. Member runs are never touched -- see the module docstring's
    "DELETION" section. Idempotent: deleting an id with no explicit row (e.g. a session only ever seen
    via the lazy-derived view) still reports ok=True with deleted_row=False, since the end state --
    'no sessions row for this id' -- is identical either way."""
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
    """Every known session (explicit `sessions` rows, plus any session_key referenced by at least one run
    that was never materialized), newest-activity-first. Bounded, not cursor-paginated -- unlike
    list_session_runs, where a SINGLE session can hold thousands of turns, the total number of distinct
    sessions on one install is already capped by store.KEEP (1000 total runs store-wide), so a bounded
    query plus an in-Python merge/sort is the right amount of machinery, not premature."""
    store._ensure()
    with closing(store._connect()) as db:
        explicit_rows = db.execute("SELECT * FROM sessions").fetchall()
        agg_rows = db.execute(
            "SELECT session_key, MIN(recorded_ts) AS first_ts, MAX(recorded_ts) AS last_ts, "
            "COUNT(*) AS n FROM runs WHERE session_key IS NOT NULL GROUP BY session_key"
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
    seen = set()
    out = []
    for row in explicit_rows:
        doc = _row_to_document(row)
        agg = agg_by_key.get(doc["id"])
        if agg is not None:
            doc["first_activity_ts"] = float(agg["first_ts"])
            doc["last_activity_ts"] = float(agg["last_ts"])
            doc["run_count"] = int(agg["n"])
        if doc["id"] in previews:
            doc["preview"] = previews[doc["id"]]
        seen.add(doc["id"])
        out.append(doc)
    for key, agg in agg_by_key.items():
        if key in seen:
            continue
        derived = {
            "first_activity_ts": float(agg["first_ts"]),
            "last_activity_ts": float(agg["last_ts"]),
            "run_count": int(agg["n"]),
        }
        doc = _lazy_document(key, derived)
        doc.update(derived)
        if key in previews:
            doc["preview"] = previews[key]
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
    (already relied on by `store.runs_after`), and migration 2's `runs_session_latest_idx`, so this is a
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
    "materialize_session",
    "resolve_session_id",
    "update_session",
]
