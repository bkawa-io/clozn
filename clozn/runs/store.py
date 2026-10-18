"""SQLite run metadata with content-addressed trace artifacts.

SQLite is authoritative.  A run's queryable metadata and JSON document live in the
``runs`` table; the potentially large normalized trace lives in an immutable SHA-256
blob.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import logging
import os
import re
import sqlite3
import secrets
import threading
import time
import uuid
from clozn._io import atomic_write_json
from clozn.paths import data_path

from . import schema
from .summaries import SUMMARY_FIELDS, _flags, _summ, _summary, confidence_facts
from .trace import (
    TRACE_KEYS,
    _clean_alt,
    _clean_alts,
    _clean_step,
    _entropy_from_probs,
    _float_or_none,
    _int_or_none,
    _logprob,
    _norm_trace,
    _rounded_prob,
    _steps_from_parallel,
    accumulate_ar_events,
    finish_reason_from_frames,
    raw_finish_reason_from_frames,
    generation_timing_from_frames,
    steps_to_trace,
)


# v1 intentionally does not reuse the historical ~/.clozn/runs journal.  Keep
# this as a module global because tests and embedders already inject a store by
# replacing RUNS_DIR.
RUNS_DIR = data_path("runs")
KEEP = 1000
_SAFE_RID = re.compile(r"^[A-Za-z0-9_-]+$")
_OPAQUE_CLIENT_KEY = re.compile(r"^client_[0-9a-f]{24}$")
_log = logging.getLogger(__name__)


class SchemaLockTimeout(RuntimeError):
    """Raised by `_ensure()` when the cross-process schema-init lock (`_ensure_schema_locked`) could not
    be acquired within `_SCHEMA_LOCK_TIMEOUT_S`. A typed, honest failure -- per the store's "omit, never
    null-pad" contract, a caller must never be handed a connection to a possibly half-initialized
    database just because a lock was slow. In practice this means another process or thread is (or was,
    and crashed mid-init -- SQLite's own hot-journal recovery cleans that up automatically on the next
    attempt) already bringing the SAME store up to date; retrying is the right move, not proceeding."""


_SCHEMA_LOCK_TIMEOUT_S = 30.0     # mirrors _connect()'s own busy_timeout convention below

# Per-process cache of db paths already confirmed to have the current schema. Keyed
# by path, not a bare bool: tests legitimately repoint the module-global RUNS_DIR at a fresh tmp_path per
# test within the SAME process, and a bare "already initialized" flag would silently skip real schema
# creation for every subsequent test's brand-new store.
_schema_verified: set[str] = set()
_schema_verify_lock = threading.Lock()


def _db_path() -> str:
    return os.path.join(RUNS_DIR, "runs.sqlite3")


def _blob_root() -> str:
    return os.path.join(RUNS_DIR, "blobs", "sha256")


def _schema_lock_path() -> str:
    return os.path.join(RUNS_DIR, ".schema_init.lock")


def _ensure() -> None:
    """Create the run directory and ensure its current database schema exists.

    Concurrency: `_connect()`'s `PRAGMA journal_mode=WAL` is a genuine, one-time on-disk file-format
    conversion the first time it runs against a non-WAL file, and SQLite requires exclusive access to
    perform it -- unlike ordinary write-lock contention, that specific conversion does NOT retry through
    the busy handler / `busy_timeout` if another connection holds so much as a pending write transaction.
    Measured against unmodified code: concurrent first-time `_ensure()` calls (20 threads, barrier-
    synchronized, fresh store) failed at a 5-12% rate with `sqlite3.OperationalError: database is
    locked`, always at that exact PRAGMA (tests/test_runs_store_concurrency.py reproduces this). The fix
    below is two-layered: a per-process cache (`_schema_verified`) short-circuits everything after this
    process has itself confirmed the schema is current, and the one-time work -- the WAL switch plus
    current-schema bootstrap -- is serialized (both cross-thread AND cross-process) by
    `_ensure_schema_locked()`."""
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(_blob_root(), exist_ok=True)
    db_path = _db_path()
    if db_path in _schema_verified:
        return
    with _schema_verify_lock:
        if db_path in _schema_verified:      # a same-process racer may have just finished the slow path
            return                           # while we were waiting for _schema_verify_lock
        _ensure_schema_locked(db_path)
        _schema_verified.add(db_path)


def _ensure_schema_locked(db_path: str) -> None:
    """Cross-process mutual exclusion around WAL setup and schema initialization.

    The lock is a SEPARATE, tiny SQLite file (`_schema_lock_path()`) that -- deliberately -- is NEVER
    itself switched to WAL, so acquiring it never triggers the very race it exists to prevent: ordinary
    rollback-journal `BEGIN IMMEDIATE` contention IS reliably retried by `busy_timeout` (measured: 0
    failures across 500+ concurrent acquisitions of exactly this primitive, vs. the WAL-conversion race
    above). Holding it, only ONE thread/process at a time performs `_connect()` (the WAL switch) and
    `schema.bootstrap()`; every other caller blocks -- bounded by `_SCHEMA_LOCK_TIMEOUT_S` -- and then
    finds the work already done, at which point its OWN `PRAGMA journal_mode=WAL` reissue is a safe no-op
    (measured: 0 failures reissuing the pragma against an already-WAL file under concurrent write load --
    the exclusive-access requirement only applies to an actual mode CHANGE). A crashed lock holder needs
    no separate staleness heuristic: SQLite's own hot-journal recovery rolls back an abandoned transaction
    the moment the next connection touches the file.

    The bootstrap is idempotent and transactional; this outer lock prevents the WAL-conversion race from
    landing two callers inside initialization at the same moment.
    """
    lock_conn = sqlite3.connect(_schema_lock_path(), timeout=_SCHEMA_LOCK_TIMEOUT_S)
    try:
        lock_conn.execute(f"PRAGMA busy_timeout={int(_SCHEMA_LOCK_TIMEOUT_S * 1000)}")
        lock_conn.isolation_level = None      # manual transaction control: DDL/PRAGMA needs an explicit BEGIN
                                               # to be governed by it at all under Python's sqlite3 module.
        try:
            lock_conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise SchemaLockTimeout(
                f"could not acquire the schema-init lock for {db_path!r} within "
                f"{_SCHEMA_LOCK_TIMEOUT_S:.0f}s -- another process or thread appears to be "
                f"initializing this store"
            ) from exc
        try:
            with closing(_connect()) as db:
                schema.bootstrap(db)
        finally:
            lock_conn.execute("COMMIT")       # release the lock even if bootstrap raised -- a real
                                               # schema failure must not also wedge every OTHER
                                               # caller behind a lock nobody will ever release.
    finally:
        lock_conn.close()


def _connect() -> sqlite3.Connection:
    os.makedirs(RUNS_DIR, exist_ok=True)
    db = sqlite3.connect(_db_path(), timeout=30.0)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=30000")
    return db


def association_secret() -> bytes:
    """Stable install-local HMAC key in SQLite metadata; never exported with a run payload."""
    _ensure()
    with closing(_connect()) as db, db:
        row = db.execute("SELECT value FROM schema_meta WHERE key = 'association_hmac_key'").fetchone()
        if row is None:
            candidate = secrets.token_hex(32)
            db.execute("INSERT OR IGNORE INTO schema_meta(key, value) VALUES('association_hmac_key', ?)",
                       (candidate,))
            row = db.execute("SELECT value FROM schema_meta WHERE key = 'association_hmac_key'").fetchone()
    return bytes.fromhex(str(row["value"]))


def _valid_rid(rid) -> bool:
    return bool(isinstance(rid, str) and rid and _SAFE_RID.fullmatch(rid))


def _blob_path(digest: str) -> str:
    return os.path.join(_blob_root(), digest[:2], digest + ".json")


def _store_blob(value: dict, *, kind: str) -> dict:
    """Write a content-addressed JSON blob if it isn't already on disk, returning its ref. Shared by the
    trace blob and the persisted context<->answer influence-map artifact (Phase 3.7): both are
    potentially-large derived evidence that a run's queryable row shouldn't carry inline.
    BACKLOG §2 (evidence-write failures): the old trace-only code let an `atomic_write_json` failure
    (disk full, permission error, ...) propagate all the way up through `_pack`/`_put`/`record`'s blanket
    `except Exception: return None` -- one write hiccup silently discarded the ENTIRE run (prompt,
    response, everything), with nothing but a bare None to show for it. We now catch the write failure
    HERE, log it, and hand it back via the ref itself (`write_failed`) so `_pack` can mark the run row
    evidence-missing and still persist everything else -- degrading honestly instead of vanishing."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    path = _blob_path(digest)
    ref = {"sha256": digest, "media_type": "application/json", "bytes": len(encoded)}
    if not os.path.isfile(path):
        try:
            atomic_write_json(path, value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        except Exception as exc:
            _log.warning("%s blob write failed (digest=%s, bytes=%d): %s: %s",
                         kind, digest, len(encoded), type(exc).__name__, exc)
            ref["write_failed"] = f"{type(exc).__name__}: {exc}"
    return ref


def _load_blob(ref, *, kind: str) -> dict:
    """Load a content-addressed JSON blob, VERIFYING its SHA-256 against the recorded digest. A missing,
    unreadable, or corrupt (digest-mismatch) blob is SURFACED as {"unavailable": <reason>, "sha256": ...}
    rather than silently returned as an empty {} -- a run must never present "no <kind>" when the truth is
    "the evidence was lost or tampered with." An absent/invalid ref (the run genuinely carried none) still
    returns {} (honestly none, not corruption)."""
    ref = ref or {}
    digest = str(ref.get("sha256") or "")
    write_failed = ref.get("write_failed")
    if write_failed:
        # The write itself failed back in _store_blob (see that docstring) -- there is nothing useful to
        # try opening; we already know the verdict and must report exactly that, not a possibly-different
        # one derived from whatever partial/absent file happens to sit at this digest's path right now.
        return {"unavailable": f"{kind} evidence write failed: {write_failed}", "sha256": digest or None}
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        return {}
    try:
        with open(_blob_path(digest), "rb") as handle:
            raw = handle.read()
    except FileNotFoundError:
        return {"unavailable": f"{kind} blob missing", "sha256": digest}
    except Exception as exc:
        return {"unavailable": f"{kind} blob unreadable: {type(exc).__name__}", "sha256": digest}
    actual = hashlib.sha256(raw).hexdigest()
    if actual != digest:
        return {"unavailable": f"{kind} blob corrupt (digest mismatch)", "sha256": digest, "actual_sha256": actual}
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        return {"unavailable": f"{kind} blob not valid JSON: {type(exc).__name__}", "sha256": digest}
    return value if isinstance(value, dict) else {}


def _store_trace(trace: dict) -> dict:
    return _store_blob(trace, kind="trace")


def _load_trace(ref) -> dict:
    return _load_blob(ref, kind="trace")


def _store_influence_map(value: dict) -> dict:
    return _store_blob(value, kind="influence map")


def _load_influence_map(ref) -> dict:
    return _load_blob(ref, kind="influence map")


def _mark_blob_write_failure(payload: dict, ref: dict, *, flag: str, meta_key: str) -> None:
    """Mirror a blob write failure onto the run's own flags/meta so it survives even though the
    underlying evidence file didn't. Used for both the trace blob and the influence-map blob."""
    if not ref.get("write_failed"):
        return
    payload["flags"] = sorted(set(payload.get("flags") or []) | {flag})
    meta = dict(payload.get("meta") or {})
    meta[meta_key] = ref["write_failed"]
    payload["meta"] = meta


def _pack(rec: dict) -> tuple[str, dict]:
    # Validate the complete document before writing a blob. A bad metadata value must not leave an
    # orphaned trace artifact merely because trace serialization happened first.
    json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
    payload = dict(rec)
    trace_ref = _store_trace(payload.pop("trace", {}) or {})
    # BACKLOG §2 honesty invariant: the run row still lands (losing the WHOLE run over a trace-write
    # hiccup would be strictly worse), but it must never read back as "no trace" -- a plain {} would
    # look identical to a run that genuinely carried none. Every surface that reads a run (Runs list
    # via `flags`, run detail via `meta`, a future `clozn doctor`) gets the same honest marker, applied
    # here once rather than re-derived per call site.
    _mark_blob_write_failure(payload, trace_ref, flag="evidence-missing", meta_key="evidence_write_failed")
    payload["trace_ref"] = trace_ref

    # Phase 3.7: the persisted context<->answer influence-map artifact is potentially-large derived
    # evidence (a complete answer-token x context-span matrix) -- it goes through the same
    # content-addressed blob machinery as trace, keyed by influence_map_ref, so the run row itself stays
    # small. A run that never had a map computed carries no influence_map key at all (not an empty one).
    influence_map_value = payload.pop("influence_map", None)
    if isinstance(influence_map_value, dict) and influence_map_value:
        influence_map_ref = _store_influence_map(influence_map_value)
        _mark_blob_write_failure(payload, influence_map_ref, flag="influence-evidence-missing",
                                 meta_key="influence_evidence_write_failed")
        payload["influence_map_ref"] = influence_map_ref

    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return serialized, payload


def _unpack(payload_json: str) -> dict:
    rec = json.loads(payload_json)
    rec["trace"] = _load_trace(rec.pop("trace_ref", None))
    influence_map_ref = rec.pop("influence_map_ref", None)
    if influence_map_ref is not None:
        rec["influence_map"] = _load_influence_map(influence_map_ref)
    return rec


def _row_values(rec: dict, payload_json: str) -> tuple:
    timing = rec.get("timing") or {}
    return (
        rec["id"],
        float(rec.get("created_ts") or 0.0),
        float(rec.get("recorded_ts") or rec.get("created_ts") or 0.0),
        str(rec.get("created_at") or ""),
        str(rec.get("source") or ""),
        str(rec.get("client") or "unknown"),
        rec.get("client_key"),
        rec.get("session_key"),
        str(rec.get("model") or ""),
        str(rec.get("substrate") or ""),
        rec.get("parent_run_id"),
        rec.get("finish_reason"),
        rec.get("error"),
        str(rec.get("prompt_summary") or ""),
        str(rec.get("response_summary") or ""),
        int(timing.get("duration_ms") or 0),
        payload_json,
    )


_INSERT = """
    INSERT INTO runs(
        id, created_ts, recorded_ts, created_at, source, client, client_key, session_key,
        model, substrate, parent_run_id, finish_reason, error, prompt_summary, response_summary,
        duration_ms, payload_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def _put(rec: dict, *, replace: bool = False, ignore: bool = False) -> bool:
    if not isinstance(rec, dict) or not _valid_rid(rec.get("id")):
        return False
    payload_json, _ = _pack(rec)
    statement = _INSERT
    if replace:
        statement = _INSERT.replace("INSERT INTO", "INSERT OR REPLACE INTO", 1)
    elif ignore:
        statement = _INSERT.replace("INSERT INTO", "INSERT OR IGNORE INTO", 1)
    with closing(_connect()) as db, db:
        session_id = rec.get("session_key")
        if session_id:
            created_ts = float(rec.get("recorded_ts") or rec.get("created_ts") or time.time())
            created_at = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(created_ts))
            session_client = rec.get("client_key")
            if not isinstance(session_client, str) or not _OPAQUE_CLIENT_KEY.fullmatch(session_client):
                session_client = None
            db.execute(
                "INSERT OR IGNORE INTO sessions"
                "(id, created_ts, created_at, client_key, title, privacy_json) "
                "VALUES (?, ?, ?, ?, NULL, ?)",
                (session_id, created_ts, created_at, session_client, '{"visibility":"visible"}'),
            )
        cursor = db.execute(statement, _row_values(rec, payload_json))
        return cursor.rowcount > 0


def record(*, source: str, client: str = "unknown", model: str = "", substrate: str = "",
           messages=None, response: str = "",
           trace: dict | None = None, started: float | None = None, ended: float | None = None,
           parent_run_id: str | None = None, changes_applied: dict | None = None,
           error: str | None = None, finish_reason: str | None = None,
           meta: dict | None = None, assembled_messages=None, final_prompt: str | None = None,
           identity: dict | None = None,
           reasoning: dict | None = None, session_key: str | None = None,
           client_key: str | None = None, client_key_source: str | None = None,
           project_key: str | None = None,
           output_contract: dict | None = None,
           _reserved_run_id: str | None = None) -> str | None:
    """Persist a completed run and return its id. Logging failures remain non-fatal.

    `identity` (roadmap S4.3): the immutable reproduction-identity block from
    clozn.runs.identity.runtime_identity -- model_sha256, template_fingerprint, engine_build,
    clozn_version, captured_at. A top-level field (like trace), not folded into
    `meta`, so analysis consumers can read it without picking through generation metadata.
    Callers that don't pass one (older call sites, replay/fork, the CLI run path) simply get {} --
    honestly "no identity captured for this run," not a fabricated one.

    """
    try:
        _ensure()
        from .association import session_key as normalize_session_key
        session_key = normalize_session_key(session_key)
        started = started if started is not None else time.time()
        ended = ended if ended is not None else time.time()
        rid = _reserved_run_id or f"run_{int(started * 1000):013x}_{uuid.uuid4().hex[:6]}"
        if not _valid_rid(rid):
            raise ValueError("_reserved_run_id must be a valid run ID")
        from .think_tags import prompt_opens_think, sanitize_messages, sanitize_reply, sanitize_steps
        implicit_think = prompt_opens_think(final_prompt)
        msgs = sanitize_messages(messages or [])
        # A gateway caller that supplies `reasoning` has already separated the public response.  Do not
        # apply the prompt-prefilled state to that clean answer a second time (it would hide the answer as
        # an unclosed block).  Direct store callers still get the full safety-net sanitation here.
        think = sanitize_reply(response, implicit_open=implicit_think if not reasoning else False)
        response = think.public_text
        prompt = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        norm_trace = _norm_trace(trace)
        reasoning_doc = dict(reasoning or {})
        if (think.stripped or reasoning_doc) and isinstance(norm_trace.get("steps"), list):
            public_steps, reasoning_steps, trace_think = sanitize_steps(
                norm_trace["steps"], implicit_open=implicit_think
            )
            public_trace = steps_to_trace(public_steps)
            if reasoning_steps:
                public_trace["reasoning_steps"] = reasoning_steps
            norm_trace = public_trace
            trace_public = trace_think.public_text
            alignment = "matched" if trace_public == response else (
                "matched_ignoring_outer_whitespace"
                if trace_public.strip() == response.strip() else "mismatch"
            )
            if not reasoning_doc:
                reasoning_doc = think.journal(reasoning_steps=reasoning_steps,
                                               trace_alignment=alignment)
            else:
                reasoning_doc.setdefault("trace_step_count", len(reasoning_steps))
                reasoning_doc.setdefault("trace_alignment", alignment)
        elif think.stripped and not reasoning_doc:
            reasoning_doc = think.journal()
        from .context_receipt import build_context_receipt, warnings_for
        context_receipt = build_context_receipt(
            messages=msgs,
            assembled_messages=assembled_messages,
            final_prompt=final_prompt,
            finish_reason=finish_reason,
            meta=meta,
            trace=norm_trace,
            run_id=rid,
            identity=identity,
            error=error,
        )
        rec = {
            "id": rid,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
            "created_ts": started,
            "recorded_ts": time.time(),
            "source": source,
            "client": client or "unknown",
            "client_key": client_key,
            "client_key_source": client_key_source,
            "session_key": session_key,
            "project_key": project_key,
            "model": model,
            "substrate": substrate,
            "prompt_summary": _summ(prompt),
            "response_summary": _summ(response),
            # Confidence shape for the run index, derived here because `norm_trace` is in hand and the
            # trace itself is about to become a content-addressed blob (`trace_ref`) that list_runs
            # deliberately does not open. Spreads to nothing when the run has no confidence, so a
            # traceless run carries no confidence keys at all rather than zeros.
            **confidence_facts(norm_trace),
            "messages": msgs,
            "response": response,
            "reasoning": reasoning_doc,
            "assembled_messages": assembled_messages if assembled_messages is not None else None,
            "final_prompt": final_prompt,
            "trace": norm_trace,
            "timing": {"started_at": started, "ended_at": ended,
                       "duration_ms": int((ended - started) * 1000)},
            "parent_run_id": parent_run_id,
            "changes_applied": changes_applied,
            "error": error,
            "finish_reason": finish_reason,
            "meta": meta or {},
            "identity": identity or {},
            # Structured-I/O evidence lives in the JSON payload because none of its members are indexed
            # columns. Copy only a real object so a malformed direct caller cannot make _flags() raise and
            # silently lose the whole run.
            "output_contract": dict(output_contract) if isinstance(output_contract, dict) else {},
            "context_receipt": context_receipt,
            "warnings": warnings_for(finish_reason, meta),
        }
        # Construct the default source universe only after receipt capture has
        # minted and persisted canonical seg_/src_ identities.  The manifest
        # therefore contains IDs the strict source-removal path can consume
        # directly; derivation failures produce an empty universe rather than a
        # speculative coordinate mapping.
        from . import context_units
        context_unit_manifest = context_units.build_context_unit_manifest(rec)
        rec["context_units"] = context_unit_manifest
        rec["flags"] = _flags(rec)
        if not _put(rec):
            return None
        _prune()
        return rid
    except Exception:
        return None


def _prune() -> None:
    _ensure()
    with closing(_connect()) as db, db:
        db.execute(
            "DELETE FROM runs WHERE id IN ("
            "SELECT id FROM runs ORDER BY recorded_ts DESC, id DESC LIMIT -1 OFFSET ?)",
            (int(KEEP),),
        )


def list_runs(limit: int = 50, *, include_replays: bool = True) -> list[dict]:
    """Newest-first summaries, filtered and limited in SQLite rather than in Python."""
    _ensure()
    where = "" if include_replays else "WHERE source <> 'replay'"
    with closing(_connect()) as db, db:
        rows = db.execute(
            f"SELECT payload_json FROM runs {where} ORDER BY created_ts DESC, id DESC LIMIT ?",
            (max(0, int(limit)),),
        ).fetchall()
    out = []
    for row in rows:
        try:
            out.append(_summary(json.loads(row["payload_json"])))
        except Exception:
            continue
    return out


_DERIVED_SOURCES = frozenset({"replay", "branch", "fork"})


def find_runs(limit: int = 50, *, client: str | None = None, session_id: str | None = None,
              client_id: str | None = None, model: str | None = None,
              include_derived: bool = False) -> list[dict]:
    """Newest matching summaries for sidecars/watchers, over the bounded local journal.

    Session input may be the raw caller-known token or an already-opaque ``session_...`` key.  Client
    matching is case-insensitive because User-Agent normalization produces human-facing labels.
    """
    from .association import client_key, session_key
    wanted_limit = max(0, int(limit))
    if wanted_limit == 0:
        return []
    wanted_session = session_key(session_id)
    wanted_client_id = client_key(client_id)
    wanted_client = str(client).strip().casefold() if client is not None and str(client).strip() else None
    wanted_model = str(model).strip().casefold() if model is not None and str(model).strip() else None
    clauses = []
    params: list = []
    if not include_derived:
        clauses.append("source NOT IN ('replay', 'branch', 'fork')")
    if wanted_client is not None:
        clauses.append("lower(client) = ?")
        params.append(wanted_client)
    if wanted_client_id is not None:
        clauses.append("client_key = ?")
        params.append(wanted_client_id)
    if wanted_session is not None:
        clauses.append("session_key = ?")
        params.append(wanted_session)
    if wanted_model is not None:
        clauses.append("lower(model) = ?")
        params.append(wanted_model)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    _ensure()
    with closing(_connect()) as db:
        rows = db.execute(
            f"SELECT recorded_ts, payload_json FROM runs {where} "
            "ORDER BY recorded_ts DESC, id DESC LIMIT ?",
            (*params, wanted_limit),
        ).fetchall()
    out = []
    for row in rows:
        try:
            run = json.loads(row["payload_json"])
            run.setdefault("recorded_ts", float(row["recorded_ts"]))
            out.append(_summary(run))
        except Exception:
            continue
    return out


def latest_run(*, client: str | None = None, session_id: str | None = None,
               client_id: str | None = None, model: str | None = None,
               include_derived: bool = False) -> dict | None:
    rows = find_runs(1, client=client, session_id=session_id, client_id=client_id,
                     model=model, include_derived=include_derived)
    return rows[0] if rows else None


def get_run(rid: str) -> dict | None:
    if not _valid_rid(rid):
        return None
    _ensure()
    with closing(_connect()) as db, db:
        row = db.execute("SELECT payload_json FROM runs WHERE id = ?", (rid,)).fetchone()
    if row is None:
        return None
    try:
        return _unpack(row["payload_json"])
    except Exception:
        return None


def replace_run(rec: dict) -> bool:
    """Atomically replace one existing run document (used for run attachments)."""
    rid = rec.get("id") if isinstance(rec, dict) else None
    if not _valid_rid(rid) or get_run(rid) is None:
        return False
    try:
        return _put(rec, replace=True)
    except Exception:
        return False


def iter_runs(*, limit: int | None = None) -> list[dict]:
    """Return full records newest-first for bounded local analysis."""
    _ensure()
    sql = "SELECT payload_json FROM runs ORDER BY created_ts DESC, id DESC"
    params: tuple = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (max(0, int(limit)),)
    with closing(_connect()) as db, db:
        rows = db.execute(sql, params).fetchall()
    out = []
    for row in rows:
        try:
            out.append(_unpack(row["payload_json"]))
        except Exception:
            continue
    return out
