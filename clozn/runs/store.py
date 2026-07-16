"""SQLite run metadata with content-addressed trace artifacts.

SQLite is authoritative.  A run's queryable metadata and JSON document live in the
``runs`` table; the potentially large normalized trace lives in an immutable SHA-256
blob.  Old ``run_*.json`` journals are accepted only through :func:`import_json_dir`.
"""
from __future__ import annotations

from contextlib import closing
import glob
import hashlib
import json
import os
import re
import sqlite3
import time
import uuid

from clozn._io import atomic_write_json

from .summaries import SUMMARY_FIELDS, _flags, _summ, _summary
from .trace import (
    TRACE_KEYS,
    _clean_alt,
    _clean_alts,
    _clean_step,
    _entropy_from_probs,
    _float_or_none,
    _int_or_none,
    _logprob,
    _normalize_workspace_readouts,
    _norm_trace,
    _rounded_prob,
    _steps_from_parallel,
    _with_workspace_readouts,
    accumulate_ar_events,
    finish_reason_from_frames,
    steps_to_trace,
)


RUNS_DIR = os.path.join(os.path.expanduser("~/.clozn"), "runs")
KEEP = 1000
SCHEMA_VERSION = 1
_SAFE_RID = re.compile(r"^[A-Za-z0-9_-]+$")


def _db_path() -> str:
    return os.path.join(RUNS_DIR, "runs.sqlite3")


def _blob_root() -> str:
    return os.path.join(RUNS_DIR, "blobs", "sha256")


def _ensure() -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)
    os.makedirs(_blob_root(), exist_ok=True)
    with closing(_connect()) as db, db:
        db.executescript(
            """
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                created_ts REAL NOT NULL,
                created_at TEXT NOT NULL,
                source TEXT NOT NULL,
                client TEXT NOT NULL,
                model TEXT NOT NULL,
                substrate TEXT NOT NULL,
                parent_run_id TEXT,
                finish_reason TEXT,
                error TEXT,
                prompt_summary TEXT NOT NULL,
                response_summary TEXT NOT NULL,
                duration_ms INTEGER NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS runs_created_idx ON runs(created_ts DESC, id DESC);
            CREATE INDEX IF NOT EXISTS runs_source_idx ON runs(source, created_ts DESC);
            CREATE INDEX IF NOT EXISTS runs_parent_idx ON runs(parent_run_id, created_ts ASC);
            CREATE INDEX IF NOT EXISTS runs_model_idx ON runs(model, created_ts DESC);
            """
        )
        db.execute(
            "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )


def _connect() -> sqlite3.Connection:
    os.makedirs(RUNS_DIR, exist_ok=True)
    db = sqlite3.connect(_db_path(), timeout=30.0)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=NORMAL")
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=30000")
    return db


def _valid_rid(rid) -> bool:
    return bool(isinstance(rid, str) and rid and _SAFE_RID.fullmatch(rid))


def _blob_path(digest: str) -> str:
    return os.path.join(_blob_root(), digest[:2], digest + ".json")


def _store_trace(trace: dict) -> dict:
    encoded = json.dumps(trace, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    path = _blob_path(digest)
    if not os.path.isfile(path):
        atomic_write_json(path, trace, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {"sha256": digest, "media_type": "application/json", "bytes": len(encoded)}


def _load_trace(ref) -> dict:
    digest = str((ref or {}).get("sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        return {}
    try:
        with open(_blob_path(digest), encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _pack(rec: dict) -> tuple[str, dict]:
    # Validate the complete document before writing a blob. A bad metadata value must not leave an
    # orphaned trace artifact merely because trace serialization happened first.
    json.dumps(rec, ensure_ascii=False, separators=(",", ":"))
    payload = dict(rec)
    payload["trace_ref"] = _store_trace(payload.pop("trace", {}) or {})
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return serialized, payload


def _unpack(payload_json: str) -> dict:
    rec = json.loads(payload_json)
    rec["trace"] = _load_trace(rec.pop("trace_ref", None))
    return rec


def _row_values(rec: dict, payload_json: str) -> tuple:
    timing = rec.get("timing") or {}
    return (
        rec["id"],
        float(rec.get("created_ts") or 0.0),
        str(rec.get("created_at") or ""),
        str(rec.get("source") or ""),
        str(rec.get("client") or "unknown"),
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
        id, created_ts, created_at, source, client, model, substrate, parent_run_id,
        finish_reason, error, prompt_summary, response_summary, duration_ms, payload_json
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        before = db.total_changes
        db.execute(statement, _row_values(rec, payload_json))
        return db.total_changes > before


def record(*, source: str, client: str = "unknown", model: str = "", substrate: str = "",
           messages=None, response: str = "", memory: dict | None = None, behavior: dict | None = None,
           trace: dict | None = None, started: float | None = None, ended: float | None = None,
           parent_run_id: str | None = None, changes_applied: dict | None = None,
           error: str | None = None, finish_reason: str | None = None,
           meta: dict | None = None, assembled_messages=None, final_prompt: str | None = None,
           workspace_provider=None, sections: list | None = None) -> str | None:
    """Persist a completed run and return its id. Logging failures remain non-fatal.

    `sections` (prompt-section influence foundation, clozn.runs.sections): the run's ablatable-section
    manifest, if the caller built one -- explicit `clozn_section` tags, the deterministic auto-chunker, or
    memory-card locations, per clozn.runs.sections's own docstring. Stored as the top-level `sections`
    field ONLY when non-empty; an old caller that never passes it (every replay/fork/timetravel call site
    today) gets exactly the schema it always got -- no `sections` key at all, not a null or empty list --
    so this is purely additive and nothing downstream needs to special-case its absence."""
    try:
        _ensure()
        started = started if started is not None else time.time()
        ended = ended if ended is not None else time.time()
        rid = f"run_{int(started * 1000):013x}_{uuid.uuid4().hex[:6]}"
        msgs = messages or []
        prompt = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        norm_trace = _with_workspace_readouts(rid, _norm_trace(trace), workspace_provider)
        rec = {
            "id": rid,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
            "created_ts": started,
            "source": source,
            "client": client or "unknown",
            "model": model,
            "substrate": substrate,
            "prompt_summary": _summ(prompt),
            "response_summary": _summ(response),
            "messages": msgs,
            "response": response,
            "assembled_messages": assembled_messages if assembled_messages is not None else None,
            "final_prompt": final_prompt,
            "memory": memory or {},
            "behavior": behavior or {},
            "trace": norm_trace,
            "timing": {"started_at": started, "ended_at": ended,
                       "duration_ms": int((ended - started) * 1000)},
            "parent_run_id": parent_run_id,
            "changes_applied": changes_applied,
            "error": error,
            "finish_reason": finish_reason,
            "meta": meta or {},
        }
        if sections:
            rec["sections"] = list(sections)
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
            "SELECT id FROM runs ORDER BY created_ts DESC, id DESC LIMIT -1 OFFSET ?)",
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
    """Return full records newest-first for lineage and offline analysis."""
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


def import_json_dir(path: str) -> dict:
    """Explicit one-shot import of a legacy ``run_*.json`` directory."""
    _ensure()
    result = {"found": 0, "imported": 0, "skipped": 0, "invalid": 0}
    for filename in sorted(glob.glob(os.path.join(os.path.abspath(path), "run_*.json"))):
        result["found"] += 1
        try:
            with open(filename, encoding="utf-8") as handle:
                rec = json.load(handle)
            if not isinstance(rec, dict) or not _valid_rid(rec.get("id")):
                result["invalid"] += 1
                continue
            rec.setdefault("trace", {})
            rec.setdefault("created_ts", os.path.getmtime(filename))
            rec.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%S",
                                                        time.localtime(rec["created_ts"])))
            rec.setdefault("source", "legacy")
            rec.setdefault("client", "unknown")
            rec.setdefault("model", "")
            rec.setdefault("substrate", "")
            rec.setdefault("prompt_summary", "")
            rec.setdefault("response_summary", "")
            rec.setdefault("timing", {"started_at": rec["created_ts"], "ended_at": rec["created_ts"],
                                      "duration_ms": 0})
            rec.setdefault("flags", _flags(rec))
            if _put(rec, ignore=True):
                result["imported"] += 1
            else:
                result["skipped"] += 1
        except Exception:
            result["invalid"] += 1
    _prune()
    return result


def update_tiny_tests(rid: str, tiny_tests: list) -> bool:
    from .attachments import update_tiny_tests as _update_tiny_tests
    return _update_tiny_tests(rid, tiny_tests)


def _load_runs() -> list[dict]:
    from .lineage import _load_runs as _load
    return _load()


def _change_label(run: dict) -> str | None:
    from .lineage import _change_label as _label
    return _label(run)


def _lineage_summary(run: dict, current_id: str | None = None) -> dict:
    from .lineage import _lineage_summary as _summary_for_lineage
    return _summary_for_lineage(run, current_id)


def lineage(rid: str, limit: int = 500) -> dict | None:
    from .lineage import lineage as _lineage
    return _lineage(rid, limit)


def lineage_family(rid: str, limit: int = 2000) -> list[dict] | None:
    from .lineage import lineage_family as _lineage_family
    return _lineage_family(rid, limit)
