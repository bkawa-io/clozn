"""Transactional privacy and retention mutations for the local run journal.

SQLite is authoritative. A database mutation commits before trace cleanup; cleanup
then rechecks all references under a SQLite writer lock. The safe failure mode is an
orphan for existing GC, never a surviving row whose evidence was deleted.
"""
from __future__ import annotations

from contextlib import closing
import json
import os
import re
import sqlite3
import time
from typing import Any

from . import store

REDACTION_SCHEMA = "clozn.run_redaction.v1"
LITERAL_REDACTION_SCHEMA = "clozn.run_literal_redaction.v1"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_REMOVED_FIELDS = (
    "applied_corrections", "assembled_messages", "behavior", "changes_applied", "client_key",
    "client_key_source", "context_receipt", "correction_conflicts", "error", "final_prompt", "memory",
    "influence_map", "messages", "meta", "output_contract", "project_key", "prompt_summary",
    "reasoning", "response", "response_summary", "session_key", "tiny_tests",
    "trace", "warnings",
)
_LITERAL_PLACEHOLDER = "[REDACTED]"
# Structural/identity fields a literal scrub must never rewrite: the row id, its timestamps, the
# content-addressed trace pointer (a hash, not text -- see store._store_trace), and the redaction receipt
# itself (about to be overwritten with the fresh one below anyway).
_NO_SCRUB_KEYS = frozenset({"id", "created_at", "created_ts", "recorded_ts", "trace_ref", "redaction"})


class MutationError(RuntimeError):
    """A journal mutation could not be completed atomically."""


class RunHasChildrenError(MutationError):
    """A run has replay/branch children; deletion is refused unless cascade is requested explicitly."""

    def __init__(self, run_id: str, children: "list[str]"):
        self.run_id = run_id
        self.children = list(children)
        preview = ", ".join(self.children[:5]) + (", ..." if len(self.children) > 5 else "")
        super().__init__(
            f"run {run_id!r} has {len(self.children)} child run(s) ({preview}) and cannot be deleted "
            f"without cascade=True"
        )


def _require_run_id(run_id: Any) -> str:
    if not store._valid_rid(run_id):
        raise MutationError("run_id must be an exact valid run ID")
    return run_id


def _blob_digest(payload: Any, key: str) -> str | None:
    ref = payload.get(key) if isinstance(payload, dict) else None
    digest = ref.get("sha256") if isinstance(ref, dict) else None
    return digest if isinstance(digest, str) and _DIGEST_RE.fullmatch(digest) else None


def _trace_digest(payload: Any) -> str | None:
    return _blob_digest(payload, "trace_ref")


def _influence_map_digest(payload: Any) -> str | None:
    return _blob_digest(payload, "influence_map_ref")


def _begin(db: sqlite3.Connection) -> None:
    db.isolation_level = None
    db.execute("BEGIN IMMEDIATE")


def _end(db: sqlite3.Connection, error: BaseException | None) -> None:
    try:
        db.execute("ROLLBACK" if error is not None else "COMMIT")
    except sqlite3.Error:
        if error is None:
            raise


def _load_payload(raw: str) -> dict | None:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else None
    except Exception:
        return None


def _scrub(value: Any, literal: str, placeholder: str, *, key: str | None = None) -> "tuple[Any, int]":
    """Deep-replace one exact literal in every string reachable from ``value``, returning the scrubbed
    value and how many occurrences were replaced. Recurses through dicts/lists; a dict key listed in
    ``_NO_SCRUB_KEYS`` is passed through untouched (and not recursed into) so structural/identity fields
    can never be corrupted by a literal that happens to overlap a hash or timestamp string."""
    if key is not None and key in _NO_SCRUB_KEYS:
        return value, 0
    if isinstance(value, str):
        count = value.count(literal)
        return (value.replace(literal, placeholder) if count else value), count
    if isinstance(value, list):
        total = 0
        out = []
        for item in value:
            scrubbed, n = _scrub(item, literal, placeholder)
            out.append(scrubbed)
            total += n
        return out, total
    if isinstance(value, dict):
        total = 0
        out = {}
        for k, v in value.items():
            scrubbed, n = _scrub(v, literal, placeholder, key=k)
            out[k] = scrubbed
            total += n
        return out, total
    return value, 0


def _apply_literal_redaction(db: sqlite3.Connection, row: sqlite3.Row, payload: dict, run_id: str,
                             literals: "list[str]") -> dict[str, Any]:
    """Scrub only the given literals from a run's content, leaving its shape and every other field
    intact. Unlike the full tombstone, this never touches the content-addressed trace blob: that blob may
    be shared (deduplicated) across other runs, and mutating it would silently change what THEY show too.
    Callers who need a guaranteed trace wipe should use the default (no ``literals``) full-redaction path.
    """
    ordered = sorted(set(literals), key=lambda text: (-len(text), text))  # longest-first: a short literal
    scrubbed = payload                                                     # must never eat part of a longer one
    total_replacements = 0
    for literal in ordered:
        scrubbed, count = _scrub(scrubbed, literal, _LITERAL_PLACEHOLDER)
        total_replacements += count
    receipt = {
        "schema": LITERAL_REDACTION_SCHEMA,
        "status": "literal_redacted",
        "redacted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "literal_count": len(ordered),
        "replacement_count": total_replacements,
        "trace_untouched": True,
    }
    scrubbed["redaction"] = receipt
    encoded = json.dumps(scrubbed, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)

    def _scrub_column(text) -> "str | None":
        if not isinstance(text, str) or not text:
            return text
        for literal in ordered:
            text = text.replace(literal, _LITERAL_PLACEHOLDER)
        return text

    db.execute(
        "UPDATE runs SET prompt_summary=?, response_summary=?, error=?, payload_json=? WHERE id=?",
        (_scrub_column(row["prompt_summary"]) or "", _scrub_column(row["response_summary"]) or "",
         _scrub_column(row["error"]), encoded, run_id),
    )
    return {"ok": True, "action": "redact", "run_id": run_id, "already_redacted": False,
            "redaction": receipt}


def _tombstone(row: sqlite3.Row, payload: dict | None) -> dict:
    payload = payload or {}
    receipt = {
        "schema": REDACTION_SCHEMA,
        "status": "redacted",
        "redacted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "removed_fields": list(_REMOVED_FIELDS),
        "trace_evidence_removed": _trace_digest(payload) is not None,
        "influence_evidence_removed": _influence_map_digest(payload) is not None,
        "source_payload_readable": bool(payload),
    }
    identity = payload.get("identity")
    identity = identity if isinstance(identity, dict) else {}
    safe_identity = {
        key: identity[key]
        for key in ("model_sha256", "model_size_bytes", "template_fingerprint",
                    "engine_build", "clozn_version", "captured_at")
        if key in identity
    }
    return {
        "id": row["id"], "created_at": row["created_at"],
        "created_ts": float(row["created_ts"]), "recorded_ts": float(row["recorded_ts"]),
        "source": row["source"], "client": row["client"],
        "client_key": None, "client_key_source": None, "session_key": None,
        "project_key": None, "model": row["model"], "substrate": row["substrate"],
        "prompt_summary": "", "response_summary": "", "messages": [], "response": "",
        "reasoning": {}, "assembled_messages": None, "final_prompt": None,
        "memory": {}, "behavior": {}, "trace_ref": {},
        "timing": {"duration_ms": int(row["duration_ms"] or 0)},
        "parent_run_id": row["parent_run_id"], "changes_applied": None, "error": None,
        "finish_reason": row["finish_reason"], "meta": {},
        "identity": safe_identity,
        "output_contract": {}, "context_receipt": {}, "warnings": [],
        "flags": ["redacted"], "redaction": receipt,
    }


def _cleanup_blob(digest: str | None) -> dict[str, Any]:
    """Synchronously remove one content-addressed blob (trace or influence-map) IF nothing else still
    references it -- the immediate counterpart to the deferred, explicitly-requested `clozn migrate --gc`
    sweep. Generic over which ref key produced the digest; `gc._referenced_digests` already scans both."""
    if digest is None:
        return {"status": "not_applicable", "sha256": None}
    path, root = store._blob_path(digest), store._blob_root()
    try:
        from . import gc
        with closing(store._connect()) as db:
            _begin(db)
            error = None
            try:
                if digest in gc._referenced_digests(db):
                    result = {"status": "retained_shared", "sha256": digest}
                elif not os.path.exists(path):
                    result = {"status": "already_missing", "sha256": digest}
                elif not gc._is_contained(root, path):
                    result = {"status": "failed", "sha256": digest,
                              "error": "refused: path escapes blob root"}
                else:
                    try:
                        os.remove(path)
                    except OSError as exc:
                        result = {"status": "failed", "sha256": digest,
                                  "error": f"{type(exc).__name__}: {exc}"}
                    else:
                        result = {"status": "deleted", "sha256": digest}
            except BaseException as exc:
                error = exc
                raise
            finally:
                _end(db, error)
        return result
    except Exception as exc:
        return {"status": "failed", "sha256": digest,
                "error": f"{type(exc).__name__}: {exc}"}


def redact_run(run_id: str, *, literals: "list[str] | None" = None) -> dict[str, Any]:
    """Replace exactly one run with a content-free audit tombstone.

    With ``literals`` given, redaction is instead scoped to exactly those substrings: only text matching
    one of them is scrubbed (deep, everywhere it appears), the rest of the run's content and shape survive
    untouched, and the trace blob is left alone (see ``_apply_literal_redaction``). Omit ``literals`` (the
    default) for the original, guaranteed-total wipe used by ``clozn runs redact <id>``.
    """
    run_id = _require_run_id(run_id)
    if literals is not None:
        if not isinstance(literals, (list, tuple)):
            raise MutationError("literals must be a list of strings")
        cleaned = []
        for item in literals:
            if not isinstance(item, str) or not item:
                raise MutationError("each literal must be a non-empty string")
            cleaned.append(item)
        literals = cleaned
    store._ensure()
    digest = None
    influence_digest = None
    try:
        with closing(store._connect()) as db:
            _begin(db)
            error = None
            try:
                row = db.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
                if row is None:
                    raise MutationError(f"run {run_id!r} was not found")
                else:
                    payload = _load_payload(row["payload_json"])
                    existing = payload.get("redaction") if payload else None
                    if isinstance(existing, dict) and existing.get("status") == "redacted":
                        result = {"ok": True, "action": "redact", "run_id": run_id,
                                  "already_redacted": True, "redaction": dict(existing)}
                    elif literals:
                        result = _apply_literal_redaction(db, row, payload or {}, run_id, literals)
                    else:
                        digest = _trace_digest(payload)
                        influence_digest = _influence_map_digest(payload)
                        redacted = _tombstone(row, payload)
                        encoded = json.dumps(redacted, ensure_ascii=False, sort_keys=True,
                                             separators=(",", ":"), allow_nan=False)
                        db.execute(
                            "UPDATE runs SET client_key=NULL, session_key=NULL, error=NULL, "
                            "prompt_summary='', response_summary='', payload_json=? WHERE id=?",
                            (encoded, run_id),
                        )
                        result = {"ok": True, "action": "redact", "run_id": run_id,
                                  "already_redacted": False, "redaction": redacted["redaction"]}
            except BaseException as exc:
                error = exc
                raise
            finally:
                _end(db, error)
    except MutationError:
        raise
    except Exception as exc:
        raise MutationError(
            f"could not redact run {run_id!r}: {type(exc).__name__}: {exc}") from None
    # MERGE UNION (3.5 x 3.7): full-tombstone redaction cleans up BOTH content-addressed blobs;
    # literal-scoped redaction leaves BOTH untouched -- either blob may be shared across runs,
    # and the receipt states it (trace_untouched). Same rule, two evidence kinds.
    not_applicable = {"status": "not_applicable", "sha256": None}
    applicable = result.get("ok") and not result.get("already_redacted") and not literals
    result["trace_cleanup"] = _cleanup_blob(digest) if applicable else not_applicable
    result["influence_map_cleanup"] = _cleanup_blob(influence_digest) if applicable else not_applicable
    return result


def _direct_children(db: sqlite3.Connection, run_id: str) -> "list[str]":
    rows = db.execute(
        "SELECT id FROM runs WHERE parent_run_id = ? ORDER BY recorded_ts, id", (run_id,)
    ).fetchall()
    return [row["id"] for row in rows]


def _descendant_ids(db: sqlite3.Connection, run_id: str) -> "list[str]":
    """Every run reachable by following ``parent_run_id`` down from ``run_id`` -- children, grandchildren,
    and so on -- so a cascade delete never leaves a grandchild pointing at an already-deleted parent."""
    rows = db.execute(
        "WITH RECURSIVE descendants(id) AS ("
        "  SELECT id FROM runs WHERE parent_run_id = ?"
        "  UNION ALL"
        "  SELECT r.id FROM runs r JOIN descendants d ON r.parent_run_id = d.id"
        ") SELECT id FROM descendants",
        (run_id,),
    ).fetchall()
    return [row["id"] for row in rows]


def delete_run(run_id: str, *, cascade: bool = False) -> dict[str, Any]:
    """Delete exactly one run row; no tombstone survives.

    A run with replay/branch children is refused with a typed :class:`RunHasChildrenError` listing the
    exact child ids, unless ``cascade=True`` -- in which case every descendant (children, grandchildren,
    ...) is deleted in the same transaction, since deleting only the direct children would still leave a
    grandchild's ``parent_run_id`` pointing at nothing.
    """
    run_id = _require_run_id(run_id)
    if not isinstance(cascade, bool):
        raise MutationError("cascade must be a boolean")
    store._ensure()
    digests: dict[str, "tuple[str | None, str | None]"] = {}   # target -> (trace, influence_map)
    deleted_ids: "list[str]" = []
    try:
        with closing(store._connect()) as db:
            _begin(db)
            error = None
            try:
                row = db.execute("SELECT payload_json FROM runs WHERE id=?", (run_id,)).fetchone()
                if row is None:
                    raise MutationError(f"run {run_id!r} was not found")
                children = _direct_children(db, run_id)
                if children and not cascade:
                    raise RunHasChildrenError(run_id, children)
                targets = [run_id] + (_descendant_ids(db, run_id) if children else [])
                for target in targets:
                    target_row = db.execute(
                        "SELECT payload_json FROM runs WHERE id=?", (target,)).fetchone()
                    if target_row is not None:
                        payload = _load_payload(target_row["payload_json"])
                        digests[target] = (_trace_digest(payload), _influence_map_digest(payload))
                    else:
                        digests[target] = (None, None)
                db.executemany("DELETE FROM runs WHERE id=?", [(target,) for target in targets])
                deleted_ids = targets
                result = {"ok": True, "action": "delete", "run_id": run_id, "cascade": cascade,
                          "deleted_run_ids": deleted_ids}
            except BaseException as exc:
                error = exc
                raise
            finally:
                _end(db, error)
    except MutationError:
        raise
    except Exception as exc:
        raise MutationError(
            f"could not delete run {run_id!r}: {type(exc).__name__}: {exc}") from None
    if len(deleted_ids) <= 1:
        # The common, non-cascading case keeps its original single-object shape exactly.
        tr, inf = digests.get(run_id, (None, None))
        result["trace_cleanup"] = _cleanup_blob(tr)
        result["influence_map_cleanup"] = _cleanup_blob(inf)
    else:
        result["trace_cleanup"] = [
            dict(_cleanup_blob(digests.get(target, (None, None))[0]), run_id=target)
            for target in deleted_ids]
        result["influence_map_cleanup"] = [
            dict(_cleanup_blob(digests.get(target, (None, None))[1]), run_id=target)
            for target in deleted_ids]
        result["cascade_deleted_count"] = len(deleted_ids) - 1
    return result


def _plan_from_partition(retained: "list[sqlite3.Row]", removed: "list[sqlite3.Row]",
                         total: int) -> dict[str, Any]:
    """Shared plan-shaping for both keep-N and age-cutoff retention: which rows go, their orphaned trace
    blobs, and any RETAINED run whose ``parent_run_id`` is about to be deleted (an honest, non-blocking
    signal -- bulk retention still proceeds, but the lineage break is never silently hidden)."""

    def digest_of(row) -> "str | None":
        return _trace_digest(_load_payload(row["payload_json"]))

    removed_ids = {row["id"] for row in removed}
    retained_digests = {digest for row in retained if (digest := digest_of(row)) is not None}
    entries, removed_digests = [], set()
    for row in reversed(removed):  # report exact deletion order oldest-first
        digest = digest_of(row)
        if digest is not None:
            removed_digests.add(digest)
        entry = {"run_id": row["id"], "recorded_ts": float(row["recorded_ts"])}
        if digest is not None:
            entry["trace_sha256"] = digest
        entries.append(entry)
    orphaned_parent_refs = sorted(
        row["id"] for row in retained
        if row["parent_run_id"] is not None and row["parent_run_id"] in removed_ids
    )
    return {
        "total_count": total, "kept_count": len(retained),
        "delete_count": len(removed), "runs": entries,
        "orphaned_trace_digests": sorted(removed_digests - retained_digests),
        "orphaned_parent_refs": orphaned_parent_refs,
    }


def _retention_plan(db: sqlite3.Connection, keep: int) -> dict[str, Any]:
    rows = db.execute(
        "SELECT id, recorded_ts, parent_run_id, payload_json FROM runs "
        "ORDER BY recorded_ts DESC, id DESC"
    ).fetchall()
    plan = _plan_from_partition(rows[:keep], rows[keep:], len(rows))
    plan["keep"] = keep
    return plan


def _retention_plan_by_age(db: sqlite3.Connection, cutoff_ts: float) -> dict[str, Any]:
    rows = db.execute(
        "SELECT id, recorded_ts, parent_run_id, payload_json FROM runs "
        "ORDER BY recorded_ts DESC, id DESC"
    ).fetchall()
    retained = [row for row in rows if float(row["recorded_ts"]) >= cutoff_ts]
    removed = [row for row in rows if float(row["recorded_ts"]) < cutoff_ts]
    plan = _plan_from_partition(retained, removed, len(rows))
    plan["cutoff_ts"] = cutoff_ts
    return plan


def _apply_or_preview(plan_fn, *, dry_run: bool) -> dict[str, Any]:
    """Run ``plan_fn(db)`` read-only for a dry run, or inside one transaction that then deletes exactly
    the rows it planned -- shared by both retention flavors so their transaction handling stays identical.
    """
    with closing(store._connect()) as db:
        if dry_run:
            return plan_fn(db)
        _begin(db)
        error = None
        try:
            plan = plan_fn(db)
            db.executemany("DELETE FROM runs WHERE id=?",
                           [(entry["run_id"],) for entry in plan["runs"]])
            return plan
        except BaseException as exc:
            error = exc
            raise
        finally:
            _end(db, error)


def prune_to(keep: int, *, dry_run: bool = True) -> dict[str, Any]:
    """Plan or apply exact oldest-row retention; blob removal stays with existing GC."""
    if not isinstance(keep, int) or isinstance(keep, bool) or keep < 0:
        raise MutationError("keep must be a non-negative integer")
    if not isinstance(dry_run, bool):
        raise MutationError("dry_run must be a boolean")
    store._ensure()
    try:
        plan = _apply_or_preview(lambda db: _retention_plan(db, keep), dry_run=dry_run)
    except Exception as exc:
        raise MutationError(f"could not apply retention: {type(exc).__name__}: {exc}") from None
    return {
        "ok": True, "action": "retention", "dry_run": dry_run, **plan,
        "run_ids": [entry["run_id"] for entry in plan["runs"]],
        "deleted_count": 0 if dry_run else plan["delete_count"],
        "blob_cleanup": "deferred_to_gc",
    }


def prune_older_than(days: int, *, dry_run: bool = True) -> dict[str, Any]:
    """Plan or apply the age-based ``clozn privacy retention --days N`` policy: delete every run recorded
    more than ``days`` days ago. Backs the delete-on-GC path (`clozn migrate --gc`); blob removal stays
    with the existing blob GC, exactly like :func:`prune_to`."""
    if not isinstance(days, int) or isinstance(days, bool) or days <= 0:
        raise MutationError("days must be a positive integer")
    if not isinstance(dry_run, bool):
        raise MutationError("dry_run must be a boolean")
    store._ensure()
    cutoff_ts = time.time() - float(days) * 86400.0
    try:
        plan = _apply_or_preview(lambda db: _retention_plan_by_age(db, cutoff_ts), dry_run=dry_run)
    except Exception as exc:
        raise MutationError(f"could not apply age-based retention: {type(exc).__name__}: {exc}") from None
    return {
        "ok": True, "action": "retention_by_age", "dry_run": dry_run, "days": days, **plan,
        "run_ids": [entry["run_id"] for entry in plan["runs"]],
        "deleted_count": 0 if dry_run else plan["delete_count"],
        "blob_cleanup": "deferred_to_gc",
    }


__all__ = [
    "LITERAL_REDACTION_SCHEMA", "MutationError", "REDACTION_SCHEMA", "RunHasChildrenError",
    "delete_run", "prune_older_than", "prune_to", "redact_run",
]
