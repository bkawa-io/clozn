"""F5: the scoped correction store ("Teach Once") -- clozn.correction.v1 / clozn.correction-resolution.v1
/ clozn.correction-export.v1, migration 5's `corrections` + `correction_events` tables.

READ THIS BEFORE EXTENDING ANYTHING HERE -- THE HAZARD THIS MODULE EXISTS TO AVOID
------------------------------------------------------------------------------------
`clozn/server/message_assembly.py` (renamed from `memory_assembly.py`) documents what got cut on
2026-07-27: the prompt-mode CARD pipeline -- active cards -> TOPIC GATE -> compiled system block ->
injection, auto-applied without the user confirming anything. "Steering is the only personalization
surface" was the stated reason. This module is a DIFFERENT, narrower thing, and every function below
preserves the properties that make it different:

  * NO topic gate. NO relevance matching. NO inference about applicability, anywhere in this file.
    `resolve_corrections()` -- the ONE function that decides what applies to a run -- takes ONLY discrete
    scope identifiers (session/client/project/model keys, a global-local flag) as arguments. It has no
    parameter for message content, a prompt, or anything else a similarity/keyword/topic matcher could
    run over. Adding relevance-based selection later would require changing THIS signature to accept
    text -- a visible, deliberate diff, not a hidden internal tweak to existing logic.
  * NO auto-apply without confirmation. A correction is DRAFTED (inert; `confirmed_ts` absent) and then
    separately, explicitly CONFIRMED. `resolve_corrections()`'s own SQL predicate requires
    `confirmed_ts IS NOT NULL AND enabled = 1` -- an unconfirmed draft is not merely ignored by
    convention, it is never a row the query can return, structurally, regardless of what any caller
    forgets to check.
  * Scope is always explicit, always one of exactly five declared kinds (session / client / model /
    project / global_local), and is recorded on the correction at creation -- never guessed, never
    defaulted from "what the last few turns were about."
  * `content` (the correction text) is opaque payload to every selection path in this file. Grouping for
    conflict detection uses `type` (a closed four-value enum) and scope equality only -- never `content`.
    grep this file for every read of `row["content"]` / `doc["content"]` if you want to verify that by
    hand; there are exactly two, both purely for building/exporting the document, neither feeding a
    decision.
  * No claim about model weights anywhere in this module, its schemas, or the events it writes. A
    correction is declared context threaded into a request; nothing here trains, fine-tunes, or updates
    a parameter.

RECEIPT INTEGRATION (the acceptance criterion this module exists to make structural)
----------------------------------------------------------------------------------------
`apply_and_record()` is the reusable forcing function that both (a) passes a resolution's `applied` list
into `clozn.runs.store.record(applied_corrections=...)`, which folds it into the SAME immutable,
schema-validated `context_receipt` the run is created with, and (b) calls `record_applications()` to
write "applied"/"conflict_lost" events into this module's own ledger -- both derived from the identical
`resolution` dict, computed once. The live generation adapter follows the same two-step seam directly
because it must materialize the selected content before the worker call: it saves one resolution on the
handler, passes its receipt fields to `store.record()`, then invokes `record_applications()` only after
that record succeeds. Thus "this module's ledger says correction X applied to run R" cannot be true in
the shipped request path without "run R's receipt lists correction X" also being true -- the two writes
are structurally paired from one resolution, not merely kept in sync by convention.

Two honest limits on that guarantee, stated plainly rather than glossed over:
  1. The live request adapter is intentionally separate from this storage module (`clozn.server.routes.
     corrections`) and has to keep its handler-local resolution paired with `store.record()` and
     `record_applications()`. Other callers should use `apply_and_record()` rather than calling
     `store.record()` directly if they want the same guarantee. This is the same single-writer boundary
     as the execution-fork receipt seam.
  2. Python does not enforce that a future caller goes through either paired path -- someone COULD call
     `store.record(applied_corrections=...)` directly and skip `record_applications()`, or vice versa.
     That is a real gap this module cannot close by itself; it is convention beyond this one point, not a
     mechanical impossibility. What IS mechanical is the direction that matters most for the acceptance
     criterion ("no correction applies without appearing in the receipt"): the shipped code has no path
     that marks a correction "applied to run R" in this module's ledger without R's receipt already
     carrying it, because record_applications() takes a run_id that only exists after store.record()
     already returned successfully with that same resolution's applied list on board.

WHY ONE EVENT LEDGER, NOT A SEPARATE "HISTORY" TABLE
---------------------------------------------------------
`correction_events` carries BOTH lifecycle transitions (drafted/confirmed/disabled/enabled/deleted) AND
per-run outcomes (applied/conflict_lost) as one append-only, ordered (by `seq`) log per correction. This
is what makes "disable != delete history" a query guarantee rather than a promise: disabling a correction
writes a `disabled` event and flips `enabled`, but every prior `applied`/`conflict_lost` row for that
correction_id is untouched (nothing here ever UPDATEs or DELETEs a correction_events row), and stays
retrievable through `export_correction()` regardless of the correction's current enabled/deleted state.

DELETE IS NOT UNDOABLE; DISABLE IS -- READ THIS BEFORE "FIXING" undo_last_change
---------------------------------------------------------------------------------
`delete_correction()` scrubs `content` (never null-padded -- the key is removed) and stamps `deleted_ts`.
There is deliberately no shadow copy of the scrubbed text anywhere that `undo_last_change()` could restore
from, so undo refuses once a correction is deleted (CorrectionStateError). This is the concrete difference
between disable (fully reversible, content intact, `undo_last_change` flips it back) and delete
(permanent, content gone, only the hash and the event ledger survive) -- see clozn.correction.v1's own
`deleted_ts` description.
"""
from __future__ import annotations

import hashlib
import json
import re
import secrets
import time
from contextlib import closing

from clozn import schemas

from . import association, store

SCHEMA_NAME = "clozn.correction.v1"
RESOLUTION_SCHEMA = "clozn.correction-resolution.v1"
EXPORT_SCHEMA = "clozn.correction-export.v1"

TYPES = ("output_format", "source_requirement", "style", "forbidden_behavior")
SCOPE_KINDS = ("session", "client", "model", "project", "global_local")

# Most-specific-and-most-temporary first. A POLICY choice, not a mechanical necessity -- reorder this
# tuple (and nothing else) to change precedence. Reasoning, most to least contextual/temporary:
#   session       "just for this conversation" -- the most deliberate, shortest-lived override.
#   client        "how I use this specific app/integration" -- a stable per-tool habit.
#   project       "rules for this codebase" -- stable, but shared across whoever touches the project.
#   model         "this specific model needs different handling" -- a technical accommodation.
#   global_local  "my default everywhere on this machine" -- the broadest fallback, wins only when
#                 nothing more specific is declared.
PRECEDENCE_ORDER = ("session", "client", "project", "model", "global_local")
_RANK = {kind: index for index, kind in enumerate(PRECEDENCE_ORDER)}

# Tie-break when two candidates share the exact same precedence rank (only possible when two DIFFERENT
# correction rows carry the identical (scope_kind, scope_value, type) triple, or two global_local
# corrections of the same type both exist -- see resolve_corrections()'s own docstring).
SAME_RANK_TIEBREAK = "most_recently_confirmed"

_ID_RE = re.compile(r"^corr_[0-9a-f]{24}$")
_MODEL_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_UNDOABLE_EVENTS = frozenset({"confirmed", "disabled", "enabled"})
_LIFECYCLE_LIMIT = 500  # export/list bound; mirrors clozn.behavior.corrective_flow's bounded-receipt convention


class CorrectionError(ValueError):
    """Base class for every typed, catchable correction-store failure -- a route layer (a later slice)
    turns these into 400s, never a silent no-op. Mirrors clozn.runs.sessions.SessionValueError's role."""


class CorrectionValueError(CorrectionError):
    """A caller-supplied argument (scope, type, content, id shape) is malformed."""


class CorrectionNotFoundError(CorrectionError):
    """No correction exists with the given id."""


class CorrectionStateError(CorrectionError):
    """The requested transition is illegal from the correction's CURRENT state (e.g. disabling a
    never-confirmed draft, undoing past a deletion, confirming a deleted correction)."""


# ---------------------------------------------------------------------------------------------- identity

def _new_id() -> str:
    return "corr_" + secrets.token_hex(12)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]


def _normalize_model_sha256(value) -> str:
    text = str(value or "").strip().lower()
    if not _MODEL_SHA256_RE.fullmatch(text):
        raise CorrectionValueError("model scope value must be a 64-hex-char model_sha256 digest")
    return text


def validate_scope(scope_kind, scope_value) -> "tuple[str, str | None]":
    """Normalize and validate a (kind, value) pair. Raises CorrectionValueError on anything malformed.
    Returns (scope_kind, normalized_value_or_None). `global_local` never carries a value; every other
    kind requires one. session/client/project reuse clozn.runs.association's own opaque-key
    normalization (accept either a raw caller token or an already-opaque key) so a correction scoped to
    "this session" resolves against the IDENTICAL key runs.session_key already uses -- no second
    identity scheme invented here. `model` has no privacy property to protect (a model_sha256 is already
    a public content hash, not a caller-supplied raw token), so it is validated directly, never HMAC'd.
    """
    if scope_kind not in SCOPE_KINDS:
        raise CorrectionValueError(f"scope kind must be one of {SCOPE_KINDS}, got {scope_kind!r}")
    if scope_kind == "global_local":
        if scope_value not in (None, ""):
            raise CorrectionValueError("global_local scope carries no value")
        return scope_kind, None
    if scope_value is None or not str(scope_value).strip():
        raise CorrectionValueError(f"scope kind {scope_kind!r} requires a non-empty scope value")
    if scope_kind == "session":
        return scope_kind, association.session_key(scope_value)
    if scope_kind == "client":
        return scope_kind, association.client_key(scope_value)
    if scope_kind == "project":
        return scope_kind, association.project_key(scope_value)
    return scope_kind, _normalize_model_sha256(scope_value)


# ------------------------------------------------------------------------------------------ row <-> document

def _row_to_document(row) -> dict:
    doc = {
        "schema_version": SCHEMA_NAME,
        "id": row["id"],
        "scope": _scope_doc(row),
        "type": row["type"],
        "content_hash": row["content_hash"],
        "enabled": bool(row["enabled"]),
        "created_ts": float(row["created_ts"]),
        "created_at": row["created_at"],
    }
    if row["content"] is not None:
        doc["content"] = row["content"]
    if row["confirmed_ts"] is not None:
        doc["confirmed_ts"] = float(row["confirmed_ts"])
    if row["disabled_ts"] is not None:
        doc["disabled_ts"] = float(row["disabled_ts"])
    if row["deleted_ts"] is not None:
        doc["deleted_ts"] = float(row["deleted_ts"])
    if row["deleted_reason"]:
        doc["deleted_reason"] = row["deleted_reason"]
    return doc


def _fetch_row(db, correction_id: str):
    return db.execute("SELECT * FROM corrections WHERE id = ?", (correction_id,)).fetchone()


def _require_row(db, correction_id: str):
    if not isinstance(correction_id, str) or not _ID_RE.fullmatch(correction_id):
        raise CorrectionValueError("correction_id must match corr_<24 hex chars>")
    row = _fetch_row(db, correction_id)
    if row is None:
        raise CorrectionNotFoundError(f"no correction {correction_id!r}")
    return row


# --------------------------------------------------------------------------------------------- event ledger

def _append_event(db, correction_id: str, event_type: str, *, run_id=None, detail=None, ts=None) -> None:
    detail_json = (json.dumps(detail, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                   if detail else None)
    db.execute(
        "INSERT INTO correction_events(correction_id, event_type, event_ts, run_id, detail_json) "
        "VALUES (?, ?, ?, ?, ?)",
        (correction_id, event_type, ts if ts is not None else time.time(), run_id, detail_json),
    )


def _event_row_to_doc(row) -> dict:
    doc = {"seq": int(row["seq"]), "event_type": row["event_type"], "event_ts": float(row["event_ts"])}
    if row["run_id"]:
        doc["run_id"] = row["run_id"]
    if row["detail_json"]:
        try:
            doc["detail"] = json.loads(row["detail_json"])
        except Exception:
            pass
    return doc


def _events_for(db, correction_id: str, *, limit: int = _LIFECYCLE_LIMIT) -> list:
    rows = db.execute(
        "SELECT seq, event_type, event_ts, run_id, detail_json FROM correction_events "
        "WHERE correction_id = ? ORDER BY seq ASC LIMIT ?",
        (correction_id, limit),
    ).fetchall()
    return [_event_row_to_doc(r) for r in rows]


# ------------------------------------------------------------------------------------------------- lifecycle

def draft_correction(*, scope_kind: str, correction_type: str, content: str, scope_value=None) -> dict:
    """Create an INERT, unconfirmed correction. Never selected by resolve_corrections() until
    confirm_correction() is called separately and explicitly -- see the module docstring."""
    if correction_type not in TYPES:
        raise CorrectionValueError(f"type must be one of {TYPES}, got {correction_type!r}")
    if not isinstance(content, str) or not content.strip():
        raise CorrectionValueError("content must be a non-empty string")
    content = content.strip()
    if len(content) > 4000:
        raise CorrectionValueError("content must be at most 4000 characters")
    kind, value = validate_scope(scope_kind, scope_value)

    now = time.time()
    doc = {
        "id": _new_id(),
        "scope_kind": kind,
        "scope_value": value,
        "type": correction_type,
        "content": content,
        "content_hash": _content_hash(content),
        "enabled": 0,
        "created_ts": now,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
    }
    store._ensure()
    with closing(store._connect()) as db, db:
        db.execute(
            "INSERT INTO corrections(id, scope_kind, scope_value, type, content, content_hash, enabled, "
            "created_ts, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (doc["id"], doc["scope_kind"], doc["scope_value"], doc["type"], doc["content"],
             doc["content_hash"], doc["enabled"], doc["created_ts"], doc["created_at"]),
        )
        _append_event(db, doc["id"], "drafted", ts=now)
        row = _fetch_row(db, doc["id"])
    result = _row_to_document(row)
    schemas.validate(result, SCHEMA_NAME)
    return result


def _overlapping_candidates(db, *, correction_type: str, scope_kind: str, scope_value, exclude_id: str) -> list:
    """Every OTHER confirmed+enabled, non-deleted correction of the same type that COULD apply together
    with (scope_kind, scope_value) in some future resolve_corrections() call -- either because one of the
    two is global_local (always co-applies), or because both target the identical (kind, value). This is
    a hypothetical-overlap check for confirm-time warning only; resolve_corrections() itself never calls
    this and instead groups the CONCRETE rows a real query already matched."""
    rows = db.execute(
        "SELECT id, scope_kind, scope_value, confirmed_ts FROM corrections "
        "WHERE type = ? AND confirmed_ts IS NOT NULL AND enabled = 1 AND deleted_ts IS NULL AND id != ?",
        (correction_type, exclude_id),
    ).fetchall()
    out = []
    for row in rows:
        same_target = row["scope_kind"] == scope_kind and row["scope_value"] == scope_value
        either_global = scope_kind == "global_local" or row["scope_kind"] == "global_local"
        if same_target or either_global:
            out.append({
                "correction_id": row["id"],
                "scope": ({"kind": row["scope_kind"], "value": row["scope_value"]}
                          if row["scope_value"] else {"kind": row["scope_kind"]}),
                "confirmed_ts": float(row["confirmed_ts"]),
            })
    return out


def _confirm_row(db, correction_id: str, *, now: float, detail: "dict | None" = None) -> None:
    """The exact confirm mutation (confirmed_ts/enabled) plus its 'confirmed' event -- factored out of
    confirm_correction() so a caller that must commit an ADDITIONAL row in the SAME transaction (F6,
    clozn.runs.teaching_loop.verify_and_promote(), which appends a verification event immediately before
    promoting) can do so atomically, through this one mutation, rather than re-deriving the UPDATE
    statement a second time. Callers must have already validated state (not deleted, not already
    confirmed) and must be inside an open `db` transaction -- this helper does not itself validate or
    commit. `detail` is optional and defaults to None (the ordinary hand-confirm path via
    confirm_correction() below never sets it); a non-None detail is how a caller records WHY this
    particular confirmation happened without inventing a second event type."""
    db.execute("UPDATE corrections SET confirmed_ts = ?, enabled = 1 WHERE id = ?", (now, correction_id))
    _append_event(db, correction_id, "confirmed", ts=now, detail=detail)


def confirm_correction(correction_id: str) -> dict:
    """THE explicit-confirmation gate: a drafted correction becomes selectable by resolve_corrections()
    only after this call. Idempotent if already confirmed (returns the current document, no duplicate
    event) -- mirrors clozn.runs.sessions.create_session's idempotent-create philosophy. Returns
    {"correction": <doc>, "potential_conflicts": [...]} -- the latter is a best-effort, confirm-time
    surfacing of same-type corrections this one COULD conflict with; it never blocks confirmation (the
    user may genuinely want two competing corrections and let resolve_corrections()'s precedence decide
    and record the outcome later)."""
    store._ensure()
    with closing(store._connect()) as db, db:
        row = _require_row(db, correction_id)
        if row["deleted_ts"] is not None:
            raise CorrectionStateError(f"correction {correction_id!r} is deleted and cannot be confirmed")
        if row["confirmed_ts"] is not None:
            return {"correction": _row_to_document(row),
                    "potential_conflicts": _overlapping_candidates(
                        db, correction_type=row["type"], scope_kind=row["scope_kind"],
                        scope_value=row["scope_value"], exclude_id=correction_id)}
        now = time.time()
        _confirm_row(db, correction_id, now=now)
        conflicts = _overlapping_candidates(
            db, correction_type=row["type"], scope_kind=row["scope_kind"],
            scope_value=row["scope_value"], exclude_id=correction_id)
        row = _fetch_row(db, correction_id)
    return {"correction": _row_to_document(row), "potential_conflicts": conflicts}


def disable_correction(correction_id: str) -> dict:
    """Reversible: excludes this correction from future resolve_corrections() calls without touching its
    content, content_hash, confirmed_ts, or ANY past correction_events row (disable != delete history).
    Idempotent if already disabled."""
    store._ensure()
    with closing(store._connect()) as db, db:
        row = _require_row(db, correction_id)
        if row["deleted_ts"] is not None:
            raise CorrectionStateError(f"correction {correction_id!r} is deleted")
        if row["confirmed_ts"] is None:
            raise CorrectionStateError(
                f"correction {correction_id!r} was never confirmed; nothing to disable")
        if not row["enabled"]:
            return _row_to_document(row)
        now = time.time()
        db.execute("UPDATE corrections SET enabled = 0, disabled_ts = ? WHERE id = ?", (now, correction_id))
        _append_event(db, correction_id, "disabled", ts=now)
        row = _fetch_row(db, correction_id)
    return _row_to_document(row)


def enable_correction(correction_id: str) -> dict:
    """Re-enable a disabled (but not deleted) correction. Idempotent if already enabled."""
    store._ensure()
    with closing(store._connect()) as db, db:
        row = _require_row(db, correction_id)
        if row["deleted_ts"] is not None:
            raise CorrectionStateError(f"correction {correction_id!r} is deleted")
        if row["confirmed_ts"] is None:
            raise CorrectionStateError(
                f"correction {correction_id!r} was never confirmed; confirm it first")
        if row["enabled"]:
            return _row_to_document(row)
        now = time.time()
        db.execute("UPDATE corrections SET enabled = 1, disabled_ts = NULL WHERE id = ?", (correction_id,))
        _append_event(db, correction_id, "enabled", ts=now)
        row = _fetch_row(db, correction_id)
    return _row_to_document(row)


def delete_correction(correction_id: str, *, reason: "str | None" = None) -> dict:
    """PERMANENT within this slice -- see the module docstring's "DELETE IS NOT UNDOABLE" section.
    Scrubs `content` (key removed, never null-padded) but keeps `content_hash` and every
    correction_events row forever, so a run's applied_corrections receipt entry citing this correction
    stays checkable. Idempotent if already deleted."""
    store._ensure()
    with closing(store._connect()) as db, db:
        row = _require_row(db, correction_id)
        if row["deleted_ts"] is not None:
            return _row_to_document(row)
        now = time.time()
        db.execute(
            "UPDATE corrections SET enabled = 0, content = NULL, deleted_ts = ?, deleted_reason = ? "
            "WHERE id = ?",
            (now, (reason or None), correction_id),
        )
        _append_event(db, correction_id, "deleted", ts=now, detail=({"reason": reason} if reason else None))
        row = _fetch_row(db, correction_id)
    return _row_to_document(row)


def undo_last_change(correction_id: str) -> dict:
    """Revert the most recent STATE transition (confirmed / disabled / enabled) -- never a deletion, see
    the module docstring. Each undo itself records a new event of the inverse type (tagged
    detail.undo_of=<seq>), so calling this repeatedly toggles back and forth through real, inspectable
    history rather than needing a separate 'undone' marker or a redo concept."""
    store._ensure()
    with closing(store._connect()) as db, db:
        row = _require_row(db, correction_id)
        if row["deleted_ts"] is not None:
            raise CorrectionStateError(
                f"correction {correction_id!r} was deleted; deletion is permanent, nothing to undo")
        target = db.execute(
            "SELECT seq, event_type FROM correction_events WHERE correction_id = ? "
            "AND event_type IN ('confirmed', 'disabled', 'enabled') ORDER BY seq DESC LIMIT 1",
            (correction_id,),
        ).fetchone()
        if target is None:
            raise CorrectionStateError(f"correction {correction_id!r} has nothing undoable yet")
        now = time.time()
        kind = target["event_type"]
        if kind == "confirmed":
            db.execute(
                "UPDATE corrections SET confirmed_ts = NULL, enabled = 0 WHERE id = ?", (correction_id,))
            _append_event(db, correction_id, "drafted", ts=now, detail={"undo_of": target["seq"]})
        elif kind == "disabled":
            db.execute(
                "UPDATE corrections SET enabled = 1, disabled_ts = NULL WHERE id = ?", (correction_id,))
            _append_event(db, correction_id, "enabled", ts=now, detail={"undo_of": target["seq"]})
        else:  # "enabled"
            db.execute(
                "UPDATE corrections SET enabled = 0, disabled_ts = ? WHERE id = ?", (now, correction_id))
            _append_event(db, correction_id, "disabled", ts=now, detail={"undo_of": target["seq"]})
        row = _fetch_row(db, correction_id)
    return _row_to_document(row)


# --------------------------------------------------------------------------------------------------- reads

def get_correction(correction_id: str) -> "dict | None":
    if not isinstance(correction_id, str) or not _ID_RE.fullmatch(correction_id):
        return None
    store._ensure()
    with closing(store._connect()) as db:
        row = _fetch_row(db, correction_id)
    return _row_to_document(row) if row is not None else None


def list_corrections(*, scope_kind=None, scope_value=None, correction_type=None,
                      include_disabled: bool = True, include_deleted: bool = False,
                      limit: int = 200) -> list:
    clauses = []
    params: list = []
    if scope_kind is not None:
        clauses.append("scope_kind = ?")
        params.append(scope_kind)
    if scope_value is not None:
        clauses.append("scope_value = ?")
        params.append(scope_value)
    if correction_type is not None:
        clauses.append("type = ?")
        params.append(correction_type)
    if not include_disabled:
        clauses.append("enabled = 1")
    if not include_deleted:
        clauses.append("deleted_ts IS NULL")
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    store._ensure()
    with closing(store._connect()) as db:
        rows = db.execute(
            f"SELECT * FROM corrections {where} ORDER BY created_ts DESC LIMIT ?",
            (*params, max(0, int(limit))),
        ).fetchall()
    return [_row_to_document(r) for r in rows]


def has_active_corrections() -> bool:
    """Return whether any confirmed, enabled, non-deleted correction exists.

    This existence probe intentionally selects no content.  The gateway uses it to keep the normal
    no-correction request path byte-for-byte quiet while preserving ``resolve_corrections``'s
    content-blind selection boundary.
    """
    store._ensure()
    with closing(store._connect()) as db:
        row = db.execute(
            "SELECT 1 FROM corrections WHERE confirmed_ts IS NOT NULL AND enabled = 1 "
            "AND deleted_ts IS NULL LIMIT 1"
        ).fetchone()
    return row is not None


def export_correction(correction_id: str) -> "dict | None":
    """A portable, schema-governed bundle: the live document plus its complete event ledger, oldest
    first. Read-only -- never mutates the correction or writes an event for the export itself."""
    if not isinstance(correction_id, str) or not _ID_RE.fullmatch(correction_id):
        return None
    store._ensure()
    with closing(store._connect()) as db:
        row = _fetch_row(db, correction_id)
        if row is None:
            return None
        events = _events_for(db, correction_id)
    doc = {
        "schema_version": EXPORT_SCHEMA,
        "correction_id": correction_id,
        "exported_ts": time.time(),
        "correction": _row_to_document(row),
        "events": events,
    }
    schemas.validate(doc, EXPORT_SCHEMA)
    return doc


# ------------------------------------------------------------------------------------------------ resolution

def resolve_corrections(*, session_id=None, client_id=None, project_id=None, model_sha256=None,
                        include_global_local: bool = True, now: "float | None" = None) -> dict:
    """PURE, content-blind resolution: given ONLY discrete scope identifiers (never message content, a
    prompt, or anything text-similarity could run over -- see the module docstring), return which
    confirmed+enabled corrections apply and which same-type conflicts were surfaced deciding that.

    Arguments accept either a raw caller-known token or an already-opaque key, normalized the same way
    clozn.runs.store.find_runs already does for session_id/client_id (clozn.runs.association).
    `model_sha256` is the exact model identity digest (clozn.runs.identity), not a friendly model name --
    this project prefers exact reproducibility identity over labels everywhere else, and correction scope
    follows that precedent.
    """
    now = now if now is not None else time.time()
    resolved = {
        "session": association.session_key(session_id) if session_id is not None else None,
        "client": association.client_key(client_id) if client_id is not None else None,
        "project": association.project_key(project_id) if project_id is not None else None,
        "model": _normalize_model_sha256(model_sha256) if model_sha256 is not None else None,
    }
    scope_context: dict = {}
    if resolved["session"]:
        scope_context["session_key"] = resolved["session"]
    if resolved["client"]:
        scope_context["client_key"] = resolved["client"]
    if resolved["project"]:
        scope_context["project_key"] = resolved["project"]
    if resolved["model"]:
        scope_context["model_sha256"] = resolved["model"]
    scope_context["include_global_local"] = bool(include_global_local)

    store._ensure()
    clauses = []
    params: list = []
    if include_global_local:
        clauses.append("scope_kind = 'global_local'")
    for kind in ("session", "client", "project", "model"):
        value = resolved[kind]
        if value:
            clauses.append("(scope_kind = ? AND scope_value = ?)")
            params.extend([kind, value])
    if not clauses:
        candidates = []
    else:
        where = "confirmed_ts IS NOT NULL AND enabled = 1 AND deleted_ts IS NULL AND (" \
                + " OR ".join(clauses) + ")"
        with closing(store._connect()) as db:
            rows = db.execute(f"SELECT * FROM corrections WHERE {where}", params).fetchall()
        candidates = list(rows)

    by_type: dict = {}
    for row in candidates:
        by_type.setdefault(row["type"], []).append(row)

    applied: list = []
    conflicts: list = []
    for correction_type, group in by_type.items():
        if len(group) == 1:
            applied.append(_applied_entry(group[0]))
            continue
        winner, rule = _break_tie(group)
        applied.append(_applied_entry(winner))
        losers = [r for r in group if r["id"] != winner["id"]]
        conflicts.append({
            "type": correction_type,
            "candidates": [_candidate_entry(r) for r in group],
            "winner_id": winner["id"],
            "losing_ids": sorted(r["id"] for r in losers),
            "rule": rule,
        })

    doc = {
        "schema_version": RESOLUTION_SCHEMA,
        "created_ts": now,
        "scope_context": scope_context,
        "applied": applied,
        "conflicts": conflicts,
    }
    schemas.validate(doc, RESOLUTION_SCHEMA)
    return doc


def _scope_doc(row) -> dict:
    return {"kind": row["scope_kind"], "value": row["scope_value"]} if row["scope_value"] \
        else {"kind": row["scope_kind"]}


def _applied_entry(row) -> dict:
    entry = {
        "correction_id": row["id"],
        "type": row["type"],
        "scope": _scope_doc(row),
        "content_hash": row["content_hash"],
    }
    if row["confirmed_ts"] is not None:
        entry["confirmed_ts"] = float(row["confirmed_ts"])
    return entry


def _candidate_entry(row) -> dict:
    entry = {"correction_id": row["id"], "scope": _scope_doc(row)}
    if row["confirmed_ts"] is not None:
        entry["confirmed_ts"] = float(row["confirmed_ts"])
    return entry


def _break_tie(group: list):
    """Deterministic, always-recorded winner selection across a same-type conflict group -- see
    PRECEDENCE_ORDER's docstring for the policy and SAME_RANK_TIEBREAK for the fallback. Never reads
    `content`; ranks purely on scope_kind, confirmed_ts, and id."""
    best_rank = min(_RANK[r["scope_kind"]] for r in group)
    tied = [r for r in group if _RANK[r["scope_kind"]] == best_rank]
    if len(tied) == 1:
        return tied[0], "precedence"
    newest = max(float(r["confirmed_ts"]) for r in tied)
    newest_rows = [r for r in tied if float(r["confirmed_ts"]) == newest]
    if len(newest_rows) == 1:
        return newest_rows[0], "most_recently_confirmed"
    return sorted(newest_rows, key=lambda r: r["id"])[0], "correction_id"


def receipt_fields(resolution: dict) -> dict:
    """Trim a resolve_corrections() document to the exact shape clozn.runs.context_receipt.
    build_context_receipt's `applied_corrections`/`correction_conflicts` parameters expect -- the two
    are already 1:1 by construction (see resolve_corrections()); this is a named seam so a caller does
    not need to know that fact directly."""
    return {"applied_corrections": list(resolution.get("applied") or []),
            "correction_conflicts": list(resolution.get("conflicts") or [])}


def messages_for_resolution(resolution: dict) -> list[dict]:
    """Materialize the already-selected corrections as one system message.

    Resolution deliberately remains content-blind: selection is made from exact scope identities and
    the returned document contains only hashes/metadata.  Generation is the one point where the
    selected opaque payloads are allowed to be read.  This helper keeps that boundary explicit and
    refuses a missing/deleted payload rather than silently applying an incomplete correction set.
    The caller still owns the original request messages; this function returns only the additive
    system message so it cannot mutate a caller-owned list.
    """
    entries = list((resolution or {}).get("applied") or [])
    if not entries:
        return []
    lines: list[str] = []
    for entry in entries:
        correction_id = entry.get("correction_id") if isinstance(entry, dict) else None
        if not correction_id:
            raise CorrectionStateError("correction resolution contained an entry without an id")
        doc = get_correction(correction_id)
        if not isinstance(doc, dict) or not isinstance(doc.get("content"), str):
            raise CorrectionStateError(
                f"selected correction {correction_id!r} is no longer available for generation"
            )
        # The type is metadata only; the user-approved content remains the instruction payload.
        kind = str(entry.get("type") or "correction")
        lines.append(f"[{kind}] {doc['content']}")
    return [{
        "role": "system",
        "content": "Clozn confirmed corrections (explicitly approved by the user):\n- "
                   + "\n- ".join(lines),
    }]


def record_applications(run_id: str, resolution: dict) -> dict:
    """Append 'applied' events (winners) and 'conflict_lost' events (losers) to each affected
    correction's ledger, all tagged with `run_id`. Best-effort by design, mirroring
    clozn.runs.store._store_blob's "a side-write failure must cost its own field, never the run it
    describes" philosophy: `run_id` already exists (store.record() already succeeded and already
    embedded this same resolution's applied list in the run's receipt) by the time this is called, so a
    failure here must not be reported as if the run itself failed. Returns {"recorded_ids": [...],
    "errors": [...]}."""
    recorded: list = []
    errors: list = []
    try:
        store._ensure()
        with closing(store._connect()) as db, db:
            for entry in resolution.get("applied") or []:
                cid = entry.get("correction_id")
                if not cid:
                    continue
                try:
                    _append_event(db, cid, "applied", run_id=run_id,
                                 detail={"type": entry.get("type"), "scope": entry.get("scope")})
                    recorded.append(cid)
                except Exception as exc:                       # noqa: BLE001 -- degrade, never raise
                    errors.append(f"{cid}: {type(exc).__name__}: {exc}")
            for conflict in resolution.get("conflicts") or []:
                for cid in conflict.get("losing_ids") or []:
                    try:
                        _append_event(db, cid, "conflict_lost", run_id=run_id,
                                     detail={"type": conflict.get("type"),
                                             "winner_id": conflict.get("winner_id"),
                                             "rule": conflict.get("rule")})
                    except Exception as exc:                    # noqa: BLE001
                        errors.append(f"{cid}: {type(exc).__name__}: {exc}")
    except Exception as exc:                                    # noqa: BLE001 -- the run must survive this
        errors.append(f"{type(exc).__name__}: {exc}")
    return {"recorded_ids": recorded, "errors": errors}


def apply_and_record(*, run_kwargs: dict, session_id=None, client_id=None, project_id=None,
                     model_sha256=None, include_global_local: bool = True):
    """The single, structural forcing function described in the module docstring's "RECEIPT
    INTEGRATION" section: resolve corrections for the given scope context, thread the result into
    clozn.runs.store.record() so it lands in the created run's context receipt, then log the same
    resolution into this module's own event ledger. Returns (run_id_or_None, resolution).

    `run_kwargs` are forwarded to store.record() verbatim (this function does not know or care about the
    rest of a run's shape) with `applied_corrections`/`correction_conflicts` added -- the caller's own
    dict is never mutated."""
    resolution = resolve_corrections(
        session_id=session_id, client_id=client_id, project_id=project_id,
        model_sha256=model_sha256, include_global_local=include_global_local)
    fields = receipt_fields(resolution)
    kwargs = dict(run_kwargs)
    kwargs["applied_corrections"] = fields["applied_corrections"]
    kwargs["correction_conflicts"] = fields["correction_conflicts"]
    rid = store.record(**kwargs)
    if rid:
        record_applications(rid, resolution)
    return rid, resolution


__all__ = [
    "SCHEMA_NAME", "RESOLUTION_SCHEMA", "EXPORT_SCHEMA", "TYPES", "SCOPE_KINDS",
    "PRECEDENCE_ORDER", "SAME_RANK_TIEBREAK",
    "CorrectionError", "CorrectionValueError", "CorrectionNotFoundError", "CorrectionStateError",
    "validate_scope", "draft_correction", "confirm_correction", "disable_correction",
    "enable_correction", "delete_correction", "undo_last_change", "get_correction",
    "list_corrections", "has_active_corrections", "export_correction", "resolve_corrections", "receipt_fields",
    "messages_for_resolution",
    "record_applications", "apply_and_record",
]
