"""checkpoint_pin_store.py -- FORK-PIN-01: durable persistence for pinned execution-fork checkpoints.

WHY THIS EXISTS
----------------
clozn.replay.checkpoint_capture materializes an EPHEMERAL checkpoint: bounded worker-memory state
that is gone the moment the worker restarts, gets FIFO-evicted, or the gateway shuts down (see that
module's own docstring, and clozn.schemas.defs's clozn.checkpoint-reference.v1.json, which hard-codes
``pinned: const false`` for exactly this reason). This module is what makes a checkpoint OUTLIVE that
-- it takes the export envelope the engine's POST /v1/checkpoint/export returns (see
engine/core/serve/checkpoint_codec.hpp) and persists it:

  - the raw KV bytes go into a content-addressed BINARY blob (mirrors clozn.analysis.tensor_store's
    ``.bin`` blob + ``.json`` sidecar convention exactly -- that module's own docstring explains why
    JSON-only storage is wrong for large binary payloads: ~5-10x the bytes as decimal-literal text).
  - everything else (identity, state, both hashes) is the sidecar, plus a SQLite row
    (``pinned_checkpoints``, clozn/runs/migrations.py migration 3) that makes "is run X pinned"
    answerable without ever touching the blob file.
  - the full manifest (identity + state + blob refs) is schema-governed
    (clozn.schemas.defs.clozn.pinned-checkpoint.v1.json) and stored verbatim in the row's
    ``manifest_json`` column, so a reader never has to reconstruct it field-by-field from separate
    columns.

LOSSLESS ONLY, FAIL CLOSED ON PARTIAL WRITES
---------------------------------------------
Unlike clozn.runs.store's blob helpers (which degrade a write failure to a `write_failed` ref so one
lost trace doesn't cost an entire run), a pin's entire purpose is durability -- so `pin_checkpoint`
raises on ANY write failure rather than recording a pin that claims bytes exist when they might not.
The write order is deliberate: blob bytes are fsync'd to disk FIRST (an orphaned blob with no SQLite
row is harmless -- content-addressed, GC-able, and simply invisible to `list_pins`/`resolve_pin`);
only once that succeeds does the SQLite INSERT run inside its own transaction. So the only possible
half-written state after any failure is an orphan blob with NO row -- never a row that lies about
having bytes. SQLite remains authoritative (clozn/runs/store.py's own header comment), exactly as for
runs themselves.

SENSITIVITY
-----------
A KV cache is a lossy encoding of the prompt (and often the response) it was built from -- treat pin
blobs and their sidecars at least as sensitively as a run's own prompt/response text. This module does
not implement its own redaction path; `unpin_checkpoint` is the deletion primitive redaction/retention
tooling should call (mirroring clozn.runs.mutations's own trace/influence-map blob cleanup discipline:
the digest is checked for OTHER referencing rows before the blob file is removed, so a
deduplicated blob is never pulled out from under a still-live pin).
"""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
import base64
import hashlib
import json
import os
import re
import sqlite3
import tempfile
import time
import uuid

from clozn import schemas
from clozn.runs import store as runs_store

SCHEMA_VERSION = "clozn.pinned-checkpoint.v1"
PIN_ROOT = os.path.join(runs_store.RUNS_DIR, "blobs", "checkpoints", "sha256")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class PinStoreError(RuntimeError):
    """A pin/unpin/resolve operation could not be completed."""


class PinHasDependentsError(PinStoreError):
    """The pinned run has child runs; unpin refused without cascade=True (mirrors
    clozn.runs.mutations.RunHasChildrenError's contract exactly -- a pinned checkpoint's whole point
    is to be a durable fork point, so silently orphaning runs already forked from it is the one thing
    unpin must never do by default)."""

    def __init__(self, run_id: str, children: list[str]):
        self.run_id = run_id
        self.children = list(children)
        preview = ", ".join(self.children[:5]) + (", ..." if len(self.children) > 5 else "")
        super().__init__(
            f"run {run_id!r} has {len(self.children)} child run(s) ({preview}) depending on its "
            f"pinned checkpoint and cannot be unpinned without cascade=True"
        )


def _blob_path(digest: str) -> str:
    return os.path.join(PIN_ROOT, digest[:2], digest + ".bin")


def _sidecar_path(digest: str) -> str:
    return os.path.join(PIN_ROOT, digest[:2], digest + ".json")


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Binary sibling of clozn._io.atomic_write_json / clozn.analysis.tensor_store._atomic_write_bytes:
    write to a temp file in the same directory, flush+fsync, then os.replace -- there is never a
    moment where `path` is truncated or half-written."""
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-atomic-", suffix=".bin")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def _require_envelope(envelope: Mapping) -> dict:
    if not isinstance(envelope, Mapping):
        raise PinStoreError("envelope must be an object (see POST /v1/checkpoint/export)")
    if envelope.get("envelope_version") != "clozn.checkpoint-export.v1":
        raise PinStoreError(
            f"unsupported envelope_version {envelope.get('envelope_version')!r} "
            "(expected clozn.checkpoint-export.v1)"
        )
    identity = envelope.get("identity")
    state = envelope.get("state")
    payload_sha256 = envelope.get("payload_sha256")
    if not isinstance(identity, Mapping):
        raise PinStoreError("envelope missing 'identity'")
    if not isinstance(state, Mapping):
        raise PinStoreError("envelope missing 'state'")
    if not (isinstance(payload_sha256, str) and _DIGEST_RE.fullmatch(payload_sha256)):
        raise PinStoreError("envelope missing a valid 'payload_sha256'")
    return dict(envelope)


def pin_checkpoint(
    run_id: str,
    envelope: Mapping,
    *,
    checkpoint_id: str,
    source_worker_generation_id: str,
    note: str | None = None,
    clock=time.time,
) -> dict:
    """Durably persist one checkpoint export envelope against `run_id`, replacing any prior pin for
    that run. Returns the schema-validated manifest (clozn.pinned-checkpoint.v1) that was stored.

    `checkpoint_id` / `source_worker_generation_id` are the EPHEMERAL identity of the in-memory
    checkpoint this pin was captured from -- recorded for audit only (`source` in the manifest), never
    compared on resolution (see clozn.pinned-checkpoint.v1.json's Source $def docstring).
    """
    if not runs_store._valid_rid(run_id):
        raise PinStoreError("run_id must be an exact valid run ID")
    if not (isinstance(checkpoint_id, str) and checkpoint_id):
        raise PinStoreError("checkpoint_id must be a non-empty string")
    if not (isinstance(source_worker_generation_id, str) and source_worker_generation_id):
        raise PinStoreError("source_worker_generation_id must be a non-empty string")

    env = _require_envelope(envelope)
    identity = dict(env["identity"])
    state = dict(env["state"])
    payload_sha256 = env["payload_sha256"]

    kv_b64 = state.get("kv_data_b64")
    if not (isinstance(kv_b64, str) and kv_b64):
        raise PinStoreError("envelope state missing 'kv_data_b64'")
    try:
        raw_kv = base64.b64decode(kv_b64, validate=True)
    except Exception as exc:
        raise PinStoreError(f"envelope kv_data_b64 is not valid base64: {exc}") from None
    if not raw_kv:
        raise PinStoreError("envelope kv_data_b64 decoded to zero bytes")

    declared_kv_bytes = state.get("kv_bytes")
    if declared_kv_bytes != len(raw_kv):
        raise PinStoreError("envelope kv_bytes does not match the decoded kv_data_b64 length")

    envelope_bytes = len(
        json.dumps(env, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))

    digest = hashlib.sha256(raw_kv).hexdigest()
    sidecar = {
        "sha256": digest,
        "kv_bytes": len(raw_kv),
        # The FULL identity object, worker_generation_id included -- resolve_pin() reconstructs the
        # exact original export envelope from this sidecar, and the ORIGINAL envelope's identity DID
        # carry it (checkpoint_identity_json() on the C++ side always includes it). The manifest below
        # deliberately does NOT duplicate it here (see manifest_identity) -- it already has its own,
        # canonical home at manifest["source"]["worker_generation_id"].
        "identity": identity,
        "state": {k: v for k, v in state.items() if k != "kv_data_b64"},
        "payload_sha256": payload_sha256,
    }
    # The schema-governed manifest's `identity` is the fail-closed COMPATIBILITY fingerprint only
    # (clozn.pinned-checkpoint.v1.json's Identity $def is deliberately closed/additionalProperties:
    # false over exactly those axes) -- worker_generation_id is PROVENANCE, not a compatibility axis
    # (see checkpoint_identity_json's own comment: comparing it at import would defeat pinning's whole
    # point), and already lives at manifest["source"]["worker_generation_id"]. Strip it here rather
    # than widen the schema to accept a field nothing should ever compare against.
    manifest_identity = {k: v for k, v in identity.items() if k != "worker_generation_id"}

    pin_id = "pin_" + uuid.uuid4().hex[:20]
    pinned_ts = float(clock())
    pinned_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(pinned_ts))

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "pin_id": pin_id,
        "run_id": run_id,
        "pinned_ts": pinned_ts,
        "pinned_at": pinned_at,
        "source": {
            "checkpoint_id": checkpoint_id,
            "worker_generation_id": source_worker_generation_id,
        },
        "identity": manifest_identity,
        "state": {
            "n_tokens": state.get("n_tokens"),
            "n_past": state.get("n_past"),
            "prompt_tokens": state.get("prompt_tokens"),
            "causal": bool(state.get("causal", True)),
            "has_sampler": bool(state.get("has_sampler", False)),
            "has_steer": bool(state.get("has_steer", False)),
        },
        "blob": {
            "sha256": digest,
            "kv_bytes": len(raw_kv),
            "envelope_bytes": envelope_bytes,
            "payload_sha256": payload_sha256,
        },
    }
    if note:
        manifest["note"] = str(note)
    schemas.validate(manifest, SCHEMA_VERSION)   # schema-first: refuse before ANY bytes are written

    # Write the blob bytes FIRST (lossless-only discipline -- see module docstring): a write failure
    # here raises immediately and NOTHING is recorded in SQLite, so a failed pin never claims bytes
    # that do not durably exist. Content-addressed: skip the write if this exact digest already
    # exists (a re-pin of byte-identical state, or a dedup across runs).
    blob_path = _blob_path(digest)
    if not os.path.isfile(blob_path):
        _atomic_write_bytes(blob_path, raw_kv)
    sidecar_path = _sidecar_path(digest)
    if not os.path.isfile(sidecar_path):
        encoded_sidecar = json.dumps(
            sidecar, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        _atomic_write_bytes(sidecar_path, encoded_sidecar)

    runs_store._ensure()
    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    try:
        with closing(runs_store._connect()) as db:
            db.isolation_level = None
            db.execute("BEGIN IMMEDIATE")
            try:
                db.execute(
                    "INSERT INTO pinned_checkpoints("
                    " run_id, pin_id, pinned_ts, pinned_at, note, checkpoint_id,"
                    " source_worker_generation_id, model_sha256, architecture, n_embd, n_layer,"
                    " vocab_size, n_ctx, protocol_version, engine_version, build_id, llama_cpp_commit,"
                    " n_tokens, n_past, prompt_tokens, causal, has_sampler, has_steer,"
                    " blob_sha256, kv_bytes, envelope_bytes, payload_sha256, manifest_json"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(run_id) DO UPDATE SET"
                    " pin_id=excluded.pin_id, pinned_ts=excluded.pinned_ts, pinned_at=excluded.pinned_at,"
                    " note=excluded.note, checkpoint_id=excluded.checkpoint_id,"
                    " source_worker_generation_id=excluded.source_worker_generation_id,"
                    " model_sha256=excluded.model_sha256, architecture=excluded.architecture,"
                    " n_embd=excluded.n_embd, n_layer=excluded.n_layer, vocab_size=excluded.vocab_size,"
                    " n_ctx=excluded.n_ctx, protocol_version=excluded.protocol_version,"
                    " engine_version=excluded.engine_version, build_id=excluded.build_id,"
                    " llama_cpp_commit=excluded.llama_cpp_commit, n_tokens=excluded.n_tokens,"
                    " n_past=excluded.n_past, prompt_tokens=excluded.prompt_tokens, causal=excluded.causal,"
                    " has_sampler=excluded.has_sampler, has_steer=excluded.has_steer,"
                    " blob_sha256=excluded.blob_sha256, kv_bytes=excluded.kv_bytes,"
                    " envelope_bytes=excluded.envelope_bytes, payload_sha256=excluded.payload_sha256,"
                    " manifest_json=excluded.manifest_json",
                    (
                        run_id, pin_id, pinned_ts, pinned_at, note, checkpoint_id,
                        source_worker_generation_id, identity.get("model_sha256"),
                        identity.get("architecture"), identity.get("n_embd"), identity.get("n_layer"),
                        identity.get("vocab_size"), identity.get("n_ctx"),
                        identity.get("protocol_version"), identity.get("engine_version"),
                        identity.get("build_id"), identity.get("llama_cpp_commit"),
                        state.get("n_tokens"), state.get("n_past"), state.get("prompt_tokens"),
                        1 if state.get("causal", True) else 0,
                        1 if state.get("has_sampler", False) else 0,
                        1 if state.get("has_steer", False) else 0,
                        digest, len(raw_kv), envelope_bytes, payload_sha256, manifest_json,
                    ),
                )
            except BaseException:
                db.execute("ROLLBACK")
                raise
            else:
                db.execute("COMMIT")
    except sqlite3.Error as exc:
        raise PinStoreError(f"could not record pin for run {run_id!r}: {type(exc).__name__}: {exc}") from None

    return manifest


def get_pin(run_id: str) -> dict | None:
    """The stored manifest for `run_id`'s pin, or None if it has no pin. Does not touch the blob."""
    if not runs_store._valid_rid(run_id):
        return None
    runs_store._ensure()
    with closing(runs_store._connect()) as db:
        row = db.execute(
            "SELECT manifest_json FROM pinned_checkpoints WHERE run_id = ?", (run_id,)
        ).fetchone()
    if row is None:
        return None
    try:
        value = json.loads(row["manifest_json"])
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def list_pins() -> list[dict]:
    """Every pinned checkpoint's manifest, most recently pinned first."""
    runs_store._ensure()
    with closing(runs_store._connect()) as db:
        rows = db.execute(
            "SELECT manifest_json FROM pinned_checkpoints ORDER BY pinned_ts DESC, run_id DESC"
        ).fetchall()
    out = []
    for row in rows:
        try:
            value = json.loads(row["manifest_json"])
        except Exception:
            continue
        if isinstance(value, dict):
            out.append(value)
    return out


def resolve_pin(run_id: str) -> dict:
    """Reconstruct the full export envelope for `run_id`'s pin, re-verifying every digest along the
    way -- the read-time honesty discipline clozn.runs.store._load_blob / clozn.analysis.tensor_store
    .load_tensor both use: a missing/corrupt blob or sidecar is SURFACED as {"unavailable": reason},
    never silently treated as "no pin" or a guessed/partial reconstruction. Success shape:
    {"ok": True, "manifest": {...}, "envelope": {...same shape POST /v1/checkpoint/export returns...}}.
    """
    manifest = get_pin(run_id)
    if manifest is None:
        return {"unavailable": f"run {run_id!r} has no pinned checkpoint"}

    digest = manifest.get("blob", {}).get("sha256")
    if not (isinstance(digest, str) and _DIGEST_RE.fullmatch(digest)):
        return {"unavailable": "pin manifest has no valid blob.sha256", "sha256": digest}

    blob_path = _blob_path(digest)
    try:
        with open(blob_path, "rb") as handle:
            raw_kv = handle.read()
    except FileNotFoundError:
        return {"unavailable": "pinned checkpoint blob missing", "sha256": digest}
    except OSError as exc:
        return {"unavailable": f"pinned checkpoint blob unreadable: {type(exc).__name__}", "sha256": digest}
    actual = hashlib.sha256(raw_kv).hexdigest()
    if actual != digest:
        return {"unavailable": "pinned checkpoint blob corrupt (digest mismatch)",
                "sha256": digest, "actual_sha256": actual}

    sidecar_path = _sidecar_path(digest)
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            sidecar = json.load(handle)
    except FileNotFoundError:
        return {"unavailable": "pinned checkpoint sidecar missing -- identity/state cannot be "
                                "confirmed, refusing to guess", "sha256": digest}
    except (OSError, json.JSONDecodeError) as exc:
        return {"unavailable": f"pinned checkpoint sidecar unreadable: {type(exc).__name__}",
                "sha256": digest}
    if not isinstance(sidecar, dict) or sidecar.get("sha256") != digest:
        return {"unavailable": "pinned checkpoint sidecar does not match this blob's digest",
                "sha256": digest}

    state = dict(sidecar.get("state") or {})
    state["kv_data_b64"] = base64.b64encode(raw_kv).decode("ascii")
    envelope = {
        "envelope_version": "clozn.checkpoint-export.v1",
        "identity": sidecar.get("identity"),
        "state": state,
        "payload_sha256": sidecar.get("payload_sha256"),
    }
    return {"ok": True, "manifest": manifest, "envelope": envelope}


def _direct_children(db: sqlite3.Connection, run_id: str) -> list[str]:
    rows = db.execute(
        "SELECT id FROM runs WHERE parent_run_id = ? ORDER BY recorded_ts, id", (run_id,)
    ).fetchall()
    return [row["id"] for row in rows]


def _blob_referenced_elsewhere(db: sqlite3.Connection, digest: str, *, excluding_run_id: str) -> bool:
    row = db.execute(
        "SELECT 1 FROM pinned_checkpoints WHERE blob_sha256 = ? AND run_id != ? LIMIT 1",
        (digest, excluding_run_id),
    ).fetchone()
    return row is not None


def unpin_checkpoint(run_id: str, *, cascade: bool = False) -> dict:
    """Remove `run_id`'s pin: the SQLite row first (transactional -- see module docstring), then the
    blob/sidecar files IF no other pin still references the same digest (content-addressed dedup
    safety, mirroring clozn.runs.mutations._cleanup_blob). Refuses if `run_id` has child runs unless
    `cascade=True` (PinHasDependentsError) -- a pin's whole point is to be a durable fork point for
    children that may not have any other record of it once the ephemeral in-memory checkpoint is gone.
    """
    if not runs_store._valid_rid(run_id):
        raise PinStoreError("run_id must be an exact valid run ID")
    if not isinstance(cascade, bool):
        raise PinStoreError("cascade must be a boolean")

    runs_store._ensure()
    digest = None
    try:
        with closing(runs_store._connect()) as db:
            db.isolation_level = None
            db.execute("BEGIN IMMEDIATE")
            try:
                row = db.execute(
                    "SELECT blob_sha256 FROM pinned_checkpoints WHERE run_id = ?", (run_id,)
                ).fetchone()
                if row is None:
                    db.execute("ROLLBACK")
                    raise PinStoreError(f"run {run_id!r} has no pinned checkpoint")
                digest = row["blob_sha256"]
                children = _direct_children(db, run_id)
                if children and not cascade:
                    db.execute("ROLLBACK")
                    raise PinHasDependentsError(run_id, children)
                referenced_elsewhere = _blob_referenced_elsewhere(db, digest, excluding_run_id=run_id)
                db.execute("DELETE FROM pinned_checkpoints WHERE run_id = ?", (run_id,))
            except BaseException:
                try:
                    db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
                raise
            else:
                db.execute("COMMIT")
    except (PinStoreError, PinHasDependentsError):
        raise
    except sqlite3.Error as exc:
        raise PinStoreError(f"could not unpin run {run_id!r}: {type(exc).__name__}: {exc}") from None

    blob_cleanup = {"status": "retained_shared", "sha256": digest} if referenced_elsewhere else None
    if blob_cleanup is None:
        blob_path, sidecar_path = _blob_path(digest), _sidecar_path(digest)
        try:
            if os.path.isfile(blob_path):
                os.remove(blob_path)
            if os.path.isfile(sidecar_path):
                os.remove(sidecar_path)
            blob_cleanup = {"status": "deleted", "sha256": digest}
        except OSError as exc:
            blob_cleanup = {"status": "failed", "sha256": digest, "error": f"{type(exc).__name__}: {exc}"}

    return {"ok": True, "action": "unpin", "run_id": run_id, "cascade": cascade, "blob_cleanup": blob_cleanup}


__all__ = [
    "PIN_ROOT", "SCHEMA_VERSION", "PinStoreError", "PinHasDependentsError",
    "get_pin", "list_pins", "pin_checkpoint", "resolve_pin", "unpin_checkpoint",
]
