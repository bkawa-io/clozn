"""Blob garbage collection for the SQLite run store (BACKLOG §2).

Every blob (a run's trace, or its persisted Phase-3.7 context<->answer influence-map artifact) lives at
`blobs/sha256/<xx>/<digest>.json`, content-addressed and referenced by exactly the `trace_ref.sha256`,
`influence_map_ref.sha256`, or `minimal_context_support_ref.sha256` embedded in a run's `payload_json`.
A blob becomes ORPHANED the moment its only
referencing run row is gone -- pruned by `store._prune()` (KEEP=1000 rows), replaced by `replace_run()`
with a different trace/map, or deleted by hand. Nothing has ever cleaned these up; they just accumulate.

This module never deletes anything on its own -- `store.py` does not import it, and nothing calls it
except the `clozn migrate --gc` CLI path. That is intentional: blob deletion is a destructive,
explicitly-requested operation, not something that happens as a side effect of an unrelated read/write.

Safety invariants (read before touching this file):
  - The reference set is built EXCLUSIVELY by reading supported blob refs out of every row's payload_json
    in the DB, read fresh at call time -- never inferred from directory structure, a cache, or a prior
    call's result.
  - Every candidate path is re-verified with `os.path.realpath` + `os.path.commonpath` against the blob
    root immediately before an unlink is attempted -- refused (not deleted) if it would land outside the
    root. The sha256-hex filename regex already makes an escape structurally impossible, but the
    containment check is unconditional anyway: never trust a single layer of "should be impossible" next
    to a delete call.
  - `collect(dry_run=True)` (the default) only computes and returns the plan; nothing is unlinked unless a
    caller explicitly passes `dry_run=False`.
  - Live deletion re-reads the referenced set immediately before deleting (not the one `plan()` computed
    moments earlier) so a run recorded between planning and deleting can never be destroyed out from under
    itself (TOCTOU-safe by construction, not by timing luck).
"""
from __future__ import annotations

import glob
import json
import os
import re
from contextlib import closing

from . import store

_BLOB_REL_RE = re.compile(r"^([0-9a-f]{2})[/\\]([0-9a-f]{64})\.json$")


_BLOB_REF_KEYS = ("trace_ref", "influence_map_ref", "minimal_context_support_ref")


def _referenced_digests(db) -> set[str]:
    """Every trace_ref.sha256 or influence_map_ref.sha256 currently reachable from a run row. Both live
    in the SAME content-addressed blobs/sha256 namespace (see store._pack), so one shared scan covers
    both. Parsed out of payload_json per row -- there is no separate indexed column for either (they live
    inside the JSON document) -- KEEP=1000 caps this at a small, cheap full scan."""
    out: set[str] = set()
    rows = db.execute("SELECT payload_json FROM runs").fetchall()
    for row in rows:
        try:
            payload = json.loads(row["payload_json"])
        except Exception:
            continue                                    # a corrupt row references nothing we can trust
        payload = payload or {}
        for key in _BLOB_REF_KEYS:
            ref = payload.get(key) or {}
            digest = ref.get("sha256") if isinstance(ref, dict) else None
            if isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest):
                out.add(digest)
    return out


def _digest_from_path(root: str, path: str) -> str | None:
    """The digest a blob file's path claims to hold, or None if the path doesn't match the exact
    `<2-hex>/<64-hex>.json` shape `store._blob_path` produces -- anything else is left alone (`malformed`
    in the plan, never a deletion candidate)."""
    rel = os.path.relpath(path, root).replace(os.sep, "/")
    m = _BLOB_REL_RE.match(rel)
    if not m or m.group(1) != m.group(2)[:2]:           # dir prefix must match the digest's own first byte
        return None
    return m.group(2)


def _is_contained(root: str, path: str) -> bool:
    """True iff `path` resolves to somewhere inside `root` (symlinks/`..` resolved first). The one check
    every deletion in this module passes through right before `os.remove` -- see module docstring."""
    root_real = os.path.realpath(root)
    path_real = os.path.realpath(path)
    try:
        return os.path.commonpath([root_real, path_real]) == root_real
    except ValueError:
        return False                                     # e.g. different drives on Windows -> not contained


def _safe_size(path: str) -> int:
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def plan(*, blob_root: str | None = None) -> dict:
    """Compute the GC plan without touching disk. `keep`/`delete` are lists of {"path", "digest", "bytes"};
    `malformed` lists anything on disk that doesn't even look like a content-addressed blob (never a
    deletion candidate, reported so `doctor`-style output can flag it as an anomaly)."""
    store._ensure()                                       # guarantee the `runs` table (and migrations) exist
    root = os.path.abspath(blob_root or store._blob_root())
    with closing(store._connect()) as db:
        referenced = _referenced_digests(db)
    keep, delete, malformed = [], [], []
    for path in glob.glob(os.path.join(root, "*", "*.json")):
        digest = _digest_from_path(root, path)
        entry = {"path": path, "digest": digest, "bytes": _safe_size(path)}
        if digest is None:
            malformed.append(entry)
        elif digest in referenced:
            keep.append(entry)
        else:
            delete.append(entry)
    return {
        "blob_root": root,
        "referenced_count": len(referenced),
        "total_blobs": len(keep) + len(delete),
        "keep": keep,
        "delete": delete,
        "malformed": malformed,
    }


def collect(*, dry_run: bool = True, blob_root: str | None = None) -> dict:
    """Run GC. `dry_run=True` (default) returns the plan untouched, `deleted`/`failed` both empty.
    `dry_run=False` unlinks every `delete` candidate that STILL isn't referenced (re-checked fresh, see
    module docstring) and STILL passes the containment check, and reports exactly what was removed vs. what
    was refused/failed and why -- so a caller never has to guess what actually happened on disk."""
    result = plan(blob_root=blob_root)
    result["dry_run"] = dry_run
    if dry_run:
        result["deleted"] = []
        result["failed"] = []
        return result

    root = result["blob_root"]
    with closing(store._connect()) as db:
        referenced_now = _referenced_digests(db)         # re-read fresh -- NOT result["keep"]'s digests

    deleted, failed = [], []
    for entry in result["delete"]:
        digest, path = entry["digest"], entry["path"]
        if digest in referenced_now:
            continue                                      # a run landed since plan() -- treat as kept, never delete
        if not _is_contained(root, path):
            failed.append({**entry, "error": "refused: path escapes the blob root"})
            continue
        try:
            os.remove(path)
        except OSError as exc:
            failed.append({**entry, "error": f"{type(exc).__name__}: {exc}"})
        else:
            deleted.append(entry)

    result["deleted"] = deleted
    result["failed"] = failed
    return result
