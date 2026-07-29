"""tensor_store.py -- content-addressed BINARY blobs for float32 tensors (slice 3.2's storage
primitive for clozn.analysis.mechanistic_diff, but generic: anything that needs to persist a raw
float32 vector/tensor off a model's residual stream can use this).

WHY NOT clozn.runs.store's blob store
--------------------------------------
clozn/runs/store.py already has a proven content-addressed blob pattern (`_store_blob`/`_load_blob`,
~lines 107-161): sha256-addressed, sharded on disk, digest re-verified on every read, a missing/corrupt
blob surfaced as `{"unavailable": reason}` rather than silently empty. That pattern is exactly right
here and is deliberately mirrored below -- but that store is JSON-only (`json.dumps`/`json.loads`). A
single captured residual row is `n_embd` (often 3072-8192) float32 values; as a JSON array of decimal
literals that is roughly 5-10x the bytes of the raw float32 data (ASCII digits + separators for every
component, vs 4 bytes each). Storing a mechanistic-diff run's worth of per-layer/per-position residuals
as JSON blobs would make the run store's own directory dominated by tensor bloat. This module is the
binary sibling: same content-addressing and honesty discipline, `array`-packed float32 bytes instead.

STDLIB ONLY
-----------
Per docs/SEAMS.md rule 1, no numpy: the `array` module packs/unpacks float32 exactly (`array('f', ...)`)
and `array.byteswap()` normalizes to little-endian on a big-endian host, matching the wire convention
`engine/client/clozn_engine.py`'s `decode_tensor` already documents (SPEC.md's tensor codec is
little-endian float32) -- a stored blob's bytes are therefore byte-identical to what a caller would get
by base64-decoding the same values straight off `/score`'s `captured` field.

WHY A SIDECAR, NOT SHAPE-IN-FILENAME
-------------------------------------
The blob's byte count alone cannot disambiguate shape: a [4, 3] and a [3, 4] and a [12] tensor of the
same values all serialize to the same 48 bytes. The whole point of "a blob is never ambiguous about
what it holds" is that shape/dtype/provenance travel in a JSON sidecar next to the binary blob, keyed by
the SAME digest. If that sidecar is missing or does not parse, this module refuses to guess a shape
rather than degrade to a bare flat array -- see `load_tensor`.

The sidecar is NOT part of the content address (only the raw float32 bytes are hashed): two identical
vectors captured under different provenance (different layer, different position, even a different
model) dedupe to the same blob, which is correct -- the BYTES are the same evidence regardless of why
they were captured. `provenance` is caller-defined and opaque to this module (e.g. {"model": "reference",
"layer": 14, "position": 37, "role": "residual_stream"} for mechanistic_diff's use), stored purely for a
reader's benefit.
"""
from __future__ import annotations

import array
import hashlib
import json
import os
import re
import sys
import tempfile

TENSOR_ROOT = os.path.join(os.path.expanduser("~/.clozn"), "tensors")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


def _blob_root() -> str:
    return os.path.join(TENSOR_ROOT, "blobs", "sha256")


def _blob_path(digest: str) -> str:
    return os.path.join(_blob_root(), digest[:2], digest + ".bin")


def _sidecar_path(digest: str) -> str:
    return os.path.join(_blob_root(), digest[:2], digest + ".json")


# ------------------------------------------------------------------------------------------- codec

def _encode_f32(values) -> bytes:
    """Little-endian float32, row-major. `array` uses the platform's native byte order, so a
    big-endian host gets its bytes flipped before writing -- the ON-DISK format is always LE,
    regardless of host, matching the engine wire codec's own convention."""
    arr = array.array("f", (float(v) for v in values))
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tobytes()


def _decode_f32(raw: bytes) -> list:
    arr = array.array("f")
    arr.frombytes(raw)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tolist()


def _atomic_write_bytes(path: str, data: bytes) -> None:
    """Binary sibling of clozn._io.atomic_write_json: write to a temp file in the same directory,
    flush+fsync, then os.replace -- there is never a moment where `path` is truncated or half-written."""
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


# ------------------------------------------------------------------------------------------- public API

def store_tensor(values, *, shape, dtype: str = "float32", provenance: dict | None = None) -> dict:
    """Persist a flat float32 tensor content-addressed; returns its ref.

    `values` is a flat, row-major sequence of numbers whose length must equal `shape`'s product (the
    caller's responsibility -- this function encodes, it does not reshape). Only `dtype="float32"` is
    supported (raises ValueError otherwise): this store exists for engine capture output, which is
    always float32 on the wire.

    Returns `{"sha256", "dtype", "shape", "bytes"}` -- everything a caller needs to embed in a stored
    artifact (e.g. clozn.mechanistic-diff.v1's `tensors` field) without touching disk again. Mirrors
    clozn.runs.store's blob-write-failure discipline: if the write itself fails (disk full, permission
    error, ...) the ref carries `write_failed` instead of raising, so ONE tensor's evidence loss costs
    only that tensor -- never propagated up to abort an entire capture run over a full disk.
    """
    if dtype != "float32":
        raise ValueError(f"tensor_store only supports dtype='float32', got {dtype!r}")
    shape_list = [int(d) for d in shape]
    expected = 1
    for dim in shape_list:
        expected *= dim
    values_list = list(values)
    if len(values_list) != expected:
        raise ValueError(
            f"values has {len(values_list)} elements, shape {shape_list} implies {expected}")

    raw = _encode_f32(values_list)
    digest = hashlib.sha256(raw).hexdigest()
    ref: dict = {"sha256": digest, "dtype": "float32", "shape": shape_list, "bytes": len(raw)}

    sidecar = {"sha256": digest, "dtype": "float32", "shape": shape_list, "bytes": len(raw)}
    if provenance:
        sidecar["provenance"] = dict(provenance)
    try:
        blob_path = _blob_path(digest)
        if not os.path.isfile(blob_path):
            _atomic_write_bytes(blob_path, raw)
        sidecar_path = _sidecar_path(digest)
        if not os.path.isfile(sidecar_path):
            text = json.dumps(sidecar, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            _atomic_write_bytes(sidecar_path, text.encode("utf-8"))
    except Exception as exc:      # noqa: BLE001 -- a write hiccup costs this tensor's ref, never raises
        ref["write_failed"] = f"{type(exc).__name__}: {exc}"
    return ref


def load_tensor(ref: dict) -> dict:
    """Load + verify a tensor by its ref, mirroring clozn.runs.store's `_load_blob` honesty discipline:
    a missing, unreadable, or digest-mismatched blob is SURFACED as `{"unavailable": reason, "sha256":
    ...}` rather than an empty/zero result -- a caller must never read "no data" as "no divergence." An
    absent/malformed ref, or a ref whose write already failed in `store_tensor`, is reported the same
    honest way rather than as a plain `{}` that could be mistaken for "zero tensor."

    The sidecar (shape/dtype/provenance) is re-read and re-verified alongside the blob; if it is
    missing or does not parse, the WHOLE tensor is reported unavailable rather than falling back to a
    guessed flat shape -- see the module docstring's "WHY A SIDECAR" section for why a shape guess here
    would be dishonest (byte count alone cannot disambiguate shape).

    Success shape: `{"ok": True, "sha256", "dtype", "shape", "bytes", "values": [...],
    "provenance": {...}?}` -- `provenance` is present only if the tensor was stored with one.
    """
    ref = ref or {}
    digest = str(ref.get("sha256") or "")
    write_failed = ref.get("write_failed")
    if write_failed:
        return {"unavailable": f"tensor evidence write failed: {write_failed}", "sha256": digest or None}
    if not _DIGEST_RE.match(digest):
        return {"unavailable": "no valid tensor ref (missing or malformed sha256)"}

    blob_path = _blob_path(digest)
    try:
        with open(blob_path, "rb") as handle:
            raw = handle.read()
    except FileNotFoundError:
        return {"unavailable": "tensor blob missing", "sha256": digest}
    except OSError as exc:
        return {"unavailable": f"tensor blob unreadable: {type(exc).__name__}", "sha256": digest}
    actual = hashlib.sha256(raw).hexdigest()
    if actual != digest:
        return {"unavailable": "tensor blob corrupt (digest mismatch)", "sha256": digest,
                "actual_sha256": actual}

    sidecar_path = _sidecar_path(digest)
    try:
        with open(sidecar_path, encoding="utf-8") as handle:
            sidecar = json.load(handle)
    except FileNotFoundError:
        return {"unavailable": "tensor sidecar missing -- shape/dtype cannot be confirmed, refusing "
                                "to guess", "sha256": digest}
    except (OSError, json.JSONDecodeError) as exc:
        return {"unavailable": f"tensor sidecar unreadable: {type(exc).__name__}", "sha256": digest}
    if not isinstance(sidecar, dict) or sidecar.get("sha256") != digest:
        return {"unavailable": "tensor sidecar does not match this blob's digest", "sha256": digest}
    shape = sidecar.get("shape")
    dtype = sidecar.get("dtype")
    if not isinstance(shape, list) or dtype != "float32":
        return {"unavailable": "tensor sidecar has an invalid shape/dtype", "sha256": digest}

    try:
        values = _decode_f32(raw)
    except Exception as exc:      # noqa: BLE001 -- surfaced as unavailable, never a half-decoded result
        return {"unavailable": f"tensor blob not valid float32 data: {type(exc).__name__}",
                "sha256": digest}

    out = {"ok": True, "sha256": digest, "dtype": "float32", "shape": shape,
           "bytes": len(raw), "values": values}
    provenance = sidecar.get("provenance")
    if isinstance(provenance, dict) and provenance:
        out["provenance"] = provenance
    return out
