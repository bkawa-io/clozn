"""clozn/runs/identity.py -- immutable reproduction identity for run records (roadmap S4.3).

The external audit (notes/POSITIONING_AUDIT_B_2026-07.md S1.3/S3.2) and Hugging Face's own analysis
(96.5% of 50k+ eval records missing minimal reproduction fields) both name the same gap: a run record
can carry a filename and a quant guess, but nothing that PROVES which exact model bytes and which exact
chat-template rendering produced a reply. This module assembles that block:

    {model_path, model_sha256, model_size_bytes, template_fingerprint, engine_build, clozn_version,
     captured_at}

Every key is OMITTED, never null-padded, when it cannot be honestly measured -- absence must stay
visible (the same "don't invent a value" rule the rest of clozn.runs follows for meta/trace fields).

WHY A CACHE IS MANDATORY
-------------------------
Hashing a 5-8 GB GGUF takes tens of seconds. `clozn.artifacts.contracts.gguf_identity()` already pays
this cost, uncached, once per engine boot (clozn/cli/engine_process.py's spawn_engine()). This module's
`model_sha256()` adds a persistent cache keyed by (absolute path, size, mtime_ns) at
~/.clozn/cache/model_hashes.json, so re-hashing an unchanged file -- across runs, across processes, even
across the engine boots gguf_identity() itself pays for -- costs one os.stat() instead of a full re-read.

WHY THE HOT PATH NEVER PAYS FOR THIS
-------------------------------------
The product's own engine already reports a `model_sha256` field on GET /health (server_main.cpp),
computed at process boot via the exact gguf_identity()->sha256_file() path above and handed to the
worker with `--model-sha256`. clozn.server.substrates.EngineSubstrate.run_meta() -- which already
fetches /health once per process and caches the result -- is where this module gets wired in
(EngineSubstrate.identity_meta()): it prefers that already-computed, zero-marginal-cost health field
over calling model_sha256() itself. Only when an engine build/launch path doesn't report model_sha256
(e.g. an older worker binary, or CLOZN_ENGINE_BIN pointed at a file clozn never hashed at boot) does
run_meta() fall back to this module's own model_sha256(), and even then only once per process (the
existing run_meta() cache), matching this task's documented acceptable option: "compute at
engine boot/first-run and cache in the process." No request after the first ever re-hashes anything.

template_fingerprint() takes the same zero-request-cost approach: it renders one FIXED short
conversation through the caller's chat-template function (the engine's own /apply_template, when
available) once per process and hashes the result -- a cheap local HTTP round trip, not a file read.
"""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone

from clozn._io import atomic_write_json
from clozn.artifacts.contracts import sha256_file
from clozn.runs import identity_ext


_CACHE_DIR = os.path.join(os.path.expanduser("~/.clozn"), "cache")
_CACHE_PATH = os.path.join(_CACHE_DIR, "model_hashes.json")

_CHUNK_SIZE = 8 << 20   # 8 MiB -- matches clozn.artifacts.contracts.sha256_file's own default chunk size

# The FIXED canonical conversation template_fingerprint() renders. Module-level and never varied per
# call: the whole point is a comparable fingerprint across runs/models/processes. This does not need to
# be a REALISTIC prompt -- it only needs to exercise system+user role formatting the same way every time.
CANONICAL_CONVERSATION = (
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
)


def _load_cache() -> dict:
    try:
        with open(_CACHE_PATH, encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _save_cache(cache: dict) -> None:
    try:
        atomic_write_json(_CACHE_PATH, cache, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        pass


def _cache_hit(entry, st) -> str | None:
    """The cached sha256 for `entry` IF it matches `st` (an os.stat() result) on both size and mtime_ns,
    else None. Shared by model_sha256 (which falls through to a real hash on a miss) and
    cached_model_sha256 (which does not) so the two never drift on what counts as "still valid"."""
    if (
        isinstance(entry, dict)
        and entry.get("size") == st.st_size
        and entry.get("mtime_ns") == st.st_mtime_ns
        and isinstance(entry.get("sha256"), str)
        and entry["sha256"]
    ):
        return entry["sha256"]
    return None


def model_sha256(path) -> str | None:
    """SHA-256 hex digest of the model file at `path`, MANDATORY-cached by (absolute path, size,
    mtime_ns) at ~/.clozn/cache/model_hashes.json.

    A cache hit costs exactly one os.stat() call -- the file itself is never opened, so an unchanged
    multi-GB GGUF is hashed at most once per file version (any size or mtime change is treated as a new
    version and re-hashed). Reads the file in 8 MiB chunks on a miss (clozn.artifacts.contracts.
    sha256_file). Never raises: a missing file, a permission error, or a corrupt/unreadable cache file
    all fall through to None (or, for a corrupt cache, simply a fresh compute) rather than propagating.
    """
    if not path:
        return None
    try:
        resolved = os.path.abspath(os.fspath(path))
        st = os.stat(resolved)
    except Exception:
        return None

    cache = _load_cache()
    hit = _cache_hit(cache.get(resolved), st)
    if hit is not None:
        return hit

    try:
        digest = sha256_file(resolved, chunk_size=_CHUNK_SIZE)
    except Exception:
        return None

    cache[resolved] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
    _save_cache(cache)
    return digest


def cached_model_sha256(path) -> str | None:
    """This exact file version's sha256 IF it is already sitting in the cache -- otherwise None, WITHOUT
    ever reading (let alone hashing) the file itself. For callers that must never pay model_sha256()'s
    multi-GB-file hashing cost inline -- e.g. a synchronous web request handler, see
    clozn.models.inventory.inventory() -- where blocking on a full read of every local GGUF just to answer
    a listing request would turn a fast page load into a multi-minute hang.

    model_sha256() remains the only place that actually COMPUTES and populates this cache (at engine boot
    or first CLI use); a miss here honestly means "nobody has hashed this exact file version yet," not
    "this file can't be hashed" -- callers should render that as null/absent, never as an error. Never
    raises: a missing file or a permission error on the stat both yield None, same as model_sha256."""
    if not path:
        return None
    try:
        resolved = os.path.abspath(os.fspath(path))
        st = os.stat(resolved)
    except Exception:
        return None
    return _cache_hit(_load_cache().get(resolved), st)


def template_fingerprint(apply_template_fn) -> str | None:
    """First 16 hex chars of the SHA-256 of CANONICAL_CONVERSATION rendered through
    `apply_template_fn` (a `messages -> prompt string` callable, e.g. the engine client's
    apply_template). This fingerprints the chat template AND the tokenizer's rendering behavior
    without needing the raw template string -- two models/builds that render the fixed conversation
    identically get the same fingerprint; any difference in template, special tokens, or whitespace
    handling changes it. Never raises: a missing/erroring/non-string-returning callable yields None."""
    if apply_template_fn is None:
        return None
    try:
        rendered = apply_template_fn(list(CANONICAL_CONVERSATION))
    except Exception:
        return None
    if not isinstance(rendered, str) or not rendered:
        return None
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]


# Forward-compatible health-field lookup.  Current workers expose the build identity from
# engine_build_info() on /health; older workers may omit it, in which case this function still omits the
# field rather than inventing a value.  The tuple also accepts historical/managed aliases so a worker
# upgrade remains additive.
_ENGINE_BUILD_KEYS = ("engine_build", "build", "build_id", "build_commit")


def _engine_build(engine_health) -> str | None:
    if not isinstance(engine_health, dict):
        return None
    for key in _ENGINE_BUILD_KEYS:
        value = engine_health.get(key)
        if value:
            return str(value)
    return None


def _clozn_version() -> str | None:
    try:
        from clozn import __version__
        return str(__version__) or None
    except Exception:
        return None


def runtime_identity(*, model_path=None, model_sha256_hint=None, apply_template_fn=None,
                      engine_health=None, clozn_version=None, extra_context=None) -> dict:
    """Assemble the run record's `identity` block.

    `model_sha256_hint` lets a caller pass an already-known digest (e.g. the engine's own /health
    model_sha256, computed once at boot -- see the module docstring) so this never re-hashes a file the
    caller already identified; `model_sha256(model_path)` is only invoked as a fallback when no hint is
    given. Every field is OMITTED (never None-padded) when it cannot be honestly established -- absence
    must stay visible, not be faked as null. Never raises.

    `extra_context` is a dict of already-computed facts the CALLER knows and this function cannot
    measure for itself -- engine discovery source, backend, artifact sha256, adapter path, machine
    identity. It is merged into the context handed to clozn.runs.identity_providers.* and is the
    supported way for a facet provider to see a value that lives in the CLI/supervisor layer rather
    than in this module's own arguments. Without it a provider could only report what runtime_identity
    already knows, which makes most of the facets rule 2 asks for unreachable.

    The measured arguments above always WIN over same-named `extra_context` keys: a caller's hint must
    never be able to overwrite a value this module actually established. Unknown keys pass through
    untouched."""
    out: dict = {}
    try:
        if model_path:
            try:
                resolved = os.path.abspath(os.fspath(model_path))
            except Exception:
                resolved = str(model_path)
            out["model_path"] = resolved
            try:
                out["model_size_bytes"] = os.path.getsize(resolved)
            except Exception:
                pass

        sha = model_sha256_hint or (model_sha256(model_path) if model_path else None)
        if sha:
            out["model_sha256"] = str(sha)

        fingerprint = template_fingerprint(apply_template_fn)
        if fingerprint:
            out["template_fingerprint"] = fingerprint

        build = _engine_build(engine_health)
        if build:
            out["engine_build"] = build

        version = clozn_version or _clozn_version()
        if version:
            out["clozn_version"] = version

        # Feature-owned identity facets (engine artifact, adapter, machine/runtime, ...) contributed by
        # clozn/runs/identity_providers/*.py. Kept out of this function's body on purpose: those facets
        # belong to different features, and merging them here by hand would make this one function the
        # thing every feature branch has to edit. collect() never raises and omits a namespace it could
        # not establish, so the "omit, never null-pad" rule above holds inside `ext` too.
        context = dict(extra_context) if isinstance(extra_context, dict) else {}
        context.update({
            "model_path": model_path,
            "model_sha256": out.get("model_sha256"),
            "engine_health": engine_health,
            "apply_template_fn": apply_template_fn,
        })
        extensions = identity_ext.collect(context)
        if extensions:
            out["ext"] = extensions
    except Exception:
        pass
    out["captured_at"] = datetime.now(timezone.utc).isoformat()
    return out
