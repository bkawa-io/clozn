"""clozn/models/lockfile.py -- read and validate a checked-in `clozn.model-lock.v1` lockfile.

Feature 02 ("GitHub Action for model-change gating", notes/agent_roadmap/02-github-action-model-gate.md)
calls for a model lockfile convention so a gate's "run mode" can pin exactly which remote model artifacts
it resolves, by SHA-256, rather than trusting whatever a floating model name happens to point at.

SCOPE -- WHAT THIS MODULE DOES NOT DO
--------------------------------------
This module parses and validates a lockfile ALREADY ON DISK. It never opens a socket, never follows a
URL, and never resolves a pinned entry into a local model file. Downloading with SHA-256 verification,
HTTPS-redirect enforcement at connection time, and caching keyed on artifact/engine/suite/tokenizer/
template fingerprints are real, separate work (the spec's "run mode"); they were deliberately deferred
out of this slice because they need real network I/O to test honestly (a local HTTP test server or heavy
mocking), and because verify mode -- the free-CPU-runner path -- does not need them at all. See the
feature-02 plan for the full reasoning.

Route any future downloader that reads a lockfile produced here through `clozn.network_policy.
guarded_urlopen` (the process-wide urlopen wrapper with the local-only gate and audit ledger) rather than
calling `urllib`/`http.client` directly -- that is the established pattern for every other outbound call
in this codebase.
"""
from __future__ import annotations

import json

from clozn import schemas

LOCKFILE_SCHEMA = "clozn.model-lock.v1"


class LockfileError(ValueError):
    """The lockfile could not be read, parsed, or does not conform to `clozn.model-lock.v1`."""


def load_lockfile(path: str) -> dict:
    """Read, parse, and validate a model lockfile at `path`. Never touches the network.

    Raises LockfileError -- never lets json.JSONDecodeError/OSError/clozn.schemas.ValidationError/
    SchemaError escape directly -- on:
      * the file not existing or not being readable,
      * the file not being valid JSON,
      * the document not conforming to `clozn.model-lock.v1` (missing sha256, wrong schema_version, a
        `url` that does not match the schema's `^https://` pattern, etc: today the schema's own
        `pattern` already rejects every non-HTTPS `url` the fixtures exercise, so this is the check
        that actually fires in practice),
      * a `url` that is not HTTPS, checked again explicitly here as defense-in-depth independent of the
        schema file -- if `clozn.model-lock.v1.json`'s `pattern` is ever loosened by a future edit, an
        HTTPS-only lockfile reader must not silently start accepting `http://`.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            document = json.load(handle)
    except OSError as exc:
        raise LockfileError(f"could not read lockfile {path!r}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise LockfileError(f"lockfile {path!r} is not valid JSON: {exc}") from exc

    try:
        schemas.validate(document, LOCKFILE_SCHEMA)
    except (schemas.ValidationError, schemas.SchemaError) as exc:
        raise LockfileError(
            f"lockfile {path!r} does not conform to {LOCKFILE_SCHEMA}: {exc}") from exc

    for role, pinned in (document.get("models") or {}).items():
        url = pinned.get("url") if isinstance(pinned, dict) else None
        if not isinstance(url, str) or not url.lower().startswith("https://"):
            raise LockfileError(
                f"lockfile {path!r} model {role!r}: url must be HTTPS, got {url!r}")

    return document


def model_roles(document: dict) -> list[str]:
    """The pinned model role names (e.g. 'baseline', 'candidate') in a validated lockfile, sorted."""
    return sorted((document.get("models") or {}).keys())


def pinned_model(document: dict, role: str) -> dict:
    """The pinned artifact dict for `role`. Raises LockfileError if that role is not in the lockfile --
    never returns an empty/default dict for a role that was never pinned."""
    models = document.get("models") or {}
    if role not in models:
        available = ", ".join(sorted(models)) or "none"
        raise LockfileError(f"lockfile has no model pinned for role {role!r} (available: {available})")
    return models[role]


__all__ = ["LOCKFILE_SCHEMA", "LockfileError", "load_lockfile", "model_roles", "pinned_model"]
