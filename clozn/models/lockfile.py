"""clozn/models/lockfile.py -- read and validate a checked-in `clozn.model-lock.v1` lockfile.

Feature 02 ("GitHub Action for model-change gating", notes/agent_roadmap/02-github-action-model-gate.md)
calls for a model lockfile convention so a gate's "run mode" can pin exactly which remote model artifacts
it resolves, by SHA-256, rather than trusting whatever a floating model name happens to point at.

SCOPE -- WHAT THIS MODULE DOES NOT DO
--------------------------------------
This module parses and validates a lockfile ALREADY ON DISK. It never opens a socket, follows a URL, or
resolves a pin. The separate, explicitly networked resolver is ``clozn.models.fetch``; keeping the
boundary between them is what guarantees that ``model-lock verify`` stays safe on a network-free CI
runner.
"""
from __future__ import annotations

import json
from copy import deepcopy
from urllib.parse import urlsplit

from clozn import schemas

LOCKFILE_SCHEMA = "clozn.model-lock.v1"


class LockfileError(ValueError):
    """The lockfile could not be read, parsed, or does not conform to `clozn.model-lock.v1`."""


def _loopback_http(url: object) -> bool:
    if not isinstance(url, str):
        return False
    try:
        parsed = urlsplit(url)
    except ValueError:
        return False
    return (
        parsed.scheme.casefold() == "http"
        and (parsed.hostname or "").rstrip(".").casefold() in {"127.0.0.1", "localhost", "::1"}
    )


def load_lockfile(path: str, *, allow_loopback_http: bool = False) -> dict:
    """Read, parse, and validate a model lockfile at `path`. Never touches the network.

    ``allow_loopback_http`` is an internal test seam for ``clozn.models.fetch``.
    It accepts only exact loopback fixture hosts; the default and the public
    ``model-lock verify`` command remain strictly HTTPS-only.

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

    # The released v1 schema correctly remains HTTPS-only.  The networked fetcher's in-process test seam
    # may explicitly admit a loopback HTTP fixture; validate an HTTPS-normalized copy in that one case,
    # then apply the transport check to the original document below. Normal callers (especially
    # `model-lock verify`) never set this flag and therefore validate the unmodified document.
    schema_document = document
    if allow_loopback_http and isinstance(document, dict):
        schema_document = deepcopy(document)
        for pinned in (schema_document.get("models") or {}).values():
            if isinstance(pinned, dict) and _loopback_http(pinned.get("url")):
                pinned["url"] = "https://" + pinned["url"][len("http://"):]
    try:
        schemas.validate(schema_document, LOCKFILE_SCHEMA)
    except (schemas.ValidationError, schemas.SchemaError) as exc:
        raise LockfileError(
            f"lockfile {path!r} does not conform to {LOCKFILE_SCHEMA}: {exc}") from exc

    for role, pinned in (document.get("models") or {}).items():
        url = pinned.get("url") if isinstance(pinned, dict) else None
        if not isinstance(url, str) or (
                not url.lower().startswith("https://")
                and not (allow_loopback_http and _loopback_http(url))):
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
