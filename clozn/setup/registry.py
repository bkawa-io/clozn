"""Read/write ~/.clozn/engines/registry.json (clozn.engine-registry.v1) -- the single source of truth
for which managed engines are installed and which is active/previous.

Self-healing like clozn/cli/engine_process.py's daemons.json registry: an `installed` entry whose
on-disk entrypoint no longer exists is dropped rather than trusted (see prune_missing()), and a corrupt
or missing file reads back as an empty-but-valid registry rather than raising -- a hand-edited or
half-written file does not deserve to brick every future `clozn setup`/`doctor` call.
"""
from __future__ import annotations

import json
import os

from clozn import schemas
from clozn._io import atomic_write_json
from clozn.setup.errors import RegistryError

SCHEMA_NAME = "clozn.engine-registry.v1"


def engines_dir(home: str) -> str:
    return os.path.join(home, "engines")


def registry_path(home: str) -> str:
    return os.path.join(engines_dir(home), "registry.json")


def lock_path(home: str) -> str:
    return os.path.join(engines_dir(home), ".lock")


def _empty() -> dict:
    return {"schema_version": SCHEMA_NAME}


def load(home: str) -> dict:
    """The registry document, or an empty-but-valid one if none exists yet or the file is unreadable/
    corrupt. Never raises."""
    try:
        with open(registry_path(home), encoding="utf-8") as handle:
            document = json.load(handle)
    except Exception:
        return _empty()
    return document if isinstance(document, dict) else _empty()


def save(home: str, document: dict) -> None:
    """Validate-then-atomically-write. Raises RegistryError on a document that fails
    clozn.engine-registry.v1 -- every caller in this package builds `document` through record_install()/
    rollback() below, so a validation failure here means one of THOSE has a bug, not that the user did
    anything wrong."""
    document = dict(document)
    document["schema_version"] = SCHEMA_NAME
    try:
        schemas.validate(document, SCHEMA_NAME)
    except (schemas.ValidationError, schemas.SchemaError) as error:
        raise RegistryError(f"refusing to write an invalid engine registry: {error}") from None
    atomic_write_json(registry_path(home), document, indent=2, sort_keys=True)


def prune_missing(document: dict) -> dict:
    """A NEW dict with every `installed` entry whose entrypoint file no longer exists dropped, and
    `active`/`previous` cleared if they named a now-missing key. Pure -- does not touch disk; callers
    decide whether/when to persist the result (clozn/setup/install.read_status() does, matching
    engine_process.py's `_find_warm` prune-then-write-if-dirty pattern)."""
    installed = dict(document.get("installed") or {})
    alive = {
        key: record for key, record in installed.items()
        if isinstance(record, dict) and os.path.isfile(str(record.get("entrypoint") or ""))
    }
    out = dict(document)
    out["installed"] = alive
    if out.get("active") not in alive:
        out.pop("active", None)
    if out.get("previous") not in alive:
        out.pop("previous", None)
    return out


def record_install(document: dict, key: str, record: dict, *, make_active: bool) -> dict:
    """A NEW dict with `record` added under `key`, and -- when `make_active` -- `active` moved to `key`
    with whatever was previously active becoming `previous` (unless it IS `key`, e.g. a `--force`
    reinstall of the currently active version, which must not overwrite `previous` with itself). Pure --
    does not touch disk."""
    out = dict(document)
    installed = dict(out.get("installed") or {})
    installed[key] = dict(record)
    out["installed"] = installed
    if make_active:
        previous_active = out.get("active")
        if previous_active and previous_active != key:
            out["previous"] = previous_active
        out["active"] = key
    return out


def rollback(document: dict) -> dict:
    """A NEW dict with `active` and `previous` swapped. Raises RegistryError if there is no `previous`,
    or if `previous` no longer names an installed entry (its directory was removed out of band)."""
    previous = document.get("previous")
    if not previous:
        raise RegistryError("no previous engine recorded; nothing to roll back to")
    installed = document.get("installed") or {}
    if previous not in installed:
        raise RegistryError(
            f"the previous engine ({previous}) is no longer in the registry; it may have been removed")
    out = dict(document)
    out["previous"] = out.get("active")
    out["active"] = previous
    if not out.get("previous"):
        out.pop("previous", None)
    return out
