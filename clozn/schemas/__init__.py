"""Versioned schemas for clozn's stored artifacts -- the one place a new artifact type gets defined.

Roadmap rule 7 ("Schema-first changes") requires that every new stored artifact ship a versioned schema,
a validator, a migration policy, and fixtures BEFORE any UI work. This package is that contract, and it
is deliberately shaped so that adding an artifact type is a pure ADDITION -- no shared Python file needs
editing, so parallel work on different artifacts cannot conflict:

    1. Write clozn/schemas/defs/<schema_version>.json   (e.g. clozn.context-receipt.v1.json)
    2. Drop fixtures in tests/fixtures/schemas/<schema_version>/valid__*.json and invalid__*.json
    3. Call clozn.schemas.validate(doc) wherever you read or write the artifact

tests/test_schema_contracts.py discovers both directories by walking the filesystem, so step 2 gives you
round-trip tests without touching a shared test module.

NAMING
------
A schema's name IS the `schema_version` string its documents carry -- `clozn.context-receipt.v1` lives in
`defs/clozn.context-receipt.v1.json`. That 1:1 mapping is what lets validate() take an untrusted document
and find the right schema with no registry to keep in sync and no dispatch table to edit.

VERSIONING AND MIGRATION
------------------------
Schemas are immutable once released. A backward-compatible change (adding an optional property, widening
an enum) edits the existing `.v1` file. A breaking change (new required field, removed property, narrowed
type) adds `.v2` alongside it and leaves `.v1` in place -- readers keep validating old artifacts on disk.
Never renumber, never delete a version that has been written to a user's disk. A `.vN` file gaining a
required property that older artifacts lack is the one mistake that silently invalidates existing local
data; add it as optional, or bump the version.
"""
from __future__ import annotations

import json
import os

from clozn.schemas._validator import SchemaError, ValidationError, check_keywords
from clozn.schemas._validator import validate as _validate_against

__all__ = ["SchemaError", "ValidationError", "load", "validate", "list_schemas", "schema_path",
           "DEFS_DIR"]

DEFS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defs")

# name -> parsed schema. Populated by load(); each entry has already passed check_keywords(), so the
# unsupported-keyword scan is paid once per process per schema rather than on every validate() call.
_CACHE: dict[str, dict] = {}


def schema_path(name: str) -> str:
    """Absolute path to `name`'s schema file. Does not check that it exists."""
    return os.path.join(DEFS_DIR, f"{name}.json")


def list_schemas() -> list[str]:
    """Every schema name available in defs/, sorted. Empty list if defs/ is missing entirely."""
    try:
        return sorted(f[:-5] for f in os.listdir(DEFS_DIR) if f.endswith(".json"))
    except OSError:
        return []


def load(name: str) -> dict:
    """The parsed schema registered as `name`, cached per process.

    Raises SchemaError if the file is missing, unparseable, or uses a keyword the stdlib validator does
    not implement -- all three are authoring bugs that should surface loudly and immediately, never be
    swallowed into a permissive "no schema, so everything passes".
    """
    cached = _CACHE.get(name)
    if cached is not None:
        return cached

    path = schema_path(name)
    try:
        with open(path, encoding="utf-8") as handle:
            schema = json.load(handle)
    except FileNotFoundError:
        available = ", ".join(list_schemas()) or "none"
        raise SchemaError(
            f"no schema named {name!r} in {DEFS_DIR} (available: {available}). A stored artifact must "
            f"have a schema before it is written -- see clozn/schemas/__init__.py."
        ) from None
    except (OSError, json.JSONDecodeError) as exc:
        raise SchemaError(f"schema {name!r} at {path} could not be read: {exc}") from None

    if not isinstance(schema, dict):
        raise SchemaError(f"schema {name!r} must be a JSON object, got {type(schema).__name__}")
    check_keywords(schema, name)
    _CACHE[name] = schema
    return schema


def validate(document, name: str | None = None) -> None:
    """Validate `document` against its schema. Returns None on success, raises on failure.

    `name` defaults to the document's own `schema_version` field, which is why every clozn artifact
    carries one: a reader can hand an untrusted blob straight to this function without first knowing
    what it is.

    Raises ValidationError if the document does not conform, or SchemaError if the named schema is
    missing or malformed. It never returns a bool -- a validator whose failure mode is a falsy return
    value is one `if` away from being ignored.
    """
    if name is None:
        if not isinstance(document, dict):
            raise ValidationError(
                f"cannot infer a schema: expected an object carrying 'schema_version', got "
                f"{type(document).__name__}")
        name = document.get("schema_version")
        if not isinstance(name, str) or not name:
            raise ValidationError(
                "cannot infer a schema: document has no 'schema_version' string. Every stored clozn "
                "artifact must carry one (see clozn/schemas/__init__.py).")
    _validate_against(document, load(name))
