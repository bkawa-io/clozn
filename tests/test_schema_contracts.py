"""Fixture-driven contract tests for every schema in clozn/schemas/defs/.

WHY THIS FILE NEVER NEEDS EDITING
---------------------------------
Both halves of this suite discover their work by walking the filesystem, so adding an artifact type is
a pure addition -- drop a schema in clozn/schemas/defs/ and fixtures in tests/fixtures/schemas/<name>/
and the tests below pick them up. Nobody editing a shared test module means parallel work on different
artifact types cannot collide here.

    tests/fixtures/schemas/<schema_version>/valid__<case>.json     must validate
    tests/fixtures/schemas/<schema_version>/invalid__<case>.json   must raise ValidationError

An `invalid__` fixture that passes validation is a REAL failure, not a nuisance: it means the schema is
looser than its author believed. Name the case for the defect it encodes (invalid__missing_run_id.json,
invalid__sha256_too_short.json) so a failure reads as a sentence.
"""
from __future__ import annotations

import json
import os

import pytest

from clozn import schemas

FIXTURE_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "schemas")


def _fixtures(kind: str) -> list[tuple[str, str]]:
    """Every (schema_name, fixture_path) pair for fixtures whose basename starts with `kind`."""
    found: list[tuple[str, str]] = []
    for name in schemas.list_schemas():
        directory = os.path.join(FIXTURE_ROOT, name)
        if not os.path.isdir(directory):
            continue
        for entry in sorted(os.listdir(directory)):
            if entry.startswith(kind) and entry.endswith(".json"):
                found.append((name, os.path.join(directory, entry)))
    return found


def _ident(pair: tuple[str, str]) -> str:
    return f"{pair[0]}/{os.path.basename(pair[1])}"


def test_at_least_one_schema_is_registered():
    """A guard against this suite silently passing because defs/ went missing or stopped shipping."""
    assert schemas.list_schemas(), (
        f"no schemas found in {schemas.DEFS_DIR} -- either none are registered yet or the package data "
        f"is not shipping (see setup.py package_data)")


@pytest.mark.parametrize("name", schemas.list_schemas())
def test_schema_loads_and_uses_only_supported_keywords(name):
    """load() runs check_keywords(), so this fails loudly on a keyword the stdlib validator ignores."""
    schema = schemas.load(name)
    assert isinstance(schema, dict) and schema, f"{name} parsed to an empty schema"


@pytest.mark.parametrize("name", schemas.list_schemas())
def test_schema_id_matches_its_filename(name):
    """The name/filename/`$id` 1:1 mapping is what lets validate() infer a schema from an untrusted
    document's schema_version with no registry to keep in sync. A mismatch breaks that inference."""
    declared = schemas.load(name).get("$id")
    if declared is not None:
        assert declared == name, f"{name}.json declares $id {declared!r}; they must match"


@pytest.mark.parametrize("pair", _fixtures("valid__"), ids=_ident)
def test_valid_fixtures_validate(pair):
    name, path = pair
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    schemas.validate(document, name)


@pytest.mark.parametrize("pair", _fixtures("invalid__"), ids=_ident)
def test_invalid_fixtures_are_rejected(pair):
    name, path = pair
    with open(path, encoding="utf-8") as handle:
        document = json.load(handle)
    with pytest.raises(schemas.ValidationError):
        schemas.validate(document, name)


def test_schema_version_is_inferred_from_the_document():
    """validate() with no explicit name must find the schema via the document's own schema_version."""
    name = schemas.list_schemas()[0]
    with pytest.raises(schemas.ValidationError, match="schema_version"):
        schemas.validate({"not_a_schema_version": name})


def test_unknown_schema_raises_schema_error_not_a_silent_pass():
    with pytest.raises(schemas.SchemaError, match="no schema named"):
        schemas.validate({"schema_version": "clozn.definitely-not-real.v1"})


def test_forward_compatibility_unknown_fields_are_tolerated_by_default():
    """Roadmap rule: 'Unknown fields may be retained for forward compatibility.' An older clozn reading a
    newer artifact must not reject it for carrying a field it has not learned about yet."""
    schemas.validate({
        "schema_version": "clozn.run-identity.v1",
        "captured_at": "2026-07-27T00:00:00+00:00",
        "a_field_from_a_future_version": {"nested": True},
    }, "clozn.run-identity.v1")
