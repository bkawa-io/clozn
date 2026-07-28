"""A stdlib-only validator for the JSON Schema subset clozn's stored artifacts actually use.

WHY NOT jsonschema
------------------
pyproject.toml declares `dependencies = []` on purpose: the product CLI/gateway supervisor never imports
Torch, transformers, or anything else outside the stdlib, and the `product-minimal` CI lane asserts
`clozn.server.app` imports with none of them installed (see docs/RUNTIME_SPLIT.md). A schema validator
that every stored artifact passes through sits squarely on that path, so it cannot be the thing that
introduces clozn's first hard install-time dependency. This module is the alternative: ~200 lines
covering the keywords the artifact schemas in defs/ actually need, and raising a clear error on any
keyword it does NOT implement rather than silently ignoring it.

THAT LAST PART IS THE WHOLE POINT
---------------------------------
A validator that skips keywords it doesn't understand is worse than no validator: it reports "valid"
for a document it never actually checked. `_check_keywords()` walks every schema at load time and
raises SchemaError on an unsupported keyword, so an agent who writes `"multipleOf": 4` finds out when
the schema loads, not when a bad artifact sails through in production. Adding a keyword here is cheap
and expected -- silently tolerating one is not.

SUPPORTED KEYWORDS
------------------
    type (single or list)     required            properties
    additionalProperties      items               enum
    const                     pattern             format (annotation only, never enforced)
    minimum / maximum         minLength / maxLength   minItems / maxItems
    anyOf / oneOf             $ref (local "#/$defs/NAME" only)

`additionalProperties` DEFAULTS TO PERMISSIVE, deliberately: roadmap rule 7 wants versioned schemas, and
feature 01 states "Unknown fields may be retained for forward compatibility, but required fields must
never be inferred." An older clozn reading a newer artifact must not reject it for carrying a field it
has not learned about yet. Set `"additionalProperties": false` explicitly on a schema that must stay
closed.
"""
from __future__ import annotations

import re

_SUPPORTED = frozenset({
    "$schema", "$id", "$defs", "$ref", "title", "description", "examples", "default", "deprecated",
    "type", "required", "properties", "additionalProperties", "items", "enum", "const", "pattern",
    "format", "minimum", "maximum", "minLength", "maxLength", "minItems", "maxItems", "anyOf", "oneOf",
})

# JSON Schema type name -> the Python types json.load actually produces for it. bool is excluded from
# "integer"/"number" explicitly: in Python `isinstance(True, int)` is True, so without this a schema
# demanding an integer token count would happily accept `true`.
_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


class SchemaError(Exception):
    """The SCHEMA is malformed or uses a keyword this validator does not implement.

    Distinct from ValidationError on purpose: this one means a developer wrote a bad schema (a bug to
    fix at authoring time), not that a document failed to conform (a runtime condition to handle).
    """


class ValidationError(Exception):
    """A document did not conform to its schema. `path` locates the offending node."""

    def __init__(self, message: str, path: str = ""):
        self.path = path or "<root>"
        self.message = message
        super().__init__(f"{self.path}: {message}")


def _type_ok(value, name: str) -> bool:
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    expected = _TYPES.get(name)
    if expected is None:
        raise SchemaError(f"unknown type name {name!r}")
    if expected is bool:
        return isinstance(value, bool)
    return isinstance(value, expected)


def check_keywords(schema, path: str = "<root>") -> None:
    """Raise SchemaError if `schema` uses any keyword this validator does not implement.

    Called once per schema at load time (see clozn.schemas.load) rather than per document, so the cost
    is paid on the first use of a schema and never on the hot path.
    """
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise SchemaError(f"{path}: schema must be an object or boolean, got {type(schema).__name__}")
    for key in schema:
        if key not in _SUPPORTED:
            raise SchemaError(
                f"{path}: unsupported keyword {key!r}. This validator is a deliberate subset -- add "
                f"support in clozn/schemas/_validator.py rather than relying on it being ignored."
            )
    for key in ("properties", "$defs"):
        for name, sub in (schema.get(key) or {}).items():
            check_keywords(sub, f"{path}.{key}.{name}")
    if "items" in schema:
        check_keywords(schema["items"], f"{path}.items")
    if isinstance(schema.get("additionalProperties"), dict):
        check_keywords(schema["additionalProperties"], f"{path}.additionalProperties")
    for key in ("anyOf", "oneOf"):
        for index, sub in enumerate(schema.get(key) or []):
            check_keywords(sub, f"{path}.{key}[{index}]")


def _resolve(schema: dict, root: dict, path: str) -> dict:
    """Follow a local `$ref`. Only "#/$defs/NAME" is supported -- remote refs would mean network I/O
    during validation, which a local-first artifact validator must never do."""
    ref = schema["$ref"]
    if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
        raise SchemaError(f"{path}: only local '#/$defs/NAME' refs are supported, got {ref!r}")
    name = ref[len("#/$defs/"):]
    target = (root.get("$defs") or {}).get(name)
    if not isinstance(target, dict):
        raise SchemaError(f"{path}: $ref target {ref!r} not found in $defs")
    return target


def validate(document, schema: dict, root: dict | None = None, path: str = "") -> None:
    """Raise ValidationError if `document` does not conform to `schema`. Returns None on success.

    `root` carries the top-level schema so `$ref` can resolve against its `$defs`; callers normally omit
    it. `path` is the dotted location used in error messages.
    """
    root = schema if root is None else root
    if isinstance(schema, bool):
        if not schema:
            raise ValidationError("schema `false` rejects every value", path)
        return
    if "$ref" in schema:
        schema = _resolve(schema, root, path or "<root>")

    if "type" in schema:
        names = schema["type"]
        names = [names] if isinstance(names, str) else list(names)
        if not any(_type_ok(document, n) for n in names):
            got = "null" if document is None else type(document).__name__
            raise ValidationError(f"expected type {'|'.join(names)}, got {got}", path)

    if "const" in schema and document != schema["const"]:
        raise ValidationError(f"expected the constant {schema['const']!r}, got {document!r}", path)

    if "enum" in schema and document not in schema["enum"]:
        raise ValidationError(
            f"{document!r} is not one of the {len(schema['enum'])} permitted values "
            f"({', '.join(repr(v) for v in schema['enum'][:8])}"
            f"{', ...' if len(schema['enum']) > 8 else ''})", path)

    if isinstance(document, str):
        pattern = schema.get("pattern")
        if pattern is not None and not re.search(pattern, document):
            raise ValidationError(f"{document!r} does not match pattern {pattern!r}", path)
        if "minLength" in schema and len(document) < schema["minLength"]:
            raise ValidationError(f"string shorter than minLength {schema['minLength']}", path)
        if "maxLength" in schema and len(document) > schema["maxLength"]:
            raise ValidationError(f"string longer than maxLength {schema['maxLength']}", path)

    if isinstance(document, (int, float)) and not isinstance(document, bool):
        if "minimum" in schema and document < schema["minimum"]:
            raise ValidationError(f"{document} is below the minimum {schema['minimum']}", path)
        if "maximum" in schema and document > schema["maximum"]:
            raise ValidationError(f"{document} is above the maximum {schema['maximum']}", path)

    if isinstance(document, list):
        if "minItems" in schema and len(document) < schema["minItems"]:
            raise ValidationError(f"array has {len(document)} items, minItems is {schema['minItems']}", path)
        if "maxItems" in schema and len(document) > schema["maxItems"]:
            raise ValidationError(f"array has {len(document)} items, maxItems is {schema['maxItems']}", path)
        if "items" in schema:
            for index, item in enumerate(document):
                validate(item, schema["items"], root, f"{path}[{index}]")

    if isinstance(document, dict):
        for name in schema.get("required", ()):
            if name not in document:
                raise ValidationError(f"missing required property {name!r}", path)
        properties = schema.get("properties") or {}
        for name, sub in properties.items():
            if name in document:
                validate(document[name], sub, root, f"{path}.{name}" if path else name)
        extra = schema.get("additionalProperties", True)
        if extra is not True:
            unknown = sorted(set(document) - set(properties))
            if unknown and extra is False:
                raise ValidationError(
                    f"unknown propert{'y' if len(unknown) == 1 else 'ies'} "
                    f"{', '.join(repr(u) for u in unknown)} (schema is closed)", path)
            if isinstance(extra, dict):
                for name in unknown:
                    validate(document[name], extra, root, f"{path}.{name}" if path else name)

    for key, require_exactly_one in (("anyOf", False), ("oneOf", True)):
        options = schema.get(key)
        if not options:
            continue
        failures = []
        passed = 0
        for index, option in enumerate(options):
            try:
                validate(document, option, root, path)
                passed += 1
            except ValidationError as exc:
                failures.append(f"[{index}] {exc.message}")
        if passed == 0:
            raise ValidationError(f"matched none of the {len(options)} {key} branches: "
                                  f"{'; '.join(failures)}", path)
        if require_exactly_one and passed > 1:
            raise ValidationError(f"matched {passed} oneOf branches, expected exactly 1", path)
