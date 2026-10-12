"""Fail-closed structured I/O primitives for the OpenAI Chat Completions shim.

This module intentionally does not route a request or call a model.  It owns the pure
Phase-2.8 contract that those integration points can share:

* validate the deliberately small tools / structured-output request vocabulary;
* normalize OpenAI tool history and lower it to the engine's role+text template seam;
* qualify an exact loaded model + rendered-template identity from an explicit registry;
* render a versioned prompt protocol and strictly parse its result; and
* turn the parsed result into ordinary OpenAI response message/delta shapes.

Nothing is qualified by default.  A configured registry is an allow-list, not a hint:
both ``model_sha256`` and ``template_fingerprint`` must match the loaded substrate.
The request's user-controlled ``model`` label is never consulted.

The implementation is standard-library-only so Clozn's minimal product install remains
dependency-free.  Consequently JSON Schema support is an explicit, recursively validated
subset rather than a claim to implement all of draft 2020-12.  Unknown keywords fail.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
import re
import secrets
from collections.abc import Mapping
from typing import Any


QUALIFICATIONS_ENV = "CLOZN_STRUCTURED_IO_QUALIFICATIONS"
QUALIFICATION_SCHEMA = "clozn.structured_io.qualifications.v2"
EVIDENCE_SCHEMA = "clozn.structured_io.v1"
PYTHON_ENVELOPE_RENDERER_ID = "clozn.structured_io.renderer.v1"
PYTHON_ENVELOPE_PARSER_ID = "clozn.structured_io.parser.v1"

# Qualification-bearing native execution contracts.  These IDs describe the
# private atomic worker pipeline, not the model-free Python envelope above.  Any
# behavior-bearing change to the corresponding worker stage must mint a new ID
# so existing exact-model evidence fails closed.
NATIVE_EXECUTOR_ID = "clozn.chat_io.atomic_executor.v1"
NATIVE_RENDERER_ID = "clozn.chat_io.llama_common.renderer.v1"
NATIVE_GRAMMAR_ID = "clozn.chat_io.ar_grammar.v1"
NATIVE_PARSER_ID = "clozn.chat_io.llama_common.parser.v1"
NATIVE_MESSAGE_VALIDATOR_ID = "clozn.structured_io.native_message_validator.v1"
JSON_SCHEMA_SUBSET_ID = "clozn.structured_io.json_schema_subset.v1"
QUALIFICATION_EVIDENCE_SCHEMA = "clozn.chat_io.qualification_evidence.v2"
QUALIFICATION_SUITE_ID = "clozn.chat_io.qualification_suite.v1"

NATIVE_WORKER_PIPELINE = {
    "executor_id": NATIVE_EXECUTOR_ID,
    "renderer_id": NATIVE_RENDERER_ID,
    "grammar_id": NATIVE_GRAMMAR_ID,
    "parser_id": NATIVE_PARSER_ID,
}
NATIVE_QUALIFICATION_PIPELINE = {
    **NATIVE_WORKER_PIPELINE,
    "validator_id": NATIVE_MESSAGE_VALIDATOR_ID,
}

MAX_TOOLS = 32
MAX_SCHEMA_DEPTH = 8
MAX_SCHEMA_PROPERTIES = 128
MAX_SCHEMA_BYTES = 64 * 1024
MAX_OUTPUT_BYTES = 256 * 1024
MAX_OUTPUT_DEPTH = 32
MAX_OUTPUT_NODES = 10_000

_NAME_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CALL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_FINGERPRINT_RE = re.compile(r"^[0-9a-fA-F]{16,64}$")
_FENCE_RE = re.compile(
    r"\A[ \t\r\n]*```json[ \t]*\r?\n(?P<body>.*?)\r?\n```[ \t]*[\r\n \t]*\Z",
    re.DOTALL,
)

_JSON_TYPES = frozenset({"object", "array", "string", "number", "integer", "boolean", "null"})
_COMMON_SCHEMA_KEYS = frozenset({"type", "description", "enum"})
_TYPE_SCHEMA_KEYS = {
    "object": frozenset({"properties", "required", "additionalProperties"}),
    "array": frozenset({"items", "minItems", "maxItems"}),
    "string": frozenset({"minLength", "maxLength"}),
    "number": frozenset({"minimum", "maximum"}),
    "integer": frozenset({"minimum", "maximum"}),
    "boolean": frozenset(),
    "null": frozenset(),
}


class StructuredIOError(ValueError):
    """A typed, serializable contract failure.

    ``evidence`` contains only JSON-compatible diagnostic data and is deliberately
    separate from the human-readable message.  Route glue can put :meth:`as_error`
    inside its usual ``{"error": ...}`` envelope without losing the typed cause.
    """

    def __init__(self, message: str, *, code: str, param: str | None,
                 evidence: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.code = str(code)
        self.param = param
        self.evidence = dict(evidence or {})

    def as_error(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "message": str(self),
            "type": "invalid_request_error" if self.param else "model_output_error",
            "param": self.param,
            "code": self.code,
        }
        if self.evidence:
            out["evidence"] = self.evidence
        return out


def _fail(message: str, *, code: str = "invalid_parameter", param: str | None,
          evidence: Mapping[str, Any] | None = None) -> None:
    raise StructuredIOError(message, code=code, param=param, evidence=evidence)


def _exact_keys(value: Mapping[str, Any], allowed: set[str] | frozenset[str], *, param: str) -> None:
    extra = sorted(set(value) - set(allowed))
    if extra:
        _fail(f"unsupported field '{extra[0]}'", code="unsupported_parameter",
              param=f"{param}.{extra[0]}")


def _json_size(value: Any, *, param: str) -> int:
    try:
        return len(json.dumps(value, ensure_ascii=False, allow_nan=False,
                              separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        _fail(f"{param} must contain only finite JSON values: {exc}", param=param)
    raise AssertionError("unreachable")


def _json_type_matches(value: Any, kind: str) -> bool:
    if kind == "object":
        return isinstance(value, dict)
    if kind == "array":
        return isinstance(value, list)
    if kind == "string":
        return isinstance(value, str)
    if kind == "number":
        return (isinstance(value, int) and not isinstance(value, bool)) or (
            isinstance(value, float) and math.isfinite(value)
        )
    if kind == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    return False


def _nonnegative_int(value: Any, *, param: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        _fail(f"{param} must be a non-negative integer", param=param)
    return int(value)


def normalize_json_schema(schema: Any, *, param: str = "response_format.json_schema.schema",
                          root_object: bool = True) -> dict[str, Any]:
    """Validate and normalize Clozn's strict JSON Schema subset.

    Every schema node requires one scalar ``type``.  Composition, references, regexes,
    formats, defaults, and conditional schemas are deliberately outside v1.  Object
    schemas default ``additionalProperties`` to ``false`` and may not opt back into
    undeclared keys.  This lets validation be strict without silently interpreting an
    omitted keyword as broader behavior.
    """
    if _json_size(schema, param=param) > MAX_SCHEMA_BYTES:
        _fail(f"{param} exceeds {MAX_SCHEMA_BYTES} encoded bytes", param=param)
    property_count = [0]

    def visit(node: Any, node_param: str, depth: int) -> dict[str, Any]:
        if depth > MAX_SCHEMA_DEPTH:
            _fail(f"JSON schema exceeds maximum depth {MAX_SCHEMA_DEPTH}", param=node_param)
        if not isinstance(node, Mapping):
            _fail("each JSON schema node must be an object", param=node_param)
        kind = node.get("type")
        if not isinstance(kind, str) or kind not in _JSON_TYPES:
            _fail(f"schema type must be one of {sorted(_JSON_TYPES)}", param=f"{node_param}.type")
        allowed = _COMMON_SCHEMA_KEYS | _TYPE_SCHEMA_KEYS[kind]
        extra = sorted(set(node) - allowed)
        if extra:
            _fail(
                f"unsupported JSON schema keyword '{extra[0]}'",
                code="unsupported_schema_keyword",
                param=f"{node_param}.{extra[0]}",
                evidence={"supported_keywords": sorted(allowed), "schema_type": kind},
            )
        out: dict[str, Any] = {"type": kind}
        if "description" in node:
            if not isinstance(node["description"], str):
                _fail("schema description must be a string", param=f"{node_param}.description")
            out["description"] = node["description"]
        if "enum" in node:
            enum = node["enum"]
            if not isinstance(enum, list) or not enum:
                _fail("schema enum must be a non-empty list", param=f"{node_param}.enum")
            seen: set[str] = set()
            for index, item in enumerate(enum):
                if not _json_type_matches(item, kind):
                    _fail(f"enum value does not match schema type {kind!r}",
                          param=f"{node_param}.enum[{index}]")
                encoded = json.dumps(item, ensure_ascii=False, allow_nan=False,
                                     sort_keys=True, separators=(",", ":"))
                if encoded in seen:
                    _fail("schema enum values must be unique", param=f"{node_param}.enum[{index}]")
                seen.add(encoded)
            out["enum"] = list(enum)

        if kind == "object":
            properties = node.get("properties", {})
            if not isinstance(properties, Mapping):
                _fail("object schema properties must be an object", param=f"{node_param}.properties")
            normalized_properties = {}
            for name, child in properties.items():
                if not isinstance(name, str) or not name:
                    _fail("property names must be non-empty strings", param=f"{node_param}.properties")
                property_count[0] += 1
                if property_count[0] > MAX_SCHEMA_PROPERTIES:
                    _fail(f"JSON schema exceeds {MAX_SCHEMA_PROPERTIES} total properties",
                          param=f"{node_param}.properties")
                normalized_properties[name] = visit(child, f"{node_param}.properties.{name}", depth + 1)
            required = node.get("required", [])
            if not isinstance(required, list) or any(not isinstance(x, str) for x in required):
                _fail("object schema required must be a list of property names",
                      param=f"{node_param}.required")
            if len(set(required)) != len(required):
                _fail("object schema required names must be unique", param=f"{node_param}.required")
            missing = [name for name in required if name not in normalized_properties]
            if missing:
                _fail(f"required property {missing[0]!r} is not declared in properties",
                      param=f"{node_param}.required")
            additional = node.get("additionalProperties", False)
            if additional is not False:
                _fail("strict v1 schemas require additionalProperties:false",
                      param=f"{node_param}.additionalProperties")
            out.update(properties=normalized_properties, required=list(required), additionalProperties=False)

        elif kind == "array":
            if "items" not in node:
                _fail("array schemas require an items schema", param=f"{node_param}.items")
            out["items"] = visit(node["items"], f"{node_param}.items", depth + 1)
            minimum = _nonnegative_int(node.get("minItems", 0), param=f"{node_param}.minItems")
            out["minItems"] = minimum
            if "maxItems" in node:
                maximum = _nonnegative_int(node["maxItems"], param=f"{node_param}.maxItems")
                if maximum < minimum:
                    _fail("maxItems must be at least minItems", param=f"{node_param}.maxItems")
                out["maxItems"] = maximum

        elif kind == "string":
            minimum = _nonnegative_int(node.get("minLength", 0), param=f"{node_param}.minLength")
            out["minLength"] = minimum
            if "maxLength" in node:
                maximum = _nonnegative_int(node["maxLength"], param=f"{node_param}.maxLength")
                if maximum < minimum:
                    _fail("maxLength must be at least minLength", param=f"{node_param}.maxLength")
                out["maxLength"] = maximum

        elif kind in ("number", "integer"):
            for key in ("minimum", "maximum"):
                if key in node:
                    value = node[key]
                    finite_number = (
                        isinstance(value, int) and not isinstance(value, bool)
                    ) or (isinstance(value, float) and math.isfinite(value))
                    if not finite_number:
                        _fail(f"{key} must be a finite number", param=f"{node_param}.{key}")
                    out[key] = value
            if "minimum" in out and "maximum" in out and out["maximum"] < out["minimum"]:
                _fail("maximum must be at least minimum", param=f"{node_param}.maximum")
        return out

    normalized = visit(schema, param, 0)
    if root_object and normalized["type"] != "object":
        _fail("the root schema must have type 'object'", param=f"{param}.type")
    return normalized


def _normalize_tool(tool: Any, index: int) -> dict[str, Any]:
    param = f"tools[{index}]"
    if not isinstance(tool, Mapping):
        _fail("each tool must be an object", param=param)
    _exact_keys(tool, {"type", "function"}, param=param)
    if tool.get("type") != "function":
        _fail("only function tools are supported", code="unsupported_parameter", param=f"{param}.type")
    fn = tool.get("function")
    if not isinstance(fn, Mapping):
        _fail("function must be an object", param=f"{param}.function")
    _exact_keys(fn, {"name", "description", "parameters", "strict"}, param=f"{param}.function")
    name = fn.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        _fail("function name must be 1-64 ASCII letters, digits, '_' or '-'",
              param=f"{param}.function.name")
    description = fn.get("description")
    if description is not None and not isinstance(description, str):
        _fail("function description must be a string", param=f"{param}.function.description")
    strict = fn.get("strict", True)
    if strict is not True:
        _fail("structured I/O v1 requires function.strict=true when present",
              param=f"{param}.function.strict")
    parameters = fn.get("parameters", {"type": "object", "properties": {}})
    normalized_schema = normalize_json_schema(parameters, param=f"{param}.function.parameters")
    normalized_fn: dict[str, Any] = {
        "name": name,
        "parameters": normalized_schema,
        "strict": True,
    }
    if description is not None:
        normalized_fn["description"] = description
    return {"type": "function", "function": normalized_fn}


def _normalize_response_format(value: Any) -> dict[str, Any]:
    if value is None:
        return {"type": "text"}
    if not isinstance(value, Mapping):
        _fail("response_format must be an object", param="response_format")
    kind = value.get("type")
    if kind == "text":
        _exact_keys(value, {"type"}, param="response_format")
        return {"type": "text"}
    if kind == "json_object":
        _exact_keys(value, {"type"}, param="response_format")
        return {"type": "json_object"}
    if kind != "json_schema":
        _fail("response_format.type must be text, json_object, or json_schema",
              code="unsupported_parameter", param="response_format.type")
    _exact_keys(value, {"type", "json_schema"}, param="response_format")
    spec = value.get("json_schema")
    if not isinstance(spec, Mapping):
        _fail("response_format.json_schema must be an object", param="response_format.json_schema")
    _exact_keys(spec, {"name", "description", "strict", "schema"},
                param="response_format.json_schema")
    name = spec.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        _fail("json_schema.name must be 1-64 ASCII letters, digits, '_' or '-'",
              param="response_format.json_schema.name")
    if spec.get("strict") is not True:
        _fail("structured I/O v1 requires json_schema.strict=true",
              param="response_format.json_schema.strict")
    if "schema" not in spec:
        _fail("json_schema.schema is required", param="response_format.json_schema.schema")
    normalized_spec: dict[str, Any] = {
        "name": name,
        "strict": True,
        "schema": normalize_json_schema(spec["schema"]),
    }
    if "description" in spec:
        if not isinstance(spec["description"], str):
            _fail("json_schema.description must be a string",
                  param="response_format.json_schema.description")
        normalized_spec["description"] = spec["description"]
    return {"type": "json_schema", "json_schema": normalized_spec}


def normalize_contract(body: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the structured fields of one already-object request body.

    The returned ``mode`` is ``None``, ``tools``, ``json_object``, or
    ``json_schema``.  This function intentionally ignores unrelated chat fields; the
    existing OpenAI validator remains their authority.
    """
    if not isinstance(body, Mapping):
        _fail("request body must be an object", param="body")
    raw_tools = body.get("tools")
    if raw_tools is None:
        tools: list[dict[str, Any]] = []
    elif not isinstance(raw_tools, list):
        _fail("tools must be a list", param="tools")
    else:
        if len(raw_tools) > MAX_TOOLS:
            _fail(f"tools supports at most {MAX_TOOLS} definitions", param="tools")
        tools = [_normalize_tool(tool, index) for index, tool in enumerate(raw_tools)]
    names = [tool["function"]["name"] for tool in tools]
    if len(set(names)) != len(names):
        duplicate = next(name for name in names if names.count(name) > 1)
        _fail(f"tool function names must be unique; duplicate {duplicate!r}", param="tools")

    choice = body.get("tool_choice")
    if choice is None:
        choice = "auto" if tools else "none"
    if choice not in ("auto", "none"):
        _fail("tool_choice supports only 'auto' and 'none' in v1",
              code="unsupported_parameter", param="tool_choice")
    if choice == "auto" and not tools:
        _fail("tool_choice:'auto' requires at least one tool", param="tool_choice")

    parallel = body.get("parallel_tool_calls")
    if parallel is not None and not isinstance(parallel, bool):
        _fail("parallel_tool_calls must be a boolean", param="parallel_tool_calls")
    if tools and choice != "none" and parallel not in (None, False):
        _fail("structured I/O v1 permits at most one tool call; set parallel_tool_calls:false",
              code="unsupported_parameter", param="parallel_tool_calls")

    response_format = _normalize_response_format(body.get("response_format"))
    active_tools = tools if choice == "auto" else []
    if active_tools and response_format["type"] != "text":
        _fail("tools and structured response_format cannot be active in the same v1 request",
              code="unsupported_parameter", param="response_format")
    mode = "tools" if active_tools else (
        response_format["type"] if response_format["type"] != "text" else None
    )
    return {
        "schema": EVIDENCE_SCHEMA,
        "mode": mode,
        "tools": tools,
        "active_tools": active_tools,
        "tool_choice": choice,
        "parallel_tool_calls": False if active_tools else None,
        "response_format": response_format,
        "renderer_id": PYTHON_ENVELOPE_RENDERER_ID,
        "parser_id": PYTHON_ENVELOPE_PARSER_ID,
    }


def _strict_json_loads(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-standard JSON constant {value!r}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in pairs:
            if key in out:
                raise ValueError(f"duplicate object key {key!r}")
            out[key] = value
        return out

    def finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON number {value!r}")
        return parsed

    return json.loads(text, parse_constant=reject_constant, parse_float=finite_float,
                      object_pairs_hook=unique_object)


def structured_parser_input(raw_output: Any, sanitized_output: Any) -> tuple[str, str]:
    """Admit only an explicit leading think-block normalization.

    General think-tag sanitation is intentionally too powerful for a strict parser:
    removing tags inside a JSON string could repair an undeclared tool name or alter
    an argument value.  Structured mode may drop one or more complete leading
    ``<think>...</think>`` blocks, but never edits the remaining JSON document.
    """
    if not isinstance(raw_output, str) or not isinstance(sanitized_output, str):
        _fail("structured model output must be text", code="malformed_model_output", param=None)
    if raw_output == sanitized_output:
        return raw_output, "none"

    lower = raw_output.lower()
    cursor = 0
    while cursor < len(raw_output) and raw_output[cursor].isspace():
        cursor += 1
    consumed = False
    while lower.startswith("<think>", cursor):
        close = lower.find("</think>", cursor + len("<think>"))
        if close < 0:
            _fail("structured reasoning prefix has no closing </think> tag",
                  code="malformed_model_output", param=None,
                  evidence={"normalization": "rejected_think_rewrite"})
        consumed = True
        cursor = close + len("</think>")
        while cursor < len(raw_output) and raw_output[cursor].isspace():
            cursor += 1
    candidate = raw_output[cursor:]
    if (not consumed or "<think>" in candidate.lower() or "</think>" in candidate.lower()
            or candidate.strip() != sanitized_output.strip()):
        _fail("think tags may only appear as complete leading blocks before structured JSON",
              code="malformed_model_output", param=None,
              evidence={"normalization": "rejected_think_rewrite"})
    return candidate, "strip_leading_think_blocks"


def _validate_output_tree(value: Any) -> None:
    """Bound parsed JSON iteratively so later validation/serialization cannot recurse explosively."""
    stack = [(value, 0)]
    nodes = 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > MAX_OUTPUT_NODES:
            _fail(f"structured model output exceeds {MAX_OUTPUT_NODES} JSON nodes",
                  code="malformed_model_output", param=None,
                  evidence={"maximum_nodes": MAX_OUTPUT_NODES})
        if depth > MAX_OUTPUT_DEPTH:
            _fail(f"structured model output exceeds depth {MAX_OUTPUT_DEPTH}",
                  code="malformed_model_output", param=None,
                  evidence={"maximum_depth": MAX_OUTPUT_DEPTH})
        if isinstance(item, dict):
            stack.extend((child, depth + 1) for child in item.values())
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif isinstance(item, float) and not math.isfinite(item):
            _fail("structured model output contains a non-finite number",
                  code="malformed_model_output", param=None)


def _normalize_history_tool_call(call: Any, *, param: str,
                                 tools_by_name: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(call, Mapping):
        _fail("assistant tool_calls items must be objects", param=param)
    _exact_keys(call, {"id", "type", "function"}, param=param)
    call_id = call.get("id")
    if not isinstance(call_id, str) or not _CALL_ID_RE.fullmatch(call_id):
        _fail("tool call id must be a 1-128 character ASCII identifier", param=f"{param}.id")
    if call.get("type") != "function":
        _fail("only function tool calls are supported", param=f"{param}.type")
    fn = call.get("function")
    if not isinstance(fn, Mapping):
        _fail("tool call function must be an object", param=f"{param}.function")
    _exact_keys(fn, {"name", "arguments"}, param=f"{param}.function")
    name = fn.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        _fail("historical function name must be 1-64 ASCII letters, digits, '_' or '-'",
              param=f"{param}.function.name")
    arguments = fn.get("arguments")
    if not isinstance(arguments, str):
        _fail("historical tool call arguments must be a JSON string",
              param=f"{param}.function.arguments")
    try:
        value = _strict_json_loads(arguments)
    except (json.JSONDecodeError, ValueError) as exc:
        _fail(f"historical tool call arguments are not strict JSON: {exc}",
              param=f"{param}.function.arguments")
    if not isinstance(value, dict):
        _fail("historical tool call arguments must decode to an object",
              param=f"{param}.function.arguments")
    # A continuation may intentionally omit tools to force an ordinary final answer.
    # Shape-check that history regardless; schema-check it when the same definition is
    # present in this request, without pretending a missing historical schema is known.
    if name in tools_by_name:
        validate_instance(value, tools_by_name[name]["function"]["parameters"],
                          param=f"{param}.function.arguments")
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                                    separators=(",", ":")),
        },
    }


def normalize_and_lower_messages(messages: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate tool history and return journal + role/text generation messages.

    ``messages`` preserves the normalized OpenAI objects for the journal.  The
    ``generation_messages`` value is safe for the current engine API, which accepts only
    ``role`` and string ``content``.  Tool results become user turns and carry a
    versioned JSON envelope; this is a renderer convention, not a claim that the model's
    native template understands OpenAI's ``tool`` role.
    """
    if not isinstance(messages, list) or not messages:
        _fail("messages must be a non-empty list", param="messages")
    tools_by_name = {
        tool["function"]["name"]: tool for tool in (contract.get("tools") or [])
        if isinstance(tool, Mapping) and isinstance(tool.get("function"), Mapping)
    }
    normalized: list[dict[str, Any]] = []
    lowered: list[dict[str, str]] = []
    pending: dict[str, str] = {}
    answered: set[str] = set()

    for index, raw in enumerate(messages):
        param = f"messages[{index}]"
        if not isinstance(raw, Mapping):
            _fail("each message must be an object", param=param)
        role = raw.get("role")
        if pending and role != "tool":
            call_id = next(iter(pending))
            _fail(f"tool result for {call_id!r} must immediately follow its assistant call",
                  param=f"{param}.role")
        if role in ("developer", "system", "user"):
            _exact_keys(raw, {"role", "content"}, param=param)
            if not isinstance(raw.get("content"), str):
                _fail("message content must be a string", param=f"{param}.content")
            item = {"role": role, "content": raw["content"]}
            normalized.append(item)
            lowered.append({"role": "system" if role == "developer" else role,
                            "content": raw["content"]})
            continue

        if role == "assistant":
            _exact_keys(raw, {"role", "content", "tool_calls"}, param=param)
            calls = raw.get("tool_calls")
            content = raw.get("content")
            if calls is None:
                if not isinstance(content, str):
                    _fail("assistant content must be a string when tool_calls is absent",
                          param=f"{param}.content")
                normalized.append({"role": "assistant", "content": content})
                lowered.append({"role": "assistant", "content": content})
                continue
            if not isinstance(calls, list) or len(calls) != 1:
                _fail("structured I/O v1 requires exactly one assistant tool call",
                      code="unsupported_parameter", param=f"{param}.tool_calls")
            if content not in (None, ""):
                _fail("assistant tool-call messages require null or empty content",
                      param=f"{param}.content")
            call = _normalize_history_tool_call(
                calls[0], param=f"{param}.tool_calls[0]", tools_by_name=tools_by_name
            )
            call_id = call["id"]
            if call_id in pending or call_id in answered:
                _fail(f"duplicate tool call id {call_id!r}", param=f"{param}.tool_calls[0].id")
            pending[call_id] = call["function"]["name"]
            normalized.append({"role": "assistant", "content": None, "tool_calls": [call]})
            envelope = {
                "protocol": PYTHON_ENVELOPE_RENDERER_ID,
                "type": "tool_call",
                "id": call_id,
                "name": call["function"]["name"],
                "arguments": _strict_json_loads(call["function"]["arguments"]),
            }
            lowered.append({"role": "assistant", "content": _history_envelope(envelope)})
            continue

        if role == "tool":
            _exact_keys(raw, {"role", "content", "tool_call_id", "name"}, param=param)
            content = raw.get("content")
            call_id = raw.get("tool_call_id")
            if not isinstance(content, str):
                _fail("tool result content must be a string", param=f"{param}.content")
            if not isinstance(call_id, str) or call_id not in pending:
                _fail("tool_call_id must name an unanswered preceding assistant call",
                      param=f"{param}.tool_call_id")
            expected_name = pending[call_id]
            name = raw.get("name", expected_name)
            if name != expected_name:
                _fail(f"tool result name must match {expected_name!r}", param=f"{param}.name")
            normalized_item = {"role": "tool", "content": content, "tool_call_id": call_id}
            if "name" in raw:
                normalized_item["name"] = name
            normalized.append(normalized_item)
            envelope = {
                "protocol": PYTHON_ENVELOPE_RENDERER_ID,
                "type": "tool_result",
                "id": call_id,
                "name": expected_name,
                "content": content,
            }
            lowered.append({"role": "user", "content": _history_envelope(envelope)})
            answered.add(call_id)
            del pending[call_id]
            continue

        _fail("message role must be developer, system, user, assistant, or tool",
              param=f"{param}.role")
    if pending:
        call_id = next(iter(pending))
        _fail(f"assistant tool call {call_id!r} has no following tool result",
              param="messages")
    return {"messages": normalized, "generation_messages": lowered}


def _history_envelope(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return f"<clozn-structured-history-v1>\n{encoded}\n</clozn-structured-history-v1>"


def renderer_instruction(contract: Mapping[str, Any]) -> str:
    """Return the deterministic, versioned system instruction for an active contract."""
    mode = contract.get("mode")
    common = (
        f"Clozn structured I/O protocol {PYTHON_ENVELOPE_RENDERER_ID}.\n"
        "Follow this output contract exactly. Return one complete JSON object and nothing else: "
        "no Markdown fence, prose, prefix, or suffix. Do not copy structured-history envelopes; "
        "they describe prior calls and results."
    )
    if mode == "tools":
        definitions = [tool["function"] for tool in contract.get("active_tools") or []]
        encoded = json.dumps(definitions, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return (
            common
            + "\nChoose exactly one shape. Ordinary answer: "
              '{"type":"message","content":"answer text"}. '
              "One tool call: "
              '{"type":"tool_call","name":"declared_name","arguments":{}}. '
              "Call only a declared function and make arguments satisfy its parameters schema. "
              "At most one call is permitted.\nTOOL_DEFINITIONS_JSON=" + encoded
        )
    if mode == "json_object":
        return common + "\nThe object itself is the requested answer."
    if mode == "json_schema":
        schema = contract["response_format"]["json_schema"]["schema"]
        encoded = json.dumps(schema, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return common + "\nThe object itself is the requested answer and must satisfy SCHEMA_JSON=" + encoded
    _fail("structured renderer requires an active contract mode",
          code="structured_mode_inactive", param=None)
    raise AssertionError("unreachable")


def render_messages(messages: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize/lower history and prepend the renderer instruction.

    Returns the same keys as :func:`normalize_and_lower_messages` plus a
    ``renderer_instruction`` field.  The normalized journal messages never contain
    the synthetic instruction.
    """
    if contract.get("mode") not in ("tools", "json_object", "json_schema"):
        _fail("structured renderer requires an active contract mode",
              code="structured_mode_inactive", param=None)
    plan = normalize_and_lower_messages(messages, contract)
    instruction = renderer_instruction(contract)
    plan["renderer_instruction"] = instruction
    plan["generation_messages"] = [
        {"role": "system", "content": instruction}, *plan["generation_messages"]
    ]
    return plan


def validate_instance(value: Any, schema: Mapping[str, Any], *,
                      param: str | None = "model_output") -> None:
    """Validate a JSON value against an already-normalized v1 schema."""
    errors: list[dict[str, str]] = []

    def err(path: str, message: str) -> None:
        errors.append({"path": path, "message": message})

    def visit(item: Any, node: Mapping[str, Any], path: str) -> None:
        kind = str(node.get("type") or "")
        if not _json_type_matches(item, kind):
            err(path, f"expected {kind}")
            return
        if "enum" in node and item not in node["enum"]:
            err(path, "value is not in enum")
            return
        if kind == "object":
            properties = node.get("properties") or {}
            for required in node.get("required") or []:
                if required not in item:
                    err(path, f"missing required property {required!r}")
            for key, child in item.items():
                if key not in properties:
                    err(f"{path}.{key}", "additional property is not allowed")
                else:
                    visit(child, properties[key], f"{path}.{key}")
        elif kind == "array":
            if len(item) < int(node.get("minItems", 0)):
                err(path, f"array has fewer than {node.get('minItems')} items")
            if "maxItems" in node and len(item) > int(node["maxItems"]):
                err(path, f"array has more than {node['maxItems']} items")
            for index, child in enumerate(item):
                visit(child, node["items"], f"{path}[{index}]")
        elif kind == "string":
            if len(item) < int(node.get("minLength", 0)):
                err(path, f"string is shorter than {node.get('minLength')}")
            if "maxLength" in node and len(item) > int(node["maxLength"]):
                err(path, f"string is longer than {node['maxLength']}")
        elif kind in ("number", "integer"):
            if "minimum" in node and item < node["minimum"]:
                err(path, f"number is below minimum {node['minimum']}")
            if "maximum" in node and item > node["maximum"]:
                err(path, f"number is above maximum {node['maximum']}")

    visit(value, schema, "$")
    if errors:
        _fail("structured value does not satisfy its JSON schema",
              code="schema_validation_failed", param=param, evidence={"errors": errors})


def _parse_whole_output(raw_output: Any) -> tuple[Any, str, str]:
    if not isinstance(raw_output, str):
        _fail("model output must be text", code="malformed_model_output", param=None,
              evidence={"output_type": type(raw_output).__name__})
    if len(raw_output.encode("utf-8")) > MAX_OUTPUT_BYTES:
        _fail(f"model output exceeds {MAX_OUTPUT_BYTES} UTF-8 bytes",
              code="malformed_model_output", param=None,
              evidence={"output_bytes": len(raw_output.encode("utf-8")),
                        "maximum_bytes": MAX_OUTPUT_BYTES})
    try:
        return _strict_json_loads(raw_output), "none", raw_output
    except (json.JSONDecodeError, ValueError, RecursionError, OverflowError) as first:
        match = _FENCE_RE.fullmatch(raw_output)
        if match is not None:
            inner = match.group("body")
            try:
                return _strict_json_loads(inner), "strip_single_json_fence", inner
            except (json.JSONDecodeError, ValueError, RecursionError, OverflowError) as second:
                detail = str(second)
        else:
            detail = str(first)
        _fail("model output is not exactly one strict JSON value",
              code="malformed_model_output", param=None,
              evidence={"raw_output": raw_output, "normalization": "none", "parse_error": detail})
    raise AssertionError("unreachable")


def parse_output(raw_output: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly parse one model result according to an active normalized contract."""
    mode = contract.get("mode")
    if mode not in ("tools", "json_object", "json_schema"):
        _fail("structured parser requires an active contract mode",
              code="structured_mode_inactive", param=None)
    value, normalization, normalized_text = _parse_whole_output(raw_output)
    _validate_output_tree(value)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "mode": mode,
        "raw_output": raw_output,
        "normalization": normalization,
        "normalized_output": normalized_text,
        "parser_id": PYTHON_ENVELOPE_PARSER_ID,
    }
    if not isinstance(value, dict):
        _fail("structured model output must be a JSON object",
              code="malformed_model_output", param=None, evidence=evidence)

    if mode == "tools":
        kind = value.get("type")
        if kind == "message":
            extra = sorted(set(value) - {"type", "content"})
            if extra or not isinstance(value.get("content"), str):
                _fail("tool-mode message must contain exactly type and string content",
                      code="malformed_model_output", param=None,
                      evidence={**evidence, "unexpected_fields": extra})
            return {"kind": "message", "content": value["content"], "evidence": evidence}
        if kind != "tool_call":
            _fail("tool-mode output type must be 'message' or 'tool_call'",
                  code="malformed_model_output", param=None, evidence=evidence)
        extra = sorted(set(value) - {"type", "name", "arguments"})
        if extra:
            _fail("tool call contains unsupported fields", code="malformed_model_output", param=None,
                  evidence={**evidence, "unexpected_fields": extra})
        name = value.get("name")
        tools = {tool["function"]["name"]: tool for tool in contract.get("active_tools") or []}
        if name not in tools:
            _fail(f"model selected undeclared tool {name!r}", code="unknown_tool", param=None,
                  evidence={**evidence, "allowed_tools": sorted(tools)})
        arguments = value.get("arguments")
        if not isinstance(arguments, dict):
            _fail("tool arguments must be a JSON object", code="malformed_model_output", param=None,
                  evidence=evidence)
        try:
            validate_instance(arguments, tools[name]["function"]["parameters"], param=None)
        except StructuredIOError as exc:
            exc.evidence = {**evidence, **exc.evidence, "tool": name}
            raise
        canonical = json.dumps(arguments, ensure_ascii=False, allow_nan=False,
                               sort_keys=True, separators=(",", ":"))
        return {"kind": "tool_call", "name": name, "arguments": arguments,
                "arguments_json": canonical, "evidence": evidence}

    if mode == "json_schema":
        schema = contract["response_format"]["json_schema"]["schema"]
        try:
            validate_instance(value, schema, param=None)
        except StructuredIOError as exc:
            exc.evidence = {**evidence, **exc.evidence}
            raise
    canonical = json.dumps(value, ensure_ascii=False, allow_nan=False,
                           sort_keys=True, separators=(",", ":"))
    return {"kind": mode, "value": value, "content": canonical, "evidence": evidence}


def validate_native_message(message: Any, contract: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate one message produced by the qualified native parser.

    The native parser owns model/template-specific syntax such as reasoning and tool
    markers.  This validator does not parse or repair that syntax: it accepts only the
    parser's OpenAI-shaped assistant message, then independently enforces Clozn's
    public one-call contract and JSON Schema subset.  Public tool-call IDs are minted
    later and native IDs are never trusted for association.
    """
    mode = contract.get("mode")
    if mode not in ("tools", "json_object", "json_schema"):
        _fail("native message validator requires an active contract mode",
              code="structured_mode_inactive", param=None)
    if not isinstance(message, Mapping):
        _fail("native parser output must be an object",
              code="malformed_model_output", param=None,
              evidence={"validator_id": NATIVE_MESSAGE_VALIDATOR_ID,
                        "output_type": type(message).__name__})

    try:
        encoded_message = json.dumps(
            message, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        _fail("native parser output is not finite JSON",
              code="malformed_model_output", param=None,
              evidence={"validator_id": NATIVE_MESSAGE_VALIDATOR_ID,
                        "parse_error": str(exc)})
    if len(encoded_message) > MAX_OUTPUT_BYTES:
        _fail(f"native parser output exceeds {MAX_OUTPUT_BYTES} UTF-8 bytes",
              code="malformed_model_output", param=None,
              evidence={"validator_id": NATIVE_MESSAGE_VALIDATOR_ID,
                        "output_bytes": len(encoded_message),
                        "maximum_bytes": MAX_OUTPUT_BYTES})

    value = dict(message)
    _validate_output_tree(value)
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "mode": mode,
        "validator_id": NATIVE_MESSAGE_VALIDATOR_ID,
        "source": "native_parser_message",
        "normalization": "none",
    }
    allowed_message_fields = {"role", "content", "reasoning_content", "tool_calls"}
    extra = sorted(set(value) - allowed_message_fields)
    if extra:
        _fail("native assistant message contains unsupported fields",
              code="malformed_model_output", param=None,
              evidence={**evidence, "unexpected_fields": extra})
    if value.get("role") != "assistant":
        _fail("native parser output role must be 'assistant'",
              code="malformed_model_output", param=None, evidence=evidence)
    if "reasoning_content" in value and not isinstance(value["reasoning_content"], str):
        _fail("native reasoning_content must be a string when present",
              code="malformed_model_output", param=None, evidence=evidence)

    calls = value.get("tool_calls")
    has_calls = calls not in (None, [])
    if mode != "tools":
        if has_calls:
            _fail("native JSON output must not contain tool calls",
                  code="malformed_model_output", param=None, evidence=evidence)
        content = value.get("content")
        if not isinstance(content, str):
            _fail("native JSON output content must be a string",
                  code="malformed_model_output", param=None, evidence=evidence)
        if len(content.encode("utf-8")) > MAX_OUTPUT_BYTES:
            _fail(f"native JSON content exceeds {MAX_OUTPUT_BYTES} UTF-8 bytes",
                  code="malformed_model_output", param=None,
                  evidence={**evidence, "maximum_bytes": MAX_OUTPUT_BYTES})
        try:
            parsed_value = _strict_json_loads(content)
        except (json.JSONDecodeError, ValueError, RecursionError, OverflowError) as exc:
            _fail("native JSON content is not exactly one strict JSON value",
                  code="malformed_model_output", param=None,
                  evidence={**evidence, "parse_error": str(exc)})
        _validate_output_tree(parsed_value)
        if not isinstance(parsed_value, dict):
            _fail("native structured content must be a JSON object",
                  code="malformed_model_output", param=None, evidence=evidence)
        if mode == "json_schema":
            schema = contract["response_format"]["json_schema"]["schema"]
            try:
                validate_instance(parsed_value, schema, param=None)
            except StructuredIOError as exc:
                exc.evidence = {**evidence, **exc.evidence}
                raise
        canonical = json.dumps(parsed_value, ensure_ascii=False, allow_nan=False,
                               sort_keys=True, separators=(",", ":"))
        return {"kind": mode, "value": parsed_value, "content": canonical,
                "evidence": evidence}

    if not has_calls:
        content = value.get("content")
        if not isinstance(content, str):
            _fail("native tool-mode message must have string content when no call is present",
                  code="malformed_model_output", param=None, evidence=evidence)
        return {"kind": "message", "content": content, "evidence": evidence}

    if not isinstance(calls, list) or len(calls) != 1:
        _fail("native tool-mode output must contain exactly one tool call",
              code="malformed_model_output", param=None, evidence=evidence)
    if value.get("content") not in (None, ""):
        _fail("native tool-call message requires null or empty content",
              code="malformed_model_output", param=None, evidence=evidence)
    call = calls[0]
    if not isinstance(call, Mapping):
        _fail("native tool call must be an object",
              code="malformed_model_output", param=None, evidence=evidence)
    call_extra = sorted(set(call) - {"id", "type", "function"})
    if call_extra or call.get("type") != "function":
        _fail("native tool call must contain only an optional id, type:function, and function",
              code="malformed_model_output", param=None,
              evidence={**evidence, "unexpected_fields": call_extra})
    native_id = call.get("id")
    if native_id not in (None, "") and (
        not isinstance(native_id, str) or not _CALL_ID_RE.fullmatch(native_id)
    ):
        _fail("native tool call id has an invalid shape",
              code="malformed_model_output", param=None, evidence=evidence)
    fn = call.get("function")
    if not isinstance(fn, Mapping):
        _fail("native tool call function must be an object",
              code="malformed_model_output", param=None, evidence=evidence)
    fn_extra = sorted(set(fn) - {"name", "arguments"})
    if fn_extra:
        _fail("native tool call function contains unsupported fields",
              code="malformed_model_output", param=None,
              evidence={**evidence, "unexpected_fields": fn_extra})
    name = fn.get("name")
    tools = {tool["function"]["name"]: tool for tool in contract.get("active_tools") or []}
    if name not in tools:
        _fail(f"native parser selected undeclared tool {name!r}",
              code="unknown_tool", param=None,
              evidence={**evidence, "allowed_tools": sorted(tools)})
    arguments_text = fn.get("arguments")
    if not isinstance(arguments_text, str):
        _fail("native tool arguments must be a JSON string",
              code="malformed_model_output", param=None, evidence=evidence)
    if len(arguments_text.encode("utf-8")) > MAX_OUTPUT_BYTES:
        _fail(f"native tool arguments exceed {MAX_OUTPUT_BYTES} UTF-8 bytes",
              code="malformed_model_output", param=None,
              evidence={**evidence, "maximum_bytes": MAX_OUTPUT_BYTES})
    try:
        arguments = _strict_json_loads(arguments_text)
    except (json.JSONDecodeError, ValueError, RecursionError, OverflowError) as exc:
        _fail("native tool arguments are not strict JSON",
              code="malformed_model_output", param=None,
              evidence={**evidence, "parse_error": str(exc), "tool": name})
    _validate_output_tree(arguments)
    if not isinstance(arguments, dict):
        _fail("native tool arguments must decode to an object",
              code="malformed_model_output", param=None,
              evidence={**evidence, "tool": name})
    try:
        validate_instance(arguments, tools[name]["function"]["parameters"], param=None)
    except StructuredIOError as exc:
        exc.evidence = {**evidence, **exc.evidence, "tool": name}
        raise
    canonical = json.dumps(arguments, ensure_ascii=False, allow_nan=False,
                           sort_keys=True, separators=(",", ":"))
    return {"kind": "tool_call", "name": name, "arguments": arguments,
            "arguments_json": canonical, "evidence": evidence}


def serialize_openai_result(parsed: Mapping[str, Any], *, call_id: str | None = None) -> dict[str, Any]:
    """Return ``message`` + ``finish_reason`` for an OpenAI chat choice."""
    kind = parsed.get("kind")
    if kind == "tool_call":
        call_id = call_id or parsed.get("call_id") or ("call_" + secrets.token_hex(12))
        if not isinstance(call_id, str) or not _CALL_ID_RE.fullmatch(call_id):
            _fail("call_id must be a 1-128 character ASCII identifier", param="call_id")
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {"name": parsed["name"], "arguments": parsed["arguments_json"]},
            }],
        }
        return {"message": message, "finish_reason": "tool_calls"}
    if kind == "message":
        return {"message": {"role": "assistant", "content": parsed["content"]},
                "finish_reason": "stop"}
    if kind in ("json_object", "json_schema"):
        return {"message": {"role": "assistant", "content": parsed["content"]},
                "finish_reason": "stop"}
    _fail(f"cannot serialize parsed result kind {kind!r}", code="invalid_parsed_result", param=None)
    raise AssertionError("unreachable")


def openai_stream_deltas(parsed: Mapping[str, Any], *, call_id: str | None = None) -> dict[str, Any]:
    """Return buffered SSE deltas plus the terminal finish reason.

    Route glue still owns SSE framing, IDs, timestamps, and the terminal Clozn run ID.
    A tool call is one complete delta in v1; clients may accumulate it exactly like a
    fragmented arguments stream.
    """
    result = serialize_openai_result(parsed, call_id=call_id)
    message = result["message"]
    deltas: list[dict[str, Any]] = [{"role": "assistant"}]
    if message.get("tool_calls"):
        call = message["tool_calls"][0]
        deltas.append({"tool_calls": [{"index": 0, **call}]})
    else:
        deltas.append({"content": message["content"]})
    return {"deltas": deltas, "finish_reason": result["finish_reason"]}


def _empty_registry() -> dict[str, Any]:
    return {"schema_version": QUALIFICATION_SCHEMA, "entries": []}


def _schema_subsets_for(features: list[str]) -> dict[str, str]:
    subsets: dict[str, str] = {}
    if "tools" in features:
        subsets["tool_parameters"] = JSON_SCHEMA_SUBSET_ID
    if "json_schema" in features:
        subsets["json_schema"] = JSON_SCHEMA_SUBSET_ID
    return subsets


def validate_qualification_registry(registry: Any) -> dict[str, Any]:
    """Validate a qualification allow-list without treating it as evidence itself."""
    if not isinstance(registry, Mapping):
        _fail("qualification registry must be an object",
              code="qualification_registry_invalid", param=None)
    extra = sorted(set(registry) - {"schema_version", "entries"})
    if extra:
        _fail(f"qualification registry has unsupported field {extra[0]!r}",
              code="qualification_registry_invalid", param=None)
    if registry.get("schema_version") != QUALIFICATION_SCHEMA:
        _fail(f"qualification registry schema_version must be {QUALIFICATION_SCHEMA!r}",
              code="qualification_registry_invalid", param=None)
    entries = registry.get("entries")
    if not isinstance(entries, list):
        _fail("qualification registry entries must be a list",
              code="qualification_registry_invalid", param=None)
    normalized = []
    seen: set[tuple[str, str]] = set()
    allowed_features = {"tools", "json_object", "json_schema"}
    for index, raw in enumerate(entries):
        if not isinstance(raw, Mapping):
            _fail(f"qualification entry {index} must be an object",
                  code="qualification_registry_invalid", param=None)
        required = {
            "model_sha256", "template_fingerprint", "features", "schema_subsets",
            "pipeline", "evidence",
        }
        extra = sorted(set(raw) - required)
        missing = sorted(required - set(raw))
        if extra or missing:
            _fail(f"qualification entry {index} fields are invalid",
                  code="qualification_registry_invalid", param=None,
                  evidence={"entry": index, "extra": extra, "missing": missing})
        sha = raw["model_sha256"]
        fingerprint = raw["template_fingerprint"]
        features = raw["features"]
        if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
            _fail(f"qualification entry {index} has invalid model_sha256",
                  code="qualification_registry_invalid", param=None)
        if not isinstance(fingerprint, str) or not _FINGERPRINT_RE.fullmatch(fingerprint):
            _fail(f"qualification entry {index} has invalid template_fingerprint",
                  code="qualification_registry_invalid", param=None)
        if (not isinstance(features, list) or not features
                or any(not isinstance(feature, str) for feature in features)
                or len(set(features)) != len(features)
                or any(feature not in allowed_features for feature in features)):
            _fail(f"qualification entry {index} has invalid features",
                  code="qualification_registry_invalid", param=None)
        expected_subsets = _schema_subsets_for(features)
        if raw["schema_subsets"] != expected_subsets:
            _fail(f"qualification entry {index} has invalid schema_subsets",
                  code="qualification_registry_invalid", param=None)
        pipeline = raw["pipeline"]
        if not isinstance(pipeline, Mapping) or dict(pipeline) != NATIVE_QUALIFICATION_PIPELINE:
            _fail(f"qualification entry {index} names an unknown native pipeline",
                  code="qualification_registry_invalid", param=None,
                  evidence={"entry": index, "expected_pipeline": NATIVE_QUALIFICATION_PIPELINE})
        evidence = raw["evidence"]
        if not isinstance(evidence, Mapping):
            _fail(f"qualification entry {index} evidence must be an object",
                  code="qualification_registry_invalid", param=None)
        evidence_required = {
            "schema_version", "suite_id", "artifact_version", "payload_sha256",
        }
        if set(evidence) != evidence_required:
            _fail(f"qualification entry {index} evidence fields are invalid",
                  code="qualification_registry_invalid", param=None)
        if (evidence.get("schema_version") != QUALIFICATION_EVIDENCE_SCHEMA
                or evidence.get("suite_id") != QUALIFICATION_SUITE_ID
                or evidence.get("artifact_version") != 2
                or not isinstance(evidence.get("payload_sha256"), str)
                or not _SHA256_RE.fullmatch(evidence["payload_sha256"])):
            _fail(f"qualification entry {index} has invalid qualification evidence",
                  code="qualification_registry_invalid", param=None)
        key = (sha.lower(), fingerprint.lower())
        if key in seen:
            _fail(f"qualification registry repeats identity at entry {index}",
                  code="qualification_registry_invalid", param=None)
        seen.add(key)
        entry = {
            "model_sha256": key[0],
            "template_fingerprint": key[1],
            "features": list(features),
            "schema_subsets": dict(expected_subsets),
            "pipeline": dict(NATIVE_QUALIFICATION_PIPELINE),
            "evidence": dict(evidence),
        }
        normalized.append(entry)
    return {"schema_version": QUALIFICATION_SCHEMA, "entries": normalized}


def load_qualification_registry(*, environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Load the explicit registry path, or return an empty allow-list when unset."""
    environ = os.environ if environ is None else environ
    path = str(environ.get(QUALIFICATIONS_ENV) or "").strip()
    if not path:
        return _empty_registry()
    try:
        with open(os.path.abspath(os.path.expanduser(path)), encoding="utf-8") as handle:
            registry = json.load(handle)
    except Exception as exc:
        _fail(f"configured qualification registry is unreadable: {type(exc).__name__}: {exc}",
              code="qualification_registry_invalid", param=None,
              evidence={"environment": QUALIFICATIONS_ENV, "path": path})
    return validate_qualification_registry(registry)


@lru_cache(maxsize=8)
def _cached_gguf_identity(path: str, size: int, mtime_ns: int) -> dict[str, Any]:
    """Read immutable GGUF metadata once per exact on-disk file version."""
    del size, mtime_ns  # They are cache-key material; gguf_identity reads the path itself.
    from clozn.artifacts.contracts import gguf_identity
    return gguf_identity(path)


def resolve_qualification_registry(
    identity: Mapping[str, Any] | None,
    *,
    artifact_root: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve the explicit registry or one installed exact-model chat-I/O artifact.

    Explicit configuration is exclusive.  Without it, the installed artifact is
    revalidated on every structured request so removal, ambiguity, or evidence
    tampering cannot silently retain authorization.  Only the expensive GGUF header
    identity is cached, keyed by the file's size and mtime.
    """
    environ = os.environ if environ is None else environ
    if str(environ.get(QUALIFICATIONS_ENV) or "").strip():
        return load_qualification_registry(environ=environ)

    identity = identity if isinstance(identity, Mapping) else {}
    model_path = identity.get("model_path")
    live_sha = identity.get("model_sha256")
    fingerprint = identity.get("template_fingerprint")
    if not all(isinstance(value, str) and value for value in (
            model_path, live_sha, fingerprint)):
        return _empty_registry()

    resolved = os.path.abspath(os.path.expanduser(model_path))
    try:
        stat = os.stat(resolved)
        model_identity = _cached_gguf_identity(resolved, stat.st_size, stat.st_mtime_ns)
    except Exception:
        return _empty_registry()
    if str(model_identity.get("sha256") or "").lower() != live_sha.lower():
        return _empty_registry()

    root = artifact_root or str(environ.get("CLOZN_ARTIFACTS_DIR") or "").strip()
    if not root:
        root = os.path.join(os.path.expanduser("~"), ".clozn", "artifacts")
    try:
        from clozn.artifacts.contracts import (
            ArtifactContractError, find_compatible_chat_io_profile,
        )
        profile = find_compatible_chat_io_profile(
            model_identity, fingerprint, root,
        )
    except ArtifactContractError as exc:
        _fail(
            f"installed structured-I/O qualification artifact is invalid: {exc}",
            code="qualification_registry_invalid", param=None,
            evidence={"artifact_root": os.path.abspath(os.path.expanduser(root))},
        )
    if profile is None:
        return _empty_registry()
    return validate_qualification_registry({
        "schema_version": QUALIFICATION_SCHEMA,
        "entries": [profile["registry_entry"]],
    })


def require_qualification(identity: Mapping[str, Any] | None, feature: str, *,
                          runtime_pipeline: Mapping[str, Any] | None = None,
                          registry: Mapping[str, Any] | None = None,
                          environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Return the exact matching allow-list entry or fail closed.

    ``feature`` is one of ``tools``, ``json_object``, or ``json_schema``.  The
    request's model label is intentionally absent from this API.
    """
    param = "tools" if feature == "tools" else "response_format"
    if feature not in {"tools", "json_object", "json_schema"}:
        _fail(f"unknown structured feature {feature!r}", code="invalid_parameter", param=param)
    if not isinstance(identity, Mapping):
        identity = {}
    sha = identity.get("model_sha256")
    fingerprint = identity.get("template_fingerprint")
    if not isinstance(sha, str) or not _SHA256_RE.fullmatch(sha):
        _fail("loaded model has no exact SHA-256 identity; structured I/O is not qualified",
              code="model_not_qualified", param=param, evidence={"reason": "missing_model_sha256"})
    if not isinstance(fingerprint, str) or not _FINGERPRINT_RE.fullmatch(fingerprint):
        _fail("loaded model has no exact template fingerprint; structured I/O is not qualified",
              code="model_not_qualified", param=param,
              evidence={"reason": "missing_template_fingerprint", "model_sha256": sha.lower()})
    checked = validate_qualification_registry(registry) if registry is not None else (
        load_qualification_registry(environ=environ)
    )
    for entry in checked["entries"]:
        if (entry["model_sha256"] == sha.lower()
                and entry["template_fingerprint"] == fingerprint.lower()):
            if feature not in entry["features"]:
                _fail(f"exact model/template identity is not qualified for {feature}",
                      code="model_not_qualified", param=param,
                      evidence={"reason": "feature_not_qualified", "feature": feature,
                                "qualified_features": entry["features"]})
            actual_pipeline = runtime_pipeline
            if actual_pipeline is None and isinstance(identity, Mapping):
                candidate = identity.get("chat_io_pipeline")
                actual_pipeline = candidate if isinstance(candidate, Mapping) else None
            if not isinstance(actual_pipeline, Mapping):
                _fail("loaded worker did not report a native structured-I/O pipeline",
                      code="model_not_qualified", param=param,
                      evidence={"reason": "missing_runtime_pipeline"})
            actual = dict(actual_pipeline)
            allowed_key_sets = {
                frozenset(NATIVE_WORKER_PIPELINE),
                frozenset(NATIVE_QUALIFICATION_PIPELINE),
            }
            if frozenset(actual) not in allowed_key_sets:
                _fail("loaded worker reported an invalid native structured-I/O pipeline",
                      code="model_not_qualified", param=param,
                      evidence={"reason": "runtime_pipeline_mismatch",
                                "runtime_pipeline": actual,
                                "qualified_pipeline": entry["pipeline"]})
            for name, expected in entry["pipeline"].items():
                if name == "validator_id" and name not in actual:
                    actual_value = NATIVE_MESSAGE_VALIDATOR_ID
                else:
                    actual_value = actual.get(name)
                if actual_value != expected:
                    _fail("loaded worker pipeline does not match exact qualification",
                          code="model_not_qualified", param=param,
                          evidence={"reason": "runtime_pipeline_mismatch",
                                    "field": name, "runtime_value": actual_value,
                                    "qualified_value": expected})
            return dict(entry)
    _fail("exact loaded model/template identity is not in the structured-I/O qualification registry",
          code="model_not_qualified", param=param,
          evidence={"reason": "identity_not_qualified", "model_sha256": sha.lower(),
                    "template_fingerprint": fingerprint.lower()})
    raise AssertionError("unreachable")


def qualification_evidence(entry: Mapping[str, Any], feature: str,
                           contract: Mapping[str, Any]) -> dict[str, Any]:
    """Build the small qualification block suitable for run evidence."""
    request_basis: dict[str, Any] = {
        "feature": feature,
        "mode": contract.get("mode"),
        "active_tools": contract.get("active_tools") or [],
        "tool_choice": contract.get("tool_choice"),
        "parallel_tool_calls": contract.get("parallel_tool_calls"),
        "response_format": contract.get("response_format"),
    }
    encoded = json.dumps(request_basis, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return {
        "schema": EVIDENCE_SCHEMA,
        "feature": feature,
        "model_sha256": entry["model_sha256"],
        "template_fingerprint": entry["template_fingerprint"],
        "pipeline": dict(entry["pipeline"]),
        "schema_subsets": dict(entry["schema_subsets"]),
        "request_contract_sha256": hashlib.sha256(encoded).hexdigest(),
        "qualification_evidence": dict(entry.get("evidence") or {}),
    }
