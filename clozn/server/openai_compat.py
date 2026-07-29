"""Honest request validation for Clozn's intentionally small OpenAI-compatible surface.

The gateway used to pick a few fields out of a request and silently discard the rest.  That is especially
dangerous for tools, stop sequences, structured output, penalties, and ``n``: a client sees HTTP 200 and
reasonably assumes the requested behavior happened.  This module is the single policy table used by both
OpenAI routes.  It accepts fields Clozn implements, strips only documented neutral/no-op values, and raises
an OpenAI-shaped 400 for every behavior-bearing field the runtime cannot honor.

Keep docs/OPENAI_COMPATIBILITY.md in lockstep with the exported field sets below.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any


class CompatibilityError(ValueError):
    """A request error that can be serialized as an OpenAI ``invalid_request_error``."""

    def __init__(self, message: str, *, param: str | None, code: str = "unsupported_parameter"):
        super().__init__(message)
        self.param = param
        self.code = code


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fail(message: str, param: str, *, code: str = "invalid_parameter") -> None:
    raise CompatibilityError(message, param=param, code=code)


def _neutral(value: Any, predicate: Callable[[Any], bool]) -> bool:
    return value is None or predicate(value)


def _zero(value: Any) -> bool:
    return _is_number(value) and float(value) == 0.0


def _empty_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and not value


def _empty_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and not value


CHAT_SUPPORTED_FIELDS = frozenset({
    "model", "messages", "max_tokens", "max_completion_tokens", "temperature", "top_p", "seed",
    "stream", "top_k", "repeat_penalty", "clozn_trust", "clozn_receipt", "clozn_lens", "clozn_selective",
    "clozn_guard", "clozn_sources", "tools", "tool_choice", "parallel_tool_calls", "response_format",
})

# Accepted only at the listed neutral value, then removed.  These are compatibility affordances, not
# claimed features.  A behavior-bearing value fails with the field named in ``error.param``.
CHAT_NEUTRAL_FIELDS: dict[str, Callable[[Any], bool]] = {
    "n": lambda v: _is_int(v) and v == 1,
    "user": lambda v: isinstance(v, str),
    "frequency_penalty": _zero,
    "presence_penalty": _zero,
    "logprobs": lambda v: v is False,
    "top_logprobs": lambda v: _is_int(v) and v == 0,
    "stop": lambda _v: False,                 # only null is neutral (handled by _neutral)
    "logit_bias": _empty_mapping,
    "functions": _empty_sequence,
    "function_call": lambda v: v == "none",
    "modalities": lambda v: v == ["text"],
    "audio": lambda _v: False,
    "prediction": lambda _v: False,
    "store": lambda v: v is False,
    "metadata": _empty_mapping,
    "service_tier": lambda v: v in ("auto", "default"),
    "stream_options": lambda v: v == {"include_usage": False},
}


COMPLETION_SUPPORTED_FIELDS = frozenset({
    "model", "prompt", "max_tokens", "temperature", "top_p", "seed", "stream", "top_k",
    "repeat_penalty", "rep_penalty",
})

COMPLETION_NEUTRAL_FIELDS: dict[str, Callable[[Any], bool]] = {
    "n": lambda v: _is_int(v) and v == 1,
    "best_of": lambda v: _is_int(v) and v == 1,
    "echo": lambda v: v is False,
    "user": lambda v: isinstance(v, str),
    "frequency_penalty": _zero,
    "presence_penalty": _zero,
    "logprobs": lambda _v: False,
    "stop": lambda _v: False,
    "suffix": lambda _v: False,
    "logit_bias": _empty_mapping,
}


def _check_known_fields(body: Mapping[str, Any], supported: frozenset[str],
                        neutral: Mapping[str, Callable[[Any], bool]]) -> None:
    for field, value in body.items():
        if field in supported:
            continue
        predicate = neutral.get(field)
        if predicate is None:
            raise CompatibilityError(
                f"unsupported parameter '{field}'; see docs/OPENAI_COMPATIBILITY.md",
                param=field,
            )
        if not _neutral(value, predicate):
            raise CompatibilityError(
                f"parameter '{field}' is supported only at its neutral/default value; "
                "Clozn cannot honor the requested behavior",
                param=field,
            )


def _positive_int(body: Mapping[str, Any], field: str) -> int | None:
    if field not in body or body[field] is None:
        return None
    value = body[field]
    if not _is_int(value) or value < 1:
        _fail(f"{field} must be an integer of at least 1", field)
    return int(value)


def _number_in(body: Mapping[str, Any], field: str, low: float, high: float,
               *, low_inclusive: bool = True) -> float | None:
    if field not in body or body[field] is None:
        return None
    value = body[field]
    if not _is_number(value):
        _fail(f"{field} must be a number", field)
    number = float(value)
    below = number < low if low_inclusive else number <= low
    if not math.isfinite(number) or below or number > high:
        left = "[" if low_inclusive else "("
        _fail(f"{field} must be in {left}{low}, {high}]", field)
    return number


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        _fail("messages must be a non-empty list of text {role, content} objects", "messages")
    out: list[dict[str, str]] = []
    for index, message in enumerate(value):
        param = f"messages[{index}]"
        if not isinstance(message, Mapping):
            _fail("each message must be an object", param)
        extra = set(message) - {"role", "content"}
        if extra:
            field = sorted(extra)[0]
            _fail(f"message field '{field}' is unsupported; Clozn accepts text-only messages", f"{param}.{field}")
        role = message.get("role")
        content = message.get("content")
        if role not in ("developer", "system", "user", "assistant"):
            _fail("message role must be developer, system, user, or assistant", f"{param}.role")
        if not isinstance(content, str):
            _fail("message content must be a string (multimodal parts are unsupported)", f"{param}.content")
        # Local GGUF templates generally predate the developer role.  Its instruction semantics map to
        # system for this text-only surface; the public matrix documents this normalization explicitly.
        out.append({"role": "system" if role == "developer" else str(role), "content": content})
    return out


def _normalize_sources(value: Any, *, message_count: int) -> list[dict[str, Any]]:
    """Validate the explicit source-identity extension without touching message content.

    The extension is a list of ``{message_index, source_id, label?}`` records.  It is deliberately
    separate from OpenAI message objects so ordinary clients remain byte-for-byte compatible and
    unsupported message fields still fail clearly.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        _fail("clozn_sources must be a list", "clozn_sources")
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_indexes: set[int] = set()
    for position, raw in enumerate(value):
        param = f"clozn_sources[{position}]"
        if not isinstance(raw, Mapping):
            _fail("each clozn_sources entry must be an object", param)
        extra = set(raw) - {"message_index", "source_id", "label"}
        if extra:
            field = sorted(extra)[0]
            _fail(f"source metadata field '{field}' is unsupported", f"{param}.{field}")
        index = raw.get("message_index")
        if not _is_int(index) or index < 0 or index >= message_count:
            _fail(
                f"message_index must identify one of the request's {message_count} messages",
                f"{param}.message_index",
            )
        source_id = raw.get("source_id")
        if not isinstance(source_id, str) or not source_id.strip() or len(source_id) > 256:
            _fail("source_id must be a non-empty string of at most 256 characters",
                  f"{param}.source_id")
        source_id = source_id.strip()
        if source_id in seen_ids:
            _fail("source_id values must be unique within a request", f"{param}.source_id")
        if index in seen_indexes:
            _fail("each message_index may have at most one source identity",
                  f"{param}.message_index")
        item: dict[str, Any] = {"message_index": int(index), "source_id": source_id}
        label = raw.get("label")
        if label is not None:
            if not isinstance(label, str) or not label.strip() or len(label) > 256:
                _fail("label must be a non-empty string of at most 256 characters",
                      f"{param}.label")
            item["label"] = label.strip()
        normalized.append(item)
        seen_ids.add(source_id)
        seen_indexes.add(int(index))
    return normalized


def normalize_chat_request(body: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(body, Mapping):
        _fail("request body must be a JSON object", "body")
    _check_known_fields(body, CHAT_SUPPORTED_FIELDS, CHAT_NEUTRAL_FIELDS)
    from clozn.server.structured_io import StructuredIOError, normalize_and_lower_messages, normalize_contract
    try:
        structured = normalize_contract(body)
    except StructuredIOError as exc:
        raise CompatibilityError(str(exc), param=exc.param, code=exc.code) from exc

    out = {key: value for key, value in body.items() if key in CHAT_SUPPORTED_FIELDS}
    for field in ("max_tokens", "max_completion_tokens", "temperature", "top_p", "seed", "stream",
                  "top_k", "repeat_penalty", "clozn_trust", "clozn_receipt", "clozn_lens", "clozn_selective",
                  "clozn_guard"):
        if out.get(field) is None:
            out.pop(field, None)

    has_tool_history = any(
        isinstance(message, Mapping)
        and (message.get("role") == "tool" or "tool_calls" in message)
        for message in (body.get("messages") if isinstance(body.get("messages"), list) else [])
    )
    if structured.get("mode") or has_tool_history:
        try:
            plan = normalize_and_lower_messages(body.get("messages"), structured)
        except StructuredIOError as exc:
            raise CompatibilityError(str(exc), param=exc.param, code=exc.code) from exc
        out["messages"] = plan["messages"]
        out["_structured_contract"] = structured
    else:
        out["messages"] = _normalize_messages(body.get("messages"))
    sources = _normalize_sources(body.get("clozn_sources"), message_count=len(out["messages"]))
    out.pop("clozn_sources", None)
    if sources:
        out["_clozn_sources"] = sources

    if structured.get("mode"):
        if sources:
            _fail("clozn_sources cannot be combined with structured I/O in v1", "clozn_sources",
                  code="unsupported_parameter")
        for extension in ("clozn_trust", "clozn_receipt", "clozn_lens", "clozn_selective", "clozn_guard"):
            if body.get(extension):
                _fail(f"{extension} cannot be combined with structured I/O in v1", extension,
                      code="unsupported_parameter")

    # Neutral structured fields remain compatibility no-ops, just as before.  Active
    # fields are represented by the private normalized contract consumed by the route.
    for field in ("tools", "tool_choice", "parallel_tool_calls", "response_format"):
        out.pop(field, None)
    if "model" in out and (not isinstance(out["model"], str) or not out["model"].strip()):
        _fail("model must be a non-empty string", "model")
    if "stream" in out and not isinstance(out["stream"], bool):
        _fail("stream must be a boolean", "stream")

    old_max = _positive_int(body, "max_tokens")
    new_max = _positive_int(body, "max_completion_tokens")
    if old_max is not None and new_max is not None:
        _fail("use only one of max_tokens or max_completion_tokens", "max_completion_tokens")
    out.pop("max_completion_tokens", None)
    if new_max is not None:
        out["max_tokens"] = new_max
    elif old_max is not None:
        out["max_tokens"] = old_max

    temperature = _number_in(body, "temperature", 0.0, 2.0)
    top_p = _number_in(body, "top_p", 0.0, 1.0, low_inclusive=False)
    if temperature is not None:
        out["temperature"] = temperature
    if top_p is not None:
        out["top_p"] = top_p
    if "seed" in out and out["seed"] is not None and not _is_int(out["seed"]):
        _fail("seed must be an integer", "seed")
    if "top_k" in out:
        if not _is_int(out["top_k"]) or out["top_k"] < 0:
            _fail("top_k must be a non-negative integer", "top_k")
    if "repeat_penalty" in out:
        if (not _is_number(out["repeat_penalty"]) or not math.isfinite(float(out["repeat_penalty"]))
                or float(out["repeat_penalty"]) <= 0):
            _fail("repeat_penalty must be a positive number", "repeat_penalty")
        out["repeat_penalty"] = float(out["repeat_penalty"])
    for field in ("clozn_trust", "clozn_receipt", "clozn_selective"):
        if field in out and not isinstance(out[field], bool):
            _fail(f"{field} must be a boolean", field)
    if "clozn_lens" in out and not isinstance(out["clozn_lens"], (bool, dict)):
        _fail("clozn_lens must be a boolean or object", "clozn_lens")
    # clozn_guard's own semantics (concepts/threshold/counter_strength/max_fires validation) are the
    # generation_guard module's job (it needs to report which sub-field is wrong, and a server-wide
    # default spec bypasses this normalizer entirely) -- this is only a wire-shape gate, matching
    # clozn_lens's own boolean-or-object split.
    if "clozn_guard" in out and not isinstance(out["clozn_guard"], (bool, dict)):
        _fail("clozn_guard must be an object", "clozn_guard")
    return out


def normalize_completion_request(body: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(body, Mapping):
        _fail("request body must be a JSON object", "body")
    _check_known_fields(body, COMPLETION_SUPPORTED_FIELDS, COMPLETION_NEUTRAL_FIELDS)
    out = {key: value for key, value in body.items() if key in COMPLETION_SUPPORTED_FIELDS}
    for field in ("max_tokens", "temperature", "top_p", "seed", "stream", "top_k", "repeat_penalty",
                  "rep_penalty"):
        if out.get(field) is None:
            out.pop(field, None)

    if not isinstance(out.get("prompt"), str):
        _fail("prompt must be a string (prompt arrays and token arrays are unsupported)", "prompt")
    if "model" in out and (not isinstance(out["model"], str) or not out["model"].strip()):
        _fail("model must be a non-empty string", "model")
    if "stream" in out and not isinstance(out["stream"], bool):
        _fail("stream must be a boolean", "stream")
    maximum = _positive_int(body, "max_tokens")
    if maximum is not None:
        out["max_tokens"] = maximum
    temperature = _number_in(body, "temperature", 0.0, 2.0)
    top_p = _number_in(body, "top_p", 0.0, 1.0, low_inclusive=False)
    if temperature is not None:
        out["temperature"] = temperature
    if top_p is not None:
        out["top_p"] = top_p
    if "seed" in out and out["seed"] is not None and not _is_int(out["seed"]):
        _fail("seed must be an integer", "seed")
    if "top_k" in out and (not _is_int(out["top_k"]) or out["top_k"] < 0):
        _fail("top_k must be a non-negative integer", "top_k")
    if "repeat_penalty" in out and "rep_penalty" in out:
        _fail("use only one of repeat_penalty or rep_penalty", "repeat_penalty")
    if "repeat_penalty" in out:
        out["rep_penalty"] = out.pop("repeat_penalty")
    if "rep_penalty" in out:
        if (not _is_number(out["rep_penalty"]) or not math.isfinite(float(out["rep_penalty"]))
                or float(out["rep_penalty"]) <= 0):
            _fail("repeat_penalty must be a positive number", "repeat_penalty")
        out["rep_penalty"] = float(out["rep_penalty"])
    return out
