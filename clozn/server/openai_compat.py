"""Honest request validation for Clozn's intentionally small OpenAI-compatible surface.

Silently discarding behavior-bearing fields is dangerous for tools, stop sequences, structured output,
penalties, and ``n``: a client sees HTTP 200 and reasonably assumes the requested behavior happened.
This module is the single policy table for the public
OpenAI Chat Completions route. It accepts fields Clozn implements, strips only documented neutral/no-op
values, and raises an OpenAI-shaped 400 for every behavior-bearing field the runtime cannot honor.

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
    "stream", "stop", "stream_options", "top_k", "repeat_penalty",
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
    "logit_bias": _empty_mapping,
    "modalities": lambda v: v == ["text"],
    "audio": lambda _v: False,
    "prediction": lambda _v: False,
    "store": lambda v: v is False,
    "metadata": _empty_mapping,
    "service_tier": lambda v: v in ("auto", "default"),
    "stream_options": lambda _v: False,
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


_MESSAGE_KNOWN_FIELDS = {"role", "content"}


def _normalize_messages(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        _fail("messages must be a non-empty list of text {role, content} objects", "messages")
    out: list[dict[str, str]] = []
    for index, message in enumerate(value):
        param = f"messages[{index}]"
        if not isinstance(message, Mapping):
            _fail("each message must be an object", param)
        extra = set(message) - _MESSAGE_KNOWN_FIELDS
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
        normalized = {"role": "system" if role == "developer" else str(role), "content": content}
        out.append(normalized)
    return out

def _normalize_stop(value: Any) -> list[str] | None:
    """Validate and normalize the behavior-bearing native stop contract."""
    if value is None:
        return None
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not 1 <= len(values) <= 4:
        _fail("stop must be a non-empty string or an array of 1 to 4 strings", "stop")
    normalized: list[str] = []
    total_bytes = 0
    for index, item in enumerate(values):
        if not isinstance(item, str) or not item:
            _fail("stop sequences must be non-empty strings", f"stop[{index}]")
        size = len(item.encode("utf-8"))
        if size > 1024:
            _fail("each stop sequence must be at most 1024 UTF-8 bytes", f"stop[{index}]")
        if item in normalized:
            _fail("stop sequences must not contain duplicates", f"stop[{index}]")
        normalized.append(item)
        total_bytes += size
    if total_bytes > 4096:
        _fail("stop sequences must total at most 4096 UTF-8 bytes", "stop")
    return normalized


def _normalize_stream_options(value: Any) -> dict[str, bool] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"include_usage"}:
        _fail("stream_options supports only include_usage", "stream_options")
    if not isinstance(value["include_usage"], bool):
        _fail("stream_options.include_usage must be a boolean", "stream_options.include_usage")
    return {"include_usage": bool(value["include_usage"])}


def normalize_chat_request(body: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(body, Mapping):
        _fail("request body must be a JSON object", "body")
    _check_known_fields(body, CHAT_SUPPORTED_FIELDS, CHAT_NEUTRAL_FIELDS)
    out = {key: value for key, value in body.items() if key in CHAT_SUPPORTED_FIELDS}
    for field in ("max_tokens", "max_completion_tokens", "temperature", "top_p", "seed", "stream",
                  "top_k", "repeat_penalty"):
        if out.get(field) is None:
            out.pop(field, None)
    out["messages"] = _normalize_messages(body.get("messages"))

    if "model" in out and (not isinstance(out["model"], str) or not out["model"].strip()):
        _fail("model must be a non-empty string", "model")
    if "stream" in out and not isinstance(out["stream"], bool):
        _fail("stream must be a boolean", "stream")
    if "stop" in body:
        normalized_stop = _normalize_stop(body.get("stop"))
        if normalized_stop is None:
            out.pop("stop", None)
        else:
            out["stop"] = normalized_stop
    if "stream_options" in body:
        stream_options = _normalize_stream_options(body.get("stream_options"))
        if stream_options is not None:
            out["stream_options"] = stream_options

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
    return out
