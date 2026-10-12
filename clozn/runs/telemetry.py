"""Privacy-safe, offline OpenTelemetry/OpenInference export for stored runs.

The public functions in this module only transform already-selected run dictionaries or write the
resulting JSONL to a local file.  There is deliberately no collector URL, HTTP client, environment
auto-discovery, or background sender here -- ``clozn runs export-otel`` writes exactly one local file and
nothing else ever leaves the machine through this module.

Each JSONL line is one OTLP/JSON ``TracesData`` object, matching the OpenTelemetry file-export format.
It contains one root span with OpenInference LLM semantic attributes.  Prompt, message, response,
reasoning, and summary text are absent by default; callers must explicitly request content via
``include_content=True`` (CLI: ``--include-content``).

Field mapping (stored run -> OTLP/OpenInference span), see :func:`run_to_span`:

=====================================  ===================================  =========================
Run field                              Span field / attribute               Notes
=====================================  ===================================  =========================
``id``                                 ``spanId`` (+ ``traceId``)           Both are a stable SHA-256
                                                                             of ``(schema, kind, id)`` --
                                                                             same run id always yields
                                                                             the same ids, no run state.
``timing.started_at``/``created_ts``   ``startTimeUnixNano``                First present value wins.
``timing.ended_at``/``duration_ms``    ``endTimeUnixNano``                  Derived if only a duration
                                                                             is recorded.
``model`` / ``meta.model_file``        ``llm.model_name``                   A local absolute path is
                                                                             reduced to its filename.
``finish_reason``                      ``llm.finish_reason``
``error`` (presence)                   ``status.code`` (2=error, 1=ok)      ``clozn.run.status`` mirrors
                                                                             this as a string.
``meta.<sampler keys>``                ``llm.invocation_parameters``        JSON-encoded object; only
                                                                             the known sampler keys.
``meta.<timing keys>``                 ``clozn.timing.<key>``               Load/prefill/eval durations.
``identity.model_sha256`` etc.         ``clozn.identity.*``                 Reproduction identity only;
                                                                             never a local file path.
``messages``/``assembled_messages``    ``llm.input_messages.<i>.message.*`` ``.role`` always included;
                                                                             ``.content`` only with
                                                                             ``include_content=True``.
``response``                           ``llm.output_messages.0.message.*``  Same content gating.
``final_prompt``/``prompt``            ``input.value``                      Only when there are no
                                                                             structured messages, and
                                                                             only with content included.
(n/a)                                  ``clozn.content.policy``             ``"omitted"``, ``"included"``,
                                                                             or ``"redacted"`` -- always
                                                                             present, so a reader can
                                                                             never mistake silence for an
                                                                             empty run.
=====================================  ===================================  =========================

Everything not listed above (reasoning traces, memory cards, tool/output-contract detail, influence maps)
is intentionally NOT exported -- OTel/OpenInference has no semantic slot for it, and guessing one would
misrepresent clozn-specific evidence as a generic LLM span.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
import math
import os
import tempfile


TRACE_SCHEMA = "clozn.otel.run.v1"
SCOPE_NAME = "clozn.runs.telemetry"

_INVOCATION_KEYS = (
    "max_tokens", "temperature", "top_p", "top_k", "seed", "repeat_penalty", "stream",
)
_TIMING_META_KEYS = (
    "load_duration_ms", "prefill_duration_ms", "prompt_eval_duration_ms",
    "generation_duration_ms", "eval_duration_ms", "generation_tokens_per_second",
)
_MESSAGE_ROLES = frozenset({"system", "developer", "user", "assistant", "tool", "function"})


class TelemetryExportError(ValueError):
    """A selected run or requested export policy cannot produce an honest span."""


def _version() -> str:
    try:
        from clozn import __version__
        return str(__version__)
    except Exception:
        return "unknown"


def _stable_id(run_id: str, kind: str, hex_chars: int) -> str:
    material = f"{TRACE_SCHEMA}\0{kind}\0{run_id}".encode("utf-8")
    value = hashlib.sha256(material).hexdigest()[:hex_chars]
    # A valid OTel trace/span id cannot be all zeroes. SHA-256 makes this effectively impossible,
    # but keep the validity guarantee explicit rather than probabilistic.
    return value if int(value, 16) else ("0" * (hex_chars - 1) + "1")


def trace_id_for_run(run_id: str) -> str:
    """Stable 16-byte OTel trace id for one journal run id."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise TelemetryExportError("run id must be a non-empty string")
    return _stable_id(run_id.strip(), "trace", 32)


def span_id_for_run(run_id: str) -> str:
    """Stable 8-byte OTel span id for one journal run id."""
    if not isinstance(run_id, str) or not run_id.strip():
        raise TelemetryExportError("run id must be a non-empty string")
    return _stable_id(run_id.strip(), "span", 16)


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _number(value, *, nonnegative: bool = False) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or (nonnegative and numeric < 0):
        return None
    return value


def _integer(value) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _first_integer(*values) -> int | None:
    return next((clean for value in values if (clean := _integer(value)) is not None), None)


def _seconds(value) -> float | None:
    number = _number(value, nonnegative=True)
    return float(number) if number is not None else None


def _iso_seconds(value) -> float | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (ValueError, OverflowError, OSError):
        return None


def _timestamps(run: Mapping) -> tuple[str, str, int | float]:
    timing = _mapping(run.get("timing"))
    start = _seconds(timing.get("started_at"))
    if start is None:
        start = _seconds(run.get("created_ts"))
    if start is None:
        start = _iso_seconds(run.get("created_at"))
    if start is None:
        start = 0.0

    end = _seconds(timing.get("ended_at"))
    duration = _number(timing.get("duration_ms"), nonnegative=True)
    if end is None and duration is not None:
        end = start + float(duration) / 1000.0
    if end is None or end < start:
        end = start
    if duration is None:
        duration = round((end - start) * 1000.0, 6)
    def nanos(seconds: float) -> str:
        try:
            return str(int(Decimal(str(seconds)) * Decimal(1_000_000_000)))
        except (InvalidOperation, ValueError):
            return "0"

    return nanos(start), nanos(end), duration


def _any_value(value) -> dict:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        # OTLP JSON follows protobuf JSON: 64-bit integers are decimal strings.
        return {"intValue": str(value)}
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TelemetryExportError("telemetry attributes must not contain non-finite numbers")
        return {"doubleValue": value}
    if isinstance(value, str):
        return {"stringValue": value}
    if isinstance(value, (list, tuple)):
        return {"arrayValue": {"values": [_any_value(item) for item in value]}}
    raise TelemetryExportError(f"unsupported telemetry attribute type: {type(value).__name__}")


def _attributes(values: Mapping[str, object]) -> list[dict]:
    return [
        {"key": key, "value": _any_value(value)}
        for key, value in sorted(values.items()) if value is not None
    ]


def _model_name(run: Mapping) -> str | None:
    meta = _mapping(run.get("meta"))
    for value in (run.get("model"), meta.get("model")):
        if isinstance(value, str) and value.strip():
            model = value.strip()
            # A stored model label is occasionally the local GGUF path. Preserve ordinary registry/HF
            # names, but never export an absolute path containing local directory or account names.
            if (model.startswith(("/", "\\"))
                    or (len(model) >= 3 and model[1] == ":" and model[2] in ("/", "\\"))):
                return model.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
            return model
    model_file = meta.get("model_file")
    if isinstance(model_file, str) and model_file.strip():
        # Export the model filename, never a local absolute path containing a username or directory.
        return model_file.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return None


def _token_counts(run: Mapping) -> tuple[int | None, int | None]:
    meta = _mapping(run.get("meta"))
    receipt = _mapping(run.get("context_receipt"))
    rendered = _mapping(receipt.get("rendered"))
    termination = _mapping(receipt.get("termination"))
    trace = _mapping(run.get("trace"))
    tokens = trace.get("tokens")
    traced = len(tokens) if isinstance(tokens, list) else None
    prompt = _first_integer(rendered.get("tokens"), meta.get("prompt_tokens"))
    completion = _first_integer(
        termination.get("generated_tokens"), meta.get("completion_tokens"),
        meta.get("generation_tokens"), traced,
    )
    return prompt, completion


def _redaction_pairs(redactions) -> tuple[tuple[str, str], ...]:
    if redactions is None:
        return ()
    if not isinstance(redactions, Mapping):
        raise TelemetryExportError("redactions must be a mapping of literal text to replacement text")
    pairs = []
    for literal, replacement in redactions.items():
        if not isinstance(literal, str) or not literal:
            raise TelemetryExportError("redaction literals must be non-empty strings")
        if not isinstance(replacement, str):
            raise TelemetryExportError("redaction replacements must be strings")
        pairs.append((literal, replacement))
    # Longest-first avoids a shorter literal consuming part of a longer one; lexical tie-breaking makes
    # the result independent of mapping insertion order.
    return tuple(sorted(pairs, key=lambda pair: (-len(pair[0]), pair[0], pair[1])))


def _content(text, redactions: tuple[tuple[str, str], ...]):
    if not isinstance(text, str):
        return None
    for literal, replacement in redactions:
        text = text.replace(literal, replacement)
    return text


def _messages(run: Mapping) -> list:
    assembled = run.get("assembled_messages")
    if isinstance(assembled, list) and assembled:
        return assembled
    messages = run.get("messages")
    return messages if isinstance(messages, list) else []


def _content_attributes(run: Mapping, *, include_content: bool,
                        redactions: tuple[tuple[str, str], ...]) -> dict:
    out: dict[str, object] = {}
    messages = _messages(run)
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            continue
        role = message.get("role")
        if role in _MESSAGE_ROLES:
            out[f"llm.input_messages.{index}.message.role"] = role
        if include_content:
            content = _content(message.get("content"), redactions)
            if content is not None:
                out[f"llm.input_messages.{index}.message.content"] = content

    if include_content and not messages:
        prompt = next((value for value in (run.get("final_prompt"), run.get("prompt"))
                       if isinstance(value, str)), None)
        if prompt is not None:
            out["input.value"] = _content(prompt, redactions)
            out["input.mime_type"] = "text/plain"

    response = run.get("response")
    if isinstance(response, str):
        out["llm.output_messages.0.message.role"] = "assistant"
        if include_content:
            out["llm.output_messages.0.message.content"] = _content(response, redactions)
    return out


def run_to_span(run: Mapping, *, include_content: bool = False,
                redactions: Mapping[str, str] | None = None) -> dict:
    """Transform one stored run into an OTLP/JSON Span dictionary.

    ``include_content=False`` is the privacy boundary. ``redactions`` are applied only after content is
    explicitly enabled; supplying them without that opt-in is rejected so a caller cannot mistakenly
    believe literals were searched while all content was silently omitted.
    """
    if not isinstance(run, Mapping):
        raise TelemetryExportError("each selected run must be an object")
    if not isinstance(include_content, bool):
        raise TelemetryExportError("include_content must be a boolean")
    pairs = _redaction_pairs(redactions)
    if pairs and not include_content:
        raise TelemetryExportError("redactions require include_content=True")
    run_id = run.get("id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise TelemetryExportError("selected run is missing a non-empty id")
    run_id = run_id.strip()

    meta = _mapping(run.get("meta"))
    identity = _mapping(run.get("identity"))
    start_ns, end_ns, duration_ms = _timestamps(run)
    prompt_tokens, completion_tokens = _token_counts(run)
    has_error = bool(run.get("error"))
    attrs: dict[str, object] = {
        "openinference.span.kind": "LLM",
        "llm.system": "clozn",
        "llm.provider": "local",
        "llm.model_name": _model_name(run),
        "llm.finish_reason": run.get("finish_reason") if isinstance(run.get("finish_reason"), str) else None,
        "llm.token_count.prompt": prompt_tokens,
        "llm.token_count.completion": completion_tokens,
        "llm.token_count.total": (
            prompt_tokens + completion_tokens
            if prompt_tokens is not None and completion_tokens is not None else None
        ),
        "clozn.run.id": run_id,
        "clozn.run.source": run.get("source") if isinstance(run.get("source"), str) else None,
        "clozn.run.client": run.get("client") if isinstance(run.get("client"), str) else None,
        "clozn.run.substrate": run.get("substrate") if isinstance(run.get("substrate"), str) else None,
        "clozn.run.status": "error" if has_error else "ok",
        "clozn.content.policy": (
            "redacted" if include_content and pairs else "included" if include_content else "omitted"
        ),
        "clozn.timing.duration_ms": duration_ms,
        "clozn.identity.model_sha256": (
            identity.get("model_sha256") if isinstance(identity.get("model_sha256"), str) else None
        ),
        "clozn.identity.template_fingerprint": (
            identity.get("template_fingerprint")
            if isinstance(identity.get("template_fingerprint"), str) else None
        ),
        "clozn.identity.engine_build": (
            identity.get("engine_build") if isinstance(identity.get("engine_build"), str) else None
        ),
    }
    invocation = {
        key: meta[key] for key in _INVOCATION_KEYS
        if key in meta and isinstance(meta[key], (str, int, float, bool)) and (
            not isinstance(meta[key], float) or math.isfinite(meta[key])
        )
    }
    if invocation:
        attrs["llm.invocation_parameters"] = json.dumps(
            invocation, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    for key in _TIMING_META_KEYS:
        value = _number(meta.get(key), nonnegative=True)
        if value is not None:
            attrs[f"clozn.timing.{key}"] = value
    attrs.update(_content_attributes(run, include_content=include_content, redactions=pairs))

    status = {"code": 2, "message": "run recorded an error"} if has_error else {"code": 1}
    return {
        "traceId": trace_id_for_run(run_id),
        "spanId": span_id_for_run(run_id),
        "name": "clozn.chat",
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": start_ns,
        "endTimeUnixNano": end_ns,
        "attributes": _attributes(attrs),
        "status": status,
    }


def _record_for_span(span: dict) -> dict:
    version = _version()
    return {
        "resourceSpans": [{
            "resource": {"attributes": _attributes({
                "service.name": "clozn",
                "service.version": version,
                "telemetry.sdk.language": "python",
                "telemetry.sdk.name": "clozn.offline-export",
            })},
            "scopeSpans": [{
                "scope": {"name": SCOPE_NAME, "version": version},
                "spans": [span],
            }],
        }],
    }


def export_runs(runs: Iterable[Mapping], *, include_content: bool = False,
                redactions: Mapping[str, str] | None = None) -> list[dict]:
    """Return one OTLP/JSON ``TracesData`` record per already-selected stored run."""
    if isinstance(runs, (str, bytes, bytearray, Mapping)) or not isinstance(runs, Iterable):
        raise TelemetryExportError("runs must be an iterable of selected run objects")
    if not isinstance(include_content, bool):
        raise TelemetryExportError("include_content must be a boolean")
    pairs = _redaction_pairs(redactions)
    if pairs and not include_content:
        raise TelemetryExportError("redactions require include_content=True")
    return [
        _record_for_span(run_to_span(run, include_content=include_content, redactions=redactions))
        for run in runs
    ]


def format_jsonl(records: Iterable[Mapping]) -> str:
    """Serialize OTLP records as deterministic UTF-8 JSON Lines text."""
    if isinstance(records, (str, bytes, bytearray, Mapping)) or not isinstance(records, Iterable):
        raise TelemetryExportError("records must be an iterable of OTLP objects")
    try:
        lines = [json.dumps(record, sort_keys=True, separators=(",", ":"),
                            ensure_ascii=False, allow_nan=False) for record in records]
    except (TypeError, ValueError) as exc:
        raise TelemetryExportError(f"telemetry record is not canonical JSON: {exc}") from exc
    return "" if not lines else "\n".join(lines) + "\n"


def write_jsonl(path: str, records: Iterable[Mapping]) -> str:
    """Atomically write local JSONL and return ``path``; performs no network operation."""
    target = os.path.abspath(os.fspath(path))
    text = format_jsonl(records)  # serialize before opening anything; failure cannot truncate target
    directory = os.path.dirname(target) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=directory, prefix=".tmp-otel-", suffix=".jsonl")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise
    return target


__all__ = [
    "SCOPE_NAME", "TRACE_SCHEMA", "TelemetryExportError", "export_runs", "format_jsonl",
    "run_to_span", "span_id_for_run", "trace_id_for_run", "write_jsonl",
]
