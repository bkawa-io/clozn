"""Derive a clock-domain-aware performance trace from a stored run.

Worker and gateway timing documents remain separate at capture time. This derived view flattens their
phases for presentation while retaining ``clock_owner``/``clock_domain`` on each entry. Only measured,
exclusive, in-request durations contribute to known time; overlapping transport spans and process-startup
context stay visible without being double-counted.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from clozn import schemas

SCHEMA = "clozn.performance-trace.v1"

# clozn/runs/identity.py's top-level identity fields that identify the engine/model/build, as opposed to
# the machine it ran on (that's identity["ext"]["machine"], a separate namespace -- see _machine_identity).
_ENGINE_IDENTITY_KEYS = ("model_sha256", "template_fingerprint", "engine_build", "clozn_version")


class PerfTraceError(ValueError):
    """The given run cannot honestly produce a clozn.performance-trace.v1 document."""


def _object(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, *, minimum: float = 0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if result != result or result in (float("inf"), float("-inf")) or result < minimum:
        return None
    return result


def _integer(value: Any, *, minimum: int = 0) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= minimum else None


def _duration_ns(ms) -> int | None:
    value = _number(ms)
    return None if value is None else round(value * 1_000_000)


def _machine_identity(identity: Mapping[str, Any]) -> dict:
    machine = _object(_object(identity.get("ext")).get("machine"))
    return dict(machine) if machine else {}


def _engine_identity(identity: Mapping[str, Any]) -> dict:
    return {key: identity[key] for key in _ENGINE_IDENTITY_KEYS
            if key in identity and isinstance(identity[key], str) and identity[key]}


def _decode_phase(meta: Mapping[str, Any]) -> dict | None:
    """Legacy-worker fallback for builds that predate the versioned timing wrapper."""
    duration_ns = _duration_ns(meta.get("generation_duration_ms"))
    if duration_ns is None:
        return None
    # Keep the original v1 shape byte-compatible for old worker frames. The versioned worker wrapper
    # below carries the richer clock-domain/measurement labels.
    return {"name": "decode", "owner": "clozn_worker", "duration_ns": duration_ns}


def _captured_phases(meta: Mapping[str, Any]) -> list[dict]:
    phases: list[dict] = []
    for key, expected_owner in (
        ("gateway_timing", "clozn_gateway"),
        ("worker_timing", "clozn_worker"),
    ):
        document = _object(meta.get(key))
        if document.get("unit") != "nanoseconds":
            continue
        clock = document.get("clock") if isinstance(document.get("clock"), str) else "monotonic"
        clock_owner = (
            document.get("clock_owner")
            if isinstance(document.get("clock_owner"), str)
            else expected_owner
        )
        source_schema = document.get("schema_version")
        for raw in document.get("phases", []) if isinstance(document.get("phases"), list) else []:
            phase = _object(raw)
            name = phase.get("name")
            duration_ns = _integer(phase.get("duration_ns"))
            if not isinstance(name, str) or not name or duration_ns is None:
                continue
            entry = {
                "name": name,
                "owner": (
                    phase.get("owner") if isinstance(phase.get("owner"), str) else expected_owner
                ),
                "clock_owner": clock_owner,
                "clock_domain": f"{clock_owner}:{clock}",
                "duration_ns": duration_ns,
                "measurement": (
                    phase.get("measurement")
                    if phase.get("measurement") in {"measured", "estimated"}
                    else "measured"
                ),
                "aggregation": (
                    phase.get("aggregation")
                    if phase.get("aggregation") in {"exclusive", "overlapping", "context_only"}
                    else "exclusive"
                ),
            }
            start_ns = _integer(phase.get("start_ns"))
            if start_ns is not None:
                entry["start_ns"] = start_ns
            if isinstance(phase.get("scope"), str):
                entry["scope"] = phase["scope"]
            includes = phase.get("includes")
            if isinstance(includes, list):
                entry["includes"] = [value for value in includes if isinstance(value, str)]
            if isinstance(source_schema, str) and source_schema:
                entry["source_schema"] = source_schema
            phases.append(entry)
    if not any(phase["name"] == "decode" for phase in phases):
        legacy = _decode_phase(meta)
        if legacy is not None:
            phases.append(legacy)
    return phases


def _metrics(run: Mapping[str, Any], meta: Mapping[str, Any]) -> dict:
    receipt_limits = _object(_object(run.get("context_receipt")).get("limits"))
    timing = _object(run.get("timing"))
    metrics: dict[str, Any] = {}

    prompt_tokens = _integer(receipt_limits.get("prompt_tokens"))
    if prompt_tokens is not None:
        metrics["prompt_tokens"] = prompt_tokens

    generated_tokens = _integer(receipt_limits.get("generated_tokens"))
    if generated_tokens is None:
        generated_tokens = _integer(meta.get("generation_tokens"))
    if generated_tokens is not None:
        metrics["generated_tokens"] = generated_tokens

    decode_tok_s = _number(meta.get("generation_tokens_per_second"))
    if decode_tok_s is not None:
        metrics["decode_tokens_per_second"] = decode_tok_s

    prompt_tok_s = _number(meta.get("prompt_tokens_per_second"))
    if prompt_tok_s is not None:
        metrics["prompt_tokens_per_second"] = prompt_tok_s

    # prompt_tokens_per_second needs a prefill duration this codebase does not measure yet (see module
    # docstring) -- deliberately never populated in v1, kept in the schema for forward compatibility.

    wall_clock_ms = _number(timing.get("duration_ms"))
    if wall_clock_ms is not None:
        metrics["wall_clock_total_ms"] = wall_clock_ms

    return metrics


def _aggregation(phases: Sequence[Mapping[str, Any]], wall_clock_ms) -> dict:
    known_ns = sum(
        int(phase["duration_ns"])
        for phase in phases
        if phase.get("measurement", "measured") == "measured"
        and phase.get("aggregation", "exclusive") == "exclusive"
        and isinstance(phase.get("duration_ns"), int)
    )
    out: dict[str, Any] = {
        "known_duration_ns": known_ns,
        "phase_count": len(phases),
        "exclusive_phase_count": sum(
            1 for phase in phases
            if phase.get("measurement", "measured") == "measured"
            and phase.get("aggregation", "exclusive") == "exclusive"
        ),
    }
    wall_ms = _number(wall_clock_ms)
    if wall_ms is not None:
        wall_ns = round(wall_ms * 1_000_000)
        out["wall_clock_total_ns"] = wall_ns
        out["unaccounted_duration_ns"] = max(0, wall_ns - known_ns)
        out["consistency"] = "consistent" if known_ns <= wall_ns else "known_exceeds_wall"
        out["measurement_coverage"] = (
            min(1.0, known_ns / wall_ns) if wall_ns > 0 else (1.0 if known_ns == 0 else 0.0)
        )
    return out


def build_trace(run) -> dict:
    """A schema-validated clozn.performance-trace.v1 document derived from `run` (a clozn.runs.store
    record). Never null-pads: a field/phase/metric this run's evidence cannot support is simply absent.

    Raises PerfTraceError if `run` has no usable id -- the one field this artifact cannot honestly omit,
    since a trace with no run_id cannot be attributed to anything (mirrors
    clozn.runs.telemetry.TelemetryExportError's contract for the same reason).
    """
    record = _object(run)
    run_id = record.get("id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise PerfTraceError("cannot build a performance trace: run has no non-empty string id")

    identity = _object(record.get("identity"))
    meta = _object(record.get("meta"))

    doc: dict[str, Any] = {
        "schema_version": SCHEMA,
        "run_id": run_id.strip(),
        "clock": "monotonic",
    }

    machine_identity = _machine_identity(identity)
    if machine_identity:
        doc["machine_identity"] = machine_identity

    engine_identity = _engine_identity(identity)
    for key in ("backend", "device", "gpu_layers", "quant", "mode"):
        value = meta.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            engine_identity.setdefault(key, value)
    if engine_identity:
        doc["engine_identity"] = engine_identity

    phases = _captured_phases(meta)
    if phases:
        doc["phases"] = phases

    metrics = _metrics(record, meta)
    if metrics:
        doc["metrics"] = metrics
    if phases or "wall_clock_total_ms" in metrics:
        doc["aggregation"] = _aggregation(phases, metrics.get("wall_clock_total_ms"))

    schemas.validate(doc, SCHEMA)
    return doc


def attach_diagnoses(trace: dict, diagnoses: Sequence[Mapping[str, Any]]) -> dict:
    """Return a copy of `trace` with `diagnoses` attached and re-validated.

    Kept separate from build_trace() so clozn.runs.perf_diagnosis (the rule engine) stays an independent,
    optional layer: a caller that only wants phases/metrics never has to evaluate rules to get a valid
    document, and a caller that wants both composes the two explicitly.
    """
    out = dict(trace)
    entries = [dict(item) for item in diagnoses if isinstance(item, Mapping)]
    if entries:
        out["diagnoses"] = entries
    schemas.validate(out, SCHEMA)
    return out
