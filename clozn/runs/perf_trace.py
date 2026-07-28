"""clozn/runs/perf_trace.py -- derive the clozn.performance-trace.v1 artifact from a stored run.

WHY THIS IS A DERIVED VIEW, NOT A NEW STORED FIELD
----------------------------------------------------
Like clozn.runs.diagnosis.diagnose() and clozn.runs.timeline.timeline(), build_trace() reads an
already-recorded run and reshapes it -- it does not add a new block to clozn.runs.store.record()'s
payload. Persisting a full performance_trace document on every run today would mean writing a mostly-empty
artifact to disk forever: see the module docstring below for exactly which phases this run's evidence can
and cannot support. Once real phase instrumentation lands (queue wait, model load, prefill -- see the
09-performance-diagnosis plan's Slice 6, deferred), this function's job shrinks to "pass through what was
actually captured" rather than reconstructing from scraps; it does not need to change shape to get there.

WHAT THIS RUN'S EVIDENCE CAN ACTUALLY SUPPORT TODAY
------------------------------------------------------
A repo-wide survey (see the 09-performance-diagnosis plan) found exactly one phase genuinely measured on
a production run: decode. The C++ worker times it with std::chrono::steady_clock (engine/core/src/
generate_ar.cpp) and it reaches this run's `meta` as generation_duration_ms/generation_tokens_per_second
(clozn/runs/trace.py's generation_timing_from_frames(), merged in at clozn/server/substrates.py). Model
load, prompt prefill, queue wait, and KV/context allocation are not measured anywhere in this codebase --
clozn.runs.diagnosis already reads for load_duration_ms/prefill_duration_ms/kv_allocation_ms/
context_allocation_ms and reports "unavailable" for exactly this reason. build_trace() inherits that
honesty: only the decode phase is emitted into `phases`, and the total wall-clock duration is surfaced
separately (metrics.wall_clock_total_ms) rather than dressed up as a monotonic phase -- see the schema's
own docstring for why mixing those two clock domains under one `duration_ns` array would be exactly the
kind of narration-ahead-of-evidence this feature exists to prevent.
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
    """The one phase this codebase genuinely times with a monotonic clock today -- see the module
    docstring. No `start_ns`: nothing upstream of decode is timed yet, so decode's offset within the
    request isn't known even though its own duration is (omit, don't guess)."""
    duration_ns = _duration_ns(meta.get("generation_duration_ms"))
    if duration_ns is None:
        return None
    return {"name": "decode", "owner": "clozn_worker", "duration_ns": duration_ns}


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

    # prompt_tokens_per_second needs a prefill duration this codebase does not measure yet (see module
    # docstring) -- deliberately never populated in v1, kept in the schema for forward compatibility.

    wall_clock_ms = _number(timing.get("duration_ms"))
    if wall_clock_ms is not None:
        metrics["wall_clock_total_ms"] = wall_clock_ms

    return metrics


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
    if engine_identity:
        doc["engine_identity"] = engine_identity

    phases = [phase for phase in (_decode_phase(meta),) if phase is not None]
    if phases:
        doc["phases"] = phases

    metrics = _metrics(record, meta)
    if metrics:
        doc["metrics"] = metrics

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
