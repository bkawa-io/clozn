"""Versioned evidence rules over ``clozn.performance-trace.v1``.

Every report contains all seven named rules. ``fired`` means the required measured evidence was present
and crossed the versioned threshold; ``not_fired`` means it was present and checked; ``unavailable``
means it was absent. The distinction prevents missing instrumentation from reading as a clean bill of
health. No rule changes settings, and no correlation-only rule is upgraded to causal support.

Slow-decode baselines match model identity plus recorded backend/offload shape, so CPU and GPU runs are
not mixed. Large-context prefers the measured prefill phase and retains a clearly labeled legacy
non-decode fallback for old workers. Startup, queue, stream-flush, and adapter rules only evaluate when
their corresponding measured phase exists.
"""
from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

from clozn import schemas

RULE_VERSION = "1"

# Runs derived from another run (replay/branch/fork) are not independent peer evidence -- mirrors
# clozn.runs.diagnosis's own _DERIVED_SOURCES (kept as a private local copy rather than importing a
# private name across modules; each module here owns its own tiny constants, matching this codebase's
# existing convention -- see e.g. clozn.runs.telemetry's own _TIMING_META_KEYS).
_DERIVED_SOURCES = frozenset({"replay", "branch", "fork"})

# Thresholds, named and documented so they're easy to find and reconsider -- stored alongside
# RULE_VERSION rather than hidden inside the rule bodies.
_CONTEXT_OCCUPANCY_THRESHOLD = 0.6     # prompt_tokens / context_window_tokens
_NONDECODE_SHARE_THRESHOLD = 0.3       # (wall_clock_ms - decode_ms) / wall_clock_ms
_SLOW_DECODE_RATIO_THRESHOLD = 0.6     # this_run_tok_s <= peer_median_tok_s * this ratio
_MIN_PEER_RUNS = 2                     # fewer than this many comparable runs -> no baseline yet
_QUEUE_SHARE_THRESHOLD = 0.1           # measured queue wait / request wall
_BACKPRESSURE_SHARE_THRESHOLD = 0.1    # measured write+flush blocking / request wall


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


def _evidence(path: str, value: Any) -> dict:
    return {"path": path, "value": value}


def _entry(rule: str, *, status: str, reason: str | None = None, evidence_state: str | None = None,
           likely_cause: str | None = None, possible_fix: str | None = None, evidence=None) -> dict:
    out: dict[str, Any] = {"rule": rule, "rule_version": RULE_VERSION, "status": status}
    if reason:
        out["reason"] = reason
    if evidence_state:
        out["evidence_state"] = evidence_state
    if likely_cause:
        out["likely_cause"] = likely_cause
    if possible_fix:
        out["possible_fix"] = possible_fix
    if evidence:
        out["evidence"] = list(evidence)
    return out


def _measured_phase(run: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    """First measured phase named ``name`` from the derived trace, or None.

    Going through perf_trace centralizes validation of units and clock ownership. A malformed timing
    document therefore makes only this rule unavailable; it never breaks the report.
    """
    try:
        from clozn.runs.perf_trace import build_trace
        for phase in build_trace(run).get("phases", []):
            if phase.get("name") == name and phase.get("measurement", "measured") == "measured":
                return phase
    except Exception:
        pass
    return None


def _phase_ms(run: Mapping[str, Any], name: str) -> float | None:
    phase = _measured_phase(run, name)
    duration_ns = _integer(_object(phase).get("duration_ns")) if phase is not None else None
    return None if duration_ns is None else duration_ns / 1_000_000


# ------------------------------------------------------------------------------------- large_context

def _large_context(run: Mapping[str, Any], related_runs) -> dict:
    limits = _object(_object(run.get("context_receipt")).get("limits"))
    prompt_tokens = _integer(limits.get("prompt_tokens"))
    context_window = _integer(limits.get("context_window_tokens"), minimum=1)
    if prompt_tokens is None or context_window is None:
        return _entry(
            "large_context", status="unavailable",
            reason="context_receipt.limits.prompt_tokens/context_window_tokens were not both recorded "
                   "on this run")

    timing = _object(run.get("timing"))
    meta = _object(run.get("meta"))
    wall_ms = _number(timing.get("duration_ms"))
    prefill_ms = _phase_ms(run, "prefill")
    decode_ms = _number(meta.get("generation_duration_ms"))
    if wall_ms is None or (prefill_ms is None and decode_ms is None):
        return _entry(
            "large_context", status="unavailable",
            reason="timing.duration_ms and neither a measured prefill phase nor a legacy decode duration "
                   "were recorded")

    occupancy = prompt_tokens / context_window
    if prefill_ms is not None:
        prompt_share = (prefill_ms / wall_ms) if wall_ms > 0 else 0.0
        prompt_path = "phases.prefill.duration_ms"
        prompt_duration_ms = prefill_ms
    else:
        nondecode_ms = max(0.0, wall_ms - decode_ms)
        prompt_share = (nondecode_ms / wall_ms) if wall_ms > 0 else 0.0
        prompt_path = "derived.non_decode_duration_ms"
        prompt_duration_ms = nondecode_ms
    evidence = [
        _evidence("context_receipt.limits.prompt_tokens", prompt_tokens),
        _evidence("context_receipt.limits.context_window_tokens", context_window),
        _evidence("timing.duration_ms", wall_ms),
        _evidence(prompt_path, round(prompt_duration_ms, 3)),
    ]
    if occupancy >= _CONTEXT_OCCUPANCY_THRESHOLD and prompt_share >= _NONDECODE_SHARE_THRESHOLD:
        attribution = (
            "measured prompt prefill"
            if prefill_ms is not None
            else "legacy non-decode time (this worker does not yet time prefill separately)"
        )
        return _entry(
            "large_context", status="fired", evidence_state="correlated",
            likely_cause=(
                f"The prompt used {occupancy * 100:.0f}% of the context window, and {attribution} "
                f"accounted for {prompt_share * 100:.0f}% of request wall time."),
            possible_fix="Reduce the prompt size or context window, or inspect what's being included "
                         "in context.",
            evidence=evidence)
    return _entry(
        "large_context", status="not_fired",
        reason=(f"the prompt used {occupancy * 100:.0f}% of the context window and prompt-processing "
                f"time was {prompt_share * 100:.0f}% of the request -- below this rule's thresholds "
                f"({_CONTEXT_OCCUPANCY_THRESHOLD * 100:.0f}% / {_NONDECODE_SHARE_THRESHOLD * 100:.0f}%)"),
        evidence=evidence)


# --------------------------------------------------------------------------------------- slow_decode

def _peer_decode_rates(run: Mapping[str, Any], related_runs) -> list[float]:
    """Recorded generation_tokens_per_second from other, non-derived runs sharing this run's model
    identity -- preferring the exact model_sha256 when both runs have one, else the model label."""
    this_id = run.get("id")
    this_identity = _object(run.get("identity"))
    this_sha = this_identity.get("model_sha256") if isinstance(this_identity.get("model_sha256"), str) else None
    this_model = run.get("model") if isinstance(run.get("model"), str) and run.get("model") else None
    this_meta = _object(run.get("meta"))

    rates: list[float] = []
    for candidate in related_runs if isinstance(related_runs, Sequence) else ():
        candidate = _object(candidate)
        if not candidate or candidate.get("id") == this_id:
            continue
        if candidate.get("source") in _DERIVED_SOURCES:
            continue
        candidate_identity = _object(candidate.get("identity"))
        candidate_sha = (candidate_identity.get("model_sha256")
                         if isinstance(candidate_identity.get("model_sha256"), str) else None)
        if this_sha and candidate_sha:
            if candidate_sha != this_sha:
                continue
        elif this_model:
            if candidate.get("model") != this_model:
                continue
        else:
            continue      # this run identifies no model at all -- nothing safe to compare against
        candidate_meta = _object(candidate.get("meta"))
        # A model-relative baseline also keeps the backend/offload shape fixed when this run recorded
        # it. Comparing a CPU run to CUDA peers would manufacture a "regression" from configuration.
        comparable = True
        for key in ("device", "gpu_layers", "quant", "backend"):
            if key in this_meta and candidate_meta.get(key) != this_meta.get(key):
                comparable = False
                break
        if not comparable:
            continue
        rate = _number(candidate_meta.get("generation_tokens_per_second"))
        if rate is not None and rate > 0:
            rates.append(rate)
    return rates


def _slow_decode(run: Mapping[str, Any], related_runs) -> dict:
    meta = _object(run.get("meta"))
    this_rate = _number(meta.get("generation_tokens_per_second"))
    if this_rate is None:
        return _entry(
            "slow_decode", status="unavailable",
            reason="meta.generation_tokens_per_second was not recorded on this run")

    peer_rates = _peer_decode_rates(run, related_runs)
    if len(peer_rates) < _MIN_PEER_RUNS:
        return _entry(
            "slow_decode", status="unavailable",
            reason=(f"fewer than {_MIN_PEER_RUNS} other recorded runs share this run's model identity "
                    f"with a recorded decode rate ({len(peer_rates)} found), so there is no baseline to "
                    f"compare against yet"))

    peer_median = statistics.median(peer_rates)
    evidence = [
        _evidence("meta.generation_tokens_per_second", this_rate),
        _evidence("related_runs[*].meta.generation_tokens_per_second (peer median, n="
                  f"{len(peer_rates)})", round(peer_median, 3)),
    ]
    if peer_median > 0 and this_rate <= peer_median * _SLOW_DECODE_RATIO_THRESHOLD:
        return _entry(
            "slow_decode", status="fired", evidence_state="correlated",
            likely_cause=(
                f"Decode ran at {this_rate:.1f} tok/s, well below the {peer_median:.1f} tok/s median of "
                f"{len(peer_rates)} other recorded runs on the same model."),
            possible_fix="Check for GPU contention from another process, or whether this run fell back "
                         "to a slower backend or precision.",
            evidence=evidence)
    return _entry(
        "slow_decode", status="not_fired",
        reason=(f"decode ran at {this_rate:.1f} tok/s against a peer median of {peer_median:.1f} tok/s "
                f"(n={len(peer_rates)}) -- not below this rule's threshold"),
        evidence=evidence)


def _cold_model_load(run: Mapping[str, Any], related_runs) -> dict:
    phase = _measured_phase(run, "model_load")
    if phase is None:
        return _entry(
            "cold_model_load", status="unavailable",
            reason="no measured model_load phase was recorded on this run")
    duration_ms = _integer(phase.get("duration_ns")) / 1_000_000
    evidence = [_evidence("phases.model_load.duration_ms", round(duration_ms, 3))]
    if duration_ms > 0:
        return _entry(
            "cold_model_load", status="fired", evidence_state="observed",
            likely_cause=f"This run was associated with a {duration_ms:.0f} ms worker model load.",
            possible_fix="Keep the worker warm between requests or start it before latency-sensitive work.",
            evidence=evidence)
    return _entry(
        "cold_model_load", status="not_fired",
        reason="the worker explicitly recorded a zero-duration model_load phase", evidence=evidence)


def _client_backpressure(run: Mapping[str, Any], related_runs) -> dict:
    flush_ms = _phase_ms(run, "stream_flush")
    wall_ms = _number(_object(run.get("timing")).get("duration_ms"))
    if flush_ms is None or wall_ms is None:
        return _entry(
            "client_backpressure", status="unavailable",
            reason="a measured stream_flush phase and timing.duration_ms were not both recorded")
    share = flush_ms / wall_ms if wall_ms > 0 else 0.0
    evidence = [
        _evidence("phases.stream_flush.duration_ms", round(flush_ms, 3)),
        _evidence("timing.duration_ms", wall_ms),
    ]
    if share >= _BACKPRESSURE_SHARE_THRESHOLD:
        return _entry(
            "client_backpressure", status="fired", evidence_state="correlated",
            likely_cause=(
                f"Gateway writes/flushes blocked for {share * 100:.0f}% of request wall time. The "
                "span overlaps worker progress, so this is correlated with a slow reader, not proof."
            ),
            possible_fix="Read streamed chunks promptly or avoid a slow proxy between the client and gateway.",
            evidence=evidence)
    return _entry(
        "client_backpressure", status="not_fired",
        reason=f"stream writes/flushes used {share * 100:.1f}% of wall time, below the rule threshold",
        evidence=evidence)


def _queue_contention(run: Mapping[str, Any], related_runs) -> dict:
    queue_ms = _phase_ms(run, "gateway_queue")
    wall_ms = _number(_object(run.get("timing")).get("duration_ms"))
    if queue_ms is None or wall_ms is None:
        return _entry(
            "queue_contention", status="unavailable",
            reason="a measured gateway_queue phase and timing.duration_ms were not both recorded")
    share = queue_ms / wall_ms if wall_ms > 0 else 0.0
    evidence = [
        _evidence("phases.gateway_queue.duration_ms", round(queue_ms, 3)),
        _evidence("timing.duration_ms", wall_ms),
    ]
    if share >= _QUEUE_SHARE_THRESHOLD:
        return _entry(
            "queue_contention", status="fired", evidence_state="observed",
            likely_cause=f"The request spent {share * 100:.0f}% of its wall time waiting at the gateway.",
            possible_fix="Reduce concurrent generations, increase worker capacity, or schedule this run later.",
            evidence=evidence)
    return _entry(
        "queue_contention", status="not_fired",
        reason=f"gateway queue wait used {share * 100:.1f}% of wall time, below the rule threshold",
        evidence=evidence)


def _adapter_reload(run: Mapping[str, Any], related_runs) -> dict:
    phase = _measured_phase(run, "adapter_attach")
    if phase is None:
        return _entry(
            "adapter_reload", status="unavailable",
            reason="no measured adapter_attach phase was recorded on this run")
    duration_ms = _integer(phase.get("duration_ns")) / 1_000_000
    evidence = [_evidence("phases.adapter_attach.duration_ms", round(duration_ms, 3))]
    if duration_ms > 0:
        return _entry(
            "adapter_reload", status="fired", evidence_state="observed",
            likely_cause=f"An adapter attach added {duration_ms:.0f} ms of process-startup work.",
            possible_fix="Reuse a worker with the adapter already attached instead of restarting it.",
            evidence=evidence)
    return _entry(
        "adapter_reload", status="not_fired",
        reason="the worker explicitly recorded a zero-duration adapter_attach phase", evidence=evidence)


def _memory_pressure(run: Mapping[str, Any], related_runs) -> dict:
    """The one always-unavailable-today rule with a real evidence check behind it: clozn.runs.diagnosis
    already reads meta.cpu_spill_bytes/meta.cpu_spill (its own `cpu_spill` finding), so this rule reuses
    that exact, already-established field contract rather than inventing a new one. No production code
    path writes either field yet (same survey finding as the rest of this module), so in practice this
    also reports `unavailable` against real traffic today -- but unlike cold_model_load/
    client_backpressure/queue_contention/adapter_reload, it will correctly fire or not_fire the moment
    a writer exists, with no further change needed here."""
    meta = _object(run.get("meta"))
    spill_bytes = _integer(meta.get("cpu_spill_bytes"))
    spill_flag = meta.get("cpu_spill") if isinstance(meta.get("cpu_spill"), bool) else None
    if spill_bytes is None and spill_flag is None:
        return _entry(
            "memory_pressure", status="unavailable",
            reason="meta.cpu_spill_bytes/meta.cpu_spill are read by clozn.runs.diagnosis but no code "
                   "path writes them yet")

    evidence = []
    if spill_bytes is not None:
        evidence.append(_evidence("meta.cpu_spill_bytes", spill_bytes))
    if spill_flag is not None:
        evidence.append(_evidence("meta.cpu_spill", spill_flag))
    spilled = (spill_bytes or 0) > 0 or spill_flag is True
    if spilled:
        return _entry(
            "memory_pressure", status="fired", evidence_state="observed",
            likely_cause="The worker explicitly recorded a CPU/offload spill during this run.",
            possible_fix="Select a smaller quant/model, or reduce concurrent memory pressure on this "
                         "machine.",
            evidence=evidence)
    return _entry(
        "memory_pressure", status="not_fired",
        reason="the worker explicitly recorded no CPU spill", evidence=evidence)


# Fixed, documented order -- every real run gets exactly these seven entries, always, regardless of which
# ones could fire. See the module docstring for why silence here would be worse than an honest list of
# unavailables.
_RULES = (
    _large_context,
    _slow_decode,
    _cold_model_load,
    _client_backpressure,
    _queue_contention,
    _adapter_reload,
    _memory_pressure,
)


def diagnose_performance(run, related_runs=()) -> list[dict]:
    """One entry per named rule, always all seven, in the fixed order above. `related_runs` is only
    consulted by slow_decode; other rules ignore it. Never raises: every coercion helper above degrades
    a malformed field to None/absent rather than throwing, matching clozn.runs.diagnosis's own style."""
    record = run if isinstance(run, Mapping) else {}
    return [rule(record, related_runs) for rule in _RULES]


def build_performance_report(run, related_runs=()) -> dict:
    """clozn.performance-trace.v1 document for `run`, with `diagnoses` attached -- the one-call entry
    point CLI/route callers should use. Equivalent to composing
    clozn.runs.perf_trace.build_trace()/attach_diagnoses() with diagnose_performance() by hand."""
    from clozn.runs import perf_trace

    trace = perf_trace.build_trace(run)
    diagnoses = diagnose_performance(run, related_runs)
    report = perf_trace.attach_diagnoses(trace, diagnoses)
    fired = [entry["rule"] for entry in diagnoses if entry.get("status") == "fired"]
    evaluable = [entry for entry in diagnoses if entry.get("status") != "unavailable"]
    report["regression_attribution"] = {
        "status": "attributed" if fired else ("not_detected" if evaluable else "unavailable"),
        "rules": fired,
        "evaluable_rule_count": len(evaluable),
        "evidence_state": "correlated" if fired else "observed",
    }
    schemas.validate(report, perf_trace.SCHEMA)
    return report
