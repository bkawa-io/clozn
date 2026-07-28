"""clozn/runs/perf_diagnosis.py -- the versioned rule engine over clozn.performance-trace.v1 evidence.

Distinct from (and does not replace) clozn.runs.diagnosis, which already renders honest per-phase
observed/unavailable findings and a why-slow/why-cut-off narrative. This module adds the one thing that
was missing on top of it: the spec's seven named rules (cold_model_load, large_context, slow_decode,
client_backpressure, queue_contention, adapter_reload, memory_pressure), each producing a single
"likely cause + possible fix" pair naming the evidence that triggered it -- never auto-changing settings.

THE BINDING DISCIPLINE: A SURVEY FINDING THAT REFRAMED THIS MODULE
----------------------------------------------------------------------
A repo-wide grep for the fields these rules would need (load_duration_ms, prefill/prompt_eval duration,
kv/context allocation duration, per-request queue-wait duration, adapter-reload evidence, client
stream-backpressure timestamps) turns up no production writer for any of them -- only
clozn.runs.diagnosis's reader, the OTel exporter, and synthetic test fixtures. So today, only two of the
seven rules (large_context, slow_decode) can genuinely fire against real traffic; the other five
structurally cannot, for lack of instrumentation this cycle deliberately does not add (see the
09-performance-diagnosis plan's Slice 6, deferred pending coordination with feature 01's engine-discovery
refactor). The failure this discipline exists to prevent: a rule engine that silently returns nothing for
five of seven rules reads to a user as "nothing wrong here," which is exactly the false reassurance this
whole feature exists to rule out. diagnose_performance() therefore ALWAYS returns one entry per rule, in a
fixed order, whether or not it could fire.

THREE STATUSES, NOT TWO
--------------------------
Each entry's `status` is one of:

  fired       -- the rule's evidence was present and its condition was met. Carries `evidence_state`
                 (never stronger than "correlated" in this module: nothing here re-runs a request with
                 one variable changed, which is the only thing that would earn "causally_supported" --
                 see roadmap rule 1), `likely_cause`, `possible_fix`, and the `evidence` that triggered it.
  not_fired   -- the rule's evidence WAS present and was evaluated, but the condition was not met: an
                 explicit, present "checked and it's clean" entry, never silence. Only large_context and
                 slow_decode can produce this today, since they are the only two with a genuine evidence
                 path on real runs.
  unavailable -- the evidence this rule needs does not exist on this run at all, almost always because no
                 code path records it yet. Carries `reason` naming exactly what's missing. This is what
                 the other five rules (and large_context/slow_decode when their specific fields are
                 absent) report on every real run today.

`not_fired` and `unavailable` are deliberately different words for a deliberately different fact:
conflating "we checked and it's fine" with "we have no way to check" would itself be a narration-ahead-
of-evidence bug -- a user seeing "large_context: unavailable" on a run with a tiny prompt and fast decode
would wrongly conclude the rule never works, when it actually ran and found nothing wrong.

HOW large_context AND slow_decode EARN REAL EVIDENCE WITHOUT NEW INSTRUMENTATION
------------------------------------------------------------------------------------
large_context: context_receipt.limits (feature 06's already-shipped artifact) gives prompt/context token
counts on real runs today, and meta.generation_duration_ms (genuinely worker-timed, see
clozn.runs.perf_trace's module docstring) lets this rule isolate a "non-decode share" of the request's
wall-clock time. That is NOT proof the non-decode time was spent on prefill specifically -- it could be
queue wait or model load too, none of which is separated yet -- so this rule is careful to say "non-decode
time" in its likely_cause text, never "prefill time", and never claims more than `correlated`.

slow_decode: rather than a stored baseline-key system (the spec's "Baselines" section, out of scope this
cycle) or the CLI's roofline throughput predictor (clozn.cli.throughput_predictor -- deliberately not
imported here: clozn/runs/ has no existing dependency on clozn/cli/, and pulling one in for a single rule
would be a new, unprecedented layering direction for one module to introduce), this rule compares the
run's own meta.generation_tokens_per_second against the median of OTHER already-journaled runs sharing its
model identity -- the exact `related_runs` pattern clozn.runs.diagnosis already established for
client_auxiliary_calls. No new storage, no cross-layer import: it is evidence already sitting in the
journal, read the same way a sibling module already reads it.
"""
from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any

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
    decode_ms = _number(meta.get("generation_duration_ms"))
    if wall_ms is None or decode_ms is None:
        return _entry(
            "large_context", status="unavailable",
            reason="timing.duration_ms and/or meta.generation_duration_ms were not recorded, so the "
                   "non-decode share of elapsed time cannot be isolated")

    occupancy = prompt_tokens / context_window
    nondecode_ms = max(0.0, wall_ms - decode_ms)
    nondecode_share = (nondecode_ms / wall_ms) if wall_ms > 0 else 0.0
    evidence = [
        _evidence("context_receipt.limits.prompt_tokens", prompt_tokens),
        _evidence("context_receipt.limits.context_window_tokens", context_window),
        _evidence("timing.duration_ms", wall_ms),
        _evidence("meta.generation_duration_ms", decode_ms),
    ]
    if occupancy >= _CONTEXT_OCCUPANCY_THRESHOLD and nondecode_share >= _NONDECODE_SHARE_THRESHOLD:
        return _entry(
            "large_context", status="fired", evidence_state="correlated",
            likely_cause=(
                f"The prompt used {occupancy * 100:.0f}% of the context window, and decode accounted "
                f"for only {(1 - nondecode_share) * 100:.0f}% of the request's total wall time -- "
                f"consistent with (not proven to be) slow prompt processing. This codebase does not "
                f"yet time prefill separately from queueing or model load, so the non-decode share "
                f"cannot be attributed to prompt evaluation alone."),
            possible_fix="Reduce the prompt size or context window, or inspect what's being included "
                         "in context.",
            evidence=evidence)
    return _entry(
        "large_context", status="not_fired",
        reason=(f"the prompt used {occupancy * 100:.0f}% of the context window and non-decode time was "
                f"{nondecode_share * 100:.0f}% of the request -- below this rule's thresholds "
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
        rate = _number(_object(candidate.get("meta")).get("generation_tokens_per_second"))
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


# ---------------------------------------------------------------- rules with no evidence path today

def _cold_model_load(run: Mapping[str, Any], related_runs) -> dict:
    return _entry(
        "cold_model_load", status="unavailable",
        reason="no model-load duration is recorded on this run; no code path in this codebase writes "
               "load_duration_ms yet (clozn.cli.engine_process.spawn_engine times its own health-poll "
               "loop but never stores the duration)")


def _client_backpressure(run: Mapping[str, Any], related_runs) -> dict:
    return _entry(
        "client_backpressure", status="unavailable",
        reason="no stream-write-vs-generation-complete timestamps are recorded; this codebase has no "
               "instrumentation on the SSE write path yet")


def _queue_contention(run: Mapping[str, Any], related_runs) -> dict:
    return _entry(
        "queue_contention", status="unavailable",
        reason="no per-request queue-wait duration is recorded on any run; "
               "clozn.server.request_gate.RequestGate tracks live active/waiting counts but does not "
               "persist them onto a run")


def _adapter_reload(run: Mapping[str, Any], related_runs) -> dict:
    return _entry(
        "adapter_reload", status="unavailable",
        reason="no adapter-load timing or repeat-load evidence is recorded on any run; adapter loading "
               "is the LoRA adapter workflow's domain and is not yet instrumented")


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
    return perf_trace.attach_diagnoses(trace, diagnoses)
