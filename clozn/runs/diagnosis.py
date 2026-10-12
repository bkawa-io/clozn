"""Evidence-only explanations for slow or cut-off recorded runs.

This module interprets journal fields; it does not benchmark the machine, infer
unrecorded phases, or assign causality from elapsed time alone.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


SCHEMA = "clozn.run_diagnosis.v1"
AUXILIARY_WINDOW_SECONDS = 5.0
_DERIVED_SOURCES = frozenset({"replay", "branch", "fork"})


def _object(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, *, minimum: float = 0.0) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) and result >= minimum else None


def _integer(value: Any) -> int | None:
    return int(value) if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _evidence(path: str, value: Any, meaning: str | None = None) -> dict[str, Any]:
    item = {"path": path, "value": value}
    if meaning:
        item["meaning"] = meaning
    return item


def _finding(name: str, status: str, text: str, evidence=()) -> dict[str, Any]:
    return {"id": name, "status": status, "text": text, "evidence": list(evidence)}


def _recorded_duration(run: Mapping[str, Any], names: Sequence[str]):
    for container_name in ("timing", "meta"):
        container = _object(run.get(container_name))
        for name in names:
            value = _number(container.get(name))
            if value is not None:
                return value, f"{container_name}.{name}"
    return None, None


def _decode_intervals(run: Mapping[str, Any]) -> tuple[int, int, float]:
    steps = _object(run.get("trace")).get("steps")
    if not isinstance(steps, list):
        return 0, 0, 0.0
    values = []
    for step in steps:
        value = _number(_object(step).get("dt_ms"))
        if value is not None:
            values.append(value)
    return len(values), len(steps), sum(values)


def _slow_findings(run: Mapping[str, Any]) -> list[dict[str, Any]]:
    findings = []
    timing = _object(run.get("timing"))
    total_ms = _number(timing.get("duration_ms"))
    if total_ms is None:
        findings.append(_finding(
            "total_wall_time", "unavailable",
            "The journal did not record a valid end-to-end duration for this request."))
    else:
        findings.append(_finding(
            "total_wall_time", "observed",
            f"This request took {total_ms:g} ms end to end; that total alone does not identify the slow phase.",
            [_evidence("timing.duration_ms", timing.get("duration_ms"),
                       "measured request wall-clock duration")]))

    load_ms, load_path = _recorded_duration(run, ("load_duration_ms",))
    if load_ms is None:
        findings.append(_finding(
            "model_load", "unavailable",
            "No per-request model-load duration was recorded, so model loading cannot be blamed or ruled out."))
    else:
        findings.append(_finding(
            "model_load", "observed", f"The recorded model-load phase took {load_ms:g} ms.",
            [_evidence(load_path, load_ms)]))

    prefill_ms, prefill_path = _recorded_duration(
        run, ("prefill_duration_ms", "prompt_eval_duration_ms"))
    if prefill_ms is None:
        findings.append(_finding(
            "prefill", "unavailable",
            "Prompt-prefill time was not separated from the request total."))
    else:
        findings.append(_finding(
            "prefill", "observed", f"Prompt prefill took {prefill_ms:g} ms.",
            [_evidence(prefill_path, prefill_ms)]))

    generation_ms, generation_path = _recorded_duration(
        run, ("generation_duration_ms", "eval_duration_ms"))
    if generation_ms is not None:
        findings.append(_finding(
            "generation", "observed", f"Token generation took {generation_ms:g} ms.",
            [_evidence(generation_path, generation_ms)]))
    else:
        covered, total_steps, interval_ms = _decode_intervals(run)
        if covered:
            coverage = "all" if covered == total_steps else f"{covered} of {total_steps}"
            rate = (covered * 1000.0 / interval_ms) if interval_ms > 0 else None
            rate_text = f" ({rate:.1f} recorded intervals/s)" if rate is not None else ""
            findings.append(_finding(
                "generation", "observed",
                f"Per-token evidence covers {coverage} decode intervals totaling {interval_ms:g} ms{rate_text}; "
                "it is not a complete generation-phase timer.",
                [_evidence("trace.steps[*].dt_ms", {"covered": covered, "steps": total_steps,
                                                     "sum_ms": interval_ms})]))
        else:
            token_count = _integer(_object(
                _object(run.get("context_receipt")).get("termination")
            ).get("generated_tokens"))
            evidence = ([_evidence("context_receipt.termination.generated_tokens", token_count)]
                        if token_count is not None else [])
            findings.append(_finding(
                "generation", "unavailable",
                "The journal has no generation-phase duration or per-token interval timing.", evidence))

    receipt = _object(run.get("context_receipt"))
    prompt_tokens = _integer(_object(receipt.get("rendered")).get("tokens"))
    n_ctx = _integer(receipt.get("context_window_tokens"))
    if prompt_tokens is not None and n_ctx and prompt_tokens <= n_ctx:
        ratio = prompt_tokens * 100.0 / n_ctx
        findings.append(_finding(
            "context_pressure", "observed",
            f"The prompt occupied {prompt_tokens} of {n_ctx} context tokens ({ratio:.1f}%); this measures "
            "capacity, not allocation time or a latency contribution.",
            [_evidence("context_receipt.rendered.tokens", prompt_tokens),
             _evidence("context_receipt.context_window_tokens", n_ctx)]))
    else:
        findings.append(_finding(
            "context_pressure", "unavailable",
            "Prompt-token and context-window counts were not both recorded, so context pressure is unavailable."))

    allocation_ms, allocation_path = _recorded_duration(
        run, ("kv_allocation_ms", "context_allocation_ms"))
    if allocation_ms is None:
        findings.append(_finding(
            "context_allocation", "unavailable",
            "No context/KV allocation duration was recorded."))
    else:
        findings.append(_finding(
            "context_allocation", "observed",
            f"Context/KV allocation took {allocation_ms:g} ms.",
            [_evidence(allocation_path, allocation_ms)]))

    meta = _object(run.get("meta"))
    spill_bytes = _integer(meta.get("cpu_spill_bytes"))
    spill_flag = meta.get("cpu_spill") if isinstance(meta.get("cpu_spill"), bool) else None
    if spill_bytes is not None or spill_flag is not None:
        spilled = (spill_bytes or 0) > 0 or spill_flag is True
        evidence = []
        if spill_bytes is not None:
            evidence.append(_evidence("meta.cpu_spill_bytes", spill_bytes))
        if spill_flag is not None:
            evidence.append(_evidence("meta.cpu_spill", spill_flag))
        findings.append(_finding(
            "cpu_spill", "observed" if spilled else "not_observed",
            (f"The worker recorded {spill_bytes} bytes of CPU spill." if spill_bytes is not None and spilled
             else "The worker explicitly recorded CPU spill." if spilled
             else "The worker explicitly recorded no CPU spill."), evidence))
    else:
        placement = meta.get("device")
        placement_evidence = ([_evidence("meta.device", placement,
                                         "placement does not distinguish planned CPU use from spill")]
                              if isinstance(placement, str) and placement else [])
        findings.append(_finding(
            "cpu_spill", "unavailable",
            "CPU-spill evidence was not recorded; device placement alone cannot prove a spill or its cost.",
            placement_evidence))
    return findings


def _cutoff_finding(run: Mapping[str, Any]) -> dict[str, Any]:
    finish = run.get("finish_reason")
    error = run.get("error")
    receipt = _object(run.get("context_receipt"))
    termination = _object(receipt.get("termination"))
    termination_reason = termination.get("reason")
    prompt = _integer(_object(receipt.get("rendered")).get("tokens"))
    generated = _integer(termination.get("generated_tokens"))
    context = _integer(receipt.get("context_window_tokens"))
    maximum = _integer(receipt.get("reserved_output_tokens"))
    evidence = [_evidence("finish_reason", finish)] if isinstance(finish, str) else []

    if termination_reason in {"max_tokens", "context_limit"} or finish == "length":
        if isinstance(termination_reason, str):
            evidence.append(_evidence("context_receipt.termination.reason", termination_reason))
        hit_context = (prompt is not None and generated is not None and context is not None
                       and prompt + generated >= context)
        hit_max = generated is not None and maximum is not None and generated >= maximum
        if hit_context and hit_max:
            cause = "the recorded context window and requested output limit were both reached"
        elif hit_context:
            cause = "the recorded context window was reached"
        elif hit_max:
            cause = "the requested output-token limit was reached"
        else:
            cause = "the worker reported a token-budget stop, but the journal cannot separate output from context budget"
        for path, value in (
            ("context_receipt.rendered.tokens", prompt),
            ("context_receipt.termination.generated_tokens", generated),
            ("context_receipt.context_window_tokens", context),
            ("context_receipt.reserved_output_tokens", maximum),
        ):
            if value is not None:
                evidence.append(_evidence(path, value))
        return _finding("output_cutoff", "observed", f"The reply was cut off: {cause}.", evidence)
    if error == "client_disconnected":
        return _finding(
            "output_cutoff", "observed",
            "The client disconnected before generation completed; the journal does not claim a normal stop.",
            [_evidence("error", error), _evidence("finish_reason", finish)])
    if isinstance(error, str) and error:
        return _finding(
            "output_cutoff", "observed",
            "The run ended with a recorded error rather than a normal model stop.",
            [_evidence("error", error), _evidence("finish_reason", finish)])
    if isinstance(finish, str):
        label = ("The worker recorded a normal stop, not a token-budget cutoff."
                 if finish == "stop" else
                 f"The worker recorded finish reason {finish!r}, not a token-budget cutoff.")
        return _finding("output_cutoff", "not_observed", label, evidence)
    return _finding(
        "output_cutoff", "unavailable",
        "No finish reason or terminal error was recorded, so whether the reply was cut off is unavailable.")


def _interval(run: Mapping[str, Any]) -> tuple[float, float] | None:
    timing = _object(run.get("timing"))
    started = _number(timing.get("started_at"))
    ended = _number(timing.get("ended_at"))
    if started is None:
        started = _number(run.get("created_ts"))
    if started is None:
        return None
    if ended is None or ended < started:
        ended = started
    return started, ended


def _gap_seconds(left: tuple[float, float], right: tuple[float, float]) -> float:
    if left[1] < right[0]:
        return right[0] - left[1]
    if right[1] < left[0]:
        return left[0] - right[1]
    return 0.0


def _auxiliary_finding(run: Mapping[str, Any], related_runs) -> dict[str, Any]:
    session = run.get("session_key")
    client = run.get("client_key")
    if isinstance(session, str) and session:
        association_field, association_value = "session_key", session
    elif isinstance(client, str) and client:
        association_field, association_value = "client_key", client
    else:
        return _finding(
            "client_auxiliary_calls", "unavailable",
            "This run has no exact session or client association key, so nearby calls cannot be linked safely.")
    target_interval = _interval(run)
    if target_interval is None:
        return _finding(
            "client_auxiliary_calls", "unavailable",
            "This run has an exact association key but no usable timestamp for nearby-call detection.",
            [_evidence(association_field, association_value)])

    matches = []
    for candidate in related_runs if isinstance(related_runs, Sequence) else ():
        candidate = _object(candidate)
        if not candidate or candidate.get("id") == run.get("id"):
            continue
        if candidate.get("source") in _DERIVED_SOURCES:
            continue
        if candidate.get(association_field) != association_value:
            continue
        candidate_interval = _interval(candidate)
        if candidate_interval is None:
            continue
        gap = _gap_seconds(target_interval, candidate_interval)
        if gap <= AUXILIARY_WINDOW_SECONDS:
            matches.append({"run_id": candidate.get("id"), "source": candidate.get("source"),
                            "prompt_summary": candidate.get("prompt_summary"),
                            "gap_seconds": round(gap, 6),
                            "duration_ms": _object(candidate.get("timing")).get("duration_ms")})
    if not matches:
        return _finding(
            "client_auxiliary_calls", "not_observed",
            f"No other {association_field[:-4]}-associated calls were found within "
            f"{AUXILIARY_WINDOW_SECONDS:g} seconds in the supplied related runs.",
            [_evidence(association_field, association_value)])
    return _finding(
        "client_auxiliary_calls", "observed",
        f"The journal contains {len(matches)} other call(s) with the same {association_field[:-4]} within "
        f"{AUXILIARY_WINDOW_SECONDS:g} seconds. They are possible client auxiliary calls, but their purpose "
        "and whether the client waited for them were not recorded.",
        [_evidence(association_field, association_value),
         _evidence("related_runs", matches, "separate nearby journal runs, not time charged to this run")])


def diagnose(run, related_runs=()) -> dict[str, Any]:
    """Return an honest, JSON-ready slow/cutoff diagnosis from recorded evidence only."""
    record = _object(run)
    slow = _slow_findings(record)
    observed = [finding["text"] for finding in slow if finding["status"] == "observed"]
    unavailable = [finding["id"] for finding in slow if finding["status"] == "unavailable"]
    slow_summary = " ".join(observed) if observed else "No timed request phase is available."
    if unavailable:
        slow_summary += " Unavailable: " + ", ".join(unavailable) + "."
    cutoff = _cutoff_finding(record)
    auxiliary = _auxiliary_finding(record, related_runs)
    return {
        "schema": SCHEMA,
        "run_id": record.get("id") if isinstance(record.get("id"), str) else None,
        "why_slow": {"summary": slow_summary, "findings": slow},
        "why_cut_off": {"summary": cutoff["text"], "finding": cutoff},
        "client_auxiliary_calls": auxiliary,
    }
