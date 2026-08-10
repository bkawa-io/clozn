"""Hard signals -- the "something is actually OFF" facts a run's footer flags (AMBIENT_DELIVERY.md).

These are the ONLY things worth a flag beyond a close call, and every one is a hard fact or a named check
that actually ran -- never a proxy/vibe (the fragility/stability terminology's binding rule). All free
from the recorded run, no model call. High precision by design: we would rather miss a soft problem than
raise a false one, so only unambiguous facts are here (fuzzy refusal detection / runtime-integrity
comparison are deliberately left out until they can be done without false alarms).

Signals (each a human phrase): errored · truncated (hit the token limit) · got stuck repeating (a real
degeneracy loop, via runs/degeneracy.detect_loop) · empty reply · a fenced JSON block that doesn't
parse (real verification -- a check ran and failed).
"""
from __future__ import annotations

import json
import re

from clozn.runs.degeneracy import detect_loop

# mirrors clozn/runs/actuary.py's machine-source set -- studio probes, not user turns.
_MACHINE_SOURCES = {"replay", "branch", "fork", "receipt", "receipts", "counterfactual", "rederive",
                    "swap_receipt", "anchored_receipt", "experiment"}

# capture the WHOLE fenced block body (not just a {...}) so trailing junk before the close fence -- a
# comment, a stray line -- makes the parse fail and the check fire, instead of matching only the valid
# prefix and silently passing.
_FENCE = re.compile(r"```(json)?\s*\n(.*?)```", re.S)


# -------------------------------------------------------------------------------------- Turn Receipt v1
#
# This registry is deliberately kept beside the legacy hard-signal detector.  The receipt builder and
# the in-band footer both consume these same codes; a new fact therefore gets one detector and one
# priority order, rather than two subtly different ambient/read-side implementations.
SIGNAL_LEVELS = frozenset({"attention", "info"})
SIGNAL_PRIORITY = (
    "run_errored",
    "answer_truncated",
    "invalid_structured_output",
    "repetition_detected",
    "empty_reply",
    "context_omitted",
    "context_tension_detected",
    "context_window_pressure",
)

# These are code-owned footer phrases.  They intentionally contain no source labels, prompt snippets,
# model output, or other persisted user-controlled text.
FOOTER_PHRASES = {
    "run_errored": "run errored",
    "answer_truncated": "cut off at token limit",
    "invalid_structured_output": "invalid structured output",
    "repetition_detected": "repetition detected",
    "empty_reply": "empty reply",
    "context_omitted": "context omissions detected",
    "context_tension_detected": "competing measured context effects",
    "context_window_pressure": None,       # rendered from the factual occupancy percentage
}

_SIGNAL_EVIDENCE = {
    "run_errored": "run",
    "answer_truncated": "outcome",
    "invalid_structured_output": "output_contract",
    "repetition_detected": "trace",
    "empty_reply": "response",
    "context_omitted": "context_receipt",
    "context_tension_detected": "context_tension",
    "context_window_pressure": "context",
    "influence_coverage_partial": "context_utilization",
    "measured_effect_sparse": "context_utilization",
    "first_divergence_available": "first_divergence_view",
}


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _number(value, *, minimum=0.0):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (result >= minimum) or result in (float("inf"), float("-inf")):
        return None
    return result


def _signal(code: str, level: str, summary: str, *, evidence: str | None = None, **extra) -> dict:
    item = {
        "code": code,
        "level": level,
        "summary": summary,
        "evidence": evidence or _SIGNAL_EVIDENCE.get(code, code),
    }
    item.update(extra)
    return item


def _structured_output_is_invalid(run: dict) -> bool:
    """Return true only when durable structured-output evidence supports that fact.

    The structured pipeline records parser/qualification evidence on the run.  Older runs may not have
    that field, so the historical malformed-fenced-JSON check remains a narrow fallback.  This helper
    never parses a new model response and never calls a structured-output worker.
    """
    contract = _mapping(run.get("output_contract"))
    status = contract.get("status")
    outcome = _mapping(contract.get("outcome"))
    if status in {"error", "invalid", "failed", "malformed"} or outcome.get("status") in {
        "error", "invalid", "failed", "malformed",
    }:
        return True
    if contract.get("valid") is False or contract.get("parse_valid") is False:
        return True
    if any(key in contract for key in ("parse_error", "validation_error", "unexpected_fields")):
        return True
    error = _mapping(contract.get("error"))
    if error:
        return True

    # A persisted structured evidence record plus a run error is the strong legacy signal for the
    # failure path where the parser threw before a status field was added.
    if contract.get("schema") == "clozn.structured_io.v1" and run.get("error"):
        return True

    response = run.get("response")
    reply = response if isinstance(response, str) else ""
    for match in _FENCE.finditer(reply):
        language, body = match.group(1), (match.group(2) or "").strip()
        if body and (language == "json" or body[0] in "{["):
            try:
                json.loads(body)
            except Exception:
                return True
    return False


def _repetition_detected(run: dict) -> bool:
    trace = _mapping(run.get("trace"))
    tokens = trace.get("tokens")
    response = run.get("response")
    pieces = tokens if isinstance(tokens, list) and tokens else (
        response.split() if isinstance(response, str) and response else []
    )
    try:
        return bool(pieces) and bool(detect_loop(pieces, window=8))
    except Exception:
        return False


def build_structured_signals(run: dict | None, *, outcome=None, context=None,
                             what_mattered=None, context_tension=None, comparison=None) -> list[dict]:
    """Build the canonical Turn Receipt signal list from recorded evidence only.

    ``outcome``, ``context``, ``what_mattered``, ``context_tension`` and ``comparison`` are optional
    already-composed projections.  Supplying them keeps this registry independent of the Turn Receipt
    module while ensuring the footer can consume the exact same structured facts.  The function is
    deterministic, defensive, and never mutates ``run``.
    """
    if not isinstance(run, dict):
        return []
    outcome = _mapping(outcome)
    context = _mapping(context)
    what_mattered = _mapping(what_mattered)
    context_tension = _mapping(context_tension)
    comparison = _mapping(comparison)
    signals_out: list[dict] = []

    if run.get("error") or outcome.get("state") == "errored":
        signals_out.append(_signal("run_errored", "attention", "The run recorded an error."))

    if outcome.get("state") == "truncated" or outcome.get("finish_reason") in {
        "max_tokens", "context_limit", "length",
    } or run.get("finish_reason") == "length":
        signals_out.append(_signal(
            "answer_truncated", "attention", "Generation stopped at an output or context limit.",
        ))

    if _structured_output_is_invalid(run):
        signals_out.append(_signal(
            "invalid_structured_output", "attention", "Structured-output validation evidence recorded a failure.",
        ))

    if _repetition_detected(run):
        signals_out.append(_signal(
            "repetition_detected", "attention", "A deterministic repetition check fired.",
        ))

    response_present = "response" in run
    response = run.get("response")
    if response_present and isinstance(response, str) and not response.strip() and not run.get("error"):
        signals_out.append(_signal("empty_reply", "attention", "The recorded reply was present but empty."))

    sources = _mapping(context.get("sources"))
    omitted = sources.get("omitted")
    if isinstance(omitted, int) and not isinstance(omitted, bool) and omitted > 0:
        signals_out.append(_signal(
            "context_omitted", "attention", "One or more delivered context segments did not reach assembly.",
        ))

    occupancy = _number(context.get("window_occupancy"), minimum=0.0)
    if occupancy is not None and occupancy >= 0.85:
        percentage = round(occupancy * 100)
        signals_out.append(_signal(
            "context_window_pressure", "attention",
            f"Prompt occupies {percentage}% of the context window.",
            occupancy=round(occupancy, 4), percentage=percentage,
        ))

    tension_summary = _mapping(context_tension)
    if (tension_summary.get("measurement_state") == "available"
            and isinstance(tension_summary.get("tension_pairs"), int)
            and tension_summary["tension_pairs"] > 0):
        signals_out.append(_signal(
            "context_tension_detected", "attention",
            "Competing measured context effects were detected.",
        ))

    if what_mattered.get("measurement_state") == "available":
        coverage = _mapping(what_mattered.get("coverage"))
        prompt_sources = coverage.get("prompt_sources")
        measured_sources = coverage.get("measured_sources")
        if (isinstance(prompt_sources, int) and isinstance(measured_sources, int)
                and measured_sources < prompt_sources):
            signals_out.append(_signal(
                "influence_coverage_partial", "info",
                f"Influence measured {measured_sources} of {prompt_sources} context sources.",
            ))
        effects = _mapping(what_mattered.get("effect_summary"))
        clear = effects.get("sources_with_clear_measured_effect")
        if isinstance(clear, int) and isinstance(measured_sources, int) and clear < measured_sources:
            signals_out.append(_signal(
                "measured_effect_sparse", "info",
                f"{clear} of {measured_sources} measured sources showed a clear above-floor effect.",
            ))

    divergence = _mapping(comparison.get("first_divergence"))
    if comparison.get("state") == "available" and divergence:
        signals_out.append(_signal(
            "first_divergence_available", "info",
            f"This branch first diverged at token {divergence.get('index', '?')}.",
        ))

    # The list is a stable policy surface: attention signals use the documented footer priority; info
    # signals follow their registry order.  Unknown future codes are retained after known facts.
    priority = {code: index for index, code in enumerate(SIGNAL_PRIORITY)}
    signals_out.sort(key=lambda item: (0, priority[item["code"]]) if item["code"] in priority
                    else (1, item["code"]))
    return signals_out


def is_organic(run: dict) -> bool:
    """A genuine user turn, not a studio probe (a known machine source or any derived run is not)."""
    if not isinstance(run, dict):
        return False
    if str(run.get("source") or "").lower() in _MACHINE_SOURCES:
        return False
    return not run.get("parent_run_id")


def hard_signals(run: dict | None) -> list[str]:
    """The list of hard-fact flags for this run (human phrases), or [] when nothing is off. Never raises."""
    try:
        if not isinstance(run, dict):
            return []
        out = []
        if run.get("error"):
            out.append("the run errored")
        if run.get("finish_reason") == "length":
            out.append("cut off mid-answer (hit the token limit)")
        resp = run.get("response")
        reply = str(resp) if resp is not None else ""
        trace = run.get("trace") if isinstance(run.get("trace"), dict) else {}
        toks = trace.get("tokens")
        pieces = toks if isinstance(toks, list) and toks else reply.split()
        # only flag EMPTY when the reply is present-and-empty -- an absent `response` key (a trace-only
        # fixture, a diffusion run that stores final_text elsewhere) is unknown, not empty.
        if resp is not None and not reply.strip() and not run.get("error"):
            out.append("returned an empty reply")
        elif detect_loop(pieces, window=8):
            out.append("got stuck repeating")
        for fm in _FENCE.finditer(reply):
            lang, blk = fm.group(1), (fm.group(2) or "").strip()
            if blk and (lang == "json" or blk[0] in "{["):    # a JSON block (declared, or looks like one)
                try:
                    json.loads(blk)
                except Exception:
                    out.append("the JSON block it returned doesn't parse")
                    break
        return out
    except Exception:
        return []
