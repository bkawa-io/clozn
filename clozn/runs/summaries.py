"""Compact run summaries and list flags."""
from __future__ import annotations


# The slim fields returned by list_runs() (the Runs page doesn't need full messages/trace).
SUMMARY_FIELDS = (
    "id",
    "created_at",
    "source",
    "client",
    "client_key",
    "client_key_source",
    "session_key",
    "created_ts",
    "recorded_ts",
    "model",
    "substrate",
    "prompt_summary",
    "response_summary",
    "memory",
    "behavior",
    "timing",
    "finish_reason",
    "parent_run_id",
    "flags",
    "warnings",
    # Confidence shape, computed once by `confidence_facts()` when the run is recorded (the trace is
    # a content-addressed blob behind `trace_ref`, so deriving these at LIST time would mean one blob
    # read per row per page load). Absent on runs recorded before this existed, and on any run with
    # no trace -- see confidence_facts() for why absence is never rendered as zero.
    "token_count",
    "confidence",
    "confidence_min",
    "confidence_mean",
    "low_confidence_count",
)

# Below this the model's next token was closer to a coin flip than a choice. Matches the band the run
# reader already calls shaky, so a run cannot read as low-confidence in the index and fine on its own
# page.
_LOW_CONFIDENCE = 0.35
_SPARK_POINTS = 32

# Sent together or not at all -- see _summary() and confidence_facts().
_CONFIDENCE_FIELDS = (
    "token_count",
    "confidence",
    "confidence_min",
    "confidence_mean",
    "low_confidence_count",
)


def _summ(text: str, n: int = 220) -> str:
    """A one-line excerpt for the run index.

    Raised from 90: the Runs page uses the prompt as a card headline, and 90 characters cut most real
    prompts mid-clause, so the index read as a column of fragments. 220 is roughly the two lines the
    card clamps to at the documented 15px content size. Still a SUMMARY, never the prompt -- the full
    text lives on the run itself and nothing here should be mistaken for a replayable input.
    """
    text = (text or "").strip().replace("\n", " ")
    return text[:n] + ("…" if len(text) > n else "")


def _flags(rec: dict) -> list[str]:
    """Cheap UI flags derived from the record (the Runs page filters on these)."""
    f = []
    mem = rec.get("memory") or {}
    if mem.get("cards_applied"):
        f.append("memory")
    if mem.get("proposed_cards"):
        f.append("pending-memory")
    # ("anchored-memory" / "memory-retried" / "memory-loop-guard" were derived here from
    # memory.anchored{,_loop_guard}. Nothing writes those keys since the 2026-07-27 anchored-memory
    # removal, so the flags could only ever be false -- a filter that silently matched nothing.)
    if (rec.get("behavior") or {}).get("active_dials"):
        f.append("steered")
    if rec.get("parent_run_id"):
        f.append("replayed")
    if rec.get("error"):
        f.append("error")
    if rec.get("finish_reason") == "length":
        f.append("truncated")
    if (rec.get("reasoning") or {}).get("stripped_from_response"):
        f.append("reasoning-captured")
    output = rec.get("output_contract")
    output = output if isinstance(output, dict) else {}
    outcome = output.get("outcome")
    outcome = outcome if isinstance(outcome, dict) else {}
    if outcome.get("kind") == "tool_call" and outcome.get("status") == "parsed":
        f.append("tool-call")
    if outcome.get("status") == "error":
        f.append("output-parse-error")
    conf = (rec.get("trace") or {}).get("confidence") or []
    # Shares _LOW_CONFIDENCE with confidence_facts() on purpose. These two used to disagree (0.3 here,
    # a separate constant there), which meant a run could carry the "low-confidence" flag while the
    # index's own low_confidence_count called none of its tokens low, or the reverse.
    if conf and min(conf) < _LOW_CONFIDENCE:
        f.append("low-confidence")
    if len((rec.get("response") or "").split()) > 220:
        f.append("long")
    return f


def confidence_facts(trace: dict | None) -> dict:
    """Per-run confidence shape for the index, derived from the trace already in memory.

    Called by `store.record()` while the trace is still in hand. It cannot be derived at list time:
    the trace is a content-addressed, cross-run deduplicated blob referenced by `trace_ref`, so the
    index would pay one blob read per row per page load to recompute what is cheap to store once.

    Every key is OMITTED when the run recorded no confidence. That is deliberate and load-bearing: a
    traceless run must not arrive as `confidence: []` or `confidence_min: 0`, because zero is a real
    and terrible confidence value while absence is not a value at all. A client that receives no key
    cannot accidentally plot silence as certainty.
    """
    values = trace.get("confidence") if isinstance(trace, dict) else None
    if not isinstance(values, list):
        return {}
    numeric = [float(v) for v in values if isinstance(v, (int, float))]
    if not numeric:
        return {}

    # Downsample to at most _SPARK_POINTS. The sparkline consuming this is ~112px wide, so anything
    # finer is sub-pixel detail paid for on every row of every page load.
    if len(numeric) <= _SPARK_POINTS:
        spark = numeric
    else:
        step = (len(numeric) - 1) / (_SPARK_POINTS - 1)
        spark = [numeric[round(index * step)] for index in range(_SPARK_POINTS)]

    return {
        "token_count": len(numeric),
        "confidence": [round(value, 3) for value in spark],
        "confidence_min": round(min(numeric), 3),
        "confidence_mean": round(sum(numeric) / len(numeric), 3),
        "low_confidence_count": sum(1 for value in numeric if value < _LOW_CONFIDENCE),
    }


def _summary(r: dict) -> dict:
    """One run dict -> the compact SUMMARY_FIELDS view.

    Every field is present, EXCEPT the confidence group, which is dropped entirely when the run has
    none. That asymmetry is deliberate: the older fields have always been sent as null when absent and
    callers index them directly, so making them disappear would be a breaking change for no gain. The
    confidence group is new and has no such callers, and it is the one group where a null actively
    misleads -- `confidence_min: null` invites `?? 0`, and 0 is a real and terrible confidence rather
    than a missing one.
    """
    out = {k: r.get(k) for k in SUMMARY_FIELDS}
    for key in _CONFIDENCE_FIELDS:
        if out.get(key) is None:
            out.pop(key, None)
    return out
