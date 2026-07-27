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
)


def _summ(text: str, n: int = 90) -> str:
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
    if conf and min(conf) < 0.3:
        f.append("low-confidence")
    if len((rec.get("response") or "").split()) > 220:
        f.append("long")
    return f


def _summary(r: dict) -> dict:
    """One run dict -> the compact SUMMARY_FIELDS view."""
    return {k: r.get(k) for k in SUMMARY_FIELDS}
