"""span_receipt.py -- SPAN RECEIPTS ("injection forensics"): causally attribute one SPAN of a stored
run's own prompt/context by ablation. "Did THIS sentence in the context change the answer?" -- answered
the same two ways the existing influence receipts answer it, with the same thresholds:

  * REGEN arm (counterfactual text): rebuild the conversation with the span removed from its message and
    regenerate GREEDY through the exact replay path the existing /runs/<id>/receipt regen arm uses
    (clozn.replay.replay.replay with {"greedy": True}) -- baseline-with-span vs ablated-without-span, both
    greedy, both from the run's own stored messages. `has_effect` / `answer_changed` is the same exact
    string comparison _build_receipt uses.
  * FORCED arm (dependence): teacher-force the ORIGINAL answer tokens under the ablated context via the
    existing forced-scoring machinery (clozn.receipts.rederive.score_arm) -- per-continuation-token logprob
    deltas, summarized with forced.py's own `_delta_summary` and judged against forced.py's own thresholds
    (_FORCED_MEAN_THRESHOLD / _FORCED_SUM_THRESHOLD -- never re-invented here). The null floor replaces the
    span with register-matched filler of the SAME character length (`matched_length_neutral_filler`,
    the card_filler control's exact recipe) and applies forced.py's `_NULL_FLOOR_RATIO_MIN` ratio test.

OFFSETS ARE CHARACTER OFFSETS, NOT BYTES. `start`/`end` index Unicode code points of the message's
`content` string -- exactly Python `str` slicing (`content[start:end]`), half-open [start, end). Chosen
for JS-friendliness: a frontend that holds the same message text computes them with indexOf/slice rather
than encoding to UTF-8 first. Honest caveat: JavaScript strings count UTF-16 code units, so JS-computed
offsets diverge from code points on astral-plane characters (emoji, rare CJK extensions) -- for BMP text
(ASCII, accented Latin, standard CJK) the two are identical. The `{"find": "<substring>"}` form sidesteps
offset arithmetic entirely and is the recommended wire form for emoji-bearing contexts.

Honesty rules carried over verbatim from the sibling receipts: the regen `note` never claims beyond the
measurement ("removing this span measurably changed / did not change the answer" -- ablation-causal, not
intent); the forced arm carries forced.py's `_FORCED_CAVEAT` unchanged (a nonzero delta is dependence, not
"the answer would have differed", and certainly not "hijacked"); `silent_influence` is core.py's exact
formula (regen text unchanged AND the forced deltas clear the null floor by _NULL_FLOOR_RATIO_MIN).

Never raises into the caller except `SpanSpecError` (an invalid span spec -- the route's 400); a
generation/scoring failure returns None (regen, mirrors receipts.receipt) or a degraded
`causal_verified: False` forced sub-dict (mirrors forced.forced_receipt). Duck-typed against
`sub.chat` / `sub.score_tokens` like replay.py / rederive.py -- fully unit-testable with a fake substrate,
no model, no GPU, no server.
"""
from __future__ import annotations

from clozn.replay.replay import replay as replay_run

from . import rederive
from .deltas import _NOTE_BASELINE
from .forced import (
    _FORCED_CAVEAT,
    _FORCED_MEAN_THRESHOLD,
    _FORCED_SUM_THRESHOLD,
    _NULL_FLOOR_RATIO_MIN,
    _delta_summary,
    _forced_deltas,
    matched_length_neutral_filler,
    _top_dependent,
)
from .metrics import receipt_metrics


class SpanSpecError(ValueError):
    """An invalid span spec (bad message index, offsets out of range, find-string missing/ambiguous).
    The route maps this -- and only this -- to a 400; everything else stays a 500/None."""


_OFFSETS_NOTE = (
    "start/end are CHARACTER offsets (Unicode code points, Python str indices, half-open [start, end)) "
    "into the message's content -- NOT byte offsets. JS callers: UTF-16 code units diverge from code "
    "points on astral-plane characters (emoji); prefer the {'find': ...} form for such text."
)

_SPAN_COST_NOTE = (
    "cost: a context-span ablation changes the shared prefix, so the ablated arm re-prefills the whole "
    "context (no KV reuse) -- the expensive case, same as a memory-block ablation."
)

_SPAN_FORCED_NOTE = (
    "the with/without prompts differ in length by exactly the ablated span; deltas align per CONTINUATION "
    "token position, which is what matters -- not per prompt token. The null floor swaps the span for "
    "register-matched filler of the SAME character length, so 'any edit to the context moves logprobs a "
    "little' is measured as a floor and subtracted from the claim, never silently credited to the span."
)

_NOTE_CHANGED = (
    "ablation-causal: removing this span measurably changed the answer -- the greedy regeneration without "
    "the span differs from the greedy baseline with it. This proves the span was load-bearing for THIS "
    "answer; it does not by itself prove intent (e.g. that the span was an injection) -- read the forced "
    "arm's caveat before claiming more."
)

_NOTE_UNCHANGED = (
    "ablation-causal: removing this span did not change the greedy answer. The forced arm still measures "
    "how much the model's CONFIDENCE in the original answer depended on it (a span can matter below the "
    "text-change threshold) -- see silent_influence."
)


def _last_user_index(messages: list):
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], dict) and messages[i].get("role") == "user":
            return i
    return None


def resolve_span(messages: list, spec: dict) -> dict:
    """Resolve `spec` ({"message"?, "start", "end"} or {"message"?, "find"}) against `messages` into
    {"message": i, "start": s, "end": e, "text": content[s:e]}. `message` defaults to the LAST user turn.
    Offsets are character offsets -- see the module docstring. Raises SpanSpecError (-> the route's 400)
    on any invalid spec: missing/ambiguous find-string, out-of-range or empty offsets, a bad message
    index, or a message with no text content."""
    spec = spec or {}
    if not isinstance(messages, list) or not messages:
        raise SpanSpecError("run has no messages to locate a span in")

    idx = spec.get("message")
    if idx is None:
        idx = _last_user_index(messages)
        if idx is None:
            raise SpanSpecError("run has no user message; pass an explicit 'message' index")
    if isinstance(idx, bool) or not isinstance(idx, int):
        raise SpanSpecError("'message' must be an integer index into the run's messages")
    if not 0 <= idx < len(messages):
        raise SpanSpecError(f"'message' index {idx} is out of range (run has {len(messages)} messages)")
    content = messages[idx].get("content") if isinstance(messages[idx], dict) else None
    if not isinstance(content, str) or not content:
        raise SpanSpecError(f"message {idx} has no text content to ablate")

    find = spec.get("find")
    if find is not None:
        if not isinstance(find, str) or not find:
            raise SpanSpecError("'find' must be a non-empty string")
        n = content.count(find)
        if n == 0:
            raise SpanSpecError(f"'find' string not found in message {idx}")
        if n > 1:
            raise SpanSpecError(f"'find' string is ambiguous in message {idx} ({n} occurrences); "
                                "pass explicit start/end character offsets instead")
        start = content.index(find)
        end = start + len(find)
    else:
        start, end = spec.get("start"), spec.get("end")
        if (isinstance(start, bool) or isinstance(end, bool)
                or not isinstance(start, int) or not isinstance(end, int)):
            raise SpanSpecError("need either a 'find' string or integer 'start'/'end' character offsets")
        if start < 0 or end > len(content) or start >= end:
            raise SpanSpecError(
                f"span [{start}, {end}) is out of range or empty for message {idx} (content is "
                f"{len(content)} characters; offsets are character offsets, not bytes)")

    return {"message": idx, "start": start, "end": end, "text": content[start:end]}


def _with_span_replaced(messages: list, idx: int, start: int, end: int, replacement: str) -> list:
    """A shallow-copied message list with messages[idx]'s [start, end) replaced by `replacement`
    ("" = the ablation; matched-length filler = the null-floor control). Only the touched message dict is
    copied deeply enough to mutate safely; the run's own stored list is never modified."""
    out = [dict(m) if isinstance(m, dict) else m for m in messages]
    content = out[idx]["content"]
    out[idx]["content"] = content[:start] + replacement + content[end:]
    return out


def _forced_span_arm(run: dict, influence: dict, sub, ablated_messages: list, control_messages: list) -> dict:
    """The teacher-forced dependence arm: the run's ORIGINAL answer tokens scored under the original vs
    the span-ablated context, via rederive.score_arm (never a reimplementation). Shape mirrors
    forced.forced_receipt's output field-for-field (same thresholds, same caveat, same degraded shapes);
    the null floor is the card_filler recipe applied to the span (kind: "span_filler")."""
    conditions = rederive.with_arm_conditions(run)
    degraded = {"influence": influence, "mode": "forced", "causal_verified": False, "caveat": _FORCED_CAVEAT}

    with_tokens, with_ok = rederive.score_arm(
        sub, conditions, messages=conditions["raw_messages"], block=conditions["raw_block"])
    if not with_ok:
        return {**degraded, "note": "forced scoring needs the engine substrate (score_tokens is not "
                                    "available here)"}

    without_tokens, without_ok = rederive.score_arm(
        sub, conditions, messages=ablated_messages, block=conditions["raw_block"])
    if not without_ok:
        return {**degraded, "note": "the ablated arm could not be scored"}

    deltas = _forced_deltas(with_tokens, without_tokens)
    if deltas is None:
        return {**degraded, "note": "with/without arms did not align token-for-token (a scoring "
                                    "inconsistency)"}

    pieces = [str(t.get("piece", "")) for t in with_tokens]
    summary = _delta_summary(deltas)
    has_effect = (summary["mean_nats_per_token"] >= _FORCED_MEAN_THRESHOLD
                  or abs(summary["sum_nats"]) >= _FORCED_SUM_THRESHOLD)
    out = {
        "influence": influence,
        "mode": "forced",
        "retokenized": conditions["retokenized"],
        "causal_verified": True,
        "answer_tokens": pieces,
        "deltas": [round(d, 6) for d in deltas],
        "sum_nats": summary["sum_nats"],
        "mean_nats_per_token": summary["mean_nats_per_token"],
        "top_dependent": _top_dependent(pieces, deltas),
        "has_effect": has_effect,
        "threshold": {"mean_abs_nats_per_token": _FORCED_MEAN_THRESHOLD,
                      "abs_sum_nats": _FORCED_SUM_THRESHOLD},
        "note": _SPAN_FORCED_NOTE,
        "caveat": _FORCED_CAVEAT,
    }

    # Null floor: the span swapped for register-matched filler of the SAME character length -- its
    # failure degrades only the floor (the real with/without deltas above already stand), mirroring
    # forced_receipt's own control handling.
    control_tokens, control_ok = rederive.score_arm(
        sub, conditions, messages=control_messages, block=conditions["raw_block"])
    control_deltas = _forced_deltas(with_tokens, control_tokens) if control_ok else None
    if control_deltas is not None:
        c_summary = _delta_summary(control_deltas)
        floor_mean = c_summary["mean_nats_per_token"]
        ratio = (summary["mean_nats_per_token"] / floor_mean) if floor_mean > 0 else None
        out["null_floor"] = {
            "kind": "span_filler",
            "deltas": [round(d, 6) for d in control_deltas],
            "sum_nats": c_summary["sum_nats"],
            "mean_nats_per_token": floor_mean,
            "ratio_real_over_floor": round(ratio, 3) if ratio is not None else None,
            "exceeds_floor_by_order_of_magnitude": bool(ratio is not None
                                                        and ratio >= _NULL_FLOOR_RATIO_MIN),
        }
    return out


def span_receipt(run: dict, spec: dict, sub) -> dict | None:
    """One span receipt for `run`: regen arm + forced arm over the span `spec` resolves to.

    `run`  -- a run dict from runlog.get_run(id) (needs "id", "messages"; the forced arm additionally
              reads trace/response/memory/behavior via rederive.with_arm_conditions).
    `spec` -- the request body: {"message"?: int, "start": int, "end": int} or {"message"?: int,
              "find": str}. `message` defaults to the last user turn; offsets are CHARACTER offsets
              (module docstring).
    `sub`  -- the live substrate. `.chat` drives both greedy regen arms (via replay, exactly like
              /runs/<id>/receipt's regen mode); `.score_tokens` drives the forced arm and degrades
              honestly (causal_verified: False inside `forced`) when absent.

    Returns the receipt dict, or None when a regen arm could not be generated (mirrors
    receipts.receipt()'s None -> the route's 500). Raises SpanSpecError -- and only that -- on an
    invalid span spec (the route's 400). Shape mirrors /runs/<id>/receipt's mode="both" response:
    flat regen fields at the top level + the whole forced receipt nested under "forced" +
    silent_influence, with `influence` describing the span ({"kind": "context_span", ...}, the ablated
    text echoed in influence.text) and `answer_changed` aliasing the regen has_effect."""
    if not run or not isinstance(run, dict):
        return None
    messages = run.get("messages") or []
    span = resolve_span(messages, spec)                     # SpanSpecError propagates -> the route's 400
    idx, start, end, text = span["message"], span["start"], span["end"], span["text"]

    influence = {"kind": "context_span", "message": idx, "start": start, "end": end, "text": text}
    changes = {"ablated_span": {"message": idx, "start": start, "end": end, "text": text}}

    ablated_messages = _with_span_replaced(messages, idx, start, end, "")
    control_messages = _with_span_replaced(
        messages, idx, start, end, matched_length_neutral_filler(len(text)))

    # REGEN arm -- the exact path /receipt's regen mode takes (replay, both arms greedy, the stored
    # sampled reply never a term). The ablated arm replays a message-substituted view of the run; the
    # child run it persists carries parent_run_id + changes_applied.ablated_span for lineage.
    baseline_child = replay_run(run, {"greedy": True}, sub)
    if baseline_child is None:
        return None
    ablated_child = replay_run({**run, "messages": ablated_messages}, {**changes, "greedy": True}, sub)
    if ablated_child is None:
        return None
    baseline_reply = baseline_child.get("response") or ""
    ablated_reply = ablated_child.get("response") or ""
    answer_changed = baseline_reply != ablated_reply

    forced = _forced_span_arm(run, influence, sub, ablated_messages, control_messages)

    out = {
        "influence": influence,
        "changes_applied": changes,
        "baseline_reply": baseline_reply,
        "ablated_reply": ablated_reply,
        "delta": receipt_metrics(baseline_reply, ablated_reply),
        "has_effect": answer_changed,
        "answer_changed": answer_changed,                   # explicit alias for the span-receipt consumer
        "causal_verified": True,                            # the span ablation always actually applies
        "mode": "both",
        "note": _NOTE_CHANGED if answer_changed else _NOTE_UNCHANGED,
        "baseline_note": _NOTE_BASELINE,
        "cost_note": _SPAN_COST_NOTE,
        "offsets_note": _OFFSETS_NOTE,
        "forced": forced,
    }
    floor = (forced.get("null_floor") or {})
    out["silent_influence"] = bool(not answer_changed and floor.get("exceeds_floor_by_order_of_magnitude"))
    return out
