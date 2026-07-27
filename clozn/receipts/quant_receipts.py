"""quant_receipts.py -- QUANT-RECEIPTS: "did Q4 lobotomize my model?" Replay a user's own recorded run
under TWO quant files of the SAME model,
teacher-force the identical stored answer tokens under both via /score (reused UNCHANGED -- the two arms
are just two model files, nothing new on the wire), and diff per-token: "Q4_K_M preserved your runs; Q3
broke exactly these refusal/formatting behaviors, at these tokens."

Model-free and GPU-free by construction: this module never talks to a model. Its whole surface is a pure
function of two already-computed /score outputs (EngineClient.score / EngineSubstrate.score_tokens
responses -- see engine/client/clozn_engine.py's `score()` docstring and
engine/core/serve/routes_whitebox.cpp's /score handler for the exact wire shape reproduced in fixtures
here: `[{"id","piece","logprob","topk"?: [{"id","piece","logprob"}, ...]}, ...]`, topk sorted DESCENDING
by logprob so `topk[0]` IS that arm's argmax at that forced step). Mirrors receipts/rederive.py's own
"stdlib-only, duck-typed, never raise" discipline, and receipts/forced.py's caveat discipline: every
receipt carries the honesty labels below rather than letting a bare number imply more than it does.

Two honest signals, NEVER conflated (see _QUANT_CAVEAT, and receipts/forced.py's own regen-vs-forced
caveat for the sibling discipline this mirrors):
  * ARGMAX FLIP -- quant A's top-1 pick at this forced step differs from quant B's. This is the real
    "the quant would have changed the greedy answer here" signal: under teacher forcing (both arms fed
    the identical recorded prefix), a flip means the two quant files disagree about what the single next
    token should be. It is a ONE-STEP counterfactual, not a simulated re-generation -- it does not claim
    what either quant would say several tokens further on, since after a real divergence the actual
    prefix the model sees would differ from what's forced here.
  * DEPENDENCE SHIFT (no flip) -- the argmax token is the SAME under both quants, but the model's raw
    confidence in it changed (a nonzero logprob delta). This is NOT an answer change: greedy decoding
    would produce the identical token either way. It IS evidence the quant altered the model's internals
    around this token, just not enough to flip the choice.
Flip detection needs topk>=1 on BOTH arms (see _TOPK_CAVEAT) -- a position scored with topk=0 on either
arm has an UNKNOWN flip status, counted separately, never folded into "preserved".

Public surface:
  * diff_quant_scores(answer_tokens, tokens_a, tokens_b, label_a=, label_b=) -- the model-free core: the
    receipt from two already-computed /score token arrays. Unit-tested on FIXTURE arrays (no model, no
    GPU) in tests/test_quant_receipts.py.
  * quant_receipt_for_run(run, sub_a, sub_b, label_a=, label_b=) -- the SEAM the future live path calls:
    reconstructs a run's own forced-scoring conditions via rederive.with_arm_conditions (exactly like
    rederive.py/forced.py do) and scores them on TWO substrates -- in production, two EngineSubstrate
    instances each pointed at a different quant's GGUF file (its own engine process/port; llama.cpp's
    server loads one model per process, so "two quants" means two live processes side by side). That
    orchestration -- and the Q8 download + GPU it needs -- is DEFERRED and never exercised here; this
    function itself is model-free and unit-tested against FAKE substrates (mirrors test_rederive.py's
    FakeScoreSub), proving the wiring without touching a real engine.
"""
from __future__ import annotations

from . import rederive


_QUANT_CAVEAT = (
    "an ARGMAX FLIP at a position means quant A's and quant B's top-1 pick differ at that single forced "
    "step -- it does NOT mean the two quants would have produced different text from there on: both arms "
    "are teacher-forced on the SAME recorded continuation, so a flip is a one-step counterfactual (what "
    "THIS step's greedy choice would be under each quant, given the identical prefix so far), never a "
    "simulated re-generation under the flipped token. A nonzero logprob delta with NO flip is a "
    "confidence/DEPENDENCE SHIFT, not an answer change: the argmax token is the same token under both "
    "quants, just held with different conviction. Read 'flipped' and 'dependence shift' as different "
    "claims about different things -- never interchangeably (mirrors receipts/forced.py's regen-vs-forced "
    "discipline: 'both are interventions; they measure different outcomes')."
)

_TOPK_CAVEAT = (
    "argmax-flip detection needs topk>=1 on BOTH arms' /score calls (rank 0 of the returned topk list IS "
    "the argmax under that arm -- the engine partial_sorts it by descending logprob). A position scored "
    "with topk=0 (or a missing/empty topk) on either arm has an UNKNOWN flip status, not a 'no flip' one "
    "-- it is counted and reported separately, never folded into 'preserved'."
)

_TOP_DEPENDENCE_K = 5
_FLIP_DETAIL_CAP = 20


def _as_int_or_none(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _topk_list(token: dict) -> list:
    tk = token.get("topk")
    return tk if isinstance(tk, list) else []


def _argmax_entry(topk: list):
    """topk[0], or None if topk is empty/absent -- topk is already sorted descending by the engine."""
    return topk[0] if topk and isinstance(topk[0], dict) else None


def _rank_of(token_id, topk: list):
    """This token's 0-based position within its OWN arm's topk list, or None if it isn't there (either
    because topk wasn't requested, or because its true rank is worse than k -- both read as 'unknown',
    never as rank 0)."""
    for i, entry in enumerate(topk):
        if isinstance(entry, dict) and entry.get("id") == token_id:
            return i
    return None


def diff_quant_scores(answer_tokens: list, tokens_a: list, tokens_b: list, *,
                      label_a: str = "quant_a", label_b: str = "quant_b") -> dict | None:
    """The model-free QUANT-RECEIPTS core: given a run's recorded answer (as continuation TOKEN IDS --
    the same `continuation_ids` rederive.with_arm_conditions extracts) and the two /score outputs from
    teacher-forcing that SAME continuation under quant A and quant B, compute the per-token receipt.

    `tokens_a`/`tokens_b`: exactly what EngineSubstrate.score_tokens / EngineClient.score return --
    `[{"id","piece","logprob","topk"?}, ...]`, one entry per continuation token, same order as
    `answer_tokens`. Both arms MUST have been scored on the identical continuation (this is checked,
    not assumed): a caller accidentally diffing two different runs' scores is a bug, not a receipt.

    Returns None on malformed input (not lists / empty). Returns a dict with `causal_verified: False`
    and an explanatory `note` (never raises) when the arms don't align -- length mismatch or a
    per-position id mismatch (the two arms scored DIFFERENT continuations) or a missing logprob. On
    success, returns `causal_verified: True` plus the per-position `positions` list and a `summary`
    (see module docstring for the flip-vs-dependence-shift distinction the summary preserves).
    """
    try:
        if not isinstance(answer_tokens, list) or not answer_tokens:
            return None
        if not isinstance(tokens_a, list) or not tokens_a or not isinstance(tokens_b, list) or not tokens_b:
            return None

        n = len(answer_tokens)
        if len(tokens_a) != n or len(tokens_b) != n:
            return {"mode": "quant_diff", "causal_verified": False, "label_a": label_a, "label_b": label_b,
                    "note": (f"the recorded answer and the two quant arms do not align in length -- "
                             f"answer={n} tokens, {label_a}={len(tokens_a)}, {label_b}={len(tokens_b)} -- "
                             "cannot diff token-for-token"),
                    "caveat": _QUANT_CAVEAT}

        positions = []
        for i in range(n):
            expected_id = _as_int_or_none(answer_tokens[i])
            ta, tb = tokens_a[i], tokens_b[i]
            if not isinstance(ta, dict) or not isinstance(tb, dict):
                return {"mode": "quant_diff", "causal_verified": False, "label_a": label_a,
                        "label_b": label_b,
                        "note": f"position {i}: a score entry was not a dict on at least one arm",
                        "caveat": _QUANT_CAVEAT}

            id_a, id_b = _as_int_or_none(ta.get("id")), _as_int_or_none(tb.get("id"))
            if expected_id is None or id_a != expected_id or id_b != expected_id:
                return {"mode": "quant_diff", "causal_verified": False, "label_a": label_a,
                        "label_b": label_b,
                        "note": (f"position {i}: the two arms (or the recorded answer) scored DIFFERENT "
                                 f"continuations -- answer_id={expected_id}, {label_a}_id={id_a}, "
                                 f"{label_b}_id={id_b} -- these are not two scores of the SAME answer"),
                        "caveat": _QUANT_CAVEAT}

            lp_a, lp_b = ta.get("logprob"), tb.get("logprob")
            if not isinstance(lp_a, (int, float)) or not isinstance(lp_b, (int, float)):
                return {"mode": "quant_diff", "causal_verified": False, "label_a": label_a,
                        "label_b": label_b,
                        "note": f"position {i}: a logprob is missing on at least one arm -- cannot diff",
                        "caveat": _QUANT_CAVEAT}

            topk_a, topk_b = _topk_list(ta), _topk_list(tb)
            argmax_a, argmax_b = _argmax_entry(topk_a), _argmax_entry(topk_b)
            rank_a, rank_b = _rank_of(expected_id, topk_a), _rank_of(expected_id, topk_b)

            flip_known = argmax_a is not None and argmax_b is not None
            flipped = (argmax_a.get("id") != argmax_b.get("id")) if flip_known else None
            status = "unknown" if not flip_known else ("flipped" if flipped else "preserved")

            piece = str(ta.get("piece", tb.get("piece", "")))
            positions.append({
                "index": i,
                "token_id": expected_id,
                "piece": piece,
                "logprob_a": float(lp_a),
                "logprob_b": float(lp_b),
                "delta_nats": round(float(lp_b) - float(lp_a), 6),
                "rank_a": rank_a,
                "rank_b": rank_b,
                "rank_change": (rank_b - rank_a) if (rank_a is not None and rank_b is not None) else None,
                "argmax_a_id": argmax_a.get("id") if argmax_a else None,
                "argmax_a_piece": argmax_a.get("piece") if argmax_a else None,
                "argmax_b_id": argmax_b.get("id") if argmax_b else None,
                "argmax_b_piece": argmax_b.get("piece") if argmax_b else None,
                "argmax_flip": flipped,
                "status": status,
            })

        summary = _summarize(positions, label_a=label_a, label_b=label_b)
        return {
            "mode": "quant_diff",
            "causal_verified": True,
            "label_a": label_a,
            "label_b": label_b,
            "n_tokens": n,
            "positions": positions,
            "summary": summary,
            "caveat": _QUANT_CAVEAT,
            "topk_note": _TOPK_CAVEAT,
        }
    except Exception:
        return None


def _summarize(positions: list, *, label_a: str, label_b: str) -> dict:
    n = len(positions)
    flipped = [p for p in positions if p["status"] == "flipped"]
    preserved = [p for p in positions if p["status"] == "preserved"]
    unknown = [p for p in positions if p["status"] == "unknown"]

    def _mean_abs(rows):
        return round(sum(abs(p["delta_nats"]) for p in rows) / len(rows), 6) if rows else None

    top_dependence_shifts = sorted(preserved, key=lambda p: -abs(p["delta_nats"]))[:_TOP_DEPENDENCE_K]

    flip_detail = [{"index": p["index"], "piece": p["piece"],
                    f"{label_a}_would_say": p["argmax_a_piece"], f"{label_b}_would_say": p["argmax_b_piece"],
                    "delta_nats": p["delta_nats"]} for p in flipped[:_FLIP_DETAIL_CAP]]

    positions_str = ", ".join(str(p["index"]) for p in flipped[:_FLIP_DETAIL_CAP])
    tokens_str = ", ".join(repr(p["piece"]) for p in flipped[:_FLIP_DETAIL_CAP])
    if not flipped:
        summary_text = f"{len(preserved)}/{n} tokens preserved (no argmax flips between {label_a} and {label_b})"
    else:
        summary_text = (f"{len(preserved)}/{n} tokens preserved; diverged at {len(flipped)} position(s) "
                        f"[{positions_str}] on tokens [{tokens_str}]")
        if len(flipped) > _FLIP_DETAIL_CAP:
            summary_text += f" (showing first {_FLIP_DETAIL_CAP} of {len(flipped)})"
    if unknown:
        summary_text += (f"; {len(unknown)} position(s) have unknown flip status (no topk on at least "
                         "one arm)")

    return {
        "n_flipped": len(flipped),
        "n_preserved": len(preserved),
        "n_unknown": len(unknown),
        "flipped_positions": [p["index"] for p in flipped],
        "flipped_detail": flip_detail,
        "top_dependence_shifts": top_dependence_shifts,
        "mean_abs_delta_nats_all": _mean_abs(positions),
        "mean_abs_delta_nats_preserved": _mean_abs(preserved),
        "summary_text": summary_text,
    }


def quant_receipt_for_run(run: dict, sub_a, sub_b, *, label_a: str, label_b: str, topk: int = 8) -> dict | None:
    """SEAM for the live path (DEFERRED -- see module docstring; not wired to any server route, never
    run against a real engine here). Reconstructs `run`'s own recorded
    forced-scoring conditions via rederive.with_arm_conditions -- exactly the same reconstruction
    rederive.py/forced.py already use -- then scores that SAME continuation on two substrates via
    rederive.score_arm (duck-typed against `.score_tokens`, just like rederive.py itself). In production
    `sub_a`/`sub_b` would be two EngineSubstrate instances, each backed by a different quant's GGUF file
    on its own engine process/port (llama.cpp loads one model per process); that orchestration -- and the
    Q8 download + GPU it needs -- does not exist yet. This function is model-free and GPU-free ITSELF:
    it is unit-tested against FAKE substrates (mirrors test_rederive.py's FakeScoreSub) to prove the
    wiring, never against a real engine.

    Returns None for a bad run. Returns a dict with `causal_verified: False` and a `note` (never raises)
    when the run has no per-token continuation ids (an old/light-tier run -- retokenized-text scoring is
    too approximate to trust a token-position diff on) or when either substrate fails to score. Otherwise
    delegates to `diff_quant_scores` for the receipt.
    """
    try:
        if not run or not isinstance(run, dict):
            return None
        conditions = rederive.with_arm_conditions(run)
        answer_ids = conditions.get("continuation_ids")
        if not answer_ids:
            return {"mode": "quant_diff", "causal_verified": False, "label_a": label_a, "label_b": label_b,
                    "note": ("this run has no per-token continuation ids in its stored trace (an old or "
                             "light-tier run) -- quant-diffing needs exact token ids to force the "
                             "IDENTICAL answer on both quants; a retokenized-text fallback would drift "
                             "at the BPE boundary and make the per-position diff meaningless"),
                    "caveat": _QUANT_CAVEAT}

        tokens_a, ok_a = rederive.score_arm(sub_a, conditions, block=conditions["block"],
                                            steer_strengths=conditions["steer_strengths"], topk=topk)
        if not ok_a:
            return {"mode": "quant_diff", "causal_verified": False, "label_a": label_a, "label_b": label_b,
                    "note": f"quant '{label_a}' could not be scored (no score_tokens, or scoring raised)",
                    "caveat": _QUANT_CAVEAT}

        tokens_b, ok_b = rederive.score_arm(sub_b, conditions, block=conditions["block"],
                                            steer_strengths=conditions["steer_strengths"], topk=topk)
        if not ok_b:
            return {"mode": "quant_diff", "causal_verified": False, "label_a": label_a, "label_b": label_b,
                    "note": f"quant '{label_b}' could not be scored (no score_tokens, or scoring raised)",
                    "caveat": _QUANT_CAVEAT}

        return diff_quant_scores(answer_ids, tokens_a, tokens_b, label_a=label_a, label_b=label_b)
    except Exception:
        return None
