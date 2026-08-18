"""rederive.py -- deterministic teacher-forced re-derivation of
a stored run's exact answer + per-token internals, built entirely from the run record and
EngineSubstrate.score_tokens (S0/S1) -- NO generation anywhere, ever.

Two consumers share the reconstruction logic here: this module's own `rederive()` (S3 deliverable 1 --
re-derive a run's own answer under its own conditions), and receipts.py's forced-mode receipts (S3
deliverable 2 -- score the SAME answer under a DELIBERATELY-CHANGED block/steer, to measure how much the
answer depended on one influence). Both need "what were this run's with-arm conditions" from the record
alone; only receipts.py additionally needs to SWAP the block/steer between two calls, which is why
`with_arm_conditions` carries both an assembled-preferring pair (`messages`/`block`) and a raw pair
(`raw_messages`/`raw_block`) -- see its docstring.

Public surface:

  * with_arm_conditions(run) -- everything needed to reconstruct a run's WITH-arm forced-scoring
    conditions from the record ALONE: the prompt messages, `steer_strengths` (always {} -- a regular
    run carries no recorded steering state; named-dial personalization is retired and nothing else
    logs a raw steer vector onto an ordinary run record), and the continuation as TOKEN IDS from the
    stored trace (falling back to the raw `response` TEXT, flagged `retokenized: True` --
    boundary-approximate, per REPRODUCE_AND_PROVE_PLAN.md's tokenization-boundary caveat).
  * score_arm(sub, conditions, ...) -- one forced-scoring call reusing `conditions`'s messages/
    continuation under a possibly-different block/steer_strengths/steer_vec -- what makes a with/
    without receipt arm just two calls that share everything except the one knob under test.
  * rederive(run, sub) -- the S3 deliverable: re-derive `run`'s own stored answer under its own
    conditions and return {"text", "steps": [{"piece","token_id","logprob","conf"}, ...], "meta"}.
    Deterministic; never raises (mirrors replay.py/receipts.py's "never raise into the caller"
    contract) -- returns None on any failure.

Stdlib-only (just `math`), duck-typed against `sub.score_tokens` exactly like replay.py is duck-typed
against `sub.chat` -- fully unit-testable with a fake substrate, no model, no GPU.
"""
from __future__ import annotations

import math


def _continuation_ids(run: dict):
    """(ids, retokenized) -- the stored trace's per-token ids (v1 `trace.token_ids`, falling back to the
    v2 `trace.steps`' per-step `token_id`) when EVERY position has one; else (None, True) so the caller
    falls back to scoring the raw `response` TEXT (boundary-approximate -- see module docstring). A
    partially-populated id list (some positions real, some missing) is treated the SAME as "none at
    all" -- never silently mixed with a fabricated id, and never fed to score_tokens's
    `[int(t) for t in ...]` where a None would raise."""
    trace = run.get("trace") or {}
    ids = trace.get("token_ids")
    if isinstance(ids, list) and ids and all(isinstance(i, int) for i in ids):
        return list(ids), False
    steps = trace.get("steps")
    if isinstance(steps, list) and steps:
        got = [s.get("token_id") for s in steps if isinstance(s, dict)]
        if got and all(isinstance(i, int) for i in got):
            return got, False
    return None, True


def with_arm_conditions(run: dict) -> dict:
    """Everything needed to reconstruct `run`'s WITH-arm forced-scoring conditions from the record
    ALONE:
      "messages"/"block"        -- the ASSEMBLED-preferring pair: `run.assembled_messages` when it was
                                    recorded (what was actually fed to the model; `block` is None here
                                    since it's already folded in), else the raw `messages` PLUS the
                                    stored `memory.prompt_block` as an explicit block. This is what
                                    rederive()'s single-arm re-derivation uses.
      "raw_messages"/"raw_block" -- the UN-folded `messages` plus the bare block string, always. A
                                    with/without receipt pair MUST reconstruct both arms from these,
                                    never from "messages" above: clozn_server._inject_block APPENDS a
                                    block to an existing system message rather than replacing it, so
                                    re-injecting a DIFFERENT block onto already-ASSEMBLED messages would
                                    double up old+new block text instead of swapping it.
      "steer_strengths"          -- always {}: a regular run carries no recorded steering state to
                                    reconstruct (named-dial personalization is retired).
      "continuation_ids"/"retokenized" -- see _continuation_ids.
      "response"                 -- the stored reply text (the continuation-TEXT fallback's input).
    """
    run = run or {}
    assembled = run.get("assembled_messages")
    raw_messages = list(run.get("messages") or [])
    raw_block = (run.get("memory") or {}).get("prompt_block")
    if isinstance(assembled, list) and assembled:
        messages, block, block_source = list(assembled), None, "assembled_messages"
    else:
        messages, block = raw_messages, raw_block
        block_source = "prompt_block" if block else "none"
    ids, retokenized = _continuation_ids(run)
    return {
        "messages": messages,
        "block": block,
        "block_source": block_source,
        "raw_messages": raw_messages,
        "raw_block": raw_block,
        "steer_strengths": {},
        "continuation_ids": ids,
        "retokenized": retokenized,
        "response": run.get("response") or "",
    }


def score_arm(sub, conditions: dict, *, messages=None, block=None, steer_strengths=None, steer_vec=None,
             topk: int = 0):
    """One forced-scoring call on `sub.score_tokens`, reusing `conditions`'s messages/continuation
    (from with_arm_conditions) under the given block/steer_strengths/steer_vec -- so a with/without
    receipt arm is just two calls that share everything except the one knob under test. Continuation
    ids are primary; when `conditions["continuation_ids"]` is None (an old/light-tier run), falls back
    to scoring `conditions["response"]` as continuation TEXT (boundary-approximate; the caller should
    surface `conditions["retokenized"]`). Returns (tokens, ok) -- ([], False) on any failure, including a
    substrate with no score_tokens -- never raises."""
    score_tokens = getattr(sub, "score_tokens", None)
    if not callable(score_tokens):
        return [], False
    msgs = messages if messages is not None else conditions.get("messages")
    ids = conditions.get("continuation_ids")
    try:
        kwargs = {"block": block, "topk": int(topk)}
        if steer_vec is not None:
            kwargs["steer_vec"] = steer_vec
        if steer_strengths is not None:
            kwargs["steer_strengths"] = steer_strengths
        if ids is not None:
            tokens = score_tokens(msgs, ids, **kwargs)
        else:
            response = conditions.get("response") or ""
            if not response:
                return [], False
            tokens = score_tokens(msgs, None, continuation=response, **kwargs)
        return (tokens if isinstance(tokens, list) else []), True
    except Exception:
        return [], False


def rederive(run: dict, sub) -> dict | None:
    """The S3 deliverable: deterministically re-derive `run`'s own stored answer under its own recorded
    conditions -- teacher-forced scoring of the SAME continuation tokens, NO generation anywhere.
    Returns {"text", "steps": [{"piece","token_id","logprob","conf"}, ...], "meta"}, or None on any
    failure (mirrors replay.py/receipts.py's own "never raise into the caller" contract)."""
    try:
        if not run or not isinstance(run, dict):
            return None
        conditions = with_arm_conditions(run)
        tokens, ok = score_arm(sub, conditions, block=conditions["block"],
                               steer_strengths=conditions["steer_strengths"])
        if not ok or not tokens:
            return None
        steps = []
        pieces = []
        for t in tokens:
            if not isinstance(t, dict):
                continue
            piece = str(t.get("piece", ""))
            logprob = t.get("logprob")
            conf = math.exp(logprob) if isinstance(logprob, (int, float)) else None
            steps.append({"piece": piece, "token_id": t.get("id"), "logprob": logprob, "conf": conf})
            pieces.append(piece)
        return {
            "text": "".join(pieces),
            "steps": steps,
            "meta": {
                "retokenized": conditions["retokenized"],
                "block_source": conditions["block_source"],
                "n_tokens": len(steps),
            },
        }
    except Exception:
        return None
