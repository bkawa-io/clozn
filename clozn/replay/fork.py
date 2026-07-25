"""fork.py -- fork-at-token: regenerate a stored run's reply from token position t with a chosen
alternative token FORCED at t, recorded as a CHILD run ("click an almost-said token to fork reality").

The sibling of timetravel.branch (rewind & branch from a TURN) at TOKEN granularity. Where a branch
truncates the *transcript* and re-generates a whole turn, a fork keeps the run's exact prompt AND the
reply's own committed pieces [0..position), splices in one forced alternative piece, and lets the model
continue GREEDY from there. The result is persisted exactly like a branch: runlog.record with
parent_run_id + changes_applied = {"fork": {...}}, source "fork", so the child is an inspectable run
the Studio can diff against its parent.

Prompt source (canonical, in order -- the same records rederive.with_arm_conditions treats as truth):
  1. run["final_prompt"]      -- the EXACT rendered string the model saw (recorded by EngineSubstrate
                                 on every run: backlog #5). Best: zero reconstruction.
  2. re-render via the model's own chat template (engine.apply_template) from rederive's
     with-arm messages (assembled_messages preferred; else raw messages + the stored prompt_block,
     folded in the same way clozn_server._inject_block does).

Generation seam: the substrate's ENGINE (sub.engine.complete -- the raw-prompt completion the branch
path's chat() ultimately rides). chat() itself can't be used here: it re-renders the template and
closes the assistant turn, so a partial reply can't be teacher-forced through it. Decode is ALWAYS
greedy (temperature 0, rep_penalty 1, seed 0 -- byte-identical to _engine_complete_traced's greedy
regime): a fork is a deterministic what-if, not a sample. The run's recorded tone dials
(behavior.active_dials) are re-applied from the RECORD via sub.steer.steer_vector, mirroring
score_tokens' explicit-conditions reconstruction -- never read from the live dial state.

HONESTY -- retokenization: the engine takes a prompt STRING, so the prefix pieces + forced piece are
concatenated as text and retokenized by the engine; BPE boundaries can shift (e.g. the forced piece
merging with the last prefix piece), in which case the continuation is not running on the exact
recorded token ids. We DETECT this where possible -- re-tokenize the prefix+forced text through
sub.score_tokens (the same /score text path rederive's fallback uses) and compare piece-for-piece --
and flag the child `retokenized`: False only when verified exact; True when a shift is detected OR
when the substrate has no score seam to verify with (can't prove exact => flagged, the same honest
convention rederive.with_arm_conditions applies to its text fallback). NOTE the check retokenizes the
continuation independently of the prompt (the /score text path's own boundary_approximate caveat), so
the prompt/prefix junction itself stays approximate either way.

Contract mirrors the siblings: validation errors raise ValueError (the route maps them to 400);
everything after validation never raises into the handler -- generation/persistence failures return
None. Stdlib-only; the substrate is passed in (the live SUB), never imported, so this module is
unit-testable against a fake substrate with no model and no engine.
"""
from __future__ import annotations

import time

import clozn.runs.store as runlog
from clozn.runs.think_tags import sanitize_messages

MAX_NEW = 256          # continuation budget, mirroring branch/replay's chat(max_new=256)

FORK_NOTE = ("greedy continuation (sample=false): a deterministic what-if from the forced token "
             "onward, not a sample from the original decode regime; the kept prefix and the forced "
             "token are spliced as text, the rest is the model's own greedy path")

_UNVERIFIED_NOTE = ("; retokenization could not be verified on this substrate (no score seam), so "
                    "the spliced prefix is conservatively flagged retokenized")


# ------------------------------------------------------------------------------- pure validation
def _alt_pairs(trace: dict, position: int) -> list[tuple[str, int | None]]:
    """The recorded alternatives at `position` as (piece, token_id|None) pairs; [] when absent.
    Reads the v1 parallel array (trace.alternatives[position]) -- the shape runlog._clean_alts pins."""
    alts = trace.get("alternatives") or []
    at = alts[position] if position < len(alts) and isinstance(alts[position], list) else []
    out = []
    for a in at:
        if not isinstance(a, dict):
            continue
        piece = str(a.get("piece", a.get("text", "")))
        tid = a.get("token_id", a.get("id"))
        try:
            tid = int(tid) if tid is not None else None
        except (TypeError, ValueError):
            tid = None
        out.append((piece, tid))
    return out


def resolve_forced_token(trace: dict, position: int, token=None, token_id=None) -> tuple[str, bool]:
    """(forced_piece, was_recorded_alternative) for the caller's choice at `position`. PURE; raises
    ValueError on anything unresolvable (the route's 400).

    `token` (piece text) wins when both are given: ANY non-empty piece is allowed -- a free token is a
    legitimate what-if -- but only a piece matching a RECORDED alternative earns
    was_recorded_alternative=True (the honest distinction the response carries). `token_id` alone can
    only resolve against the recorded alternatives (or the committed token itself -- a re-derive, not
    an alternative, so False): with no tokenizer here, an arbitrary id has no text to splice."""
    pairs = _alt_pairs(trace, position)
    if token is not None:
        piece = str(token)
        if not piece:
            raise ValueError("forced 'token' must be a non-empty piece string")
        return piece, any(p == piece for p, _ in pairs)
    if token_id is not None:
        try:
            tid = int(token_id)
        except (TypeError, ValueError):
            raise ValueError("token_id must be an integer")
        for piece, aid in pairs:
            if aid is not None and aid == tid:
                return piece, True
        ids = trace.get("token_ids") or []
        if position < len(ids) and ids[position] == tid:      # the committed pick itself
            return str((trace.get("tokens") or [])[position]), False
        raise ValueError(f"token_id {tid} is not among the recorded alternatives at position "
                         f"{position}; pass 'token' with the piece text to force a free token")
    raise ValueError("need a forced 'token' (piece text) or 'token_id' (a recorded alternative's id)")


# ------------------------------------------------------------------------------- prompt assembly
def _inject_block(messages, block):
    """`messages` with a memory block folded in as system context -- a faithful mirror of
    clozn_server._inject_block (append to an existing system message, else prepend one), kept local so
    this module never imports the server app. Only the no-final_prompt fallback path needs it."""
    if not block:
        return list(messages or [])
    msgs = [dict(m) for m in (messages or [])]
    for m in msgs:
        if m.get("role") == "system":
            m["content"] = (str(m.get("content") or "") + "\n\n" + block).strip()
            return msgs
    return [{"role": "system", "content": block}] + msgs


def _prompt_base(run: dict, sub):
    """(prompt_string, source) for the fork's generation base -- the text the model saw BEFORE the
    reply began. Prefers the recorded final_prompt (exact); falls back to re-rendering rederive's
    with-arm messages through the model's own chat template (engine.apply_template). (None, None)
    when neither is possible (the caller turns that into a clean failure, never a guess)."""
    fp = run.get("final_prompt")
    if isinstance(fp, str) and fp:
        return fp, "final_prompt"
    from clozn.receipts import rederive
    conditions = rederive.with_arm_conditions(run)
    tmpl = getattr(getattr(sub, "engine", None), "apply_template", None)
    if not callable(tmpl):
        return None, None
    try:
        return str(tmpl(_inject_block(conditions["messages"], conditions["block"]))), "apply_template"
    except Exception:
        return None, None


# ------------------------------------------------------------------------------- honesty: retokenization
def _detect_retokenization(sub, run: dict, expected_pieces: list) -> bool | None:
    """Re-tokenize the spliced prefix+forced text through the substrate's score seam and compare
    piece-for-piece with what we spliced. True == a token boundary shifted (the continuation is NOT
    running on the exact recorded pieces); False == verified identical; None == unverifiable here
    (no score_tokens on this substrate, or the call failed) -- the caller flags None as retokenized,
    since exactness can't be proven."""
    score = getattr(sub, "score_tokens", None)
    if not callable(score):
        return None
    expected = [str(p) for p in expected_pieces]
    try:
        from clozn.receipts import rederive
        conditions = rederive.with_arm_conditions(run)
        toks = score(conditions["messages"], None, continuation="".join(expected),
                     block=conditions["block"])
    except Exception:
        return None
    if not isinstance(toks, list) or not toks:
        return None
    got = [str(t.get("piece", "")) for t in toks if isinstance(t, dict)]
    return got != expected


# ------------------------------------------------------------------------------- generation (greedy)
def _steer_kwargs(sub, run: dict) -> dict:
    """The engine steer kwargs reconstructing the run's RECORDED dials (behavior.active_dials) --
    the same explicit-conditions construction EngineSubstrate.score_tokens uses, never the live dial
    state. {} when there are no dials, no steer, or the vector can't be built (best-effort: a fork
    without dials is still an honest greedy continuation; the dials recorded on the child are the
    ones actually applied)."""
    try:
        strengths = dict(((run.get("behavior") or {}).get("active_dials")) or {})
    except Exception:
        strengths = {}
    steer = getattr(sub, "steer", None)
    if steer is None or not strengths or not any(strengths.values()):
        return {}
    try:
        sv = steer.steer_vector(strengths)
    except Exception:
        return {}
    if not sv:
        return {}
    return {"steer_vec": sv, "steer": {"coef": 1.0, "layer": getattr(steer, "layer", 0)},
            "_dials": strengths}


def _complete_greedy(engine, prompt: str, max_new: int, extra_kw: dict):
    """One raw-prompt greedy completion on the engine -- temperature 0 / rep_penalty 1 / seed 0,
    byte-identical to _engine_complete_traced's greedy fallback regime. Returns
    (continuation_text, finish_reason) or (None, None) on an unparseable reply."""
    r = engine.complete(prompt, max_tokens=int(max_new), temperature=0.0, rep_penalty=1.0, seed=0,
                        **extra_kw)
    ch = r.get("choices") if isinstance(r, dict) else None
    if not (isinstance(ch, list) and ch and isinstance(ch[0], dict)):
        return None, None
    return str(ch[0].get("text", "")), ch[0].get("finish_reason")


def _complete_traced(engine, prompt: str, max_new: int, extra_kw: dict):
    """The same greedy completion, but through the server's traced seam so the child gets per-token
    steps (the gap that left every forked child with trace {} -- unforkable-onward, uncastable,
    invisible to the token timeline). Returns (continuation, steps, finish) or None when the seam is
    unavailable or fails -- the caller then uses _complete_greedy, so a fork is NEVER lost to tracing.

    The import is lazy and guarded: this module's contract is stdlib-only/unit-testable-with-a-fake-
    substrate, and the server package imports THIS module (routes/fork.py), so a top-level import
    would be a cycle. Under a fake substrate the import (or the seam itself) simply fails and the
    plain path runs, exactly as before."""
    try:
        from clozn.server.substrates import _engine_complete_traced
    except Exception:
        return None
    try:
        reply, steps, finish, _div = _engine_complete_traced(
            engine, prompt, int(max_new), dict(extra_kw), sample=None)
    except Exception:
        return None
    if not isinstance(reply, str):
        return None
    return reply, steps, finish


def _spliced_child_trace(parent_trace: dict, position: int, forced_piece: str, cont_steps) -> list | None:
    """Assemble the CHILD's full-reply trace: parent's prefix steps + the forced step + the fresh
    continuation steps. Returns a raw steps list for runlog.record(trace=...) or None when it cannot
    be built honestly.

    HONESTY PRECONDITIONS (the caller enforces the first):
      * Only valid when the spliced prefix verified TOKEN-EXACT (retokenized is False): greedy decode
        is deterministic, so an identical prompt + identical prefix tokens + the run's own dials
        reproduce identical per-position distributions -- the parent's recorded measurements ARE the
        child's, not an approximation of them.
      * The forced step keeps the parent's distribution-level values (alternatives, topk_entropy):
        the distribution at `position` depends only on the context BEFORE it, which is unchanged.
        Its committed prob/token_id come from the parent's recorded alternative when the forced piece
        is one; when it isn't (a custom token), they are simply ABSENT -- never invented."""
    rich = parent_trace.get("steps")
    if not isinstance(rich, list) or len(rich) <= position:
        return None
    out = []
    for s in rich[:position]:
        if not isinstance(s, dict):
            return None
        c = dict(s)
        c.pop("pos", None)
        out.append(c)
    forced = dict(rich[position]) if isinstance(rich[position], dict) else {}
    forced.pop("pos", None)
    forced["piece"] = forced_piece
    forced["text"] = forced_piece
    for k in ("token_id", "prob", "confidence", "logprob"):   # the ORIGINAL committed token's, not ours
        forced.pop(k, None)
    for a in forced.get("alternatives") or []:
        if isinstance(a, dict) and str(a.get("piece", a.get("text", ""))) == forced_piece:
            if a.get("token_id") is not None:
                forced["token_id"] = a["token_id"]
            if a.get("prob") is not None:
                forced["prob"] = a["prob"]
            break
    out.append(forced)
    for s in cont_steps or []:
        if isinstance(s, dict):
            c = dict(s)
            c.pop("pos", None)
            out.append(c)
    if len(out) <= position + 1:            # continuation contributed nothing usable
        return None
    for i, c in enumerate(out):             # one sequential index space; the parent's and the
        c["index"] = i                      # continuation's own indices both restart at 0
    return out


# ------------------------------------------------------------------------------- the fork itself
def fork(run: dict, sub, position, token=None, token_id=None, max_new: int = MAX_NEW) -> dict | None:
    """Fork `run`'s reply at trace `position` with the forced `token` (piece text) or `token_id`
    (a recorded alternative's id), continue greedy on the live substrate's engine, and record the
    result as a CHILD run. Returns the child run dict -- extended (NOT persisted; the same convention
    as replay's generated_ids) with:

      prefix_kept        -- the unchanged reply text [0..position) (the UI's divergence anchor)
      forked_from_piece  -- the ORIGINAL committed piece at `position` (what the fork replaced)
      retokenized        -- False only when the spliced prefix verified token-exact (see
                            _detect_retokenization); True on a detected shift OR when unverifiable
      note               -- the greedy-what-if honesty note

    Raises ValueError on invalid input (no trace / position out of range / unresolvable token) -- the
    route maps those to 400. After validation it NEVER raises: any generation or persistence failure
    returns None (the route's 500), mirroring branch/replay."""
    if not run or not isinstance(run, dict):
        raise ValueError("run record is empty")
    trace = run.get("trace") or {}
    pieces = trace.get("tokens")
    if not isinstance(pieces, list) or not pieces:
        raise ValueError("run has no trace to fork from")
    position = int(position)
    if position < 0 or position >= len(pieces):
        raise ValueError(f"fork position {position} out of range "
                         f"(the reply has {len(pieces)} trace tokens)")
    forced_piece, was_recorded = resolve_forced_token(trace, position, token=token, token_id=token_id)

    try:
        engine = getattr(sub, "engine", None)
        if engine is None or not callable(getattr(engine, "complete", None)):
            return None                                     # fork regenerates on the raw-prompt engine seam
        prompt_base, prompt_source = _prompt_base(run, sub)
        if prompt_base is None:
            return None
        prefix = "".join(str(p) for p in pieces[:position])
        forked_prompt = prompt_base + prefix + forced_piece

        retok = _detect_retokenization(
            sub, run, [str(p) for p in pieces[:position]] + [forced_piece])

        steer_kw = _steer_kwargs(sub, run)
        applied_dials = steer_kw.pop("_dials", {})
        t0 = time.time()
        cont_steps = None
        traced = _complete_traced(engine, forked_prompt, max_new, steer_kw)
        if traced is not None:
            continuation, cont_steps, finish = traced
        else:
            continuation, finish = _complete_greedy(engine, forked_prompt, max_new, steer_kw)
        if continuation is None:
            return None
        reply = prefix + forced_piece + continuation

        # The child's own per-token trace -- only when it can be assembled honestly: fresh steps for
        # the continuation AND a token-exact-verified prefix (retok is False; True/None means the
        # parent's measurements cannot be claimed for this child's prefix, so no trace at all).
        child_trace = None
        trace_provenance = None
        if cont_steps and retok is False:
            child_trace = _spliced_child_trace(trace, position, forced_piece, cont_steps)
            if child_trace is not None:
                trace_provenance = ("spliced: prefix steps are the parent's own records (valid -- the "
                                    "prefix verified token-exact and greedy decode is deterministic); "
                                    "the forced step keeps the parent's distribution with the recorded "
                                    "alternative's probability (absent if the token wasn't recorded); "
                                    "the continuation is measured fresh")
        elif cont_steps:
            trace_provenance = ("omitted: continuation steps were captured, but the spliced prefix did "
                                "not verify token-exact, so the parent's prefix measurements cannot be "
                                "claimed for this child")

        changes = {"fork": {"position": position, "token": forced_piece,
                            "was_recorded_alternative": bool(was_recorded),
                            "trace_provenance": trace_provenance}}
        mem = getattr(sub, "memory", None) or getattr(sub, "_mem", None)
        try:
            strength = float(getattr(mem, "memory_strength", 1.0)) if mem is not None else 1.0
        except (TypeError, ValueError):
            strength = 1.0
        memd = {"strength": strength,                        # a fork never touches the live knobs --
                "has_prefix": (getattr(mem, "prefix", None) is not None) if mem is not None else False,
                "cards_applied": [], "proposed_cards": []}   # whatever memory rode the parent is baked
        #                                                      into its final_prompt already
        rid = runlog.record(
            source="fork", client="studio",
            model=run.get("model"), substrate=run.get("substrate"),
            messages=sanitize_messages(run.get("messages") or []), response=reply,
            memory=memd, behavior={"active_dials": applied_dials},
            trace=child_trace,                              # None when it couldn't be built honestly
            final_prompt=forked_prompt,                     # the exact spliced string this child saw
            finish_reason=finish,
            parent_run_id=run.get("id"), changes_applied=changes, started=t0,
        )
        if rid is None:
            return None
        child = runlog.get_run(rid)
        if child is None:
            child = {"id": rid, "response": reply, "parent_run_id": run.get("id"),
                     "changes_applied": changes}
        # response-only extensions (the same convention as replay's generated_ids): the UI's
        # divergence-point rendering + the honesty flags.
        child["prefix_kept"] = prefix
        child["forked_from_piece"] = str(pieces[position])
        child["retokenized"] = True if retok is None else bool(retok)
        child["note"] = FORK_NOTE + (_UNVERIFIED_NOTE if retok is None else "")
        child["prompt_source"] = prompt_source
        return child
    except Exception:
        return None
