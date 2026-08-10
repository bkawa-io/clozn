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

FORK-02 -- compat_fork(), below: the compatibility wrapper POST /runs/<id>/fork now runs through.
Exact execution forks (checkpoint capture -> plan -> execute, clozn.replay.checkpoint_capture /
execution_fork / execution_fork_execute -- FORK-00/01/CKPT-01, proven bit-exact at the engine level)
reach the gateway but nothing called them; this splice remained the only path a "fork" button could
take. compat_fork() tries exact FIRST whenever a request could possibly be exact, degrades to this
module's fork() (the text splice, unchanged above) only when no exact state is honestly available --
per the SAME clozn.replay.execution_fork.plan_execution_fork classifier docs/EXECUTION_FORK_CONTRACT.md
defines, never a second, softer vocabulary for the same facts -- and returns a typed `unavailable`
rather than a silent empty result when neither path can run.
"""
from __future__ import annotations

from copy import deepcopy
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
        # Branch Fan's reconstructed horizon is the parent's remaining recorded horizon.  A final
        # response token therefore has a truthful zero-token continuation; do not turn that
        # boundary case into an unrelated extra model generation.
        if int(max_new) == 0:
            continuation, finish = "", "branch_horizon_exhausted"
        else:
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


# ============================================================================== FORK-02: compatibility wrapper
COMPAT_OUTCOMES = ("exact_execution_fork", "reconstructed_replay", "unavailable")

EXACT_NOTE = ("exact execution fork: the worker restored its exact recorded KV state and applied the "
             "forced token there directly on its token id -- no text splice, nothing to retokenize")


def _compat_reason(code, message) -> dict:
    """The {code, message} shape clozn.execution-fork.v1 already uses for its own `reasons` -- reused
    verbatim so this wrapper never invents a second vocabulary for the same facts."""
    return {"code": str(code or "unknown"), "message": str(message or "")}


def _exact_token_id(trace: dict, position: int, forced_piece: str, token_id_arg) -> int | None:
    """The numeric token id the EXACT worker wire needs for `forced_piece` at `position`, or None when
    only free text is available. `token_id_arg` (the caller's raw token_id, if supplied) wins and is
    trusted as-is -- resolve_forced_token already proved it names `forced_piece` (a recorded alternative
    or the committed pick's own id). Otherwise a `token` (piece text) is looked up against the recorded
    alternatives; a free piece with no recorded id has no numeric id to force exactly, so exact is not
    attempted for it -- reconstruction is the only honest path, never a guess at an id."""
    if token_id_arg is not None:
        try:
            return int(token_id_arg)
        except (TypeError, ValueError):
            return None
    for piece, tid in _alt_pairs(trace, position):
        if piece == forced_piece and tid is not None:
            return tid
    return None


def capture_exact_fork_context(run: dict, engine, *, runtime_identity: dict,
                               worker_identity: dict) -> dict:
    """Capture one exact-fork checkpoint context for reuse by one or more candidates.

    This is the shared exact-first capture policy used by ``compat_fork`` and Branch Fan.  A capture
    that never becomes available is an eligibility result, not an execution failure; callers may then
    consult the existing reconstructed-replay policy.  Once a checkpoint is handed to a plan, later
    exact execution failures must remain failures and must never be hidden behind reconstruction.
    """
    from clozn.replay.checkpoint_capture import CheckpointCaptureError, capture_parent_checkpoint

    try:
        capture = capture_parent_checkpoint(
            run, engine, runtime_identity=runtime_identity, worker_identity=worker_identity)
    except CheckpointCaptureError as exc:
        return {"status": "ineligible",
                "reason": _compat_reason("checkpoint_capture_request_invalid", str(exc))}
    if capture.get("status") != "available":
        reason = (capture.get("reasons") or [{}])[0]
        return {"status": "ineligible", "reason": _compat_reason(
            reason.get("code", "checkpoint_unavailable"),
            reason.get("message", "no exact checkpoint could be captured"))}
    return {
        "status": "available",
        "checkpoint_reference": deepcopy(capture["checkpoint_reference"]),
        "capture": capture,
    }


def plan_exact_force_token(run: dict, request: dict, *, checkpoint_reference: dict,
                           runtime_identity: dict, worker_identity: dict) -> dict:
    """Plan one force-token candidate against an already captured exact checkpoint."""
    from clozn.replay.execution_fork import plan_execution_fork

    return plan_execution_fork(
        run, request, checkpoint=checkpoint_reference,
        runtime_identity=runtime_identity, worker_identity=worker_identity,
    )


def execute_exact_force_token(run: dict, plan: dict, engine, *, runtime_identity: dict,
                              worker_identity: dict, reload_parent=None, cancel_check=None) -> dict:
    """Execute one planned exact candidate using the canonical execution-fork executor."""
    from clozn.replay.execution_fork_execute import execute_exact_fork

    return execute_exact_fork(
        run, plan, engine,
        runtime_identity=runtime_identity, worker_identity=worker_identity,
        reload_parent=reload_parent, cancel_check=cancel_check,
    )


def _try_exact(run: dict, engine, request: dict, runtime_identity: dict, worker_identity: dict) -> dict:
    """Attempt the exact path for one already-token-id-resolvable request: capture an ephemeral
    checkpoint of the immutable parent (clozn.replay.checkpoint_capture), plan against it
    (clozn.replay.execution_fork), and -- only when that plan is exact -- execute it
    (clozn.replay.execution_fork_execute). All three modules are composed read-only; this function adds
    no new eligibility rule of its own.

    Returns {"status": "success", "child": ..., "exactness": ..., "unchanged_control": ...,
    "execution_id": ...} when the intervention completed; {"status": "failed", "reason": ..., ...} when
    an exact plan existed and RAN but its execution genuinely diverged/errored/went stale (the
    checkpoint DID exist -- never masked behind reconstruction); or {"status": "ineligible",
    "reason": ...} when no exact state was honestly available at all (the caller falls through to the
    reconstructed-replay gate for that case)."""
    capture = capture_exact_fork_context(
        run, engine, runtime_identity=runtime_identity, worker_identity=worker_identity)
    if capture["status"] != "available":
        return capture

    plan = plan_exact_force_token(
        run, request, checkpoint_reference=capture["checkpoint_reference"],
        runtime_identity=runtime_identity, worker_identity=worker_identity)
    if plan["classification"] != "exact_execution_fork":
        reason = (plan.get("reasons") or [{}])[0]
        return {"status": "ineligible", "reason": _compat_reason(
            reason.get("code", "exact_plan_unavailable"),
            reason.get("message", "the captured checkpoint did not plan as exact"))}

    result = execute_exact_force_token(
        run, plan, engine,
        runtime_identity=runtime_identity, worker_identity=worker_identity,
        reload_parent=runlog.get_run)
    receipt = result["receipt"]
    if receipt["phase"] == "completed":
        return {
            "status": "success",
            "child": result["child"],
            "exactness": receipt["exactness"],
            "unchanged_control": receipt["unchanged_control"],
            "execution_id": receipt["execution_id"],
        }
    reason = (receipt.get("reasons") or [{}])[0]
    return {
        "status": "failed",
        "reason": _compat_reason(
            reason.get("code", "exact_execution_failed"),
            reason.get("message", "the exact execution fork did not complete")),
        "exactness": receipt.get("exactness"),
        "unchanged_control": receipt.get("unchanged_control"),
        "execution_id": receipt.get("execution_id"),
    }


def compat_fork(run: dict, sub, position, *, token=None, token_id=None,
                runtime_identity: dict | None = None, worker_identity: dict | None = None,
                max_new: int = MAX_NEW) -> dict | None:
    """POST /runs/<id>/fork's FORK-02 compatibility wrapper: one honest outcome, never a silent empty
    result.

    Tries the exact execution-fork path FIRST whenever this request could possibly be exact (a
    resolvable numeric token id, a live engine, and a current worker/runtime identity) via _try_exact,
    above. Only when no exact state is honestly available does it degrade to the legacy text splice
    (fork(), above) -- and the SAME classifier (clozn.replay.execution_fork.plan_execution_fork,
    checkpoint=None) decides whether even that is eligible, so this module never invents a softer
    reconstruction gate than the one docs/EXECUTION_FORK_CONTRACT.md already defines. "Never silently
    prefer the splice because it is easier": exact is always attempted first when it could possibly
    apply, and a genuinely FAILED exact attempt (a checkpoint that existed and ran, but diverged) is
    reported as `unavailable`, never quietly swapped for a plausible-looking splice.

    Returns a dict carrying `outcome` (one of COMPAT_OUTCOMES) and `reasons`
    ({code, message} pairs, the same shape clozn.execution-fork.v1 uses):
      * exact_execution_fork / reconstructed_replay -- the CHILD run record (the pre-FORK-02 200 shape,
        prefix_kept/forked_from_piece/retokenized/note included exactly as fork() always produced them),
        plus `outcome`/`reasons` and, when exact ran, `exactness` + `unchanged_control` lifted straight
        from its execution-fork receipt (never re-derived or renamed).
      * unavailable -- no child was created: {"outcome": "unavailable", "reasons": [...]}, plus whatever
        exactness/execution_fork_execution_id facts an attempted-but-failed exact plan already produced.

    Returns None ONLY for the one case that mirrors the pre-FORK-02 500: reconstruction was eligible but
    its own generation/persistence step failed (fork() returned None) -- the same failure, the same
    contract, so the route's existing 500 handling for `child is None` still applies unchanged.

    Raises ValueError on invalid input (no trace / position out of range / unresolvable token) -- same
    contract as fork(), the route maps those to 400. After validation it never raises."""
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
    forced_piece, _was_recorded = resolve_forced_token(trace, position, token=token, token_id=token_id)
    exact_token_id = _exact_token_id(trace, position, forced_piece, token_id)

    change = {"type": "force_token", "token_piece": forced_piece}
    if exact_token_id is not None:
        change["token_id"] = exact_token_id
    request = {"position": position, "change": change}

    reasons: list[dict] = []
    engine = getattr(sub, "engine", None)
    can_attempt_exact = (
        exact_token_id is not None and engine is not None
        and isinstance(runtime_identity, dict) and isinstance(worker_identity, dict)
    )

    if can_attempt_exact:
        outcome = _try_exact(run, engine, request, runtime_identity, worker_identity)
        if outcome["status"] == "success":
            child = outcome["child"]
            pieces_str = [str(p) for p in pieces]
            child["outcome"] = "exact_execution_fork"
            child["reasons"] = [_compat_reason(
                "exact_preconditions_met",
                "an exact checkpoint was captured and its intervention completed")]
            child["exactness"] = deepcopy(outcome["exactness"])
            child["unchanged_control"] = deepcopy(outcome["unchanged_control"])
            child["execution_fork_execution_id"] = outcome["execution_id"]
            child["prefix_kept"] = "".join(pieces_str[:position])
            child["forked_from_piece"] = pieces_str[position]
            child["retokenized"] = False
            child["note"] = EXACT_NOTE
            return child
        if outcome["status"] == "failed":
            # An exact plan existed and ran, but its own execution genuinely diverged, errored, or went
            # stale. The checkpoint DID exist -- this is not "no exact state", so it is never masked
            # behind the splice; report it honestly (GET /execution-forks/<id> holds the full receipt).
            return {
                "outcome": "unavailable",
                "reasons": [outcome["reason"]],
                "execution_fork_execution_id": outcome.get("execution_id"),
                "exactness": deepcopy(outcome.get("exactness")),
                "unchanged_control": deepcopy(outcome.get("unchanged_control")),
            }
        reasons.append(outcome["reason"])                          # ineligible -- fall through below
    elif exact_token_id is None:
        reasons.append(_compat_reason(
            "exact_requires_token_id",
            "the forced token has no recorded numeric id, so only the reconstructed text path can "
            "force it"))
    elif engine is None:
        reasons.append(_compat_reason(
            "no_engine", "no engine is available to attempt an exact checkpoint"))
    else:
        reasons.append(_compat_reason(
            "worker_identity_unavailable",
            "the current worker's exact runtime/worker identity is unavailable"))

    from clozn.replay.execution_fork import plan_execution_fork
    recon_plan = plan_execution_fork(
        run, request, checkpoint=None,
        runtime_identity=runtime_identity, worker_identity=worker_identity)
    if recon_plan["classification"] != "reconstructed_replay":
        plan_reason = (recon_plan.get("reasons") or [{}])[0]
        reasons.append(_compat_reason(
            plan_reason.get("code", "reconstruction_unavailable"),
            plan_reason.get("message", "the legacy text-splice path is also unavailable")))
        return {"outcome": "unavailable", "reasons": reasons}

    child = fork(run, sub, position, token=token, token_id=token_id, max_new=max_new)
    if child is None:
        return None
    child["outcome"] = "reconstructed_replay"
    child["reasons"] = reasons or [_compat_reason(
        "checkpoint_not_supplied", "no exact checkpoint was supplied; reconstructing from text")]
    child["exactness"] = deepcopy(recon_plan["exactness"])
    child["unavoidable_differences"] = list(recon_plan["unavoidable_differences"])
    return child
