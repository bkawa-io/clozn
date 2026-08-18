"""Sequential execution of a bounded fan over a run's recorded token alternatives.

Branch Fan is deliberately an orchestration layer.  It selects only alternatives already recorded on
the immutable parent, delegates exactness to the existing execution-fork policy, delegates
reconstruction to its retained Branch Fan child seam, and delegates comparison to ``diff_runs``.  It does not
create a fan/experiment object and does not perform a new model-analysis operation.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
import time

from clozn import schemas
import clozn.runs.store as runlog
from clozn.runs.think_tags import sanitize_messages

SCHEMA_VERSION = "clozn.branch-fan.v1"
DEFAULT_LIMIT = 3
MIN_LIMIT = 1
MAX_LIMIT = 4
MAX_NEW = 256

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
    substrate, and the server package imports THIS module, so a top-level import
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
def reconstruct_branch_child(run: dict, sub, position, token=None, token_id=None, max_new: int = MAX_NEW) -> dict | None:
    """Reconstruct one Branch Fan child from text and record the resulting child Run.

    Returns the child run dict -- extended (NOT persisted; the same convention
    as replay's generated_ids) with:

      prefix_kept        -- the unchanged reply text [0..position) (the UI's divergence anchor)
      forked_from_piece  -- the ORIGINAL committed piece at `position` (what the fork replaced)
      retokenized        -- False only when the spliced prefix verified token-exact (see
                            _detect_retokenization); True on a detected shift OR when unverifiable
      note               -- the greedy-what-if honesty note

    Raises ValueError on invalid input (no trace / position out of range / unresolvable token) -- the
    Branch Fan maps validation failures to its typed unavailable result. After validation it NEVER
    raises: any generation or persistence failure returns None."""
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




class BranchFanInputError(ValueError):
    """A typed caller-input error suitable for the HTTP route's stable 400 response."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_limit(limit: int) -> int:
    if not _is_int(limit) or not MIN_LIMIT <= limit <= MAX_LIMIT:
        raise BranchFanInputError("invalid_limit", f"limit must be an integer from {MIN_LIMIT} to {MAX_LIMIT}")
    return limit


def _trace_tokens(parent: Mapping) -> list[str]:
    trace = parent.get("trace")
    tokens = trace.get("tokens") if isinstance(trace, Mapping) else None
    if not isinstance(tokens, list) or not tokens or not all(isinstance(piece, str) for piece in tokens):
        raise BranchFanInputError("invalid_position", "parent has no recorded response token pieces")
    return tokens


def _candidate_projection(candidate: Mapping) -> dict:
    out = {"rank": candidate["recorded_rank"]}
    if candidate.get("token_id") is not None:
        out["token_id"] = candidate["token_id"]
    if candidate.get("probability") is not None:
        out["probability"] = candidate["probability"]
    return out


def _recorded_candidates(parent: Mapping, position: int, limit: int) -> tuple[list[dict], int]:
    """Select usable recorded alternatives in their source-array order."""
    tokens = _trace_tokens(parent)
    if not _is_int(position) or position < 0 or position >= len(tokens):
        raise BranchFanInputError("invalid_position", "position is outside the recorded response token range")
    trace = parent["trace"]
    raw_alternatives = trace.get("alternatives")
    at_position = (
        raw_alternatives[position]
        if isinstance(raw_alternatives, list) and position < len(raw_alternatives)
        and isinstance(raw_alternatives[position], list)
        else []
    )
    candidates = []
    seen_ids = set()
    seen_pieces = set()
    committed_piece = tokens[position]
    token_ids = trace.get("token_ids")
    committed_id = (
        token_ids[position]
        if isinstance(token_ids, list) and position < len(token_ids)
        and _is_int(token_ids[position]) and token_ids[position] >= 0
        else None
    )
    for recorded_rank, raw in enumerate(at_position):
        if not isinstance(raw, Mapping):
            continue
        piece = raw.get("piece", raw.get("text"))
        if not isinstance(piece, str) or not piece or piece == committed_piece:
            continue

        token_id = raw.get("token_id", raw.get("id"))
        if token_id is not None:
            if not _is_int(token_id) or token_id < 0:
                continue
            if committed_id is not None and token_id == committed_id:
                continue
        probability = raw.get("prob", raw.get("probability", raw.get("confidence")))
        if probability is not None:
            if (
                not isinstance(probability, (int, float))
                or isinstance(probability, bool)
                or not math.isfinite(float(probability))
                or probability < 0
                or probability > 1
            ):
                continue
            probability = float(probability)

        if token_id is not None:
            if token_id in seen_ids:
                continue
            seen_ids.add(token_id)
        elif piece in seen_pieces:
            continue
        seen_pieces.add(piece)
        candidate = {
            "recorded_rank": recorded_rank,
            "piece": piece,
        }
        if token_id is not None:
            candidate["token_id"] = token_id
        if probability is not None:
            candidate["probability"] = probability
        candidates.append(candidate)
    return candidates[:limit], len(at_position)


def recorded_alternatives_available(parent: Mapping, position: int) -> bool:
    """Read-only availability check using Branch Fan's own candidate authority.

    Dispatchers may use this to avoid waking a worker for the typed no-candidate result.  It does
    not expose, sort, or duplicate candidate selection; the actual Branch Fan call still owns the
    complete filtering and ordering operation.
    """
    candidates, _ = _recorded_candidates(parent, position, 1)
    return bool(candidates)


def recorded_alternative_candidates(parent: Mapping, position: int, *, limit: int = MAX_LIMIT) -> list[dict]:
    """Return the Branch Fan candidate projection without executing anything.

    This is a small read-side seam for affordance builders.  Candidate filtering, deduplication, and
    recorded-array ordering remain owned by :func:`_recorded_candidates`; callers must not reproduce
    those rules merely to render a Test This or Selection Inspector descriptor.
    """
    if not _is_int(limit) or not MIN_LIMIT <= limit <= MAX_LIMIT:
        raise BranchFanInputError("invalid_limit", f"limit must be an integer from {MIN_LIMIT} to {MAX_LIMIT}")
    candidates, _ = _recorded_candidates(parent, position, limit)
    return [dict(candidate) for candidate in candidates]


def _cancelled(cancel_check) -> bool:
    if not callable(cancel_check):
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return False


def _reason(code: str, message: str) -> dict:
    return {"code": code, "message": message}


_SAFE_REASON_MESSAGES = {
    "control_diverged": "the unchanged exact control did not match",
    "stale_plan": "exact execution preconditions changed after planning",
    "checkpoint_expired": "the shared exact checkpoint expired",
    "stale_worker_generation": "the exact checkpoint belongs to another worker generation",
    "execution_cancelled": "exact execution was cancelled",
    "intervention_failed": "the exact intervention failed",
    "persistence_failed": "the completed child could not be persisted",
    "reconstructed_execution_failed": "reconstructed replay did not produce a child",
    "exact_execution_failed": "exact execution failed",
}


def _public_reasons(raw) -> list[dict]:
    out = []
    for item in raw or []:
        if not isinstance(item, Mapping):
            continue
        code = item.get("code")
        if isinstance(code, str) and code:
            message = _SAFE_REASON_MESSAGES.get(code, "branch execution was unavailable")
            out.append({"code": code, "message": message})
    return out


def comparison_projection_from_diff(parent: Mapping, child: Mapping, diff: Mapping) -> dict:
    """Project only values returned by an already-computed canonical run diff."""
    view = diff.get("first_divergence_view")
    if not isinstance(view, Mapping):
        view = {"schema_version": "clozn.first-divergence-view.v1", "state": "trace_unavailable"}
    if not diff.get("trace_available"):
        return {
            "state": "trace_unavailable",
            "first_divergence_view": deepcopy(dict(view)),
        }
    out = {"state": "available", "first_divergence_view": deepcopy(dict(view))}
    for key in ("common_prefix_len", "first_divergence"):
        if key in diff:
            out[key] = deepcopy(diff[key])
    summary = diff.get("summary")
    if isinstance(summary, Mapping):
        value = summary.get("char_similarity")
        label = summary.get("char_similarity_label")
        if value is not None or label is not None:
            surface = {}
            if value is not None:
                surface["value"] = deepcopy(value)
            if label is not None:
                surface["label"] = deepcopy(label)
            out["surface_similarity"] = surface
    return out


def comparison_projection(parent: Mapping, child: Mapping) -> dict:
    """Compute and project one canonical run diff."""
    from clozn.analysis.model_diff import diff_runs

    return comparison_projection_from_diff(parent, child, diff_runs(parent, child))


# Keep the original local name for older tests/callers while exposing one shared projection seam to
# Test This.  Both features still call the canonical ``diff_runs`` implementation exactly once per
# successful child; neither feature invents a second token diff.
_comparison = comparison_projection


def _branch_alternative(candidate: Mapping) -> dict:
    return {"recorded_alternative": _candidate_projection(candidate)}


def _not_attempted(candidate: Mapping, code: str, message: str) -> dict:
    out = _branch_alternative(candidate)
    out.update({
        "state": "not_attempted",
        "outcome": "unavailable",
        "reasons": [_reason(code, message)],
        "comparison": None,
    })
    return out


def _unavailable(candidate: Mapping, reasons, *, exactness=None, unchanged_control=None) -> dict:
    out = _branch_alternative(candidate)
    out.update({
        "state": "unavailable",
        "outcome": "unavailable",
        "reasons": _public_reasons(reasons) or [_reason("branch_unavailable", "branch could not be produced")],
        "comparison": None,
    })
    if isinstance(exactness, Mapping):
        out["exactness"] = deepcopy(dict(exactness))
    if isinstance(unchanged_control, Mapping):
        out["unchanged_control"] = {"status": unchanged_control.get("status", "unavailable")}
    return out


def _fidelity(branches: list[Mapping]) -> str:
    outcomes = [branch.get("outcome") for branch in branches if branch.get("state") == "completed"]
    if not outcomes:
        return "none_completed"
    exact = sum(outcome == "exact_execution_fork" for outcome in outcomes)
    reconstructed = sum(outcome == "reconstructed_replay" for outcome in outcomes)
    if exact and reconstructed:
        return "mixed"
    if exact == len(outcomes):
        return "all_exact"
    if reconstructed == len(outcomes):
        return "all_reconstructed"
    return "mixed"


def _summary(branches: list[Mapping], requested: int, *, status: str) -> dict:
    return {
        "status": status,
        "requested_branches": requested,
        "attempted_branches": sum(branch.get("state") != "not_attempted" for branch in branches),
        "children_created": sum(branch.get("state") == "completed" for branch in branches),
        "exact_children": sum(branch.get("outcome") == "exact_execution_fork" for branch in branches),
        "reconstructed_children": sum(branch.get("outcome") == "reconstructed_replay" for branch in branches),
        "unavailable_branches": sum(branch.get("state") == "unavailable" for branch in branches),
        "not_attempted_branches": sum(branch.get("state") == "not_attempted" for branch in branches),
    }


def _execution_base(capture_state: str, *, reused: bool, fidelity: str, reason=None) -> dict:
    capture = {"state": capture_state, "reused_for_exact_candidates": bool(reused)}
    if reason is not None:
        capture["reason"] = deepcopy(reason)
    return {
        "policy": "exact_first",
        "order": "sequential",
        "checkpoint_capture": capture,
        "fidelity": fidelity,
    }


def _completed_exact(candidate, child, receipt, parent) -> dict:
    out = _branch_alternative(candidate)
    out.update({
        "state": "completed",
        "outcome": "exact_execution_fork",
        "child_run_id": child.get("id"),
        "exactness": deepcopy(receipt.get("exactness") or {"proof_status": "confirmed"}),
        "unchanged_control": {"status": (receipt.get("unchanged_control") or {}).get("status", "matched")},
        "reasons": [],
        "comparison": _comparison(parent, child),
    })
    if receipt.get("execution_id"):
        out["execution_fork_execution_id"] = receipt["execution_id"]
    return out


def _completed_reconstructed(candidate, child, plan, parent) -> dict:
    out = _branch_alternative(candidate)
    out.update({
        "state": "completed",
        "outcome": "reconstructed_replay",
        "child_run_id": child.get("id"),
        "exactness": deepcopy(plan.get("exactness") or {
            "regime": "reconstructed_text",
            "proof_status": "not_applicable",
        }),
        "unavoidable_differences": deepcopy(plan.get("unavoidable_differences") or []),
        "reasons": [],
        "comparison": _comparison(parent, child),
    })
    return out


_SHARED_FAILURE_CODES = frozenset({
    "stale_plan", "checkpoint_expired", "stale_worker_generation", "runtime_identity_mismatch",
    "worker_generation_changed", "checkpoint_invalidated", "execution_cancelled",
})


def _run_reconstructed(parent, sub, candidate, position, remaining, runtime_identity, worker_identity):
    from clozn.replay.execution_fork import plan_execution_fork

    request = {
        "position": position,
        "change": {"type": "force_token", "token_piece": candidate["piece"]},
    }
    if candidate.get("token_id") is not None:
        request["change"]["token_id"] = candidate["token_id"]
    plan = plan_execution_fork(
        parent, request, checkpoint=None,
        runtime_identity=runtime_identity, worker_identity=worker_identity,
    )
    if plan.get("classification") != "reconstructed_replay":
        return _unavailable(candidate, plan.get("reasons"), exactness=plan.get("exactness"))
    child = reconstruct_branch_child(parent, sub, position, token=candidate["piece"], max_new=remaining)
    if not isinstance(child, Mapping) or not child.get("id"):
        return _unavailable(candidate, [_reason("reconstructed_execution_failed", "reconstructed replay did not produce a child")])
    return _completed_reconstructed(candidate, child, plan, parent)


def _branch_failure_code(branch: Mapping) -> str | None:
    reasons = branch.get("reasons")
    if isinstance(reasons, list) and reasons and isinstance(reasons[0], Mapping):
        code = reasons[0].get("code")
        return code if isinstance(code, str) else None
    return None


def branch_fan(
    parent_run: dict,
    sub,
    position: int,
    *,
    limit: int = DEFAULT_LIMIT,
    runtime_identity: dict | None = None,
    worker_identity: dict | None = None,
    reload_parent=None,
    cancel_check=None,
) -> dict:
    """Create bounded child forks for the parent's already-recorded alternatives."""
    if not isinstance(parent_run, Mapping):
        raise BranchFanInputError("invalid_parent", "parent run must be an object")
    parent_id = parent_run.get("id")
    if not isinstance(parent_id, str) or not parent_id:
        raise BranchFanInputError("invalid_parent", "parent run id is unavailable")
    if not _is_int(position) or position < 0:
        raise BranchFanInputError("invalid_position", "position must be a non-negative integer")
    limit = _validate_limit(limit)
    tokens = _trace_tokens(parent_run)
    if position >= len(tokens):
        raise BranchFanInputError("invalid_position", "position is outside the recorded response token range")

    candidates, recorded_count = _recorded_candidates(parent_run, position, limit)
    selection = {
        "source": "recorded_alternatives",
        "recorded_alternatives": recorded_count,
        "selected_alternatives": len(candidates),
        "requested_limit": limit,
    }
    if not candidates:
        selection.update({"state": "unavailable", "reason": "no_recorded_alternatives"})
        result = {
            "schema_version": SCHEMA_VERSION,
            "parent_run_id": parent_id,
            "position": position,
            "selection": selection,
            "execution": _execution_base("not_attempted", reused=False, fidelity="none_completed"),
            "branches": [],
            "summary": _summary([], 0, status="unavailable"),
        }
        schemas.validate(result, SCHEMA_VERSION)
        return result
    selection["state"] = "available"

    branches = []
    capture_state = "not_attempted"
    capture_reason = None
    checkpoint_reference = None
    exact_candidate_exists = any(candidate.get("token_id") is not None for candidate in candidates)
    engine = getattr(sub, "engine", None) if sub is not None else None
    exact_possible = (
        exact_candidate_exists and callable(getattr(engine, "execution_fork", None))
        and isinstance(runtime_identity, Mapping) and isinstance(worker_identity, Mapping)
    )

    if _cancelled(cancel_check):
        branches = [_not_attempted(candidate, "branch_fan_cancelled", "branch fan cancelled before execution")
                    for candidate in candidates]
        result = {
            "schema_version": SCHEMA_VERSION, "parent_run_id": parent_id, "position": position,
            "selection": selection,
            "execution": _execution_base("not_attempted", reused=False, fidelity="none_completed"),
            "branches": branches,
            "summary": _summary(branches, len(candidates), status="cancelled"),
        }
        schemas.validate(result, SCHEMA_VERSION)
        return result

    if exact_possible:
        from clozn.replay.execution_fork import capture_exact_force_token_context
        try:
            capture = capture_exact_force_token_context(
                parent_run, engine, runtime_identity=dict(runtime_identity), worker_identity=dict(worker_identity))
        except Exception:
            capture = {
                "status": "ineligible",
                "reason": _reason("checkpoint_capture_unavailable", "an exact checkpoint could not be captured"),
            }
        if capture.get("status") == "available":
            reference = capture.get("checkpoint_reference")
            if isinstance(reference, Mapping):
                capture_state = "available"
                checkpoint_reference = deepcopy(dict(reference))
            else:
                capture_state = "unavailable"
                capture_reason = _reason("checkpoint_capture_unavailable", "an exact checkpoint reference was unavailable")
        else:
            capture_state = "unavailable"
            capture_reason = (_public_reasons([capture.get("reason")]) or [
                _reason("checkpoint_capture_unavailable", "an exact checkpoint could not be captured")
            ])[0]
    elif exact_candidate_exists:
        capture_state = "unavailable"
        capture_reason = _reason("exact_execution_unavailable", "exact checkpoint prerequisites were unavailable")

    from clozn.replay.execution_fork import execute_exact_force_token, plan_exact_force_token
    import clozn.runs.store as runlog
    reload_parent = reload_parent or runlog.get_run
    remaining = max(0, len(tokens) - position - 1)
    stop_scheduling = None
    cancelled = False

    for offset, candidate in enumerate(candidates):
        if stop_scheduling is not None:
            branches.append(_not_attempted(candidate, *stop_scheduling))
            continue
        if _cancelled(cancel_check):
            cancelled = True
            for rest in candidates[offset:]:
                branches.append(_not_attempted(rest, "branch_fan_cancelled", "branch fan cancelled between branches"))
            break

        exact_candidate = checkpoint_reference is not None and candidate.get("token_id") is not None
        if exact_candidate:
            request = {
                "position": position,
                "change": {
                    "type": "force_token",
                    "token_id": candidate["token_id"],
                    "token_piece": candidate["piece"],
                },
            }
            try:
                plan = plan_exact_force_token(
                    parent_run, request, checkpoint_reference=checkpoint_reference,
                    runtime_identity=dict(runtime_identity), worker_identity=dict(worker_identity),
                )
                if plan.get("classification") != "exact_execution_fork":
                    branch = _unavailable(candidate, plan.get("reasons"), exactness=plan.get("exactness"))
                else:
                    execution = execute_exact_force_token(
                        parent_run, plan, engine,
                        runtime_identity=dict(runtime_identity), worker_identity=dict(worker_identity),
                        reload_parent=reload_parent, cancel_check=cancel_check,
                    )
                    receipt = execution.get("receipt") or {}
                    child = execution.get("child")
                    if receipt.get("phase") == "completed" and isinstance(child, Mapping) and child.get("id"):
                        branch = _completed_exact(candidate, child, receipt, parent_run)
                    else:
                        branch = _unavailable(
                            candidate, receipt.get("reasons"), exactness=receipt.get("exactness"),
                            unchanged_control=receipt.get("unchanged_control"),
                        )
                        if receipt.get("execution_id"):
                            branch["execution_fork_execution_id"] = receipt["execution_id"]
                        if receipt.get("phase") == "cancelled":
                            cancelled = True
            except Exception:
                branch = _unavailable(candidate, [_reason("exact_execution_failed", "exact execution failed")])
            branches.append(branch)
            code = _branch_failure_code(branch)
            if code in _SHARED_FAILURE_CODES:
                stop_scheduling = ("shared_exact_precondition_failed", "later branches were not attempted after a shared exact precondition failed")
            if cancelled:
                for rest in candidates[offset + 1:]:
                    branches.append(_not_attempted(rest, "branch_fan_cancelled", "branch fan cancelled after a child attempt"))
                break
            continue

        # Missing numeric ids (or an unavailable shared checkpoint) take the same reconstructed
        # planner/reconstructed path. No candidate is invented or rescored here.
        try:
            branch = _run_reconstructed(
                parent_run, sub, candidate, position, remaining, runtime_identity, worker_identity)
        except Exception:
            branch = _unavailable(candidate, [_reason("reconstructed_execution_failed", "reconstructed replay failed")])
        branches.append(branch)

    if cancelled:
        status = "partial_cancelled" if any(branch.get("state") == "completed" for branch in branches) else "cancelled"
    elif any(branch.get("state") == "not_attempted" for branch in branches):
        status = "partial" if any(branch.get("state") == "completed" for branch in branches) else "unavailable"
    elif any(branch.get("state") == "completed" for branch in branches):
        status = "completed" if all(branch.get("state") == "completed" for branch in branches) else "partial"
    else:
        status = "unavailable"
    result = {
        "schema_version": SCHEMA_VERSION,
        "parent_run_id": parent_id,
        "position": position,
        "selection": selection,
        "execution": _execution_base(
            capture_state, reused=checkpoint_reference is not None,
            fidelity=_fidelity(branches), reason=capture_reason),
        "branches": branches,
        "summary": _summary(branches, len(candidates), status=status),
    }
    schemas.validate(result, SCHEMA_VERSION)
    return result


__all__ = [
    "DEFAULT_LIMIT", "MAX_LIMIT", "MIN_LIMIT", "BranchFanInputError", "branch_fan",
    "comparison_projection", "comparison_projection_from_diff", "recorded_alternatives_available",
    "recorded_alternative_candidates",
]
