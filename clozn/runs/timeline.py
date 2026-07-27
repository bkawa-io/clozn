"""run_timeline.py -- reshape a stored run (runlog.get_run()) into an ordered list of typed, semantic
RunEvent dicts: what happened, in order, in plain language, for the Studio's Run Inspector timeline. Sibling
of explain.py (same contract shape: a pure `timeline(run) -> list[dict]`, zero model calls, zero
generation, only assembly of signals runlog already captured for that turn) -- but where explain.py answers
"why" (confidence / influences / concepts, each its own standalone panel), this module answers "what
happened, when" as one ordered sequence a UI can render as a timeline strip.

This is a RESHAPE, not a new signal: every field emitted below already lives on the run record (runlog.py's
schema) or one of its sub-dicts (`trace`, `memory`, `behavior`, `timing`). Nothing here is invented,
estimated, or synthesized -- an event fires only when its underlying data actually exists on the run.

The event taxonomy, in emission order (each only when its data is present -- see `timeline()`'s guards):
  1. run_started    -- once, always (for a non-empty run): source/client/model/created_at.
  2. branched_from  -- only when parent_run_id is set (this run is a replay/branch of another).
  3. memory_applied -- only when memory.cards_applied is non-empty; one card entry per applied card,
                       zipped against applied_ids / relevance BY INDEX (a short or missing list is None for
                       that slot -- e.g. internalized mode logs no applied_ids at all -- never fabricated).
  4. dials_applied  -- only when behavior.active_dials is non-empty.
  5. generation     -- only when there's a response or a trace: token count (from the trace, when there is
                       one) else a word count of the response -- NEVER presented as if it were a real token
                       count -- plus wall time from timing.duration_ms.
  6. hesitation     -- one event per token whose trace confidence fell below LOW_CONF, carrying that
                       token's recorded alternatives. Same threshold as explain.py's uncertain_moments (see
                       LOW_CONF below): the Explain panel and this timeline must never disagree about what
                       counts as "unsure".
  7. finished       -- when the run carries no error: the stop cause (finish_reason), or an honest
                       "Finished" with finish_reason: None when the run never recorded one.
  8. error          -- when the run carries one; mutually exclusive with `finished` (an errored run has no
                       stop cause to report).

Zero imports beyond stdlib typing syntax: unlike explain.py this module never resolves card provenance
(memory_cards) -- cards_applied/applied_ids/relevance already ride flat on the run's own `memory` block, and
a timeline event only needs to say a card fired THIS turn, not re-prove its provenance. That keeps this
module trivially unit-testable against bare fixture dicts, no card-store isolation required.

Never raises: `timeline()` reduces a non-dict (or empty) run to [] rather than erroring, and each event kind
is assembled in its own guarded step so one malformed section drops only that one event -- never blanks the
rest of the timeline (mirrors explain.py's per-panel try/except).
"""
from __future__ import annotations

# Matches explain.py's LOW_CONF -- ONE "unsure" cutoff
# read in all three places, so this timeline's hesitation events and the Explain panel's uncertain_moments
# never disagree about which tokens counted as a hesitation. Kept as its own constant (not imported from
# explain) so this module stays a zero-dependency sibling -- if one changes, change the other.
LOW_CONF = 0.5


def _as_list(x) -> list:
    return x if isinstance(x, list) else []


def _as_dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _card_text(c) -> str:
    """cards_applied entries are plain strings on every current logging path, but tolerate a {"text": ...}
    shape too, defensively -- mirrors explain._card_text."""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return str(c.get("text", ""))
    return "" if c is None else str(c)


def _rounded(x):
    """A relevance float, rounded to 4dp like runlog does -- or None for anything missing/unusable."""
    try:
        return round(float(x), 4) if x is not None else None
    except (TypeError, ValueError):
        return None


def _trace_steps(trace: dict) -> list[dict]:
    """Return rich token steps when present; synthesize the old v1 arrays into the same shape otherwise.

    The synthesis also folds in the v2 parallel arrays (token_ids/logprobs/topk_entropy -- see runlog.py's
    TRACE_KEYS docstring for exactly what each means/how approximate it is) when a trace carries them
    without a `steps` list. A trace persisted before v2 (or any trace missing these arrays) simply has no
    such keys to read -- .get() below degrades that to nothing rather than raising."""
    trace = _as_dict(trace)
    steps = _as_list(trace.get("steps"))
    if steps and all(isinstance(s, dict) for s in steps):
        return steps
    tokens = _as_list(trace.get("tokens"))
    confidence = _as_list(trace.get("confidence"))
    alternatives = _as_list(trace.get("alternatives"))
    token_ids = _as_list(trace.get("token_ids"))
    logprobs = _as_list(trace.get("logprobs"))
    topk_entropy = _as_list(trace.get("topk_entropy"))
    out = []
    for i, piece in enumerate(tokens):
        step = {"index": i, "piece": piece, "text": piece,
                "alternatives": alternatives[i] if i < len(alternatives) and isinstance(alternatives[i], list) else []}
        if i < len(confidence):
            step["confidence"] = confidence[i]
            step["prob"] = confidence[i]
        if i < len(token_ids) and token_ids[i] is not None:
            step["token_id"] = token_ids[i]
        if i < len(logprobs) and logprobs[i] is not None:
            step["logprob"] = logprobs[i]
        if i < len(topk_entropy) and topk_entropy[i] is not None:
            step["topk_entropy"] = topk_entropy[i]
        out.append(step)
    return out


# --------------------------------------------------------------------------------------------- memory_applied
def _memory_applied(run: dict) -> dict | None:
    """One event for the memory cards compiled into this turn's prompt. None when none were."""
    mem = _as_dict(run.get("memory"))
    texts = _as_list(mem.get("cards_applied"))
    if not texts:
        return None
    ids = _as_list(mem.get("applied_ids"))
    rels = _as_list(mem.get("relevance"))
    cards = [{"text": _card_text(t), "id": ids[i] if i < len(ids) else None,
              "relevance": _rounded(rels[i]) if i < len(rels) else None} for i, t in enumerate(texts)]
    n_cards = len(cards)
    return {"type": "memory_applied", "label": f"{n_cards} memory card{'' if n_cards == 1 else 's'} applied",
            "count": n_cards, "gate": mem.get("gate"), "mode": mem.get("mode"), "cards": cards}


# ---------------------------------------------------------------------------------------------- dials_applied
def _dials_applied(run: dict) -> dict | None:
    """One event for the whole set of active tone dials this turn. None when no dial is active."""
    behavior = _as_dict(run.get("behavior"))
    dials = _as_dict(behavior.get("active_dials"))
    if not dials:
        return None
    n = len(dials)
    return {"type": "dials_applied", "label": f"{n} behavior dial{'' if n == 1 else 's'}", "dials": dict(dials)}


# ------------------------------------------------------------------------------------------------- generation
def _generation(run: dict) -> dict | None:
    """Token count -- real, from the trace, when there is one -- else an honestly-labeled fallback (a word
    count of the response; never presented as if it were a token count). None when there's neither a
    response nor a trace to report on at all."""
    response = run.get("response")
    trace = _as_dict(run.get("trace"))
    if not response and not trace:
        return None
    steps = _trace_steps(trace)
    tokens = _as_list(trace.get("tokens"))
    n = len(steps) if steps else (len(tokens) if tokens else len(str(response or "").split()))
    duration = _as_dict(run.get("timing")).get("duration_ms")
    return {"type": "generation", "label": f"Generated {n} tokens", "n_tokens": n, "duration_ms": duration}


# ------------------------------------------------------------------------------------------------- hesitation
def _hesitations(run: dict) -> list[dict]:
    """One `hesitation` event per token whose recorded confidence fell below LOW_CONF, carrying that
    token's recorded alternatives. Mirrors explain._confidence's uncertain-moments loop exactly (same
    threshold, same tolerance for a missing/malformed confidence or alternatives entry) -- just emitted as
    ordered timeline events instead of folded into one aggregate panel. [] when there's no per-token trace."""
    trace = _as_dict(run.get("trace"))
    steps = _trace_steps(trace)
    if not steps:
        return []
    out = []
    for i, step in enumerate(steps):
        piece = step.get("piece", step.get("text", ""))
        try:
            c = float(step.get("confidence", step.get("prob"))) if step.get("confidence", step.get("prob")) is not None else None
        except (TypeError, ValueError):
            c = None
        if c is None or c >= LOW_CONF:
            continue
        alts = step.get("alternatives")
        alts = alts if isinstance(alts, list) else []
        ev = {"type": "hesitation", "label": f'Unsure at "{piece}"',
              "index": step.get("index", i), "token": piece,
              "confidence": round(c, 4), "prob": round(c, 4), "alternatives": alts}
        for k in ("token_id", "logprob", "entropy", "topk_entropy", "wall_ms", "dt_ms"):
            if step.get(k) is not None:
                ev[k] = step[k]
        out.append(ev)
    return out


# ------------------------------------------------------------------------------------------------------ API
def timeline(run: dict | None) -> list[dict]:
    """Assemble the ordered RunEvent list for one run (as returned by runlog.get_run()). Never raises: a
    non-dict or empty `run` degrades to [], and each event kind is assembled in its own try/except so one
    malformed section drops only that event -- never blanks the rest of the timeline."""
    run = run if isinstance(run, dict) else {}
    if not run:
        return []
    events: list[dict] = []

    try:
        events.append({"type": "run_started", "label": "Run started", "source": run.get("source"),
                       "client": run.get("client"), "model": run.get("model"), "at": run.get("created_at")})
    except Exception:
        pass

    try:
        if run.get("parent_run_id"):
            events.append({"type": "branched_from", "label": "Branched from an earlier run",
                           "parent_run_id": run["parent_run_id"]})
    except Exception:
        pass

    try:
        ev = _memory_applied(run)
        if ev:
            events.append(ev)
    except Exception:
        pass

    try:
        ev = _dials_applied(run)
        if ev:
            events.append(ev)
    except Exception:
        pass

    try:
        ev = _generation(run)
        if ev:
            events.append(ev)
    except Exception:
        pass

    try:
        events.extend(_hesitations(run))
    except Exception:
        pass

    try:
        err = run.get("error")
        if err:
            events.append({"type": "error", "label": "Error", "message": err})
        else:
            reason = run.get("finish_reason")
            events.append({"type": "finished", "label": f"Finished ({reason})" if reason else "Finished",
                           "finish_reason": reason, "truncated": reason == "length"})
    except Exception:
        pass

    return events
