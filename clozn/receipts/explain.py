"""explain.py -- "Explain this answer" Milestone 1 (EXPLAIN_THIS_ANSWER_SPEC.md): assemble a run's
ALREADY-LOGGED free signals into one structured `explanation` object. ZERO model calls, zero generation --
pure read + reshape of what `runlog`/`behavior` already captured for that turn. Per the
spec: "the missing work is assembly ... not new primitives" -- this module is exactly that assembly, done
once in a single testable place so the endpoint (clozn_server.py) stays a thin dispatcher.

The three panels (mirrors the spec's principle section -- every panel answers a question the model cannot
be trusted to answer about itself):
  * confidence         -- the token trace's "uncertain moments" (tokens below LOW_CONF), each with its
                          recorded alternatives, plus a one-line "N hesitations" count. NEVER a single
                          aggregate confidence % -- that scalar self-report probe is dead (it saturates at
                          every scale -- a measured dead end). {"available": false, ...}
                          when the run carries no per-token trace at all (the HF chat path may not).
  * influences_active -- the run's `sections` manifest (prompt sections: rag_context/system-prompt/etc.,
                          each with its name/source/char_count/preview -- see clozn.runs.sections).
                          Memory cards, anchored memory, and named-dial personalization were all cut
                          from the product, so sections are the whole story now. Every section entry is
                          tagged causal_verified:null -- ACTIVE is not PROOF; only M2's on-demand
                          ablation receipt may ever set this true. This module never runs that receipt.
                          A run recorded before section capture landed carries no `sections` field at
                          all -- surfaced as an explicit {"available": false, "note": ...}, never a
                          silently-empty list.
  * concepts           -- the engine's sae:<id> feature readouts, when a run happens to carry them.
                          {"available": false, ...} otherwise (true today for every run: no current
                          logging path threads concept readouts onto the stored record -- see the
                          docstring on _concepts below). Written forward-compatible so the day capture
                          lands, this assembly needs no changes.

Honesty invariants enforced HERE (so the endpoint can't drift from the spec by construction):
  * no aggregate confidence number, ever, anywhere in the returned object.
  * every active-influence entry carries causal_verified: null.
  * a missing signal is an explicit {"available": false, "note": "..."} field -- never a silently absent
    key ("no receipt for that" is a first-class answer, per the spec).

Mirrors replay.py: stdlib-only -- no model,
no substrate, no GPU. This whole module runs on a plain `run` dict (e.g. from runlog.get_run()) and is
fully unit-testable against fixture dicts. Never raises: `explain()` degrades field-by-field on anything
missing or malformed, so one bad field can't blank out the panels that DID assemble cleanly.
"""
from __future__ import annotations

from clozn.runs import close_calls

# Matches the run timeline and confidence-span `LOW_CONF` values -- one convention, so every recorded
# hesitation and the Studio Explain panel's uncertain moments agree about what counts as unsure.
LOW_CONF = 0.5

_NO_TRACE_NOTE = "token trace captured on the engine path"
_NO_CONCEPTS_NOTE = "no named concept readout was captured for this run."
_NO_SECTIONS_NOTE = ("no section manifest was captured for this run (predates section capture, or this "
                     "run's source doesn't produce one).")


def _as_list(x) -> list:
    return x if isinstance(x, list) else []


def _as_dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _card_text(c) -> str:
    """cards_applied entries are plain strings on every current path (_log_run's prompt AND internalized
    branches both store texts, not dicts) -- but tolerate a {"text": ...} shape too, defensively, exactly
    like the timeline assembler does, so a future shape change degrades instead of breaking the endpoint."""
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return str(c.get("text", ""))
    return "" if c is None else str(c)


# --------------------------------------------------------------------------------------------- confidence
def _confidence(run: dict) -> dict:
    """The K "uncertain moments" -- tokens whose recorded confidence fell below LOW_CONF -- each with its
    recorded alternatives, plus a one-line "N hesitations" count. NEVER a single aggregate %: this function
    only ever emits per-token numbers that were already measured at generation time, or plain counts --
    never a synthesized overall "confidence score" (the principle this whole endpoint exists to honor)."""
    trace = _as_dict(run.get("trace"))
    tokens = _as_list(trace.get("tokens"))
    if not tokens:
        # The HF chat path (and any run that predates trace capture) has no per-token record at all. An
        # explicit, honestly-labeled field beats a silently-missing key.
        return {"available": False, "note": _NO_TRACE_NOTE}
    confidence = _as_list(trace.get("confidence"))
    alternatives = _as_list(trace.get("alternatives"))
    uncertain = []
    for i, tok in enumerate(tokens):
        try:
            c = float(confidence[i]) if i < len(confidence) else None
        except (TypeError, ValueError):
            c = None
        if c is None or c >= LOW_CONF:
            continue
        alts = alternatives[i] if i < len(alternatives) and isinstance(alternatives[i], list) else []
        uncertain.append({"index": i, "token": tok, "confidence": round(c, 4), "alternatives": alts})
    n = len(uncertain)
    return {
        "available": True,
        "threshold": LOW_CONF,
        "n_tokens": len(tokens),
        "uncertain_moments": uncertain,
        "summary": f"{n} hesitation{'' if n == 1 else 's'}",
    }


# ---------------------------------------------------------------------------------------- influences_active
def _section_entry(sec: dict) -> dict:
    """One section from the run's manifest, reshaped for display: id/name/source/char_count/preview,
    always tagged causal_verified:null (ACTIVE is not PROOF; only M2's on-demand section-ablation
    receipt, `exclude_sections`, may ever prove one caused the reply, and it never runs from here).
    Guards every field individually so one malformed section in the manifest can't take the others
    down with it."""
    return {
        "id": sec.get("id"),
        "name": sec.get("name"),
        "source": sec.get("source"),
        "char_count": sec.get("char_count"),
        "preview": sec.get("preview"),
        "causal_verified": None,
    }


def _sections_active(run: dict) -> dict:
    """The run's `sections` manifest (prompt sections -- rag_context/system-prompt/etc., the fixed
    schema clozn.runs.sections/store produce), reshaped into the same {"available": ..., ...} degrade
    shape as `_confidence`/`_concepts`: a run predating section capture (or whose source never produces
    one) carries no `sections` field at all, and that is reported honestly rather than as a silently
    empty list -- there's a real difference between "captured, and there were none" (not currently
    distinguished further here, since the manifest producer always omits the field entirely when it has
    nothing to say) and "never captured"."""
    manifest = run.get("sections")
    if not isinstance(manifest, list):
        return {"available": False, "note": _NO_SECTIONS_NOTE}
    sections = [_section_entry(s) for s in manifest if isinstance(s, dict)]
    return {"available": True, "sections": sections}


def _influences_active(run: dict) -> dict:
    """The run's section manifest -- the only logged "what shaped this reply" signal left.

    Memory cards, anchored memory, and named-dial personalization were all cut from the product, so
    `sections` (which PARTS of the prompt could have shaped the reply) is the whole story now. Every
    section entry is tagged causal_verified:null -- ACTIVE is not PROOF, and only an on-demand ablation
    receipt may ever flip it."""
    return {"sections": _sections_active(run)}


# -------------------------------------------------------------------------------------------------- concepts
def _concepts(run: dict) -> dict:
    """Top SAE features per span, when the run happens to carry them (the engine path's sae:<id> readouts;
    see engine/core/serve/clozn_server.cpp's StepFeatures). Honest as of this writing: NO current logging
    path threads concept readouts onto the stored run record -- runlog._norm_trace only keeps
    tokens/confidence/alternatives (runlog.TRACE_KEYS), so this is, today, always the "not available"
    branch. Written forward-compatible (reads trace["concepts"], falling back to a top-level run["concepts"])
    so the day capture wiring lands here, this assembly function needs no changes -- only the producer does.
    """
    trace = _as_dict(run.get("trace"))
    spans = trace.get("concepts", run.get("concepts"))
    if not isinstance(spans, list) or not spans:
        return {"available": False, "note": _NO_CONCEPTS_NOTE}
    cleaned = []
    for span in spans:
        if not isinstance(span, dict):
            continue
        feats = [f for f in _as_list(span.get("features")) if isinstance(f, dict)]
        feats = sorted(feats, key=lambda f: f.get("score", 0) or 0, reverse=True)[:5]   # top features per span
        entry = {"features": feats}
        if "position" in span:
            entry["position"] = span["position"]
        if "piece" in span:
            entry["piece"] = span["piece"]
        cleaned.append(entry)
    if not cleaned:
        return {"available": False, "note": _NO_CONCEPTS_NOTE}
    return {"available": True, "spans": cleaned}


# ----------------------------------------------------------------------------------------------------- forks
def _forks(run: dict) -> dict:
    """Close calls -- steps where the two co-leading tokens near-tied (a coin-flip fork). CORRELATIONAL
    locator, never a verdict: it points at WHERE a branch-stability test would pay off, never claims the
    step was "wrong" or "fragile". Reconstructs each step's distribution from {emitted} u alternatives (the
    emitted token is excluded from `alternatives`), so a fork is chosen-vs-strongest-rival. `meaningful_count`
    is the answer-changing slice (digit forks + polarity flips); most forks are harmless phrasing splits."""
    trace = _as_dict(run.get("trace"))
    if not trace or not _as_list(trace.get("alternatives")):
        return {"available": False, "note": _NO_TRACE_NOTE}
    calls = close_calls.close_calls(run)
    return {"available": True, "forks": calls,
            "meaningful_count": len(close_calls.meaningful(calls)),
            "summary": close_calls.summarize(calls)}


# ------------------------------------------------------------------------------------------------------ API
def explain(run: dict | None) -> dict:
    """Assemble the M1 explanation object for one run (as returned by runlog.get_run()). Never raises: a
    non-dict `run` (None, a stray string, ...) degrades to a fully-honest empty explanation rather than
    erroring, and a failure assembling ONE panel never blanks out the others -- each is guarded separately."""
    run = run if isinstance(run, dict) else {}
    try:
        confidence = _confidence(run)
    except Exception:
        confidence = {"available": False, "note": _NO_TRACE_NOTE}
    try:
        influences = _influences_active(run)
    except Exception:
        influences = {"sections": {"available": False, "note": _NO_SECTIONS_NOTE},
                      "note": "influence manifest unavailable"}
    try:
        concepts = _concepts(run)
    except Exception:
        concepts = {"available": False, "note": _NO_CONCEPTS_NOTE}
    try:
        forks = _forks(run)
    except Exception:
        forks = {"available": False, "note": _NO_TRACE_NOTE}
    return {
        "run_id": run.get("id"),
        "confidence": confidence,
        "influences_active": influences,
        "concepts": concepts,
        "forks": forks,
    }
