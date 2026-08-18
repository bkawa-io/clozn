"""semantic_matcher.py -- the REAL support matcher for narrate.py's confabulation-diff (EXPLAIN_THIS_ANSWER
_SPEC.md M4, the deferred on-model pass).

narrate.py ships `confabulation_diff` with a pluggable `support_matcher` and a DELIBERATELY WEAK default
(`lexical_default`, keyword overlap), and leaves the real judgment as a documented, RAISING hook
(`narrate.semantic_support_matcher`) -- because deciding whether a self-narration claim is actually supported
by the measured receipts is a semantic judgment this project's own findings say a model can be confidently
WRONG about (research/FINDINGS.md law #1: content is legible, PROCESS is not). This module is that deferred
pass: a real matcher you pass via `support_matcher=`, validated by a gated test that seeds a KNOWN divergence
and proves the diff catches it (the spec's M4 "Done" line).

    narrate.narrate(run, sub, support_matcher=semantic_matcher.nli_support_matcher)

THE JUDGE IS INDEPENDENT ON PURPOSE. The matcher is a cross-encoder NLI model
(cross-encoder/nli-deberta-v3-base) -- a DIFFERENT model from the Qwen being audited. That independence is
the point: handing "was my reasoning X?" back to the audited model is exactly the self-consistency /
sycophancy trap law #1 warns about (a model fluently rationalizes its own output). A third-party entailment
model has no stake in the answer; it only judges "does premise P entail hypothesis H?".

WHY NLI, NOT EMBEDDING COSINE. The cheaper option -- cosine similarity from the on-disk MiniLM sentence
embedder (topic_gate) -- was MEASURED and REJECTED: on the seed cases it separated supported claims from
confabulations by a margin of only ~0.02 (worst true 0.306 vs best confab 0.283), because cosine measures
TOPICAL overlap, not entailment ("I took the weather into account" is structurally similar to a terse
instruction whether or not it is true). The cross-encoder, asked entailment(premise=receipt, hypothesis=
claim), separated the SAME cases by ~0.83 (true claims 0.83-0.99 entailment, confabulations <=0.001). That
measurement -- not a preference -- is why this module carries the heavier dependency; the receipt is in
test_semantic_matcher_gated.py.

THE HONEST CEILING (do not oversell this). A cross-encoder entailment score is a strong separator for the
failure this module targets -- a fluent claim crediting an influence that was never on record -- and it did
model per-influence ATTRIBUTION when there was more than one named influence to attribute to (measured,
asserted in the gated test, back when memory cards and named steering dials were both live evidence
sources; both are retired now -- see _premises). It is still NOT a complete honesty oracle:
  * it judges one (receipt, claim) pair at a time; a COMPOUND claim ("concise because you asked, and warm
    because I like you") is now split by narrate.clause_split (the default claim_splitter) into separately
    judged clauses, so a half-confabulation no longer hides behind a supported partner clause -- but that
    splitter is a documented heuristic (narrate._DIFF_NOTE), not a parser, so nested/implicit coordination
    can still under-split;
  * "unsupported" conflates "no receipt entails this" with "a receipt CONTRADICTS this" -- the raw
    contradiction probability is returned alongside (`contradiction`) so a caller CAN tell them apart, but the
    boolean does not;
  * the premise is only as good as the manifest text -- a receipt with no human-readable text (a bare concept
    id) gives the NLI nothing to reason over, and is skipped;
  * one model, one threshold, English; the threshold (0.5, the entailment-class midpoint) is chosen against a
    wide measured margin, NOT learned, and is exposed as a parameter.

Lazy + degrade, exactly like topic_gate: the ~440MB checkpoint loads on first use; if sentence-transformers
or the model is unavailable the matcher returns a LABELED unsupported result (`method: "nli-unavailable"`)
rather than crashing -- fail closed, but SAY so, so a caller can fall back to lexical_default instead of
silently trusting a blank judge. No torch at import time (every heavy import is inside a function), so
`import semantic_matcher` stays cheap and the plain (non `-m model`) test suite never drags a checkpoint at
collection -- mirrors topic_gate.py's discipline.
"""
from __future__ import annotations

import threading

_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
# entailment-class midpoint. The measured true/confab entailment margin is ~0.83 wide (gated test), so the
# exact cut is not delicate; 0.5 = "the NLI calls this more-likely-than-not an entailment".
NLI_THRESHOLD = 0.5

_model = None
_ent_idx = 1        # resolved from the checkpoint's id2label on load; these are the deberta-v3-nli defaults
_contra_idx = 0
_ok = True
_lock = threading.Lock()


def _ensure_model() -> bool:
    """Load the cross-encoder once (thread-safe, double-checked). Returns True if usable, False (and latches
    _ok=False) if sentence-transformers or the checkpoint is unavailable -- after which every call degrades to
    a labeled 'nli-unavailable' unsupported result instead of raising."""
    global _model, _ent_idx, _contra_idx, _ok
    if _model is not None:
        return True
    if not _ok:
        return False
    with _lock:
        if _model is not None:
            return True
        try:
            import os
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")   # this machine: HF symlinks crash (CLAUDE.md)
            from sentence_transformers import CrossEncoder
            m = CrossEncoder(_MODEL_NAME)
            id2label = dict(m.model.config.id2label)
            ent = [i for i, l in id2label.items() if "entail" in str(l).lower()]
            con = [i for i, l in id2label.items() if "contradict" in str(l).lower()]
            _ent_idx = ent[0] if ent else 1
            _contra_idx = con[0] if con else 0
            _model = m
        except Exception:
            _ok = False
            _model = None
            return False
    return True


def available() -> bool:
    """True if the cross-encoder is loaded/loadable. A caller that wants to fall back to lexical_default when
    the NLI checkpoint is missing branches on this rather than getting fail-closed 'unsupported' everywhere."""
    return _ensure_model()


def _premises(explanation: dict) -> list[tuple[str, str]]:
    """Turn an M1 (explain.explain) object's ACTIVE INFLUENCES into [(id, premise_sentence)] -- the evidence
    each claim is checked against. Memory cards and named-dial personalization -- the two sources this used
    to read (card text/quoted words, and a sentence naming the steered tone) -- are both retired, and
    influences_active carries no other named-influence field (only the section manifest, which this does
    not index), so this is always empty now. Hesitations and concepts were never premises here either: "I
    hesitated" is not an INFLUENCE the model is crediting, and a bare concept id has no text to entail from.
    Kept as its own function rather than inlined so nli_support_matcher's contract (an honest "no measured
    influence on record" degrade) stays unchanged if a future influence source is added here."""
    return []


def _entail_scores(premises: list[tuple[str, str]], claim: str) -> list[tuple[str, float, float]]:
    """Batched (entailment, contradiction) probability of `claim` given each premise. Returns
    [(id, entail_prob, contra_prob), ...]. numpy is imported locally (kept out of module import). Assumes
    _ensure_model() already succeeded."""
    import numpy as np
    pairs = [(prem, claim) for _, prem in premises]
    logits = np.array(_model.predict(pairs), dtype="float64")
    if logits.ndim == 1:
        logits = logits[None, :]
    ex = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = ex / ex.sum(axis=1, keepdims=True)
    return [(fid, float(row[_ent_idx]), float(row[_contra_idx])) for (fid, _), row in zip(premises, probs)]


def _normalize_claim(claim: str) -> str:
    """Make a claim look like a well-formed SENTENCE to the cross-encoder. MEASURED fragility: the NLI model
    is punctuation-sensitive on bare clause fragments -- "I answered concisely." entails the concise card at
    0.842, but the period-less fragment "I answered concisely" (exactly what `narrate.clause_split` hands over
    from the middle of a compound sentence) scores only 0.234, while "I kept my answer concise" scores 0.958
    -- so it is the fragment SHAPE, not the meaning. Appending a terminal period recovers the entailment. Any
    claim not already ending in .!? gets one before scoring; the claim TEXT returned to the caller is
    unchanged (this only affects what the model sees)."""
    c = (claim or "").strip()
    return c if (c and c[-1] in ".!?") else c + "."


def nli_support_matcher(claim: str, explanation: dict, threshold: float = NLI_THRESHOLD) -> dict:
    """THE real support_matcher: is `claim` (one atomic self-narration claim from the unconstrained "why")
    entailed by any measured influence in `explanation`? Drop-in for narrate.confabulation_diff's
    `support_matcher` parameter -- narrate(run, sub, support_matcher=semantic_matcher.nli_support_matcher).

    For each active influence, score entailment(premise=receipt, hypothesis=claim) with the cross-encoder; the
    claim is SUPPORTED iff the best entailment >= threshold. Returns (extra keys flow through
    confabulation_diff onto the claim's entry, for transparency):
        {"supported": bool,
         "score": <best entailment prob over all influences>,
         "matched_id": <the influence that supports it, or None when unsupported>,
         "closest_id": <highest-entailment influence regardless of threshold, for a "closest was X" trace>,
         "contradiction": <that influence's contradiction prob -- lets a caller tell "contradicted" from
                           "no receipt", which the boolean cannot>,
         "method": "nli-deberta-v3", "threshold": <threshold>}

    Degrades LABELED, never raises:
      * no influence text to check -> supported False, method "nli" (an honest "no receipt for that" -- the
        safe default, same as lexical_default on an empty explanation);
      * model unavailable          -> supported False, method "nli-unavailable" (fail CLOSED but say so, so a
        caller can fall back to lexical_default rather than silently trust a blank judge);
      * a prediction that throws    -> supported False, method "nli-error"."""
    def _degrade(method: str, note: str) -> dict:
        return {"supported": False, "score": 0.0, "matched_id": None, "closest_id": None,
                "contradiction": 0.0, "method": method, "threshold": threshold, "note": note}

    premises = _premises(explanation)
    if not premises:
        return _degrade("nli", "no measured influence on record")
    if not isinstance(claim, str) or not claim.strip():
        return _degrade("nli", "empty claim")
    if not _ensure_model():
        return _degrade("nli-unavailable", "cross-encoder unavailable; caller should fall back to lexical_default")
    try:
        scores = _entail_scores(premises, _normalize_claim(claim))   # normalize fragments -> well-formed sentence
    except Exception:
        return _degrade("nli-error", "prediction failed")

    best_id, best_ent, best_contra = max(scores, key=lambda t: t[1])
    supported = bool(best_ent >= threshold)
    return {"supported": supported,
            "score": round(best_ent, 4),
            "matched_id": best_id if supported else None,
            "closest_id": best_id,
            "contradiction": round(best_contra, 4),
            "method": "nli-deberta-v3",
            "threshold": threshold}
