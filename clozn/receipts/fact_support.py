"""Fact extraction and support matching for accountable narration."""
from __future__ import annotations

import re


def _as_dict(x) -> dict:
    return x if isinstance(x, dict) else {}


def _as_list(x) -> list:
    return x if isinstance(x, list) else []


def _citable_facts(explanation: dict) -> list[dict]:
    """Flatten an explain() object into citeable facts for constrained narration."""
    explanation = _as_dict(explanation)
    facts: list[dict] = []

    conf = _as_dict(explanation.get("confidence"))
    for m in _as_list(conf.get("uncertain_moments")):
        if not isinstance(m, dict):
            continue
        idx = m.get("index")
        fid = f"hesitation:{idx}" if idx is not None else f"hesitation:{len(facts)}"
        alt_pieces = [a.get("piece", a) if isinstance(a, dict) else a for a in _as_list(m.get("alternatives"))]
        alt_text = ", ".join(str(a) for a in alt_pieces if a)
        text = f'wavered on "{m.get("token", "")}"'
        if alt_text:
            text += f" (also considered: {alt_text})"
        facts.append({"id": fid, "text": text, "category": "hesitation"})

    concepts = _as_dict(explanation.get("concepts"))
    for si, span in enumerate(_as_list(concepts.get("spans"))):
        if not isinstance(span, dict):
            continue
        for fi, feat in enumerate(_as_list(span.get("features"))):
            if not isinstance(feat, dict):
                continue
            fid = feat.get("id") or f"concept:{si}:{fi}"
            text = str(feat.get("label") or fid)
            facts.append({"id": str(fid), "text": text, "category": "concept"})

    return facts


def _influence_lexicon(explanation: dict) -> list[dict]:
    """The narrower influence evidence set used by lexical_default. Memory cards and named-dial
    personalization are both retired, and influences_active carries no other named-influence field
    (only the section manifest, which this lexicon does not index), so this is always empty now --
    kept as its own function rather than inlined so lexical_default's contract stays unchanged if a
    future influence source is added here."""
    return []


_STOPWORDS = frozenset(
    "a an the to of and or in on for with is are i you it that this be as your my we do have can will what "
    "how not but so if they them their there here just like about into because since s t re ve ll d m he "
    "she his her its our us was were been being at by from also".split()
)
_WORD_RE = re.compile(r"[a-z0-9']+")


def _words(text) -> set[str]:
    return {w for w in _WORD_RE.findall(str(text or "").lower()) if len(w) > 2 and w not in _STOPWORDS}


def lexical_default(claim: str, explanation: dict) -> dict:
    """Weak model-free support matcher based on token overlap with the influence lexicon."""
    claim_words = _words(claim)
    lexicon = _influence_lexicon(explanation)
    if not claim_words or not lexicon:
        return {"supported": False, "matched_ids": [], "matched_terms": []}

    matched_ids: list[str] = []
    matched_terms: set[str] = set()
    for fact in lexicon:
        overlap = claim_words & _words(fact["text"])
        if overlap:
            matched_ids.append(fact["id"])
            matched_terms |= overlap
    return {"supported": bool(matched_ids), "matched_ids": matched_ids, "matched_terms": sorted(matched_terms)}


def semantic_support_matcher(claim: str, explanation: dict) -> dict:
    """Deferred gated semantic support matcher; intentionally not a fake implementation."""
    raise NotImplementedError(
        "semantic_support_matcher is the deferred M4 on-model pass (EXPLAIN_THIS_ANSWER_SPEC.md M4). "
        "Pass a real callable via the support_matcher parameter instead."
    )
