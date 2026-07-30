"""Map E1's extracted claims to supplied-source evidence (E2).

``clozn.claim-support.v1`` is a DERIVED, read-only projection over one run's ``clozn.answer-claims.v1``
document (E1, ``clozn.runs.claims``) and its persisted influence measurement
(``run["influence_map"]``, ``clozn.context_answer_influence.v1``). The run document is never altered.
Every cited source is a real ``clozn.text-span-addresses.v1`` address -- built with
``clozn.runs.text_span_addresses.project_influence_addresses``, the exact same projection the Sources
lens already uses. This module does not invent a second addressing scheme, and it does not re-derive
``evidence_state``: it reads ``causally_supported`` vs ``observed`` off the influence artifact's own
``Link`` objects exactly as recorded.

THE THREE HONESTY RULES (structural, not incidental)
------------------------------------------------------
1. **Presence is not support.** ``influence_map["prompt_sources"]`` licenses only "this text was in the
   assembled prompt" (its own schema description says so) and is never consulted here. Only
   ``prompt_spans`` (guaranteed real text) and ``links`` (real measurements) are. A claim's textual
   overlap with a prompt span, by itself, caps at ``weakly_supported``. ``supported`` requires a
   ``Link`` whose ``evidence_state == "causally_supported"`` AND ``effect == "supports"`` overlapping
   the claim's own answer range -- the forced-score-intervention evidence trail, not a coincidence of
   words.
2. **Missing evidence is never "unsupported."** If ``run["influence_map"]`` does not exist, did not run
   (``status != "ok"``), or cannot be reconciled with this run's recorded answer text (a hash mismatch,
   or no resolvable answer-span offsets), every factual claim's status is ``measurement_unavailable``.
   ``unsupported_by_supplied_materials`` is reserved for the case where the measurement machinery WAS
   consulted and genuinely found nothing -- a different claim entirely.
3. **"Contradicted" needs explicit evidence, not absence of support.** The gate is narrow and
   deterministic on purpose: a prompt span must share most of the claim's own content words (a high,
   conservative overlap bar -- this is clearly the same fact, not a coincidence) AND either assert a
   disjoint number/date, or carry an explicit negation the claim itself does not. Anything short of
   that stays short of ``contradicted`` -- a false ``contradicted`` is the worst failure this artifact
   can produce, so ambiguous cases fall to ``unsupported_by_supplied_materials`` or
   ``measurement_unavailable``, never to a confident false accusation.

STATUS AND CATEGORY, ONE RULE EACH
-------------------------------------
Non-factual claim categories (``recommendation``, ``uncertainty_statement``, ``instruction_procedure``,
``non_verifiable_prose``) get ``unverifiable_from_available_evidence`` unconditionally -- a category
rule, never a per-claim judgment call, because verification targets checkable factual assertions.
``factual_claim`` is the only category that can produce any of the other five statuses; it can never
land on ``unverifiable_from_available_evidence`` itself.

``unsupported_by_supplied_materials`` never reads as "false" anywhere in this module's own vocabulary or
the schema's field descriptions -- it means exactly "the supplied materials do not support this," which
says nothing about whether the claim is true.

DETERMINISM
-----------
Pure function of ``(run, claims_document)``: no randomness, no wall-clock, no model calls. Every
collection this module emits is either drawn directly from a deterministically-ordered source list or
explicitly sorted before use, so the same input produces byte-identical output, proven in
``tests/test_claim_support.py``.
"""
from __future__ import annotations

import hashlib
import re

from clozn.runs.claims import CATEGORIES as CLAIM_CATEGORIES
from clozn.runs.claims import SCHEMA_VERSION as CLAIMS_SCHEMA_VERSION
from clozn.runs.text_span_addresses import (
    INFLUENCE_SCHEMA,
    OFFSET_CONTRACT,
    project_influence_addresses,
)

SCHEMA_VERSION = "clozn.claim-support.v1"

STATUSES = frozenset({
    "supported",
    "weakly_supported",
    "contradicted",
    "unsupported_by_supplied_materials",
    "unverifiable_from_available_evidence",
    "measurement_unavailable",
})

# Statuses that must carry non-empty `source_span_ids` -- the only ones backed by a specific cited span.
_EVIDENCE_BEARING_STATUSES = frozenset({"supported", "weakly_supported", "contradicted"})

_PROMPT_ADDRESS_KINDS = frozenset({
    "attached_source_span", "delivered_message", "rendered_prompt_segment",
})

# ======================================================================================================
# Overlap heuristics -- stdlib `re` only, deterministic. Two thresholds, on purpose: `weakly_supported`
# is meant to be an inclusive "there is SOME textual connection" signal; `contradicted` requires a much
# higher bar (clearly the same fact) before a mismatch is trusted at all -- see honesty rule 3.
# ======================================================================================================

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "and", "or", "but", "if",
    "of", "to", "in", "on", "at", "for", "with", "as", "by", "from", "that", "this", "these", "those",
    "it", "its", "you", "your", "they", "their", "he", "she", "his", "her", "we", "our", "will",
    "would", "should", "could", "can", "may", "might", "not", "no", "do", "does", "did", "has", "have",
    "had", "about", "into", "than", "then", "so", "such", "also", "there", "here", "which", "who",
    "what", "when", "where", "how", "why",
})
_CONTENT_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_NUMBER_RE = re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")
_APOS = "['’]"
_NEGATION_PATTERNS = (
    r"\bis not\b", rf"\bisn{_APOS}t\b", r"\bwas not\b", rf"\bwasn{_APOS}t\b",
    r"\bare not\b", rf"\baren{_APOS}t\b", r"\bwere not\b", rf"\bweren{_APOS}t\b",
    r"\bdid not\b", rf"\bdidn{_APOS}t\b", r"\bdoes not\b", rf"\bdoesn{_APOS}t\b",
    r"\bdo not\b", rf"\bdon{_APOS}t\b", r"\bno longer\b", r"\bnever\b",
    r"\bcannot\b", rf"\bcan{_APOS}t\b",
)
_NEGATION_RE = re.compile("|".join(f"(?:{pattern})" for pattern in _NEGATION_PATTERNS), re.IGNORECASE)

_WEAK_OVERLAP_THRESHOLD = 0.34
_STRONG_OVERLAP_THRESHOLD = 0.6
_MIN_SHARED_WORDS = 2


def _content_words(text: str) -> frozenset[str]:
    return frozenset(word.lower() for word in _CONTENT_WORD_RE.findall(text)) - _STOPWORDS


def _numbers(text: str) -> frozenset[str]:
    return frozenset(match.group(0).replace(",", "") for match in _NUMBER_RE.finditer(text))


def _overlap_fraction(claim_words: frozenset[str], source_text: str) -> float:
    """Fraction of the CLAIM's own content words echoed in `source_text` -- recall, not Jaccard, so a
    long source span containing a short claim's content is not penalized for its own extra words."""
    if not claim_words:
        return 0.0
    shared = claim_words & _content_words(source_text)
    if len(shared) < _MIN_SHARED_WORDS:
        return 0.0
    return len(shared) / len(claim_words)


def _check_textual_overlap(
    claim_text: str, prompt_text_by_id: dict[str, str], prompt_address_by_id: dict[str, str],
) -> tuple[list[str], float] | None:
    claim_words = _content_words(claim_text)
    matches: list[str] = []
    best = 0.0
    for native_id, text in prompt_text_by_id.items():
        fraction = _overlap_fraction(claim_words, text)
        if fraction < _WEAK_OVERLAP_THRESHOLD:
            continue
        address_id = prompt_address_by_id.get(native_id)
        if address_id is None:
            continue
        matches.append(address_id)
        best = max(best, fraction)
    if not matches:
        return None
    return sorted(set(matches)), round(best, 4)


def _check_contradiction(
    claim_text: str, prompt_text_by_id: dict[str, str], prompt_address_by_id: dict[str, str],
) -> tuple[list[str], str] | None:
    claim_words = _content_words(claim_text)
    claim_numbers = _numbers(claim_text)
    claim_negated = bool(_NEGATION_RE.search(claim_text))
    numeric_matches: list[str] = []
    negation_matches: list[str] = []
    for native_id, text in prompt_text_by_id.items():
        fraction = _overlap_fraction(claim_words, text)
        if fraction < _STRONG_OVERLAP_THRESHOLD:
            continue
        address_id = prompt_address_by_id.get(native_id)
        if address_id is None:
            continue
        source_numbers = _numbers(text)
        if claim_numbers and source_numbers and claim_numbers.isdisjoint(source_numbers):
            numeric_matches.append(address_id)
            continue
        if not claim_negated and _NEGATION_RE.search(text):
            negation_matches.append(address_id)
    if numeric_matches:
        return sorted(set(numeric_matches)), "numeric_or_date_mismatch"
    if negation_matches:
        return sorted(set(negation_matches)), "direct_negation"
    return None


def _check_causal_support(
    overlapping_answer_ids: list[str],
    links_by_answer_id: dict[str, list[dict]],
    prompt_address_by_id: dict[str, str],
) -> tuple[list[str], float] | None:
    address_ids: list[str] = []
    best_delta = 0.0
    for answer_id in overlapping_answer_ids:
        for link in links_by_answer_id.get(answer_id, ()):
            if link.get("effect") != "supports" or link.get("evidence_state") != "causally_supported":
                continue
            context_id = link.get("context_span_id")
            address_id = prompt_address_by_id.get(context_id) if isinstance(context_id, str) else None
            if address_id is None:
                continue
            address_ids.append(address_id)
            delta = link.get("abs_delta_nats")
            if isinstance(delta, (int, float)) and not isinstance(delta, bool):
                best_delta = max(best_delta, float(delta))
    if not address_ids:
        return None
    return sorted(set(address_ids)), best_delta


# ======================================================================================================
# Influence-map gating (honesty rule 2) and geometry resolution
# ======================================================================================================

def _influence_gate(run: dict) -> tuple[dict | None, str | None]:
    """(influence_map, None) when usable, else (None, method_name) naming exactly why not."""
    influence = run.get("influence_map")
    if not isinstance(influence, dict) or not influence:
        return None, "no_influence_map"
    if isinstance(influence.get("unavailable"), str):
        # The blob-backed-reference-not-resolved marker clozn.runs.store.get_run produces -- see
        # text_span_addresses.build_persisted_text_span_addresses' own docstring for this exact shape.
        return None, "no_influence_map"
    if influence.get("schema") != INFLUENCE_SCHEMA:
        return None, "no_influence_map"
    status = influence.get("status")
    if status == "unavailable":
        return None, "influence_measurement_unavailable"
    if status != "ok" or influence.get("available") is not True:
        return None, "influence_measurement_error"
    from clozn import schemas
    try:
        schemas.validate(influence, INFLUENCE_SCHEMA)
    except schemas.ValidationError:
        return None, "influence_measurement_error"
    return influence, None


class _Geometry:
    """Everything a per-claim check needs, once the influence map has passed the gate above."""

    __slots__ = ("answer_offsets", "prompt_address_by_id", "prompt_text_by_id", "links_by_answer_id")

    def __init__(self, answer_offsets, prompt_address_by_id, prompt_text_by_id, links_by_answer_id):
        self.answer_offsets = answer_offsets
        self.prompt_address_by_id = prompt_address_by_id
        self.prompt_text_by_id = prompt_text_by_id
        self.links_by_answer_id = links_by_answer_id


def _resolve_geometry(
    run_id: str, influence_map: dict, response: str, *, privacy: str,
) -> tuple[_Geometry | None, str | None]:
    """(_Geometry, None) on success, else (None, method_name) -- always measurement_unavailable."""
    try:
        addresses = project_influence_addresses(run_id, influence_map, privacy=privacy)
    except (ValueError, TypeError):
        return None, "no_resolvable_answer_spans"

    response_sha256 = hashlib.sha256(response.encode("utf-8")).hexdigest()
    answer_offsets: dict[str, tuple[int, int]] = {}
    prompt_address_by_id: dict[str, str] = {}
    saw_answer_span = False
    saw_hash_mismatch = False
    for address in addresses:
        native_id = address.get("native_ref", {}).get("id")
        if not isinstance(native_id, str):
            continue
        kind = address.get("kind")
        if kind == "answer_span":
            saw_answer_span = True
            canonical = (address.get("resolution") or {}).get("canonical")
            if not isinstance(canonical, dict):
                continue
            if canonical.get("basis_sha256") != response_sha256:
                saw_hash_mismatch = True
                continue
            start, end = canonical.get("start"), canonical.get("end")
            if isinstance(start, int) and isinstance(end, int) and not isinstance(start, bool):
                answer_offsets[native_id] = (start, end)
        elif kind in _PROMPT_ADDRESS_KINDS:
            prompt_address_by_id[native_id] = address["address_id"]

    if not answer_offsets:
        reason = "answer_text_mismatch" if (saw_answer_span and saw_hash_mismatch) else (
            "no_resolvable_answer_spans"
        )
        return None, reason

    prompt_text_by_id: dict[str, str] = {}
    for index, span in enumerate(influence_map.get("prompt_spans") or []):
        if not isinstance(span, dict):
            continue
        native_id = span.get("id")
        native_id = native_id if isinstance(native_id, str) and native_id else f"prompt-span-{index}"
        text = span.get("text")
        if isinstance(text, str):
            prompt_text_by_id[native_id] = text

    links_by_answer_id: dict[str, list[dict]] = {}
    for link in influence_map.get("links") or []:
        if isinstance(link, dict) and isinstance(link.get("answer_span_id"), str):
            links_by_answer_id.setdefault(link["answer_span_id"], []).append(link)

    return _Geometry(answer_offsets, prompt_address_by_id, prompt_text_by_id, links_by_answer_id), None


# ======================================================================================================
# Per-claim status
# ======================================================================================================

def _support_for_claim(claim: dict, *, gate_method: str | None, geometry: _Geometry | None,
                       geometry_failure: str | None, response: str | None) -> dict:
    text_span = claim.get("text_span") or {}
    result = {
        "claim_index": claim.get("index"),
        "claim_address_id": text_span.get("address_id"),
    }

    category = claim.get("category")
    if category != "factual_claim":
        return {**result, "status": "unverifiable_from_available_evidence",
                "method": {"name": "category_rule"}}

    if gate_method is not None:
        return {**result, "status": "measurement_unavailable", "method": {"name": gate_method}}
    if geometry_failure is not None:
        return {**result, "status": "measurement_unavailable", "method": {"name": geometry_failure}}
    assert geometry is not None and response is not None  # both gates above already ruled out None

    canonical = (text_span.get("resolution") or {}).get("canonical") or {}
    claim_start, claim_end = canonical.get("start"), canonical.get("end")
    if not isinstance(claim_start, int) or not isinstance(claim_end, int) or isinstance(claim_start, bool):
        return {**result, "status": "measurement_unavailable",
                "method": {"name": "no_resolvable_answer_spans"}}
    claim_text = response[claim_start:claim_end]

    contradiction = _check_contradiction(
        claim_text, geometry.prompt_text_by_id, geometry.prompt_address_by_id,
    )
    if contradiction is not None:
        address_ids, method_name = contradiction
        return {**result, "status": "contradicted", "method": {"name": method_name},
                "source_span_ids": address_ids}

    overlapping_answer_ids = [
        native_id for native_id, (start, end) in geometry.answer_offsets.items()
        if start < claim_end and end > claim_start
    ]
    support = _check_causal_support(
        overlapping_answer_ids, geometry.links_by_answer_id, geometry.prompt_address_by_id,
    )
    if support is not None:
        address_ids, max_abs_delta_nats = support
        return {**result, "status": "supported",
                "method": {"name": "forced_score_intervention", "max_abs_delta_nats": max_abs_delta_nats},
                "source_span_ids": address_ids}

    weak = _check_textual_overlap(claim_text, geometry.prompt_text_by_id, geometry.prompt_address_by_id)
    if weak is not None:
        address_ids, overlap_fraction = weak
        return {**result, "status": "weakly_supported",
                "method": {"name": "textual_overlap", "overlap_fraction": overlap_fraction},
                "source_span_ids": address_ids}

    return {**result, "status": "unsupported_by_supplied_materials",
            "method": {"name": "measured_comparison_no_match"}}


# ======================================================================================================
# The derived artifact
# ======================================================================================================

def build_claim_support(run: dict, claims_document: dict, *, privacy: str = "metadata_only") -> dict:
    """Build and validate one derived `clozn.claim-support.v1` document.

    `claims_document` must be a `clozn.answer-claims.v1` document already built for this same run (see
    `clozn.runs.claims.build_answer_claims`) -- this function does not build one itself. Never reads or
    writes anything but its two arguments (no blob store, no model call).
    """
    if privacy not in {"full", "metadata_only"}:
        raise ValueError("privacy must be full or metadata_only")
    run_id = str(run.get("id") or "")
    if not run_id:
        raise ValueError("run.id must be a non-empty string")
    if not isinstance(claims_document, dict) or claims_document.get("schema_version") != CLAIMS_SCHEMA_VERSION:
        raise ValueError(f"claims_document must be a {CLAIMS_SCHEMA_VERSION} document")
    if claims_document.get("run_id") != run_id:
        raise ValueError("claims_document.run_id does not match run.id")

    response = run.get("response")
    response = response if isinstance(response, str) else None

    gate_method: str | None
    influence_map, gate_method = (None, "no_influence_map") if response is None else _influence_gate(run)

    geometry: _Geometry | None = None
    geometry_failure: str | None = None
    if gate_method is None:
        geometry, geometry_failure = _resolve_geometry(run_id, influence_map, response, privacy=privacy)

    results = []
    for claim in claims_document.get("claims") or []:
        if not isinstance(claim, dict):
            continue
        if claim.get("category") not in CLAIM_CATEGORIES:
            raise ValueError(f"claims_document contains an unknown category {claim.get('category')!r}")
        results.append(_support_for_claim(
            claim, gate_method=gate_method, geometry=geometry,
            geometry_failure=geometry_failure, response=response,
        ))

    influence_summary = {"gate": gate_method or "ok"}
    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "privacy": privacy,
        "offset_contract": dict(OFFSET_CONTRACT),
        "source": {
            "claims_schema_version": claims_document.get("schema_version"),
            "influence_map": influence_summary,
        },
        "results": results,
    }
    from clozn import schemas
    schemas.validate(document)
    return document


__all__ = [
    "SCHEMA_VERSION",
    "STATUSES",
    "build_claim_support",
]
