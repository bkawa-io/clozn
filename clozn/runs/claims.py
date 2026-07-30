"""Deterministic answer segmentation and claim extraction (E1).

``clozn.answer-claims.v1`` is a DERIVED, read-only projection over one run's recorded answer text
(``run["response"]``) -- never a run migration.  The run document itself is never altered.  A claim's
identity and offsets are built with ``clozn.runs.text_span_addresses.make_text_span_address`` -- the
address kind ``"claim"`` and the ``native_ref.collection`` value ``"derived.claims"`` are already
reserved there for exactly this purpose (see that module's ``KINDS``/the schema's ``native_ref``
collection enum). This module does not invent a second offset/hash/identity scheme: every claim's
``text_span`` is that exact address shape, unmodified.

WHY STDLIB-ONLY, DETERMINISTIC, RULE-BASED
-------------------------------------------
``pyproject.toml`` declares ``dependencies = []`` (see ``tests/test_stdlib_only_boundary.py``) -- no
nltk, no spaCy, no regex packages beyond stdlib ``re``. Category assignment is a small, ORDERED table of
surface-pattern rules (``RULES`` below), not a model: same input always produces the same claims, in the
same order, with the same category and the same reason. ``tests/test_claims.py`` proves this by running
extraction twice and diffing the two documents byte-for-byte.

BE CONSERVATIVE -- THE LOAD-BEARING RULE
-----------------------------------------
A false ``factual_claim`` tag propagates into E2 (claim-to-source mapping) as a claim that needs
support, and from there into whatever E3 renders as "unsupported." Mislabeling ordinary prose as a
factual claim is a worse failure than under-labeling a real one, so ``factual_claim`` requires POSITIVE,
narrow evidence (a copula plus a digit/quote/internal capitalized word -- see ``RULES``), and every
other ambiguous case falls through to ``non_verifiable_prose``. Known, accepted, and documented
limitations of a surface-pattern approach: no negation scope (a "not likely" sentence still hits the
hedge pattern on "likely"), the month name "May" collides with the modal hedge word "may", and a
framing/introductory sentence that happens to contain a copula and a capitalized word (e.g. "Here is
how to use Python:") can register as ``factual_claim`` alongside genuine ones. All are deliberately left
as-is rather than adding narrower and narrower disambiguation heuristics, each with its own failure
surface to audit.

CATEGORIES
----------
    factual_claim          recommendation          uncertainty_statement
    instruction_procedure  non_verifiable_prose (the honest default)

STRUCTURE, NOT JUST SENTENCES
------------------------------
Three structural kinds feed the same category rules: a fenced ``` code block (always its own claim,
never split, never ``factual_claim``), a numbered/bulleted list item (its own claim, including any
wrapped continuation lines -- a list item is not further sentence-split), and an ordinary prose
sentence. List-item detection is a SEGMENTATION concern only; category assignment for a list item's
text uses the exact same rule table as any other sentence, evaluated on the text after its leading
marker (``"1. "``, ``"- "``, ...) is stripped for matching purposes -- the STORED span still covers the
marker, because the marker is part of the answer's real text.

LIMITED STATES, NEVER BROKEN SPANS
------------------------------------
Four typed ``segmentation.state`` values, and nothing in between:

    ok                    normal case; ``claims`` is the real (possibly empty) result
    empty                 ``response`` is the literal empty string
    unavailable           no response text exists to segment (redacted, or never recorded)
    segmentation_limited  the answer is dense in a script these ASCII-punctuation heuristics do not
                           reliably tokenize (CJK, Hangul, Thai) -- zero claims are produced rather
                           than guessing wrong sentence boundaries in a script this module cannot
                           reason about

A Latin-scripted non-English answer (French, Spanish, ...) segments normally under ``ok`` -- the
category rules are English-phrase lists, so most of its sentences will honestly land on
``non_verifiable_prose`` via the conservative default, which is the correct, non-fabricated answer,
not a bug.
"""
from __future__ import annotations

import hashlib
import re
from typing import NamedTuple

from clozn.runs.text_span_addresses import OFFSET_CONTRACT, make_text_span_address

SCHEMA_VERSION = "clozn.answer-claims.v1"

CATEGORIES = frozenset({
    "factual_claim",
    "recommendation",
    "uncertainty_statement",
    "instruction_procedure",
    "non_verifiable_prose",
})

STRUCTURAL_KINDS = frozenset({"fence", "list_item", "sentence"})

# ======================================================================================================
# The rules table -- ORDER IS THE CONTRACT. First match wins. Keep this table and `categorize_claim`'s
# body in lockstep; `tests/test_claims.py` asserts every reason `categorize_claim` can return appears
# here exactly once, and the schema's `category_reason` enum is this table's `rule_id` column.
# ======================================================================================================
RULES = (
    # (rule_id, category, description)
    ("code_fence_block", "non_verifiable_prose",
     "Text inside a fenced ``` code block. Code is never itself a checkable natural-language claim."),
    ("interrogative_sentence", "non_verifiable_prose",
     "The segment ends in '?'. A question asserts nothing to verify."),
    ("hedge_marker", "uncertainty_statement",
     "Contains an uncertainty/hedge phrase (might, may, could, I think, unclear, seems, probably, "
     "...). Checked before recommendation/instruction so a hedged suggestion reads as uncertain, not "
     "confident."),
    ("recommendation_marker", "recommendation",
     "Contains a recommendation phrase (should, recommend, suggest, consider, best practice, ...)."),
    ("list_item_imperative", "instruction_procedure",
     "A numbered/bulleted list item whose first word, after its marker, is a closed-list imperative "
     "verb (Install, Run, Open, Configure, ...)."),
    ("imperative_lead", "instruction_procedure",
     "A non-list segment whose very first word is that same closed-list imperative verb."),
    ("factual_declarative", "factual_claim",
     "A declarative segment with a copula/definitional verb (is/are/was/were/equals/means/contains/"
     "consists) AND concrete evidence: a digit, a quoted span, or a capitalized word other than the "
     "segment's own first word. Narrow on purpose -- a bare 'X is a Y' with none of those stays "
     "non_verifiable_prose."),
    ("no_deterministic_category_match", "non_verifiable_prose",
     "Nothing above matched. The conservative default: an uncategorizable segment is prose, not a "
     "claim."),
)
_RULE_IDS = tuple(rule_id for rule_id, _category, _description in RULES)


class ClaimUnit(NamedTuple):
    """One claim-worthy span of the answer text, before categorization."""
    start: int
    end: int
    structural_kind: str  # "fence" | "list_item" | "sentence"


# ======================================================================================================
# Segmentation: text -> ClaimUnit list. Pure, deterministic, stdlib `re` only.
# ======================================================================================================

_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_LIST_MARKER_RE = re.compile(r"^[ \t]*(?:[-*•]|\d{1,3}[.)])[ \t]+")
_SENTENCE_PUNCT_RE = re.compile(r"[.!?]+")
_ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "e.g", "i.e", "fig", "no", "st",
    "vol", "approx", "inc", "ltd", "co",
})

_DENSE_SCRIPT_RANGES = (
    (0x4E00, 0x9FFF),   # CJK Unified Ideographs
    (0x3400, 0x4DBF),   # CJK Extension A
    (0x3040, 0x309F),   # Hiragana
    (0x30A0, 0x30FF),   # Katakana
    (0xAC00, 0xD7A3),   # Hangul syllables
    (0x0E00, 0x0E7F),   # Thai
)
_DENSE_SCRIPT_THRESHOLD = 0.3  # fraction of non-whitespace characters


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _blocks(text: str) -> list[tuple[int, int, bool]]:
    """(start, end, is_fence) covering the whole string, in order, no gaps."""
    blocks: list[tuple[int, int, bool]] = []
    cursor = 0
    for match in _FENCE_RE.finditer(text):
        if match.start() > cursor:
            blocks.append((cursor, match.start(), False))
        blocks.append((match.start(), match.end(), True))
        cursor = match.end()
    if cursor < len(text):
        blocks.append((cursor, len(text), False))
    return blocks


def _line_spans(text: str, base: int, limit: int) -> list[tuple[int, int]]:
    spans = []
    pos = base
    for line in text[base:limit].splitlines(keepends=True):
        spans.append((pos, pos + len(line)))
        pos += len(line)
    return spans


def _prose_blocks(text: str, base: int, limit: int) -> list[tuple[int, int, bool]]:
    """(start, end, is_list_item) blocks within one non-fence region, blank lines dropped.

    Consecutive non-blank lines merge into one block (an ordinary wrapped paragraph), UNLESS a line
    itself starts a new list marker, which always begins a fresh block even with no blank line before
    it -- "1. Do X\\n2. Do Y\\n" is two blocks, not one.
    """
    blocks: list[tuple[int, int, bool]] = []
    current_start: int | None = None
    current_end = 0
    current_is_list = False
    for line_start, line_end in _line_spans(text, base, limit):
        raw = text[line_start:line_end]
        if not raw.strip():
            if current_start is not None:
                blocks.append((current_start, current_end, current_is_list))
                current_start = None
            continue
        is_list_start = bool(_LIST_MARKER_RE.match(raw))
        if current_start is None:
            current_start, current_end, current_is_list = line_start, line_end, is_list_start
        elif is_list_start:
            blocks.append((current_start, current_end, current_is_list))
            current_start, current_end, current_is_list = line_start, line_end, True
        else:
            current_end = line_end
    if current_start is not None:
        blocks.append((current_start, current_end, current_is_list))
    return blocks


def _preceding_word(text: str, pos: int) -> str:
    """The lowercase token immediately before `pos` (the start of a punctuation run).

    Includes single internal periods immediately between two letters, so "e.g." and "i.e." resolve to
    one token ("e.g", "i.e") rather than stopping at the first internal period and seeing only "g"/"e".
    """
    start = pos
    while start > 0 and (
        text[start - 1].isalpha()
        or (text[start - 1] == "." and start > 1 and text[start - 2].isalpha())
    ):
        start -= 1
    return text[start:pos].lower()


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """One non-blank prose block -> its (start, end) sentence sub-spans, offsets relative to `text`.

    A period/question mark/exclamation run is a boundary only when it is followed by whitespace (a bare
    "3.14" or "e.g.com" has none, so it never qualifies) AND the next non-space character looks like a
    new sentence start (uppercase, digit, quote, or opening bracket) AND the word immediately before it
    is not one of a small closed abbreviation list. No terminator at all reduces to the single
    (0, len(text)) span -- the pathological no-punctuation case, an honest answer, not a bug.
    """
    boundaries = {0, len(text)}
    for match in _SENTENCE_PUNCT_RE.finditer(text):
        end = match.end()
        if end >= len(text) or not text[end].isspace():
            continue
        if _preceding_word(text, match.start()) in _ABBREVIATIONS:
            continue
        rest = text[end:].lstrip()
        if not rest or not (rest[0].isupper() or rest[0].isdigit() or rest[0] in "\"'([‘“"):
            continue
        boundaries.add(end)
    points = sorted(boundaries)
    spans = []
    for a, b in zip(points, points[1:]):
        ts, te = _trim(text, a, b)
        if te > ts:
            spans.append((ts, te))
    return spans


def iter_claim_units(text: str) -> list[ClaimUnit]:
    """The full ordered list of claim-worthy spans in `text`. Pure; no categorization, no I/O."""
    units: list[ClaimUnit] = []
    for block_start, block_end, is_fence in _blocks(text):
        if is_fence:
            ts, te = _trim(text, block_start, block_end)
            if te > ts:
                units.append(ClaimUnit(ts, te, "fence"))
            continue
        for sub_start, sub_end, is_list in _prose_blocks(text, block_start, block_end):
            if is_list:
                ts, te = _trim(text, sub_start, sub_end)
                if te > ts:
                    units.append(ClaimUnit(ts, te, "list_item"))
            else:
                block_text = text[sub_start:sub_end]
                for rel_start, rel_end in _sentence_spans(block_text):
                    units.append(ClaimUnit(sub_start + rel_start, sub_start + rel_end, "sentence"))
    return units


def _dense_script_fraction(text: str) -> float:
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return 0.0
    dense = sum(
        1 for ch in non_space
        if any(low <= ord(ch) <= high for low, high in _DENSE_SCRIPT_RANGES)
    )
    return dense / len(non_space)


# ======================================================================================================
# Categorization: (text, structural_kind) -> (category, rule_id). Pure, deterministic.
# ======================================================================================================

_APOS = "['’]"
_HEDGE_PATTERNS = (
    rf"\bi{_APOS}m not (?:entirely |completely |100% )?sure\b",
    r"\bnot (?:entirely |completely )?certain\b",
    rf"\bcan{_APOS}t be (?:entirely |completely )?certain\b",
    r"\bcannot be (?:entirely |completely )?certain\b",
    r"\bmight\b", r"\bmay\b", r"\bcould\b",
    r"\bpossibly\b", r"\bperhaps\b", r"\bpresumably\b",
    r"\bi think\b", r"\bi believe\b",
    rf"\bit{_APOS}s unclear\b", r"\bit is unclear\b",
    r"\bseems?\b", r"\bappears?\b",
    r"\blikely\b", r"\bprobably\b", r"\buncertain\b",
    r"\bhard to say\b", r"\bdifficult to say\b",
    r"\bno guarantee\b",
    rf"\bit{_APOS}s possible\b", r"\bit is possible\b",
)
_RECOMMENDATION_PATTERNS = (
    r"\byou should\b", r"\bshould\b",
    r"\bwe recommend\b", r"\bi recommend\b", r"\brecommends?\b",
    r"\bsuggests?\b", rf"\bi{_APOS}d suggest\b", r"\bwe suggest\b",
    r"\bconsider\b", r"\btry\b", r"\bmake sure to\b",
    rf"\bit{_APOS}s best to\b", r"\bit is best to\b", r"\bbest practice\b",
    rf"\bi{_APOS}d advise\b", r"\bi would advise\b", r"\badvisable\b",
    r"\bought to\b",
)
_IMPERATIVE_VERBS = frozenset({
    "run", "install", "open", "click", "add", "remove", "delete", "create", "set", "configure",
    "check", "ensure", "use", "navigate", "type", "enter", "select", "download", "restart",
    "reboot", "enable", "disable", "verify", "replace", "copy", "save", "execute", "start",
    "stop", "go", "visit", "press", "choose", "update", "upgrade", "launch", "close", "edit",
    "modify", "review", "confirm", "follow", "clone", "build", "compile", "test", "deploy",
    "initialize", "activate", "deactivate", "uninstall", "reinstall", "clear", "reset",
})
_COPULA_RE = re.compile(r"\b(?:is|are|was|were|equals|means|contains|consists)\b", re.IGNORECASE)
_DIGIT_RE = re.compile(r"\d")
_QUOTE_RE = re.compile(r"[\"‘’“”][^\"‘’“”]+[\"‘’“”]")
_PROPER_NOUN_RE = re.compile(r"[A-Z][a-zA-Z]+")
_LEADING_ALPHA_RE = re.compile(r"[A-Za-z]+")

_HEDGE_RE = re.compile("|".join(f"(?:{pattern})" for pattern in _HEDGE_PATTERNS), re.IGNORECASE)
_RECOMMENDATION_RE = re.compile(
    "|".join(f"(?:{pattern})" for pattern in _RECOMMENDATION_PATTERNS), re.IGNORECASE,
)


def _leading_word(text: str, structural_kind: str) -> str:
    body = text
    if structural_kind == "list_item":
        marker = _LIST_MARKER_RE.match(text)
        if marker:
            body = text[marker.end():]
    match = _LEADING_ALPHA_RE.match(body.strip())
    return match.group(0).lower() if match else ""


def _has_concrete_evidence(text: str) -> bool:
    """`text` is already stripped (the caller passes `stripped`), so `first_word.end()` is directly a
    valid offset into it -- excluding only the sentence's own leading word from the proper-noun check."""
    if _DIGIT_RE.search(text):
        return True
    if _QUOTE_RE.search(text):
        return True
    first_word = _LEADING_ALPHA_RE.match(text)
    tail_start = first_word.end() if first_word else 0
    return bool(_PROPER_NOUN_RE.search(text[tail_start:]))


def categorize_claim(text: str, structural_kind: str) -> tuple[str, str]:
    """(category, rule_id) for one claim unit's text, per `RULES` above, in that exact order."""
    if structural_kind not in STRUCTURAL_KINDS:
        raise ValueError(f"unknown structural_kind {structural_kind!r}")
    if structural_kind == "fence":
        return "non_verifiable_prose", "code_fence_block"
    stripped = text.strip()
    if stripped.endswith("?"):
        return "non_verifiable_prose", "interrogative_sentence"
    if _HEDGE_RE.search(stripped):
        return "uncertainty_statement", "hedge_marker"
    if _RECOMMENDATION_RE.search(stripped):
        return "recommendation", "recommendation_marker"
    if _leading_word(stripped, structural_kind) in _IMPERATIVE_VERBS:
        reason = "list_item_imperative" if structural_kind == "list_item" else "imperative_lead"
        return "instruction_procedure", reason
    if _COPULA_RE.search(stripped) and _has_concrete_evidence(stripped):
        return "factual_claim", "factual_declarative"
    return "non_verifiable_prose", "no_deterministic_category_match"


# ======================================================================================================
# The derived artifact: run -> clozn.answer-claims.v1
# ======================================================================================================

def _text_details(text: str) -> dict:
    encoded = text.encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "code_points": len(text),
            "utf8_bytes": len(encoded)}


def _run_redaction_status(run: dict) -> str | None:
    redaction = run.get("redaction")
    status = redaction.get("status") if isinstance(redaction, dict) else None
    if status == "redacted" or "redacted" in (run.get("flags") or []):
        return "redacted"
    return status if isinstance(status, str) else None


def build_answer_claims(run: dict, *, privacy: str = "metadata_only") -> dict:
    """Build and validate one derived `clozn.answer-claims.v1` document from a run's recorded answer.

    Never reads or writes anything but `run` itself (no I/O, no blob store, no model call). `privacy`
    matches `text_span_addresses`' own contract exactly: "full" embeds each claim's exact substring
    text in its `text_span`; "metadata_only" (the default) keeps offsets and hashes only.
    """
    if privacy not in {"full", "metadata_only"}:
        raise ValueError("privacy must be full or metadata_only")
    run_id = str(run.get("id") or "")
    if not run_id:
        raise ValueError("run.id must be a non-empty string")

    response = run.get("response")
    response = response if isinstance(response, str) else None
    if _run_redaction_status(run) == "redacted":
        response = None
        unavailable_reason = "answer_text_redacted"
    else:
        unavailable_reason = "no_answer_text"

    claims: list[dict] = []
    if response is None:
        segmentation = {"state": "unavailable", "reason": unavailable_reason}
        answer_source = {"basis": "recorded_answer"}
    elif response == "":
        details = _text_details("")
        segmentation = {"state": "empty", "reason": "answer_text_empty"}
        answer_source = {
            "basis": "recorded_answer", "basis_sha256": details["sha256"],
            "basis_code_points": details["code_points"], "basis_utf8_bytes": details["utf8_bytes"],
        }
    elif _dense_script_fraction(response) > _DENSE_SCRIPT_THRESHOLD:
        details = _text_details(response)
        segmentation = {"state": "segmentation_limited", "reason": "unsupported_script_density"}
        answer_source = {
            "basis": "recorded_answer", "basis_sha256": details["sha256"],
            "basis_code_points": details["code_points"], "basis_utf8_bytes": details["utf8_bytes"],
        }
    else:
        details = _text_details(response)
        answer_source = {
            "basis": "recorded_answer", "basis_sha256": details["sha256"],
            "basis_code_points": details["code_points"], "basis_utf8_bytes": details["utf8_bytes"],
        }
        for index, unit in enumerate(iter_claim_units(response)):
            category, reason = categorize_claim(response[unit.start:unit.end], unit.structural_kind)
            text_span = make_text_span_address(
                run_id=run_id,
                kind="claim",
                native_ref={
                    "artifact_schema": SCHEMA_VERSION,
                    "collection": "derived.claims",
                    "id": f"claim-{index}",
                },
                relation_anchor={"claim_index": index},
                basis="recorded_answer",
                start=unit.start,
                end=unit.end,
                privacy=privacy,
                basis_text=response,
                redacted=False,
            )
            claims.append({
                "index": index,
                "category": category,
                "category_reason": reason,
                "text_span": text_span,
            })
        segmentation = {"state": "ok", "claim_count": len(claims)}

    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "privacy": privacy,
        "offset_contract": dict(OFFSET_CONTRACT),
        "segmentation": segmentation,
        "answer_source": answer_source,
        "claims": claims,
    }
    from clozn import schemas
    schemas.validate(document)
    return document


__all__ = [
    "CATEGORIES",
    "ClaimUnit",
    "RULES",
    "SCHEMA_VERSION",
    "STRUCTURAL_KINDS",
    "build_answer_claims",
    "categorize_claim",
    "iter_claim_units",
]
