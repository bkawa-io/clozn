"""clozn/behavior/compare.py -- deterministic before/after compare-view metrics for corrective
retries (roadmap feature 08's "Compare view").

Deliberately separate from clozn.receipts.metrics.receipt_metrics(): that function's return shape
(``{"words", "wps", "changed"}``) is a widely shared contract -- every receipt type (generic,
forced, corrective) asserts it verbatim, including exact-equality tests in tests/test_receipts.py.
Adding fields there would be a shared-surface change outside this feature's approved scope.
compare_metrics() below is a SEPARATE, additive dict that clozn.replay.corrective.retry_compare()
merges alongside receipt_metrics()'s own output; it never replaces or reshapes it.

Both metrics here are pure text counting, no model call, no dependency beyond the stdlib --
exactly as honest, and exactly as crude, as clozn.replay.counterfactual._coherence's degeneracy
check (which this module deliberately does not duplicate: that one only flags "degenerate: bool"
for a single text; this one COUNTS repeats across a before/after pair).

"Source coverage" -- the spec's third compare-view row -- is deliberately NOT computed here: it
depends on the Sources lens (a different, not-yet-shipped roadmap feature). Inventing a number for
it would be exactly the "claiming improved factuality merely because measured dependence increased"
the spec explicitly warns against. Its absence from compare_metrics()'s output is intentional, not
an oversight (SEAMS.md rule 2: omit, never null-pad) -- a caller that wants it must ask the Sources
lens directly, once one exists, and merge it in separately.

Computation only, no rendering: the coordinator deferred all frontend work this cycle (the Studio
rebuild is unverified), so this module produces the data a compare view would show without
attempting to draw it.
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[\w']+")
_CODE_FENCE_RE = re.compile(r"^```", re.MULTILINE)
_BULLET_RE = re.compile(r"^\s*[-*+]\s+\S", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^\s*\d+[.)]\s+\S", re.MULTILINE)
_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)


def _repeated_phrase_count(text: str, n: int = 3) -> int:
    """Count of DISTINCT n-word phrases that appear more than once in `text` (default trigrams --
    the same window clozn.replay.counterfactual._coherence's repeat-3gram check uses, just counted
    instead of only flagged). Case-insensitive, word-boundary tokenized; punctuation is not part of
    a word. A text shorter than `n` words has no phrases to repeat -- 0, not an error."""
    words = _WORD_RE.findall((text or "").lower())
    if len(words) < n:
        return 0
    counts: dict[tuple, int] = {}
    for i in range(len(words) - n + 1):
        gram = tuple(words[i:i + n])
        counts[gram] = counts.get(gram, 0) + 1
    return sum(1 for count in counts.values() if count > 1)


def _format_fingerprint(text: str) -> tuple:
    """(code_fences, bullet_lines, numbered_lines, heading_lines) -- a cheap structural signature.
    Two texts with the same fingerprint are not proven identically formatted (this is deliberately
    crude, matching the rest of this module), but a DIFFERENT fingerprint is a real, checkable
    structural change: a fence count that dropped, a heading that vanished, bullets that became a
    numbered list."""
    t = text or ""
    return (
        len(_CODE_FENCE_RE.findall(t)),
        len(_BULLET_RE.findall(t)),
        len(_NUMBERED_RE.findall(t)),
        len(_HEADING_RE.findall(t)),
    )


def compare_metrics(orig: str, repl: str) -> dict:
    """Additive compare-view metrics for a corrective-retry before/after pair:

        {"repeated_phrases": [orig_count, repl_count], "format_changed": bool}

    Merge into a caller's existing delta dict (e.g. receipts.receipt_metrics()'s output) rather than
    using this as a caller's only comparison -- it does not replicate word counts or the changed%
    Jaccard delta that function already provides."""
    orig_text, repl_text = orig or "", repl or ""
    return {
        "repeated_phrases": [_repeated_phrase_count(orig_text), _repeated_phrase_count(repl_text)],
        "format_changed": _format_fingerprint(orig_text) != _format_fingerprint(repl_text),
    }
