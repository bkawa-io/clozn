"""coherence.py -- the mandatory degeneration/repetition proxy shared by corrective retries
(clozn/replay/corrective.py) and the concept-guard counter-steer loop (clozn/server/generation_guard.py).

Originally lived in counterfactual.py (EXPLAIN_THIS_ANSWER_SPEC.md law #6: a slider that pushes a raw
steering vector past where a model can absorb it must not report "huge delta = huge effect" -- a huge
delta can just as easily be the model degenerating into repetition or a script switch). That module was
the tone-dial what-if slider and was retired with the rest of named-dial personalization; this check is
generic text-only degeneracy detection with no dial dependency of its own, so it moved here rather than
being deleted with its old home.

A crude, eyeball-informed, NOT-learned proxy -- mirrors memory_disorders.py's is_degenerate() checks
exactly (empty output / immediate 3-gram word repetition / character runaway / script switch). Duplicated
rather than imported: callers of this module are stdlib-only / no-model, and memory_disorders.py is out
of scope to touch or depend on here.
"""
from __future__ import annotations

import re
import unicodedata

_FOREIGN_MIN = 3


def _foreign_letters(t: str) -> int:
    """Count non-ASCII LETTER characters. A real language/script switch (the failure this flags -- steering
    derailing into Cyrillic/CJK word-salad) is many non-ASCII *letters*; emoji, curly quotes, and em-dashes
    are non-ASCII SYMBOLS/PUNCTUATION and must NOT count -- Gemma-2 is emoji-heavy and perfectly coherent, so
    the old `[^\\x00-\\x7F]` catch-all false-flagged it 100% degenerate. A stray accent ('cafe') is one
    letter, below the threshold, so ordinary loanwords pass."""
    return sum(1 for ch in t if ord(ch) > 127 and unicodedata.category(ch).startswith("L"))


def _coherence(text: str) -> dict:
    """{"degenerate": bool, "reason": str} for `text` -- flags an over-dosed steering vector or corrective
    rewrite that derailed into gibberish, never silently read as "big delta = big effect". Pure text
    counting, no model call -- exactly as honest, and exactly as crude, as the one other place this
    codebase already does this (memory_disorders.is_degenerate)."""
    t = (text or "").strip()
    if not t:
        return {"degenerate": True, "reason": "empty"}
    words = t.split()
    for i in range(len(words) - 2):
        if words[i] == words[i + 1] == words[i + 2]:
            return {"degenerate": True, "reason": "repeat-3gram"}
    if re.search(r"(.)\1{4,}", t):
        return {"degenerate": True, "reason": "char-runaway"}
    if _foreign_letters(t) >= _FOREIGN_MIN:
        return {"degenerate": True, "reason": "script-switch"}
    return {"degenerate": False, "reason": ""}
