"""Repeated-cycle detection on a generated token stream.

Relocated from clozn/memory/anchored.py when the anchored-memory program was removed. It never had
anything to do with memory -- it is a generic "is this reply eating itself" check, and
clozn/runs/signals.py has always used it that way. Kept verbatim so the behaviour it guards is
unchanged; only its address moved.
"""
from __future__ import annotations


def detect_loop(pieces, window: int = 8) -> bool:
    """True when the last `window` generated pieces are a VERBATIM repeated cycle -- i.e. periodic
    with some period p <= window//2, so at least two full cycles are present. Fires on
    'the cake the cake the cake the cake' and on a single stuttered token; never on ordinary prose,
    which has no exact short period. Fewer than `window` pieces, or window < 2, is always False
    (not enough evidence to call a loop)."""
    toks = [p for p in (str(x) for x in (pieces or [])) if p.strip()]
    try:
        w = int(window)
    except (TypeError, ValueError):
        return False
    if w < 2 or len(toks) < w:
        return False
    tail = toks[-w:]
    for period in range(1, w // 2 + 1):
        if all(tail[i] == tail[i - period] for i in range(period, w)):
            return True
    return False
