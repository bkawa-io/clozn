"""feedback.py -- the preference-signal store (the CAPTURE layer of preference plumbing).

The Run Inspector already lets a user say a reply was off ("Too verbose" -> nudge `concise` + replay,
heavn Replay's quick repairs) and persist that fix (the F2 save-fix -> /steer/set). That is the MANUAL loop:
the user drives it, one reply at a time. What's missing is LEARNING -- noticing "you keep asking for
concise" and proactively offering to make it a default. This module is the foundation for that: it
records each directional feedback signal, tied to the run that prompted it, so a later accumulate-and-
propose step can mine the pattern.

Deliberately the CAPTURE layer ONLY -- it records signals and rolls them up; it never CHANGES a dial or
proposes anything (that's the agency-carrying consumer, built once the propose-vs-auto call is made). No
model, no torch, no gradients -- a stdlib JSON append store at ~/.clozn/feedback.json, so it's cheap,
inspectable, and unit-testable on plain dicts. Every signal carries provenance (the run_id) so a proposal
it later feeds can show its receipts, exactly like a memory card cites the turn it came from.

A signal:
    {"id", "ts", "run_id", "kind", "dial", "direction", "meta"}
  kind      -- what the user did: "quick_repair" (clicked a "Too X" button), "save_fix" (persisted a
               replay's dial change), "thumb" (a future 👍/👎), ... an open string, not an enum, so new
               signal sources don't need a schema change.
  dial      -- the tone dial the signal is about (e.g. "concise"); None for a signal that isn't dial-shaped.
  direction -- +1 toward the dial's + pole / -1 away; None when not applicable.
"""
from __future__ import annotations

import json
import os
import time

from clozn._io import atomic_write_json

_PATH = os.path.join(os.path.expanduser("~"), ".clozn", "feedback.json")


def _path() -> str:
    return _PATH


def _load() -> list:
    """The whole signal list, or [] on a missing/corrupt file (never raises -- a feedback read must never
    break a chat turn)."""
    p = _path()
    if not os.path.isfile(p):
        return []
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(signals: list) -> None:
    # Atomic: one bad write used to leave a truncated file, and _load answers any parse error
    # with [] -- erasing EVERY prior entry, not just the failed one. Same class, same remedy
    # as the settings/cards/dials writers.
    atomic_write_json(_path(), signals, indent=2)


def record(run_id, kind: str, dial=None, direction=None, meta=None, _now=None) -> dict:
    """Append one feedback signal and return it. `_now` is injectable so tests are deterministic (the store
    is otherwise wall-clock). A blank kind is coerced to 'unknown' rather than rejected -- losing a signal
    is worse than a vague one. Best-effort persistence: a write failure returns the (unsaved) signal rather
    than raising, so the caller (a request handler) never fails a user action over a feedback write."""
    now = float(_now if _now is not None else time.time())
    signals = _load()
    sig = {
        "id": f"fb_{int(now * 1000):x}_{len(signals):x}",
        "ts": now,
        "run_id": str(run_id) if run_id is not None else None,
        "kind": (str(kind).strip() or "unknown"),
        "dial": (str(dial) if dial else None),
        "direction": (int(direction) if direction in (1, -1, "1", "-1") else None),
        "meta": meta if isinstance(meta, dict) else None,
    }
    signals.append(sig)
    try:
        _save(signals)
    except Exception:
        pass
    return sig


def list_signals(limit=None, run_id=None) -> list:
    """All signals (newest first), optionally filtered to one run or capped at `limit`."""
    out = list(reversed(_load()))
    if run_id is not None:
        out = [s for s in out if s.get("run_id") == str(run_id)]
    if isinstance(limit, int) and limit >= 0:
        out = out[:limit]
    return out


def summary(window_seconds=None, _now=None) -> dict:
    """The rollup a learning step reads: per-(dial, direction) counts + the last run that drove each, over
    an optional recent window. This is the 'what you keep asking for' aggregate -- NOT a proposal (no dial
    is changed here); a consumer decides whether a count crosses a threshold worth suggesting. Directionless
    signals (no dial) are tallied under `other` so nothing is silently dropped."""
    now = float(_now if _now is not None else time.time())
    signals = _load()
    if isinstance(window_seconds, (int, float)) and window_seconds > 0:
        signals = [s for s in signals if now - float(s.get("ts", 0)) <= window_seconds]
    by_dial: dict = {}
    other = 0
    for s in signals:
        dial = s.get("dial")
        if not dial:
            other += 1
            continue
        d = str(s.get("direction")) if s.get("direction") is not None else "0"
        key = f"{dial}:{d}"
        ent = by_dial.setdefault(key, {"dial": dial, "direction": s.get("direction"),
                                       "count": 0, "last_run_id": None, "last_ts": 0.0})
        ent["count"] += 1
        if float(s.get("ts", 0)) >= ent["last_ts"]:
            ent["last_ts"] = float(s.get("ts", 0))
            ent["last_run_id"] = s.get("run_id")
    ranked = sorted(by_dial.values(), key=lambda e: e["count"], reverse=True)
    return {"total": len(signals), "by_dial": ranked, "other": other}
