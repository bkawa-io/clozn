"""preferences.py -- the propose-and-review CONSUMER of the feedback capture layer (feedback.py).

The capture layer records directional signals ("you clicked Too verbose on run_D"). This module reads the
accumulated pattern and, when a (dial, direction) preference crosses a threshold, creates a PENDING PROPOSAL
-- "you've asked for concise 3x; make it your default?" -- for the user to APPROVE or DISMISS. It is
propose-AND-REVIEW by construction: this module decides WHAT to propose and tracks each proposal's status;
it NEVER changes a dial itself. Approving a proposal (persisting the dial) is done by the caller, which has
the live substrate's steer -- so this module stays model-free and unit-testable on plain dicts.

Not-nagging is a first-class concern: one active proposal per (dial, direction); a DISMISSED preference is
not re-proposed until fresh signals push its count a full threshold past where it was dismissed (so "no, I
don't want that" sticks, but a renewed, stronger pattern can still resurface). An APPROVED proposal is done
-- the dial now leans that way, so the pattern should quiet on its own.

Store: ~/.clozn/preferences.json, a list of proposals:
    {"id", "dial", "direction", "suggested_value", "count", "evidence"(run_ids), "status", "label",
     "created_ts", "resolved_ts", "dismissed_at_count"}
  status -- "pending" (awaiting review) / "approved" (applied) / "dismissed" (declined).
"""
from __future__ import annotations

import json
import os
import time

from clozn._io import atomic_write_json

_PATH = os.path.join(os.path.expanduser("~"), ".clozn", "preferences.json")

DEFAULT_THRESHOLD = 3        # signals for a (dial,direction) before it's worth proposing
DEFAULT_LEAN = 0.5           # the value a proposal suggests setting the dial to (capped per-axis on apply)
_EVIDENCE_CAP = 6            # run_ids kept as a proposal's receipts


def _path() -> str:
    return _PATH


def _load() -> list:
    p = _path()
    if not os.path.isfile(p):
        return []
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save(props: list) -> None:
    # Atomic: one bad write used to leave a truncated file, and _load answers any parse error
    # with [] -- erasing EVERY prior entry, not just the failed one. Same class, same remedy
    # as the settings/cards/dials writers.
    atomic_write_json(_path(), props, indent=2)


def _label(dial: str, direction, count: int) -> str:
    toward = "more " + dial if (direction or 0) >= 0 else "less " + dial
    return f"You've asked for {toward} {count}x -- make it your default?"


def _aggregate(signals: list) -> list:
    """Per (dial, direction): count + the recent run_ids that drove it (the proposal's receipts). Signals
    with no dial are ignored here (they aren't dial preferences). Newest run_ids first, capped."""
    agg: dict = {}
    for s in signals:
        dial = s.get("dial")
        if not dial:
            continue
        direction = s.get("direction")
        key = (dial, direction)
        ent = agg.setdefault(key, {"dial": dial, "direction": direction, "count": 0, "evidence": []})
        ent["count"] += 1
        rid = s.get("run_id")
        if rid and rid not in ent["evidence"]:
            ent["evidence"].append(rid)
    for ent in agg.values():
        ent["evidence"] = ent["evidence"][:_EVIDENCE_CAP]
    return list(agg.values())


def refresh(signals: list, threshold: int = DEFAULT_THRESHOLD, lean: float = DEFAULT_LEAN,
            _now=None) -> list:
    """Fold the current signal pattern into the proposal store and return the PENDING proposals. For each
    (dial, direction) whose count >= threshold: refresh an existing pending proposal; leave an approved one
    alone; re-open a dismissed one only if the count grew a full threshold past the dismissal; else create a
    new pending proposal. Deterministic given `signals` + `_now` (injectable clock for tests)."""
    now = float(_now if _now is not None else time.time())
    props = _load()
    by_key = {(p.get("dial"), p.get("direction")): p for p in props}
    for a in _aggregate(signals):
        if a["count"] < threshold:
            continue
        key = (a["dial"], a["direction"])
        ex = by_key.get(key)
        if ex is None:
            p = {"id": f"pref_{int(now * 1000):x}_{len(props):x}", "dial": a["dial"],
                 "direction": a["direction"], "suggested_value": round((a["direction"] or 1) * lean, 3),
                 "count": a["count"], "evidence": a["evidence"], "status": "pending",
                 "label": _label(a["dial"], a["direction"], a["count"]),
                 "created_ts": now, "resolved_ts": None, "dismissed_at_count": None}
            props.append(p)
            by_key[key] = p
        elif ex["status"] == "pending":
            ex["count"] = a["count"]; ex["evidence"] = a["evidence"]
            ex["label"] = _label(a["dial"], a["direction"], a["count"])
        elif ex["status"] == "dismissed":
            if a["count"] >= int(ex.get("dismissed_at_count") or 0) + threshold:
                ex.update(status="pending", count=a["count"], evidence=a["evidence"],
                          label=_label(a["dial"], a["direction"], a["count"]),
                          created_ts=now, resolved_ts=None)
        # approved -> leave it; the dial already leans this way.
    _save(props)
    return [p for p in props if p["status"] == "pending"]


def list_proposals(status=None) -> list:
    props = _load()
    return [p for p in props if status is None or p.get("status") == status]


def get(proposal_id: str):
    return next((p for p in _load() if p.get("id") == proposal_id), None)


def resolve(proposal_id: str, action: str, _now=None):
    """Mark a proposal approved or dismissed. Returns the updated proposal, or None if unknown. Dismissing
    records the count at dismissal, so `refresh` knows how much fresher evidence must accrue before it dares
    resurface. Applying an approved proposal (setting the dial) is the CALLER's job -- this only records
    intent, keeping the module model-free."""
    now = float(_now if _now is not None else time.time())
    if action not in ("approve", "dismiss"):
        return None
    props = _load()
    p = next((x for x in props if x.get("id") == proposal_id), None)
    if p is None:
        return None
    if action == "approve":
        p["status"] = "approved"
    else:
        p["status"] = "dismissed"
        p["dismissed_at_count"] = p.get("count")
    p["resolved_ts"] = now
    _save(props)
    return p
