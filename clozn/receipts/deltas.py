"""Influence ablation changes and receipt object assembly."""
from __future__ import annotations

from .metrics import receipt_metrics


_NOTE_BASELINE = (
    "the run's stored sampled reply is NOT the baseline for this receipt -- greedy-with-the-influence is. "
    "The sampled reply is context only; it is never a term in this subtraction "
    "(EXPLAIN_THIS_ANSWER_SPEC.md M2: diffing sampled-vs-greedy would mix two changes at once)."
)


def _key(influence: dict) -> str:
    influence = influence or {}
    if influence.get("card_id"):
        return f"card:{influence['card_id']}"
    if influence.get("dial"):
        return f"dial:{influence['dial']}"
    if influence.get("section"):
        return f"section:{influence['section']}"
    if influence.get("memory_off"):
        return "memory_off"
    if influence.get("behavior_off"):
        return "behavior_off"
    return "unknown"


def _ablation_changes(influence: dict) -> dict | None:
    """One influence spec -> replay changes for its ablated arm."""
    if not isinstance(influence, dict):
        return None
    cid = influence.get("card_id")
    if cid:
        return {"disabled_memory_ids": [str(cid)]}
    dial = influence.get("dial")
    if dial:
        return {"behavior_overrides": {str(dial): 0.0}}
    section = influence.get("section")
    if section:
        return {"exclude_sections": [str(section)]}
    if influence.get("memory_off"):
        return {"memory_off": True}
    if influence.get("behavior_off"):
        return {"behavior_off": True}
    return None


def _section_influences(manifest: dict) -> tuple[list, list]:
    """Sections from the M1 manifest (`influences_active.sections`, produced by explain.py from a run's
    `sections` field) as receipt influence specs -- (influences, skipped), the same two-list shape
    `_fired_influences` already returns for cards/dials, so core.py's caller can just `.extend()` both.

    DEDUP DECISION (why a "memory_card"-sourced section is never turned into its own ablation arm here):
    a section's `source` field can be "memory_card" -- the manifest's per-turn view of a memory card that
    ALSO already appears, resolved by id with its own provenance, in `influences_active.cards`. That is
    the SAME fired influence described twice by two different producers (the card path and the section
    path around it). Ablating both would double-count one real cause: two receipts for the same change,
    and a phantom "these two are jointly redundant" pair from prove_all's guard (ablating the card AND its
    own section leaves nothing standing, which looks like redundancy between two things that are really
    one thing). Cards keep their existing, richer path (a resolved card_id, real per-card ablation in
    prompt mode via `disabled_memory_ids`); a section only becomes its own arm when its source is "api" or
    "auto" -- prompt content that has no other representation in the manifest at all, and is therefore the
    ONLY way to ablate it."""
    sections = ((manifest or {}).get("influences_active") or {}).get("sections") or {}
    if isinstance(sections, dict):                     # explain.py's {"available": .., "sections": [...]}
        sections = sections.get("sections") or []
    influences: list = []
    skipped: list = []
    for sec in sections if isinstance(sections, list) else []:
        if not isinstance(sec, dict):
            continue
        if sec.get("source") not in ("api", "auto"):
            continue                                    # "memory_card" (or anything else): dedup, see above
        name = sec.get("name")
        if not name:
            skipped.append({"influence": {"section_source": sec.get("source")},
                            "reason": "section has no name recorded; per-section ablation needs a name"})
            continue
        influences.append({"section": name, "source": sec.get("source")})
    return influences, skipped


def _merge_ablation_changes(influences: list) -> dict:
    """Joint replay changes that ablate every influence at once."""
    ids: list = []
    overrides: dict = {}
    sections: list = []
    memory_off = behavior_off = False
    for inf in influences:
        c = _ablation_changes(inf) or {}
        ids.extend(c.get("disabled_memory_ids") or [])
        overrides.update(c.get("behavior_overrides") or {})
        sections.extend(c.get("exclude_sections") or [])
        memory_off = memory_off or bool(c.get("memory_off"))
        behavior_off = behavior_off or bool(c.get("behavior_off"))
    merged: dict = {}
    if ids:
        merged["disabled_memory_ids"] = ids
    if overrides:
        merged["behavior_overrides"] = overrides
    if sections:
        merged["exclude_sections"] = sections
    if memory_off:
        merged["memory_off"] = True
    if behavior_off:
        merged["behavior_off"] = True
    return merged


def _cost_note(influence: dict) -> str:
    influence = influence or {}
    if influence.get("card_id") or influence.get("memory_off"):
        return ("cost: a front-of-context memory ablation changes the shared prefix, so the ablated arm "
                "re-prefills the whole context (no KV reuse) -- the expensive case.")
    if influence.get("section"):
        return ("cost: a section ablation edits the prompt content itself, so the ablated arm re-prefills "
                "the whole context (no KV reuse) -- the same expensive case as a memory ablation.")
    return ("cost: a dial ablation acts at decode time, so the prompt KV stays reusable -- cheap relative "
            "to a memory ablation.")


def _unapplied_note(ablated_child: dict, changes: dict) -> str | None:
    notes = ((ablated_child or {}).get("memory") or {}).get("notes") or {}
    if changes.get("disabled_memory_ids") and "disabled_memory_ids" in notes:
        return notes["disabled_memory_ids"]
    if changes.get("edited_memory") and "edited_memory" in notes:
        return notes["edited_memory"]
    if changes.get("exclude_sections") and "exclude_sections" in notes:
        sec_notes = notes["exclude_sections"]
        if isinstance(sec_notes, dict):
            return "; ".join(f"{k}: {v}" for k, v in sec_notes.items())
        return str(sec_notes)
    return None


def _build_receipt(influence: dict, baseline_child: dict, ablated_child: dict, changes: dict) -> dict:
    baseline_reply = baseline_child.get("response") or ""
    ablated_reply = ablated_child.get("response") or ""
    unapplied = _unapplied_note(ablated_child, changes)
    out = {
        "influence": influence,
        "changes_applied": changes,
        "baseline_reply": baseline_reply,
        "ablated_reply": ablated_reply,
        "delta": receipt_metrics(baseline_reply, ablated_reply),
        "has_effect": baseline_reply != ablated_reply,
        "causal_verified": unapplied is None,
        "note": _NOTE_BASELINE,
        "cost_note": _cost_note(influence),
    }
    if isinstance(influence, dict) and influence.get("section"):
        # section receipts carry the same honesty fields every other receipt does (above); this just
        # tags WHICH kind of influence it is and its manifest identity, for a consumer rendering cards/
        # dials/sections side by side without having to sniff `influence`'s shape.
        out["kind"] = "section"
        out["section_name"] = influence.get("section")
        out["section_source"] = influence.get("source")
    # Early-stop (prove-all perf): the ablated arm halted at the first token that left the baseline, so the
    # ablated reply above is a bit-exact PREFIX up to that divergence -- NOT the full ablated reply (the rest
    # was never generated; that's the saved decode). has_effect is still exact (a prefix that already differs
    # proves the answer changed), but the `delta` word/Jaccard numbers describe the divergence POINT, not a
    # full-reply diff -- so flag it, honestly, rather than let a consumer read the truncation as a huge change.
    if ablated_child.get("diverged") is True:
        out["ablated_reply_truncated"] = True
        out["diverged_at"] = ablated_child.get("diverged_at")
        out["early_stop_note"] = (
            "ablated arm early-stopped at the first token that diverged from the baseline (token index "
            f"{ablated_child.get('diverged_at')}): the reply shown is the bit-exact prefix up to that point, "
            "not the full ablated reply. has_effect is exact; the delta numbers reflect the divergence point.")
    if unapplied:
        out["ablation_note"] = unapplied
    return out
