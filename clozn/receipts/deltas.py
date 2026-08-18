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
    if influence.get("section"):
        return f"section:{influence['section']}"
    return "unknown"


def _ablation_changes(influence: dict) -> dict | None:
    """One influence spec -> replay changes for its ablated arm. `section` is the only ablatable influence
    kind left in the product (memory-card and tone-dial ablation were cut along with cards and dials);
    anything else -- including a stray legacy `card_id`/`memory_off`/`behavior_off` spec from before that
    cut -- is not something this codebase can ablate, and returns None rather than claim otherwise."""
    if not isinstance(influence, dict):
        return None
    section = influence.get("section")
    if section:
        return {"exclude_sections": [str(section)]}
    return None


def _section_influences(manifest: dict) -> tuple[list, list]:
    """Sections from the M1 manifest (`influences_active.sections`, produced by explain.py from a run's
    `sections` field) as receipt influence specs -- (influences, skipped), the same two-list shape
    `_fired_influences` in core.py returns directly to its own caller.

    SOURCE WHITELIST: `clozn.runs.sections` only ever produces `source: "api"` or `"auto"` today (memory
    cards, and the third `"memory_card"` section source that used to shadow a fired card's own richer
    `card_id` ablation path, were cut from the product on 2026-07-27 along with the rest of memory). The
    `in ("api", "auto")` check below is kept as a defensive whitelist rather than dropped outright -- a
    run recorded before that cut may still carry a stray `"memory_card"`-sourced entry in its manifest,
    and this must skip it rather than try to ablate a source this codebase no longer knows how to."""
    sections = ((manifest or {}).get("influences_active") or {}).get("sections") or {}
    if isinstance(sections, dict):                     # explain.py's {"available": .., "sections": [...]}
        sections = sections.get("sections") or []
    influences: list = []
    skipped: list = []
    for sec in sections if isinstance(sections, list) else []:
        if not isinstance(sec, dict):
            continue
        if sec.get("source") not in ("api", "auto"):
            continue                                    # unknown/legacy source (e.g. old "memory_card"
                                                          # entries) -- see SOURCE WHITELIST above
        name = sec.get("name")
        if not name:
            skipped.append({"influence": {"section_source": sec.get("source")},
                            "reason": "section has no name recorded; per-section ablation needs a name"})
            continue
        influences.append({"section": name, "source": sec.get("source")})
    return influences, skipped


def _merge_ablation_changes(influences: list) -> dict:
    """Joint replay changes that ablate every (section) influence at once."""
    sections: list = []
    for inf in influences:
        c = _ablation_changes(inf) or {}
        sections.extend(c.get("exclude_sections") or [])
    return {"exclude_sections": sections} if sections else {}


def _cost_note(influence: dict) -> str:
    influence = influence or {}
    if influence.get("section"):
        return ("cost: a section ablation edits the prompt content itself, so the ablated arm re-prefills "
                "the whole context (no KV reuse) -- the expensive case.")
    return "cost: unknown ablation kind; no cost model for this influence spec."


def _unapplied_note(ablated_child: dict, changes: dict) -> str | None:
    """Honest "this ablation didn't (fully) apply" text, read off the child replay's own
    `section_notes` (replay.py's `_apply_section_exclusions` -- keyed by requested section name, one
    entry per name that did NOT fully apply: unknown name, no manifest, no usable parts, or a
    final_prompt-anchored part replay's chat(messages, ...) surface can't splice). None when the change
    wasn't a section exclusion, or every requested section applied cleanly."""
    if not changes.get("exclude_sections"):
        return None
    sec_notes = (ablated_child or {}).get("section_notes") or {}
    if not sec_notes:
        return None
    return "; ".join(f"{k}: {v}" for k, v in sec_notes.items())


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
        # sections side by side without having to sniff `influence`'s shape.
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
