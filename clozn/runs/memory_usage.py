"""Evidence-only memory usage receipts for stored runs.

This module only reshapes evidence captured on the run itself.  In particular it
never consults the current card store: cards may have changed since the run was
recorded, so live state cannot prove what happened on an earlier turn.
"""
from __future__ import annotations

import math
from collections.abc import Mapping


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _card_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        text = value.get("text")
        return text if isinstance(text, str) else ""
    return "" if value is None else str(value)


def _relevance(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return round(number, 4) if math.isfinite(number) else None


def _nonnegative_int(value):
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _cards(memory: Mapping) -> list[dict]:
    texts = _list(memory.get("cards_applied"))
    ids = _list(memory.get("applied_ids"))
    relevance = _list(memory.get("relevance"))
    scopes = _list(memory.get("applied_scope_kinds"))
    cards = []
    for index, text in enumerate(texts):
        card = {
            "text": _card_text(text),
            "id": ids[index] if index < len(ids) else None,
            "relevance": _relevance(relevance[index]) if index < len(relevance) else None,
        }
        if index < len(scopes) and scopes[index] in {"global", "app", "project"}:
            card["scope_kind"] = scopes[index]
        cards.append(card)
    return cards


def _snapshot_cards(value) -> list[dict]:
    """Copy an additive capture-time card snapshot without enriching it from live state."""
    return [dict(card) for card in _list(value) if isinstance(card, Mapping)]


def _total_prompt_tokens(run: Mapping) -> tuple[int | None, str | None]:
    receipt = _mapping(run.get("context_receipt"))
    limits = _mapping(receipt.get("limits"))
    value = _nonnegative_int(limits.get("prompt_tokens"))
    if value is not None:
        return value, "context_receipt.limits.prompt_tokens"
    value = _nonnegative_int(_mapping(run.get("meta")).get("prompt_tokens"))
    if value is not None:
        return value, "meta.prompt_tokens"
    return None, None


def _anchored(memory: Mapping) -> dict:
    """Anchored-memory evidence. DELIBERATELY OUTLIVES the 2026-07-27 anchored-memory removal.

    No code writes these keys anymore, so every run recorded since reads "not_observed". But runs
    recorded BEFORE the cut carry real evidence of what actually rode those turns, and this module's
    contract (see the module docstring) is to reshape what the run recorded -- never to re-judge it
    against today's feature set. Deleting the reader would silently rewrite that history to "no memory
    was applied", which is the one thing a receipt may not do."""
    keys = ("anchored", "anchored_layer", "anchored_s_total", "anchored_skipped",
            "anchored_loop_guard", "anchored_scope_excluded_count")
    if not any(key in memory for key in keys):
        return {"status": "not_observed"}
    bags = [dict(item) for item in _list(memory.get("anchored")) if isinstance(item, Mapping)]
    return {
        "status": "observed",
        "bags": bags,
        "count": len(bags),
        "layer": memory.get("anchored_layer"),
        "s_total": memory.get("anchored_s_total"),
        "skipped": memory.get("anchored_skipped"),
        "scope_excluded_count": _nonnegative_int(memory.get("anchored_scope_excluded_count")),
        "loop_guard": (dict(memory["anchored_loop_guard"])
                       if isinstance(memory.get("anchored_loop_guard"), Mapping) else None),
        "evidence": [f"memory.{key}" for key in keys if key in memory],
    }


def _facts(memory: Mapping) -> dict:
    facts = memory.get("facts")
    if not isinstance(facts, Mapping):
        return {"status": "not_observed"}
    return {"status": "observed", "evidence": dict(facts), "source": "memory.facts"}


def memory_usage(run: dict | None) -> dict:
    """Build a JSON-ready receipt from one stored run, without reading mutable state."""
    run = run if isinstance(run, Mapping) else {}
    memory = _mapping(run.get("memory"))
    changes = _mapping(run.get("changes_applied"))
    mode = memory.get("mode") if isinstance(memory.get("mode"), str) else None
    has_card_evidence = isinstance(memory.get("cards_applied"), list)
    recorded_cards = _cards(memory) if has_card_evidence else []
    candidate_evidence = isinstance(memory.get("candidate_cards"), list)
    candidate_cards = _snapshot_cards(memory.get("candidate_cards"))

    if mode == "prompt" and has_card_evidence:
        injected = {
            "status": "observed",
            "cards": recorded_cards,
            "count": len(recorded_cards),
            "evidence": [f"memory.{key}" for key in ("cards_applied", "applied_ids", "relevance")
                         if key in memory],
        }
    elif mode == "internalized" and has_card_evidence:
        injected = {
            "status": "not_applicable",
            "cards": [],
            "count": 0,
            "note": "Internalized memory is fused; cards_applied records the active set, not prompt injection.",
        }
    else:
        injected = {
            "status": "unavailable",
            "cards": [],
            "count": None,
            "note": "No prompt-mode cards_applied evidence was recorded for this run.",
        }

    if mode == "prompt" and candidate_evidence:
        selected = {
            "status": "observed",
            "cards": candidate_cards,
            "count": len(candidate_cards),
            "selection_stage": memory.get("selection_stage"),
            "evidence": [key for key in ("memory.candidate_cards", "memory.selection_stage")
                         if key.rsplit(".", 1)[1] in memory],
            "note": "This is the capture-time card set considered by the turn gate.",
        }
    elif mode == "prompt" and has_card_evidence:
        selected = {
            "status": "derived",
            "cards": recorded_cards,
            "count": len(recorded_cards),
            "basis": "same_as_injected",
            "note": ("Historical fallback: prompt memory has no later per-card stage after its "
                     "whole-block gate; selected is the exact injected set recorded for this run."),
        }
    elif mode == "internalized":
        selected = {
            "status": "unavailable",
            "cards": [],
            "count": None,
            "note": "A per-turn selected-card set is not recorded for internalized memory.",
        }
    else:
        selected = {
            "status": "unavailable",
            "cards": [],
            "count": None,
            "note": "A selected-card set cannot be recovered from this run record.",
        }

    disabled = changes.get("disabled_memory_ids")
    omitted_evidence = isinstance(memory.get("omitted_cards"), list)
    if omitted_evidence:
        omitted_cards = _snapshot_cards(memory.get("omitted_cards"))
        omitted_ids = [card.get("id") for card in omitted_cards if card.get("id") is not None]
        if isinstance(disabled, list):
            omitted_ids.extend(value for value in disabled if value not in omitted_ids)
        evidence = ["memory.omitted_cards"]
        if "omission_reason" in memory:
            evidence.append("memory.omission_reason")
        if isinstance(disabled, list):
            evidence.append("changes_applied.disabled_memory_ids")
        omitted = {
            "status": "observed",
            "cards": omitted_cards,
            "ids": omitted_ids,
            "reason": memory.get("omission_reason"),
            "evidence": evidence,
            "note": "Capture-time omissions; explicitly disabled IDs are included when present.",
        }
    elif isinstance(disabled, list):
        omitted = {
            "status": "observed",
            "cards": [],
            "ids": list(disabled),
            "reason": None,
            "evidence": ["changes_applied.disabled_memory_ids"],
            "note": "These card IDs were explicitly disabled for this run.",
        }
    else:
        omitted = {
            "status": "unavailable",
            "cards": [],
            "ids": [],
            "reason": None,
            "note": "Candidate and omitted card identities were not recorded for this run.",
        }

    prompt_block = memory.get("prompt_block")
    prompt_block = prompt_block if isinstance(prompt_block, str) and prompt_block else None
    total_tokens, total_source = _total_prompt_tokens(run)
    exact_cost = _nonnegative_int(memory.get("prompt_token_cost"))
    baseline_tokens = _nonnegative_int(memory.get("baseline_prompt_tokens"))
    unavailable_reason = memory.get("prompt_token_cost_unavailable_reason")
    unavailable_reason = unavailable_reason if isinstance(unavailable_reason, str) else None
    if exact_cost is not None:
        token_cost = {
            "status": "observed",
            "memory_prompt_tokens": exact_cost,
            "prompt_block_utf8_bytes": len(prompt_block.encode("utf-8")) if prompt_block else 0,
            "baseline_prompt_tokens": baseline_tokens,
            "total_prompt_tokens": total_tokens,
            "total_prompt_tokens_source": total_source,
            "evidence": [key for key in ("memory.prompt_token_cost", "memory.baseline_prompt_tokens")
                         if key.rsplit(".", 1)[1] in memory],
            "note": "Exact matched prompt-token delta captured for this run.",
        }
    elif prompt_block is None:
        token_cost = {
            "status": "observed",
            "memory_prompt_tokens": 0,
            "prompt_block_utf8_bytes": 0,
            "baseline_prompt_tokens": baseline_tokens,
            "total_prompt_tokens": total_tokens,
            "total_prompt_tokens_source": total_source,
            "note": "No prompt block was recorded, so prompt-memory token cost was zero.",
        }
    else:
        token_cost = {
            "status": "unavailable",
            "memory_prompt_tokens": None,
            "prompt_block_utf8_bytes": len(prompt_block.encode("utf-8")),
            "baseline_prompt_tokens": baseline_tokens,
            "total_prompt_tokens": total_tokens,
            "total_prompt_tokens_source": total_source,
            "unavailable_reason": unavailable_reason,
            "note": ("The exact prompt block bytes and total prompt tokens are recorded, but no "
                     "memory-specific token delta was captured; it is not estimated."),
        }

    internalized = {"status": "not_applicable"}
    if mode == "internalized" and has_card_evidence:
        internalized = {
            "status": "observed",
            "active_cards": recorded_cards,
            "count": len(recorded_cards),
            "evidence": "memory.cards_applied",
        }

    return {
        "schema": "clozn.memory_usage.v1",
        "run_id": run.get("id"),
        "mode": mode,
        "prompt_cards": {
            "injected": injected,
            "selected": selected,
            "omitted": omitted,
            "gate": memory.get("gate"),
            "strength": memory.get("strength"),
            "prompt_block": prompt_block,
        },
        "internalized": internalized,
        "anchored": _anchored(memory),
        "facts": _facts(memory),
        "token_cost": token_cost,
    }
