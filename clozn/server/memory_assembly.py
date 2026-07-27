"""clozn.server.memory_assembly -- how one turn's memory reaches the model.

The prompt-mode pipeline (active cards -> topic gate -> assembled block -> injection), the anchored-
memory apply + loop guard, card migration/sync, and the provenance/risk helpers for proposed cards.
Extracted from clozn.server.app; app remains the seam (state + patchable helpers) -- this module reads
every patched name (`_prompt_gate`, `_prompt_mem_cards`, `ctx.PROMPT_GATE_MIN`, ...) through the late-bound
`ctx` so monkeypatches on the app module are always seen, and app re-exports every public name here.
"""
from __future__ import annotations

from clozn.server import app as ctx   # the seam: live server state + patchable helpers (see docstring)

# ------- memory cards <-> the working prefix (D2 + E1) --------------------------------------------
# The cards (research/memory_cards.py) are the metadata + review layer; the trained soft-prefix is
# UNCHANGED. The contract that keeps the prefix safe: m.rules is ALWAYS the texts of the ACTIVE cards,
# and the prefix is built from m.rules via m.consolidate(rules) exactly as before. So a card's STATUS
# decides what's in m.rules, which drives the prefix. We only ever retrain when the active set actually
# changes (a no-op transition -- e.g. approving a card whose text is already active -- never touches it).

_SUSPICIOUS = ("ignore ", "disregard ", "system prompt", "you are now", "forget ", "override",
               "jailbreak", "developer mode", "instead of", "from now on you", "pretend ")


def _risk_of(text: str) -> str:
    """Cheap heuristic: flag instruction-like / prompt-injection-ish memory text as 'suspicious' so the
    reviewer sees it. A memory is meant to be a fact/preference ABOUT the user, not a command to the model."""
    t = (text or "").lower()
    return "suspicious" if any(s in t for s in _SUSPICIOUS) else "low"


def _dial_suggestion(text: str):
    """If a memory's text is really a STYLE preference that maps to a tone dial, return that suggestion
    ({axis, value, pole_label}); else None. Guarded import of steering.suggest_dial_for_preference so a
    missing/broken steering module (or the pure-engine substrate) can never break /memory/add or propose.
    Pure + deterministic (a lexicon match, no model) -- see steering.suggest_dial_for_preference."""
    try:
        from clozn.behavior.steering.catalog import suggest_dial_for_preference
        return suggest_dial_for_preference(text)
    except Exception:
        return None


QUOTE_SPAN_MAX = 240   # a "you said this" quote is for recognizing your own words, not re-reading the essay


def _provenance_of(messages):
    """The (source_turn, quoted_span) pair for a card proposed from `messages` (the
    OBEY defense -- see memory_cards.has_provenance). source_turn is the index of the LAST user message in
    the list (mirrors _last_user's "most recent user turn" convention, and matches dream_consolidation.py's
    `"turn": i` = index into a run's messages); quoted_span is that message's own verbatim text, truncated
    to QUOTE_SPAN_MAX chars -- never paraphrased, never the model's synthesized third-person card text.
    (None, "") when there's no user message to cite at all (defensive: propose_memory needs user content to
    work from, so this should be rare) -- that is exactly the "claimed a run but can't back it up" case the
    approve-gate refuses on, and the Memory page flags."""
    for i in range(len(messages or []) - 1, -1, -1):
        m = messages[i]
        if isinstance(m, dict) and m.get("role") == "user":
            content = str(m.get("content") or "").strip()
            if content:
                span = content if len(content) <= QUOTE_SPAN_MAX else content[:QUOTE_SPAN_MAX].rstrip() + "…"
                return i, span
    return None, ""


# ------- memory MODE: prompt-carried cards vs the internalized prefix (MEMORY_MODE_SWAP_SPEC) ------
# mode "prompt" (the fresh-install default): the ACTIVE card texts are compiled into ONE system block
# (memory_mode.compile_prompt_block -- verbatim the sys_rule wording the prefix trains toward) and
# prepended to the chat, topic-gated PER TURN; generation runs with use_prefix=False. Card mutations
# skip consolidate()/_TRAIN_LOCK entirely (instant). mode "internalized": today's prefix path, exactly
# as before. An existing trained prefix keeps "internalized" until the user toggles (memory_mode.py).

PROMPT_GATE_MIN = 0.05     # gate below this -> the block is OMITTED for the turn. Prompt mode controls
                           # over-bleed by omission (binary), not by the prefix's continuous scaling.


class PromptBlockDecision(tuple):
    """Backward-compatible 3-tuple carrying the per-turn card selection receipt."""
    def __new__(cls, block, applied, gate, *, candidates=(), omitted=(), omission_reason=None):
        value = super().__new__(cls, (block, applied, gate))
        value.candidates = [dict(card) for card in candidates if isinstance(card, dict)]
        value.omitted = [dict(card) for card in omitted if isinstance(card, dict)]
        value.omission_reason = omission_reason
        return value


def _capture_prompt_decision(mem_out, decision) -> None:
    """Copy a decision onto request-local evidence without consulting mutable card state later."""
    if mem_out is None:
        return
    applied = decision[1] if isinstance(decision, (tuple, list)) and len(decision) > 1 else []
    candidates = getattr(decision, "candidates", None)
    if candidates is None:
        # Test doubles and third-party adapters may still return the historical plain tuple.
        candidates = [dict(card) for card in (applied or []) if isinstance(card, dict)]
    mem_out["candidate_cards"] = [dict(card) for card in candidates]
    mem_out["omitted_cards"] = [dict(card) for card in getattr(decision, "omitted", [])]
    mem_out["selection_stage"] = "active_prompt_cards_considered_by_turn_gate"
    reason = getattr(decision, "omission_reason", None)
    if reason:
        mem_out["omission_reason"] = reason


def _baseline_prompt_tokens(engine, messages) -> int | None:
    """Count the same chat template without memory when a supporting worker is available."""
    try:
        info = engine.apply_template_info(messages)
        count = info.get("prompt_tokens") if isinstance(info, dict) else None
        return count if isinstance(count, int) and not isinstance(count, bool) and count >= 0 else None
    except Exception:
        return None


def _memory_mode():
    """Product always uses prompt cards; the optional lab can still select internalized memory."""
    try:
        import clozn.memory.mode as memory_mode
        return memory_mode.get_mode()
    except Exception:
        return "internalized" if getattr(ctx, "RUNTIME_KIND", "product") == "lab" else "prompt"


def _last_user(messages):
    """The last user turn's content ('' if none) -- the topic-gate input, same as the prefix path."""
    return next((m.get("content", "") for m in reversed(messages or []) if m.get("role") == "user"), "")


def _prompt_gate(_last_user, _texts):
    """Ungated: every active card enters the prompt block.

    This used to call topic_gate.scalar(). BEHAVIOUR IS UNCHANGED by removing it, because the gate
    needed `sentence-transformers` -- which is not a product dependency (pyproject `dependencies = []`,
    enforced by the torch-free CI lane) -- and returned 1.0 whenever the embedder was missing. In the
    shipped configuration it therefore always returned 1.0 and gated nothing. Kept as a function
    rather than inlined so every call site and the run-record shape stay identical."""
    return 1.0


def _prompt_relevance(_last_user, _texts):
    """No per-card relevance signal. Same reason as _prompt_gate: this read cosines from the MiniLM
    embedder and returned {} without it, which was the shipped case. The inspector already treats {}
    as "no relevance data", so nothing downstream changes."""
    return {}


def _prompt_mem_cards(mem, exclude_ids=(), request_scope=None):
    """The ACTIVE cards ({id, text}) that feed the prompt block, minus exclude_ids (replay's REAL
    per-card ablation). Reads the card store directly (memory_mode.active_cards) -- in prompt mode the
    cards ARE the memory (m.rules is bookkeeping that can lag right after boot). Falls back to
    mem.rules (id-less) only if the store module is unavailable, so a broken store degrades to the old
    rule list rather than to amnesia."""
    import clozn.memory.mode as memory_mode
    cards = memory_mode.active_cards(exclude_ids, request_scope=request_scope)
    if cards is not None:
        return cards
    return [{"id": None, "text": t} for t in (getattr(mem, "rules", []) or []) if t]


def _prompt_block_for(mem, last_user, strength=None, request_scope=None):
    """Prompt-mode injection decision for THIS turn -> (block_text | None, applied_cards, gate).

    None == omit the block entirely: no active cards, strength == 0 (the dial maps to on/off in prompt
    mode -- 0 never injects, >0 injects when gated in), or the topic gate is ~0 (off-topic turn).
    applied_cards is [] whenever the block is omitted. Honors mem._exclude_card_ids (set temporarily
    by replay.py for per-card receipts). `strength` overrides mem.memory_strength (the pure-engine
    path reads it from disk); `mem` may be None on that path -- every read of it is defensive."""
    cards = ctx._prompt_mem_cards(mem, getattr(mem, "_exclude_card_ids", None) or (),
                                  request_scope=request_scope)
    texts = [c["text"] for c in cards]
    s = float(strength if strength is not None else getattr(mem, "memory_strength", 1.0))
    if not texts:
        return PromptBlockDecision(None, [], 0.0, candidates=cards,
                                   omission_reason="no_active_cards")
    if s <= 0.0:
        return PromptBlockDecision(None, [], 0.0, candidates=cards, omitted=cards,
                                   omission_reason="memory_strength_zero")
    g = ctx._prompt_gate(last_user, texts)
    if g < ctx.PROMPT_GATE_MIN:
        return PromptBlockDecision(None, [], g, candidates=cards, omitted=cards,
                                   omission_reason="topic_gate_below_threshold")
    rel = ctx._prompt_relevance(last_user, texts)          # {text: cosine} per card (best-effort; {} if no embedder)
    applied = [dict(c, relevance=rel.get(c["text"])) for c in cards]
    import clozn.memory.mode as memory_mode
    return PromptBlockDecision(memory_mode.compile_prompt_block(texts), applied, g,
                               candidates=cards)


# ANCHORED MEMORY REMOVED (2026-07-27). _anchored_gates / _apply_anchored_memory /
# _anchored_loop_guard lived here. They composed a k-sparse {token: alpha} bag into one steer
# vector at L21. Measured on its own qualified config and it could not recall: the bag for a card
# naming a cat 'Miso' adopted in 'Kyoto' stored {cat, grey, named, adopted} -- dir(c) needs
# single-token words, so every proper noun was dropped -- and the injection induced a fabricated
# "your cat named Qwen". See notes/ANCHORED_MEMORY_FINDINGS.md.


def _inject_block(messages, block):
    """`messages` with the memory block folded in as system context (a copy -- never mutates the
    caller's list). Appends to an existing system message (the client's own instructions keep first
    position) or prepends a new one; a None/empty block returns the messages unchanged."""
    if not block:
        return list(messages)
    msgs = [dict(m) for m in messages]
    for m in msgs:
        if m.get("role") == "system":
            m["content"] = (str(m.get("content") or "") + "\n\n" + block).strip()
            return msgs
    return [{"role": "system", "content": block}] + msgs


def _mem_migrate(m):
    """Seed the card store from a memory object's legacy rule-strings, ONCE. migrate_from_rules is a
    no-op when the store already has cards, and it creates them as ACTIVE -- the prefix is already trained
    on these exact rules, so we do NOT re-consolidate here. Returns the cards created (or [])."""
    import clozn.memory.cards as memory_cards
    try:
        return memory_cards.migrate_from_rules(list(getattr(m, "rules", []) or []))
    except Exception:
        return []


def _export_markdown(run: dict, xr: dict | None) -> str:
    """Render a run (+ its M1 explain) as a human-readable Markdown receipt: the conversation, what memory
    and which dials shaped it (with per-card relevance), why it stopped, and where it hesitated. Pure / no
    model -- the JSON export carries the full structured bundle; this is its readable companion."""
    import clozn.receipts.bundle as receipt_bundle
    return receipt_bundle.to_markdown(receipt_bundle.build(run, explain=xr))


def _runs_for_card(card_id):
    """Best-effort: the run summaries whose memory.cards_applied names this card (by id OR by text).
    cards_applied currently records the active rule TEXTS (see _log_run), so we match on text primarily
    and on id as a forward-compatible fallback. Returns [] when the card / runs are gone (never raises)."""
    import clozn.memory.cards as memory_cards
    import clozn.runs.store as runlog
    try:
        card = memory_cards.get(card_id)
        text = (card or {}).get("text", "")
        needles = {n for n in (card_id, text) if n}
        if not needles:
            return []
        out = []
        for r in runlog.list_runs(500):
            applied = ((r.get("memory") or {}).get("cards_applied")) or []
            applied = [str(a) for a in applied]
            if needles & set(applied):
                out.append(r)
        return out
    except Exception:
        return []


def _mem_sync_rules(m, reconsolidate=True, force=False):
    """Make m.rules == the active-card texts, then rebuild the prefix ONLY if the active set changed.

    This is the one place the prefix can move. If the active texts are identical to what m.rules already
    holds, we leave the prefix completely untouched (the expensive, working artifact is preserved). When
    the set changed and reconsolidate is on, we retrain from the active texts (SLOW -- expected on
    approve/reject/disable/edit). If the active set became EMPTY (e.g. the last card was disabled), we
    reset() so the now-unused prefix stops biting -- reset() is zero-arg on both memory backends.
    `force` retrains even when m.rules already matches the store -- used when toggling BACK to
    internalized mode, where the rules are synced but the PREFIX may be stale (prompt-mode card edits
    never consolidate)."""
    import clozn.memory.cards as memory_cards
    new_rules = memory_cards.active_texts()
    changed = list(getattr(m, "rules", []) or []) != list(new_rules)
    m.rules = list(new_rules)
    result = None
    if (changed or force) and reconsolidate:
        if new_rules:
            result = m.consolidate(list(new_rules))
        else:                                    # nothing active anymore -> drop the prefix entirely
            try:
                result = m.reset()
            except Exception:
                pass
            m.rules = []                          # reset() may clear rules; keep them in sync
    return {"changed": changed, "rules": list(new_rules), "consolidate": result}

