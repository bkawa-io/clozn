"""memory_mode -- which mechanism carries the studio's memory cards.

Two modes, one persisted setting in ~/.clozn/studio_settings.json:

  "prompt"       (default for FRESH installs) -- the active card texts are compiled into one system
                 block and prepended to the chat, topic-gated per turn. Card edits are instant (no
                 retrain), per-card ablation is real, and the card text IS what's applied, verbatim.
  "internalized" -- today's trained soft-prefix path, untouched: cards drive consolidate() (a ~4-5 min
                 TTT retrain per active-set change). Kept as the research mode (the self-audit
                 experiments REQUIRE a non-text memory) and the context-constrained fallback.

Migration rule (don't silently change a live personality): when no mode was ever chosen, an existing
trained prefix on disk (~/.clozn/studio_memory.pt / studio_dream_memory.pt) resolves to "internalized"
until the user toggles; a fresh install resolves to "prompt".

BLOCK STYLE: a second, independent persisted setting, "block_style" -- "soft" (default,
unchanged) | "strict". Both are prompt-mode wording variants of the SAME rules; block_style never
affects "internalized" mode (the prefix has no prompt block to reword). The measured problem
(measured in an A/B follow-up): the soft block's "...use it naturally to tailor how you
respond" phrasing is a distillation-target wording that a strong instruction-follower (7B) over-satisfies
but a 1.5B under-fires on plain neutral probes -- two of four traits (space, question) came out
PREFIX-STRONGER at 1.5B, inverting the 7B verdict (PROMPT >= PREFIX everywhere). "strict" states the same
rules as direct imperatives (no "naturally"/"tailor" hedge) to test whether closing that soft-wording gap
closes the inversion. Soft stays the default and stays byte-identical to consolidate()'s sys_rule (the
lockstep test enforces this at the "soft" style only -- strict is not a distillation target and is free
to reword).

Mirrors memory_cards.py: stdlib only (torch-free, so replay.py stays model-free-testable), module-level
path globals tests can point at a tmp dir, and IO that NEVER raises -- a broken settings file degrades
to the migration default, never to a crashed request.
"""
from __future__ import annotations

import os

from clozn.settings import _load_settings, get_setting, set_setting   # noqa: F401  (re-exported)

_CLOZN = os.path.expanduser("~/.clozn")

# A trained prefix on disk == a live personality someone invested minutes of TTT in. Its existence is
# the migration signal: with no explicit choice recorded, keep serving it (internalized) rather than
# silently swapping the mechanism under the user. Module global so tests can isolate.
LEGACY_PREFIX_PATHS = [os.path.join(_CLOZN, "studio_memory.pt"),
                       os.path.join(_CLOZN, "studio_dream_memory.pt")]

MODES = ("prompt", "internalized")
PRODUCT_MODES = ("prompt",)

BLOCK_STYLES = ("soft", "strict")
DEFAULT_BLOCK_STYLE = "soft"      # unchanged wording; strict is opt-in (see module docstring)


def get_mode() -> str:
    """Return prompt-card mode in product processes; honor the persisted lab choice otherwise."""
    if os.environ.get("CLOZN_RUNTIME_KIND") == "product":
        return "prompt"
    mode = _load_settings().get("memory_mode")
    if mode in MODES:
        return mode
    try:
        if any(os.path.isfile(p) for p in LEGACY_PREFIX_PATHS):
            return "internalized"
    except Exception:
        pass
    return "prompt"


def set_mode(mode: str) -> bool:
    """Persist a valid mode; product processes refuse the lab-only internalized mode."""
    if mode not in MODES or (
        os.environ.get("CLOZN_RUNTIME_KIND") == "product" and mode not in PRODUCT_MODES
    ):
        return False
    return set_setting("memory_mode", mode)


def get_block_style() -> str:
    """The active prompt-block wording ("soft" | "strict"): the persisted choice if valid, else
    DEFAULT_BLOCK_STYLE ("soft" -- byte-identical to today's wording, no behaviour change for anyone
    who hasn't opted in). Independent of memory_mode -- this only matters when mode == "prompt"."""
    style = get_setting("block_style")
    return style if style in BLOCK_STYLES else DEFAULT_BLOCK_STYLE


def set_block_style(style: str) -> bool:
    """Persist the block-style choice. False on an invalid style or IO failure (never raises)."""
    if style not in BLOCK_STYLES:
        return False
    return set_setting("block_style", style)


def active_cards(exclude_ids=(), request_scope=None) -> list[dict] | None:
    """The ACTIVE memory cards as [{id, text}], minus exclude_ids (replay's per-card ablation) -- the
    prompt block's source of truth. [] when the store is simply empty; None if memory_cards itself is
    unavailable, so callers can tell "no cards" from "no store" and keep their own fallbacks."""
    exclude = {str(i) for i in (exclude_ids or ())}
    try:
        from clozn.memory import cards as memory_cards
        from clozn.memory import scope as memory_scope
        current = request_scope if isinstance(request_scope, memory_scope.MemoryScope) \
            else memory_scope.MemoryScope()
        active = [card for card in memory_cards.list_cards(status="active")
                  if card.get("text") and str(card.get("id")) not in exclude]
        eligible = memory_scope.eligible_cards(active, current)
        return [{"id": card.get("id"), "text": card["text"],
                 "scope_kind": memory_scope.scope_for_card(card)["kind"]}
                for card in eligible]
    except Exception:
        return None


def compile_prompt_block(texts: list[str], style: str | None = None) -> str:
    """The active card texts as ONE system block -- what prompt mode prepends to every gated-in turn.

    `style` selects the wording: "soft" (default) or "strict". None (the default, and
    every pre-existing call site's behaviour) reads the persisted setting via get_block_style() -- so
    this signature is back-compatible: no caller needs to change to keep behaving exactly as before.
    Pass an explicit style to override the setting (e.g. the A/B rig comparing both on one process).

    "soft" -- the ORIGINAL wording, MUST stay verbatim-identical to SelfTeach.consolidate's `sys_rule`
    (self_teach_server.py): that string is the distillation target the internalized prefix is trained to
    imitate, so keeping them in lockstep is what makes the two modes behaviourally comparable (the
    black-box A/B rests on it). Never reword "soft" -- add a new style instead.

    "strict" -- the SAME rules as direct imperatives, no "use it naturally to tailor" hedge. Measured
    motivation (the measured A/B follow-up): that softer framing is under-satisfied by a
    1.5B on plain neutral probes (2/4 traits inverted vs the trained prefix there, though 7B satisfied it
    fine) -- strict tests whether stating the rules as instructions closes that gap. Not a distillation
    target; free to reword independently of "soft"/consolidate's sys_rule.

    Empty/blank-only input -> "" (the caller omits the block entirely) regardless of style. Pure;
    preserves text order."""
    rules = [str(t).strip() for t in (texts or []) if str(t or "").strip()]
    if not rules:
        return ""
    resolved = get_block_style() if style is None else style
    if resolved == "strict":
        return ("Follow these facts and rules about the user exactly, in every reply, without exception:\n"
                + "\n".join("- " + r for r in rules))
    return ("You are a helpful assistant talking with a returning user. Here is what you know "
            "about them; use it naturally to tailor how you respond:\n"
            + "\n".join("- " + r for r in rules))
