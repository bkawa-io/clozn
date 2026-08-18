"""clozn.server.message_assembly -- shaping the message list and rendering a run.

This was `memory_assembly.py`, and it held the prompt-mode card pipeline: active cards -> topic gate ->
compiled system block -> injection, plus card migration/sync, the anchored-memory apply + loop guard,
and the risk/provenance helpers for proposed cards. Memory cards were cut from the product on
2026-07-27 so that steering is the only personalization surface, and all of that went with them.

What is left has nothing to do with memory, hence the rename:

  * `_inject_block` -- fold ANY system block into a message list. Branch Fan and
    `cli/commands/quant_check.py` each keep a local copy and document matching this exact shape
    (append to an existing system message so the client's own instructions keep first position, else
    prepend a new one), so this stays the canonical definition they are checked against.
  * `_last_user` -- the most recent user turn's content. Used by app's proposal/title path.
  * `_export_markdown` -- render a run + its explain as a readable receipt.

app remains the seam (state + patchable helpers) and re-exports these; this module reads anything
patchable through the late-bound `ctx` so monkeypatches on the app module are always seen.
"""
from __future__ import annotations

from clozn.server import app as ctx   # noqa: F401  the seam: live server state + patchable helpers


def _last_user(messages):
    """The last user turn's content ('' if none)."""
    return next((m.get("content", "") for m in reversed(messages or []) if m.get("role") == "user"), "")


def _inject_block(messages, block):
    """Compatibility name for the canonical experiment effective-prompt seam."""
    from clozn.experiments.effective_prompt import inject_block
    return inject_block(messages, block)


def _export_markdown(run: dict, xr: dict | None) -> str:
    """Render a run (+ its M1 explain) as a human-readable Markdown receipt: the conversation, why it
    stopped, and where it hesitated. Pure / no model -- the JSON export carries the full structured
    bundle; this is its readable companion."""
    import clozn.receipts.bundle as receipt_bundle
    return receipt_bundle.to_markdown(receipt_bundle.build(run, explain=xr))
