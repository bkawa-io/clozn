"""Message inspection and readable run export helpers used by the gateway."""
from __future__ import annotations

from clozn.server import app as ctx   # noqa: F401  the seam: live server state + patchable helpers


def _last_user(messages):
    """The last user turn's content ('' if none)."""
    return next((m.get("content", "") for m in reversed(messages or []) if m.get("role") == "user"), "")


def _export_markdown(run: dict, xr: dict | None) -> str:
    """Render a run (+ its M1 explain) as a human-readable Markdown receipt: the conversation, why it
    stopped, and where it hesitated. Pure / no model -- the JSON export carries the full structured
    bundle; this is its readable companion."""
    import clozn.receipts.bundle as receipt_bundle
    return receipt_bundle.to_markdown(receipt_bundle.build(run, explain=xr))
