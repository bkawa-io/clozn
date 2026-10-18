"""Privacy level for metadata-only Context Receipts.

Run content follows the journal's delete lifecycle. This setting controls how much source metadata the
separate Context Receipt retains: ``full``, ``metadata_only``, ``hashes_only``, or ``off``.
"""
from __future__ import annotations

import clozn.settings as settings


TIERS = ("full", "metadata_only", "hashes_only", "off")
DEFAULT = "full"
_KEY = "receipt_privacy"


def tier() -> str:
    """Return the active tier; missing or invalid settings use ``full``."""
    value = str(settings.get_setting(_KEY, DEFAULT) or "").strip().lower()
    return value if value in TIERS else DEFAULT


def set_tier(name: str) -> bool:
    """Persist a valid tier atomically through the shared settings store."""
    value = str(name or "").strip().lower()
    return value in TIERS and settings.set_setting(_KEY, value)


def builds_receipt(name: str | None = None) -> bool:
    """Return whether the selected tier builds a receipt at all."""
    return (name if name is not None else tier()) != "off"


__all__ = ["DEFAULT", "TIERS", "builds_receipt", "set_tier", "tier"]
