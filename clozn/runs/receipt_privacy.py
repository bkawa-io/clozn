"""receipt_privacy -- how much of a context receipt's OWN duplicated content is retained.

Distinct from clozn.runs.capture_mode (which governs per-token TRACE depth) and distinct from the
run-level redact/delete lifecycle in clozn.runs.mutations (which governs run['messages'] etc.). This
module governs only what clozn.runs.context_receipt.build_context_receipt() writes into a run's
`context_receipt` field, which today only ever duplicates content already present on the run
('survived.assembled_messages', 'survived.final_prompt') -- everything this tier hides is still
reachable via the run's own fields at 'full' redaction, exactly like today; a stricter tier trims what
the RECEIPT repeats, not what the run itself stores.

Four tiers, in the shape feature 06's spec asks for:

  full            everything build_context_receipt can capture -- today's default behavior.
  metadata_only   segment types/labels/order/hashes/reasons kept; full text (final_prompt,
                  assembled_messages) dropped from the receipt.
  hashes_only     segment ids/hashes/reason kept; everything else metadata-shaped drops too
                  (no source_label, no byte counts).
  off             the receipt is not built at all beyond the required schema_version/run_id/privacy
                  marker -- an explicit "disabled", never a silently empty-looking full receipt.

Mirrors capture_mode.py's settings-gate pattern exactly: stdlib only, the setting lives in the same
studio_settings.json via clozn.settings' never-raise get/set helpers.
"""
from __future__ import annotations

import clozn.settings as settings

TIERS = ("full", "metadata_only", "hashes_only", "off")
DEFAULT = "full"
_KEY = "receipt_privacy"


def tier() -> str:
    """The active receipt-privacy tier; absent / unknown / garbage -> "full" (today's existing behavior,
    unchanged for anyone who never touches this setting)."""
    v = str(settings.get_setting(_KEY, DEFAULT) or "").strip().lower()
    return v if v in TIERS else DEFAULT


def set_tier(name: str) -> bool:
    """Persist the tier (merge-write to studio_settings.json). False on an unknown name OR an IO failure
    (never raises) -- the caller reports, the request survives."""
    name = str(name or "").strip().lower()
    if name not in TIERS:
        return False
    return settings.set_setting(_KEY, name)


def includes_full_text(name: str | None = None) -> bool:
    """Does this tier retain full rendered/assembled TEXT on the receipt (final_prompt,
    assembled_messages)?  Only "full" does."""
    return (name if name is not None else tier()) == "full"


def includes_segment_metadata(name: str | None = None) -> bool:
    """Does this tier retain per-segment source_label/byte counts, or only id+hash+reason?"""
    return (name if name is not None else tier()) in ("full", "metadata_only")


def builds_receipt(name: str | None = None) -> bool:
    """Does this tier build a receipt at all? Only "off" declines."""
    return (name if name is not None else tier()) != "off"
