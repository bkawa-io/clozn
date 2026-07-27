"""capture_mode -- how much white-box data a run stores: the Light / Standard / Deep / Lab tier.

A capture TIER trades latency + storage for detail. The ladder (agreed with the studio design):

  light     text + finish_reason + metadata           -- no per-token trace: fastest, smallest
  standard  + per-token trace (tokens/conf/alts)       -- the default; what every run captured before tiers
  deep      + raw activations (state="full")           -- needs the engine activation tap (staged)
  lab       + SAE concept features + probes            -- needs --sae at launch (staged)

What v1 ENFORCES is the one dimension that is pure record policy: whether the per-token trace is STORED
(light drops it; standard/deep/lab keep it). The heavier deep/lab capture -- raw activations, SAE features
-- needs engine wiring the chat path does not have yet, so those tiers are DEFINED and RECORDED on the run
(so they light up the day that lands) but until then store exactly what standard does. Either way the
active tier is written onto every run's meta, so a reader always knows how much was captured.

Mirrors memory_mode's own settings-gate pattern: stdlib only (torch-free, model-free-testable); the
setting lives in the SAME studio_settings.json via memory_mode's never-raise get/set.
"""
from __future__ import annotations

import clozn.settings as settings  # the single settings file + its never-raise get/set helpers

TIERS = ("light", "standard", "deep", "lab")
DEFAULT = "standard"
_KEY = "capture_tier"


def tier() -> str:
    """The active capture tier; absent / unknown / garbage -> "standard" (the everyday default)."""
    v = str(settings.get_setting(_KEY, DEFAULT) or "").strip().lower()
    return v if v in TIERS else DEFAULT


def set_tier(name: str) -> bool:
    """Persist the tier (merge-write to studio_settings.json). False on an unknown name OR an IO failure
    (never raises) -- the caller reports, the request survives."""
    name = str(name or "").strip().lower()
    if name not in TIERS:
        return False
    return settings.set_setting(_KEY, name)


def captures_trace(name: str | None = None) -> bool:
    """Does this tier STORE the per-token trace? Light does not (text-only); standard/deep/lab do. The one
    record policy v1 enforces -- deep/lab's extra capture (activations, SAE) is staged for engine wiring."""
    return (name if name is not None else tier()) != "light"
