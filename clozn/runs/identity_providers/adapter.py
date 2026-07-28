"""Fine-tune adapter identity: which adapter weights, if any, produced this run.

Lands at `identity["ext"]["adapter"]`. Absent entirely when no adapter is attached -- which is the
common case, and must stay distinguishable from "an adapter was attached but we failed to describe it".

WHY THIS MATTERS MORE THAN MOST FACETS
--------------------------------------
Roadmap rule 2 wants a run to record the exact weights that answered. A base model plus a LoRA is a
DIFFERENT set of effective weights from the base model alone, and the difference is invisible in the
model path, the model sha256, and the template fingerprint -- every field run identity already records
would be byte-identical across an adapted and an unadapted run of the same base. Without this facet,
two runs that answered differently for a completely explicable reason would look identical in their
receipts, and `clozn compare-runs` would report "no difference found" about a real one.

The engine is the source of truth. It reports the attached adapter on GET /health as a top-level `lora`
object (path, scale, and the adapter GGUF's own metadata read back off the file), present only when one
is actually attached. This provider reads that -- it never infers an adapter from a CLI flag, because
the flag records what was REQUESTED and the health block records what was LOADED, and the whole point of
an identity block is to record the second.
"""
from __future__ import annotations

NAME = "adapter"

# Metadata keys worth promoting out of the adapter's raw GGUF KV block. Everything else stays under
# `meta` rather than being dropped -- an adapter may declare fields this list has not learned about.
_PROMOTED = {
    "adapter.lora.alpha": "alpha",
    "general.architecture": "architecture",
}


def identity(context) -> dict:
    """The adapter facet, read off the engine's /health block. `{}` when no adapter is attached."""
    health = (context or {}).get("engine_health")
    if not isinstance(health, dict):
        return {}

    lora = health.get("lora")
    if not isinstance(lora, dict) or not lora:
        # No adapter attached. Note this is NOT the same as the engine lacking the capability -- that
        # shows up as capabilities.lora being absent/false and is the engine's fact to report, not a
        # per-run identity fact.
        return {}

    out: dict = {}
    path = lora.get("path")
    if isinstance(path, str) and path:
        out["path"] = path

    # Scale is recorded even at 0.0: "attached and contributing nothing" is a materially different run
    # from "no adapter", and it is exactly the identity control an experiment would use. `if scale:`
    # here would erase that distinction.
    scale = lora.get("scale")
    if isinstance(scale, (int, float)) and not isinstance(scale, bool):
        out["scale"] = float(scale)

    meta = lora.get("meta")
    if isinstance(meta, dict) and meta:
        for key, promoted in _PROMOTED.items():
            value = meta.get(key)
            if value not in (None, ""):
                out[promoted] = value
        out["meta"] = meta

    return out
