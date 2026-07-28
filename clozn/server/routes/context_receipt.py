"""GET /runs/<id>/context-receipt -- feature 06's dedicated context-receipt endpoint.

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4), spliced in before the generic GET /runs/<id>
fallback, so this /runs/<id>/<suffix> route is reachable at all -- see clozn/server/routes/_autoload.py.

Also serves GET/POST /receipt-privacy for clozn.runs.receipt_privacy's tier setting, kept in THIS module
rather than added to clozn/server/routes/health.py (which already hosts the analogous /capture/tier
endpoint) -- health.py is a hand-wired file several other route families also touch; this feature owns
none of its lines.

`?include_content=false` is a READ-time redaction, independent of the tier the receipt was stored with:
a caller must not gain access to content that was stored (e.g. under a "full" tier) just because they ask
without the flag -- and conversely must not be MISTAKEN into thinking `include_content=true` restores
content genuinely never stored (a "hashes_only"/"off" receipt has nothing to reveal either way).
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

CLOZN_ROUTE_AUTOLOAD = True

_SUFFIX = "/context-receipt"
_SEGMENT_METADATA_KEEP = ("segment_id", "source_type", "original_order", "content_hash", "reason",
                          "included")


def _wants_content(h) -> bool:
    query = parse_qs(urlparse(h.path).query)
    raw = (query.get("include_content") or ["true"])[0]
    return raw.strip().lower() not in ("false", "0", "no")


def _redact_for_read(receipt: dict, shape: str) -> dict:
    """Trim `receipt` down to hashes/metadata only, for a caller that passed include_content=false --
    independent of (and possibly stricter than) whatever privacy tier the document was stored with."""
    out = dict(receipt)
    survived = out.get("survived")
    if isinstance(survived, dict) and ("final_prompt" in survived or "assembled_messages" in survived):
        survived = dict(survived)
        survived.pop("final_prompt", None)
        survived.pop("assembled_messages", None)
        survived["content_withheld_by_request"] = "include_content=false"
        out["survived"] = survived

    if shape == "legacy":
        delivered = out.get("delivered")
        if isinstance(delivered, dict) and "messages" in delivered:
            delivered = dict(delivered)
            delivered.pop("messages", None)
            delivered["content_withheld_by_request"] = "include_content=false"
            out["delivered"] = delivered
        return out

    for key in ("delivered", "assembled"):
        segments = out.get(key)
        if not isinstance(segments, list):
            continue
        trimmed = []
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            keep = {k: seg[k] for k in _SEGMENT_METADATA_KEEP if k in seg}
            keep["redaction_state"] = "hash_only"
            trimmed.append(keep)
        out[key] = trimmed
    return out


def try_get(h, p):
    if p.startswith("/runs/") and p.endswith(_SUFFIX):
        import clozn.runs.store as runlog
        from clozn.runs.context_receipt import read_receipt

        rid = p[len("/runs/"):-len(_SUFFIX)]
        run = runlog.get_run(rid)
        if not run:
            h._json(404, {"error": "run not found"})
            return True

        view = read_receipt(run)
        if view["shape"] == "absent":
            h._json(200, {"run_id": rid, "shape": "absent", "context_receipt": {},
                          "note": "no context receipt was recorded for this run"})
            return True

        receipt = view["receipt"]
        if not _wants_content(h):
            receipt = _redact_for_read(receipt, view["shape"])
        h._json(200, {"run_id": rid, "shape": view["shape"], "context_receipt": receipt})
        return True

    if p == "/receipt-privacy":
        from clozn.runs import receipt_privacy
        h._json(200, {"tier": receipt_privacy.tier(), "tiers": list(receipt_privacy.TIERS)})
        return True

    return False


def try_post(h, p, body):
    if p == "/receipt-privacy":
        from clozn.runs import receipt_privacy
        name = str((body or {}).get("tier", "")).strip().lower()
        if name not in receipt_privacy.TIERS:
            h._json(400, {"error": f"unknown tier (want one of {list(receipt_privacy.TIERS)})"})
            return True
        if not receipt_privacy.set_tier(name):
            h._json(200, {"ok": False, "reason": "could not persist the tier setting"})
            return True
        h._json(200, {"ok": True, "tier": name})
        return True
    return False
