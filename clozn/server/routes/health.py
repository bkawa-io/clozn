"""Liveness, readiness, and product runtime state."""

from clozn.server import app as ctx


def _managed_status():
    router = getattr(ctx, "MODEL_ROUTER", None)
    if router is None or not hasattr(router, "runtime_status"):
        raise RuntimeError("model router unavailable")
    return router.runtime_status()


def try_get(h, p):
    if p == "/healthz":
        h._json(200, {"status": "ok", "service": "clozn"})
        return True
    if p == "/readyz":
        try:
            managed = _managed_status()
        except Exception as error:
            h._json(503, {
                "status": "not_ready",
                "service": "clozn",
                "reason": f"managed routing unavailable: {error}",
            })
            return True
        if managed["resident_count"] < 1:
            h._json(503, {
                "status": "not_ready",
                "service": "clozn",
                "reason": "no configured model worker is ready",
                "models": managed,
            })
            return True
        queue = ctx.POST_GATE.snapshot() if getattr(ctx, "POST_GATE", None) else None
        from clozn import protocol
        h._json(200, {
            "status": "ok",
            "service": "clozn",
            "active": "engine",
            "protocol_version": protocol.PROTOCOL_VERSION,
            "model": managed["default_model_id"],
            "models": managed,
            "queue": queue,
        })
        return True
    if p == "/runtime/models":
        try:
            managed = _managed_status()
        except Exception as error:
            h._json(503, {"error": f"managed routing unavailable: {error}"})
            return True
        h._json(200, managed)
        return True
    if p == "/substrate":
        h._json(200, {"active": "engine", "available": ["engine"]})
        return True
    if p == "/engine/health":
        try:
            managed = _managed_status()
        except Exception as error:
            h._json(502, {"error": f"managed routing unavailable: {error}"})
            return True
        # This endpoint is the public router projection. Private worker ports and raw worker health stay
        # private; the explicit discriminator keeps the Studio contract unambiguous even when the router
        # currently contains a single model.
        h._json(200, {"engine": {"managed": True, **managed}})
        return True
    if p == "/state":
        h._json(200, {"substrate": ctx.active_subname(h)})
        return True
    if p == "/capture/tier":
        from clozn.runs import capture_mode
        h._json(200, {"tier": capture_mode.tier(), "tiers": list(capture_mode.TIERS)})
        return True
    return False


def try_post(h, p, body):
    if p == "/capture/tier":
        from clozn.runs import capture_mode
        name = str(body.get("tier", "")).strip().lower()
        if name not in capture_mode.TIERS:
            h._json(400, {"error": f"unknown tier (want one of {list(capture_mode.TIERS)})"})
            return True
        if not capture_mode.set_tier(name):
            h._json(200, {"ok": False, "reason": "could not persist the tier setting"})
            return True
        h._json(200, {"ok": True, "tier": name})
        return True
    return False
