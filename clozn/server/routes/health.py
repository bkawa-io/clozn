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
    return False
