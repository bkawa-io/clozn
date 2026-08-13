"""Liveness, readiness, and product runtime state."""

from clozn.server import app as ctx


def _managed_status():
    router = getattr(ctx, "MODEL_ROUTER", None)
    if router is None or not hasattr(router, "runtime_status"):
        return None
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
        if managed is not None:
            if managed["resident_count"] < 1:
                h._json(503, {
                    "status": "not_ready",
                    "service": "clozn",
                    "reason": "no configured model worker is ready",
                    "models": managed,
                })
                return True
            queue = (
                ctx.POST_GATE.snapshot()
                if getattr(ctx, "POST_GATE", None) else None
            )
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
        if ctx.active_sub(h) is None or ctx.ENGINE is None:
            h._json(503, {"status": "not_ready", "service": "clozn", "reason": "model worker unavailable"})
            return True
        try:
            worker = ctx.ENGINE.health()
        except Exception as e:
            h._json(503, {"status": "not_ready", "service": "clozn", "reason": str(e)})
            return True
        if worker.get("status") != "ok":
            h._json(503, {"status": "not_ready", "service": "clozn", "worker": worker})
            return True
        queue = ctx.POST_GATE.snapshot() if getattr(ctx, "POST_GATE", None) else None
        from clozn import protocol
        h._json(200, {"status": "ok", "service": "clozn", "active": "engine",
                      "protocol_version": protocol.PROTOCOL_VERSION,   # gateway <-> worker wire contract
                      "capabilities": worker.get("capabilities", {}),  # the live worker's negotiated flags
                      "model": worker.get("model"), "mode": worker.get("mode"), "worker": worker,
                      "queue": queue})
        return True
    if p == "/runtime/models":
        try:
            managed = _managed_status()
        except Exception as error:
            h._json(503, {"error": f"managed routing unavailable: {error}"})
            return True
        if managed is None:
            h._json(200, {
                "default_model_id": None,
                "preload_model_ids": [],
                "max_loaded_models": 1,
                "configured_count": 1 if ctx.active_sub(h) is not None else 0,
                "resident_count": 1 if ctx.active_sub(h) is not None else 0,
                "models": [],
                "managed": False,
            })
            return True
        h._json(200, {**managed, "managed": True})
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
        if managed is not None:
            # Private worker ports exist only in the supervisor projection.
            # Never lower them into a public runtime response.
            h._json(200, {"engine": {"managed": True, **managed}})
            return True
        try:
            info = ctx.ENGINE.health()
            # The raw C++ worker's own base URL. The gateway never serves /score & friends over HTTP
            # itself -- it calls the worker internally -- and this address was previously discoverable
            # only by walking the OS process tree (the export-bundle live-check trap: pointing
            # --engine-url at the GATEWAY port 409s). Ephemeral: changes on every gateway restart.
            base = getattr(ctx.ENGINE, "base", None)
            if isinstance(info, dict) and base:
                info = dict(info)
                info["worker_url"] = base
            h._json(200, {"engine": info})
        except Exception as e:
            h._json(502, {"error": f"engine unreachable: {e}"})
        return True
    if p == "/state":
        h._json(200, {"substrate": ctx.active_subname(h), "memory_mode": ctx._memory_mode(),
                      **(ctx.active_sub(h).state() if ctx.active_sub(h) else {})})
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
