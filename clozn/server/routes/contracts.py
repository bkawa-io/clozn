"""GET /contracts/hooks and POST /contracts/replay for versioned interventions.

GET /contracts/hooks is a pure static read: it serves clozn.receipts.hook_vocabulary.hook_vocabulary()
verbatim, no substrate/engine required.

POST /contracts/replay validates an untrusted `clozn.intervention_manifest.v1` body (vendored
validation in clozn.receipts.intervention_manifest -- this route never imports clozn-client) and
executes it against the active substrate's engine. A structurally invalid manifest is a 400; a
well-formed manifest the CURRENT engine cannot satisfy (missing capability, e.g. attn_knockout with
flash-attn on) is a typed 409 listing exactly what's missing, never a silent no-op and never a 500.

The route has no run identity, so it resolves the gateway's default model through the mandatory
router. That keeps intervention execution on the same explicit request-selection path as generation
without allowing a manifest to choose an unrelated worker.
"""

def try_get(h, p):
    if p != "/contracts/hooks":
        return False
    from clozn.receipts.hook_vocabulary import hook_vocabulary
    h._json(200, hook_vocabulary())
    return True


def try_post(h, p, body):
    if p != "/contracts/replay":
        return False
    body = body if isinstance(body, dict) else {}
    manifest = body.get("manifest")
    if not isinstance(manifest, dict):
        h._json(400, {"error": "'manifest' must be an object (a clozn.intervention_manifest.v1 document)"})
        return True

    from clozn.server.model_routing import select_for_handler
    selection = select_for_handler(
        h,
        {},
        surface="native",
        route="/contracts/replay",
    )
    if selection is None:
        return True
    sub = selection.sub
    engine = getattr(sub, "engine", None)
    if engine is None or not callable(getattr(engine, "score", None)):
        h._json(503, {"error": "intervention replay requires an engine-backed substrate"})
        return True

    try:
        health = engine.health() if callable(getattr(engine, "health", None)) else {}
    except Exception:
        health = {}
    if not isinstance(health, dict):
        health = {}

    from clozn.receipts.intervention_manifest import ManifestError, replay_manifest

    try:
        result = replay_manifest(manifest, engine, health=health)
    except ManifestError as exc:
        h._json(400, {"error": str(exc)})
        return True
    except Exception as exc:
        h._json(500, {"error": f"intervention replay failed: {type(exc).__name__}: {exc}"})
        return True

    if result.get("performed") is False:
        h._json(409, result)
        return True
    h._json(200, result)
    return True
