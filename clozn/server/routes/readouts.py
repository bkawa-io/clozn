"""Concept and activation readouts from an explicitly selected model worker."""

# numpy is used exactly once here (the harvest norms below) and is NOT installed by clozn: pyproject
# declares `dependencies = []` deliberately, so anything beyond the stdlib is imported lazily inside the
# function that needs it. app.py imports this module at startup, so a module-scope `import numpy` made
# the whole gateway unimportable on a fresh `pip install clozn`. Kept inside the existing try so a
# missing numpy degrades to this route's ordinary 502, not a dead gateway.


def try_post(h, p, body):
    if p not in {"/engine/harvest", "/engine/layers", "/engine/concepts"}:
        return False

    from clozn.server.model_routing import select_control_model_for_request
    selection = select_control_model_for_request(
        h,
        body.get("model"),
        field_present="model" in body,
        route=p,
    )
    if selection is None:
        return True
    engine = selection.engine
    sub = selection.sub

    if p == "/engine/harvest":
        try:
            import numpy as np
            hv = engine.harvest(str(body.get("text", ""))[:300])
            norms = np.linalg.norm(hv.activations, axis=1)
            h._json(200, {"tokens": hv.tokens, "layer": int(hv.layer), "n_embd": hv.n_embd,
                         "norms": [round(float(x), 3) for x in norms]})
        except Exception as e:
            h._json(502, {"error": f"engine: {e}"})
        return True
    if p == "/engine/layers":
        try:
            h._json(200, engine.harvest_layers(str(body.get("text", ""))[:300]))
        except Exception as e:
            h._json(502, {"error": f"engine-layers: {e}"})
        return True
    if p == "/engine/concepts":
        try:
            if not (sub and getattr(sub, "brain", None)):
                h._json(409, {"error": "named SAE concepts are available in `clozn lab qwen`; "
                                       "the product runtime exposes raw engine readouts"})
                return True
            h._json(200, sub.brain.concepts_from_engine(
                str(body.get("text", ""))[:300], engine, int(body.get("layer", 15))))
        except Exception as e:
            h._json(502, {"error": f"engine-qwen: {e}"})
        return True
    return True
