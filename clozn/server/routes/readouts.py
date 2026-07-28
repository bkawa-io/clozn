"""Concept/activation readouts straight off the C++ engine: raw per-token activation norms
(POST /engine/harvest), a per-layer activation summary (POST /engine/layers), and the brain's SAE
concepts read from the engine's Qwen GGUF (POST /engine/concepts). Mechanical extraction of the matching
`if p == "/engine/..."` branches out of clozn.server.app's do_POST; behavior unchanged. -> clozn.readouts
(through the one raw ENGINE client on clozn.server.app).
"""
from clozn.server import app as ctx

# numpy is used exactly once here (the harvest norms below) and is NOT installed by clozn: pyproject
# declares `dependencies = []` deliberately, so anything beyond the stdlib is imported lazily inside the
# function that needs it. app.py imports this module at startup, so a module-scope `import numpy` made
# the whole gateway unimportable on a fresh `pip install clozn`. Kept inside the existing try so a
# missing numpy degrades to this route's ordinary 502, not a dead gateway.


def try_post(h, p, body):
    if p == "/engine/harvest":   # READ the real C++ runtime's activations (any substrate; the engine is separate)
        try:
            import numpy as np
            hv = ctx.ENGINE.harvest(str(body.get("text", ""))[:300])
            norms = np.linalg.norm(hv.activations, axis=1)
            h._json(200, {"tokens": hv.tokens, "layer": int(hv.layer), "n_embd": hv.n_embd,
                         "norms": [round(float(x), 3) for x in norms]})
        except Exception as e:
            h._json(502, {"error": f"engine: {e}"})
        return True
    if p == "/engine/layers":    # per-layer activation SUMMARY (depth x position norms) from the C++ engine
        try:
            h._json(200, ctx.ENGINE.harvest_layers(str(body.get("text", ""))[:300]))
        except Exception as e:
            h._json(502, {"error": f"engine-layers: {e}"})
        return True
    if p == "/engine/concepts":   # named SAE concepts remain a PyTorch lab visualization
        try:
            if not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "brain", None)):
                h._json(409, {"error": "named SAE concepts are available in `clozn lab qwen`; "
                                       "the product runtime exposes raw engine readouts"})
                return True
            h._json(200, ctx.active_sub(h).brain.concepts_from_engine(
                str(body.get("text", ""))[:300], ctx.ENGINE, int(body.get("layer", 15))))
        except Exception as e:
            h._json(502, {"error": f"engine-qwen: {e}"})
        return True
    return False
