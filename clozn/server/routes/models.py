"""Local GGUF inventory: GET /models/local.

Returns clozn.models.inventory.inventory() -- path, filename, size_bytes, quant (best-effort, from the
filename), and sha256 (only when already cached; never computed inline -- see that module's docstring for
why a web request must not hash a multi-GB file). Before this route existed, no server route listed the
GGUFs on disk at all; the Model surface named the CLI (`clozn models`) instead of drawing a fake list
(see that file's prior "declared skeleton" note).

LISTING ONLY. There is deliberately no load/switch/pull route here -- which model is actively being served
is a bigger decision (a running engine process, VRAM, ...) that this route does not make; it only reports
what's sitting on disk.

clozn.models.inventory (not clozn.cli.commands.models) is the shared home for the scan itself specifically
so this route never has to import clozn.cli -- see that module's docstring for the layering rationale.
"""
from clozn.models import inventory as _inventory


def try_get(h, p):
    if p == "/models/local":
        try:
            models = _inventory.inventory()
        except Exception as e:
            h._json(502, {"error": f"model inventory scan failed: {type(e).__name__}: {e}"})
            return True
        h._json(200, {"models": models})
        return True
    return False
