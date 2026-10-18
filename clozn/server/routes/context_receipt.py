"""Context Receipt reads and metadata-disclosure settings.

The receipt endpoint returns only the schema-governed metadata artifact. Prompt and message content
belongs to the run resource and is never conditionally embedded in this response.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True

_SUFFIX = "/context-receipt"


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

        h._json(200, {
            "run_id": rid,
            "shape": view["shape"],
            "context_receipt": view["receipt"],
        })
        return True

    return False
