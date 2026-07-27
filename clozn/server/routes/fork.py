"""Fork-at-token surface: POST /runs/<id>/fork -- regenerate a past run's reply from token position t
with a chosen alternative piece forced at t, producing a CHILD run ("click an almost-said token to
fork reality"). The TOKEN-granularity sibling of timetravel's /runs/<id>/branch (turn granularity);
same load-the-run / regenerate / return-the-child-record flow. -> clozn.replay.fork.

Body: {"position": <int, index into the reply's trace tokens>, "token": "<piece>" | "token_id": <int>}.
Response: the child run record + prefix_kept / forked_from_piece / retokenized / note (see fork.py).

Registered in clozn/server/app.py: imported as `_fork_routes` and placed in `_POST_ROUTES`. Live
surface: the Observatory's fork-a-ghost flow (studio-frontend/src/data/api.ts -> Observatory.tsx). The previous shell's
Replay panel (studio/heavn) also called it; that app is no longer served from "/".
"""
from clozn.server import app as ctx


def try_post(h, p, body):
    if p.startswith("/runs/") and p.endswith("/fork"):   # fork the reply at a token -> a child run
        rid = p[len("/runs/"):-len("/fork")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        # A fork teacher-forces a raw prompt prefix through the private worker seam.
        if not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "engine", None)):
            h._json(503, {"error": "fork requires a ready product model worker"})
            return True
        if "position" not in body:
            h._json(400, {"error": "need a fork position (an index into the reply's trace tokens)"})
            return True
        try:
            position = int(body.get("position"))
        except (TypeError, ValueError):
            h._json(400, {"error": "position must be an integer"})
            return True
        import clozn.replay.fork as fork_mod
        try:
            child = fork_mod.fork(run, ctx.active_sub(h), position,
                                  token=body.get("token"), token_id=body.get("token_id"))
        except ValueError as e:                          # validation: out-of-range / no trace / bad token
            h._json(400, {"error": str(e)})
            return True
        except Exception as e:
            h._json(500, {"error": f"fork failed: {type(e).__name__}: {e}"})
            return True
        if child is None:                                # generation/persistence failure (fork never raises these)
            h._json(500, {"error": "fork failed (generation error, or the run's prompt could not "
                         "be reconstructed)"})
            return True
        h._json(200, child)
        return True
    return False
