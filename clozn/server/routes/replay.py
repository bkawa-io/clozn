"""Re-running a past run under changed state: POST /runs/<id>/replay (F1: apply changes_applied and
regenerate a child run). Needs a chat-capable substrate since it regenerates, resolved through the run's
OWN `model` field (never a client-supplied one) via clozn.server.model_routing.select_control_model_for_run
-- a managed gateway that cannot resolve a ready worker for that model refuses with a typed
`clozn.model-routing.v1` error rather than the old unconditional 503; legacy one-worker serving is
unaffected. -> clozn.replay.

(M3's POST /runs/<id>/counterfactual -- one named-dial re-gen against a `behavior_overrides:
{dial_name: value}` body -- was retired with the rest of named-dial personalization; clozn.replay.
counterfactual, the module it called into, is gone.)
"""


def try_post(h, p, body):
    if p.startswith("/runs/") and p.endswith("/replay"):   # F1: re-run a past run under changed state -> a child run
        rid = p[len("/runs/"):-len("/replay")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.server.model_routing import select_control_model_for_run
        selection = select_control_model_for_run(h, run.get("model"), route="/runs/<id>/replay")
        if selection is None:
            return True   # typed clozn.model-routing.v1 refusal already written
        sub = selection.sub
        if not (sub and getattr(sub, "chat", None)):   # replay regenerates through the product model
            h._json(503, {"error": "replay requires a ready product model worker"})
            return True
        changes = body.get("changes_applied", body.get("changes")) or {}
        try:
            from clozn import replay
            child = replay.replay(run, changes, sub)
        except Exception as e:
            h._json(500, {"error": f"replay failed: {type(e).__name__}: {e}"})
            return True
        if child is None:
            h._json(500, {"error": "replay failed"})
            return True
        h._json(200, child)
        return True
    return False
