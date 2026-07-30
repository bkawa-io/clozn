"""Re-running a past run under changed state: POST /runs/<id>/replay (F1: apply changes_applied and
regenerate a child run) and POST /runs/<id>/counterfactual (M3: one dial re-gen). Both need a chat-capable
substrate since they regenerate, resolved through the run's OWN `model` field (never a client-supplied
one) via clozn.server.model_routing.select_control_model_for_run -- a managed gateway that cannot
resolve a ready worker for that model refuses with a typed `clozn.model-routing.v1` error rather than
the old unconditional 503; legacy one-worker serving is unaffected. -> clozn.replay.
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
    if p.startswith("/runs/") and p.endswith("/counterfactual"):   # M3: one counterfactual dial re-gen
        rid = p[len("/runs/"):-len("/counterfactual")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.server.model_routing import select_control_model_for_run
        selection = select_control_model_for_run(h, run.get("model"), route="/runs/<id>/counterfactual")
        if selection is None:
            return True   # typed clozn.model-routing.v1 refusal already written
        sub = selection.sub
        if not (sub and getattr(sub, "chat", None)):   # both arms regenerate through the product model
            h._json(503, {"error": "counterfactual requires a ready product model worker"})
            return True
        overrides = body.get("behavior_overrides")
        if not isinstance(overrides, dict) or not overrides:
            h._json(400, {"error": "need a behavior_overrides dict: {dial_name: value, ...}"})
            return True
        import clozn.replay.counterfactual as counterfactual
        try:
            out = counterfactual.counterfactual(run, overrides, sub)
        except Exception as e:
            h._json(500, {"error": f"counterfactual failed: {type(e).__name__}: {e}"})
            return True
        if out is None:
            h._json(500, {"error": "counterfactual failed (bad overrides, or the replay "
                         "could not be generated)"})
            return True
        h._json(200, out)
        return True
    return False
