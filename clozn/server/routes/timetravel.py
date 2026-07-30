"""Time-travel debugger surface: the snapshot gate + ring config + store stats
(GET/POST /timetravel/mode, POST /timetravel/stats), and rewinding & branching from a turn
(POST /runs/<id>/branch) into a child run. The snapshot ring holds KV state in CPU RAM, so it is behind
ONE persisted setting (`timetravel_snapshots`, DEFAULT OFF -- the RAM rule); branch RECORDING does NOT
depend on that gate, only holding live KV for the (future) re-prefill fast path does. Mechanical
extraction of clozn.server.app's old `_timetravel` handler method + the matching do_GET/do_POST branches;
behavior unchanged. -> clozn.replay (timetravel.py).
"""
from clozn.server import app as ctx


def try_get(h, p):
    if p == "/timetravel/mode":      # #6: is per-turn KV snapshotting on? + ring config + store stats
        import clozn.replay.timetravel as timetravel
        out = {"enabled": timetravel.enabled(), **timetravel.get_config()}
        store = ctx._snap_store()
        if store is not None:
            out["store"] = store.stats()
        h._json(200, out)
        return True
    return False


def try_post(h, p, body):
    if p.startswith("/timetravel/"):   # #6: the time-travel snapshot gate + ring config (default OFF)
        import clozn.replay.timetravel as timetravel
        if p == "/timetravel/mode":       # read or set the on/off gate + ring config
            changed = False
            if "enabled" in body:
                timetravel.set_enabled(bool(body.get("enabled")))
                changed = True
            if "cap" in body or "budget_mb" in body:
                timetravel.set_config(cap=body.get("cap"), budget_mb=body.get("budget_mb"))
                changed = True
                cfg = timetravel.get_config()          # apply the (clamped) new ceilings to the LIVE store
                if ctx._snap_store() is not None:
                    ctx._snap_store().reconfigure(cap=cfg["cap"], budget_mb=cfg["budget_mb"])
            out = {"enabled": timetravel.enabled(), **timetravel.get_config()}
            store = ctx._snap_store()
            if store is not None:
                out["store"] = store.stats()
            out["changed"] = changed
            h._json(200, out)
            return True
        if p == "/timetravel/stats":      # just the store's honest memory receipt
            store = ctx._snap_store()
            h._json(200, {"enabled": timetravel.enabled(),
                         **(store.stats() if store is not None else {})})
            return True
        h._json(404, {"error": f"POST {p}"})
        return True
    if p.startswith("/runs/") and p.endswith("/branch"):   # #6: rewind & branch from a turn -> a child run
        rid = p[len("/runs/"):-len("/branch")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.server.model_routing import select_control_model_for_run
        selection = select_control_model_for_run(h, run.get("model"), route="/runs/<id>/branch")
        if selection is None:
            return True   # typed clozn.model-routing.v1 refusal already written
        sub = selection.sub
        if not (sub and getattr(sub, "chat", None)):   # a branch regenerates through the product model
            h._json(503, {"error": "branch requires a ready product model worker"})
            return True
        if "turn" not in body:
            h._json(400, {"error": "need a branch turn"})
            return True
        try:
            turn = int(body.get("turn"))
        except (TypeError, ValueError):
            h._json(400, {"error": "turn must be an integer"})
            return True
        alt = body.get("alt_user")
        # greedy by default (the receipt path: a branch's future is attributable, not sampling dice).
        sample = bool(body.get("sample", False))
        try:
            import clozn.replay.timetravel as timetravel
            child = timetravel.branch(run, turn, sub, alt_user=alt, sample=sample,
                                      store=ctx._snap_store())
        except Exception as e:
            h._json(500, {"error": f"branch failed: {type(e).__name__}: {e}"})
            return True
        if child is None:                          # None == bad turn index or a generation failure
            h._json(400, {"error": "branch failed (turn out of range, or generation error)"})
            return True
        h._json(200, child)
        return True
    return False
