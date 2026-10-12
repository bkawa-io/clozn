"""Engine-backed generation, inspection, cancellation, and sampling routes.

``/engine/*`` contains direct GGUF worker actions. OpenAI- and Ollama-compatible generation lives in
their dedicated route modules; all paths share the request-selected engine substrate.

GET/POST /sampling/mode (REPRODUCE_AND_PROVE_PLAN S5): the interactive-chat sampling settings -- on/off +
temperature/top_p/top_k/repeat_penalty -- read by EngineSubstrate.chat/chat_stream on every turn. Mirrors
/timetravel/mode's shape (GET the live config; POST any subset of keys, get the resolved config + whether
anything changed back). Turning this off (or leaving it on -- either way) never touches the receipt/
replay/forced-scoring stack: those always pass sample=False ("greedy": True), which short-circuits before
this setting is even read (see EngineSubstrate.chat / _resolve_sampling).
"""
from clozn.server import app as ctx


def try_get(h, p):
    if p == "/sampling/mode":   # S5: the live interactive-chat sampling settings (never mutates)
        h._json(200, ctx._sampling_settings())
        return True
    return False


def try_post(h, p, body):
    if p == "/cancel":
        from clozn.server.model_routing import select_control_model_for_request
        selection = select_control_model_for_request(
            h,
            body.get("model"),
            field_present="model" in body,
            route="/cancel",
        )
        if selection is None:
            return True
        engine = selection.engine
        sub = selection.sub
        if engine is None or not callable(getattr(engine, "cancel", None)):
            h._json(503, {"error": "selected model worker is unavailable"})
            return True
        # ``req`` is the worker's opaque request ID. ``req_id`` is the gateway request ID and is
        # resolved against the selected worker's live RequestContext.
        engine_req = str(body.get("req") or "")
        req_id = str(body.get("req_id") or "")
        if req_id and not engine_req:
            # Only one generation is in flight at a time (POST_GATE serializes it; /cancel itself is
            # gate-exempt precisely so it can reach that in-flight call concurrently -- see
            # app._GATE_EXEMPT_POSTS), so the substrate's live self._request IS "the" request req_id
            # could be naming. A stale/unknown req_id (already finished, or from a since-replaced
            # RequestContext) simply doesn't match -- nothing to resolve, nothing to cancel.
            current = getattr(sub, "_request", None) if sub is not None else None
            if current is not None and current.request_id == req_id:
                # Local stop signal fires unconditionally on a match, even if the worker hasn't stamped
                # its own req yet (generation just started, no frame parsed): chat_stream's read loop
                # checks is_cancelled() between frames independent of the worker-side /cancel below, so
                # this alone still halts the stream promptly.
                current.cancel()
                engine_req = current.engine_req or ""
            if not engine_req:
                # Nothing to hand the worker's /cancel (unmatched req_id, or no worker req yet) -- report
                # honestly instead of forwarding an empty id, which the worker 400s on.
                h._json(200, {"cancelled": False, "req": req_id})
                return True
        try:
            result = engine.cancel(engine_req)
        except Exception as e:
            h._json(502, {"error": f"engine cancel failed: {e}"})
            return True
        h._json(200, result)
        return True
    if p == "/sampling/mode":   # S5: adjust/toggle interactive-chat sampling (on/off + the 4 params)
        import clozn.settings as settings
        changed = False
        if "sampling" in body:
            settings.set_setting("sampling", bool(body.get("sampling")))
            changed = True
        for key in ("sample_temperature", "sample_top_p", "sample_repeat_penalty"):
            if key in body:
                try:
                    settings.set_setting(key, float(body[key]))
                    changed = True
                except (TypeError, ValueError):
                    h._json(400, {"error": f"{key} must be a number"})
                    return True
        if "sample_top_k" in body:
            try:
                settings.set_setting("sample_top_k", int(body["sample_top_k"]))
                changed = True
            except (TypeError, ValueError):
                h._json(400, {"error": "sample_top_k must be an integer"})
                return True
        out = ctx._sampling_settings()
        out["changed"] = changed
        h._json(200, out)
        return True
    if p == "/engine/observe":   # WRITE a scaled residual back at one token, OBSERVE how the prediction moves
        from clozn.server.model_routing import select_control_model_for_request
        selection = select_control_model_for_request(
            h,
            body.get("model"),
            field_present="model" in body,
            route="/engine/observe",
        )
        if selection is None:
            return True
        engine = selection.engine
        try:
            pos = int(body.get("position", 0))
            scale = float(body.get("scale", 4.0))

            def tf(a):
                a = a.copy()
                if 0 <= pos < a.shape[0]:
                    a[pos] = a[pos] * scale
                return a

            hv, obs = engine.edit_and_observe(
                str(body.get("text", ""))[:300], transform=tf, positions=[pos])
            h._json(200, {"summary": obs.summary(), "shifted": obs.shifted(),
                         "moved_l2": obs.moved_l2, "baseline_top": obs.baseline_top,
                         "edited_top": obs.edited_top, "tokens": hv.tokens,
                         "position": pos, "scale": scale})
        except Exception as e:
            h._json(502, {"error": f"engine: {e}"})
        return True
    return False
