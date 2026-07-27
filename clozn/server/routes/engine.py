"""Engine-backed chat + steering + model routing, and the studio chat surface that logs a run
(POST /say for the HF/qwen memory model). `/engine/*` here covers WRITE/generation calls (as opposed to
the pure readouts in routes/readouts.py): observe (edit a residual, watch the prediction move), deprecated
tone-dial compatibility aliases, and native GGUF chat with prompt-card memory. `/say` dispatches to
whichever substrate is active (SUB.handle) and additionally logs the run -- unlike the fully generic
SUB.handle(path, body) fallback (still in clozn.server.app's do_POST), this shapes a specific response AND
captures a trace/memory record for the Run Inspector. Mechanical extraction of the matching branches out
of do_POST; behavior unchanged. -> engine chat / substrate calls / model routing.

GET/POST /sampling/mode (REPRODUCE_AND_PROVE_PLAN S5): the interactive-chat sampling settings -- on/off +
temperature/top_p/top_k/repeat_penalty -- read by EngineSubstrate.chat/chat_stream on every turn. Mirrors
/timetravel/mode's shape (GET the live config; POST any subset of keys, get the resolved config + whether
anything changed back). Turning this off (or leaving it on -- either way) never touches the receipt/
replay/forced-scoring stack: those always pass sample=False ("greedy": True), which short-circuits before
this setting is even read (see EngineSubstrate.chat / _resolve_sampling).
"""
import time

from clozn.server import app as ctx


def try_get(h, p):
    if p == "/sampling/mode":   # S5: the live interactive-chat sampling settings (never mutates)
        h._json(200, ctx._sampling_settings())
        return True
    return False


def try_post(h, p, body):
    if p == "/cancel":
        if ctx.ENGINE is None:
            h._json(503, {"error": "no engine connected"})
            return True
        # `req` is the pre-existing contract: the worker's OWN id, forwarded to ENGINE.cancel() verbatim
        # -- untouched below, so every existing caller of that shape keeps working byte-for-byte.
        # `req_id` is new: the GATEWAY's own id (request_context.new_request_id(), e.g. from a
        # RequestContext a caller learned some other way), which this gateway can resolve to the
        # worker's own req itself -- the correlation the backlog calls out ("correlate req_ ids with the
        # worker's req"). Kept as a separate key rather than sniffing a "req_" prefix on `req` itself,
        # since `req` already accepts arbitrary opaque worker-minted ids and must not change meaning.
        engine_req = str(body.get("req") or "")
        req_id = str(body.get("req_id") or "")
        if req_id and not engine_req:
            sub = ctx.active_sub(h)
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
            result = ctx.ENGINE.cancel(engine_req)
        except Exception as e:
            h._json(502, {"error": f"engine cancel failed: {e}"})
            return True
        h._json(200, result)
        return True
    if p == "/sampling/mode":   # S5: adjust/toggle interactive-chat sampling (on/off + the 4 params)
        import clozn.memory.mode as memory_mode
        changed = False
        if "sampling" in body:
            memory_mode.set_setting("sampling", bool(body.get("sampling")))
            changed = True
        for key in ("sample_temperature", "sample_top_p", "sample_repeat_penalty"):
            if key in body:
                try:
                    memory_mode.set_setting(key, float(body[key]))
                    changed = True
                except (TypeError, ValueError):
                    h._json(400, {"error": f"{key} must be a number"})
                    return True
        if "sample_top_k" in body:
            try:
                memory_mode.set_setting("sample_top_k", int(body["sample_top_k"]))
                changed = True
            except (TypeError, ValueError):
                h._json(400, {"error": "sample_top_k must be an integer"})
                return True
        out = ctx._sampling_settings()
        out["changed"] = changed
        h._json(200, out)
        return True
    if p == "/engine/observe":   # WRITE a scaled residual back at one token, OBSERVE how the prediction moves
        try:
            pos = int(body.get("position", 0))
            scale = float(body.get("scale", 4.0))

            def tf(a):
                a = a.copy()
                if 0 <= pos < a.shape[0]:
                    a[pos] = a[pos] * scale
                return a

            hv, obs = ctx.ENGINE.edit_and_observe(str(body.get("text", ""))[:300], transform=tf, positions=[pos])
            h._json(200, {"summary": obs.summary(), "shifted": obs.shifted(),
                         "moved_l2": obs.moved_l2, "baseline_top": obs.baseline_top,
                         "edited_top": obs.edited_top, "tokens": hv.tokens,
                         "position": pos, "scale": scale})
        except Exception as e:
            h._json(502, {"error": f"engine: {e}"})
        return True
    if p == "/engine/steer/axes":   # deprecated alias; /steer/axes is the product contract
        from clozn.behavior.steering.axes import AXES
        es = ctx._engine_steer()
        sub = ctx.active_sub(h)
        # In the product process EngineSubstrate.steer IS ctx._engine_steer(). Delegate so the legacy
        # URL cannot drift from live values, calibration bounds, or custom/library axes. A lab process
        # may expose a different active substrate while an engine client exists; keep the old metadata
        # fallback there rather than pretending those are the same state owner.
        if sub is not None and getattr(sub, "steer", None) is es and hasattr(sub, "_steer"):
            out = dict(sub._steer("/steer/axes", {}) or {})
        else:
            out = {"axes": [{"name": k, "poles": AXES[k]["poles"]} for k in AXES],
                   "ready": bool(es and es.ready)}
        out.update({"engine": bool(ctx.ENGINE), "deprecated": True, "canonical": "/steer/axes"})
        h._json(200, out)
        return True
    if p == "/engine/steer/check":   # legacy body uses axis/max_tokens; canonical path is /steer/check
        es = ctx._engine_steer()
        if es is None:
            h._json(502, {"error": "model worker unavailable (CLOZN_ENGINE_PORT)"})
            return True
        try:
            prompt = str(body.get("prompt", "Tell me about the city at night."))[:300]
            axis, val = str(body.get("axis", "warm")), float(body.get("value", 1.0))
            mx = int(body.get("max_tokens", 60))
            base = es.generate(prompt, strength={}, max_new=mx)            # no dial = the baseline
            stee = es.generate(prompt, strength={axis: val}, max_new=mx)
            h._json(200, {"prompt": prompt, "axis": axis, "value": val,
                         "baseline": base.strip(), "steered": stee.strip(),
                         "deprecated": True, "canonical": "/steer/check"})
        except Exception as e:
            h._json(502, {"error": f"engine-steer: {e}"})
        return True
    if p == "/engine/chat":   # Native chat alias: GGUF generation with prompt-card memory
        if ctx.ENGINE is None:
            h._json(502, {"error": "no engine configured"})
            return True
        from clozn.runs.think_tags import sanitize_messages
        msgs = sanitize_messages(body.get("messages", []))
        t0 = time.time()
        memout = {}
        try:
            mx = int(body.get("max_tokens", 220))
            kw = {}
            mem = getattr(ctx.active_sub(h), "memory", None) if ctx.active_sub(h) else None
            # Product memory is always the legible card block. Soft-prefix training/application lives
            # only in the lab, so this route cannot import Torch or inject a stale .pt artifact.
            ms = float(getattr(mem, "memory_strength", 1.0)) if mem is not None else 1.0
            from clozn.server.generation_gateway import request_memory_scope
            memory_scope = request_memory_scope(h)
            decision = ctx._prompt_block_for(
                mem, ctx._last_user(msgs), strength=ms, request_scope=memory_scope)
            block, applied, gate = decision
            ctx._capture_prompt_decision(memout, decision)
            if applied:
                baseline_tokens = ctx._baseline_prompt_tokens(ctx.ENGINE, msgs)
                if baseline_tokens is not None:
                    memout["baseline_prompt_tokens"] = baseline_tokens
            assembled = ctx._inject_block(msgs, block)
            memout.update(mode="prompt", applied=applied, gate=gate, strength=ms,
                          prompt_block=block, assembled_messages=assembled)
            template_usage = {}
            prompt = ctx._engine_tmpl(ctx.ENGINE, assembled, usage_out=template_usage)
            if isinstance(template_usage.get("prompt_tokens"), int):
                memout["actual_prompt_tokens"] = template_usage["prompt_tokens"]
            # backlog #5: record the EXACT rendered chat-template string the model saw (both memory
            # modes render one). _log_run reads memout["final_prompt"] -> the run record's final_prompt.
            memout["final_prompt"] = prompt
            # TONE: live dial values if a substrate is up, else the saved values from disk
            st = getattr(getattr(ctx.active_sub(h), "steer", None), "strength", None) if ctx.active_sub(h) else None
            if not st:
                st = ctx._disk_dials()
            if st and any(st.values()):
                es = ctx._engine_steer()
                sv = es.steer_vector(st) if es is not None else None
                if sv:
                    kw["steer_vec"] = sv
                    kw["steer"] = {"coef": 1.0, "layer": es.layer}
            ctx._apply_anchored_memory(
                kw, memout, ctx._last_user(msgs), request_scope=memory_scope)
            # Generate + capture a per-token trace alongside (B3). Reply is byte-identical to the
            # plain complete(); the trace feeds the Run Inspector timeline. steps=[] (diffusion, or a
            # stream hiccup) -> runlog stores a clean empty trace.
            usage = {}
            reply_raw, steps, finish, _divinfo = ctx._engine_complete_traced(
                ctx.ENGINE, prompt, mx, kw, usage_out=usage)
            if isinstance(usage.get("prompt_tokens"), int):
                memout["actual_prompt_tokens"] = usage["prompt_tokens"]
            from clozn.runs.think_tags import prompt_opens_think, sanitize_reply
            reply = sanitize_reply(
                reply_raw, implicit_open=prompt_opens_think(prompt)
            ).public_text.strip()
            # Pass the raw step list; runlog.record normalizes it -> {tokens, confidence, alternatives}.
            h._log_run("engine_chat", msgs, reply_raw, "clozn-qwen (engine)", t0, trace=steps,
                       mem_out=memout, finish_reason=finish)
            # "memory" == did the prompt-card block actually ride this reply.
            h._json(200, {"reply": reply,
                         "memory": bool(memout.get("applied")),
                         "tone": bool(kw.get("steer_vec")), "via": "engine (GGUF)"})
        except Exception as e:
            h._log_run("engine_chat", msgs, "", "clozn-qwen (engine)", t0, error=str(e), mem_out=memout)
            h._json(502, {"error": f"engine-chat: {e}"})
        return True
    if p == "/say":   # studio chat (qwen memory model) -> capture it as a run
        if not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "handle", None)):
            h._json(409, {"error": f"'{p}' isn't served by the '{ctx.active_subname(h)}' substrate"})
            return True
        msg = str(body.get("message", ""))
        t0 = time.time()
        # HF studio chat: capture a per-token trace (B3) + the per-turn memory record. We hand
        # SUB.handle collectors via body["_trace_out"] / body["_mem_out"] (server-side only, never
        # echoed); QwenSubstrate's /say fills them through say()/_say_prompt -> _generate's
        # pass-through recorder. Reply text is byte-identical with or without them.
        trace_steps = []
        body["_trace_out"] = trace_steps
        memout = {}
        body["_mem_out"] = memout
        try:
            r = ctx.active_sub(h).handle(p, body)
        except Exception as e:
            h._log_run("studio_chat", [{"role": "user", "content": msg}], "",
                      "clozn-qwen", t0, error=str(e), mem_out=memout)
            h._json(500, {"error": f"{type(e).__name__}: {e}"})
            return True
        if r is None:
            h._json(409, {"error": f"'{p}' isn't served by the '{ctx.active_subname(h)}' substrate",
                         "need": "qwen", "active": ctx.active_subname(h)})
            return True
        # runlog.record normalizes the raw step list -> {tokens, confidence, alternatives}; a diffusion
        # substrate (or any path that filled nothing) yields [] -> a clean empty trace.
        raw_reply = str(r.get("reply", ""))
        h._log_run("studio_chat", [{"role": "user", "content": msg}],
                  raw_reply, "clozn-qwen", t0, trace=trace_steps, mem_out=memout)
        from clozn.runs.think_tags import prompt_opens_think, sanitize_reply
        r = dict(r)
        r["reply"] = sanitize_reply(
            raw_reply, implicit_open=prompt_opens_think(memout.get("final_prompt"))
        ).public_text
        h._json(200, r)
        return True
    return False
