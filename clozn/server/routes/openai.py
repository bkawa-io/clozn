"""Public OpenAI API plus the namespaced Clozn generation event stream."""
import time

from clozn.server import app as ctx


def _api_error(h, status: int, message: str, *, param=None, kind="invalid_request_error"):
    error = {"message": message, "type": kind, "param": param, "code": status}
    h._json(status, {"error": error})


def try_get(h, p):
    if p == "/v1/models":            # OpenAI-compatible model list (so OAI clients connect)
        from clozn.server.generation_gateway import model_id
        h._json(200, {"object": "list", "data": [
            {"id": model_id(), "object": "model", "owned_by": "clozn"}]})
        return True
    return False


def try_post(h, p, body):
    if p == "/api/clozn/generate":
        from clozn.server.generation_gateway import native_completion
        native_completion(h, body)
        return True
    if p == "/v1/completions":
        from clozn.server.generation_gateway import openai_completion
        openai_completion(h, body)
        return True
    if p != "/v1/chat/completions":   # OpenAI-compatible: chat with memory prefix + tone steering applied
        return False
    if not (ctx.active_sub(h) and getattr(ctx.active_sub(h), "chat", None)):
        h._json(503, {"error": {"message": "model worker unavailable", "type": "service_unavailable"}})
        return True
    from clozn.server.generation_gateway import model_id
    selected_model = str(body.get("model") or model_id())
    msgs = body.get("messages", [])
    if not isinstance(msgs, list) or not all(
        isinstance(message, dict)
        and message.get("role") in ("system", "user", "assistant", "tool")
        and isinstance(message.get("content", ""), str)
        for message in msgs
    ):
        _api_error(h, 400, "messages must be a list of {role, content:string} objects", param="messages")
        return True
    try:
        mx = int(body.get("max_tokens", 256))
    except (TypeError, ValueError):
        _api_error(h, 400, "max_tokens must be an integer", param="max_tokens")
        return True
    if mx < 1:
        _api_error(h, 400, "max_tokens must be at least 1", param="max_tokens")
        return True
    try:
        temperature = float(body.get("temperature", 0.7))
    except (TypeError, ValueError):
        _api_error(h, 400, "temperature must be a number", param="temperature")
        return True
    # SYMMETRY (context-contamination guard): clients echo the whole conversation back, footers and all.
    # Whatever clozn appended to a past reply, it strips here -- the model must never read its own
    # receipt footers as context (it would imitate/steer on them). No-op when no footer ever rode.
    from clozn.runs.receipt_footer import strip_footers
    msgs = strip_footers(msgs)
    # PROMPT-SECTION INFLUENCE (foundation): build the ablatable-section manifest AFTER footer-stripping
    # (which can shrink an echoed assistant message's content) but BEFORE clozn_section is popped just
    # below -- building it any earlier would let a stripped footer's length change drift a section's char
    # offsets out from under the message content this run actually records. Explicit tags win; absent any,
    # the deterministic auto-chunker stands in, so every run (tagged or not) gets a manifest for the
    # (separately-owned) receipts system to later measure against. clozn_section is then popped off every
    # message so the engine/template path never sees our extension field -- a standard OpenAI client that
    # never sends the tag gets a byte-identical prompt either way; this is additive, never a behavior
    # change to the reply itself. Guarded: a chunker bug must never break a chat.
    from clozn.runs import sections as clozn_sections
    try:
        section_manifest = clozn_sections.sections_from_messages(msgs)
        if section_manifest is None:
            section_manifest = clozn_sections.auto_chunk_messages(msgs)
    except Exception:
        section_manifest = []
    msgs = [{k: v for k, v in message.items() if k != "clozn_section"} for message in msgs]
    if body.get("stream") and getattr(ctx.active_sub(h), "chat_stream", None):
        from clozn.server import sse
        from clozn.server.routes.receipt_link import receipt_enabled
        # F1 live lens: clozn_lens {layer?, topk?, every?} (or true) is a clozn extension -- absent for
        # standard OpenAI clients, so their streams stay byte-identical (same opt-in rule as clozn_trust).
        # receipt: the in-band footer as a final content chunk (the one ambient surface).
        sse.sse_chat(h, msgs, mx, selected_model, lens=body.get("clozn_lens"),
                     receipt=body.get("clozn_receipt", receipt_enabled()), sections=section_manifest)
        return True
    t0 = time.time()
    trace_steps = []                            # HF non-stream: capture a per-token trace (B3)
    memout = {}                                 # prompt mode: what memory actually rode this turn
    chat_kw = {"trace_out": trace_steps, "mem_out": memout}
    if isinstance(ctx.active_sub(h), ctx.EngineSubstrate):
        chat_kw["apply_anchored"] = True
    try:
        reply = ctx.active_sub(h).chat(msgs, mx, temperature > 0, **chat_kw)
    except Exception as exc:
        h._log_run("openai_api", msgs, "", selected_model, t0, error=str(exc), mem_out=memout,
                  sections=section_manifest)
        _api_error(h, 502, str(exc), kind="upstream_error")
        return True
    fr = ctx.active_sub(h).last_finish_reason() if hasattr(ctx.active_sub(h), "last_finish_reason") else None
    openai_fr = ctx._openai_finish_reason(fr)
    # runlog.record normalizes the raw step list -> {tokens, confidence, alternatives}.
    rid = h._log_run("openai_api", msgs, reply, selected_model, t0,
                     trace=trace_steps, mem_out=memout, finish_reason=fr,
                     finish_reason_fallback=None if fr else openai_fr, sections=section_manifest)
    resp = {"id": "chatcmpl-clozn", "object": "chat.completion",
           "created": int(time.time()), "model": selected_model,
           "choices": [{"index": 0, "finish_reason": openai_fr,
                        "message": {"role": "assistant", "content": reply}}],
           "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
    # M5 any-client run_id bridge (EXPLAIN_THIS_ANSWER_SPEC.md): surface the id two ways so a
    # companion `clozn explain <run_id>` can inspect THIS reply from any OpenAI-compatible client
    # -- an additive top-level field (spec-compliant clients ignore unknown fields) and a response
    # header (for clients that only expose headers). Clean omission when logging failed (rid is
    # None) -- never emit a literal "null"/"None".
    extra_headers = {"X-Clozn-Run-Id": rid} if rid else None
    if rid:
        resp["clozn_run_id"] = rid
    # AMBIENT DELIVERY (AMBIENT_DELIVERY.md): two opt-in, off-by-default deliveries that reach the user
    # INSIDE whatever client they pointed at clozn -- the in-band receipt FOOTER (an exception-only
    # glass-box line + the /r/<id> permalink appended to the reply). Off by default; the logged run
    # stays the un-footered reply, only the returned content carries the footer. (Footers are the one
    # ambient surface; the earlier desktop-push path was dropped as unneeded complexity.)
    from clozn.server.routes.receipt_link import receipt_enabled
    if rid and body.get("clozn_receipt", receipt_enabled()):
        try:
            import clozn.runs.store as _runlog
            from clozn.runs import receipt_footer
            host = h.headers.get("Host") or "127.0.0.1"
            link = f"http://{host}/r/{rid}"
            foot = receipt_footer.footer(_runlog.get_run(rid), link)
            if foot:
                resp["choices"][0]["message"]["content"] = reply + foot
                resp["clozn_receipt_url"] = link
        except Exception:
            pass                          # additive -- a footer hiccup never breaks the reply
    # FRONTIER §1.1 "trust as an API field": when the caller OPTS IN (clozn_trust:true -- default
    # OFF, so a standard OpenAI response stays byte-unchanged / fully compatible), attach
    # claim-level confidence spans over the reply. Built by the SAME producer as
    # GET /runs/<id>/spans (confidence_spans.spans over the normalized token trace), from THIS
    # turn's trace -- so an agent can branch on per-claim confidence inline, without a second call.
    # HONESTY (FRONTIER §6 ledger): these are RAW, UNCALIBRATED model probabilities. clozn_spans_note
    # says so verbatim -- self-confidence != correctness; NO calibration is done here, and nothing
    # implies confidence == correctness.
    if body.get("clozn_trust"):
        try:
            from clozn.runs import confidence_spans
            import clozn.runs.store as _runlog
            _run_for_spans = {"trace": _runlog.steps_to_trace(trace_steps)}
            resp["clozn_spans"] = confidence_spans.spans(_run_for_spans)
            resp["clozn_spans_note"] = ("uncalibrated raw token confidence -- "
                                        "self-confidence != correctness")
        except Exception:
            pass                          # trust is additive: a spans hiccup never breaks the reply
    h._json(200, resp, extra_headers=extra_headers)
    return True
