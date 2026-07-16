"""SSE (server-sent events) helper for the one streaming surface clozn.server.app serves:
`/v1/chat/completions` with `stream: true`. Mechanical extraction of app.py's old `_sse_chat` method --
same OpenAI-compatible chunk shape, same run-logging on completion/error, behavior unchanged.

Reads the live substrate via `clozn.server.app` (not a captured import) so a substrate swap -- or a test's
`monkeypatch.setattr(app, "SUB", ...)` -- is observed at call time, exactly as it was when this code lived
directly in app.py.
"""
import json
import secrets
import time

from clozn.server import app as ctx
from clozn.server.http_policy import send_cors_headers


def sse_chat(handler, messages, max_new, model, lens=None, receipt=False, sections=None):
    """Stream one /v1/chat/completions reply as OpenAI-style `chat.completion.chunk` frames over SSE,
    then log the run. `handler` is the live BaseHTTPRequestHandler (needs .wfile + ._log_run).

    `sections` (prompt-section influence foundation): the caller's already-built explicit-or-auto section
    manifest (see clozn.runs.sections / clozn.server.routes.openai's try_post), threaded straight through
    to both `_log_run` calls below (success and error) so a streamed chat run gets the same manifest a
    non-streamed one does. None (the default) is exactly today's behavior for any caller that predates this.

    AMBIENT DELIVERY (AMBIENT_DELIVERY.md): `receipt` (the request's opt-in `clozn_receipt`, or the
    server-wide default) appends the exception-only in-band footer as ONE final content chunk -- so the
    run must be logged BEFORE the finish chunk (to have an id + trace for the /r/<id> link), unlike the
    old order. Off => byte-identical to before.

    F1 LIVE LENS: `lens` (from the request's opt-in `clozn_lens` field -- absent for every standard
    OpenAI client, whose stream stays byte-identical) forwards to the substrate's chat_stream, and each
    engine `jlens_live` frame is relayed as its own SSE frame `data: {"clozn_lens": {...}}` interleaved
    with the delta chunks. Only substrates whose chat_stream accepts (lens, on_frame) can serve it --
    currently the engine substrate; on any other, one honest error frame is sent and the chat proceeds
    without the lens (the reply must never be held hostage by a readout)."""
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    send_cors_headers(handler)
    handler.end_headers()

    stream_id = "chatcmpl-" + secrets.token_hex(8)
    created = int(time.time())

    def chunk(delta, finish=None, extension=None):
        o = {"id": stream_id, "object": "chat.completion.chunk", "created": created, "model": model,
             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if extension:
            o.update(extension)
        handler.wfile.write(("data: " + json.dumps(o) + "\n\n").encode("utf-8"))
        handler.wfile.flush()

    def side_frame(obj):
        # Keep the opt-in extension inside a valid chat.completion.chunk envelope. Strict clients can
        # ignore the extra top-level field without encountering a foreign event shape on /v1/*.
        chunk({}, extension={"clozn_lens": obj})

    lens_kw = {}
    if lens:
        import inspect
        try:
            params = inspect.signature(ctx.active_sub(handler).chat_stream).parameters
        except Exception:
            params = {}
        if "lens" in params and "on_frame" in params:
            lens_kw = {"lens": (lens if isinstance(lens, dict) else {}), "on_frame": side_frame}
        else:
            side_frame({"error": "live lens needs the engine substrate (post-hoc: POST /runs/<id>/jlens)"})

    # HF chat stream (QwenSubstrate.chat_stream): a pure pass-through recorder rides along and the
    # per-token trace is assembled after the stream (SUB.last_stream_trace()) -- so the run gets the
    # Run Inspector timeline while the streamed chunks stay byte-identical (B3). runlog.record
    # normalizes the raw step list; on any hiccup last_stream_trace() is [] -> a clean empty trace.
    # memout: prompt mode fills what memory ACTUALLY rode this turn (block gated in/out) for the log.
    #
    # M5 any-client run_id bridge (EXPLAIN_THIS_ANSWER_SPEC.md): DEFERRED here, deliberately. Headers
    # are already flushed above (handler.end_headers(), before a single token is generated), so
    # X-Clozn-Run-Id can never ride a header on this path -- the id only exists after _log_run runs,
    # which is after the stream ends. A trailing SSE frame isn't clean either: a frame AFTER
    # "data: [DONE]" is silently dropped by clients (incl. openai-python) that stop reading at the
    # [DONE] sentinel, and a frame BEFORE [DONE] would need a full spec-shaped chat.completion.chunk
    # (id/object/created/model/choices) just to smuggle one field, plus a stray chunk after the real
    # finish_reason:"stop" chunk -- exactly the shape drift the honesty/compat contract rules out.
    # Left unchanged; non-streaming (the required deliverable) carries clozn_run_id both ways.
    t0 = time.time(); acc = []; memout = {}
    try:
        chunk({"role": "assistant"})
        for piece in ctx.active_sub(handler).chat_stream(messages, max_new, mem_out=memout, **lens_kw):
            acc.append(piece); chunk({"content": piece})
        fr = ctx.active_sub(handler).last_finish_reason() if hasattr(ctx.active_sub(handler), "last_finish_reason") else None
        openai_fr = ctx._openai_finish_reason(fr)
        # log the run FIRST (before the finish chunk) so the receipt footer can carry its /r/<id> link.
        trace = ctx.active_sub(handler).last_stream_trace() if hasattr(ctx.active_sub(handler), "last_stream_trace") else None
        rid = handler._log_run("openai_api", messages, "".join(acc), model, t0, trace=trace,
                               mem_out=memout, finish_reason=fr,
                               finish_reason_fallback=None if fr else openai_fr, sections=sections)
        if receipt and rid:                       # the exception-only footer, as one final content chunk
            try:
                import clozn.runs.store as _runlog
                from clozn.runs import receipt_footer
                host = handler.headers.get("Host") or "127.0.0.1"
                foot = receipt_footer.footer(_runlog.get_run(rid), f"http://{host}/r/{rid}")
                if foot:
                    chunk({"content": foot})
            except Exception:
                pass                              # additive -- a footer hiccup never breaks the stream
        chunk({}, finish=openai_fr)
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
    except Exception as e:
        handler._log_run("openai_api", messages, "".join(acc), model, t0, error=str(e), mem_out=memout,
                         sections=sections)
        try:
            handler.wfile.write(("data: " + json.dumps({"error": str(e)}) + "\n\n").encode("utf-8"))
            handler.wfile.flush()
        except Exception:
            pass
