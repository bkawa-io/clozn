"""SSE (server-sent events) helper for the one streaming surface clozn.server.app serves:
`/v1/chat/completions` with `stream: true`. Mechanical extraction of app.py's old `_sse_chat` method --
same OpenAI-compatible chunk shape, same run-logging on completion/error, behavior unchanged.

Reads the live substrate via `clozn.server.app` (not a captured import) so a substrate swap -- or a test's
`monkeypatch.setattr(app, "SUB", ...)` -- is observed at call time, exactly as it was when this code lived
directly in app.py.

CLIENT DISCONNECT vs WORKER-DIES-MIDSTREAM (backlog #2): sse_chat() is the one place that both WRITES to
the client (`handler.wfile`) and READS from the worker (via the substrate's chat_stream() generator) in the
same loop, so it is the one place that can tell the two failure modes apart -- and they need genuinely
different handling:
  * CLIENT DISCONNECT (the browser tab closed, curl was Ctrl-C'd, ...): a write to `handler.wfile` raises
    BrokenPipeError/ConnectionAbortedError/ConnectionResetError (all OSError on POSIX and Windows alike).
    Nothing more can be delivered, so we stop pulling from the worker immediately -- gen.close() throws
    GeneratorExit into chat_stream at its suspended `yield`, whose own `finally` (substrates.py) turns that
    into an upstream resp.close(), aborting the worker's chunked send rather than draining a reply nobody
    reads. This is logged as a routine, expected shutdown (a distinct `stream_failure` tag), not an error.
  * WORKER-DIES-MIDSTREAM (the C++ worker crashes, resets the connection, or the read times out): the
    failure comes from ITERATING chat_stream (reading the worker), not from writing to the client -- who is
    presumably still there. HTTP status is already committed to 200 (SSE headers went out before the first
    token), so the only channel left to say "this failed" is an in-band frame: emit `data: {"error": ...}`
    honestly, then `data: [DONE]` so a well-behaved SSE consumer stops reading instead of blocking on a
    connection that's about to close. Never a hang, never a silent empty 200.
In both cases finish_reason is left unset (never "stop") -- a generation that did not finish normally must
never claim it did; a missing signal reads as missing, per the project's honesty invariant.
"""
import json
import secrets
import time

from clozn.server import app as ctx
from clozn.server.http_policy import send_cors_headers


def sse_chat(handler, messages, max_new, model, sample=True,
             stop=None, include_usage=False):
    """Stream one /v1/chat/completions reply as OpenAI-style `chat.completion.chunk` frames over SSE,
    then log the run. `handler` is the live BaseHTTPRequestHandler (needs .wfile + ._log_run).

    """
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    send_cors_headers(handler)
    handler.end_headers()

    stream_id = "chatcmpl-" + secrets.token_hex(8)
    created = int(time.time())

    def chunk(delta, finish=None, extension=None, usage=None):
        o = {"id": stream_id, "object": "chat.completion.chunk", "created": created, "model": model,
             "choices": [] if usage is not None else [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage is not None:
            o["usage"] = usage
        if extension:
            o.update(extension)
        flush_started_ns = time.monotonic_ns()
        try:
            handler.wfile.write(("data: " + json.dumps(o) + "\n\n").encode("utf-8"))
            handler.wfile.flush()
        finally:
            recorder = getattr(handler, "_record_gateway_phase", None)
            if callable(recorder):
                recorder(
                    "stream_flush", time.monotonic_ns() - flush_started_ns,
                    owner="client", aggregation="overlapping",
                )

    import inspect
    try:
        params = inspect.signature(ctx.active_sub(handler).chat_stream).parameters
    except Exception:
        params = {}
    stream_kw = {}
    if "sample" in params:
        stream_kw["sample"] = sample
    if stop is not None and "stop" in params:
        stream_kw["stop"] = list(stop)
    # An HF chat stream: a pure pass-through recorder rides along and the
    # per-token trace is assembled after the stream (SUB.last_stream_trace()) -- so the run gets the
    # Run Inspector timeline while the streamed chunks stay byte-identical (B3). runlog.record
    # normalizes the raw step list; on any hiccup last_stream_trace() is [] -> a clean empty trace.
    # memout: prompt mode fills what memory ACTUALLY rode this turn (block gated in/out) for the log.
    #
    # The run id cannot ride response headers because SSE headers are committed before generation.
    # Instead it is an additive top-level field on the ordinary terminal finish chunk, before [DONE].
    # Strict OpenAI clients ignore unknown fields while sidecars and Studio can correlate the exact run.
    t0 = time.time(); acc = []; memout = {}
    logged_messages = messages
    policy_meta = {}
    sub = ctx.active_sub(handler)
    gen = sub.chat_stream(messages, max_new, mem_out=memout, **stream_kw)
    disconnect_error = None
    think_stream = None
    think_result = None

    def _write(delta, finish=None, extension=None):
        """chunk(), but a write failure is captured as a CLIENT DISCONNECT (see the module docstring)
        instead of propagating as a generic stream failure. Returns False (the caller should stop trying
        to write/pull more) on a captured disconnect, True otherwise."""
        nonlocal disconnect_error
        try:
            chunk(delta, finish=finish, extension=extension)
            return True
        except OSError as write_err:
            disconnect_error = write_err
            return False

    try:
        if _write({"role": "assistant"}):
            for piece in gen:
                acc.append(piece)
                if think_stream is None:
                    from clozn.runs.think_tags import ThinkTagStream, prompt_opens_think
                    think_stream = ThinkTagStream(
                        implicit_open=prompt_opens_think(memout.get("final_prompt"))
                    )
                public_piece = think_stream.feed(piece)
                if public_piece and not _write({"content": public_piece}):
                    break
        if disconnect_error is None and think_stream is not None:
            tail, think_result = think_stream.finish()
            if tail and not _write({"content": tail}):
                disconnect_error = disconnect_error or OSError("client disconnected while flushing reply")
        if disconnect_error is not None:
            # CLIENT DISCONNECT: log the partial reply honestly and stop -- no further writes are
            # attempted (the client is confirmed gone), and finish_reason stays unset (never "stop").
            req_ctx = getattr(sub, "_request", None)
            if req_ctx is not None and hasattr(req_ctx, "cancel"):
                req_ctx.cancel()          # durable record on the context itself, belt to gen.close()'s suspenders
            handler._log_run("openai_api", logged_messages, "".join(acc), model, t0,
                             error=f"client disconnected mid-stream: {disconnect_error}", mem_out=memout,
                             extra_meta={**policy_meta, "stream_failure": "client_disconnected"})
            return
        fr = sub.last_finish_reason() if hasattr(sub, "last_finish_reason") else None
        openai_fr = ctx._openai_finish_reason(fr)
        # Log before the terminal chunk so a completed stream always carries its durable run id.
        trace = sub.last_stream_trace() if hasattr(sub, "last_stream_trace") else None
        usage = None
        req = getattr(sub, "_request", None)
        prompt_count = getattr(req, "prompt_tokens", None)
        completion_count = getattr(req, "completion_tokens", None)
        if (isinstance(prompt_count, int) and not isinstance(prompt_count, bool)
                and isinstance(completion_count, int) and not isinstance(completion_count, bool)):
            usage = {"prompt_tokens": prompt_count,
                     "completion_tokens": completion_count,
                     "total_tokens": prompt_count + completion_count}
        run_meta = dict(policy_meta)
        if usage:
            run_meta["usage"] = dict(usage)
        rid = handler._log_run("openai_api", logged_messages, "".join(acc), model, t0, trace=trace,
                               mem_out=memout, finish_reason=fr,
                               finish_reason_fallback=None if fr else openai_fr,
                               extra_meta=run_meta or None)
        if rid is None:
            message = "generation completed but its run could not be durably recorded"
            detail = getattr(handler, "_last_recording_error", None)
            if detail:
                message += f": {detail}"
            _write({}, extension={"error": {"message": message,
                                              "type": "upstream_error",
                                              "code": "journal_persistence_failed"}})
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
            return
        from clozn.runs.context_receipt import warnings_for
        cutoff_warnings = warnings_for(fr, {"max_tokens": int(max_new)})
        terminal = {}
        if cutoff_warnings:
            terminal["clozn_warnings"] = cutoff_warnings
        if rid:
            terminal["clozn_run_id"] = rid
        chunk({}, finish=openai_fr, extension=terminal or None)
        if include_usage and usage is not None:
            chunk({}, usage=usage)
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
    except Exception as e:
        # WORKER-DIES-MIDSTREAM (or any other chat_stream failure): see the module docstring. Distinct
        # from the client-disconnect branch above because this exception came from ITERATING `gen`
        # (reading the worker), never from writing to `handler.wfile`.
        rid = handler._log_run("openai_api", logged_messages, "".join(acc), model, t0, error=str(e), mem_out=memout,
                               extra_meta={**policy_meta, "stream_failure": "worker_disconnected"})
        recording_failed = rid is None
        if recording_failed:
            detail = getattr(handler, "_last_recording_error", None)
            e = RuntimeError("generation failed and its run could not be durably recorded" +
                             (f": {detail}" if detail else ""))
        try:
            error_frame = {"error": str(e)}
            if recording_failed:
                error_frame["error_code"] = "journal_persistence_failed"
            handler.wfile.write(("data: " + json.dumps(error_frame) + "\n\n").encode("utf-8"))
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
        except Exception:
            pass
    finally:
        # ALWAYS release the worker connection on every exit path -- normal completion, a caught client
        # disconnect, an uncaught worker failure, or anything raised by logging/the receipt footer above --
        # rather than relying on CPython refcounting to eventually GC `gen` and trigger its GeneratorExit
        # finally (substrates.py's chat_stream) at some later, unspecified time. close() on an
        # already-exhausted/closed generator is a documented no-op, so this is always safe to call.
        gen.close()
