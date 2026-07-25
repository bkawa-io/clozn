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


def sse_chat(handler, messages, max_new, model, lens=None, receipt=False, sample=True,
             journal_messages=None, corrective_evidence=None, task=None):
    """Stream one /v1/chat/completions reply as OpenAI-style `chat.completion.chunk` frames over SSE,
    then log the run. `handler` is the live BaseHTTPRequestHandler (needs .wfile + ._log_run).

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

    import inspect
    try:
        params = inspect.signature(ctx.active_sub(handler).chat_stream).parameters
    except Exception:
        params = {}
    stream_kw = {}
    if "sample" in params:
        stream_kw["sample"] = sample
    if "memory_scope" in params:
        from clozn.server.generation_gateway import request_memory_scope
        stream_kw["memory_scope"] = request_memory_scope(handler)
    if lens:
        if "lens" in params and "on_frame" in params:
            stream_kw.update(lens=(lens if isinstance(lens, dict) else {}), on_frame=side_frame)
        else:
            side_frame({"error": "live lens needs the engine substrate (post-hoc: POST /runs/<id>/jlens)"})

    # HF chat stream (QwenSubstrate.chat_stream): a pure pass-through recorder rides along and the
    # per-token trace is assembled after the stream (SUB.last_stream_trace()) -- so the run gets the
    # Run Inspector timeline while the streamed chunks stay byte-identical (B3). runlog.record
    # normalizes the raw step list; on any hiccup last_stream_trace() is [] -> a clean empty trace.
    # memout: prompt mode fills what memory ACTUALLY rode this turn (block gated in/out) for the log.
    #
    # The run id cannot ride response headers because SSE headers are committed before generation.
    # Instead it is an additive top-level field on the ordinary terminal finish chunk, before [DONE].
    # Strict OpenAI clients ignore unknown fields while sidecars and Studio can correlate the exact run.
    t0 = time.time(); acc = []; memout = {}
    logged_messages = journal_messages if journal_messages is not None else messages
    policy_meta = {}
    if corrective_evidence is not None:
        policy_meta["corrective_policy"] = corrective_evidence
    if task is not None:
        policy_meta["clozn_task"] = task
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
        # log the run FIRST (before the finish chunk) so the receipt footer can carry its /r/<id> link.
        trace = sub.last_stream_trace() if hasattr(sub, "last_stream_trace") else None
        rid = handler._log_run("openai_api", logged_messages, "".join(acc), model, t0, trace=trace,
                               mem_out=memout, finish_reason=fr,
                               finish_reason_fallback=None if fr else openai_fr,
                               extra_meta=policy_meta or None)
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
        # CALIBRATION BACKLOG #10 "ask/abstain bands": same metadata-only signal as the non-streaming
        # route (see generation_gateway.policy_signal), delivered as one side-frame before the finish
        # chunk -- silent (no frame at all) unless a matching, fitted calibration actually says "ask" or
        # "abstain".
        from clozn.server.generation_gateway import policy_verdict_and_signal
        policy_trace = trace
        if think_result is not None and think_result.stripped and isinstance(trace, list):
            from clozn.runs.think_tags import prompt_opens_think, sanitize_steps
            policy_trace, _, _ = sanitize_steps(
                trace, implicit_open=prompt_opens_think(memout.get("final_prompt"))
            )
        # ONE _policy_verdict lookup feeds both the live ask/abstain-only signal below and the
        # all-bands receipt persisted onto the run record -- policy_verdict_and_signal's own docstring
        # explains why this must not be two separate calls.
        policy, policy_verdict_meta = policy_verdict_and_signal(policy_trace, model, task=task)
        if policy:
            chunk({}, extension={"clozn_policy": policy})
        # Persist the ALL-BANDS verdict (including 'answer', which the live signal above never
        # surfaces) onto the run record, so a later reader of THIS run can tell whether the policy
        # would have answered, asked, or abstained -- not just live callers. After the fact because the
        # run must already be logged (rid assigned just above) before its own trace can be scored.
        # Additive and best-effort: any hiccup here must never touch the SSE stream already in flight.
        if rid and policy_verdict_meta:
            try:
                from clozn.runs.attachments import attach_policy_verdict
                attach_policy_verdict(rid, policy_verdict_meta)
            except Exception:
                pass
        from clozn.runs.context_receipt import warnings_for
        cutoff_warnings = warnings_for(fr, {"max_tokens": int(max_new)})
        terminal = {}
        if cutoff_warnings:
            terminal["clozn_warnings"] = cutoff_warnings
        if rid:
            terminal["clozn_run_id"] = rid
        chunk({}, finish=openai_fr, extension=terminal or None)
        handler.wfile.write(b"data: [DONE]\n\n")
        handler.wfile.flush()
    except Exception as e:
        # WORKER-DIES-MIDSTREAM (or any other chat_stream failure): see the module docstring. Distinct
        # from the client-disconnect branch above because this exception came from ITERATING `gen`
        # (reading the worker), never from writing to `handler.wfile`.
        handler._log_run("openai_api", logged_messages, "".join(acc), model, t0, error=str(e), mem_out=memout,
                         extra_meta={**policy_meta, "stream_failure": "worker_disconnected"})
        try:
            handler.wfile.write(("data: " + json.dumps({"error": str(e)}) + "\n\n").encode("utf-8"))
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
