"""NDJSON streaming for clozn's Ollama-shaped shim (roadmap PRODUCT_ROADMAP.md Phase 2 item 1:
"Ollama NDJSON streaming").

Ollama's streamed wire is newline-delimited JSON (Content-Type ``application/x-ndjson``) -- one JSON
object per line, no ``data:``/``[DONE]`` SSE framing. This module is the NDJSON twin of
``clozn.server.sse``: it reuses the EXACT SAME generation seam (``Substrate.chat_stream()``) that
``/v1/chat/completions``'s SSE branch does -- there is no second inference path for Ollama, only a
different wire serializer over the same token stream -- and mirrors sse.py's disconnect-vs-worker-
failure split and one-coherent-run contract (see sse.py's module docstring for the full rationale;
summarized again here so this file is independently reviewable):

  * CLIENT DISCONNECT (a write to ``handler.wfile`` raises OSError): stop pulling from the worker
    immediately (``gen.close()`` throws GeneratorExit at chat_stream's suspended ``yield``), mark the
    substrate's RequestContext cancelled (belt-and-suspenders alongside that ``gen.close()``), and log
    the partial reply as a distinct ``"client_disconnected"`` failure. ``finish_reason`` stays unset --
    a generation that did not finish normally must never claim ``done_reason: "stop"``.
  * WORKER-DIES-MIDSTREAM (iterating ``chat_stream`` itself raises): the client is presumably still
    there and HTTP is already committed to 200, so the only channel left is one final NDJSON line
    carrying an honest ``"error"`` field -- never a silent truncated stream, never a hang.
  * Exactly ONE ``_log_run`` call per request, on every exit path (success, disconnect, worker
    failure) -- never zero, never two.

HONESTY -- the final ``done: true`` line's timing/count fields, one bullet per field so an omission is
a documented decision, not an oversight:
  * ``total_duration`` (ns): the wall-clock span of THIS call, directly measured (``time.time()`` at
    call start and again once the stream ends) -- always fillable.
  * ``eval_count``: ``len(substrate.last_stream_trace())`` -- the number of per-token commits the
    engine itself reported for this reply. Omitted only when the active substrate has no
    ``last_stream_trace`` at all (a minimal test double).
  * ``prompt_eval_count``: the engine's own ``gen_started`` frame reports ``prompt_tokens`` verbatim;
    ``EngineSubstrate.chat_stream`` captures it onto the call's RequestContext the same way it already
    captures ``engine_req`` (see substrates.py + request_context.py). Omitted when the substrate never
    reported one (older engine build, or a test double).
  * ``load_duration``, ``prompt_eval_duration``, ``eval_duration`` are NEVER emitted: clozn's engine is
    one always-resident worker process (there is no per-request model load to time), and the engine's
    SSE frames carry per-token committal timestamps but no separate prompt-phase-vs-decode-phase
    wall-clock split -- there is nothing honest to report for any of the three, so all three are
    omitted rather than guessed or zero-filled.

``done_reason`` maps clozn's own two finish reasons (``"stop"``, ``"length"``) to themselves -- the
only two Ollama also defines for a normal completion -- and is OMITTED (never invented) for anything
else, including an unset/unknown finish reason.
"""
from __future__ import annotations

import json
import time

from clozn.server import app as ctx
from clozn.server.http_policy import send_cors_headers


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# Ollama's own two normal-completion stop causes; anything else (None included) is omitted rather than
# guessed -- see the module docstring's `done_reason` paragraph.
_DONE_REASON = {"stop": "stop", "length": "length"}


def _timing_fields(started: float, ended: float, trace, prompt_tokens) -> dict:
    out = {"total_duration": int((ended - started) * 1_000_000_000)}
    if trace is not None:
        out["eval_count"] = len(trace)
    if prompt_tokens is not None:
        out["prompt_eval_count"] = int(prompt_tokens)
    return out


def ndjson_stream(handler, messages, max_new, model, *, operation, sample=True, source="ollama_api",
                  journal_messages=None, corrective_evidence=None):
    """Stream one ``/api/chat`` (``operation="chat"``) or ``/api/generate`` (``operation="generate"``)
    reply as Ollama NDJSON, then log exactly one run.

    The two Ollama operations differ ONLY in which key carries each piece -- chat chunks nest it under
    ``message: {role: "assistant", content: <piece>}``; generate chunks carry a bare ``response: <piece>``
    -- everything else, including the generation call itself, is shared. `sample` is the same bool-or-
    override-dict contract ``EngineSubstrate.chat_stream`` already documents (from the shim's
    ``_generation_options``); `messages` is always the chat-shaped list -- ``/api/generate``'s
    prompt/system fields are folded into it by the route before this is called, exactly as the
    non-streaming path already does.
    """
    handler.send_response(200)
    handler.send_header("Content-Type", "application/x-ndjson")
    handler.send_header("Cache-Control", "no-cache")
    send_cors_headers(handler)
    handler.end_headers()

    def line(piece=None, *, thinking=None, done=False, done_reason=None, extra=None):
        o = {"model": model, "created_at": _iso_now(), "done": done}
        if operation == "chat":
            o["message"] = {"role": "assistant", "content": piece or ""}
            if thinking is not None:
                o["message"]["thinking"] = thinking
        else:
            o["response"] = piece or ""
            if thinking is not None:
                o["thinking"] = thinking
        if done_reason is not None:
            o["done_reason"] = done_reason
        if extra:
            o.update(extra)
        return o

    def _write(obj):
        """Write one NDJSON line. Returns the captured OSError on a client disconnect, else None --
        mirrors sse.py's `_write` closure so the two streaming paths fail identically."""
        try:
            handler.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
            handler.wfile.flush()
            return None
        except OSError as write_err:
            return write_err

    t0 = time.time()
    acc: list[str] = []
    memout: dict = {}
    sub = ctx.active_sub(handler)
    logged_messages = journal_messages if journal_messages is not None else messages
    policy_meta = {"corrective_policy": corrective_evidence}
    stream_kw = {"mem_out": memout, "sample": sample}
    import inspect
    try:
        params = inspect.signature(sub.chat_stream).parameters
    except Exception:
        params = {}
    gen = sub.chat_stream(messages, max_new, **stream_kw)
    disconnect_error = None
    think_stream = None

    def _emit(piece=None, *, thinking=None):
        nonlocal disconnect_error
        err = _write(line(piece, thinking=thinking))
        if err is not None:
            disconnect_error = err
            return False
        return True

    try:
        for piece in gen:
            acc.append(piece)
            if think_stream is None:
                from clozn.runs.think_tags import ThinkTagStream, prompt_opens_think
                think_stream = ThinkTagStream(
                    implicit_open=prompt_opens_think(memout.get("final_prompt"))
                )
            public_piece = think_stream.feed(piece)
            for thinking_piece in think_stream.drain_reasoning():
                if not _emit(thinking=thinking_piece):
                    break
            if disconnect_error is not None:
                break
            if public_piece and not _emit(public_piece):
                break
        if disconnect_error is None and think_stream is not None:
            tail, _ = think_stream.finish()
            for thinking_piece in think_stream.drain_reasoning():
                if not _emit(thinking=thinking_piece):
                    break
            if disconnect_error is None and tail:
                _emit(tail)
        if disconnect_error is not None:
            # CLIENT DISCONNECT: log the partial reply honestly and stop -- no further writes are
            # attempted (the client is confirmed gone), and finish_reason stays unset (never "stop").
            req_ctx = getattr(sub, "_request", None)
            if req_ctx is not None and hasattr(req_ctx, "cancel"):
                req_ctx.cancel()          # durable record on the context itself, belt to gen.close()'s suspenders
            handler._log_run(source, logged_messages, "".join(acc), model, t0,
                             error=f"client disconnected mid-stream: {disconnect_error}", mem_out=memout,
                             extra_meta={**policy_meta, "stream_failure": "client_disconnected",
                                         "compatibility_api": "ollama", "ollama_operation": operation})
            return
        fr = sub.last_finish_reason() if hasattr(sub, "last_finish_reason") else None
        trace = sub.last_stream_trace() if hasattr(sub, "last_stream_trace") else None
        prompt_tokens = sub.last_prompt_tokens() if hasattr(sub, "last_prompt_tokens") else None
        rid = handler._log_run(source, logged_messages, "".join(acc), model, t0, trace=trace, mem_out=memout,
                               finish_reason=fr,
                               extra_meta={**policy_meta, "compatibility_api": "ollama",
                                           "ollama_operation": operation})
        ended = time.time()
        timing = _timing_fields(t0, ended, trace, prompt_tokens)
        from clozn.runs.context_receipt import warnings_for
        cutoff_warnings = warnings_for(fr, {"max_tokens": int(max_new)})
        if cutoff_warnings:
            timing["clozn_warnings"] = cutoff_warnings
        if rid:
            timing["clozn_run_id"] = rid
        _write(line(done=True, done_reason=_DONE_REASON.get(fr), extra=timing))
    except Exception as e:
        # WORKER-DIES-MIDSTREAM: see the module docstring. Distinct from the client-disconnect branch
        # above because this exception came from ITERATING `gen` (reading the worker), never from
        # writing to `handler.wfile`.
        rid = handler._log_run(source, logged_messages, "".join(acc), model, t0, error=str(e), mem_out=memout,
                               extra_meta={**policy_meta, "stream_failure": "worker_disconnected",
                                           "compatibility_api": "ollama", "ollama_operation": operation})
        try:
            terminal = {"error": str(e)}
            if rid:
                terminal["clozn_run_id"] = rid
            _write(line(done=True, extra=terminal))
        except Exception:
            pass
    finally:
        # ALWAYS release the worker connection -- normal completion, a caught client disconnect, an
        # uncaught worker failure, or anything raised while logging/writing the final line above --
        # rather than relying on CPython refcounting to eventually GC `gen`. close() on an
        # already-exhausted/closed generator is a documented no-op, so this is always safe to call.
        gen.close()
