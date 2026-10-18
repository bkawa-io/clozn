"""Recorded generation at the public-gateway/private-worker boundary."""
from __future__ import annotations

from dataclasses import dataclass
import time

from clozn.server import app as ctx
from clozn.runs.context_receipt import warnings_for


@dataclass
class InstrumentedChatResult:
    """One text request after generation and durable run recording."""

    reply: str
    trace_steps: list
    finish_reason: str | None
    public_finish_reason: str
    run_id: str | None
    warnings: list[dict]
    reasoning: dict
    usage: dict | None = None


class RunRecordingError(RuntimeError):
    """Generation evidence could not be durably recorded."""

    code = "journal_persistence_failed"


def _recording_error(handler) -> RunRecordingError:
    detail = getattr(handler, "_last_recording_error", None)
    suffix = f": {detail}" if detail else ""
    return RunRecordingError(
        "generation completed but its run could not be durably recorded" + suffix
    )


def _request_usage(sub) -> dict | None:
    request = getattr(sub, "_request", None)
    if request is None:
        return None
    prompt = getattr(request, "prompt_tokens", None)
    completion = getattr(request, "completion_tokens", None)
    if not (isinstance(prompt, int) and not isinstance(prompt, bool)
            and isinstance(completion, int) and not isinstance(completion, bool)):
        return None
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def instrumented_chat(
    handler,
    messages: list,
    *,
    model: str,
    max_tokens: int = 256,
    sample=True,
    source: str,
    extra_meta: dict | None = None,
    stop: list[str] | None = None,
    journal_messages: list | None = None,
) -> InstrumentedChatResult:
    """Generate one text response and durably record the resulting run evidence."""
    sub = ctx.active_sub(handler)
    if not (sub and getattr(sub, "chat", None)):
        raise RuntimeError("model worker unavailable")

    started = time.time()
    trace_steps = []
    memout = {}
    logged_messages = journal_messages if journal_messages is not None else messages
    chat_kw = {"trace_out": trace_steps, "mem_out": memout}
    if stop is not None:
        chat_kw["stop"] = list(stop)
    try:
        reply = sub.chat(messages, int(max_tokens), sample, **chat_kw)
    except Exception as exc:
        run_id = handler._log_run(
            source,
            logged_messages,
            "",
            model,
            started,
            error=str(exc),
            mem_out=memout,
            extra_meta=extra_meta,
        )
        if run_id is None:
            raise _recording_error(handler) from exc
        raise

    raw_reply = str(reply)
    usage = _request_usage(sub)
    if usage:
        memout["usage"] = dict(usage)

    from clozn.runs.think_tags import prompt_opens_think, sanitize_reply, sanitize_steps

    implicit_think = prompt_opens_think(memout.get("final_prompt"))
    think = sanitize_reply(raw_reply, implicit_open=implicit_think)
    public_steps = trace_steps
    reasoning_steps = []
    if think.stripped:
        public_steps, reasoning_steps, _ = sanitize_steps(
            trace_steps, implicit_open=implicit_think
        )
    finish = sub.last_finish_reason() if hasattr(sub, "last_finish_reason") else None
    public_finish = ctx._openai_finish_reason(finish)

    success_meta = dict(extra_meta or {})
    if usage:
        success_meta["usage"] = dict(usage)
    run_id = handler._log_run(
        source,
        logged_messages,
        raw_reply,
        model,
        started,
        trace=trace_steps,
        mem_out=memout,
        finish_reason=finish,
        finish_reason_fallback=None if finish else public_finish,
        extra_meta=success_meta,
    )
    if run_id is None:
        raise _recording_error(handler)

    return InstrumentedChatResult(
        reply=think.public_text,
        trace_steps=public_steps,
        finish_reason=finish,
        public_finish_reason=public_finish,
        run_id=run_id,
        warnings=warnings_for(finish, {"max_tokens": int(max_tokens)}),
        reasoning=think.journal(reasoning_steps=reasoning_steps),
        usage=usage,
    )
