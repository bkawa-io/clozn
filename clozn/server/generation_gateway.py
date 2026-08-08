"""Generation transport at the public/private boundary.

The private C++ worker emits Clozn state events alongside its completion stream.  The
public gateway exposes those frames only on ``/api/clozn/generate``. Standard OpenAI Chat
Completions responses are normalized by the route layer so ordinary clients never receive
engine-internal event types.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
import secrets
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any

from clozn.server import app as ctx
from clozn.server.http_policy import send_cors_headers
from clozn.runs.context_receipt import warnings_for


@dataclass
class InstrumentedChatResult:
    """One compatibility request after it has crossed Clozn's instrumented substrate.

    Compatibility routes own their wire formats, but they must not each grow a second
    inference path.  This result is the shared seam: memory assembly, steering, trace
    capture, finish-reason capture, and run journaling have already happened by the
    time a route turns it into OpenAI- or Ollama-shaped JSON.
    """

    reply: str
    trace_steps: list
    memory: dict
    finish_reason: str | None
    public_finish_reason: str
    run_id: str | None
    warnings: list[dict]
    reasoning: dict
    structured: Any = None




def instrumented_chat(handler, messages: list, *, model: str, max_tokens: int = 256,
                      sample=True, source: str, extra_meta: dict | None = None,
                      journal_messages: list | None = None,
                      output_processor: Callable[[str, Any, str | None], Any] | None = None,
                      native_structured: Mapping[str, Any] | None = None,
                      sections: list | None = None,
                      ) -> InstrumentedChatResult:
    """Run chat through the active substrate and persist the resulting evidence.

    This is deliberately below every compatibility serializer.  The active substrate
    is where Clozn applies prompt-card memory, tone steering, and the
    traced engine call; ``handler._log_run`` is where that evidence becomes a receipt.
    A route calling ``ENGINE.complete`` directly skips all of those layers.

    Generation errors are journaled before being re-raised so callers can preserve
    their protocol-specific error envelope without losing the failed experiment.

    ``sections`` (optional): a caller-built prompt-section manifest (clozn.runs.sections) threaded
    straight through to every ``handler._log_run`` call below, so a route that builds one (today, only
    openai.py's /v1/chat/completions) gets it recorded on every outcome -- error, structured-output
    failure, and success alike -- not just the happy path. Callers that don't build one (ollama.py) simply
    don't pass it, and every _log_run call below already treats an absent/empty manifest as "nothing to
    store" (see runlog.record's own omit-not-null-pad contract).
    """
    sub = ctx.active_sub(handler)
    if not (sub and getattr(sub, "chat", None)):
        raise RuntimeError("model worker unavailable")
    if native_structured is not None and not callable(getattr(sub, "_complete_chat_native", None)):
        raise RuntimeError("active model worker does not expose atomic native structured chat")

    started = time.time()
    trace_steps = []
    memout = {}
    logged_messages = journal_messages if journal_messages is not None else messages
    chat_kw = {"trace_out": trace_steps, "mem_out": memout}
    native_result = None
    try:
        if native_structured is not None:
            contract = dict(native_structured)
            native_result = sub._complete_chat_native(
                messages,
                tools=contract.get("tools") or None,
                tool_choice=contract.get("tool_choice", "auto"),
                json_schema=contract.get("json_schema"),
                parallel_tool_calls=False,
                max_new=int(max_tokens), sample=sample,
                trace_out=trace_steps, mem_out=memout,
                enable_thinking=True,
                reasoning_format="none",
            )
            reply = native_result["raw_model_output"]
        else:
            reply = sub.chat(messages, int(max_tokens), sample, **chat_kw)
    except Exception as exc:
        handler._log_run(source, logged_messages, "", model, started, error=str(exc),
                         mem_out=memout, extra_meta=extra_meta, sections=sections)
        raise

    raw_reply = str(reply)
    from clozn.runs.think_tags import prompt_opens_think, sanitize_reply, sanitize_steps
    implicit_think = prompt_opens_think(memout.get("final_prompt"))
    think = sanitize_reply(raw_reply, implicit_open=implicit_think)
    public_steps = trace_steps
    reasoning_steps = []
    if think.stripped:
        public_steps, reasoning_steps, _ = sanitize_steps(trace_steps, implicit_open=implicit_think)
    finish = sub.last_finish_reason() if hasattr(sub, "last_finish_reason") else None
    public_finish = ctx._openai_finish_reason(finish)
    structured = None
    evidence = None
    if output_processor is not None:
        try:
            processor_value = native_result if native_result is not None else think.public_text
            structured = output_processor(raw_reply, processor_value, finish)
            evidence = (structured.get("evidence") if isinstance(structured, Mapping)
                        else getattr(structured, "evidence", None))
        except Exception as exc:
            failure_meta = dict(extra_meta or {})
            evidence = getattr(exc, "evidence", None)
            rid = handler._log_run(
                source, logged_messages, raw_reply, model, started,
                error=f"{getattr(exc, 'code', 'structured_output_error')}: {exc}",
                trace=trace_steps, mem_out=memout, finish_reason=finish,
                finish_reason_fallback=None if finish else public_finish,
                extra_meta=failure_meta,
                output_contract=evidence if isinstance(evidence, dict) else None,
                sections=sections,
            )
            try:
                exc.run_id = rid
            except Exception:
                pass
            if rid is None:
                from clozn.server.structured_io import StructuredIOError
                persistence = StructuredIOError(
                    "structured output failed and its evidence could not be durably journaled",
                    code="journal_persistence_failed", param=None,
                    evidence={"cause_code": getattr(exc, "code", type(exc).__name__),
                              "output_contract": evidence if isinstance(evidence, dict) else {}},
                )
                persistence.run_id = None
                raise persistence from exc
            raise
    success_meta = dict(extra_meta or {})
    rid = handler._log_run(source, logged_messages, raw_reply, model, started,
                           trace=trace_steps, mem_out=memout, finish_reason=finish,
                           finish_reason_fallback=None if finish else public_finish,
                           extra_meta=success_meta,
                           output_contract=evidence if isinstance(evidence, dict) else None,
                           sections=sections)
    if output_processor is not None and rid is None:
        from clozn.server.structured_io import StructuredIOError
        persistence = StructuredIOError(
            "structured output was validated but its evidence could not be durably journaled",
            code="journal_persistence_failed", param=None,
            evidence={"output_contract": evidence if isinstance(evidence, dict) else {}},
        )
        persistence.run_id = None
        raise persistence
    return InstrumentedChatResult(
        reply=think.public_text, trace_steps=public_steps, memory=memout,
        finish_reason=finish, public_finish_reason=public_finish, run_id=rid,
        warnings=warnings_for(finish, {"max_tokens": int(max_tokens)}),
        reasoning=think.journal(reasoning_steps=reasoning_steps),
        structured=structured,
    )


def model_id() -> str:
    try:
        model = str((ctx.ENGINE.health() or {}).get("model") or "clozn-local")
        name = os.path.basename(model).removesuffix(".gguf")
        return name or "clozn-local"
    except Exception:
        return "clozn-local"


_POLICY_NOTES = {
    "ask": ("confidence on this reply falls in the calibrated 'ask' band -- consider a "
            "clarifying follow-up rather than treating it as a confident answer (from "
            "clozn eval's selective-generation policy, not a live fact-check)"),
    "abstain": ("confidence on this reply falls in the calibrated 'abstain' band -- this answer is "
                "likely wrong; treat it with significant skepticism (from clozn eval's "
                "selective-generation policy, not a live fact-check)"),
}


def _policy_verdict(trace_steps, model: str | None, task: str | None = None) -> dict:
    """The RAW clozn.eval.policy.classify_run verdict for one just-completed reply, doing the SAME saved-
    profile lookup policy_signal below has always used (exact model match, task-indexed when the store
    supports it, the legacy single-report fallback otherwise). Shared by policy_signal (the always-on,
    metadata-only ANNOTATE half) and selective_generation_action (the opt-in ACTION half, calibration
    backlog #10) so both read the identical provenance gate -- a live band can never be stronger than what
    the saved report actually backs up in either caller.

    Returns whatever eval_policy.classify_run returns ({"available": False, "reason": str} or
    {"available": True, "band", "score", "score_aggregate", "answer_at", "ask_at"}), plus
    `calibration_task`/`calibration_model` folded in (the same provenance fields policy_signal has always
    exposed). Never raises -- any failure (a bad saved report, an import problem, a malformed trace)
    collapses to {"available": False, "reason": "..."}."""
    try:
        from clozn.eval import policy as eval_policy, store as eval_store
        import clozn.runs.store as runlog
        load_profile = getattr(eval_store, "load_profile", None)
        if callable(load_profile):
            # The indexed store owns selection semantics: explicit task is exact;
            # omitted task is the newest profile for the exact model.  A miss is
            # final -- never borrow another task's thresholds.
            saved = load_profile(model, task)
        else:
            saved = eval_store.load()
            if task is not None:
                legacy_task = (saved.get("task") or saved.get("set")) if isinstance(saved, dict) else None
                if legacy_task != task:
                    saved = None
        if not saved:
            return {"available": False,
                    "reason": "no calibration saved for this exact model/task -- run `clozn eval --save`"}
        trace = runlog.steps_to_trace(trace_steps)
        if not trace:
            return {"available": False, "reason": "this reply's trace carries no scored content tokens"}
        verdict = dict(eval_policy.classify_run(trace, saved, model=model))
        verdict["calibration_task"] = saved.get("task") or saved.get("set") or task
        verdict["calibration_model"] = saved.get("model")
        return verdict
    except Exception as exc:
        return {"available": False, "reason": f"policy verdict lookup failed: {exc}"}


def _signal_from_verdict(verdict: dict) -> dict | None:
    """The ask/abstain-only, note-carrying LIVE shape (`policy_signal`'s return value), derived from an
    already-computed `_policy_verdict` result without re-deriving it. Split out so
    `policy_verdict_and_signal` below can hand back both the live signal and the persisted receipt from
    ONE `_policy_verdict` call -- a caller that needs both must never pay for (or, worse, cause) two
    separate calibration-store lookups over the same reply."""
    band = verdict.get("band")
    if not verdict.get("available") or band not in ("ask", "abstain"):
        return None
    return {
        "band": band,
        "score": verdict["score"],
        "score_aggregate": verdict["score_aggregate"],
        "answer_at": verdict["answer_at"],
        "ask_at": verdict["ask_at"],
        "calibration_task": verdict.get("calibration_task"),
        "calibration_model": verdict.get("calibration_model"),
        "note": _POLICY_NOTES[band],
    }


def _meta_from_verdict(verdict: dict) -> dict | None:
    """The full, ALL-BANDS receipt shape for persistence (`meta.clozn_policy` -- see
    clozn.runs.attachments.attach_policy_verdict), derived from an already-computed `_policy_verdict`
    result. Unlike `_signal_from_verdict`, this returns a value for EVERY available band including
    'answer' -- a stored run must be able to say "the policy would have answered this" just as honestly
    as it says "asked" or "abstained," or a later reader can never reconstruct which stored answers the
    policy actually blessed. Returns None -- never a fabricated verdict -- when the verdict is
    unavailable (no calibration saved, model/task mismatch, no scored trace, ...); the caller must then
    leave the run's meta.clozn_policy key ABSENT, never default it to 'answer'."""
    if not verdict.get("available"):
        return None
    return {
        "band": verdict["band"],
        "score": verdict["score"],
        "score_aggregate": verdict["score_aggregate"],
        "answer_at": verdict["answer_at"],
        "ask_at": verdict["ask_at"],
        "calibration_task": verdict.get("calibration_task"),
        "calibration_model": verdict.get("calibration_model"),
    }


def policy_signal(trace_steps, model: str | None, task: str | None = None) -> dict | None:
    """The selective-generation policy's verdict for one just-completed /v1/chat/completions reply, or
    None when there is nothing honest to say -- no calibration saved yet (`clozn eval --save`), the saved
    calibration doesn't match this model or carries no usable score aggregate, or this reply's confidence
    is in the 'answer' band. Built on `_policy_verdict` above, which mirrors
    clozn.runs.calibrated_trust.attach_truth's provenance rules (exact model match, a fitted score
    aggregate) so this can never fabricate a verdict the saved report can't back up.

    Signals BOTH the 'ask' and 'abstain' bands -- the calibration backlog item #10 ("a retrieval/clarify
    action wired to the policy's ask band", plus its abstain follow-on: when confidence is low enough that
    the model is likely wrong, say so explicitly rather than staying silent). It is a metadata field the
    caller attaches to the response (or an SSE side-frame), never a change to the generated text; the
    caller decides what, if anything, to do with it -- the 'ask' note suggests a clarifying follow-up, the
    'abstain' note is a stronger warning that the reply is likely wrong. `trace_steps` is the RAW
    per-token step list (chat()'s trace_out, or chat_stream's last_stream_trace()) -- normalized here via
    clozn.runs.store.steps_to_trace, the same shape a stored run's trace carries. Never raises.

    ALWAYS ON -- no opt-in gate. This is metadata only; the reply text itself is never touched. The
    separate, opt-in ACTION that can replace the reply text is `selective_generation_action` below (BK
    decision: abstain/ask may become an ACTION, but OPT-IN, DEFAULT OFF).

    A caller that ALSO needs the all-bands persisted receipt for the same reply (see
    `policy_meta_for_run`) should call `policy_verdict_and_signal` instead of calling this and
    `policy_meta_for_run` separately -- each of those two independently re-runs `_policy_verdict`
    (including the calibration-store lookup), which is wasted work and, worse, double-invokes any
    caller-supplied `eval_store.load_profile`."""
    try:
        return _signal_from_verdict(_policy_verdict(trace_steps, model, task))
    except Exception:
        return None


# Backward-compat alias: earlier callers imported this name back when only the 'ask' band was wired.
ask_band_signal = policy_signal


def policy_meta_for_run(trace_steps, model: str | None, task: str | None = None) -> dict | None:
    """The full, ALL-BANDS selective-generation verdict for one just-completed reply, shaped for
    persistence onto the run record (`meta.clozn_policy` -- see clozn.runs.attachments.
    attach_policy_verdict), never for the wire response. `policy_signal` above stays the live-facing
    surface and deliberately says nothing when the band is 'answer' (there is nothing actionable to
    tell the caller); a stored run has the opposite need -- without an 'answer' verdict on file, nobody
    reading the run back later can tell a policy-blessed answer from a run the policy was never run
    against at all. This is the honest receipt for BOTH cases.

    Returns None -- never a fabricated verdict -- exactly when `_policy_verdict` reports unavailable (no
    calibration saved, model/task mismatch, no scored trace, ...). The caller must leave the run's
    meta.clozn_policy key ABSENT in that case; never default it to 'answer'. Never raises.

    See `policy_signal`'s docstring: a caller needing both this and the live signal for the SAME reply
    should call `policy_verdict_and_signal` instead, to avoid computing `_policy_verdict` twice."""
    try:
        return _meta_from_verdict(_policy_verdict(trace_steps, model, task))
    except Exception:
        return None


def policy_verdict_and_signal(trace_steps, model: str | None,
                               task: str | None = None) -> tuple[dict | None, dict | None]:
    """Compute `_policy_verdict` exactly ONCE and return `(live_signal, persisted_meta)` -- the same
    shapes `policy_signal` and `policy_meta_for_run` return, respectively. For callers (sse_chat, the
    non-streaming /v1/chat/completions handler) that need BOTH the live wire/SSE signal and the
    run-record receipt for the SAME reply: calling `policy_signal` and `policy_meta_for_run` separately
    would re-run the calibration-store lookup twice for one reply, which is wasted work and, when a
    caller's `eval_store.load_profile` is itself instrumented/counted (as
    tests/test_ask_band_server.py's task-selection test does), an observable behavior change. Never
    raises -- any failure collapses to `(None, None)`."""
    try:
        verdict = _policy_verdict(trace_steps, model, task)
    except Exception:
        return None, None
    return _signal_from_verdict(verdict), _meta_from_verdict(verdict)


# =============================================================================================================
# SELECTIVE-GENERATION ACTION (calibration backlog #10, action half) -- BK decision 2026-07-22: abstain/ask
# may become an ACTION, but OPT-IN, DEFAULT OFF. `policy_signal` above remains the always-on, metadata-only
# ANNOTATE half; everything below is the separate, explicitly-gated call that can actually replace the reply
# text when the calibrated band says the model shouldn't just answer.

# The request-body extension field name (mirrors clozn_trust/clozn_receipt/clozn_lens/clozn_task -- see
# clozn.server.openai_compat.CHAT_SUPPORTED_FIELDS, where this is whitelisted as a plain boolean). Absent or
# false -> OFF (byte-identical to policy_signal-only behavior). An explicit request value always wins over
# the server-wide default below, so one caller can opt in/out regardless of the server's setting.
SELECTIVE_FIELD = "clozn_selective"

# The server-wide default (clozn.settings, the product's settings store -- the same mechanism
# clozn.server.routes.receipt_link.RECEIPT_SETTING uses for the receipt-footer default), read only when the
# request omits SELECTIVE_FIELD entirely. Off by default, exactly like the receipt-footer setting.
SELECTIVE_SETTING = "selective_generation"

# HONESTY LAW (docs/RESEARCH_ROADMAP.md's Killed "white-box risk controller advantage" entry): the deployed
# selective-generation score is bit-identical to exp(min(logprob)) -- the same number any black-box
# OpenAI-compatible API already exposes via logprobs. Printed on every fired action so this is never
# mistaken for an internal/white-box advantage, and the known hard-tail failure mode is stated alongside it.
_SELECTIVE_CAVEAT = (
    "calibration-driven, not a live fact-check: the underlying score is TOKEN-PROBABILITY, bit-identical to "
    "the per-token logprobs any black-box OpenAI-compatible API already returns -- never a white-box "
    "advantage (docs/RESEARCH_ROADMAP.md's Killed 'white-box risk controller advantage' entry: the deployed "
    "signal is bit-identical to exp(min(logprob)) across every item it was checked against). It is fit to a "
    "labeled calibration set and known to fail on the hard tail (the confidently-wrong cases that set "
    "under-samples); a calibrated band is a fitted threshold, not a per-answer correctness guarantee."
)


def selective_generation_enabled(body: Mapping | None) -> bool:
    """Whether the OPT-IN selective-generation ACTION should even be attempted for this request -- DEFAULT
    OFF (BK decision). True only when:
      * the request body carries `clozn_selective` (SELECTIVE_FIELD) and it is truthy, OR
      * the request omits that field entirely AND the server-wide `selective_generation` setting
        (SELECTIVE_SETTING, clozn.settings, the product's settings store) is on.
    An explicit `clozn_selective: false` on the request always wins over an ON server default (lets one
    caller opt back out). Never raises -- a settings-store hiccup degrades to False (the safe default)."""
    if isinstance(body, Mapping) and SELECTIVE_FIELD in body:
        return bool(body.get(SELECTIVE_FIELD))
    try:
        import clozn.settings as settings
        return bool(settings.get_setting(SELECTIVE_SETTING, False))
    except Exception:
        return False


def _clarify_question() -> str:
    """The 'a request to rephrase' half of the ask action (see the calibration backlog's own wording: 'the
    model's own clarifying question OR a request to rephrase'). A fixed, honest clarifying request -- the
    model's-own-question alternative would need a second inference round trip, which would make this action
    non-deterministic and untestable without a live engine; this stays a pure, synchronous, model-free
    transform of already-computed inputs (trace + saved profile), matching the rest of this module's
    dependency-light style."""
    return "Could you clarify or rephrase the question so I can answer with more confidence?"


def selective_generation_action(raw_reply: str, trace_steps, model: str | None, task: str | None = None, *,
                                opted_in: bool) -> dict:
    """THE OPT-IN action half of calibration backlog #10. `opted_in` is the caller's already-resolved
    decision (see `selective_generation_enabled`) -- this function reads no request body/server setting
    itself, so its behavior is a pure function of its arguments (trivially testable with fakes).

    Returns a plain dict, NEVER raising:
      * `opted_in` is falsy -> {"applied": False, "reason": "...", "raw_reply": raw_reply}. Callers must
        treat this exactly like policy_signal-only behavior -- no reply-text change, byte-identical to
        today.
      * `opted_in` is truthy but no calibrated profile is available for this exact model/task (or one is
        available but its band is 'answer') -> {"applied": False, "reason": "...", "raw_reply": raw_reply}.
        FAIL CLOSED: the action never fires just because it was requested -- `reason` always says why
        (mirrors `_dial_calibration()`'s missing-file=no-op discipline, but here it names the reason
        instead of silently doing nothing, since the caller explicitly opted in and deserves to know).
      * `opted_in` is truthy AND the calibrated band is 'ask' or 'abstain' -> {"applied": True, "band":
        "ask"|"abstain", "reply": <replacement text>, "raw_reply": raw_reply (NEVER discarded -- the
        original model output always rides here), "verdict": {the same score/threshold/provenance fields
        policy_signal exposes}, "caveat": _SELECTIVE_CAVEAT}.
    """
    if not opted_in:
        return {"applied": False, "reason": "selective-generation action is opt-in and was not requested",
                "raw_reply": raw_reply}
    verdict = _policy_verdict(trace_steps, model, task)
    if not verdict.get("available"):
        reason = verdict.get("reason") or "no calibrated profile for this exact model/task"
        return {"applied": False, "reason": f"fail-closed (annotate-only): {reason}", "raw_reply": raw_reply}
    band = verdict.get("band")
    if band == "answer":
        return {"applied": False, "reason": "calibrated band is 'answer' -- no action needed",
                "raw_reply": raw_reply}
    model_label = verdict.get("calibration_model") or model
    task_label = verdict.get("calibration_task")
    provenance = f"{model_label}" + (f"/{task_label}" if task_label else "")
    if band == "ask":
        reply_text = (
            f"I'm not confident enough to answer directly. {_clarify_question()} "
            f"(calibration-driven: the 'ask' band for {provenance}. {_SELECTIVE_CAVEAT})"
        )
    else:  # abstain
        reply_text = (
            f"I can't confidently assert an answer here, so I won't. Treat this as unresolved rather than "
            f"a fact. (calibration-driven: the 'abstain' band for {provenance}. {_SELECTIVE_CAVEAT})"
        )
    return {
        "applied": True,
        "band": band,
        "reply": reply_text,
        "raw_reply": raw_reply,
        "verdict": {
            "band": band,
            "score": verdict.get("score"),
            "score_aggregate": verdict.get("score_aggregate"),
            "answer_at": verdict.get("answer_at"),
            "ask_at": verdict.get("ask_at"),
            "calibration_task": task_label,
            "calibration_model": model_label,
        },
        "caveat": _SELECTIVE_CAVEAT,
    }


def _request(body: dict, handler=None):
    engine = ctx.active_engine(handler) if handler is not None else ctx.ENGINE
    if engine is None:
        raise RuntimeError("model worker unavailable")
    worker_body = {key: value for key, value in body.items() if key != "model"}
    data = json.dumps(worker_body).encode("utf-8")
    request = urllib.request.Request(
        engine.base + "/v1/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=getattr(engine, "timeout", 600))


def _error(handler, exc: Exception) -> None:
    status = int(getattr(exc, "code", 502) or 502)
    detail = str(exc)
    if isinstance(exc, urllib.error.HTTPError):
        try:
            payload = json.loads(exc.read().decode("utf-8", "replace"))
            detail = str(payload.get("error") or detail)
        except Exception:
            pass
    handler._json(status, {"error": {"message": detail, "type": "upstream_error", "code": status}})


def _native_run_text(frames) -> str:
    """Extract the final native answer without inventing text when the worker sent none."""
    text = ""
    for obj in frames or []:
        if not isinstance(obj, dict):
            continue
        choices = obj.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            value = choices[0].get("text")
            if isinstance(value, str):
                text = value
    if text:
        return text
    try:
        import clozn.runs.store as runlog
        return "".join(str(step.get("piece") or "")
                        for step in runlog.accumulate_ar_events(frames))
    except Exception:
        return ""


def _native_log_run(handler, body: dict, frames: list[dict], started: float,
                    *, response_text: str | None = None, error: str | None = None):
    """Persist a native generation after its transparent worker exchange.

    Native mode deliberately keeps the worker's typed event stream on the wire, so it cannot use
    ``instrumented_chat`` without changing the protocol.  This helper folds the same event frames into
    the shared trace/timing shape and calls the handler's normal journal seam exactly once.

    STANDS DOWN when the caller declares it journals the turn itself (``X-Clozn-Client-Journals``).
    Exactly one caller does: ``clozn run``/REPL, whose ``_log_run_cli`` records a STRICTLY RICHER run
    than this function can. It has two things unavailable here -- the user's raw question (so
    ``messages`` reads as a question rather than as chat-template syntax; this route only ever sees the
    already-rendered ``prompt``) and ``final_prompt``, the exact wire input, which Gate 0 requires and
    which the ``handler._log_run`` call below does not pass at all. Without this check both writers
    fire and every CLI turn produces TWO run records -- the richer one and this weaker one.

    Deliberately a request HEADER, not a body field: ``native_completion`` proxies ``body`` verbatim to
    the C++ worker, so an extra key there would reach a process that never asked for it. Headers stop
    at the gateway. It is also not a stream frame, which keeps the native stream byte-transparent as
    that path's own comment requires.
    """
    try:
        if str(getattr(handler, "headers", {}).get("X-Clozn-Client-Journals") or "").strip():
            return None
    except Exception:
        pass
    try:
        import clozn.runs.store as runlog
        prompt = body.get("prompt", "")
        if not isinstance(prompt, str):
            prompt = str(prompt)
        messages = [{"role": "user", "content": prompt}]
        steps = runlog.accumulate_ar_events(frames)
        finish = runlog.finish_reason_from_frames(frames)
        raw_finish = runlog.raw_finish_reason_from_frames(frames)
        timing = runlog.generation_timing_from_frames(frames)
        response = _native_run_text(frames) if response_text is None else str(response_text)
        prompt_tokens = next(
            (obj.get("prompt_tokens") for obj in frames
             if isinstance(obj, dict) and obj.get("type") == "gen_started"
             and isinstance(obj.get("prompt_tokens"), int)),
            None,
        )
        mem_out = {"assembled_messages": messages, "final_prompt": prompt}
        if isinstance(prompt_tokens, int):
            mem_out["actual_prompt_tokens"] = prompt_tokens
        extra_meta = {
            "native_surface": "/api/clozn/generate",
            "native_stream": bool(body.get("stream")),
        }
        if raw_finish:
            extra_meta["raw_finish_reason"] = raw_finish
        extra_meta.update(timing or {})
        try:
            from clozn.runs import sections as clozn_sections
            sections = clozn_sections.auto_chunk_prompt(prompt)
        except Exception:
            sections = None
        model = (
            getattr(handler, "_selected_model_id", None)
            or body.get("model")
            or model_id()
        )
        return handler._log_run(
            "native_api", messages, response, str(model), started,
            error=error, trace=steps, mem_out=mem_out, finish_reason=finish,
            extra_meta=extra_meta, sections=sections,
        )
    except Exception:
        # Journaling is best effort for compatibility routes.  The route must still return the native
        # worker response if a local receipt writer is unavailable.
        return None


def native_completion(handler, body: dict) -> None:
    """Transparent Clozn event stream used by the CLI and Studio instrumentation."""
    started = time.time()
    frames: list[dict] = []
    try:
        # Keep the historical one-argument `_request` test/extension seam while
        # the built-in implementation accepts the request handler needed for
        # exact private-worker dispatch.
        import inspect
        request_params = inspect.signature(_request).parameters.values()
        handler_aware = (
            len(request_params) >= 2
            or any(param.kind == inspect.Parameter.VAR_POSITIONAL
                   for param in request_params)
        )
        response = _request(body, handler) if handler_aware else _request(body)
    except Exception as exc:
        _native_log_run(handler, body, frames, started, error=str(exc))
        _error(handler, exc)
        return
    try:
        if body.get("stream"):
            handler.send_response(getattr(response, "status", 200))
            handler.send_header("Content-Type", "text/event-stream")
            handler.send_header("Cache-Control", "no-cache")
            send_cors_headers(handler)
            handler.end_headers()
            stream_error = None
            saw_done = False
            try:
                routing = getattr(handler, "_model_routing_artifact", None)
                if routing is not None:
                    frame = {
                        "type": "model_routing",
                        "clozn_model_routing": routing,
                    }
                    handler.wfile.write(
                        ("data: " + json.dumps(frame) + "\n\n").encode("utf-8")
                    )
                    handler.wfile.flush()
                for line in response:
                    decoded = line.decode("utf-8", "replace").strip()
                    if decoded.startswith("data:"):
                        payload = decoded[5:].strip()
                        if payload == "[DONE]":
                            saw_done = True
                            # Preserve the worker's terminal marker byte-for-byte.  Native streaming is
                            # intentionally transparent; journaling happens after the exchange and does
                            # not add a protocol frame.
                            handler.wfile.write(line)
                            handler.wfile.flush()
                            continue
                        try:
                            obj = json.loads(payload)
                        except Exception:
                            obj = None
                        if isinstance(obj, dict):
                            frames.append(obj)
                    handler.wfile.write(line)
                    handler.wfile.flush()
            except Exception as exc:
                stream_error = str(exc)
            _native_log_run(
                handler, body, frames, started,
                error=(f"native stream failed: {stream_error}" if stream_error else None),
            )
            if stream_error:
                frame = {"error": {"message": stream_error, "type": "upstream_error"}}
                try:
                    handler.wfile.write(("data: " + json.dumps(frame) + "\n\n").encode("utf-8"))
                    if not saw_done:
                        handler.wfile.write(b"data: [DONE]\n\n")
                except Exception:
                    pass
            if stream_error:
                try:
                    handler.wfile.flush()
                except Exception:
                    pass
            return
        raw = response.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
            if isinstance(payload, dict):
                frames.append(payload)
        except Exception:
            payload = None
        run_id = _native_log_run(handler, body, frames, started)
        routing = getattr(handler, "_model_routing_artifact", None)
        if routing is not None:
            if isinstance(payload, dict):
                payload["clozn_model_routing"] = routing
                raw = json.dumps(payload).encode("utf-8")
        if run_id and isinstance(payload, dict):
            payload["clozn_run_id"] = run_id
            raw = json.dumps(payload).encode("utf-8")
        handler._send(getattr(response, "status", 200), raw, "application/json")
    finally:
        response.close()
