"""Public OpenAI API plus the namespaced Clozn generation event stream."""
import json
import os
import secrets
import time
import unicodedata
from collections.abc import Mapping
from typing import Any

from clozn.server import app as ctx
from clozn.server.structured_io import StructuredIOError


def _api_error(h, status: int, message: str, *, param=None, kind="invalid_request_error", code=None):
    error = {"message": message, "type": kind, "param": param, "code": code or status}
    h._json(status, {"error": error})


def _calibration_task(body) -> tuple[dict | Any, str | None]:
    """Extract Clozn's task extension before standard OpenAI normalization.

    The compatibility normalizer intentionally rejects unknown fields.  Keeping
    this narrow extraction here preserves that policy table while accepting one
    local, bounded identifier that never reaches model prompting.
    """
    if not isinstance(body, Mapping) or "clozn_task" not in body:
        return body, None
    value = body.get("clozn_task")
    if (isinstance(value, bool) or not isinstance(value, str)
            or any(unicodedata.category(char) == "Cc" for char in value)):
        raise ValueError(
            "clozn_task must be a non-empty string of at most 80 characters without control characters"
        )
    normalized_task = " ".join(value.strip().lower().split())
    if not normalized_task or len(normalized_task) > 80:
        raise ValueError(
            "clozn_task must be a non-empty string of at most 80 characters without control characters"
        )
    normalized_input = dict(body)
    normalized_input.pop("clozn_task", None)
    return normalized_input, normalized_task


def _active_identity(sub) -> dict:
    """Read only the loaded substrate's identity; never trust the request model label."""
    try:
        identity = sub.identity_meta() if callable(getattr(sub, "identity_meta", None)) else {}
    except Exception:
        identity = {}
    identity = dict(identity) if isinstance(identity, Mapping) else {}
    for field in ("model_sha256", "template_fingerprint"):
        if not identity.get(field):
            try:
                value = getattr(sub, field, None)
            except Exception:
                value = None
            if value:
                identity[field] = value
    return identity


def _structured_processor(contract: dict, qualification: dict):
    from clozn.server.structured_io import (
        StructuredIOError, parse_output, qualification_evidence, structured_parser_input,
    )

    qualified = qualification_evidence(qualification, contract["mode"], contract)
    request_evidence = {
        "mode": contract["mode"],
        "tools": contract.get("active_tools") or [],
        "tool_choice": contract.get("tool_choice"),
        "parallel_tool_calls": contract.get("parallel_tool_calls"),
        "response_format": contract.get("response_format"),
    }

    def evidence(raw_output, *, parser_input=None, reasoning_normalization="none",
                 status, finish_reason, parsed=None, error=None):
        parser = dict((parsed or {}).get("evidence") or {})
        outcome = {"status": status}
        if parsed:
            outcome["kind"] = parsed.get("kind")
            if parsed.get("kind") == "tool_call":
                outcome["tool_name"] = parsed.get("name")
                outcome["call_id"] = parsed.get("call_id")
        if error is not None:
            outcome.update(code=error.code, message=str(error))
            parser = dict(error.evidence or parser)
        normalization = parser.get("normalization", "none")
        public_finish = "tool_calls" if (parsed or {}).get("kind") == "tool_call" else "stop"
        return {
            "schema": "clozn.output_contract.v1",
            "request": request_evidence,
            "qualification": qualified,
            "raw_model_output": raw_output,
            "parser_input": parser_input,
            "parser": parser,
            "outcome": outcome,
            "recovery": {
                "policy": "one_outer_json_fence_only",
                "reasoning_normalization": reasoning_normalization,
                "json_normalization": normalization,
            },
            "substrate_finish_reason": finish_reason,
            "public_finish_reason": public_finish if status == "parsed" else None,
        }

    def process(raw_output: str, sanitized_output: str, finish_reason: str | None):
        if ctx._openai_finish_reason(finish_reason) == "length":
            exc = StructuredIOError(
                "structured model output was cut off before validation",
                code="model_output_truncated", param=None,
            )
            exc.evidence = evidence(raw_output, parser_input=None, status="error",
                                    finish_reason=finish_reason, error=exc)
            raise exc
        parser_input = None
        reasoning_normalization = "none"
        try:
            parser_input, reasoning_normalization = structured_parser_input(
                raw_output, sanitized_output
            )
            parsed = parse_output(parser_input, contract)
        except StructuredIOError as exc:
            exc.evidence = evidence(
                raw_output, parser_input=parser_input,
                reasoning_normalization=reasoning_normalization,
                status="error", finish_reason=finish_reason, error=exc,
            )
            raise
        if parsed.get("kind") == "tool_call":
            parsed = {**parsed, "call_id": "call_" + secrets.token_hex(12)}
        return {
            "parsed": parsed,
            "evidence": evidence(
                raw_output, parser_input=parser_input,
                reasoning_normalization=reasoning_normalization,
                status="parsed", finish_reason=finish_reason, parsed=parsed,
            ),
        }

    return process


def _native_runtime_pipeline(sub) -> dict:
    """Read the worker-reported atomic chat pipeline for qualification preflight."""
    try:
        engine = getattr(sub, "engine", None)
        health = engine.health() if engine and callable(getattr(engine, "health", None)) else {}
        native = health.get("native_chat_io") if isinstance(health, Mapping) else None
    except Exception:
        native = None
    if not isinstance(native, Mapping) or native.get("available") is not True:
        return {}
    return {
        key: native.get(key)
        for key in ("executor_id", "renderer_id", "grammar_id", "parser_id")
    }


def _native_generation_contract(contract: dict) -> dict:
    """Map the public v1 contract onto llama-common's atomic structured inputs."""
    mode = contract["mode"]
    if mode == "tools":
        return {
            "tools": contract.get("active_tools") or [],
            "tool_choice": "auto",
            "json_schema": None,
        }
    if mode == "json_schema":
        schema = contract["response_format"]["json_schema"]["schema"]
    else:  # json_object: constrain to one object; Python still performs strict whole-object validation.
        schema = {"type": "object"}
    return {"tools": None, "tool_choice": "none", "json_schema": schema}


def _native_structured_processor(contract: dict, qualification: dict):
    """Validate the exact native parser message and produce output-contract v2 evidence."""
    from clozn.server.structured_io import (
        NATIVE_MESSAGE_VALIDATOR_ID, StructuredIOError, qualification_evidence,
        validate_native_message,
    )

    qualified = qualification_evidence(qualification, contract["mode"], contract)
    request_evidence = {
        "mode": contract["mode"],
        "tools": contract.get("active_tools") or [],
        "tool_choice": contract.get("tool_choice"),
        "parallel_tool_calls": contract.get("parallel_tool_calls"),
        "response_format": contract.get("response_format"),
    }

    def evidence(raw_output, native, *, status, finish_reason, parsed=None, error=None):
        native = dict(native) if isinstance(native, Mapping) else {}
        parse_error = native.get("parse_error")
        native_message = native.get("message")
        outcome = {"status": status}
        if parsed:
            outcome["kind"] = parsed.get("kind")
            if parsed.get("kind") == "tool_call":
                outcome["tool_name"] = parsed.get("name")
                outcome["call_id"] = parsed.get("call_id")
        if error is not None:
            outcome.update(code=error.code, message=str(error))
        return {
            "schema": "clozn.output_contract.v2",
            "request": request_evidence,
            "qualification": qualified,
            "raw_model_output": raw_output,
            "native": {
                "model_sha256": native.get("model_sha256"),
                "pipeline": native.get("pipeline"),
                "format": native.get("format"),
                "parse_status": "error" if parse_error else "parsed",
                "parse_error": parse_error,
                "message": native_message,
                "reasoning_content": (
                    native_message.get("reasoning_content")
                    if isinstance(native_message, Mapping) else None
                ),
            },
            "validator": {
                "validator_id": NATIVE_MESSAGE_VALIDATOR_ID,
                "input": native_message,
                "result": dict(
                    (parsed or {}).get("evidence")
                    or (getattr(error, "evidence", None) if error is not None else None)
                    or {}
                ),
            },
            "outcome": outcome,
            "recovery": {
                "policy": "exact_qualified_native_parser_then_strict_validator",
                "python_repair": "none",
            },
            "substrate_finish_reason": finish_reason,
            "public_finish_reason": (
                "tool_calls" if status == "parsed" and (parsed or {}).get("kind") == "tool_call"
                else ("stop" if status == "parsed" else None)
            ),
        }

    def process(raw_output: str, native: Any, finish_reason: str | None):
        native = dict(native) if isinstance(native, Mapping) else {}
        try:
            if ctx._openai_finish_reason(finish_reason) == "length":
                raise StructuredIOError(
                    "structured model output was cut off before validation",
                    code="model_output_truncated", param=None,
                )
            if native.get("model_sha256") != qualification["model_sha256"]:
                raise StructuredIOError(
                    "worker model identity changed during structured generation",
                    code="qualification_postflight_mismatch", param=None,
                )
            runtime_pipeline = native.get("pipeline")
            expected_pipeline = {
                key: value for key, value in qualification["pipeline"].items()
                if key != "validator_id"
            }
            if runtime_pipeline != expected_pipeline:
                raise StructuredIOError(
                    "worker structured pipeline changed during generation",
                    code="qualification_postflight_mismatch", param=None,
                )
            parse_error = native.get("parse_error")
            if isinstance(parse_error, Mapping):
                raise StructuredIOError(
                    str(parse_error.get("message") or "native parser rejected model output"),
                    code=str(parse_error.get("code") or "native_parse_failed"), param=None,
                )
            parsed = validate_native_message(native.get("message"), contract)
        except StructuredIOError as exc:
            exc.evidence = evidence(
                raw_output, native, status="error", finish_reason=finish_reason, error=exc,
            )
            raise
        if parsed.get("kind") == "tool_call":
            parsed = {**parsed, "call_id": "call_" + secrets.token_hex(12)}
        return {
            "parsed": parsed,
            "evidence": evidence(
                raw_output, native, status="parsed", finish_reason=finish_reason, parsed=parsed,
            ),
        }

    return process


def _buffered_structured_sse(h, parsed: dict, *, model: str, run_id: str | None) -> None:
    """Emit protocol-valid SSE only after the complete model output has validated."""
    from clozn.server.http_policy import send_cors_headers
    from clozn.server.structured_io import openai_stream_deltas

    stream = openai_stream_deltas(parsed)
    completion_id = "chatcmpl-" + secrets.token_hex(8)
    created = int(time.time())
    h.send_response(200)
    h.send_header("Content-Type", "text/event-stream")
    h.send_header("Cache-Control", "no-cache")
    if run_id:
        h.send_header("X-Clozn-Run-Id", run_id)
    send_cors_headers(h)
    h.end_headers()

    def write(delta, finish_reason=None, extension=None):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        if extension:
            chunk.update(extension)
        h.wfile.write(("data: " + json.dumps(chunk) + "\n\n").encode("utf-8"))

    try:
        for delta in stream["deltas"]:
            write(delta)
        write({}, stream["finish_reason"], {"clozn_run_id": run_id} if run_id else None)
        h.wfile.write(b"data: [DONE]\n\n")
        h.wfile.flush()
    except OSError:
        # The buffered generation is already durably recorded.  A delivery failure
        # must not create a second, contradictory run.
        return


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
    from clozn.server.openai_compat import CompatibilityError, normalize_chat_request
    try:
        body, calibration_task = _calibration_task(body)
        body = normalize_chat_request(body)
    except CompatibilityError as exc:
        _api_error(h, 400, str(exc), param=exc.param, code=exc.code)
        return True
    except ValueError as exc:
        _api_error(h, 400, str(exc), param="clozn_task", code="invalid_parameter")
        return True
    structured = body.pop("_structured_contract", None)
    sub = ctx.active_sub(h)
    if not (sub and getattr(sub, "chat", None)):
        h._json(503, {"error": {"message": "model worker unavailable", "type": "service_unavailable"}})
        return True
    from clozn.server.generation_gateway import model_id
    selected_model = str(body.get("model") or model_id())
    msgs = body["messages"]
    mx = int(body.get("max_tokens", 256))
    temperature = float(body.get("temperature", 0.8))
    sample_request = {key: body[key] for key in ("temperature", "top_p", "top_k", "repeat_penalty", "seed")
                      if key in body and body[key] is not None}
    sample = sample_request or (temperature > 0)
    # SYMMETRY (context-contamination guard): clients echo the whole conversation back, footers and all.
    # Whatever clozn appended to a past reply, it strips here -- the model must never read its own
    # receipt footers as context (it would imitate/steer on them). No-op when no footer ever rode.
    from clozn.runs.receipt_footer import strip_footers
    msgs = strip_footers(msgs)
    from clozn.runs.think_tags import sanitize_messages
    msgs = sanitize_messages(msgs)

    # CLOSED-LOOP DISPOSITION GUARDRAILS (FRONTIER_BETS section 9.1 / experiment A1.1), opt-in, default
    # off -- see generation_guard.py's own module docstring for the full honesty framing (present-tense
    # detect-and-correct, NEVER predictive/lead-time/"acts on intent"). Parsed here, before the corrective-
    # policy/structured-output/streaming logic below, because a guarded generation bypasses ALL of that in
    # this first cut (see the module docstring's SCOPE LIMITS section) -- composing the guard with prompt-
    # card memory, tone dials, corrective retries, or structured output is deferred, not silently dropped.
    # `guard_spec` is None whenever the request/server default is off; every line below this block then
    # runs completely unchanged, so the byte-identical-when-off contract holds regardless of where the
    # guard sits in this function.
    from clozn.server import generation_guard
    try:
        guard_spec = generation_guard.parse_guard_spec(body)
    except ValueError as exc:
        _api_error(h, 400, str(exc), param="clozn_guard", code="invalid_parameter")
        return True
    if guard_spec is not None:
        if body.get("stream"):
            # STREAMING IS DEFERRED (see the module docstring): a streamed reply's early tokens are
            # already delivered before any correction could happen, so silently ignoring the guard on a
            # streaming request would be exactly the "silent pass" this feature exists to prevent --
            # refuse instead of degrading quietly.
            _api_error(
                h, 422,
                "clozn_guard is not supported together with stream:true yet -- retry without "
                "streaming (see clozn/server/generation_guard.py's module docstring)",
                param="clozn_guard", code="guard_streaming_unsupported",
            )
            return True
        guard_meta = {}
        if calibration_task is not None:
            guard_meta["clozn_task"] = calibration_task
        result = generation_guard.guarded_chat_completion(
            h, msgs, model=selected_model, max_tokens=mx, sample=sample,
            spec=guard_spec, source="openai_api", extra_meta=guard_meta or None,
        )
        if not result.get("ok", True):
            # FAIL CLOSED (see generation_guard.py's FAIL-CLOSED DECISION): a guarded concept could not be
            # resolved to a working dir(c) -- refuse the whole request rather than silently generate an
            # unguarded reply under a safety request this process cannot back.
            _api_error(h, 422, f"guard unavailable: {result.get('reason')}",
                      param="clozn_guard", code="guard_unavailable")
            return True
        resp = {
            "id": "chatcmpl-" + secrets.token_hex(8), "object": "chat.completion",
            "created": int(time.time()), "model": selected_model,
            "choices": [{"index": 0, "finish_reason": result.get("finish_reason") or "stop",
                        "message": {"role": "assistant", "content": result["reply"]}}],
            "clozn_guard_receipt": result["receipt"],
        }
        if result.get("run_id"):
            resp["clozn_run_id"] = result["run_id"]
        h._json(200, resp, extra_headers=(
            {"X-Clozn-Run-Id": result["run_id"]} if result.get("run_id") else None
        ))
        return True

    delivered_messages = msgs
    from clozn.server.generation_gateway import apply_corrective_policy
    msgs, corrective_evidence = apply_corrective_policy(h, delivered_messages)
    if structured and corrective_evidence:
        _api_error(
            h, 409,
            "active corrective response policies are prose interventions and cannot be applied "
            "to a structured-output request; undo the policy or use a different session/profile",
            param="response_format" if structured.get("mode") != "tools" else "tools",
            code="corrective_policy_inapplicable",
        )
        return True
    journal_messages = delivered_messages if corrective_evidence else None
    output_processor = None
    native_structured = None
    if structured:
        from clozn.server.structured_io import (
            normalize_and_lower_messages, require_qualification,
            resolve_qualification_registry,
        )
        try:
            if structured.get("mode"):
                active_identity = _active_identity(sub)
                qualification = require_qualification(
                    active_identity, structured["mode"],
                    runtime_pipeline=_native_runtime_pipeline(sub),
                    registry=resolve_qualification_registry(
                        active_identity,
                        artifact_root=(os.environ.get("CLOZN_ARTIFACTS_DIR") or
                                       os.path.join(ctx.CLOZN_DIR, "artifacts")),
                    ),
                )
                if not callable(getattr(sub, "_complete_chat_native", None)):
                    raise StructuredIOError(
                        "qualified worker does not expose atomic native structured chat",
                        code="model_not_qualified", param=(
                            "tools" if structured["mode"] == "tools" else "response_format"
                        ),
                    )
                plan = normalize_and_lower_messages(msgs, structured)
                # Native llama-common receives the normalized OpenAI roles directly. The lowered
                # synthetic envelope remains only for text-bypass history on the legacy chat seam.
                native_structured = _native_generation_contract(structured)
                output_processor = _native_structured_processor(structured, qualification)
            else:
                plan = normalize_and_lower_messages(msgs, structured)
            journal_messages = plan["messages"]
            msgs = plan["messages"] if native_structured is not None else plan["generation_messages"]
        except StructuredIOError as exc:
            _api_error(h, 400, str(exc), param=exc.param, code=exc.code)
            return True

    if body.get("stream") and output_processor is None and getattr(sub, "chat_stream", None):
        from clozn.server import sse
        from clozn.server.routes.receipt_link import receipt_enabled
        # F1 live lens: clozn_lens {layer?, topk?, every?} (or true) is a clozn extension -- absent for
        # standard OpenAI clients, so their streams stay byte-identical (same opt-in rule as clozn_trust).
        # receipt: the in-band footer as a final content chunk (the one ambient surface).
        sse.sse_chat(h, msgs, mx, selected_model, lens=body.get("clozn_lens"), sample=sample,
                     receipt=body.get("clozn_receipt", receipt_enabled()),
                     journal_messages=journal_messages,
                     corrective_evidence=corrective_evidence,
                     task=calibration_task)
        return True
    from clozn.server.generation_gateway import instrumented_chat
    run_meta = {}
    if corrective_evidence is not None:
        run_meta["corrective_policy"] = corrective_evidence
    if calibration_task is not None:
        run_meta["clozn_task"] = calibration_task
    try:
        generated = instrumented_chat(h, msgs, model=selected_model, max_tokens=mx,
                                      sample=sample, source="openai_api",
                                      journal_messages=journal_messages,
                                      extra_meta=run_meta or None,
                                      output_processor=output_processor,
                                      native_structured=native_structured)
    except StructuredIOError as exc:
        error = exc.as_error()
        error["type"] = "model_output_error"
        error["param"] = None
        # Full parser/qualification/raw-output evidence is durable in the run.  Keep
        # the public error compact and avoid echoing a potentially large model result.
        error.pop("evidence", None)
        rid = getattr(exc, "run_id", None)
        response = {"error": error}
        if rid:
            response["clozn_run_id"] = rid
        h._json(502, response,
                extra_headers={"X-Clozn-Run-Id": rid} if rid else None)
        return True
    except Exception as exc:
        _api_error(h, 502, str(exc), kind="upstream_error")
        return True

    if output_processor is not None:
        from clozn.server.structured_io import serialize_openai_result
        parsed = generated.structured["parsed"]
        if body.get("stream"):
            _buffered_structured_sse(
                h, parsed, model=selected_model, run_id=generated.run_id,
            )
            return True
        wire = serialize_openai_result(parsed)
        resp = {
            "id": "chatcmpl-" + secrets.token_hex(8),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": selected_model,
            "choices": [{"index": 0, **wire}],
        }
        if generated.run_id:
            resp["clozn_run_id"] = generated.run_id
        h._json(200, resp, extra_headers=(
            {"X-Clozn-Run-Id": generated.run_id} if generated.run_id else None
        ))
        return True
    reply = generated.reply
    trace_steps = generated.trace_steps
    rid = generated.run_id
    openai_fr = generated.public_finish_reason
    resp = {"id": "chatcmpl-clozn", "object": "chat.completion",
           "created": int(time.time()), "model": selected_model,
           "choices": [{"index": 0, "finish_reason": openai_fr,
                        "message": {"role": "assistant", "content": reply}}]}
    # M5 any-client run_id bridge (EXPLAIN_THIS_ANSWER_SPEC.md): surface the id two ways so a
    # companion `clozn inspect <run_id>` can inspect THIS reply from any OpenAI-compatible client
    # -- an additive top-level field (spec-compliant clients ignore unknown fields) and a response
    # header (for clients that only expose headers). Clean omission when logging failed (rid is
    # None) -- never emit a literal "null"/"None".
    extra_headers = {"X-Clozn-Run-Id": rid} if rid else None
    if rid:
        resp["clozn_run_id"] = rid
    if generated.warnings:
        resp["clozn_warnings"] = generated.warnings
        extra_headers = dict(extra_headers or {})
        extra_headers["X-Clozn-Warning"] = "output-truncated"
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
    # CALIBRATION BACKLOG #10 "ask/abstain bands": a metadata-only signal when this reply's confidence
    # lands in the saved policy's ask OR abstain band (clozn eval --save). Never opt-in-gated -- it is
    # silent (no key at all) unless a matching, fitted calibration actually says "ask" or "abstain"; see
    # generation_gateway.policy_signal.
    from clozn.server.generation_gateway import policy_verdict_and_signal
    # ONE _policy_verdict lookup feeds both the live ask/abstain-only signal below and the all-bands
    # receipt persisted onto the run record -- policy_verdict_and_signal's own docstring explains why
    # this must not be two separate calls.
    policy, policy_meta = policy_verdict_and_signal(trace_steps, selected_model, task=calibration_task)
    if policy:
        resp["clozn_policy"] = policy
    # Persist the ALL-BANDS verdict (including 'answer', which the live signal above never surfaces)
    # onto the run record, so a later reader of THIS run can tell whether the policy would have
    # answered, asked, or abstained -- not just live callers. After the fact because the run must
    # already be logged (rid assigned above) before its own trace can be scored. Additive and
    # best-effort: any hiccup here must never break the reply.
    if rid and policy_meta:
        try:
            from clozn.runs.attachments import attach_policy_verdict
            attach_policy_verdict(rid, policy_meta)
        except Exception:
            pass
    # CALIBRATION BACKLOG #10 action half (BK decision 2026-07-22: abstain/ask may become an ACTION, but
    # OPT-IN, DEFAULT OFF). `clozn_selective` on the request (or the server-wide `selective_generation`
    # setting when the request omits the field) opts in; absent/false on both keeps this response
    # BYTE-IDENTICAL to the annotate-only behavior above -- see
    # generation_gateway.selective_generation_action's docstring for the full fail-closed contract (no
    # calibrated profile -> the action never fires even when opted in, and says why).
    from clozn.server.generation_gateway import selective_generation_action, selective_generation_enabled
    if selective_generation_enabled(body):
        action = selective_generation_action(reply, trace_steps, selected_model, calibration_task,
                                             opted_in=True)
        resp["clozn_selective_action"] = action
        if action.get("applied"):
            resp["choices"][0]["message"]["content"] = action["reply"]
    h._json(200, resp, extra_headers=extra_headers)
    return True
