"""The v1 public text API: model listing and OpenAI Chat Completions."""
import time

from clozn.server import app as ctx


def _api_error(h, status: int, message: str, *, param=None, kind="invalid_request_error", code=None):
    error = {"message": message, "type": kind, "param": param, "code": code or status}
    h._json(status, {"error": error})


def try_get(h, p):
    if p == "/v1/models":            # OpenAI-compatible model list (so OAI clients connect)
        router = getattr(ctx, "MODEL_ROUTER", None)
        if router is None:
            _api_error(h, 503, "model router unavailable", code="service_unavailable")
            return True
        h._json(200, {"object": "list", "data": [
            {"id": model_id, "object": "model", "owned_by": "clozn"}
            for model_id in router.model_ids()
        ]})
        return True
    return False


def try_post(h, p, body):
    if p != "/v1/chat/completions":
        return False
    from clozn.server.openai_compat import (
        CompatibilityError, normalize_chat_request,
    )
    try:
        body = normalize_chat_request(body)
    except CompatibilityError as exc:
        _api_error(h, 400, str(exc), param=exc.param, code=exc.code)
        return True
    from clozn.server.model_routing import select_for_handler
    selection = select_for_handler(
        h, body, surface="openai", route="/v1/chat/completions"
    )
    if selection is None:
        return True
    sub = selection.sub
    if not (sub and getattr(sub, "chat", None)):
        h._json(503, {"error": {"message": "model worker unavailable", "type": "service_unavailable"}})
        return True
    selected_model = selection.model_id
    msgs = body["messages"]
    mx = int(body.get("max_tokens", 256))
    temperature = float(body.get("temperature", 0.8))
    sample_request = {key: body[key] for key in ("temperature", "top_p", "top_k", "repeat_penalty", "seed")
                      if key in body and body[key] is not None}
    sample = sample_request or (temperature > 0)
    from clozn.runs.think_tags import sanitize_messages
    msgs = sanitize_messages(msgs)

    if body.get("stream") and getattr(sub, "chat_stream", None):
        from clozn.server import sse
        sse.sse_chat(h, msgs, mx, selected_model, sample=sample,
                     stop=body.get("stop"), include_usage=bool((body.get("stream_options") or {}).get("include_usage")))
        return True
    from clozn.server.generation_gateway import instrumented_chat
    try:
        generated = instrumented_chat(h, msgs, model=selected_model, max_tokens=mx,
                                      sample=sample, source="openai_api",
                                      stop=body.get("stop"))
    except Exception as exc:
        from clozn.server.generation_gateway import RunRecordingError
        if isinstance(exc, RunRecordingError):
            _api_error(h, 502, str(exc), kind="upstream_error", code=exc.code)
            return True
        _api_error(h, 502, str(exc), kind="upstream_error")
        return True

    reply = generated.reply
    rid = generated.run_id
    openai_fr = generated.public_finish_reason
    resp = {"id": "chatcmpl-clozn", "object": "chat.completion",
           "created": int(time.time()), "model": selected_model,
           "choices": [{"index": 0, "finish_reason": openai_fr,
                        "message": {"role": "assistant", "content": reply}}]}
    # Surface the durable run id both as an additive response field and as a header so Studio and API
    # clients can reopen the recorded reply. Never emit a literal "null"/"None".
    extra_headers = {"X-Clozn-Run-Id": rid} if rid else None
    if rid:
        resp["clozn_run_id"] = rid
    if generated.usage:
        resp["usage"] = generated.usage
    if generated.warnings:
        resp["clozn_warnings"] = generated.warnings
        extra_headers = dict(extra_headers or {})
        extra_headers["X-Clozn-Warning"] = "output-truncated"
    h._json(200, resp, extra_headers=extra_headers)
    return True
