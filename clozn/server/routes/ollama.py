"""Thin Ollama API compatibility shim -- NOT a reimplementation of Ollama.

Covers just enough of Ollama's HTTP surface (`GET /api/tags`, `GET /api/version`,
`POST /api/generate`, `POST /api/chat`) that a client built against the Ollama wire
protocol (e.g. the `ollama` Python/JS client libraries, or tools that hardcode
`http://localhost:11434`) can point at clozn's engine instead and get back
Ollama-shaped JSON. There is no model pull/push, no multi-model registry, no
embeddings endpoint.

This module is a DROP-IN convenience layered on the same instrumented substrate the
OpenAI-compatible chat route uses. The wire serializers remain independent because
Ollama's and OpenAI's request/response shapes only coincidentally overlap, but both
cross generation_gateway.instrumented_chat (non-streaming) or Substrate.chat_stream via
clozn.server.ndjson (streaming) before a response is shaped -- ONE instrumented generation
path either way (Gate-0 §3.1). `GET /api/version` always answers "0.0.0-clozn" -- a fixed,
honest string so a client (or a human) can tell at a glance this is clozn wearing an
Ollama-shaped socket, not real Ollama.

STREAMING (roadmap PRODUCT_ROADMAP.md §5.1): `stream: true` on `/api/chat` or `/api/generate`
now returns Ollama-shaped NDJSON (Content-Type `application/x-ndjson`, one JSON object per
line) via clozn.server.ndjson.ndjson_stream -- the NDJSON twin of clozn.server.sse's SSE
stream, over the exact same Substrate.chat_stream() generation seam. DEFAULT-STREAM
SEMANTICS match upstream Ollama: a request that OMITS `stream` entirely streams, exactly
like real Ollama; only an explicit `stream: false` takes the non-streaming path below. (The
substrate must also expose `chat_stream`; a substrate that only implements `chat` -- a lab/
test double -- degrades to one non-streaming JSON response even for a stream-requesting
client, the same fallback clozn.server.routes.openai's SSE branch already uses.)

GATE-0 / EXPLICIT-OR-REJECTED FIELDS (roadmap §5.2): behavior-bearing Ollama fields clozn
cannot honor are REJECTED with a named, typed HTTP 400 rather than silently ignored --
`raw`, `template`, `format`, `suffix`, `context`, `think`, `images` (each accepted only at
its neutral/absent value: `false`/`None` for the booleans, `None`/empty-string/empty-list
for the rest -- a client library that echoes its own defaults must not be punished for
values that carry no actual request). `options` keys that map cleanly onto the instrumented
gateway's own sampling contract are accepted (`temperature`, `top_p`, `top_k`,
`repeat_penalty`, `seed`, `num_predict` -> `max_tokens`); any other `options` key is rejected
the same way, including `stop` -- the instrumented gateway has no stop-sequence support to
forward it to (see clozn/server/openai_compat.py's own `stop` policy on the strict OpenAI
surface). `keep_alive` is accepted and silently ignored: clozn's engine is one always-
resident worker process with no per-request unload/reload lifecycle, so no value of
`keep_alive` can change behavior either way -- it is not a "silently ignored" behavior-bearing
field in the Gate-0 sense, it is a field with nothing to opt into on this runtime.

`clozn.server.app` is imported inside each function body (not at module level) to
avoid circular imports at module load time.
"""
import os
import time


def _iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class OllamaCompatibilityError(ValueError):
    """A request error the shim can serialize as an Ollama-shaped `{"error": "..."}` 400. `field` names
    exactly what the caller sent that clozn cannot honor, always embedded in the message text too (real
    Ollama error bodies are a flat string, not a structured object -- unlike the OpenAI shim's nested
    `error.param`, so the field name has nowhere else to ride)."""

    def __init__(self, field: str, message: str):
        super().__init__(message)
        self.field = field


# ---- Gate-0 field policy (roadmap §5.2) -----------------------------------------------------------
# Each predicate answers "is this value NEUTRAL" (i.e. carries no actual request) for its field -- a
# behavior-bearing value fails with the field named in the 400. Modeled on
# clozn/server/openai_compat.py's CHAT_NEUTRAL_FIELDS table (same idiom, distinct field set: Ollama's
# wire shape is not OpenAI's).
def _neutral_false_or_none(v) -> bool:
    return v is None or v is False


def _neutral_blank_or_none(v) -> bool:
    return v is None or v == ""


def _neutral_empty_list_or_none(v) -> bool:
    return v is None or (isinstance(v, list) and not v)


_REJECTED_FIELD_NEUTRAL = {
    "raw": _neutral_false_or_none,             # raw:true skips clozn's chat-template rendering entirely
    "template": _neutral_blank_or_none,        # a caller-supplied template override string
    "format": lambda v: v is None,             # structured/JSON-schema output (no neutral non-null value)
    "suffix": _neutral_blank_or_none,          # fill-in-the-middle continuation
    "context": _neutral_empty_list_or_none,    # /api/generate raw-mode continuation token array
    "think": _neutral_false_or_none,           # explicit thinking-mode control (roadmap §5.5, not built)
    "images": _neutral_empty_list_or_none,     # vision input
}

# options{} keys the instrumented gateway can actually honor, and how they map onto its own sampling
# contract (see generation_gateway.instrumented_chat -> Substrate.chat/chat_stream's `sample` param,
# which accepts exactly temperature/top_p/top_k/repeat_penalty/seed -- the same set
# clozn/server/openai_compat.py's CHAT_SUPPORTED_FIELDS forwards for the OpenAI surface).
_MAPPED_SAMPLE_OPTION_KEYS = ("temperature", "top_p", "top_k", "repeat_penalty", "seed")


def _validate_request(body: dict) -> None:
    """Raise OllamaCompatibilityError for any top-level Gate-0 violation. `options` is validated
    separately (by _generation_options) since that pass also computes max_tokens/sample."""
    if "stream" in body and not isinstance(body["stream"], bool):
        raise OllamaCompatibilityError("stream", "'stream' must be a boolean")
    for field, is_neutral in _REJECTED_FIELD_NEUTRAL.items():
        if field in body and not is_neutral(body[field]):
            raise OllamaCompatibilityError(
                field, f"clozn does not support Ollama's '{field}' field yet "
                       "(see docs/PRODUCT_ROADMAP.md Phase 2 item 2)")


def _generation_options(options) -> tuple[int, dict | bool]:
    """Map Ollama's `options{}` to EngineSubstrate.chat/chat_stream's sampling contract. Raises
    OllamaCompatibilityError for any key that isn't in _MAPPED_SAMPLE_OPTION_KEYS plus `num_predict` --
    an unknown key is silent data loss otherwise (Gate-0 §5.2), and `stop` in particular has nowhere to
    go: the instrumented gateway has no stop-sequence support on any surface today."""
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise OllamaCompatibilityError("options", "'options' must be an object")
    for key in options:
        if key not in _MAPPED_SAMPLE_OPTION_KEYS and key != "num_predict":
            field = f"options.{key}"
            raise OllamaCompatibilityError(
                field, f"clozn does not support Ollama's '{field}' option yet "
                       "(see docs/PRODUCT_ROADMAP.md Phase 2 item 2)")
    max_tokens = 256
    if options.get("num_predict") is not None:
        try:
            n = int(options["num_predict"])
        except (TypeError, ValueError):
            n = None
        if n is not None and n > 0:
            max_tokens = n
    sample = {}
    for key in _MAPPED_SAMPLE_OPTION_KEYS:
        value = options.get(key)
        if value is not None:
            sample[key] = value
    # No explicit override means "use Clozn's configured interactive sampling",
    # which is the same contract the OpenAI chat route uses.
    return max_tokens, sample or True


def _receipt_fields(result) -> tuple[dict, dict | None]:
    fields = {}
    headers = {}
    if result.run_id:
        fields["clozn_run_id"] = result.run_id
        headers["X-Clozn-Run-Id"] = result.run_id
    if result.warnings:
        fields["clozn_warnings"] = result.warnings
        headers["X-Clozn-Warning"] = "output-truncated"
    return fields, headers or None


def _thinking_text(result) -> str | None:
    """Ollama has a separate wire field for thinking-capable models; never mix it into content."""
    blocks = (getattr(result, "reasoning", None) or {}).get("blocks") or []
    if not blocks:
        return None
    return "".join(str(block.get("text") or "") for block in blocks if isinstance(block, dict))


def try_get(h, p):
    if p == "/api/tags":
        import clozn.server.app as ctx
        if ctx.MODEL_ROUTER is not None:
            models = []
            for item in ctx.MODEL_ROUTER.catalog():
                entry = {
                    "name": item["model_id"],
                    "model": item["model_id"],
                    "size": 0,
                    "modified_at": _iso_now(),
                    "digest": f"sha256:{item['artifact_sha256']}",
                    "clozn_state": item["state"],
                }
                models.append(entry)
            h._json(200, {"models": models})
            return True
        models = []
        if ctx.ENGINE is not None:
            try:
                info = ctx.ENGINE.health() or {}
                model_path = str(info.get("model") or "")
                name = os.path.basename(model_path).removesuffix(".gguf") or "clozn-local"
                entry = {"name": name, "model": name, "size": 0, "modified_at": _iso_now()}
                # Gate-0 (roadmap §5.2): no placeholder digest. Prefer the active substrate's own
                # already-resolved+cached model_sha256 (EngineSubstrate._resolve_identity, computed once
                # at boot from this SAME /health round trip -- zero marginal cost here); fall back to the
                # raw health field when the active substrate doesn't expose one (e.g. a bare engine
                # client with no EngineSubstrate wrapper). Omitted entirely -- never an empty-string
                # placeholder -- when neither source has it (engine down/old, or unresolved so far).
                sub = ctx.active_sub(h)
                digest = getattr(sub, "model_sha256", None) or info.get("model_sha256")
                if digest:
                    entry["digest"] = f"sha256:{digest}"
                models.append(entry)
            except Exception:
                pass    # no reachable engine -- an empty list is the honest answer, not an error
        h._json(200, {"models": models})
        return True
    if p == "/api/version":
        h._json(200, {"version": "0.0.0-clozn"})   # deliberately not a real Ollama version string
        return True
    return False


def _stream_wanted(body: dict) -> bool:
    """DEFAULT-STREAM SEMANTICS: upstream Ollama streams unless the caller explicitly opts out with
    `stream: false` -- https://docs.ollama.com/api/streaming. `_validate_request` already rejected any
    non-boolean `stream` value, so the only falsy case left here is the literal `False`."""
    return body.get("stream") is not False


def try_post(h, p, body):
    if p == "/api/generate":
        import clozn.server.app as ctx
        try:
            _validate_request(body)
            max_tokens, sample = _generation_options(body.get("options"))
        except OllamaCompatibilityError as exc:
            h._json(400, {"error": str(exc)})
            return True
        from clozn.server.model_routing import select_for_handler
        selection = select_for_handler(
            h, body, surface="ollama", route="/api/generate"
        )
        if selection is None:
            return True
        sub = ctx.active_sub(h)
        if not (sub and getattr(sub, "chat", None)):
            h._json(502, {"error": "no engine configured"})
            return True
        model = selection.model_id
        prompt = str(body.get("prompt", ""))
        messages = []
        if body.get("system") is not None:
            messages.append({"role": "system", "content": str(body.get("system") or "")})
        messages.append({"role": "user", "content": prompt})
        journal_messages = messages
        from clozn.server.generation_gateway import apply_corrective_policy, apply_scoped_corrections
        messages, corrective_evidence = apply_corrective_policy(h, journal_messages)
        try:
            messages, scoped_correction_evidence = apply_scoped_corrections(h, messages)
        except Exception as exc:
            from clozn.runs.corrections import CorrectionError
            if isinstance(exc, CorrectionError):
                h._json(409, {"error": str(exc), "code": "correction_unavailable"})
                return True
            raise
        if _stream_wanted(body) and getattr(sub, "chat_stream", None):
            from clozn.server import ndjson
            ndjson.ndjson_stream(h, messages, max_tokens, model, operation="generate", sample=sample,
                                 journal_messages=journal_messages,
                                 corrective_evidence=corrective_evidence)
            return True
        from clozn.server.generation_gateway import instrumented_chat
        try:
            generated = instrumented_chat(
                h, messages, model=model, max_tokens=max_tokens, sample=sample,
                source="ollama_api",
                extra_meta={"compatibility_api": "ollama", "ollama_operation": "generate",
                            "corrective_policy": corrective_evidence,
                            "scoped_corrections": scoped_correction_evidence},
                journal_messages=journal_messages,
            )
        except Exception as e:
            h._json(502, {"error": f"engine: {e}"})
            return True
        receipt, headers = _receipt_fields(generated)
        thinking = _thinking_text(generated)
        h._json(200, {"model": model, "response": generated.reply,
                      **({"thinking": thinking} if thinking is not None else {}), "done": True,
                      **({"done_reason": generated.public_finish_reason} if generated.warnings else {}),
                      **receipt}, extra_headers=headers)
        return True
    if p == "/api/chat":
        import clozn.server.app as ctx
        try:
            _validate_request(body)
            max_tokens, sample = _generation_options(body.get("options"))
        except OllamaCompatibilityError as exc:
            h._json(400, {"error": str(exc)})
            return True
        from clozn.server.model_routing import select_for_handler
        selection = select_for_handler(
            h, body, surface="ollama", route="/api/chat"
        )
        if selection is None:
            return True
        sub = ctx.active_sub(h)
        if not (sub and getattr(sub, "chat", None)):
            h._json(502, {"error": "no engine configured"})
            return True
        model = selection.model_id
        messages = body.get("messages") or []
        from clozn.runs.receipt_footer import strip_footers
        messages = strip_footers(messages)
        from clozn.runs.think_tags import sanitize_messages
        messages = sanitize_messages(messages)
        journal_messages = messages
        from clozn.server.generation_gateway import apply_corrective_policy, apply_scoped_corrections
        messages, corrective_evidence = apply_corrective_policy(h, journal_messages)
        try:
            messages, scoped_correction_evidence = apply_scoped_corrections(h, messages)
        except Exception as exc:
            from clozn.runs.corrections import CorrectionError
            if isinstance(exc, CorrectionError):
                h._json(409, {"error": str(exc), "code": "correction_unavailable"})
                return True
            raise
        if _stream_wanted(body) and getattr(sub, "chat_stream", None):
            from clozn.server import ndjson
            ndjson.ndjson_stream(h, messages, max_tokens, model, operation="chat", sample=sample,
                                 journal_messages=journal_messages,
                                 corrective_evidence=corrective_evidence)
            return True
        from clozn.server.generation_gateway import instrumented_chat
        try:
            generated = instrumented_chat(
                h, messages, model=model, max_tokens=max_tokens, sample=sample,
                source="ollama_api",
                extra_meta={"compatibility_api": "ollama", "ollama_operation": "chat",
                            "corrective_policy": corrective_evidence,
                            "scoped_corrections": scoped_correction_evidence},
                journal_messages=journal_messages,
            )
        except Exception as e:
            h._json(502, {"error": f"engine: {e}"})
            return True
        receipt, headers = _receipt_fields(generated)
        thinking = _thinking_text(generated)
        message = {"role": "assistant", "content": generated.reply}
        if thinking is not None:
            message["thinking"] = thinking
        h._json(200, {"model": model,
                      "message": message,
                      "done": True,
                      **({"done_reason": generated.public_finish_reason} if generated.warnings else {}),
                      **receipt}, extra_headers=headers)
        return True
    return False
