# OpenAI client compatibility

**Status (2026-07-31):** Clozn implements a deliberately small, strict subset of the OpenAI HTTP API.
This is an endpoint/field contract, not a claim of full platform compatibility. Behavior-bearing fields
that Clozn cannot honor return an OpenAI-shaped HTTP 400 instead of being silently ignored.

The field inventory was checked against OpenAI's official
[Chat Completions](https://developers.openai.com/api/reference/resources/chat/methods/create) reference.
OpenAI recommends the Responses API for new platform integrations; Clozn implements only the narrow
non-streaming text subset documented below.

## Endpoint matrix

| Method and path | Status | Notes |
|---|---|---|
| `GET /v1/models` | supported | one currently loaded local model |
| `POST /v1/chat/completions` | supported subset | text, plus fail-closed qualified tools/structured output; one choice; streaming or non-streaming |
| `POST /v1/completions` | retired | returns HTTP 410 with code `endpoint_retired`; use Chat Completions |
| `POST /v1/responses` | supported subset | non-streaming text `input` with optional `instructions`; uses the same instrumented chat path |
| embeddings, audio, images, files, batches, fine-tuning | unsupported | no routes |
| stored chat list/get/update/delete | unsupported | Clozn's local run journal is a different API |

The native instrumented stream is `POST /api/clozn/generate`; it is intentionally outside `/v1` so
Clozn state events never leak into a standard OpenAI stream.

## Chat Completions request fields

| Field | Status | Exact behavior |
|---|---|---|
| `model` | supported | labels the response/run; the gateway still serves its one loaded worker |
| `messages` | supported subset | non-empty text `{role, content}` objects; roles `system`, `user`, `assistant`; `developer` is normalized to `system`. Qualified tool requests additionally accept one assistant tool call followed by a matching `tool` result message. |
| `max_tokens` | supported | positive integer |
| `max_completion_tokens` | supported alias | normalized to `max_tokens`; sending both is a 400 |
| `stream` | supported | boolean; standard `chat.completion.chunk` SSE + `[DONE]` |
| `stop` | supported subset | one string or 1--4 strings; native worker termination and cross-token matching |
| `stream_options` | supported subset | only `include_usage` boolean; `true` adds an exact terminal usage chunk |
| `temperature`, `top_p`, `seed` | supported | forwarded into the request's sampler; explicit fields override Studio's persisted sampling default |
| `n` | one only | `1`/null accepted and stripped; any other value is a 400 |
| `top_k`, `repeat_penalty` | Clozn extensions | forwarded to the engine sampler |
| `clozn_trust`, `clozn_receipt`, `clozn_receipt_mode`, `clozn_lens` | Clozn extensions | opt-in confidence spans, receipt delivery (`off`, `exceptions`, or `always`), and live J-lens readout |
| `clozn_sources` | Clozn extension | optional list of `{message_index, source_id, label?}` records; IDs must be unique and are carried unchanged into Context Receipt/source evidence |
| `tools` | qualified subset | up to 32 strict function definitions; Clozn returns at most one call and never executes it |
| `tool_choice` | qualified subset | `"auto"` activates the one-tool contract; `"none"` is an explicit text bypass |
| `parallel_tool_calls` | false for active tools | omitted/`false` accepted; `true` is rejected when the tool contract is active |
| `response_format` | qualified subset | `{"type":"json_object"}` or a restricted `strict:true` `json_schema`; `{"type":"text"}` is neutral |

Message content arrays (images/audio/files), message `name`, deprecated `functions`/`function_call`, multiple
calls in one assistant turn, and other unlisted message fields are rejected. Valid single-call tool history is
passed to the native llama-common template renderer after gateway validation. A continuation may omit current
`tools` to request an ordinary final text answer; historical arguments are schema-checked when a matching current
definition exists.
Sampler fields omitted by the client use Clozn's persisted interactive defaults (initially temperature 0.8,
top-p 0.9, top-k 40, repetition penalty 1.1); fields explicitly present in the request override them.

These fields are accepted **only at a neutral value**, removed before generation, and documented here as
ignored for client interoperability:

| Field | Accepted neutral values |
|---|---|
| `user` | string or null |
| `frequency_penalty`, `presence_penalty` | `0` or null |
| `logprobs` | `false` or null |
| `top_logprobs` | `0` or null |
| `audio`, `prediction` | null only |
| `logit_bias`, `metadata` | empty object or null |
| deprecated `functions` | empty list or null |
| deprecated `function_call` | `"none"` or null |
| `modalities` | `["text"]` or null |
| `store` | `false` or null |
| `service_tier` | `"auto"`, `"default"`, or null |
| `stream_options` | other fields are rejected; `include_usage` is supported |

Any behavior-bearing value outside the supported subset—nonzero frequency/presence penalties, multiple
choices, and so on—is a 400 with that field in `error.param`.
Unknown top-level fields are also rejected.

`clozn_sources` is journal metadata only: it does not alter the standard message list or prompt text.
It is refused on request modes that cannot preserve the identity contract (currently structured-I/O
qualification and guard-v1 transforms), rather than being silently dropped.

## Responses subset

`POST /v1/responses` accepts `model`, optional string `instructions`, required string `input` (and
optionally a text-only message array), `temperature`, `top_p`, and `max_output_tokens`. It is
`stream:false` only. Multimodal input, tools, and unknown behavior-bearing fields fail closed. The request
is normalized into the same text-chat messages and instrumented substrate used by Chat Completions, so it
creates exactly one ordinary Clozn run and uses the worker's authoritative usage counts.

## Qualified structured I/O (Phase 2.8)

Structured I/O is fail-closed and disabled by default. `CLOZN_STRUCTURED_IO_QUALIFICATIONS` may point to
an explicit qualification registry v2. Each entry binds the exact active `model_sha256`, exact
`template_fingerprint`, enabled feature (`tools`, `json_object`, and/or `json_schema`), schema-subset ID,
all four native worker IDs (atomic executor, llama-common renderer, AR grammar, and llama-common parser),
the public native-message validator ID, and passing suite evidence. The request's `model` string is only a
response label and cannot qualify or spoof the loaded substrate. Runtime-reported pipeline drift also fails
closed. Clozn ships no prequalified real-model entry at this stage.

The public contract is intentionally narrow:

- Up to 32 strict function definitions and at most one returned call. `tool_choice` supports `"auto"` and
  `"none"`; tools and a non-text `response_format` are mutually exclusive. Clozn serializes the call for
  the client but never executes it. Active structured I/O cannot be combined with `clozn_trust`,
  `clozn_receipt`, `clozn_receipt_mode`, or `clozn_lens` in this slice.
- A following request may carry the assistant's single `tool_calls` item and one immediately matching
  `role:"tool"` result. These are validated and sent as message history to the native template renderer
  without changing the journal's original OpenAI message history; the continuation may omit `tools` when it
  only wants final text.
- `json_object` requires one JSON object. `json_schema` requires `strict:true` and a root object using
  Clozn's bounded subset: scalar `type`, object properties/required keys with
  `additionalProperties:false`, arrays with typed `items`, length/item bounds, numeric bounds,
  descriptions, and same-type enums. References, composition, regexes, formats, defaults, and conditional
  schemas are rejected rather than approximated.
- The worker performs one buffered, atomic render → AR grammar-constrained generate → parse operation. It
  retains llama-common's grammar, trigger, stop, reasoning-tag, and parser state across the operation; lazy
  tool grammars are suspended only while a recognized reasoning block is active. Client-held prepared
  descriptors are not accepted for generation.
- Llama-common owns template-specific parsing into an assistant message and optional `reasoning_content`.
  The gateway then validates that native message, tool name/arguments, and schema without parsing or repairing
  model syntax. Think tags cannot be stripped to repair a tool name or argument, and native call IDs are not
  trusted for public association. Native parse failure, schema-invalid output, or length truncation becomes a
  typed HTTP 502 and one errored journal run; generation is not silently retried.
- Structured streaming is buffer-then-validate, not token-live. No model-derived SSE bytes are committed
  until parsing succeeds. Tool calls are then emitted as standard indexed `tool_calls` deltas with terminal
  `finish_reason:"tool_calls"`, the additive `clozn_run_id`, and `[DONE]`.

Every structured request that reaches generation records the raw model output, native parser message or error,
strict validator input/result, normalized request contract, exact qualification and pipeline, public call
ID/finish decision, and outcome in the run's `clozn.output_contract.v2` evidence. A validated result is not
returned unless this evidence persists; a journal failure is a typed 502. An unqualified request fails before
generation and creates no run.

This is a model-free-tested native/gateway contract, not a broad local-model compatibility claim. Exact
qualification remains opt-in through the existing registry/artifact path. `clozn serve --structured auto`
keeps the current fail-closed behavior, `--structured off` rejects active structured requests, and
`--structured required` refuses startup unless all three supported structured modes are exactly qualified.
The bounded real-model qualification battery and a successful Open WebUI two-request tool loop remain
separate acceptance work.

## Retired legacy Completions

The public `POST /v1/completions` route was retired on 2026-07-31. It returns a stable HTTP 410 response:

```json
{
  "error": {
    "message": "POST /v1/completions was retired; use POST /v1/chat/completions instead",
    "type": "invalid_request_error",
    "param": null,
    "code": "endpoint_retired"
  }
}
```

This retirement applies only to the public Python gateway. The private C++ worker still uses its internal
`/v1/completions` protocol underneath Chat Completions, native generation, and offline research tools; that
loopback-only protocol is not part of the public compatibility surface.

## Response boundary

- Chat response objects/chunks use the standard object names and one choice at index 0.
- Every accepted Chat Completions request crosses Clozn's instrumented substrate and is written to the local
  run journal with its delivered messages, rendered prompt, active dials/corrective policy, trace, and
  finish/failure state.
- Non-streaming chat responses may add `clozn_run_id` and `X-Clozn-Run-Id`; opt-in Clozn fields are additive.
- Token usage is omitted when unknown. Clozn no longer fabricates zero prompt/completion token counts.
- When available, usage comes from the executing worker's final rendered prompt and decoder accounting;
  the gateway never re-tokenizes messages. `include_usage:true` is emitted as a final choices-empty SSE
  chunk before `[DONE]`.
- Finish reasons map worker EOS to `stop` and token limits to `length`. A worker failure is an error, never a
  successful `stop`.
- A proven `length` stop also adds `clozn_warnings: [{"code":"output_truncated", ...}]` to non-stream
  responses and the terminal stream chunk. Non-stream responses additionally carry
  `X-Clozn-Warning: output-truncated`; this warns that the reply may be incomplete and does not claim the
  input prompt was truncated (overlong prompts are rejected).
- Token-live chat streams cannot return a run id in headers because the run is persisted after headers are
  sent. They are still finalized as one journal run after the stream ends. A buffered
  structured stream is validated and journaled first, so it can also return `X-Clozn-Run-Id`.
- Model-emitted `<think>...</think>` scratch text is excluded from `message.content`, streaming deltas,
  echoed assistant history, and the public token trace. Prompt-prefilled and
  unclosed think blocks are handled without leaking partial reasoning. The stripped text remains local in
  the run's versioned reasoning evidence for Replay inspection; it is not returned on the OpenAI wire.

Unqualified structured request example:

```json
{
  "error": {
    "message": "exact loaded model/template identity is not in the structured-I/O qualification registry",
    "type": "invalid_request_error",
    "param": "tools",
    "code": "model_not_qualified"
  }
}
```

## Test evidence

- `tests/test_openai_compat.py` exercises the field table without a model or network.
- `tests/test_openai_client_compat.py` starts the real gateway with a fake substrate and drives model list,
  non-streaming chat, streaming chat, and a typed 400 through the real `openai` Python package.
- `tests/test_structured_io.py` exercises the strict request contract, qualification registry v2, exact native
  pipeline matching, native-message validator, schema validator, typed errors, and model-free Python-envelope
  harness without qualifying that harness for public use.
- `tests/test_engine_chat_io.py` and `tests/test_engine_substrate_native_chat.py` exercise the private atomic
  request/response contract, pipeline and model evidence, native parse failures, steering layers, and
  trace folding without a model.
- `tests/test_openai_structured_client_compat.py` drives the real gateway through OpenAI Python 2.46.0
  against a fake atomic native result with exact identity and pipeline: one tool call, tool-result continuation,
  buffered tool SSE, `json_object`, unqualified rejection, native parse failure, and malformed-message
  journaling. It does not qualify a real model.
- `engine/core/tests/test_chat_template_renderer.cpp` model-free-tests llama-common rendering, native parsing,
  grammar extraction, tool history, JSON schema, and fail-closed structured request validation;
  `engine/core/tests/test_generate.cpp` covers AR grammar ownership and reasoning-block gating.
- CI pins `openai==2.46.0` in the CPU Python lane, so the SDK integration test cannot silently skip there.
- `tests/test_runtime_architecture.py` and `tests/test_product_smoke.py` guard the standard-vs-native stream
  envelope boundary.
- `tests/test_model_routing_gateway.py` verifies the retired legacy route returns the typed 410 migration
  response without selecting a model, loading a worker, or creating a run.
- `tests/test_think_tags.py` plus the OpenAI/Ollama streaming integration tests prove chunk-split think tags
  never enter public answer/history content while their journal evidence remains inspectable.

## Run association

Non-streaming Chat Completions responses expose `clozn_run_id` in the body and `X-Clozn-Run-Id` in headers.
Token-live chat streams cannot know the finalized journal ID before their headers are committed, so the
ordinary terminal OpenAI chat chunk carries additive `clozn_run_id` before `[DONE]`; Ollama NDJSON does the same on its
`done: true` object. Buffered structured streams finish and journal generation before committing SSE, so
they return both the header and the terminal extension.

Clients that need a side-channel can send `X-Clozn-Client-Id` and/or `X-Clozn-Session-Id` (1–128 visible
ASCII characters). Raw values are never journaled: Clozn stores install-local HMAC fingerprints, excludes
them from portable receipt exports, and supports exact `GET /runs/latest` lookup. `GET /runs/watch` and
`clozn watch` use insertion-order cursors rather than generation start time, so a slow overlapping request
cannot be skipped when it finishes late.

The released-client versions, exact exercised surfaces, expected rejections, and Open WebUI status are
published in [CLIENT_CONFORMANCE.md](CLIENT_CONFORMANCE.md). A protocol-shaped request is never counted as
a released-client pass; the matrix labels source audits and scheduled-but-not-yet-green lanes separately.
