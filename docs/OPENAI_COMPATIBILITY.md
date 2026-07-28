# OpenAI client compatibility

**Status (2026-07-21):** Clozn implements a deliberately small, strict subset of the OpenAI HTTP API.
This is an endpoint/field contract, not a claim of full platform compatibility. Behavior-bearing fields
that Clozn cannot honor return an OpenAI-shaped HTTP 400 instead of being silently ignored.

The field inventory was checked against OpenAI's official
[Chat Completions](https://developers.openai.com/api/reference/resources/chat/methods/create) and
[legacy Completions](https://developers.openai.com/api/reference/resources/completions/methods/create)
references. OpenAI recommends the Responses API for new platform integrations; Clozn does not currently
implement `/v1/responses`.

## Endpoint matrix

| Method and path | Status | Notes |
|---|---|---|
| `GET /v1/models` | supported | one currently loaded local model |
| `POST /v1/chat/completions` | supported subset | text, plus fail-closed qualified tools/structured output; one choice; streaming or non-streaming |
| `POST /v1/completions` | supported subset | legacy text prompt, one choice, streaming or non-streaming; exact raw prompt and token evidence are journaled |
| `POST /v1/responses` | unsupported | returns the normal route-level 404 |
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
| `temperature`, `top_p`, `seed` | supported | forwarded into the request's sampler; explicit fields override Studio's persisted sampling default |
| `n` | one only | `1`/null accepted and stripped; any other value is a 400 |
| `top_k`, `repeat_penalty` | Clozn extensions | forwarded to the engine sampler |
| `clozn_trust`, `clozn_receipt`, `clozn_lens` | Clozn extensions | opt-in confidence spans, receipt delivery, and live J-lens readout |
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
| `stop`, `audio`, `prediction` | null only |
| `logit_bias`, `metadata` | empty object or null |
| deprecated `functions` | empty list or null |
| deprecated `function_call` | `"none"` or null |
| `modalities` | `["text"]` or null |
| `store` | `false` or null |
| `service_tier` | `"auto"`, `"default"`, or null |
| `stream_options` | `{"include_usage":false}` or null |

Any behavior-bearing value outside the supported subset—stop sequences, nonzero frequency/presence
penalties, multiple choices, requested stream usage, and so on—is a 400 with that field in `error.param`.
Unknown top-level fields are also rejected.

`clozn_sources` is journal metadata only: it does not alter the standard message list or prompt text.
It is refused on request modes that cannot preserve the identity contract (currently structured-I/O
qualification and guard-v1 transforms), rather than being silently dropped.

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
  `clozn_receipt`, or `clozn_lens` in this slice.
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

This is a model-free-tested native/gateway contract, not a broad local-model compatibility claim. Real
exact-model qualification artifacts from the live battery and a successful Open WebUI two-request tool loop
remain acceptance work for Phase 2.8.

## Legacy Completions request fields

| Field | Status | Exact behavior |
|---|---|---|
| `model` | supported | response label for the loaded local worker |
| `prompt` | supported subset | one string; prompt/token arrays are rejected |
| `max_tokens`, `stream`, `temperature`, `top_p`, `seed` | supported | validated and forwarded |
| `n`, `best_of` | one only | `1`/null accepted and stripped |
| `echo` | false only | `false`/null accepted and stripped |
| `top_k` | Clozn extension | forwarded |
| `repeat_penalty` / `rep_penalty` | Clozn extension | one spelling only; normalized to the worker's `rep_penalty` field |
| `user` | documented ignored | string/null accepted and stripped |

`stop`, `suffix`, nonzero frequency/presence penalties, logprobs, logit bias, multiple choices, and unknown fields are rejected
unless they carry the neutral values defined in `clozn/server/openai_compat.py`.

## Response boundary

- Chat and completion response objects/chunks use the standard object names and one choice at index 0.
- Every accepted chat or legacy completion request crosses Clozn's instrumented substrate and is written
  to the local run journal with its delivered messages, rendered prompt, active dials/corrective policy,
  trace, and finish/failure state.
- Non-streaming chat and text completions may add `clozn_run_id` and `X-Clozn-Run-Id`; opt-in Clozn fields are additive.
- Token usage is omitted when unknown. Clozn no longer fabricates zero prompt/completion token counts.
- Finish reasons map worker EOS to `stop` and token limits to `length`. A worker failure is an error, never a
  successful `stop`.
- A proven `length` stop also adds `clozn_warnings: [{"code":"output_truncated", ...}]` to non-stream
  responses and the terminal stream chunk. Non-stream responses additionally carry
  `X-Clozn-Warning: output-truncated`; this warns that the reply may be incomplete and does not claim the
  input prompt was truncated (overlong prompts are rejected).
- Token-live chat and text-completion streams cannot return a run id in headers because the run is persisted
  after headers are sent. Both are still finalized as one journal run after the stream ends. A buffered
  structured stream is validated and journaled first, so it can also return `X-Clozn-Run-Id`.
- Model-emitted `<think>...</think>` scratch text is excluded from `message.content`, legacy completion
  `text`, streaming deltas, echoed assistant history, and the public token trace. Prompt-prefilled and
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
- `tests/test_legacy_completion_instrumented.py` drives the real HTTP handler with a model-free substrate
  and verifies delivered-prompt/dial/trace capture plus success, worker-failure, and disconnect runs.
- `tests/test_gate0_request_paths.py` proves legacy text completions retain their exact raw prompt, decode
  metadata, token trace, finish reason, stable terminal/non-stream run ID, and one coherent journal record.
- `tests/test_think_tags.py` plus the OpenAI/Ollama streaming integration tests prove chunk-split think tags
  never enter public answer/history content while their journal evidence remains inspectable.

## Run association

Non-streaming responses expose `clozn_run_id` in the body and `X-Clozn-Run-Id` in headers. Token-live
streams cannot know the finalized journal ID before their headers are committed, so the ordinary terminal
OpenAI completion chunk carries additive `clozn_run_id` before `[DONE]`; Ollama NDJSON does the same on its
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
