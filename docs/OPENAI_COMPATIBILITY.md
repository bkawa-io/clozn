# OpenAI text API compatibility

Clozn implements a strict text-only subset. It does not claim full OpenAI API compatibility.

## Endpoints

| Endpoint | Status |
|---|---|
| `GET /v1/models` | Supported; returns the one configured public model identity |
| `POST /v1/chat/completions` | Supported for one text choice, streaming or non-streaming |

There is no public Responses, Assistants, embeddings, images, audio, batches, tools, structured output,
or Ollama-compatible API in v1.

## Request fields

Implemented fields:

- `model`
- `messages`
- `max_tokens` or `max_completion_tokens`
- `temperature`, `top_p`, `top_k`, `repeat_penalty`, and integer `seed`
- `stream`
- `stop`
- `stream_options.include_usage`
- `clozn_sources`

Messages must be `{role, content}` objects with string content. Roles are `developer`, `system`, `user`,
or `assistant`; `developer` is normalized to `system` for local GGUF templates. Extra message fields and
multimodal content are rejected.

`clozn_sources` supplies exact source ranges without changing the message text. Each item names a
`message_index`, label/identity metadata, and a half-open Unicode range. Nested ranges require an explicit
parent and overlapping peers are rejected. Canonical source IDs are minted from the recorded bytes, not
trusted from the client.

The following common fields are accepted only at neutral values and then removed: `n:1`, `user`, zero
frequency/presence penalties, `logprobs:false`, `top_logprobs:0`, empty `logit_bias`, text-only
`modalities`, `store:false`, empty `metadata`, and the default service tier. Any behavior-bearing value
or unknown field returns HTTP 400 with a named `error.param` and typed code.

In particular, `tools`, `tool_choice`, `response_format`, and former `clozn_*` behavior controls are
unsupported.

## Responses and recording

Non-streaming replies use `chat.completion` with one choice at index 0. Streaming replies use
`chat.completion.chunk` frames followed by `[DONE]`. Native worker events never appear on this public
stream.

Unknown token usage is omitted rather than reported as zero. When the worker exposes exact prompt and
completion counts, Clozn returns standard usage fields. Worker EOS maps to `stop`; a token limit maps to
`length` and adds an output-truncated warning.

Every successful generation is durably recorded. Non-streaming replies add `clozn_run_id` to the body
and `X-Clozn-Run-Id` to headers. A live stream cannot know its ID before headers are sent, so its terminal
chunk carries `clozn_run_id`. A recording failure is an upstream error, not a successful unsaved reply.

Model-emitted `<think>` blocks are removed from public answer/history content and token traces. Their
local reasoning evidence remains separate in the run journal.

## Coverage

`tests/test_openai_compat.py` pins request normalization and rejection. `tests/test_openai_client_compat.py`
drives model listing, non-streaming chat, streaming chat, recording failures, and unsupported tools through
the real gateway with the OpenAI Python client when that optional test dependency is present.
