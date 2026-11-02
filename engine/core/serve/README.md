# clozn-server

`clozn-server` is the private native worker behind the Clozn v1 Python gateway. It loads one
autoregressive GGUF and listens on loopback. Users should start Clozn through the `clozn serve`
path rather than running this binary directly.

The worker exposes only:

| Method and path | Purpose |
|---|---|
| `GET /health` | Readiness, runtime identity, and exact capability advertisement. |
| `POST /cancel` | Cancel an active generation request. |
| `POST /v1/completions` | Text completion, optionally streamed as SSE. |
| `POST /score` | Scalar teacher-forced scoring of a recorded continuation. |
| `POST /v1/teacher-forced-score-many` | Experimental exact-prefix-trie scoring of one recorded continuation across up to 16 prompts. |
| `POST /apply_template` | Render text with the GGUF's chat template. |
| `POST /v1/reference-match/arms` | Bounded exact-reference probes. |
| `POST /v1/reference-match/persistent/create` | Create the one worker-local parent session. |
| `POST /v1/reference-match/persistent/probe` | Probe children against that parent. |
| `POST /v1/reference-match/persistent/promote` | Promote a verified child to the parent. |
| `POST /v1/reference-match/persistent/close` | Close the persistent session. |

Requests are strict: unsupported behavior-bearing fields are rejected instead of ignored. Health
advertises the protocol version, engine build, backend, model/tokenizer digests, context/batch
configuration, and retained capabilities. Persistent sessions are process-local and are
invalid after a worker restart.

The native score-many route is model-family agnostic: it operates only on token IDs and causal
sequence IDs, and capacity is based on the exact token-prefix trie. Its response is currently
marked `qualified: false`; the scalar `/score` route remains the trusted evidence path and fallback
while multi-sequence floating-point parity is characterized across GGUF architectures.
