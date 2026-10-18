# llama-server-style launch flags

Clozn accepts a small set of familiar single-model launch flags. It is not a drop-in `llama-server`.

| Flag | v1 behavior |
|---|---|
| `-m`, `--model` | alias for the positional GGUF path/name |
| `-c`, `--ctx-size`, `--ctx` | worker context size |
| `-ngl`, `--gpu-layers`, `--n-gpu-layers` | non-negative GPU offload layer count |
| `--host` | public gateway bind host; worker stays loopback-only |
| `--port` | public gateway port |
| `-a`, `--alias` | public presentation name for the one loaded model |
| `-np`, `--parallel` | only `1` is supported |

Unknown upstream flags, multiple slots, multiple loaded models, embeddings, multimodal serving, TLS,
authentication, and native llama.cpp endpoints are outside this compatibility surface.
