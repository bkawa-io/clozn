# llama-server migration compatibility

Clozn targets common single-model GGUF text-chat migrations. It is not a full
`llama-server` replacement and does not accept unknown upstream flags.

| llama-server argument | Clozn status | Clozn semantics |
|---|---|---|
| `-m`, `--model` | supported | Alias for the existing single-model positional argument. |
| `-c`, `--ctx-size` | supported | Same worker context setting as `--ctx`. |
| `-ngl`, `--gpu-layers`, `--n-gpu-layers` | supported subset | Non-negative integer layer count; omitted preserves Clozn's existing default. |
| `--host` | supported | Binds the public Clozn gateway only. The private worker remains loopback-only. |
| `--port` | supported | Public gateway port. |
| `-a`, `--alias` | supported subset | One public model presentation name in single-model mode. It never replaces loaded model identity. |
| `-np`, `--parallel` | supported subset | `1` only. Larger values are rejected because generation is serialized for evidence fidelity. |

The managed multi-model runtime remains configured by its qualified manifest;
single-model aliases and per-model GPU-layer flags are not accepted there.

Not covered by this compatibility surface include arbitrary argument
passthrough, parallel slots greater than one, multimodal serving, embeddings,
router/preset parity, native llama.cpp endpoints, TLS, authentication, and the
legacy `/v1/completions` API, which remains retired.

Compatibility claims should be dated with the upstream llama.cpp revision,
Clozn revision, model identity, and client versions used for the test.
