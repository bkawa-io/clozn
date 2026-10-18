# Client conformance

v1 targets ordinary text clients that can use OpenAI Chat Completions.

Configure a client with:

```text
base URL: http://127.0.0.1:8080/v1
API key: any non-empty local placeholder if the client requires one
model: the ID returned by GET /v1/models
```

The model-free suite exercises the real gateway with the OpenAI Python client when installed. It covers
model discovery, non-streaming text, streaming text, additive run IDs, and typed rejection of tools.

Conformance is limited to [OPENAI_COMPATIBILITY.md](OPENAI_COMPATIBILITY.md). A client that always sends
tools, multimodal parts, structured response formats, or unsupported sampling controls will receive a
named HTTP 400. Clozn does not expose Ollama-compatible endpoints in v1.
