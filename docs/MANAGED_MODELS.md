# Managed model runtime

Clozn exposes one public Python gateway and runs each GGUF model in a private supervised C++ worker.
Both `clozn serve MODEL` and `clozn serve --models-config manifest.json` use the same in-process router,
request gates, worker identity checks, run journal, and evidence paths.

## Single model

```sh
clozn serve ~/.clozn/models/model.gguf
```

The runtime derives one canonical model ID from `--alias` or the worker handshake, builds an exact
runtime identity, preloads the worker, and installs it in the router as the default model. Supported
single-model launch options include `--ctx`, `--cpu`, `--adapter`, `--adapter-scale`,
`--no-flash-attn`, `--mask`, `--eos`, `--sae`, and `--sae-k`.

## Multiple models

```sh
clozn serve --models-config manifest.json
```

The manifest uses the closed `clozn.managed-models.v1` schema. A compact example:

```json
{
  "schema_version": "clozn.managed-models.v1",
  "default_model_id": "llama-3.2-1b",
  "preload_model_ids": ["llama-3.2-1b"],
  "max_loaded_models": 2,
  "models": [
    {
      "model_id": "llama-3.2-1b",
      "model": "~/.clozn/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
      "runtime_key": {
        "key_sha256": "<derived sha256>",
        "gguf_artifact_sha256": "<gguf sha256>",
        "context_size": 4096,
        "backend": "metal",
        "adapter": {
          "present": false,
          "identity_sha256": null,
          "artifact_sha256": null,
          "scale": null
        },
        "template_fingerprint": "<16 lowercase hex characters>",
        "engine_build": "sha256:<engine executable sha256>",
        "white_box_flags": {
          "sae": false,
          "jlens": false,
          "attn_knockout": false
        }
      },
      "flags": {"ctx": 4096, "chat": true},
      "prefer_gpu": true
    }
  ]
}
```

All root fields are required:

| Field | Meaning |
|---|---|
| `schema_version` | Must be `clozn.managed-models.v1`. |
| `default_model_id` | Model used when a request omits `model`; it must be configured and preloaded. |
| `preload_model_ids` | Workers made resident before the gateway reports ready. |
| `max_loaded_models` | Hard resident-worker limit; must be at least the preload count. |
| `models` | Closed set of qualified worker definitions. |

Each model requires a canonical `model_id`, GGUF `model` path, and exact `runtime_key`. Optional launch
fields are `flags`, `prefer_gpu`, `boot_timeout`, `restart_limit`, and `restart_window`. Relative model
paths resolve from the manifest directory.

## Exact runtime identity

`runtime_key.key_sha256` is derived from every behavior-bearing runtime facet:

- exact GGUF artifact SHA-256;
- context size and backend;
- adapter presence, artifact identity, and scale;
- live chat-template fingerprint;
- engine executable SHA-256; and
- white-box feature flags.

The manifest loader checks the schema, derived key, local artifact hashes, and cross-field constraints.
At worker boot, the registry checks the live handshake against the same identity. Request selection
rechecks the live binding before a worker is used. A mismatch fails closed; it never routes to a nearby
model or the default worker.

The template fingerprint and engine digest are measured facts. The qualification helper in
[`scripts/smoke/managed_runtime_smoke.py`](../scripts/smoke/managed_runtime_smoke.py) demonstrates the
same probes used by the runtime.

## Routing and lifecycle

Native generation, OpenAI Chat/Responses, and Ollama Chat/Generate requests all resolve their `model`
through the same router. The router records a `clozn.model-routing.v1` artifact on each run containing
the canonical model ID, runtime key, and selected worker identity.

The registry provides:

- ordered preload at startup;
- single-flight cold loading for configured nonresident models;
- typed load and identity failures;
- per-worker request serialization;
- concurrency across different workers;
- worker restart supervision; and
- verified-idle least-recently-used eviction at the resident limit.

The router wraps every private-worker call in the registry's call tracker. Eviction therefore never
selects a worker with a generation or mutation in flight. Concurrent cold requests for one model share
one load event.

Run-scoped operations resolve the model recorded on the source run. If that exact configured identity is
not available, receipt, replay, context-effect, and inspection operations return an unavailable result
instead of using the default model.

## Status surfaces

- `GET /readyz` reports aggregate runtime readiness.
- `GET /runtime/models` reports each configured model's public lifecycle state, runtime key, and
  default/preload status. Private ports and local model paths are not exposed.
- `GET /v1/models` and `GET /api/tags` expose the model IDs accepted by their respective client APIs.

## Capacity and cancellation

`max_loaded_models` is a real memory limit. Cold loading may evict only a verified-idle resident; if no
safe victim exists, loading fails with a typed capacity result. Size the preload set for the available
VRAM or unified memory.

Queued work can be cancelled. Once a synchronous private-worker call is in flight, it finishes before
the worker can be evicted; the worker protocol does not interrupt that call by gateway request ID.

## Managed-model constraints

SAE and J-lens workers are rejected by the managed manifest because the runtime key does not yet carry
the fitted analysis artifact's own identity. `attn_knockout` is supported when its launch flag and
runtime-key value agree.

Single-model flags cannot be combined with `--models-config`. Managed overrides
`--default-model`, `--preload`, and `--max-loaded-models` require a manifest. Invalid combinations are
rejected instead of ignored.
