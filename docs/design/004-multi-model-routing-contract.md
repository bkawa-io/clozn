# ADR 004 — Multi-model routing contract

Status: accepted contract; runtime implementation not started  
Date: 2026-07-29

## Context

The current [product runtime](../ARCHITECTURE.md) supervises one gateway and one private GGUF worker.
The gateway exposes one model, while an OpenAI request's `model` value currently labels the response
without selecting a different worker. Multi-model serving must end that ambiguity before process
management changes.

The machine-readable decision artifact is
[`clozn.model-routing.v1`](../../clozn/schemas/defs/clozn.model-routing.v1.json). It describes one routing
attempt. It does not make the present runtime multi-model.

## Decision

### Public model identifiers

Configuration defines:

- one canonical public ID per configured GGUF artifact;
- one configured default ID;
- optional aliases; and
- an optional preload set.

`/v1/models` and `/api/tags` list canonical configured IDs, including unloaded models. Aliases are
accepted for request selection but are not separate model objects. Alias configuration may change, but
one request resolves it once: its receipt retains the literal requested label, canonical resolved ID,
and artifact digest. Historical receipts are never reinterpreted after an alias changes.

An omitted `model` resolves the configured default. Legacy `clozn serve MODEL` is represented as one
configured, preloaded default with a resident limit of one; omitted-model clients and clients using the
ID returned by discovery remain compatible. `clozn-local` may be configured as a one-model alias.

An explicit unknown label returns `unknown_model`. It must never run the default worker, reuse another
worker's evidence, or merely relabel a response. This intentionally ends the current misleading behavior
for arbitrary model strings.

### Immutable worker key

One worker is keyed by canonical JSON over all behavior-bearing runtime facets:

1. exact GGUF artifact SHA-256;
2. context size;
3. backend token;
4. adapter presence, identity/artifact SHA-256, and exact scale;
5. template fingerprint;
6. engine build; and
7. the complete sorted white-box flag map.

The SHA-256 of that canonical object is `runtime_key.key_sha256`. Model aliases, process IDs, ports,
load time, and LRU position are not key facets. Any behavior-bearing launch option added later must
either enter `white_box_flags`/a versioned facet or require a new schema version; it cannot be ignored.

`worker_identity` names one process generation of that key. A restart increments
`worker_generation`, so a stale private-port or checkpoint identity cannot collide with a later worker.
Handshake facts must match the resolved runtime key before the worker becomes ready.

### Lifecycle

Each configured runtime key is in exactly one state:

| State | Meaning | Request behavior |
|---|---|---|
| `unloaded` | Configured, no live worker | `wait` may start/coalesce a load; `fail_fast` returns `model_not_ready`. |
| `loading` | One spawn/handshake operation is active | `wait` coalesces; `fail_fast` returns `model_not_ready`. |
| `ready` | Identity-qualified and admitting work | Enter its bounded generation queue. |
| `evicting` | No new admissions; shutdown is in progress | `wait` queues for capacity; `fail_fast` returns `model_not_ready`. |
| `failed` | The last load/restart failed with typed evidence | A later `wait` request may initiate a new load event; there is no fallback. |

Ten cold requests for one runtime key produce one load event and ten request-specific wait outcomes.
Load/eviction events are runtime events, not fabricated generation runs.

### Admission, waiting, and cancellation

The default request load policy is `wait`. A future explicit Clozn extension may select `fail_fast`;
all three protocol adapters must lower it to the same internal value rather than inventing
protocol-specific semantics.

`wait` is bounded by:

- a global load-queue limit;
- a per-worker generation-queue limit;
- a queue timeout; and
- a model spawn/handshake timeout.

The limits and resolved policy are copied into the routing artifact. Queue time and model execution time
remain separate performance phases.

Cancellation is request-scoped. It removes that waiter and releases every permit it owns. Cancelling one
coalesced waiter does not cancel the shared load needed by other waiters or preload. If it was the last
waiter, the load may still finish as a recorded runtime event; no generation run is created for the
cancelled request.

### Residency and eviction

`max_loaded_workers` is a hard resident-process limit. The preload list must contain configured canonical
IDs, contain no duplicates, and fit within that limit; invalid configuration fails before serving.

The first eviction policy is deterministic `lru_idle`. A worker with an active generation, an admitted
queue entry, or an active mutation is not evictable. A wait request may queue for capacity; fail-fast
returns immediately. If no worker becomes evictable before the bound, the request receives
`no_evictable_worker` or `queue_timeout`, never an interrupted generation.

### Receipts

Every successful generation run retains the routing result's immutable receipt:

- literal `requested_model` (`null` when omitted) and `selection_source`;
- canonical `resolved_model_id`;
- `resolved_artifact` with GGUF SHA-256;
- the complete `runtime_key`;
- exact `worker_identity` and generation;
- explicit adapter presence/identity/scale; and
- the load event (`not_required`, preload, cold load, or reload), including coalescing and wait time.

Failed attempts carry the same requested fields and a load event. Resolved fields are present only if
resolution reached them. Error attempts may be logged as runtime/request events, but do not fabricate a
generation run.

### Protocol behavior

All generation surfaces resolve the model before generation and use this same contract:

| Surface | Routes | Selection |
|---|---|---|
| Native | `/api/clozn/generate` | Optional `model`; omitted means configured default. |
| OpenAI | `/v1/chat/completions`, `/v1/completions` while it exists | Standard `model`; omitted remains accepted for Clozn compatibility. |
| Ollama | `/api/chat`, `/api/generate` | Standard `model`; omitted means configured default. `keep_alive` is not repurposed as routing policy. |

If `/v1/completions` is retired separately, its retirement response occurs before routing.

OpenAI errors keep the OpenAI error envelope with `type: "model_routing_error"` and `param: "model"`.
Ollama keeps its string `error` plus the stable `code`. Native returns the common structured error.
All expose the same code, message, retryability, phase, and privacy-safe attempt receipt. Streaming does
not begin until selection/load admission succeeds. If a ready worker fails after streaming has begun,
the protocol's terminal failure frame carries the same code; `http_status` below is the status used
before response headers are committed.

### Typed error matrix

| Code | HTTP | Retryable | Phase |
|---|---:|:---:|---|
| `invalid_model_selection` | 400 | no | selection |
| `unknown_model` | 404 | no | resolution |
| `model_not_ready` | 409 | yes | resolution |
| `adapter_unavailable` | 409 | no | resolution |
| `load_queue_full` | 429 | yes | load queue |
| `generation_queue_full` | 429 | yes | generation queue |
| `queue_timeout` | 504 | yes | load or generation queue |
| `model_load_timeout` | 504 | yes | load |
| `model_load_failed` | 503 | yes | load |
| `no_evictable_worker` | 503 | yes | eviction |
| `request_cancelled` | 499 | no | request |
| `worker_failed` | 502 | yes | generation |
| `worker_identity_mismatch` | 502 | no | handshake |
| `capability_unavailable` | 422 | no | capability |

The schema binds each code to its status, retryability, and phase with a closed `oneOf`; adapters may not
choose their own mapping.

## Compatibility and versioning

This ADR changes no route or supervisor behavior. Implementation must land behind model-free contract
tests and preserve the current one-model managed smoke before enabling multiple configured models.

Stored v1 routing receipts are immutable. Optional non-behavioral fields may be added compatibly.
Changing required fields, lifecycle meaning, runtime-key canonicalization, or error mapping requires
`clozn.model-routing.v2`; historical v1 receipts are never rewritten.
