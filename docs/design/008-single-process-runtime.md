# Single-process gateway runtime

Status: implemented

## Decision

`clozn serve` owns one public Python gateway in the supervisor process and one or more private C++ GGUF
worker subprocesses. The gateway router receives live registry callbacks and immutable worker bindings
directly in memory.

```text
clients
   │
   ▼
public Python gateway
   │ exact model selection + request gate + call tracking
   ├───────────────┬───────────────┐
   ▼               ▼               ▼
private worker A  private worker B  private worker C
```

There is no second Python gateway process and no file or environment transport for routing state.

## Invariants

1. Every public generation request resolves one canonical model ID before worker access.
2. A selected worker must match its configured GGUF, context, backend, adapter, template, engine build,
   and white-box identity.
3. Requests to one worker serialize; requests to different workers may run concurrently.
4. Every routed worker call increments that worker's in-flight counter.
5. Cold loading coalesces by model ID, and capacity eviction considers only verified-idle workers.
6. Run-scoped evidence selects the source run's recorded model and never silently uses the default.
7. Worker failure is isolated; the supervisor can restart a failed worker within its configured budget.
8. Gateway shutdown stops all owned private workers.

## Launch modes

`clozn serve MODEL` constructs one exact worker definition and a one-model router.

`clozn serve --models-config MANIFEST` constructs a registry from the closed manifest, preloads the
configured set, and permits cold loading within `max_loaded_models`.

Both modes call the same `build_server(..., model_router=router)` entry point and therefore share route
selection, identity receipts, gates, and run logging.

## Failure behavior

- An unknown model returns a typed routing error.
- A configured but unavailable model returns its load or identity failure.
- A noncurrent or mismatched worker is never substituted with another worker.
- If capacity has no verified-idle victim, the cold load is refused.
- The public gateway remains available while failures are isolated to individual workers.

## Verification

The architecture is exercised by the runtime, model-routing, worker-registry, request-gate, identity,
and managed-runtime smoke suites. Live qualification additionally starts real workers, routes requests,
checks cold-load coalescing and isolation, and verifies clean shutdown.
