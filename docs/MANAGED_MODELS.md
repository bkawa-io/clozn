# Managed multi-model runtime

`clozn serve --models-config <manifest>` preloads more than one qualified GGUF worker under one
public gateway and routes each request's `model` field to the exact worker keyed to it. This is
RT-BOOT-01, specified in [ADR 004](design/004-multi-model-routing-contract.md). The public routing
artifact is `clozn.model-routing.v1`; the manifest format below is `clozn.managed-models.v1`
([`clozn/schemas/defs/clozn.managed-models.v1.json`](../clozn/schemas/defs/clozn.managed-models.v1.json)).

Read the [Limitations](#limitations-read-this-before-you-rely-on-any-of-the-above) section before
you plan around this. Several things ADR 004 describes as the eventual contract — cold loading,
eviction, request cancellation of an in-flight worker call — are implemented as library code but are
**not** reachable through `clozn serve` today.

## The manifest: `clozn.managed-models.v1`

This is a real manifest. It round-trips through `clozn.cli.managed_models.load_managed_models` (the
same loader `clozn serve --models-config` calls) on a machine that has
`~/.clozn/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf`. `gguf_artifact_sha256` for that model is the
actual SHA-256 of that exact file — copy it and it will *not* match your own download of "the same"
model; a different quantization run, a different mirror, even a different day's build can produce
different bytes and therefore a different hash. That is the entire point of keying on the artifact
digest rather than a filename (see [Qualification and identity](#qualification-and-identity) below).
`template_fingerprint` and `engine_build` here are illustrative-shaped placeholders — a real value
for either one only exists after you have booted the exact engine build against the exact model once
(see [Building a real manifest](#building-a-real-manifest)).

```json
{
  "schema_version": "clozn.managed-models.v1",
  "default_model_id": "llama-3.2-1b",
  "preload_model_ids": ["llama-3.2-1b", "qwen2.5-0.5b"],
  "max_loaded_models": 2,
  "models": [
    {
      "model_id": "llama-3.2-1b",
      "model": "~/.clozn/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
      "runtime_key": {
        "key_sha256": "d4bcaf080d11bf26d26e29a94747efd91b66e8ed6c4fb2a984fac6b6d6823dda",
        "gguf_artifact_sha256": "6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83",
        "context_size": 4096,
        "backend": "gpu",
        "adapter": {
          "present": false,
          "identity_sha256": null,
          "artifact_sha256": null,
          "scale": null
        },
        "template_fingerprint": "a1b2c3d4e5f60718",
        "engine_build": "sha256:b2c3d4e5f6a7081930415263748596a7b8c9d0e1f203142536475869708192a3",
        "white_box_flags": {
          "sae": false,
          "jlens": false,
          "attn_knockout": false
        }
      },
      "flags": {"ctx": 4096, "chat": true},
      "prefer_gpu": true
    },
    {
      "model_id": "qwen2.5-0.5b",
      "model": "~/.clozn/models/qwen2.5-0.5b-instruct-q4_k_m.gguf",
      "runtime_key": {
        "key_sha256": "3276a3b05bcd64732d4de8248431f4a69a10be1391235821d809440d6a8cdfaa",
        "gguf_artifact_sha256": "74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db",
        "context_size": 4096,
        "backend": "gpu",
        "adapter": {
          "present": false,
          "identity_sha256": null,
          "artifact_sha256": null,
          "scale": null
        },
        "template_fingerprint": "9988776655443322",
        "engine_build": "sha256:b2c3d4e5f6a7081930415263748596a7b8c9d0e1f203142536475869708192a3",
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

### Root fields (all required)

| Field | Meaning |
|---|---|
| `schema_version` | Must be the literal string `"clozn.managed-models.v1"`. |
| `default_model_id` | The `model_id` used when a client omits `model`. Must be one of `models[]`, and must currently also appear in `preload_model_ids` — the registry refuses a config where the default isn't preloaded ("the default model must be preloaded until load-on-demand exists"). |
| `preload_model_ids` | The canonical model IDs started as resident workers at boot, in this order. No duplicates. |
| `max_loaded_models` | Hard resident-worker cap. `len(preload_model_ids)` must not exceed it — that's a config error, not a warning. |
| `models` | The closed set of qualified worker definitions (at least one). |

### Per-model fields

| Field | Required? | Meaning when present | When omitted |
|---|---|---|---|
| `model_id` | yes | The canonical public ID a client's `model` field selects. Aliases are not part of v1. | — |
| `model` | yes | Path to the GGUF file. `~` is expanded; a relative path resolves against the manifest file's own directory, not your shell's cwd. | — |
| `runtime_key` | yes | The immutable identity object below. | — |
| `flags` | no | Launch flags: `ctx`, `adapter`, `adapter_scale`, `chat`, `tmpl`, `extra_args` (at most one element, and only `["--no-flash-attn"]` is accepted in v1). | `{}`. `flags.ctx` then defaults to `4096`, so `runtime_key.context_size` must be `4096` too in that case. |
| `prefer_gpu` | no | Whether this worker prefers the GPU build. | `true`. |
| `boot_timeout` | no | Seconds to wait for this worker's boot/handshake. | `180.0`. |
| `restart_limit` | no | Restarts allowed within `restart_window` before the worker gives up (matches the existing single-model restart policy). | `3`. |
| `restart_window` | no | Seconds. | `60.0`. |

### `runtime_key` fields (all required)

| Field | Meaning |
|---|---|
| `key_sha256` | SHA-256 over the canonical (sorted-key, compact) JSON of the other seven facets below. **Derived, never hand-typed** — the loader recomputes it and refuses to boot if your value doesn't match. |
| `gguf_artifact_sha256` | SHA-256 of the exact GGUF file's bytes. Recomputed from the file at load time and compared; a mismatch is a load-time error, not a warning. |
| `context_size` | The worker's `n_ctx`. Must equal `flags.ctx` (or `4096` if `flags.ctx` is omitted). |
| `backend` | `"cpu"`, `"gpu"`, `"cuda"`, or `"metal"`. Checked against the live worker's actual device for `cpu`/`cuda` at handshake time. |
| `adapter` | Closed union: either `{"present": false, "identity_sha256": null, "artifact_sha256": null, "scale": null}` (no LoRA), or `{"present": true, "identity_sha256": "<sha256>", "artifact_sha256": "<sha256>", "scale": <finite number>}`. |
| `template_fingerprint` | 16 lowercase hex characters — the canonical live apply-template fingerprint. This is a probe result, not something you compose by hand. |
| `engine_build` | `"sha256:"` followed by 64 lowercase hex characters — the exact engine executable's digest. |
| `white_box_flags` | `{"sae": bool, "jlens": bool, "attn_knockout": bool}`. In v1, `sae` and `jlens` must both be `false` (see [Limitations](#limitations-read-this-before-you-rely-on-any-of-the-above)); `attn_knockout` must be `true` exactly when `flags.extra_args` is `["--no-flash-attn"]`, and `false` otherwise. |

## Building a real manifest

There is no dedicated `clozn` CLI command yet that emits a qualified manifest for you. The only
working procedure today is the one
[`scripts/smoke/managed_runtime_smoke.py`](../scripts/smoke/managed_runtime_smoke.py)'s `_qualify()`
helper performs, and it is the exact procedure worth following by hand or scripting yourself:

1. Boot a throwaway probe worker for the GGUF (`clozn.cli.engine_process.spawn_engine`).
2. Read `n_ctx` and `device` from its handshake/health response.
3. Call `clozn.cli.runtime_process._worker_template_fingerprint(port)` — the **same** probe the real
   supervisor runs at boot — to get the canonical live template fingerprint. Do not invent one.
4. Hash the GGUF with `clozn.runs.identity.model_sha256(path)`.
5. Hash the engine executable with `clozn.artifacts.contracts.sha256_file(exe_path)`.
6. Construct a `clozn.cli.worker_registry.RuntimeKey(...)` from those facts. Its `key_sha256` is
   computed for you in `__post_init__` — read it back off the object, never fabricate it.
7. Wrap it in a `WorkerDefinition(...)`, assemble the manifest dict, and load it through
   `clozn.cli.managed_models.load_managed_models(path)` before you ever try to boot the gateway with
   it. That call gives you the full closed-schema and cross-hash validation described in the tables
   above as one pass/fail check, with a specific error message on failure.

`load_managed_models` and the supervisor's boot handshake are two independent checks, and both must
pass:

- **At load time** (`clozn/cli/managed_models.py`): schema shape, `key_sha256` against the canonical
  facets, `gguf_artifact_sha256` against the real file's current hash, adapter artifact hash if
  present, `template_fingerprint` length, and `white_box_flags.sae`/`.jlens` both `false`.
- **At boot, per worker** (`WorkerRegistry._qualify_handshake` in `clozn/cli/worker_registry.py`): the
  live worker's reported `model_sha256`, `n_ctx`, device, adapter scale, capabilities, and (when the
  worker announces them) `engine_build`/`template_fingerprint` must all agree with the configured
  key. A worker that handshakes with anything else fails closed with `worker_identity_mismatch` — it
  does not start serving under a "close enough" identity.

## Qualification and identity

The registry keys one worker on `runtime_key.key_sha256` — a digest over **all seven** behavior-bearing
facets: the exact GGUF SHA-256, context size, backend, complete adapter identity (presence, identity
hash, artifact hash, scale), template fingerprint, engine build, and the full white-box flag map.
Aliases, ports, PIDs, load time, and LRU position are deliberately excluded — they don't change what
the worker does.

In practice, this means any of the following produces a **different** identity, and therefore a
worker the registry treats as an entirely separate entity with no shared history:

- swapping the GGUF file — including re-downloading or re-quantizing "the same" nominal model, since
  that changes the file's bytes and therefore its SHA-256;
- changing the context window;
- changing backend/device;
- attaching, detaching, or rescaling a LoRA adapter;
- editing the chat template (its live fingerprint changes);
- rebuilding or replacing the engine executable;
- flipping any white-box flag (`sae`, `jlens`, `attn_knockout`).

**A near match is refused, not silently accepted, on purpose.** `_qualify_handshake` and
`PreloadedModelRouter.qualify_live_identity()` compare the live worker against the configured key on
every boot *and* on every request selection (not just once) — including model SHA-256, context size,
device, adapter scale, capabilities, and, when the worker reports them, engine build and template
fingerprint. Any disagreement is a hard failure (`WorkerIdentityMismatchError` at boot,
`worker_identity_mismatch` at selection time), never a "probably fine" pass-through. ADR 004 states the
underlying rule directly: "Any behavior-bearing launch option added later must either enter
`white_box_flags`/a versioned facet or require a new schema version; it cannot be ignored." The
generation and evidence this runtime produces — routing receipts, run identity, checkpoints, exact-fork
eligibility — are only meaningful if "the model that answered" is an exact, checkable fact rather than a
label somebody trusted.

## Serving

```
clozn serve --models-config manifest.json
```

Optional overrides — each requires `--models-config`, and each is mutually exclusive with the plain
`MODEL` argument and the single-model flags (`--ctx`, `--cpu`, `--mask`, `--eos`, `--sae`, `--sae-k`,
`--adapter`, `--adapter-scale`, `--no-flash-attn`), which the CLI rejects outright as "per-model in
--models-config" rather than silently ignoring:

- `--default-model ID` — override the manifest's `default_model_id`.
- `--preload ID` — override the manifest's `preload_model_ids`; repeat once per model ID.
- `--max-loaded-models N` — override the resident-worker cap. Its own `--help` text says exactly what
  it is today: "resident-worker limit for the qualified config (no cold loading yet)".

On boot, every model in the (possibly overridden) preload set starts, one at a time, before the
gateway is considered ready. A failed preload never tears down an already-ready sibling
(`WorkerRegistry.start_preloaded`) — you can end up serving a partial set.

Once serving:

- `GET /readyz` and `GET /runtime/models` report each configured worker's state
  (`unloaded`/`loading`/`ready`/`evicting`/`failed`), its `runtime_key_sha256`, and whether it's the
  default/preloaded — never a private worker port or local file path.
- Native (`/api/clozn/generate`), OpenAI (`/v1/chat/completions`, `/v1/completions`), and Ollama
  (`/api/chat`, `/api/generate`) requests all resolve `model` through the same router; an omitted
  `model` resolves the configured default, exactly as ADR 004 specifies.
- A successful generation's persisted run and its `meta.model_routing` receipt carry the resolved
  `model_id`, the GGUF SHA-256, and the runtime key that actually served it — not the default's.

`clozn serve MODEL` (no `--models-config`) is unchanged: see
[Legacy single-model compatibility](#legacy-single-model-compatibility).

## Limitations (read this before you rely on any of the above)

**Preloaded only — still true for `clozn serve`, for a narrower reason than before.**
`clozn.cli.worker_registry.WorkerRegistry.ensure_loaded()` implements single-flight cold-load
coalescing and idle-LRU eviction, and both router classes in `clozn.server.model_routing` now accept
an optional `loader` callback to drive exactly that: `PreloadedModelRouter` already did, and
`ProjectionFileRouter` — the class `clozn/server/app.py`'s `main()` actually constructs for
`clozn serve --models-config` — now does too (`ProjectionFileRouter.__init__(..., loader=...)`,
threaded through every `refresh()` rebuild, not just the first). This is proven end to end,
including under real concurrent HTTP dispatch with a real `WorkerRegistry` behind it (single-flight
coalescing, typed load-failure state, and eviction/resident-limit safety), by
`tests/test_projection_file_router_loader.py`.

What is **not** done: `clozn/server/app.py`'s `main()` still constructs `ProjectionFileRouter` with no
`loader` argument, so `clozn serve --models-config` itself does not cold-load on demand. The reason
changed from "the parameter doesn't exist" to a real process-topology fact: the gateway
(`python -m clozn.server.app`) is a separate OS process from the `clozn serve` supervisor that owns
the real `WorkerRegistry` and the model file paths/flags needed to spawn a worker (the routing
projection file is supervisor→gateway one-way state, never a command channel). Building a loader
inside the gateway process would mean either importing `clozn.cli` into `clozn.server` (forbidden —
see `ColdLoadOutcome`'s docstring in `clozn/server/model_routing.py`) or constructing a second,
disconnected `WorkerRegistry` there with no access to those paths and no shared state with the
supervisor's real one — spawning duplicate, desynchronized workers instead of coalescing onto the
supervisor's single-flight guarantee. Wiring a real loader across that process boundary needs its own
supervisor↔gateway IPC integration, which does not exist yet and is separately owned; see the comment
at `MODEL_ROUTER`'s construction in `clozn/server/app.py::main()`. The design for that integration —
who arbitrates, the transport, cross-process coalescing, typed failure states, eviction backpressure,
and why cooperative cancellation of an in-flight worker call still doesn't exist afterward — is
recorded in [ADR 006](design/006-cross-process-cold-load-protocol.md). It is a proposed decision, not
implemented; this section remains accurate until it ships.

So, concretely, today: only the workers named in your (possibly overridden) preload set ever exist.
Requesting a model that isn't preloaded, or one whose preload failed, returns
`model_not_ready`/`model_load_failed` immediately — it does not queue, load on demand, or wait. No
worker is ever evicted for capacity while serving. `--max-loaded-models`'s help text reflects this:
"clozn serve does not cold-load on demand yet."

**Eviction fails closed pending ADR 006 — its "never evicts an in-flight worker" guarantee was
vacuous until this was fixed.** `WorkerRegistry`'s idle-LRU eviction (used when a cold load needs to
free a resident slot) and its explicit `evict()` method both exist to avoid stopping a worker with
active generation or mutation work in flight, and both do so by consulting
`WorkerHandle.busy`/`track_call()`. But nothing in production ever calls `track_call()` — real
generation traffic flows gateway↔worker directly, bypassing the supervisor entirely (see ADR 006's
Context section) — so `busy` read permanently `False` regardless of what a worker was actually doing,
and eviction would have picked a victim on that unverified assumption the moment cold load became
reachable. `WorkerRegistry` now refuses to trust an unwired signal: it only ever picks an eviction
candidate, or lets an explicit `evict()` call proceed past its busy check, when constructed with
`busy_tracking_wired=True` — an explicit declaration that today only test code (deliberately driving
`track_call()` itself) sets. Without it, a capacity-triggered cold load fails closed with the typed
`no_verifiable_idle_worker` code (distinct from `no_evictable_worker`, which means every candidate's
busy state *was* verified and all were genuinely busy), and `evict()` raises
`UnverifiableWorkerStateError` instead of proceeding. This does not change `clozn serve`'s behavior
today, because cold load and eviction are already unreachable through it (above) — it closes the gap
for the moment they become reachable. ADR 006 designs the real fix: a live cross-process query to the
gateway's own `WorkerGateRegistry` at the moment of eviction. Until that ships and something wires
`busy_tracking_wired` for real, eviction is safe by construction but permanently un-triggerable in
production.

**Same-worker calls are serialized; the parallelism is across different models.** This part *is*
wired: `ProjectionFileRouter` defaults to building one `WorkerGateRegistry`
(`clozn/server/request_gate.py`) keyed by the configured model IDs. Two requests naming different
models run concurrently. Two requests naming the *same* model still serialize one at a time — matching
the engine's single active-generation-path limit and `EngineSubstrate`'s per-worker (not shared)
mutable request/steer state.

**Cooperative cancellation cannot interrupt an in-flight private worker call.** The worker protocol
carries no request ID for it. `WorkerHandle.track_call()`/`wait_until_idle()` only count calls in and
out; a queued request can be cancelled, but a call already dispatched to the worker runs to
completion regardless.

**Receipt / replay / influence / legacy-fork operations fail closed against a managed gateway, and
return unavailable rather than silently running on the default worker.** `active_sub(h)`
(`clozn/server/app.py`) returns `None` whenever a managed router is configured and the current request
didn't install an explicit worker selection. Today, the routes behind `/runs/<id>/receipt`,
`/runs/<id>/receipts`, `/runs/<id>/replay`, `/runs/<id>/counterfactual`, the influence-map routes, and
the legacy `/runs/<id>/fork` all resolve their worker through that same plain `active_sub(h)` — so
under a managed gateway they get no worker and answer unavailable, for any run, not only ones whose
model can't be determined. Only exact execution-fork (`/runs/<id>/execution-fork*`) and snapshot pin
have been updated to resolve a run's own recorded model explicitly through the router
(`_parent_sub_facts` in `clozn/server/routes/execution_fork.py`) and therefore work under a managed
gateway today. This is verified directly by
`test_unselected_run_engine_routes_never_use_default_worker` in `tests/test_managed_model_bootstrap.py`.

Why fail closed instead of falling back to the default worker: every one of these operations re-touches
a private worker to recompute or verify evidence about one specific historical run. If that silently
ran on whichever worker happens to be "the default," you would get a receipt, a replay, or an influence
measurement that looks like real evidence about the run — plausible output, wrong model. An explicit
"unavailable" is strictly safer than a confident answer that isn't actually about what you asked.

**SAE and J-lens configurations fail closed for managed models — at manifest load time, before the
gateway ever boots.** `load_managed_models` (`clozn/cli/managed_models.py`) raises immediately if any
model's `runtime_key.white_box_flags.sae` or `.jlens` is `true`. You cannot start a managed gateway
with a SAE- or J-lens-enabled worker in v1; every managed worker's flags for those two are always
`false`. Reason: the v1 runtime key has no field for a SAE/J-lens artifact's *own* identity (which
specific fitted readout, at what checksum) — hashing a bare boolean would let two different fitted
artifacts silently collide onto the same routing identity.

**VRAM is real, and the resident limit is not advisory.** Because `clozn serve` doesn't cold-load on
demand yet (above), everything in your preload set is resident for the runtime's entire life — there
is no lazy loading to defer memory use and no eviction to free it under pressure. Configuring
`preload_model_ids` beyond `max_loaded_models` is a hard config error at boot, not a soft cap. Size
your preload set to what actually fits, concurrently, on your hardware: two 7B-class GGUFs will not
both fit resident on a 16 GB GPU at once, and a worker that doesn't fit simply fails to boot — it does
not silently fall back to CPU or partial offload.

## Legacy single-model compatibility

`clozn serve MODEL` (no `--models-config`) is the exact original code path, unchanged: one gateway,
one private worker, no worker registry, no routing-projection file, and no `MODEL_ROUTER` installed in
the gateway process at all (`clozn/cli/runtime_process.py`: `RuntimeConfig.managed_models` is `False`
whenever `worker_definitions` is empty, and `RuntimeStack` takes the pre-existing single-`WorkerHandle`
branch). Existing clients — an omitted `model` field, or the ID returned by `/v1/models`/`/api/tags`
discovery — see no behavior change.

All the single-model flags (`--ctx`, `--cpu`, `--mask`, `--eos`, `--sae`, `--sae-k`, `--adapter`,
`--adapter-scale`, `--no-flash-attn`) still work exactly as before. Passing any of them together with
`--models-config` is a hard `CloznError` ("... is per-model in --models-config"), not a silently
ignored flag. Symmetrically, `--default-model`, `--preload`, and `--max-loaded-models` require
`--models-config`; passing one with a plain `MODEL` argument is also a `CloznError`, not a no-op.
