# protocol/ — the state-stream contract (the keystone)

One vocabulary the [engine](../engine) emits and the studio consumes —
collapsing the two spines (the engine's §5.1 typed events and the inspector's `StateStep`)
that were the same abstraction discovered twice. This is what makes Clozn *one* system
instead of two inspectors.

- **`StateStep`** — one frame of the model's evolving state: `{t, substrate, slots, values,
  kind}`. A diffusion *board pass*, an AR *token + residual*, an RWKV *recurrent-state
  update* are all `StateStep`s differing only in `substrate`.
- **`Intervention`** — the write-back: `{target (layer/slot), vector, coef}`. Steering and
  edits flow up this channel.
- **`StateSource`** — anything producing the stream (the engine over HTTP/SSE, or a
  lightweight in-process Python source).
- **Memory ops** — snapshot / restore / persist / associate operate on accumulated
  `StateStep`s: *the model's memory, made legible and editable.*
- **Worker checkpoint refs** — protocol 1.1 pairs every opaque checkpoint id with the exact
  `worker_generation_id` that owns its in-memory KV state, preventing restart collisions without
  claiming persistence or pinning.

Full contract: **[SPEC.md](SPEC.md)** — the canonical types, the engine-event → `StateStep`
mapping, and the SSE/JSON wire (light frame by default, heavy state on demand). View = read the
stream; steer = push an `Intervention`; memory = persist and recall the state.
