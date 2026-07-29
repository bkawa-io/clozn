# The state-stream protocol — spec

The one contract the [engine](../engine) emits and the studio consumes.
It collapses the two spines that were the same idea discovered twice:

| Today | Becomes |
|---|---|
| engine's §5.1 typed events (`events.hpp`) — output-oriented + white-box taps | a stream of `StateStep`s |
| inspector's `StateStep / StateSource / Spine` (`spine.py`) — state-oriented | the canonical vocabulary |

**The inspector's vocabulary is canonical** (it's already substrate-agnostic and state-first);
the engine's events are a *specialization* of it. "The model's memory" is the `state` field of a
`StateStep`; viewing = reading the stream; steering = posting an `Intervention`.

## Canonical types

```
StateStep   { step:int, token:any, state:State, readouts:[Readout], meta:{...} }
Readout     { name:str, value:any, confidence:float, causal_verified:bool|null }
Intervention{ kind:"steer"|"edit"|"restore"|"patch", target:{...}, vector?:[f], coef?:f, note?:str }
State       = { "<component>": tensor }      # diffusion canvas | RWKV state | AR residual/KV slice
```

- **`StateStep`** — one frame of the evolving state. `state` is the internal state *after* this
  step (the substrate's "memory" slice); `token` is what was consumed/committed; `readouts` are
  probe/lens/feature readings; `meta` carries `substrate`, timing, commit counts, block, etc.
- **`Readout`** — a reading that **never travels without its confidence**, and **never claims
  causality until verified** (`causal_verified`: null = unchecked, true/false = patched-and-measured).
  This is the honesty invariant on the wire.
- **`Intervention`** — the write-back channel: steer a direction, edit a slot, restore a snapshot.

`StateSource` (reset / step / get_state / set_state) and `Spine` (drive a source, fan steps to
consumers) stay as in `spine.py` — the engine is just one more `StateSource`, reached over the wire.

## Engine §5.1 event → StateStep mapping

One forward pass's events fold into one `StateStep`:

| Engine event | Folds into |
|---|---|
| `gen_started` / `gen_finished` | stream control frames (`meta.kind = "begin"/"end"`, prompt/total) |
| `tokens_committed` (items) | `StateStep.token` (the committed id(s)) + `meta.confidence` |
| `step_features` (concept scores) | `readouts`: one `Readout` per concept (`name`, `value`=score, per slot) |
| `step_lens` (top-k candidates) | a `Readout` (`name="logit-lens"`, `value`=candidates+probs) |
| `workspace_readout` | `readouts`: named latent-workspace labels, fogginess/entropy, provider metadata |
| `step_stats` (committed/remaining/ms) | `meta` (committed, remaining, ms, cache_hit) |
| `tokens_revised` | a `StateStep` with `meta.kind="revise"` (diffusion-only) |
| block start/finalize | `meta.block`, `meta.span` |
| the activation tap (`ForwardResult.activations`) | `StateStep.state` (the per-position hidden state) |

`meta.substrate ∈ {"diffusion","autoregressive","recurrent"}` tags every step.

## Readout Event Taxonomy

Clozn standardizes on one persisted readout event: `workspace_readout`.
Do not add a separate `concept_readout` event unless a future producer needs
different lifecycle semantics. Concepts, SAE features, probes, logit-lens
tokens, and the shipped Jacobian Lens adapter all use `workspace_readout`
and differentiate themselves with subtype fields.

Canonical persisted shape:

```json
{
  "type": "workspace_readout",
  "run_id": "run_...",
  "token_index": 12,
  "token_text": " example",
  "layer": 15,
  "position": 12,
  "provider": "engine_concepts",
  "provider_type": "engine_concepts",
  "readout_kind": "concept",
  "top_readouts": [{ "label": "feature_or_label", "score": 0.74 }],
  "entropy": 0.31
}
```

`provider` is the concrete adapter id. `provider_type` is the stable class:
`sae`, `probe`, `logit_lens`, `jacobian_lens`, `mock`, or `engine_concepts`.
`readout_kind` is the stable payload category: `concept`, `token`, `feature`,
`risk`, or `summary`.

Existing traces that only have `provider` remain valid. New writers should
include both subtype fields. Consumers should branch on `provider_type` and
`readout_kind` when they need behavior differences, while still displaying
unknown providers as ordinary `workspace_readout` events.

Examples:

```json
{
  "type": "workspace_readout",
  "provider": "qwen_scope_sae_l15",
  "provider_type": "sae",
  "readout_kind": "feature",
  "run_id": "run_demo",
  "token_index": 4,
  "token_text": " dragons",
  "layer": 15,
  "position": 4,
  "top_readouts": [
    { "label": "sae:1421 mythology/fantasy", "score": 0.83 },
    { "label": "sae:0774 creature/entity", "score": 0.61 }
  ],
  "entropy": 0.22
}
```

```json
{
  "type": "workspace_readout",
  "provider": "tone_probe_v1",
  "provider_type": "probe",
  "readout_kind": "concept",
  "run_id": "run_demo",
  "token_index": 7,
  "token_text": " should",
  "layer": 12,
  "position": 7,
  "top_readouts": [
    { "label": "instruction_following", "score": 0.79 },
    { "label": "uncertainty", "score": 0.18 }
  ],
  "entropy": 0.18
}
```

```json
{
  "type": "workspace_readout",
  "provider": "logit_lens_l20",
  "provider_type": "logit_lens",
  "readout_kind": "token",
  "run_id": "run_demo",
  "token_index": 9,
  "token_text": " Paris",
  "layer": 20,
  "position": 9,
  "top_readouts": [
    { "label": " Paris", "score": 0.42 },
    { "label": " London", "score": 0.21 }
  ],
  "entropy": 0.64
}
```

```json
{
  "type": "workspace_readout",
  "provider": "jacobian_lens_l25",
  "provider_type": "jacobian_lens",
  "readout_kind": "token",
  "run_id": "run_demo",
  "token_index": 11,
  "token_text": " boot",
  "layer": 25,
  "position": 11,
  "top_readouts": [
    { "label": " Italy", "score": 1.42 },
    { "label": " Italian", "score": 0.87 }
  ]
}
```

Shipped: the engine's `POST /jlens` (`unembed(J_l @ h)` on the
GGUF's own final-norm + head) and the studio's `POST /jlens` /
`POST /runs/<id>/jlens` proxy it; this event form is the opt-in
`protocol: true` shape, `provider_type: jacobian_lens`, `readout_kind: token`.
The lens is fit offline per model (today: Qwen2.5-7B only) and applied forward
on the engine's own GGUF -- a **disposition read off a fitted linear lens**,
not a decode of the model's literal thought (a linear lens always emits
*something*). These are readouts over model state or logits, not private
reasoning text, and a `jacobian_lens` provider must carry this same honest
framing -- never labeled as reading awareness or the model's actual thought,
and never presented as present before an actual adapter computes the readout.

## The wire (SSE + JSON)

### Optional worker timing

Workers advertising `capabilities.performance_timing: true` attach a versioned `timing` object to the
terminal `gen_finished` event (and to the `meta.kind="end"` StateStep control frame):

```json
{
  "schema_version": "clozn.worker-timing.v1",
  "unit": "nanoseconds",
  "clock": "steady_clock",
  "clock_owner": "clozn_worker",
  "phases": [
    {"name": "prefill", "owner": "clozn_worker", "duration_ns": 12000000,
     "measurement": "measured", "aggregation": "exclusive"},
    {"name": "decode", "owner": "clozn_worker", "duration_ns": 87000000,
     "measurement": "measured", "aggregation": "exclusive"}
  ],
  "metrics": {
    "prompt_tokens": 128,
    "generated_tokens": 32,
    "prompt_tokens_per_second": 10666.7,
    "decode_tokens_per_second": 367.8
  }
}
```

Fields are additive: an older worker may omit `timing` and clients must continue to accept the legacy
`wall_ms`/`tok_per_s` fields. Durations are measured in the named process's monotonic domain. A client
must never subtract a worker offset from a gateway offset; it may only sum non-overlapping durations.
Missing phases are absent, never reported as zero or inferred from wall time. `aggregation:"overlapping"`
and `"context_only"` phases remain useful attribution but do not contribute to known in-request time.

**Light frame by default, heavy state on demand.** Streaming the full activation tensor every
step is gigabytes; most consumers want tokens + readouts + meta. So:

- **Stream** (`GET …/stream`, SSE): each `StateStep` as `data: {json}\n\n`, with `state` **omitted
  or summarized** (shapes + a sparse/projected view) unless `?state=full`. This is the engine's
  existing event SSE, reshaped to the `StateStep` schema.
- **Snapshot** (`GET …/state`): the full `State` (named tensors, shape + base64/npy) — for
  `snapshot/restore/associate`. Pulled when the inspector needs the heavy object, not per step.
- **Intervene** (`POST …/intervene`): an `Intervention` — the engine applies it (steer = the
  control-vector `set_steer`; edit/restore = `set_state`) and the effect shows up in the next steps.

Tensors on the wire: `{dtype, shape, data}` (base64 little-endian) — `np.ndarray` ⇄ JSON both ways.

## How phases 1.2–1.3 implement this

- **1.2 (engine emits):** reshape `sse_data(...)` so each frame is a `StateStep` (the mapping
  above), tag `meta.substrate`, add the `/state` + `/intervene` endpoints. Diffusion + AR both.
- **1.3 (inspector consumes):** an `EngineStateSource(StateSource)` whose `step()` reads the SSE
  frame, `get_state()` hits `/state`, `set_state()`/steer POST `/intervene`. Then every existing
  inspector op (snapshot/restore/probe/steer/memory) runs over a *real engine model*, unchanged.
- **1.4 (gate):** drive an engine model through the `Spine`, snapshot → restore → steer → confirm
  the behavior moved. One spine, proven end-to-end.

## Invariants on the wire

Honesty (Readout confidence + `causal_verified`), the source-owns-state rule (consumers only read;
writes go through `Intervention`), and substrate-agnosticism (a new family is a new `StateSource`,
never a protocol change) all hold here — they're the same four
[carried-over invariants](../ARCHITECTURE.md#carried-over-invariants-non-negotiable), now on the wire.
