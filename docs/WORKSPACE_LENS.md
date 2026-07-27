# Workspace Lens

Workspace Lens is the UI-visible seam for per-token, per-layer latent workspace readouts.

This first implementation is deliberately small:

- `workspace_readout` is an additive event type in the generation event spine.
- `clozn/readouts/workspace_lens.py` adapts existing Clozn concept/SAE readouts into that event shape.
- `clozn/server/app.py` prefers the live C++ engine concept path (`/engine/concepts`, backed by
  `/harvest` + SAE) and stores provider `engine_concepts` when available.
- If the engine concept path is unavailable but the Python Qwen + SAE brain stack is loaded, the server
  stores provider `sae/probe`.
- The Studio Run Inspector displays the latest readout, provider name, token strip, and fogginess.
- `studio/workspace_lens_trace.jsonl` is a tiny fixture trace for demos and schema examples.

Mock readouts are not auto-attached to real runs. The deterministic mock provider remains only for fixture
or offline sample traces, where it should be treated as UI/sample data rather than interpretability evidence.
Its sample labels are:

- `code_error`
- `uncertainty`
- `memory_reference`
- `instruction_following`
- `hallucination_risk`

Future providers should keep the same payload shape and replace only the readout source. Good adapter
targets include logit lens, SAE probes, and linear probes. Do not label a provider with a `provider_type`
it doesn't actually compute.

**Jacobian Lens — shipped.** The engine serves `POST /jlens`
(`unembed(J_l @ h)` on the GGUF's own final-norm + head weights — forward-only, deterministic, no
`W_U` sidecar); the Python studio backend proxies it (`POST /jlens`, `POST /runs/<id>/jlens`) and the
Run Inspector's **"Disposed to say &middot; J-lens"** panel renders it, always alongside an unskippable
provenance caption. The lens itself is fit offline, per model (nf4 + autograd against the HF checkpoint)
and applied forward on the engine's own GGUF; today that fit covers Qwen2.5-7B only -- a model-agnostic
(fit-per-model, any AR GGUF) version is scoped but not shipped. It is a **fitted linear lens** reading a
disposition, not a decode of the model's literal thought -- a linear lens always emits *something*, so a
readout alone is never proof the model was "thinking" that token; see the honesty note in the payload
example below.

## Taxonomy Decision

Use `workspace_readout` as the generic persisted adapter event. Do not add a separate
`concept_readout` event for SAE/probe concept labels; concepts are one `readout_kind` inside the
same event. This keeps old traces renderable and gives future providers a stable plug-in slot without
renaming persisted data.

New writers should include:

- `provider`: concrete adapter name, for example `engine_concepts`, `qwen_scope_sae_l15`, or
  `logit_lens_l20`
- `provider_type`: `sae`, `probe`, `logit_lens`, `jacobian_lens`, `mock`, or `engine_concepts`
- `readout_kind`: `concept`, `token`, `feature`, `risk`, or `summary`

Existing traces that only contain `provider` are still valid. The run logger fills subtype fields when
it can infer them from known provider names.

## Event Payload

```json
{
  "type": "workspace_readout",
  "run_id": "run_...",
  "token_index": 3,
  "token_text": " maybe",
  "layer": 12,
  "position": 3,
  "top_readouts": [
    { "label": "dragon/fear/RPG", "score": 0.83 },
    { "label": "mythic creatures", "score": 0.58 }
  ],
  "entropy": 0.67,
  "provider": "engine_concepts",
  "provider_type": "engine_concepts",
  "readout_kind": "concept"
}
```

`entropy` is the current UI's fogginess signal. The engine-concepts adapter derives it from the aligned
token confidence until the engine emits a per-token concept entropy directly.

## Provider Examples

SAE feature readout:

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

Linear probe readout:

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

Logit lens token readout:

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

Jacobian Lens readout (shipped shape, `clozn/clozn_server.py`'s `_jlens_workspace_readouts`; opt-in via
`protocol: true` on `/jlens` and `/runs/<id>/jlens` -- the Run Inspector panel itself reads the raw
`{tokens, readouts, provenance}` response, not this event form):

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

`top_readouts[].score` here is the lens's raw (unnormalized) unembed score, not a probability -- shown as
a plain number, never a [0,1] fill bar. This provider does not currently emit `entropy`.

These readouts are model-state or logit observations. They are not private reasoning text, and they do
not imply causal proof unless a separate intervention/receipt verifies the effect. A J-lens readout is a
**disposition read off a fitted linear lens**, not a claim about what the model is "aware of" or
"thinking" -- the lens always emits its top-k, whether or not the readout reflects anything the model
would itself report.
