# Engine adapter (LoRA) support

**Status: supported.** The engine loads a LoRA fine-tune adapter and applies it over the base model's
weights. This document was originally a record of the *absence* of that support; the gap is closed, and
what remains is the shape of the feature plus the distinctions that made it easy to misread.

## Using it

```bash
clozn serve MODEL --adapter path/to/adapter.gguf [--adapter-scale 1.0]
```

The adapter must be a LoRA GGUF — the format llama.cpp's `convert_lora_to_gguf.py` produces from a
Hugging Face PEFT adapter. `--adapter-scale` multiplies the weight delta; `0.0` attaches the adapter
while contributing nothing, which is the identity control for *"did the adapter change this answer, or
did loading one change something else?"*

At the engine level the flags are `--lora PATH` and `--lora-scale F`.

Validate a converted adapter without loading a model or importing optional ML packages:

```bash
clozn adapter validate tune.gguf --base BASE.gguf
```

This reads the GGUF metadata and exact hash, checks that it declares `general.type=adapter` and
`adapter.type=lora`, optionally checks the base architecture, and prints an executable conversion
command pinned to Clozn's llama.cpp commit. The converter runs in a separate environment containing its
Torch/Transformers/Safetensors dependencies; they are not Clozn product dependencies.

## What happens when it cannot attach

**The worker refuses to start.** It does not fall back to the base model.

This is deliberate and it is the most important behavior here. Someone evaluating a fine-tune who
silently received base-model output would draw a conclusion about weights that were never loaded — a
wrong answer indistinguishable from a right one, and one no downstream receipt could catch. An
architecture mismatch, an unreadable file, or a GGUF that is not a LoRA all produce a clean exit with the
specific reason.

## Capability vs. state on `/health`

Two different facts, reported separately, because a client needs both:

- `capabilities.lora: true` — this build **can** attach an adapter.
- a top-level `lora` object — one **is** attached right now; carries `path`, `scale`, and the adapter
  GGUF's own metadata read back off the file. Absent when none is attached.

A build predating adapter support omits the capability key entirely rather than reporting `false`, so
*absent* means "does not know about adapters" while *false* would mean "knows, and declines."

## Run identity

`clozn/runs/identity_providers/adapter.py` records the attached adapter at `identity["ext"]["adapter"]`.

This matters more than most identity facets: a base model plus a LoRA is a different set of effective
weights from the base alone, and that difference is **invisible** in every field run identity already
recorded — model path, model sha256 and template fingerprint are all byte-identical across an adapted and
an unadapted run of the same base. Without this facet, two runs that answered differently for a
completely explicable reason would look identical in their receipts.

The facet reads the engine's `/health`, never the CLI flag: the flag records what was *requested*, the
health block records what was *loaded*, and an identity block must record the second.

## Not LoRA: two things that look like it

Both are why a keyword search for `adapter` in this repo misleads.

**`ModelAdapter` / `GgmlAdapter`** (`engine/core/include/clozn/model.hpp`,
`engine/core/src/model_ggml.cpp`) is the engine's own backend abstraction — the seam adapting llama.cpp
to clozn's interface. Nothing to do with model weights.

**Control-vector steering** — `GgmlAdapter::set_steer` calls llama.cpp's `llama_set_adapter_cvec`, which
despite the name is the *control-vector* API, and is what the tone dials use. A control vector adds one
vector to the residual stream at inference time, uniformly across a layer range. A LoRA is a pair of
low-rank matrices whose product reconstructs a per-weight-matrix delta, multiplied into specific
attention/MLP projections. Different mathematical objects, different point in the model, different API.
Having one gets you no part of the other.

llama.cpp compounds this by placing `llama_set_adapters_lora` directly beside `llama_set_adapter_cvec`
in `llama.h`.

## Testing

`scripts/dev/make_test_lora.py` builds a structurally-valid adapter matched to any base GGUF's own tensor
shapes, with deterministic pseudo-random weights. No training run, no GPU, no network, no Hugging Face
config directory:

```bash
python scripts/dev/make_test_lora.py BASE.gguf --out adapter.gguf
python scripts/dev/make_test_lora.py BASE.gguf --out mismatch.gguf --arch llama   # refusal fixture
```

The three-arm proof this feature shipped against, on Qwen2.5-0.5B:

| arm | output |
|---|---|
| no adapter | "…the **second largest** city in Europe. It is the **seat of the French government**…" |
| adapter, scale 1.0 | "…the largest city in Europe. It is the **10th largest city in the world**…" |
| adapter, scale 0.0 | byte-identical to the no-adapter arm |

The third arm is load-bearing. Without it, the second arm's difference could equally be explained by the
act of loading an adapter perturbing context creation or allocator layout; arm three pins the change to
the adapter's weights specifically.

## The comparison ladder was already adapter-ready

`clozn/cli/commands/diff_model.py` generalizes `quant_check.py`'s teacher-forced diffing into a ladder
over two duck-typed engine clients, with no assumption they are "two quants of one model."
`run_diff_model(eng_a, eng_b, args, ...)` and everything it calls — the tokenizer preflight
(`check_tokenizer_compat`), chat-template policy resolution (`check_template_match`), the two-direction
runner (`run_direction`), and the honesty-labeled verdict classifier (`classify_verdict`) — run
unmodified against a base engine and an adapter-loaded engine. Comparing base vs. base+adapter is a
boot-step change, not a diff-logic change.

## Comparison command

`clozn diff-adapter MODEL ADAPTER` is the fine-tune author's one-command "what did my adapter change?"
surface over the ladder above. It boots a base arm and a base-plus-adapter arm, holds the model,
tokenizer, template, and quantization constant, and refuses a candidate worker that cannot attach the
adapter.

## Merged-export equivalence

After merging an adapter into a deployment GGUF, validate the exact output with:

```bash
clozn validate-export BASE.gguf \
  --adapter tune.gguf \
  --merged merged.gguf \
  --suite examples/adapter-export-suite.v1.json \
  --out export-receipt.json
```

The command performs static tokenizer, template, vocabulary, architecture, adapter-metadata,
quantization-declaration, artifact-hash, and engine-artifact preflight before loading a model. It then
verifies each worker's effective `/health` identity. A missing or mismatched adapter fails before suite
cases begin.

For every case and seed, the base-plus-adapter arm generates once. Its exact continuation token IDs are
teacher-forced through all three arms. The `base_plus_adapter` versus `merged` delta is the primary
assertion; `base` versus `base_plus_adapter` remains the control. The versioned
`clozn.adapter-export-receipt.v1` artifact records per-case/seed run IDs, numeric token deltas, assertion
budgets, and one explicit verdict:

- `equivalent_within_budget`
- `behavioral_mismatch`
- `identity_mismatch`
- `execution_error`
- `inconclusive`

The receipt never claims byte equivalence or proof of semantic equivalence. When base and merged declare
different quantizations, it says so explicitly and limits the claim to measured behavior within the
suite's budgets.

## Remaining limitations

- Multiple simultaneous adapters. `llama_set_adapters_lora` takes an array with per-adapter scales; this
  build attaches exactly one.
- Hot-swapping an adapter on a live worker. `clear_lora()` exists, but nothing proves a detach returns a
  context to its exact prior state, and that proof would have to be built rather than assumed.
- The merged-export runner and bad-merge path are covered model-free, but a known live merged GGUF still
  needs to pass before that exact conversion pipeline is qualified.
