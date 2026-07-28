# Engine adapter (LoRA) support — current state

This is a factual record of what `engine/core` can and cannot do with a fine-tune adapter today, so the
next person doesn't have to re-derive it. It makes no roadmap claim and no estimate.

## 1. What the engine can and cannot do

**It cannot load a LoRA adapter, in any form.** There is no code path that applies a low-rank weight
delta to a running model.

- `engine/core/serve/server_main.cpp:84-99` is the complete argv parser for `clozn-server.exe`. Every
  flag the binary accepts is listed there (`--port`, `--host`, `--gpu-layers`, `--mask-token`, `--eos`,
  `--ctx`, `--workers`, `--sae`, `--sae-k`, `--jlens`, `--model-sha256`, `--ar`, `--diffusion`,
  `--no-flash-attn`). There is no `--lora` or `--adapter` flag.
- `clozn/cli/engine_process.py`'s `_launch_args` (the Python function that builds that argv list for a
  spawned worker) has no adapter/lora key. Its only escape hatch is a generic `extra_args` passthrough,
  currently used for `--no-flash-attn` and nothing adapter-related.
- `GET /health`'s `capabilities` object (`engine/core/serve/server_main.cpp:264-278`, mirrored in
  `protocol/fixtures/handshake.json`'s `capabilities` array and guarded by a golden-fixture test) lists
  `streaming`, `state_stream`, `sampling`, `steering`, `infill`, `revise`, `sae`, `jlens`, `readout`,
  `attn_knockout`, `score_arms`. There is no `lora` key — not even a `false` one. A client cannot detect
  "this build knows about adapters and declines" versus "this build predates the concept."
- A case-insensitive search for `lora|adapter` across `engine/core/third_party/bootstrap_llama.py` and
  `PATCHES.md` (which vendor and hand-patch llama.cpp) turns up nothing adapter-related — the one hand
  patch is `0001-llama_get_logits_tensor.patch`.
- No file in `engine/` calls llama.cpp's LoRA adapter API — `llama_adapter_lora_init` /
  `llama_set_adapter_lora` — anywhere. (Confirmed by a repo-wide search; zero hits.)

## 2. False friends: `ModelAdapter` / `GgmlAdapter` and control-vector steering are NOT LoRA

Two unrelated uses of the word "adapter" exist in this codebase and are easy to mistake for LoRA support
on a keyword search:

- **`ModelAdapter` / `GgmlAdapter`** (`engine/core/include/clozn/model.hpp:59`,
  `engine/core/src/model_ggml.cpp`) is the engine's own backend abstraction — "the ONE place a model
  backend lives" (DESIGN invariant 1), i.e. the seam that adapts llama.cpp to Clozn's internal interface.
  `model_ggml.cpp:15`'s comment ("adapter, never free — cheap, and freeing while another adapter lives
  would crash") is about this object's lifetime, not about fine-tune weights. This "adapter" has nothing
  to do with model weights at all.
- **Control-vector steering** — `GgmlAdapter::set_steer` / `clear_steer`
  (`engine/core/src/model_ggml.cpp:371-379`) calls llama.cpp's `llama_set_adapter_cvec`, which despite
  the function name is llama.cpp's **control-vector** API: it adds a fixed direction to the residual
  stream at inference time (already exposed as the `steering` capability, used by
  `clozn/behavior/steering/engine_adapter.py`'s tone dials). A control vector is one vector added
  uniformly; a LoRA is a pair of low-rank matrices (`lora_a`/`lora_b`) that reconstruct a per-weight-matrix
  delta and get multiplied into specific projection weights (attention/MLP) at load time. They are
  different mathematical objects, applied at different points in the model, loaded through different
  llama.cpp APIs. Nothing about having `set_steer` gets you any part of the way to LoRA — it is not a
  partial implementation, degraded mode, or stepping stone.

## 3. What would have to change for LoRA support to exist

This is a description of the work, not a schedule or an estimate.

- **`engine/core/src/model_ggml.cpp` / `engine/core/include/clozn/model_ggml.hpp`** (`GgmlAdapter`): wire
  llama.cpp's LoRA adapter API (`llama_adapter_lora_init` to parse a GGUF-LoRA file against the loaded
  model, `llama_set_adapter_lora` to attach it to a context with a scale, the corresponding detach/free
  calls) into the one place the model backend lives. This includes deciding how a LoRA's declared rank
  and target tensors get validated against the base model's actual shapes before attaching — llama.cpp
  will not do that checking for you in a way `GgmlAdapter` currently surfaces.
- **`engine/core/serve/server_main.cpp`**: a `--lora <path>` / `--lora-scale <f>` argv pair (extending the
  parser at line 84-99), threaded into worker construction, plus an honest `capabilities.lora` entry in
  the `/health` block (line 264-278) — `{"supported": false, ...}` today, flipped once the above lands.
  `protocol/fixtures/handshake.json`'s `capabilities` array and its golden-fixture test would need the new
  key added in lockstep.
- **`clozn/cli/engine_process.py`**: an adapter key in `_launch_args`, and (if per-worker identity should
  record what's loaded) plumbing the adapter path into `spawn_engine`'s launch-flags dict the same way
  `_model_sha256` is threaded through today.
- **GGUF-LoRA structural validation**: something that reads a LoRA GGUF's tensor names and metadata
  (`gguf_header_from_path` in `clozn/cli/fit_planner.py` already parses arbitrary GGUF headers generically
  and would likely read a LoRA file's KV metadata without modification, though this has not been tested
  against a real LoRA GGUF) and rejects a rank/architecture mismatch before ever asking the engine to
  attach it.
- **Reset-semantics proof for hot-swap** (only relevant after the above exists): nothing in the engine
  today proves that detaching a LoRA — or any capability — actually returns a context to its prior state;
  this would need to be built from scratch, not adapted from an existing verified-reset path.

None of this is estimated here. It touches the C++ model-backend layer, the serving binary's argv/health
surface, and a protocol fixture guarded by a test — that combination is why it doesn't fit inside a single
Python-only change.

## 4. The diff ladder is already adapter-ready

`clozn/cli/commands/diff_model.py` generalizes `quant_check.py`'s teacher-forced diffing into a ladder
that operates on two duct-typed engine clients with no assumption that they're "two quants of one model."
`run_diff_model(eng_a, eng_b, args, ...)` (line 317) and everything it calls — the mandatory tokenizer
preflight (`check_tokenizer_compat`, line 112), chat-template policy resolution
(`check_template_match`, line 187), the two-direction ladder runner (`run_direction`, line 207), and the
honesty-labeled verdict classifier that already distinguishes `NO_DETECTABLE_DIFF` / `CHANGED` /
`INSUFFICIENT_SAMPLE` (`classify_verdict`, line 238) — would run unmodified against a base engine and an
adapter-loaded engine the day one exists. The day LoRA loading lands, comparing base vs. base+adapter is a
boot-step change (constructing the second engine client), not a diff-logic change.

## 5. No CLI surface exists

There is no `clozn serve --adapter`, no `clozn diff-adapter`, and no `clozn validate-export` command, not
even as a refusal stub. Nobody should assume a refusal path is present — the flags simply do not parse.
