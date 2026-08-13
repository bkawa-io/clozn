# Cloze — Local Diffusion-LM Runtime — Design Doc (v0.1)

> **Historical diffusion-design archive.** This document is retained to explain the engine's research
> lineage and old code references. It is not a current product plan or command reference; see
> [CAPABILITIES.md](CAPABILITIES.md).

**Name:** Cloze (CLI: `cloze`)
**One-liner:** A local runtime and serving layer for diffusion language models (dLLMs) — quantized, cached, streamable, and model-agnostic — built on ggml/llama.cpp kernels.
**Status:** Living design doc — Phase 1 (lab) implemented; Phase 3 (C++ `core/`) in progress
**Date:** June 2026

---

## 1. Summary

Autoregressive (AR) local inference is memory-bandwidth-bound: one token per full pass over the weights. Masked-diffusion LMs commit many tokens per forward pass, so the same consumer hardware can yield a multiple of effective tokens/sec, plus native infilling/editing via bidirectional attention.

Today the pieces exist but nothing is a product:

- **llama.cpp** has `llama-diffusion-cli` (Dream, LLaDA 1.0, RND1) — an example binary, not integrated with `llama-server`. Known issues: full-vocab logits shipped GPU→CPU every step (~87% of GPU runtime), Vulkan asserts, no MoE-dLLM support. LLaDA 2.0 support was declined upstream ("not planned"); a community PR is in review.
- **dInfer** (inclusionAI) is fast but server-grade (vLLM/SGLang backends, benchmarked on 8×H800).
- The 2025–26 acceleration literature (block diffusion, delta KV caching, adaptive stopping, self-speculation) is largely unimplemented in any consumer runtime.

This project fills the layer between a research demo and a usable runtime: a **scheduler** that implements the published acceleration techniques, a **serving layer** that handles diffusion's streaming-semantics problem, and an **app shell** with a denoising visualization.

## 1a. Architecture invariants

Five load-bearing constraints that code and reviews must hold (referenced across the tree as "DESIGN invariant N"):

1. **ModelAdapter seam.** PyTorch/Transformers imports live only under `lab/cloze_lab/models/` (and the single ggml adapter under `core/src/`); everything in `scheduler/` is pure logic against the adapter interface (board tokens + attention mask + cached KV in → logits + new KV out). This is what makes the C++ port a translation rather than a rewrite.
2. **Event-sourced spine.** `generate` emits typed events (`gen_started`, `block_started`, `tokens_committed`, `tokens_revised`, `block_finalized`, `step_stats`, `gen_finished`); the TUI, benchmarks, logs, and server are consumers only. Every run logs JSONL for replay.
3. **Tests are the oracle.** Golden fixtures pin (prompt, seed, policy) → exact token picks and confidences. The same fixtures validate the C++ core: picks must match exactly; confidences within an epsilon (float reduction order differs across devices — never assert bitwise equality on sums).
4. **Scheduler writes tokens; model writes KV.** The model never mutates the board; the scheduler never fabricates KV. `cached_token` labels reconcile the two.
5. **Honesty in benchmarks.** Every speed number ships with its quality/divergence column; the cache exactness knob is exposed, never hidden.

## 2. Goals & non-goals

**Goals**

- G1: Run open dLLM checkpoints (Dream 7B, LLaDA 8B/1.5, LLaDA2-mini/flash MoE, RND1, open-dCoder) quantized on consumer hardware (Apple Silicon, single consumer GPU, CPU fallback).
- G2: OpenAI-compatible HTTP API so existing clients work unmodified on day one.
- G3: A native streaming protocol exposing progressive refinement (the basis for the visualization and for smart clients).
- G4: Implement block caching + adaptive stopping for real-world 3–10× speedups over the naive loop, with an explicit exactness knob.
- G5: Honest, reproducible benchmark harness (AR vs dLLM on identical hardware; quality + speed).

**Non-goals (v0.x)**

- Training or fine-tuning dLLMs.
- Multi-GPU / distributed serving.
- Continuous batching at vLLM sophistication (simple step-aligned batching only).
- Writing new attention kernels (reuse ggml; one small selection kernel is the exception).

## 3. System overview

```
┌─────────────────────────────────────────────┐
│  L3  App shell: CLI (pull/run/serve/bench), │
│      desktop UI w/ denoise visualization,   │
│      infill playground, benchmark mode      │
├─────────────────────────────────────────────┤
│  L2  Serving layer                          │
│      • OpenAI-compat mode (block stream)    │
│      • Native mode (refinement events)      │
│      • model registry, request queue        │
├─────────────────────────────────────────────┤
│  L1  Denoising Scheduler  ← core IP         │
│      unmask policy · step control ·         │
│      block manager · 3-tier cache ·         │
│      event emitter                          │
├─────────────────────────────────────────────┤
│  L0  ggml/llama.cpp as a library            │
│      GGUF + quants + kernels +              │
│      GPU confidence-select kernel           │
└─────────────────────────────────────────────┘
```

**Process model:** single native binary (`cloze`) embedding ggml; the desktop app is a thin client speaking the native protocol to a local `cloze serve`. Link llama.cpp as a library / git submodule rather than forking, so upstream kernel improvements flow in; upstream small fixes (e.g., the selection kernel) as PRs for credibility.

**Language:** core in C++ (matches ggml); serving layer C++ or Rust (Rust acceptable if bound via FFI — decide in spike week; bias C++ to keep one toolchain).

## 4. L0 — Model & kernel layer

### 4.1 What's reused

GGUF container, k-quant/i-quant machinery, tokenizers, Metal/CUDA/Vulkan/CPU kernels, MoE plumbing (LLaDA2 reportedly shares the BailingMoeV2 trunk already supported upstream).

### 4.2 GGUF metadata extensions

| Key | Type | Meaning |
|---|---|---|
| `diffusion.mask_token_id` | u32 | The [MASK] vocab id |
| `diffusion.default_steps` | u32 | Recommended denoise steps |
| `diffusion.block_length` | u32 | Recommended block size (0 = full-sequence) |
| `diffusion.schedule` | str | `confidence_topk` \| `threshold` \| `entropy` |
| `diffusion.attn` | str | `bidirectional` \| `block_causal` |
| `diffusion.family` | str | `dream` \| `llada` \| `llada2_moe` \| `rnd1` |

Unknown keys are ignored by upstream tools, so converted GGUFs remain loadable elsewhere.

### 4.3 The confidence-select kernel (the one new kernel)

**Problem:** naive loop computes logits for every masked position and transfers `[n_masked × vocab]` floats to CPU each step; measured at ~87% of GPU wall time in the upstream PR.

**Kernel contract:**

```
inputs : logits buffer [n_masked, vocab] (device-resident),
         temperature, top_p, k_commit (or threshold τ), rng state
outputs: per masked position → (sampled_token_id, confidence) ;
         plus the indices of the top-k_commit positions by confidence
transfer to host: 2 × n_masked ints + floats   (≈10,000× smaller)
```

Confidence = probability of the sampled token after temperature/top-p (configurable: max-prob vs margin vs negative entropy). Implement for Metal + CUDA first; CPU reference path for correctness tests. Upstream as a PR to llama.cpp where feasible.

### 4.4 Architecture support roadmap

1. **Dream 7B** — dense, already runs in upstream CLI; first correctness target.
2. **LLaDA 8B / 1.5** — dense; second target, validates family abstraction.
3. **LLaDA2-mini (16B-A1B MoE)** — first MoE target; fits 16 GB at Q4.
4. **LLaDA2-flash (100B-A6B MoE)** — stretch; needs 64 GB-class Macs / 48 GB GPUs at low-bit quant; a scaling target.
5. **RND1, open-dCoder 0.5B** — cheap to add; 0.5B is ideal for CI tests.

## 5. L1 — Denoising Scheduler

The scheduler owns the generation loop. Everything is **event-sourced**: the scheduler emits typed events; serving, logging, visualization, and benchmarking are all consumers of one stream.

### 5.1 Event model

```jsonc
// Every event: { "t": <step>, "type": ..., ...payload }
{ "type": "gen_started",     "prompt_tokens": 412, "block_len": 32, "max_new": 512 }
{ "type": "block_started",   "block": 3, "span": [96, 128] }
{ "type": "step_stats",      "block": 3, "step": 7, "committed": 21, "remaining": 11,
                             "ms": 38.2, "cache_hit": 0.82 }
{ "type": "tokens_committed","block": 3, "items": [ {"pos":101,"id":4821,"conf":0.93}, ... ] }
{ "type": "tokens_revised",  "block": 3, "items": [ {"pos":99, "old":311, "id":7,
                             "conf":0.71} ] }
{ "type": "block_finalized", "block": 3, "text": " the cache is refreshed", "steps_used": 9 }
{ "type": "gen_finished",    "reason": "eos", "new_tokens": 487, "wall_ms": 5210,
                             "steps_total": 134, "tok_per_s": 93.5 }
```

Events are append-only per request and serializable to JSONL → free flight-recorder logs and replayable visualizations.

### 5.2 Unmasking policies (pluggable)

```
interface UnmaskPolicy {
  // given per-position (token_id, confidence) for masked positions,
  // return positions to COMMIT and positions to REVISE (optional)
  select(candidates, step, block_state) -> {commit: [...], revise: [...]}
}
```

- **confidence_topk** (default): commit the k most confident positions per step; k can ramp (small early, large late).
- **threshold(τ)**: commit everything with conf ≥ τ; pairs naturally with adaptive stopping.
- **entropy**: commit lowest-entropy positions; better calibrated under quantization (see §9).
- **remask_lowconf** (revision): each step, re-mask already-committed tokens whose recomputed confidence dropped below τ_revise. Capped at R revisions per position to guarantee termination. This is the "token revision" feature ("the model changes its mind") — off by default in compat mode, on in the playground.

### 5.3 Step controller

- **fixed(T)**: exactly T steps per block (reproducible benchmarks).
- **adaptive(τ, T_max)**: stop the block when all positions ≥ τ or T_max reached. This is the user-facing quality/speed dial, surfaced in the API as `"effort": low|medium|high` mapping to (τ, T_max, k-ramp) presets.

### 5.4 Block manager

Semi-autoregressive block diffusion: generate left→right in blocks of length L; within the active block, full bidirectional diffusion over [cached prefix ‖ active block]; finalized blocks freeze into the exact KV cache.

Tradeoffs: small L → lower latency-to-first-text, more AR-like, less global coherence within long spans; large L → better parallelism and infill quality, chunkier streaming. Default L=32; expose per-request. `block_len=0` = whole-sequence mode (best for infill tasks, no inter-block caching).

**EOS handling:** in block mode, when an EOS token is committed, truncate the block at EOS and finish (mirrors the approach in the community LLaDA-2.0 PR; needs a model-independent rule keyed off `diffusion.family`).

### 5.5 Cache manager (deep dive — where the speedups live)

Three tiers, decreasing exactness:

**Tier A — Prompt cache (exact).** The prompt never changes. Prefill once; identical to AR prompt caching. Reusable across requests with shared prefixes (system prompts) exactly as llama.cpp does today.

**Tier B — Frozen-block cache (exact).** When `block_finalized` fires, the block's K/V become immutable and append to the prefix cache. From the next block's perspective this is ordinary causal-prefix caching. Cost note: blocks attend bidirectionally *within themselves* during generation but are attended *causally from later blocks* once frozen — K/V values computed during the block's last step are exactly the ones later blocks need, so freezing is free (no recompute).

**Tier C — Approximate intra-block delta cache (the big win, and the accuracy risk).**

Within a block, each denoise step changes only a few token ids (the newly committed ones and any revisions). Naively, every step recomputes K/V for the whole active window. The delta scheme:

```
state: per position p in active block:
         cached_K[p], cached_V[p], cached_token[p], stale_ctr[p]

step(t):
  changed = positions whose token id differs from cached_token (committed,
            revised, or still [MASK] but mask-embedding never changes → not changed)
  if t % FULL_REFRESH == 0 or |changed| / L > REFRESH_FRACTION:
      recompute K/V for ALL positions in block          # exact step
      stale_ctr[*] = 0
  else:
      recompute K/V only for `changed`; reuse cached K/V for the rest
      stale_ctr[unchanged] += 1
  run attention over [TierA ‖ TierB ‖ active block K/V]
  update cached_token for changed positions
```

**Why it's approximate:** when position j's token changes, the *hidden states* (and hence true K/V) of every other position in the block shift slightly, because bidirectional attention let them see j. The delta cache reuses their stale K/V anyway. Empirically (Fast-dLLM family results) the drift is small over a few steps; the periodic **full refresh** bounds it.

**Knobs (exposed, not hidden):**

| Knob | Default | Effect |
|---|---|---|
| `cache.mode` | `delta` | `off` (exact every step) / `delta` / `aggressive` |
| `cache.full_refresh_every` | 4 steps | drift bound |
| `cache.refresh_fraction` | 0.5 | if >50% of block changed, just do a full step |

Ship with a built-in A/B: `cloze bench --cache off,delta` reports speed *and* output divergence (exact-match %, logprob delta) so users see the tradeoff. Honesty here is a feature: power users will benchmark us anyway.

**Memory layout:** active-block K/V live in a fixed ring alongside the standard llama.cpp KV cache; Tier C adds `cached_token[]` and `stale_ctr[]` (ints) only — negligible overhead.

**MoE wrinkle (LLaDA2):** expert routing is a function of the token's hidden state, so when a position's token id changes, its expert assignment can change too. Implications: (1) router runs on-device inside the step, no host round-trip; (2) Tier C treats a routing change as `changed` even if K/V drift would otherwise be tolerated; (3) expert-weight paging (if we ever page experts to host RAM) must not sit on the per-step critical path — pin the hot experts, measure routing churn per step (expected to collapse quickly as the block fills in).

### 5.6 Self-speculative decoding (v0.3+, optional)

dLLMs can draft-and-verify within themselves: take a low-step "draft" pass committing aggressively, then a verify pass; accept positions where verify agrees. Slots in as another `UnmaskPolicy` + a scheduler mode — no new layers needed. Defer until the cache manager is solid; published gains overlap with Tier C so measure marginal benefit before keeping it.

## 6. L2 — Serving layer

One server, two presentations of the same event stream.

### 6.1 Compatibility mode (OpenAI-style)

> **Implementation status (2026-07-17):** the shipped, narrower endpoint/field contract is documented in
> [OPENAI_COMPATIBILITY.md](OPENAI_COMPATIBILITY.md). The bullets below are the original diffusion-serving
> design, not a claim that every OpenAI field or the proposed `cloze` object shipped.

Endpoints: `POST /v1/chat/completions` (+ `GET /v1/models`). The private worker's same-named
loopback protocol remains an internal implementation detail.

Mapping rules:

- Buffer events until `block_finalized`; emit the block's text as a normal SSE `chat.completion.chunk` delta. To clients this looks like chunky-but-valid AR streaming. With L=32 and a fast first block, time-to-first-text stays competitive.
- Revisions: impossible to express in append-only SSE → in compat mode the scheduler runs with `remask_lowconf` disabled **after block finalization** (revisions allowed only within the active block, which hasn't been emitted yet). No client ever sees text retracted.
- `finish_reason`: `stop` on EOS-commit, `length` on max_new reached.
- `usage`: proposed prompt/completion token counts plus a namespaced `cloze` extension. The current gateway
  omits usage when unknown rather than fabricating counts; the proposed extension has not shipped.
- Non-streaming requests: trivially supported (wait for `gen_finished`).
- Extra request params (all optional, namespaced): `"cloze": {"effort": "medium", "block_len": 32, "cache_mode": "delta", "seed": 7}`.

**Original target:** broad client interoperability. Actual compatibility is the tested subset linked above;
do not infer tool calling, structured output, stop sequences, or Responses API support from this design.

### 6.2 Native mode — the Diffusion Streaming Protocol (DSP/0.1)

Transport: WebSocket at `GET /v1/diffusion/stream` (SSE fallback). Frames are the §5.1 events plus a session header. Versioned (`"dsp": "0.1"`), documented as a standalone spec in the repo (`SPEC.md`) under CC-BY — owning the de facto spec is a strategic asset.

```jsonc
→ client request
{ "dsp": "0.1", "model": "llada2-mini-q4", "prompt": "...",
  "params": { "max_new": 512, "block_len": 32, "effort": "high",
              "revisions": true }, "want": ["tokens", "step_stats"] }

← server frames (selected)
{ "dsp": "0.1", "type": "session", "id": "r-91ce", "vocab_hash": "...",
  "detok": "incremental" }
{ "type": "tokens_committed", "t": 7, "items": [...] }   // as §5.1
{ "type": "tokens_revised",   "t": 9, "items": [...] }
{ "type": "block_finalized",  "block": 3, "text": "..." }
{ "type": "gen_finished",     ... }
```

Client-rendering guidance (in the spec): maintain a position-indexed buffer; render `masked` as placeholder glyphs, `committed` as solid text with optional confidence-tinted opacity, `revised` with a brief highlight. This is exactly the data the desktop visualization consumes — the UI is just a DSP client, guaranteeing the protocol stays honest.

Backpressure: server coalesces `tokens_committed` frames if the socket lags; `step_stats` are droppable (`"lossy": true` flag per frame type).

### 6.3 Request scheduling & batching

- v0.1: one active generation per model instance; FIFO queue; N parallel instances if RAM allows.
- v0.2: **step-aligned batching** — requests on the same model advance one denoise step together as a padded batch; joins happen at block boundaries. Far simpler than continuous batching and captures most local multi-client benefit (e.g., editor + chat simultaneously).
- Explicit non-goal: vLLM-class continuous batching.

## 7. L3 — App shell

### 7.1 CLI

```
cloze pull llada2-mini-q4          # registry: HF-hosted GGUFs + manifest
cloze run  llada2-mini-q4          # REPL with live denoise rendering in-terminal
cloze serve --port 11435           # compat + native endpoints
cloze bench llada2-mini-q4 --vs qwen2.5-14b-q4 --suite code,chat
cloze convert <hf-repo> --quant q4_k_m   # HF → dLLM-GGUF (wraps convert scripts)
```

Terminal renderer: the REPL renders masks as `░` resolving into text — the demo works even in a GIF of a terminal.

### 7.2 Desktop app (thin DSP client)

- Denoise visualization (confidence-tinted text, revision flashes), tok/s + steps HUD.
- **Side-by-side race mode:** same prompt to `cloze` and an Ollama/llama.cpp AR model, rendered simultaneously. The single most shareable artifact this project can produce.
- **Infill playground:** select a span in existing text/code → span re-masked → model fills bidirectionally. AR models cannot do this natively; lead demos with it.
- Implementation: Tauri or plain web app served by `cloze serve` (decide in spike; bias to served-web to keep the binary count at one).

## 8. Benchmark & eval harness

- Speed: tok/s, time-to-first-text, steps/token, cache-hit rate; standardized hardware profiles (M2/M3/M4 Mac tiers, 1×4090, CPU-only).
- Quality: small fixed suites (HumanEval subset, GSM8K subset, infill suite) run per config so speed claims always ship with a quality column.
- Divergence: cache `off` vs `delta` exact-match % and mean logprob delta (per §5.5).
- Output: one `cloze bench --report md` markdown table for sharing results.

## 9. Quantization × confidence calibration (research-credibility piece)

Open question nobody has published on: low-bit quantization perturbs logit distributions; unmasking policies *consume confidences*, so miscalibration changes which tokens commit early — quantization may hurt dLLMs differently (worse, or weirder) than AR models.

Plan: sweep {f16, q8, q5_k, q4_k, iq3} × {confidence, entropy} policies on the eval suites; measure quality, steps-to-converge, and revision rates. If miscalibration shows up, a per-quant temperature-rescale on confidences is a cheap fix (calibrate once at convert time, store as GGUF metadata `diffusion.conf_temp`). Either result is a strong technical blog post; the fix being shipped in the runtime is differentiation.

## 10. Milestones

**v0.1 — "It runs, and it's mesmerizing" (~3–5 wks)**
L0 integration (Dream 7B + LLaDA 8B), confidence-select kernel (Metal first), scheduler with `fixed(T)` + `confidence_topk`, Tier A cache only, compat-mode server, terminal renderer, `pull/run/serve`. Launchable: "run a diffusion LLM locally in one command."

**v0.2 — "It's actually fast" (~+4–6 wks)**
Block manager + Tier B, adaptive stopping + effort presets, Tier C delta cache with knobs + divergence bench, LLaDA2-mini MoE, CUDA path for the kernel, side-by-side race mode.

**v0.3 — "It's a platform" (~+4–8 wks)**
DSP/0.1 spec + WebSocket endpoint, desktop visualization + infill playground, step-aligned batching, quantization-calibration study + post, LLaDA2-flash (low-bit, 64 GB-class hardware), optional self-speculation.

## 11. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Open dLLM quality stays niche vs AR at equal size | Med | Lead with speed + infill (where dLLMs win today), not "replace your chat model"; the runtime is cheap optionality on the ecosystem improving |
| llama.cpp upstreams first-class diffusion serving | Med | Be the contributor: upstream kernels, own scheduler/protocol/app on top; speed of a focused indie vs a maintainer's side-interest |
| A frontier lab opens a strong dLLM **with** its own runtime | Low-Med | Their runtime will be server-grade (dInfer precedent); local/quantized stays ours — and their release is our demand spike |
| Tier C cache causes visible quality bugs | Med | Exactness knob + shipped divergence benchmarks; default conservative refresh |
| MoE paging/routing churn tanks LLaDA2 perf | Med | Mini (A1B) first; measure routing churn before promising flash numbers |
| Streaming-revision UX confuses users | Low | Compat mode never retracts emitted text (by construction, §6.1) |

## 12. Open questions

1. Name + repo identity (and whether the DSP spec lives in-repo or standalone).
2. C++ vs Rust for the serving layer (spike week decision).
3. Confidence definition default: max-prob vs margin (run both in v0.1 benches).
4. Whether `cloze convert` should emit upstream-compatible GGUFs only, or also a packed "clozepack" with calibration data.
5. Minimum hardware floor to officially support (M1 8 GB? CPU-only?).

## 13. Open items

Positioning and launch notes are kept out of this technical document.
