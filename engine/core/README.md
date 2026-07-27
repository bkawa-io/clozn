# core/ — the C++/ggml runtime (Phase 3)

> ## Status: IN PROGRESS — the L0/L1/L2 stack runs end-to-end
>
> Started ahead of the original phase order (by decision). The C++ runtime now does generation,
> infill, exact KV reuse, on-GPU confidence-select with **zero-copy device logits**, the typed event
> spine, a runnable CLI, and an HTTP server with SSE streaming — all validated against the lab
> goldens. It is **not yet the shipped product** (the lab remains the Phase-1 reference and oracle).
>
> Done and tested so far:
> - **The L1 scheduler, ported to C++** — `policies`, `stepper`, `blocks` (the one-way law), and the
>   Tier A/B/C `cache` manager — as `clozn_scheduler`, a pure-logic static lib with **no model/ggml
>   dependency** (the seam, enforced at the build level). 4/4 unit tests green, mirroring the lab's.
> - **The C++ `ModelAdapter` seam + the `generate` pass loop** — `model.hpp` (the seam), `sample`
>   (greedy confidence reference), and `generate` (the whole-sequence fixed(T) loop) as
>   `clozn_runtime`, still backend-free (links only the scheduler, speaks to the abstract adapter).
> - **The L0 ggml adapter, end-to-end, cross-checked against the lab goldens.** `GgmlAdapter` loads an
>   open-dCoder GGUF and drives the loop through the seam. `test_ggml_generate` (`-DCLOZN_BUILD_GGML=ON`)
>   reproduces `lab/tests/golden/dcoder_add.json` (whole-sequence) **8/8 picks exact**, text/reason
>   identical. (The raw `test_ggml_forward` smoke test — bidirectional decode + Dream shift → `' b'` —
>   still stands underneath it.)
> - **KV reuse under the one-way law — validated AND measured.** Block mode reproduces
>   `dcoder_add_blocks.json` **8/8 picks exact** (incl. conf-0.067 knife-edge slots). llama.cpp exposes
>   only causal on/off, so the block-causal law comes from **decode order**: each segment is decoded
>   while later positions don't exist, so its frozen K/V never attends forward (Tier A/B exact); the
>   active block is recomputed each pass (Tier C exact), and a captured "boundary row" bridges the
>   shifted head across the freeze. Off vs reuse on the same board: **2.78× fewer token-decodes**
>   (hardware-independent), **2.1× wall-clock** on a tiny CPU checkpoint — grows with prompt length and
>   on GPU.
>
> - **The confidence-select kernel, wired into the commit step.** The loop's per-pass sample +
>   confidence + select is now a `CommitSelector` seam (default `CpuCommitSelector`, backend-free).
>   `KernelCommitSelector` (`-DCLOZN_BUILD_CUDA=ON`) drives the Phase-2 CUDA kernel for that fusion;
>   `test_kernel_selector` proves it returns **identical picks** to the CPU selector on an RTX 5080
>   (sm_120) — 5/5 parity cases incl. a 50k-vocab row.
>
> - **Zero-copy device logits — the §4.3 transfer win, realized in the live loop.** With a CUDA
>   llama (`-DGGML_CUDA=ON`) + GPU offload, the per-step logits stay on the GPU: a small additive
>   llama patch (`llama_get_logits_tensor` + `llama_set_skip_raw_logits`, tracked in
>   [`third_party/PATCHES.md`](third_party/PATCHES.md), **not** a fork — llama.cpp is a pinned
>   submodule, the patch is a re-appliable diff) hands `GgmlAdapter` the device-resident logits
>   tensor and suppresses llama's decode-time D2H; `KernelCommitSelector` gathers the requested rows
>   on-device (D2D) and runs the kernel, so the full vocab never crosses the bus. `test_zerocopy`
>   gates it three ways — CPU, kernel host-upload, kernel device-zero-copy — for **both**
>   whole-sequence and semi-AR blocks (with exact KV reuse), all reproducing the lab goldens
>   (`dcoder_add.json` / `dcoder_add_blocks.json`) and agreeing byte-for-byte, with
>   `device_forwards>0` proving the path engaged. The block-mode shifted-head boundary row (frozen,
>   not in the active device tensor) is carried as one host row H2D'd into the on-device gather, so
>   block passes stay zero-copy too. Structural logits D2H over the test runs drops from **43.8 MB →
>   10.9 MB** (whole-seq) and **19.4 MB → 0.0 MB** (block); per steady-state step the `n_outputs×vocab`
>   copy becomes ~`2·n_masked`; grows with steps/vocab.
>
> - **The typed event spine (§5.1, invariant 2).** `generate` emits `gen_started` / `block_started` /
>   `tokens_committed` / `step_stats` / `block_finalized` / `gen_finished` — collected on the result,
>   streamed live to an optional callback, and serializable to replayable JSONL (the same
>   `{"t","type",...}` wire schema as the lab, so logs replay across both runtimes). `test_events`
>   gates the stream shape + JSONL round-trip; emission is pure observation, so goldens are untouched.
>
> - **Infill — native fill-in-the-middle (a capability AR models structurally lack).** `infill()` denoises a
>   masked gap *between* a prefix and a suffix under full bidirectional attention, so every filled
>   slot sees the fixed right-context as well as the left — which AR models structurally lack.
>   Demo: prefix `def add(a, b):` + suffix `return result` fills `result = a + b` (it used the suffix
>   to make the return correct). `test_infill` gates both sides preserved + gap filled + one-sided
>   context + the event stream.
>
> - **`clozn` — the native runtime CLI.** `core/` is runnable, not just tested:
>   `clozn <model.gguf> "<prompt>" [--max-new --steps --block-len --cache delta --gpu-layers --stream
>   --log FILE --suffix "<txt>" --gap N ...]` loads a GGUF diffusion LM and denoises (or infills)
>   through the same adapter + loop the goldens pin, printing the completion + an honest stats line
>   (`N tokens | P passes (steps/token) | tok/s | forward work`). `--stream` shows the live denoise
>   off the event spine; `--log` writes the JSONL flight recorder; `--suffix`/`--gap` switch to
>   infill. On open-dCoder 0.5B, `def gcd(a, b):` denoises to `return (a*b)//__gcd(a,b)` at **0.53
>   steps/token** — under one forward pass per committed token, the whole dLLM premise, live.
>
> - **Runs at real scale — Dream-7B.** Dream-v0-Instruct-7B (converted HF→GGUF as Qwen2, Q8_0,
>   7.6 GB) runs in the C++ runtime on the 16 GB GPU and produces correct, idiomatic code — e.g.
>   `def reverse(s):` → `return s[::-1]` + driver code, at **0.62 steps/token**, 63 tok/s, clean EOS.
>   The `clozn-bench` quant sweep (open-dCoder) reports the honest quality column: Q8_0 ~half the
>   size near-lossless (92% greedy-token agreement vs f16), Q4 trades real quality; tok/s flat on a
>   tiny model (overhead-bound, not bandwidth-bound — the quant *speed* win is on bigger models).
>
> - **The serving layer (L2) — `clozn-server`.** An HTTP server over the runtime
>   (`-DCLOZN_BUILD_SERVE=ON`, using the cpp-httplib + nlohmann/json llama.cpp already vendors):
>   `GET /health`, `POST /v1/completions`, `POST /v1/infill`. With `stream=true` it emits the §5.1
>   events as Server-Sent Events (each event a `data:` frame, then a final OpenAI frame + `[DONE]`)
>   — the native streaming protocol the spine was built for, so a client watches the denoise
>   pass-by-pass. **Concurrent**: `--workers N` runs a pool of N contexts over ONE shared model
>   (the weights load once; only KV/compute buffers replicate), so N requests run at a time — a
>   `GgmlModel` (weights+vocab+tokenizer) shared by N `GgmlAdapter` contexts. Validated on GPU:
>   4 concurrent requests complete in ~the time of one.
>
> Still ahead: an interrupted/cancelled-request path in the server, batched multi-sequence decode
> (today each context runs one sequence), and the §9 calibration extended to importance-matrix
> quants. Every speed number ships with its quality/divergence column and the
> structural-vs-wall-clock distinction — honesty is a core invariant (DESIGN invariant 5).

`core/` is the eventual L0/L1 implementation from [`docs/DESIGN.md`](../docs/DESIGN.md): the single
native `clozn` binary that embeds **ggml/llama.cpp** as a library, runs GGUF + quantized dLLM
checkpoints, and drives them with the same denoising scheduler the lab already implements. Phase 2
(the [`kernels/confidence_select`](../kernels/confidence_select) CUDA kernel) plugs into it.

---

## Why this is a translation, not a rewrite

The single most important architectural fact about Clozn is **the ModelAdapter seam**
([`lab/clozn_lab/models/base.py`](../lab/clozn_lab/models/base.py), DESIGN invariant 1).
Everything under [`lab/clozn_lab/scheduler/`](../lab/clozn_lab/scheduler) is *pure logic* against
that interface — no torch, no transformers, numpy at most for mask arithmetic. The model side
(PyTorch/HF) lives entirely behind the seam.

That split is what makes `core/` a **port**:

- The **scheduler** (`policies.py`, `stepper.py`, `blocks.py`, `cache.py`, `events.py`, the
  `generate.py` pass loop) is small, allocation-light, framework-free logic. It translates to C++
  almost statement-for-statement: `ConfidenceTopK` is a sort + a count; `Threshold` is a filter + a
  rail; `BlockPlan.blocks()` is a `while` loop; `CacheManager.plan()` is set arithmetic over position
  indices; the events are POD structs serialized to the same JSONL wire form. None of it touches a
  tensor.
- The **model** side is the only part genuinely *different* between Python and C++ (transformers
  `forward` + `DynamicCache` vs a ggml compute graph + llama.cpp KV cache). But it is **isolated
  behind one interface**, so the port replaces *one implementation of one seam* and reuses everything
  else.

The lab was built deliberately to make this true: torch imports are confined to `models/`, and the
scheduler is already proven framework-free by running against a numpy-only `FakeAdapter` in CI. The
C++ work is therefore *bounded and well-specified*, not open-ended. That is the core advantage.

---

## The seam, mapped to C++

The Python `ModelAdapter.forward` contract ([`base.py`](../lab/clozn_lab/models/base.py)) is:

```
board tokens + attention mask + (optional) cached KV + which positions to recompute
    ->  logits for the requested positions + complete KV for the board
```

The C++ side mirrors this one-for-one:

| Python (lab seam) | C++ (`core/`, planned) | Backed by |
|---|---|---|
| `board: int64[seq]` | `const llama_token* board, int n` | scheduler-owned token array |
| `attn_mask: bool[seq,seq]` | block-causal mask built into the ggml graph | `blocks.attention_mask` → graph construction |
| `kv: KVState` (opaque handle) | `llama_kv_cache` / a Clozn wrapper over ggml K/V tensors | llama.cpp KV cache |
| `recompute_kv: sorted positions` | positions to (re)decode this pass; rest reused | Tier A/B exact, Tier C delta |
| `logits_for: positions` | rows of the logits to read back (or feed the kernel) | masked positions |
| `ForwardResult.logits` | device logits buffer → **confidence-select kernel** → 2×n to host | Phase-2 kernel (§4.3) |
| `ForwardResult.kv` | updated KV handle | llama.cpp |
| `ModelConfig` | GGUF metadata keys `diffusion.*` | DESIGN §4.2, mirrored field-for-field in base.py |
| `encode` / `decode` | llama.cpp tokenizer (`llama_tokenize` / `llama_detokenize`) | GGUF tokenizer |

Two invariants travel across the port unchanged and must be re-asserted in C++:

- **Invariant 4 — scheduler writes tokens; model writes KV.** The C++ adapter reads `board`, never
  mutates it; computes K/V only for `recompute_kv`; the returned KV covers every board position. The
  `cached_token[]` built-as labels that reconcile board ↔ drawers live in the scheduler's cache
  manager, not the model.
- **The one-way law** (`blocks.py`): active block attends to frozen blocks, never the reverse — what
  keeps Tier A/B exact. In C++ it becomes the attention mask / KV-cache layout the graph is built
  with; getting it wrong silently corrupts prefix reuse (exactly the failure the Dream adapter's
  `_additive_mask` already documents for Qwen2 siblings — see [TECHNICAL §5](../docs/TECHNICAL.md)).

The Python `DreamAdapter` already demonstrates *prefix-only* (Tier A/B, contiguous-suffix) reuse
against a real append-only `DynamicCache`. The C++ adapter starts there: Tier A/B exact prefix reuse
is plain causal-prefix KV caching, which llama.cpp already does. Tier C's scattered mid-sequence
recompute is the genuinely new piece (the lab's Dream adapter raises `NotImplementedError` on
non-contiguous `recompute_kv` for exactly this reason); it requires a KV cache that can rewrite
individual positions — the main ggml-side research task of Phase 3.

---

## How `core/` links llama.cpp, and where the kernel slots in

Per DESIGN: **link llama.cpp as a library / git submodule — do not fork.** Upstream kernel and quant
improvements then flow in for free; our small additions (the confidence-select kernel, the
`diffusion.*` GGUF metadata, dLLM family handling) are upstreamed as PRs for community credibility.

```
core/                              (planned)
  CMakeLists.txt                   top-level; add_subdirectory(third_party/llama.cpp)
  third_party/llama.cpp            git submodule, pinned SHA (NOT vendored/forked)
  include/clozn/                   public headers (the seam, events, config)
  src/
    model/                         L0 — the ModelAdapter seam in C++
      adapter.hpp                  forward(board, mask, kv, recompute, logits_for)
      dream.cpp  llada.cpp ...     family quirks (shifted head, EOS rule) behind seam
      gguf_meta.cpp                read diffusion.* keys (DESIGN §4.2)
    scheduler/                     L1 — direct port of lab/clozn_lab/scheduler/
      events.hpp  policies.cpp  stepper.cpp  blocks.cpp  cache.cpp
      generate.cpp                 the pass loop (port of generate.py)
    serve/                         L2 — OpenAI-compat + native streaming protocol; later
    cli/                           L3 — pull/run/serve/bench; later
```

**The confidence-select kernel** ([`kernels/confidence_select`](../kernels/confidence_select),
Phase 2) is the join between L0 and L1. In the lab, `generate.sample_candidates` ships the full
`[n_masked × vocab]` logits to host and samples there (~87% of GPU wall time, DESIGN §4.3). In
`core/`, the device-resident logits buffer is handed straight to `clozn::confidence_select`, which
samples, scores, and selects on-device, so only `2 × n_masked` ints/floats cross to host (≈10,000×
smaller). The kernel's contract is *already pinned* by `kernels/confidence_select/reference.py` and
its parity tests, so the C++ scheduler calls the kernel exactly where the lab calls
`sample_candidates` + `policy.select`. (The kernel's deterministic paths are validated on an RTX 5080;
its sampled/`top_p` paths and `core/`'s own GPU path still need the toolchain bring-up.)

---

## What had to be true before work began (all held — work has begun)

The honest "definition of ready." `core/` work did not start until each of these held; they now do,
which is why the slices above exist:

1. **A C++ toolchain with CMake ≥ 3.18 and a CUDA (and/or Metal) compiler.** The Phase-2 kernel's
   deterministic paths now compile + validate on an RTX 5080 (`sm_120`, CUDA 13.3) where such a
   toolchain exists; `core/`'s GPU path has the same prerequisite (a CUDA-less CI box builds only the
   CPU paths). On Windows: MSVC + CUDA toolkit; on Apple Silicon: the Metal toolchain.
2. **The Python goldens stand as the cross-validation oracle.** The fixtures in
   [`lab/tests/golden/`](../lab/tests/golden) pin `(model, prompt, seed, policy, stepper, cache)` →
   per-step commits + final board/text/reason, in a versioned `clozn-golden/1` format. The C++ port
   is *correct* exactly when it reproduces them: **token picks must match exactly; confidences within
   epsilon** (float reduction order differs across devices — DESIGN invariant 3). These fixtures
   already validate the lab against the numpy `FakeAdapter`, and the kernel reference against the
   lab — the C++ runtime is just the next consumer of the same oracle.
3. **The Phase-2 confidence-select kernel is verified on real hardware** against `reference.py` — its
   deterministic paths now are (RTX 5080, `sm_120`); the sampled path and `top_p` remain to finish.
4. **Phase 1 stays the shipped product through the transition.** `core/` does not replace `lab/`;
   the lab remains the reference and the oracle. These preconditions all held before `core/` work
   began; the code that now lives here is the result.

---

## Original interface sketch (now realized in [`include/clozn/model.hpp`](include/clozn/model.hpp))

> This block was the **uncompiled picture** drawn before the seam existed. The shipped seam now lives
> in `include/clozn/model.hpp` (`ModelConfig` / `ForwardResult` / `ModelAdapter`) and the ggml
> implementation in `src/model_ggml.cpp`; they differ in surface detail from this sketch (the shipped
> `ForwardResult` owns its logits in a `std::vector` rather than a raw device pointer — the
> confidence-select kernel hand-off that the pointer anticipated is the still-pending Phase-2 wiring).
> Kept here for the record of how the shape was reasoned out. The Python seam
> ([`base.py`](../lab/clozn_lab/models/base.py)) and the goldens remain the source of truth.

```cpp
// core/include/clozn/adapter.hpp  —  UNCOMPILED SKETCH (Phase 3, not started)
// Mirrors lab/clozn_lab/models/base.py::ModelAdapter one-for-one.
namespace clozn {

struct ModelConfig {                 // == base.py ModelConfig (GGUF diffusion.* keys)
    Family   family;
    int32_t  vocab_size;
    int32_t  mask_token_id;
    int32_t  eos_token_id;           // -1 == none
    int32_t  block_length;           // 0 == whole-sequence
    AttnKind attn;                   // bidirectional | block_causal
};

struct ForwardResult {               // == base.py ForwardResult
    const float* logits;             // device-resident [n_logits, vocab]; feeds the kernel
    int32_t      n_logits;
    KvHandle     kv;                 // opaque; scheduler holds it, never inspects it
};

// Invariant 4: reads board, never writes it; computes KV only for `recompute`;
// returned kv covers every board position. The one-way-law mask is built by the
// scheduler (blocks.cpp) and passed in.
class ModelAdapter {
public:
    virtual const ModelConfig& config() const = 0;
    virtual ForwardResult forward(
        const llama_token* board, int n,
        const AttnMask&    mask,
        KvHandle           kv,            // null == cold start
        const int*         recompute, int n_recompute,   // null == all
        const int*         logits_for, int n_logits_for  // null == all
    ) = 0;
    virtual std::vector<llama_token> encode(std::string_view text) = 0;
    virtual std::string decode(const llama_token* ids, int n) = 0;
    virtual ~ModelAdapter() = default;
};

} // namespace clozn
```

---

## Competitive position (verified 2026-06-15)

- **It is not too late** — but the *easy* half ("run a diffusion LM locally at all") is converging to
  commodity. llama.cpp diffusion support sits in **unmerged PRs**; **Ollama cannot load dLLMs yet.**
  When those land, "first to run a diffusion model locally" stops being a differentiator.
- **The differentiation is scheduler depth, not the loader.** Clozn's defensible value is the L1 scheduler the
  lab already implements: the model-agnostic seam, the Tier A/B/C cache with an *exposed exactness
  knob* and a shipped divergence bench, adaptive stopping with guard rails, token revision, and
  infill. `core/` exists to carry that scheduler — already built and tested in `lab/` — onto the fast
  ggml backend.

The lab's tok/s figures are a **PyTorch reference** and are **transfer-bound** (~87% of GPU time is
the per-step logits transfer, DESIGN §4.3). The real tok/s arrives with the Phase-2 kernel and this
Phase-3 ggml runtime — that is the entire point of `core/`. Until then, the robust signals are
**steps/token** (< 1 is the whole dLLM premise) and the **quality/divergence columns**. Only real,
committed runs appear anywhere in this repo.
