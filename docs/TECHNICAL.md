# Cloze — Technical Deep-Dive

> **Historical diffusion-research archive.** The measurements and implementation notes below preserve
> the original research record. They do not describe a current Clozn product substrate; see
> [CAPABILITIES.md](CAPABILITIES.md).

> **Read this for a detailed engineering account.** It states exactly what Cloze is today, what it
> isn't, what the numbers mean, and where they
> came from. Every figure here is a real, committed run — the JSONL event logs that produced the
> tables live in [`lab/cloze_lab/bench/results/`](../lab/cloze_lab/bench/results) and are replayable.
> Where something is a reference implementation, a scaffold, or roadmap, it says so in the same
> sentence.

If you came for the architecture, the source of truth is [`docs/DESIGN.md`](DESIGN.md). This doc is
the *honest engineering account* of the parts most likely to be scrutinized: the bandwidth argument,
the KV-cache tiers and their exactness, the one new kernel, a real correctness bug we hit, and one
genuinely open research question.

---

## 0. What Cloze is right now (no hedging)

Cloze today is a **Python reference runtime** — Phase 1, everything under [`lab/`](../lab). It is a
PyTorch/Transformers playground that runs real diffusion LM checkpoints behind a clean adapter seam,
with a complete, tested denoising scheduler on top.

Cloze today is **not** the C++/ggml runtime. That is Phase 3; the [`core/`](../core) directory holds
the in-progress port — the scheduler and ggml adapter are there and validated against the lab
goldens, but it is not yet the shipped product. The README's "local-first runtime" framing is the
destination, not a description of the current binary.

What the reference runtime actually contains, working and tested:

- The `ModelAdapter` seam ([`models/base.py`](../lab/cloze_lab/models/base.py)) — the one place
  PyTorch/Transformers may be imported, so the scheduler is pure logic.
- Three real model families behind that seam: **Dream 7B**, **LLaDA 8B**, and **open-dCoder 0.5B**
  (the tiny CPU-CI checkpoint, a Dream-family sibling), plus a torch-free `FakeAdapter` oracle.
- The scheduler: typed event spine, `confidence_topk` + `threshold` unmask policies, fixed/adaptive
  stepping with guard rails, the block manager (block-causal "one-way law"), all three KV-cache
  tiers (A/B exact, C approximate), plus token revision and infill — pinned by seeded golden fixtures.
- A divergence + speed bench harness that A/Bs cache `off` vs `delta` and prints the quality column
  next to the speed column.
- A self-contained **CPU reference** of the confidence-select kernel
  ([`kernels/confidence_select/reference.py`](../kernels/confidence_select/reference.py)), fully
  tested — plus the CUDA kernel, whose **deterministic paths are compiled and validated** against
  that reference on an RTX 5080 (`sm_120`); the sampled path and `top_p` remain scaffold (§4.3).

CI runs the whole suite on CPU against open-dCoder. The FakeAdapter oracle runs everywhere.

**Competitive position, stated plainly (verified 2026-06-15).** It is *not* too late, but the easy
half — "run a diffusion LM locally" — is converging to commodity. llama.cpp has unmerged diffusion
PRs in flight; Ollama can't load dLLMs yet, but that gap is a matter of time, not depth. So Cloze
does not position on *being first to run diffusion locally*. The differentiation is **scheduler
depth**: the model-agnostic seam, the Tier A/B/C cache with an exposed exactness knob and a
divergence bench that proves the tradeoff, adaptive stopping, token revision, and native infill.

---

## 1. Why diffusion LMs can beat autoregressive on bandwidth

### 1.1 The bandwidth argument (the real one)

Local autoregressive (AR) inference is **memory-bandwidth-bound**, not compute-bound. At batch size
1 — the local single-user case — generating one token requires reading the entire weight matrix once
from memory into the compute units. The arithmetic is cheap; the *transfer of the weights* dominates.
This is why local LLM tok/s tracks memory bandwidth far more closely than FLOPS.

A masked-diffusion LM changes the unit of work. Each forward pass advances **all active masked
positions in parallel** and the scheduler commits *several* of them per pass. The expensive thing —
the full read of the weights — happens once per forward pass but yields more than one committed
token. So the metric that matters is:

> **tokens committed per forward pass** — equivalently, **steps/token < 1.**

If a dLLM commits ~2 tokens per pass, it pays roughly half the weight-reads per token an AR model
pays, and on bandwidth-bound hardware that is close to a 2× ceiling on tokens/sec — before any
caching, with bidirectional attention thrown in for free (which AR architectures structurally cannot
do; see §6). This shows up directly in our committed runs as `steps/tok`:

| run | steps/tok | reading |
|---|---|---|
| open-dCoder 0.5B, CPU, `max_new=32 steps=4 block_len=8` | **0.52** | ~1.9 tokens committed per forward |
| Dream 7B, RTX 5080, nf4, `max_new=48 steps=4 block_len=8` | **0.52** | ~1.9 tokens committed per forward |

An AR model spends exactly 1.0 steps/token by construction. `steps/tok` is the honest,
implementation-independent signal of the bandwidth win: it counts weight-reads per token regardless
of how fast any particular kernel is.

### 1.2 The honest caveat: our tok/s is transfer-bound, not the product number

Here is the part most speed posts in this niche leave out, and the part that gets you correctly torn
apart on HN if you don't say it first.

**The lab's absolute tok/s is a PyTorch reference number, and it is transfer-bound.** The naive
denoise loop in [`generate.py`](../lab/cloze_lab/generate.py) computes logits for the masked positions
and ships them GPU→host **every step** so the CPU-side sampler (`sample_candidates`) can pick tokens.
The upstream llama.cpp diffusion PR measured that full-logits transfer at **~87% of GPU wall time**
(DESIGN §4.3). Our loop has the same shape. So our tok/s is dominated by a transfer that the
*product* is explicitly designed to delete (the confidence-select kernel, §4) — it is not a
measurement of how fast diffusion decoding can be.

Concretely, the committed runs: open-dCoder, CPU: 40.9 → 53.5 tok/s (`off` → `delta`); Dream 7B nf4:
27.1 → 40.3 tok/s. Do **not** read "27.1 tok/s on a 5080" as "diffusion is slow" or as a product
claim. Read it as "the PyTorch reference loop, ferrying full logits to host every step, gets 27.1
tok/s, and that bottleneck is exactly what Phase 2 removes." Until the kernel and the ggml runtime
exist, **trust `steps/tok` and the quality/divergence columns, not the absolute tok/s.**

---

## 2. The KV-cache tiers and the one-way law

The speedups beyond raw parallelism live in the cache. Cloze's cache has three tiers of decreasing
exactness; the boundary between "exact and free" and "approximate" is set by one structural property
of block-diffusion attention. Implementation: [`scheduler/cache.py`](../lab/cloze_lab/scheduler/cache.py)
(the planner, pure logic) and the prefix-reuse path in [`models/dream.py`](../lab/cloze_lab/models/dream.py).

### 2.1 The one-way law

Semi-autoregressive block diffusion generates the output left→right in blocks of length L. *Within*
the active block, attention is fully bidirectional. *Across* blocks it is **block-causal**: an active
block attends to itself and all earlier blocks, **never forward**. That asymmetry is the **one-way
law**, built in [`scheduler/blocks.py`](../lab/cloze_lab/scheduler/blocks.py):

```python
# attention_mask(): M[q, k] = block_id(k) <= block_id(q)
return ids[None, :] <= ids[:, None]
```

The consequence is the whole reason Tiers A and B can be exact: a frozen block's attention pattern —
and therefore its K/V — is *identical whether or not later blocks exist*, because later blocks are
invisible to it. Freezing a block changes nothing it can see, so its drawers never need recomputing.

### 2.2 Tier A — prompt cache (exact)

The prompt never changes. Prefill once, reuse forever; identical to ordinary AR prompt caching, and
shareable across requests with common prefixes. Zero approximation.

### 2.3 Tier B — frozen-block cache (exact, and free)

When a block finalizes, its K/V become immutable and append to the prefix. The subtlety, handled in
`cache.py`, is that the freeze costs **no extra forward pass**: a block's first forward is always
`block_step == 0`, a full refresh, which recomputes the just-finalized block exactly once — folded
into the next block's first forward — after which `observe()` advances the `_frozen_until` boundary
and it is never recomputed again.

A real implementation constraint surfaces in the Dream adapter: its K/V cache is a Transformers
`DynamicCache`, which is **append-only**. So the adapter supports only **contiguous-suffix** reuse —
reuse an exact prefix, recompute a contiguous tail. That is precisely Tier A/B. It is *not* enough
for Tier C's scattered mid-sequence recompute, and the adapter raises `NotImplementedError` rather
than silently doing the wrong thing. In block mode this is exactly what the cache asks for, so it
composes correctly. (The general scattered-recompute Tier C is expressible against the seam and the
FakeAdapter exercises it in tests; making a *real* append-only cache do it is a future-adapter
concern — and the main ggml-side task of Phase 3.)

### 2.4 Tier C — intra-block delta cache (the big win and the accuracy risk)

Within a block, each denoise step changes only a few token ids. The delta scheme recomputes only the
positions whose token changed and reuses the rest:

```python
# cache.py: plan()
new     = {p for p in range(n) if p not in self._cached_token} - frozen
changed = {p for p in self._cached_token if int(board[p]) != self._cached_token[p]} - frozen
churn   = len(changed & active_positions) / len(active_positions)
full    = block_step % full_refresh_every == 0 or churn > refresh_fraction
recompute = (set(range(n)) - frozen) if full else (new | changed)
```

**Why it's approximate, stated precisely.** When position *j*'s token changes, the hidden states —
and therefore the true K/V — of every other position in the block shift slightly, because
bidirectional attention let them attend to *j*. The delta cache reuses their stale K/V anyway. Two
rails bound the drift: a periodic full refresh (`full_refresh_every`) and a churn trigger
(`refresh_fraction`). `full_refresh_every=1` makes `delta` behaviourally identical to `off`. The
knobs are a frozen dataclass — exposed, never hidden (DESIGN invariant 5).

---

## 3. The exactness knob and the off-vs-delta divergence methodology

Honesty here is not a vibe; it is a method, and it is shipped as code.

### 3.1 The method

The divergence harness ([`bench/divergence.py`](../lab/cloze_lab/bench/divergence.py)) is a *consumer
of the event stream* — it compares two finished runs over identical inputs and never reruns the
model. You generate twice (cache `off` = exact baseline, cache `delta` = the fast variant) and
compare the output region: exact-match %, decoded-text identity, and a mean-confidence delta. The
bench CLI does the A/B in one command (`cloze bench --model … --block-len …`) and emits a markdown
table for sharing results.

### 3.2 The committed numbers

**open-dCoder 0.5B — CPU, `max_new=32 steps=4 block_len=8`**
([results/dcoder_mn32_s4_bl8.md](../lab/cloze_lab/bench/results/dcoder_mn32_s4_bl8.md)):

| cache | forwards | cache-hit | new tok | steps/tok | tok/s | token-match | text-match | conf Δ |
|---|---|---|---|---|---|---|---|---|
| off (exact) | 12 | 0% | 23 | 0.52 | 40.9 | baseline | n/a | n/a |
| delta(refresh=1) | 12 | 56% | 23 | 0.52 | 53.5 | 100.0% | yes | −0.003 |

**Dream 7B — RTX 5080, nf4, `max_new=48 steps=4 block_len=8`**
([results/dream_mn48_s4_bl8_nf4.md](../lab/cloze_lab/bench/results/dream_mn48_s4_bl8_nf4.md)):

| cache | forwards | cache-hit | new tok | steps/tok | tok/s | token-match | text-match | conf Δ |
|---|---|---|---|---|---|---|---|---|
| off (exact) | 12 | 0% | 23 | 0.52 | 27.1 | baseline | n/a | n/a |
| delta(refresh=1) | 12 | 73% | 23 | 0.52 | 40.3 | 100.0% | yes | −0.000 |

In both runs `delta` reuses 56–73% of the drawers at **100.0% token-match, identical decoded text,
and confidence delta ≈ 0** — exact reuse at zero quality cost, proven in the same row. (Read the
tok/s deltas as "the cache helps even in the transfer-bound reference loop," not as product
throughput.)

### 3.3 The honest finding the divergence column exists to catch

Tier A/B reuse is **algorithmically exact** — the reused K/V is mathematically the same values a full
recompute would produce. But "algorithmically exact" is not "bitwise identical on every device."
Under **nf4** 4-bit weights, floating-point noise in the dequantize/matmul path can flip the pick on
a **near-tie token** — two candidates whose post-softmax probabilities are within rounding distance.

The committed `steps=4` runs above land at 100.0% token-match. But a committed **`steps=8` run dips
to 97.9% token-match**
([results/dream_mn48_s8_bl8_nf4.md](../lab/cloze_lab/bench/results/dream_mn48_s8_bl8_nf4.md)): a
handful of near-tie flips. Nothing is wrong with the cache; the reuse is still exact in exact
arithmetic. More steps means more commits means more chances for a near-tie to land on the wrong side
of fp rounding under 4-bit weights.

This is the entire reason the divergence column exists. Without it, a 97.9% would be invisible and
we'd be claiming "exact" while shipping 2.1% drift on long runs. With it, the dip is a number in a
committed table that anyone can reproduce, and we can say precisely *why* it happened (fp near-tie
flips under nf4). This is the same principle as DESIGN invariant 3, which forbids asserting
bitwise equality on confidence sums (float reduction order differs across devices) — the divergence
bench is that principle at the run level.

> **The exactness claim, stated carefully:** delta reuse is *algorithmically exact*; the divergence
> column is how you verify it stayed exact *on your hardware and quant*, and surfaces the rare cases
> (like nf4 near-ties on long runs) where floating-point reality dips below 100%. We ship the
> measurement, not just the claim.

---

## 4. The confidence-select kernel (the one new kernel)

### 4.1 What it removes

The naive loop's ~87% transfer cost (§1.2) comes from shipping `[n_masked × vocab]` float logits
GPU→host every step. The confidence-select kernel fuses, **on-device, per masked position per step**:
(1) **sample** a token (greedy argmax, or a draw from the temperature/top_p-shaped softmax), (2)
compute that pick's **confidence** (`max_prob` default, or `margin`, or `neg_entropy`), (3) **select**
which positions to commit (top-k by confidence, or those clearing τ with a min-one rail). So only
`2 × n_masked` ints/floats cross the bus per step instead of the full logits buffer — roughly
**10,000× smaller** for real-vocab models. That is the ~87% the kernel is designed to delete.

### 4.2 The contract (DESIGN §4.3, verbatim)

```
inputs : logits buffer [n_masked, vocab] (device-resident),
         temperature, top_p, k_commit (or threshold τ), rng state
outputs: per masked position → (sampled_token_id, confidence) ;
         plus the indices of the top-k_commit positions by confidence
transfer to host: 2 × n_masked ints + floats   (≈10,000× smaller)
```

### 4.3 Status: CPU reference tested; CUDA deterministic paths validated on GPU

[`kernels/confidence_select/reference.py`](../kernels/confidence_select/reference.py) is a
self-contained numpy reference oracle — it deliberately does *not* import `cloze_lab`, so the
correctness contract lives independently and can later validate a Metal/CUDA port the same way the
golden tests validate the scheduler. It is **tested**: shapes, greedy/sampled paths, top_p nucleus
filtering, all three confidence variants, both selection variants, and — crucially — **parity**
against the lab's own `generate.sample_candidates` and the `ConfidenceTopK` / `Threshold` policies,
seeded identically on both sides.

The CUDA files (`confidence_select.cu` / `.cuh`) now **compile for Blackwell** (`sm_120`, CUDA 13.3)
and their **deterministic paths are validated** against `reference.py` on an RTX 5080 via `validate.py`:
all three confidence variants × top-k/threshold selection, with token picks and selected indices
matching **exactly** and confidences within float32-vs-float64 epsilon (~1e-9 to 1e-6). Still
scaffold/unverified: the sampled path (curand can't bit-match numpy's RNG) and the `top_p` nucleus
filter (a TODO stub). And the distinction we keep carefully: this validates the kernel's **correctness
on GPU**, not yet the end-to-end **speedup** — the ~87% transfer saving is realized only once the
kernel is wired into a real GPU generate loop (and the Phase-3 ggml runtime). So today: the contract
is specified, the CPU oracle tested, and the GPU kernel's deterministic paths correctness-validated on
real Blackwell hardware.

We *can* measure the host-handoff step in isolation, though, and `cs_bench` does (RTX 5080): copying
the full `[32 × 152064]` Dream-scale logits to host takes **0.89 ms** (18.6 MB), while running the
kernel on-device and copying back the `2 × 32` results takes **0.35 ms** (256 B) — **2.6× faster**,
~76,000× less data. That's conservative (the baseline only times the transfer; the kernel also does
the select) and it's the handoff *in isolation*, not end-to-end tok/s — but it's the structural win,
measured, not asserted. (Optimization note: a first-cut single-thread argmax made the kernel *slower*
than the transfer; a block-parallel argmax reduction is what turned it into a 2.6× win — the kind of
thing you only learn by compiling and timing it.)

**End-to-end share** ([`bench/e2e_handoff.py`](../lab/cloze_lab/bench/e2e_handoff.py), CUDA): in a
real generation the handoff competes with the model forward, so the honest question is its *share* of
wall time. Measured on the 5080: **open-dCoder 0.5B → 13.9%** of wall (forward ~36.5 ms/step), **Dream
7B nf4 → 8.1%** (forward ~66.5 ms/step). Removing the handoff projects roughly **+16% / +9%** tok/s
respectively (upper bound — the kernel's own cost is far below the 5.9 ms handoff). The surprise: at a
block size of 8 the **D2H transfer is tiny (0.27 ms)**; the handoff is dominated by the **CPU
`sample_candidates` step (5.6 ms** of float64 softmax over `[8 × 152k]`). So the kernel's end-to-end
value here is mostly *moving the sample/select onto the GPU*; the transfer reduction grows with
`n_masked` (whole-sequence mode) and as the forward gets faster (the C++ path). Bottom line, stated
honestly: for a slow 7B-nf4 forward the kernel is a **single-digit-percent** end-to-end win, not a major
multiplier — the multiplier is on the *handoff step itself*, and the e2e gain scales up where
the forward is cheaper.

---

## 5. A real bug: stock Qwen2 silently makes bidirectional reuse causal

This is the kind of bug that is invisible until your divergence column catches it. open-dCoder 0.5B
is a Dream-family masked-diffusion model distilled into a **stock `Qwen2ForCausalLM`** — an
autoregressive architecture. When you call its forward with **`attention_mask=None` *and* a non-empty
KV cache**, Qwen2 falls back to building a **causal** mask over past+current positions — the correct
default for an AR model, and exactly wrong for us.

The failure is silent: with no explicit mask, the moment you enable KV reuse, the model quietly
switches the active block from bidirectional to causal attention. That breaks the one-way law in the
wrong direction, corrupts the reused prefix, and yields subtly wrong tokens with no error — the kind
of thing that shows up only as a divergence dip if you're measuring, and as nothing at all if you're
not.

The fix, in [`dream.py:_additive_mask`](../lab/cloze_lab/models/dream.py): **always emit an explicit
4-D additive mask, never `None`.** An all-zero (all-visible) mask forces true bidirectional attention
through Qwen2's mask machinery. For Dream proper (whose modeling code forwards the mask verbatim and
is hard-coded non-causal) the explicit all-zero mask is identical to `None`, so making the mask
always-explicit costs nothing on Dream and *fixes* open-dCoder. Verified bitwise-exact on both
families. The lesson generalizes: a diffusion adapter wrapping an AR base class must never let the
base class infer the mask — infer nothing, state everything.

---

## 6. Infill, for completeness on the bidirectional claim

Bidirectional attention also buys a capability AR models structurally cannot have: native **infill**.
Cloze's `infill()` ([`generate.py`](../lab/cloze_lab/generate.py)) lays the board out as
`prefix + [MASK]*gap + suffix` and denoises the masked middle under **full bidirectional attention**,
so every filled slot sees the fixed right-context (`suffix`) as well as the left. Whole-sequence and
full-recompute (exact) — infill is a one-shot fill where correctness, not cache reuse, is the point.
An AR model cannot do this without retraining tricks; for a dLLM it's the same forward pass with a
different mask. (`cloze infill "def add(a, b):" " return result" --gap 6 --model dcoder` →
`def add(a, b): result = a + b; return result`.)

---

## 7. The open research question: quantization × confidence calibration

This is a genuinely open question (DESIGN §9), and we're not going to pretend we've closed it.

**The worry.** Low-bit quantization perturbs the logit distribution. Unmasking policies *consume
confidences* — `confidence_topk` commits the most confident positions; `threshold(τ)` commits
everything over τ. So if quantization *miscalibrates* confidences, it changes *which tokens commit
early*, and early commits are load-bearing because later positions attend to them. The hypothesis is
that quantization might hurt dLLMs differently than it hurts AR models, which don't gate on a
confidence dial the same way.

**The planned study.** Sweep `{f16, q8, q5_k, q4_k, iq3} × {confidence, entropy}` policies on the
eval suites; measure quality, steps-to-converge, and revision rates. If miscalibration shows up, the
cheap fix is a per-quant temperature rescale on the confidences, stored as GGUF metadata
(`diffusion.conf_temp`).

**Preliminary finding (not the full study).** Early signal suggests **q4 does NOT systematically
break the confidence dial — *once you control for teacher-forcing bias*.** The control matters and is
the easy thing to get wrong: if you measure calibration by teacher-forcing the model along the f16
trajectory, you conflate the quant's logit perturbation with trajectory divergence and manufacture an
apparent miscalibration that isn't there. Controlling for that, q4's confidences track f16's closely
enough that the commit *order* — all the policy consumes — is largely preserved. Consistent with
this, our committed Dream 7B **nf4** run shows confidence delta ≈ 0.000 and 100% token-match vs. the
exact baseline at `steps=4` (§3.2). Caveats up front: this is one quant (nf4 via bitsandbytes, a
*lab* knob — the product quantizes via GGUF/ggml), at short step counts, on small prompts; the §3.3
nf4 near-tie flips at `steps=8` are a reminder that "doesn't break the *dial*" is not "bitwise
identical *outputs*." The full sweep is roadmap; these findings are preliminary, not published.

---

## 8. Limitations and honest answers

| If you're skeptical that… | …here's the honest answer |
|---|---|
| "diffusion is faster" is hype | The bandwidth win is `steps/tok < 1` (0.52 in our runs ≈ ~2 tokens/forward). That's real and implementation-independent. The *absolute tok/s* we report is a transfer-bound PyTorch reference, not the product number — §1.2. |
| the cache cheats on quality | Every speed row ships the divergence column (exact-match %, text identity, conf Δ). `off` is exact; `delta` is algorithmically exact and *measured* at 100% token-match at steps=4 — with the honest nf4 long-run dip to 97.9% committed and explained in §3.3. |
| the kernel speedup is vaporware | The kernel *contract* and a *tested CPU reference* exist, and the CUDA kernel now *compiles and its deterministic paths are validated* against the reference on an RTX 5080 (sm_120). That's correctness on GPU; the *realized* ~87% transfer saving still needs the kernel wired into a GPU generate loop (Phase 2→3), and we say so. |

---

## 9. Where the code is

```
docs/DESIGN.md                          architecture (source of truth)
lab/cloze_lab/
  models/base.py                        ModelAdapter seam (the one PyTorch boundary)
  models/dream.py                       Dream / open-dCoder adapter (Qwen2 mask fix, §5)
  scheduler/blocks.py                   block-causal one-way law (§2.1)
  scheduler/cache.py                    Tier A/B/C planner + exactness knob (§2)
  scheduler/policies.py                 confidence_topk / threshold / remask_lowconf
  generate.py                           the pass loop + infill (§6)
  bench/divergence.py                   the honesty column (§3)
  bench/results/                        committed runs + replayable JSONL logs
kernels/confidence_select/reference.py  tested CPU oracle of the kernel (§4)
kernels/confidence_select/*.cu,*.cuh    unverified CUDA scaffold (§4.3)
core/                                   port plan only — empty of code until Phase 3
```
