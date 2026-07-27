# confidence-select kernel

The one new kernel Clozn ships (DESIGN.md §4.3). It fuses, **per masked
position** in a denoise step:

1. sample a token (greedy argmax, or a draw from the temperature/top_p-shaped
   softmax),
2. compute that pick's **confidence** (max-prob / margin / negative-entropy),
3. **select** which positions to commit this step (top-k by confidence, or those
   clearing a threshold τ with a min-one rail),

so only `2 × n_masked` ints/floats cross GPU→host per step instead of the full
`[n_masked × vocab]` logits buffer.

## Why it exists

The naive diffusion loop ships every masked position's full-vocab logits to the
CPU each step. The upstream llama.cpp diffusion PR measured that transfer at
**~87% of GPU wall time**. Doing sample + confidence + selection device-side
collapses it to a token id and a confidence per position — roughly a
**10,000×** smaller transfer for real-vocab models (vocab ≈ 32k–150k).

## The §4.3 contract

```
inputs : logits [n_masked, vocab] (device-resident),
         temperature, top_p, k_commit (or threshold τ), rng state
outputs: per masked position -> (sampled_token_id, confidence) ;
         plus the indices of the selected positions
transfer to host: 2 × n_masked ints + floats   (~10,000× smaller)
```

Confidence variants (DESIGN open question #3, mirroring open-dCoder's
`sample_tokens`):

| variant       | definition                                  |
|---------------|---------------------------------------------|
| `max_prob`    | probability of the sampled token (default)  |
| `margin`      | top1 − top2 probability                     |
| `neg_entropy` | Σ p·log p (already negative; higher = peakier)|

Selection variants (mirroring `clozn_lab.scheduler.policies`):

- **top-k** (`k_commit`): the `min(k, n_masked)` highest-confidence positions,
  ties broken toward the **lower** index.
- **threshold** (`tau`, `min_commit`): every position with `conf ≥ τ`; if fewer
  than `min_commit` clear τ, commit the top `min_commit` instead (the
  min-one-commit progress rail).

## What's in here

| file | status | what it is |
|---|---|---|
| `reference.py` | **TESTED oracle** | self-contained numpy implementation of the full contract. Does not import `clozn_lab`. |
| `test_reference.py` | **TESTED** | pytest suite: contract/shape, greedy, top_p, confidence variants, selection, and parity against the lab. |
| `confidence_select.cuh` / `.cu` | **greedy paths validated** | CUDA implementation of the contract. Compiled on RTX 5080 / CUDA 13.3 (`sm_120`); its deterministic paths match `reference.py` exactly (see `validate.py`). The sampled (curand) path and the `top_p` filter are still scaffold/unverified. |
| `validate.cu` / `validate.py` | **validation harness** | feed identical logits to the CUDA kernel and the numpy reference and diff picks / selected / confidences across every deterministic combo. |
| `bench.cu` | **microbenchmark** | times the per-step host-handoff the kernel replaces (full-logits D2H vs on-device kernel + tiny D2H) at real model scales. |
| `test_main.cpp` | scaffold | tiny C++ smoke driver for the optional CMake target. |
| `CMakeLists.txt` | builds + validates | builds the `.cu` + `cs_test` + `cs_validate` when a CUDA compiler exists; falls back to the host-only stub otherwise. |

`reference.py` is the source of truth. The CUDA kernel's **deterministic paths are now
validated** against it on an RTX 5080 (CUDA 13.3, `sm_120`): all three confidence
variants × top-k and threshold selection produce identical token picks and selected
indices, confidences within float32-vs-float64 epsilon (~1e-9 to 1e-6). Still
**unverified** (scaffold): the sampled path — curand cannot bit-match numpy's RNG — and
the `top_p` nucleus filter (a TODO stub in the `.cu`). Treat any disagreement between a
deterministic `.cu` path and `reference.py` as a bug in the `.cu`.

This kernel lives **in-repo** for now. The eventual llama.cpp / Ollama upstream
PR (DESIGN §4.3) is deliberately **not** a blocker for Phase 2 — it ships here,
tested on CPU, and is upstreamed when the GPU path is verified.

## Running the tests

```
python -m pytest test_reference.py -q   # from kernels/confidence_select, with the project venv active
```

The parity tests (group `f`) import `clozn_lab` (numpy-only paths) and assert the
reference reproduces `generate.sample_candidates` (token ids + confidences,
greedy and sampled, seeding both sides identically) and the `ConfidenceTopK` /
`Threshold` policy selections exactly. If a parity test fails, fix
`reference.py` to match the lab — never the reverse.

## Validating the CUDA kernel (needs a CUDA toolchain)

From a developer shell with `nvcc` + `cl` + `cmake` on PATH (on Windows, after
`vcvars64.bat`):

```
cmake -S . -B build -G "NMake Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build
python validate.py build/cs_validate      # ALL PASS on RTX 5080 / sm_120
```

`validate.py` feeds one fixed set of logits to both the CUDA kernel and `reference.py`
and diffs them across every **deterministic** combo (greedy; max_prob / margin /
neg_entropy; top-k and threshold). The sampled path and `top_p` are scaffold and
intentionally not asserted — `reference.py` remains their oracle until they're finished.

## Measured host-handoff (RTX 5080, `cs_bench`)

The win this kernel exists for is removing the per-step logits transfer. `cs_bench` times
**A**: copy the full `[n_masked × vocab]` logits to host (what the naive loop transfers),
vs **B**: run the kernel on-device and copy back only `2 × n_masked` values:

| model | n_masked × vocab | A: full-logits D2H | B: kernel + tiny D2H | speedup | data moved |
|---|---|---|---|---|---|
| Dream 7B | 32 × 152064 | 0.893 ms | **0.350 ms** | **2.6×** | 18.6 MB → 256 B |
| LLaDA 8B | 32 × 126464 | 0.744 ms | 0.305 ms | 2.4× | 15.4 MB → 256 B |
| llama-32k | 16 × 32000 | 0.127 ms | 0.112 ms | 1.1× | 2.0 MB → 128 B |

The **data-movement** reduction (~10,000× fewer bytes) is structural and exact. The
**wall-time** gain depends on vocab size — it's real at LLM scale and modest below it.
Two honesty notes: (1) baseline **A only times the transfer**, while **B also does the
sample/confidence/select** that A's path would then run on the CPU — so these ratios are
*conservative*. (2) This is the host-handoff step in **isolation**, not end-to-end tok/s;
how much it moves total throughput depends on the model forward's share of wall time (the
upstream llama.cpp diffusion PR measured the handoff at ~87% for their setup). Only the
`max_prob` greedy path is block-parallel-optimized so far; `margin`/`neg_entropy` still use
a single-thread confidence loop.
