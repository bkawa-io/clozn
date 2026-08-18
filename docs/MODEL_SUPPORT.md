# Model support and qualification

**Thesis:** model support is an evidence ladder, not a yes/no label. Clozn's core is built around
autoregressive GGUF capabilities, while white-box writes, J-lenses, and SAEs require increasingly
model-specific evidence or artifacts.

> **Status (2026-07-17):** engine-side chat templating and cross-family Tier-0 core support have shipped.
> The checked-in [Wave 1 qualification ledger](qualification/wave1.json) records CPU basic/deep smoke on
> Qwen 2.5, Llama 3.1, Qwen 3.5, Gemma 4, and Ministral 3. It does **not** claim that every white-box write
> or optional artifact is qualified on every family. Open model work lives in [BACKLOG.md](BACKLOG.md) §2.

## What “supports a model” means

The ordered qualification levels are the ones serialized in `docs/qualification/wave1.json`:

| Level | What passed | Per-model work |
|---|---|---|
| **Discovered** | GGUF identity, architecture, dimensions, tokenizer, and embedded chat template can be read | no training |
| **Core** | chat, OpenAI/native streaming, run persistence, scoring, receipts, worker restart, and cleanup | run basic + deep smoke |
| **White-box** | activation taps, steering writes, and teacher-forced scoring work at valid layers for this architecture | targeted read/write qualification |
| **Lens-qualified** | a model-scoped J-lens manifest and payload passed identity, checksum, dimension, and quant-transfer checks | offline fit + transfer qualification |

SAE concept atlases are a separate bespoke feature: they require a compatible trained SAE and are never
implied by core or J-lens qualification.

“Any AR GGUF” therefore names the **core runtime contract**, not a claim that every file on the internet
has already been tested. A candidate still needs to load in the pinned engine, contain a usable embedded
chat template, and pass the appropriate qualification rung. Historical LLaDA/Dream work is not a
current product lane; its measurements remain in the archived diffusion design documents.

## Claims and evidence

| Claim | Implementation | Repeatable check or recorded measurement | Boundary |
|---|---|---|---|
| The product applies the loaded GGUF's own chat template | `engine/core/serve/chat_template_renderer.cpp`, `routes_whitebox.cpp:/apply_template`, `clozn/server/app.py:_engine_tmpl` | `tests/test_engine_apply_template.py` (5 model-free seam checks); Wave 1 records full-Jinja Gemma 4 deep smoke | a GGUF without a valid embedded template fails visibly; there is no silent Qwen fallback |
| The product path is Torch-free | `clozn/server/` + the private C++ worker | `.github/workflows/ci.yml:product-minimal`; `tests/test_runtime_architecture.py` verifies the product/lab boundary | lab fitting and research still use PyTorch |
| Core support crosses model families | the same gateway/worker and smoke gate for every model | Wave 1: Qwen 2.5 deep **26/26**; Llama 3.1, Qwen 3.5, Gemma 4, and Ministral 3 basic **24/24** + deep **26/26**, CPU | four non-Qwen rows still list targeted white-box write tests as pending |
| J-lens application is engine-native | `engine/core/serve/routes_jlens.cpp`; gateway proxy in `clozn/server/routes/receipts.py` | `tests/test_jlens_server.py` (12 contract/degradation checks); Wave 1 marks the exact Qwen2.5-7B Q4_K_M digest `qualified_q4_k_m`; the live 96–99% top-1 engine-vs-numpy result is recorded in `docs/ROADMAP.md` | the checked-in qualified fit covers Qwen2.5-7B only; a lens read is not causal proof |
| Sampling metadata describes the sampler that ran | `clozn/server/app.py:_engine_generation_meta`; `engine/core/src/sample.cpp` | `tests/test_engine_substrate.py` and `tests/test_engine_stream.py` check the full temperature/top-k/top-p/repetition/seed handoff | receipts and forced replay explicitly use the separate greedy path |

Exact checkpoint identity, GGUF SHA-256, tokenizer/template digests, dimensions, and per-rung status live in
the qualification ledger rather than prose so a marketing sentence cannot silently broaden the evidence.

## Current qualification roster

| Family / exact checkpoint | Core | White-box | J-lens |
|---|---|---|---|
| Qwen 2.5 — `Qwen/Qwen2.5-7B-Instruct` | deep CPU 26/26 | partial | exact Q4_K_M qualified |
| Llama 3.1 — `meta-llama/Llama-3.1-8B-Instruct` | basic 24/24 + deep 26/26 CPU | native probe startup passed; targeted writes pending | pending |
| Qwen 3.5 — `Qwen/Qwen3.5-9B` | basic 24/24 + deep 26/26 CPU | native probe startup passed; targeted writes pending | pending |
| Gemma 4 — `google/gemma-4-E4B-it` | basic 24/24 + deep 26/26 CPU; full Jinja template passed | native probe startup passed; targeted writes pending | pending |
| Ministral 3 — `mistralai/Ministral-3-3B-Instruct-2512` | basic 24/24 + deep 26/26 CPU | native probe startup passed; targeted writes pending | pending |

This table summarizes `wave1.json`; the JSON is authoritative when prose and the ledger disagree.

## Remaining model coupling

1. **Registry coverage is incomplete.** `clozn/server/substrates.py` recognizes the Wave 1 families, but
   model defaults and research paths still contain Qwen-specific literals. These must move behind one
   model identity/config contract instead of growing more filename heuristics.
2. **White-box portability needs targeted writes.** Core smoke is cross-family; tap/write/score parity is
   not yet fully qualified for the four non-Qwen Wave 1 rows. This is the honest boundary on “any GGUF.”
3. **J-lenses are fit per model.** Application is forward-only C++, but fitting requires the PyTorch lab
   and every product quant needs transfer evidence. The Qwen2.5-7B qualification record proves the path,
   not a universal artifact.
4. **SAEs remain bespoke.** The existing Qwen SAE/concept stack does not generalize without a compatible
   trained SAE and matching identity/dimensions.

## Next qualification work

1. Finish the artifact-contract/model-registry lane already tracked as in progress in `BACKLOG.md`.
2. Run targeted `/harvest`, `/state`/steer, and `/score` qualification on at least one non-Qwen Wave 1
   architecture, then record the exact result and GGUF digest in `wave1.json`.
3. Fit and quant-transfer a second-family J-lens before describing J-lens availability as cross-family.

The former speculative “latest model” roster was removed from this status document. Model popularity,
download counts, and hardware-fit estimates are time-sensitive selection inputs, not runtime evidence.
