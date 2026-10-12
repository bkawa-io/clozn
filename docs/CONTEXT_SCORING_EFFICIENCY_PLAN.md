# Context-scoring efficiency: implementation handoff

Status: native screening milestone implemented and qualified on the recorded CPU and Metal fixtures;
the CPU qualification was reproduced on 2026-10-09. The initial release keeps production rollout
explicitly default-off.

## Implementation status (2026-10-05)

Implemented:

- closed exact-token `POST /v1/score-variants` contract and both Python clients;
- deterministic token-trie prefix reuse, duplicate collapse, KV-aware resident waves, and
  multi-sequence continuation decoding;
- backend softmax + target gather, with measured fallback metadata when a backend cannot suppress
  full-vocabulary host transfer;
- explicit native-screen/scalar-reference separation plus opt-in proof-critical and borderline
  scalar refinement;
- optional bounded reuse of a worker-local original-generation checkpoint for late-diverging
  variants, with exact worker/baseline identity, capacity accounting, and ordinary-prefix fallback;
- an opt-in, budgeted coarse-to-fine scheduler with native coarse screening, scalar refinement of
  every selected fine/joint condition, explicit drilldown and joint arms, durable reuse, and separate
  trusted/screening/unavailable/skipped states;
- conservative workload admission for the product opt-in: ordinary batches use native scoring only
  at four or more variants with at least 25% estimated exact trie-row reuse, otherwise returning an
  honestly labeled scalar-reference result; explicit screen-then-refine orchestration may require the
  native screen without weakening any compatibility checks;
- real-GGUF parity and performance harnesses under `scripts/smoke/score_variants_parity.py` and
  `scripts/bench/score_variants.py`.

The native regime remains screening-only and default-off. The fixed CPU qualification fixture has a
0.25-nat per-token gate; its 91 gated comparisons passed with a 0.2452125 maximum after adding the
late-divergence generation-state case. An explicitly reported adversarial shallow-prefix case reached
0.6645 nats even after generation-state reuse correctly fell back to request-local scoring. That case
is excluded from the predeclared fixture rather than used to widen its threshold, and is evidence that
unselected native rows must remain non-authoritative.

The 2026-10-09 Qwen CPU rerun reproduced that result: all 91 gated comparisons passed, with
maximum/mean/p95 absolute token-logprob drift of 0.2452125/0.0180630/0.1117188 nats. The battery
covered prefix-of-another prompts, exact duplicates, 17 variants across two KV-aware waves, a one-token
continuation, applied generation-state reuse, and the declared shallow-prefix fallback. Every case used
backend softmax/target gather and none fell back to full-vocabulary host reduction.

The same fixed suite was run on an Apple M5 Metal build with all 24 Qwen2.5-0.5B Q4_K_M layers
offloaded (GGUF SHA-256 `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653`).
All 91 gated comparisons passed with maximum/mean/p95 absolute token-logprob drift of
0.0293232/0.0055313/0.0206967 nats. Every case used `backend_softmax_target_gather` with
`full_vocab_host_reduction=false`; the gated cases copied 88 target floats to the host for 91 logical
comparisons (one duplicate was collapsed). In the warm benchmark, eight late-diverging variants were
1.525x faster than independent scalar requests and eight duplicates were 2.548x faster; one- and
two-variant requests remained slower because native setup dominates. These results qualify only this
model, quantization, backend, and execution fixture; they are not a general Metal or GPU guarantee.

A second Apple M5 Metal qualification used Llama-3.2-1B-Instruct Q4_K_M with all 16 layers
offloaded (GGUF SHA-256 `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`).
All 91 gated comparisons passed at the unchanged 0.25-nat threshold, with
maximum/mean/p95 drift of 0.0450152/0.00634058/0.0255266 nats. Compact backend reduction remained
active in every case. Warm median speedups were 1.135x for four late-diverging variants, 1.275x for
eight late-diverging variants, and 4.323x for eight duplicates; one/two variants were slower and
eight early-diverging or no-overlap variants were approximately break-even. Those cross-model
results motivate the conservative product admission thresholds rather than a universal native path.

Follow-up work (not required for the initial release):

- qualification of additional model, quantization, cache, and accelerator configurations;
- a future review if automatic rollout is ever worth reconsidering. The accelerated scorer and
  coarse-to-fine scheduler remain deliberate opt-ins; adaptive workload admission operates inside
  that opt-in.

### Product rollout controls

- `CLOZN_ENABLE_NATIVE_SCORE_VARIANTS=1` enables the adaptive product seam; it remains unset by
  default.
- `CLOZN_NATIVE_SCORE_VARIANTS_MIN_ARMS` defaults to `4` (bounded to 1–64).
- `CLOZN_NATIVE_SCORE_VARIANTS_MIN_PREFIX_REUSE_RATIO` defaults to `0.25` (bounded to 0–1).

Below either threshold, the many-arm seam executes the existing scalar path and reports
`scalar_reference`, its actual serial/concurrent strategy, the scalar execution identity, and the
rollout reason. It is never mislabeled as a native screen. Explicit screen-then-refine orchestration
can require the native screen even when automatic rollout is disabled and below these performance
thresholds, because screening is the requested operation; worker capability, token-exactness, and
safety checks still apply, and every selected condition is scalar-refined before it becomes evidence.

## Objective and design decision

Reduce local inference time and memory use for context investigations without weakening the meaning of their evidence. Prioritize avoiding repeated computation, then improve batching. Do not make additional Python threads or independent model contexts the primary optimization.

Build one generic native operation that scores a recorded continuation against several tokenized prompts. Share exact prompt prefixes, process the remaining work in memory-bounded batches, and return only the requested token log probabilities plus execution metadata. Context-source selection and intervention semantics stay above the engine.

Ship incrementally: establish the reference and measurements; implement compact scoring and prefix reuse as independently measurable improvements; add multi-sequence batching; integrate and qualify; only then add adaptive investigation scheduling.

## Architecture now in the tree

- `clozn/server/substrates.py` exposes one metadata-bearing many-arm seam. It reports the actual scalar
  or native strategy and never treats endpoint availability as proof-grade qualification.
- `engine/core/src/model_ggml.cpp` executes request-local variant scoring with token-trie prefix reuse,
  duplicate collapse, KV-aware waves, multi-sequence continuation decoding, and optional compatible
  generation-state reuse.
- The native path reduces each requested row to its target-token log probability near the model and
  reports whether full-vocabulary host reduction was avoided.
- `engine/core/include/clozn/variant_score_plan.hpp` owns deterministic exact-token planning and capacity
  accounting; model-free tests cover overlap, duplicates, ragged inputs, and wave boundaries.
- Experiment orchestration reuses durable observations, validates the baseline/control, keeps screening
  separate from evidence, and scalar-refines every selected proof-bearing result.
- Context-source deletion and neutral-filler replacement remain distinct interventions above the engine.

## Shared requirements for every ticket

1. **Token-exact inputs.** Compute prefix overlap after the real chat template and tokenizer. Use recorded continuation token IDs. Character offsets, independently tokenized fragments, and equal-length text replacements do not establish shared KV history.
2. **Correct causal alignment.** The last prompt row predicts answer token 0; each subsequent answer token is scored from its predecessor's row. Preserve positions, masks, output ordering, and ragged prompt lengths. Never share answer KV between different prompt histories merely because their answer tokens match.
3. **Strict reuse identity.** Share only identical token histories under identical model, adapter/steering, positional, attention, and cache conditions. Never rejoin branches after divergence. Identical complete requests may be deduplicated and mapped back to their original IDs.
4. **Explicit numerical status.** Keep the existing scalar scorer as the reference. Prefix reuse, changed chunking, GPU reduction, and batching can all change floating-point results. Until qualified, accelerated results are screening results—not trusted observations. “Reference” does not mean mathematically exact inference.
5. **No false negative claims.** Unmeasured, cancelled, skipped, or screening-only work cannot become “context unused” or an authoritative below-floor result. Preserve coverage information. Validate both positive and negative claims through an accepted evidence path.
6. **Bounded resources and clean failure.** Respect per-sequence length, shared KV capacity, sequence limits, and compute/output workspace. Check cancellation between bounded chunks/waves. Clean up on success, error, and cancellation. Never silently retry an uncertain partially executed batch.
7. **Narrow initial support.** Start with plain text, recorded-token scoring and `topk=0`. Unsupported model/cache types or intervention hooks must explicitly fall back or report unsupported, never be silently ignored. Keep ordinary attention optimizations enabled; this task does not require attention capture.
8. **Scope discipline.** Preserve unrelated working-tree changes. Do not alter context-effect definitions, measurement floors, sampler behavior, existing CLI parallelism restrictions, or the llama.cpp pin as an incidental optimization.

## Proposed operation and result contract

Names are provisional; freeze them in Ticket 1 before dependent implementation.

```text
score_variants(
  variants = [{id, prompt_token_ids}, ...],
  continuation_token_ids,
  execution_policy,
  resource_limits
)
  -> ordered per-variant status and target-token logprobs
   + numerical regime, execution identity, and qualification status
   + timing, reuse, transfer, and capacity metrics
```

All variants in one group share the continuation and immutable execution conditions. The engine does not interpret RAG, source IDs, deletion, or filler. A baseline is just another prompt when one is needed by the caller.

The result must preserve stable variant IDs, token alignment, actual strategy/fallback, and incomplete-work status. Execution identity must cover result-affecting settings, including reduction implementation and batching/chunking policy. Actual batch/wave membership and order must be recorded where they affect reproducibility; do not reuse observations across unqualified execution shapes.

Use one metadata-bearing many-arm API. Scalar reference execution and native screening both return the same result envelope, including strategy, numerical regime, evidence use, qualification, identity, and metrics. Missing metadata is not proof of qualification.

The existing validated control remains first in the product flow. Version 1 can share prefixes among the missing variants inside one native request; it does not require retaining baseline KV across HTTP requests. Reuse the durable baseline observation where compatible. A cached scalar baseline and accelerated arm do not automatically form a trusted delta; mixed-regime screening must be labeled, and any required regime-specific baseline is additional measurement, not a replacement for the validated control.

## Bounded implementation tickets

Assign one ticket at a time to a coding agent, together with the shared requirements above. Each ticket should be a separately reviewable change. The dependencies below describe possible work ordering, not a request to launch agents automatically.

### Ticket 1 — Freeze the scoring and evidence contract

**Status:** complete.

**Outcome:** agree on inputs, result envelopes, capabilities, numerical labels, and observation identity before adding fast execution.

**Touchpoints:** `engine/client/clozn_engine/__init__.py`, `clozn/server/substrates.py`, `clozn/experiments/{scoring,multi_arm,observations,persistence}.py`, relevant protocol/schema definitions.

**Work:** define reference versus accelerated-screening policies; supported conditions; fallback and cancellation semantics; metadata propagation; cache separation. Specify how existing observations remain readable without retroactively calling them accelerated-qualified. Correct strategy terminology so scalar HTTP concurrency is not reported as native batching.

**Acceptance:** contract tests cover ordering, missing metadata, unsupported inputs, partial failure, and distinct identities for incompatible regimes. Scalar execution remains available through the shared result contract. No accelerated result can enter the trusted observation cache through a boolean capability flag alone.

### Ticket 2 — Establish measurements and the reference benchmark

**Status:** complete for the recorded CPU and Metal fixtures; new hardware identities remain ongoing
qualification work.

**Depends on:** Ticket 1 for metric names; reference measurements can begin immediately.

**Outcome:** reproducible evidence showing where time and memory go, with a scalar oracle for later changes.

**Touchpoints:** `/score`, `model_ggml.cpp`, substrate/runner timing, a new scoring-specific script under `scripts/bench/`, and performance instrumentation tests.

**Work:** measure queue/lease wait, template/tokenization, model execution, logits transfer/reduction, serialization, and persistence separately. Report decoded tokens, unique/reused prefix rows, decode calls, logical versus physical KV use, transfer volume, and peak memory where measurable. Distinguish measured values from estimates and unavailable metrics. Use local monotonic clocks; account for asynchronous device work without double-counting overlapping phases.

**Acceptance:** one command runs fixed prompt/answer fixtures and emits machine-readable configuration, timings, and scalar token scores. Separate cold and warm runs; use repeated measurements with enough samples for reported tail latency. The benchmark exercises the C++ scoring path directly.

### Ticket 3 — Compact target-logprob reduction

**Status:** complete on the qualified paths, with explicit fallback metadata elsewhere.

**Depends on:** Tickets 1–2. Can be developed independently of Ticket 4.

**Outcome:** avoid moving a full vocabulary row to the CPU for every scored token on a supported backend.

**Touchpoints:** `model_ggml.cpp`, `/score` reduction, backend/build integration, and tracked llama.cpp patches only if necessary.

**Work:** implement stable `target_logit - logsumexp(all_vocab_logits)` near the logits tensor, returning one value per requested token. The full vocabulary normalization remains necessary. Suppress llama.cpp's raw-logit host copy on this path; reducing after the copy does not achieve the goal. Correctly handle tensor layout, output-row mapping, lifetime, and synchronization. Keep the host/reference path available.

**Acceptance:** compact-only benchmarks demonstrate actual transfer reduction and report numerical differences and latency. The unchanged reference path still passes. CPU builds remain functional; CUDA support is not advertised as Metal support. Start with the backend selected by the benchmark owner; unsupported backends explicitly fall back. If there is no meaningful gain on target workloads, do not enable it by default.

**Dependency caution:** read `engine/core/third_party/PATCHES.md`. The local llama.cpp tree is gitignored; accessor/skip-copy patches are not a ready-made AR scoring implementation. Any necessary patch must be tracked and reproducible against the existing pin. Do not upgrade upstream as part of this ticket.

### Ticket 4 — Model-free exact-prefix and capacity planner

**Status:** complete.

**Depends on:** Ticket 1. Can run independently of Ticket 3 after measurement fixtures are agreed.

**Outcome:** a deterministic execution plan for related prompts, without requiring a model or GPU to test it.

**Touchpoints:** a small planner under `engine/core/include/clozn/`, the existing radix planner as reference, and C++ planner tests.

**Work:** build a token trie/radix plan across all prompts, deduplicate identical requests, preserve terminal prompts that are prefixes of other prompts, and map execution order back to caller order. Plan bounded resident waves using backend capabilities. Account for shared physical prefix rows, each divergent suffix, continuation growth, and workspace/headroom; do not merely sum all logical prompt lengths. Also check each sequence's own position limit.

**Acceptance:** model-free tests cover no overlap, complete overlap, duplicate prompts, early/middle/late divergence, prefix-of-another prompts, unequal lengths, invalid inputs, and capacity boundaries. No suffix remerging. Planning is deterministic, never knowingly exceeds its budget, and can reduce to one variant when needed. Actual allocation failure remains a handled runtime case.

### Ticket 5 — Native prefix-reusing scoring executor

**Status:** complete as an opt-in screening path.

**Depends on:** Tickets 1, 2, and 4. Ticket 3 is optional initially.

**Outcome:** a native multi-prompt scorer that removes repeated prefill work before introducing multi-sequence answer batching.

**Touchpoints:** `model_ggml.cpp` and its interface, sequence-copy/removal helpers, a narrowly scoped native route and client method.

**Work:** execute the prefix plan within an explicitly owned context lease. Reuse valid shared KV and recompute only divergent suffixes. First implement a conservative one-branch-at-a-time traversal so prefix reuse can be measured separately. Score known continuation tokens in causal chunks, not a generation loop. Preserve prompt-final logits for the first answer token. Avoid helpers that clear KV or reset causal state mid-session.

Check sequence-removal/copy capabilities and return values. Recurrent/hybrid state may not support partial rollback; unsupported cases must use a declared safe strategy or the scalar fallback. Keep cache state request-scoped for this version.

**Acceptance:** prefix-only benchmarks reduce decoded prompt work when overlap exists; no-overlap cases remain correct. Compare every output row to the scalar reference, with differences reported rather than hidden. Tests cover cancellation, decode failure, cleanup, cross-request isolation, and subsequent normal generation. This path remains opt-in screening until qualification.

### Ticket 6 — KV-aware multi-sequence, multi-token batching

**Status:** complete as an opt-in screening path.

**Depends on:** Ticket 5; use Ticket 3 where supported.

**Outcome:** improve device utilization by packing known work across variants and answer positions.

**Work:** execute the planner's resident waves with independent sequence IDs, positions, and causal masks. Pack multi-token suffix/continuation chunks across variants; do not simply call the one-token-per-sequence generation helper once per answer token. Correctly remap output rows to variant and answer index. Deduplicate only when complete histories and conditions match. Retain the prefix-only path for comparison and fallback.

Bound resident KV and compute/output buffers separately. Record actual wave shapes, re-prefill caused by eviction, and fallback reasons. Check cancellation at bounded intervals. Do not introduce permanent extra model contexts or cross-request KV persistence in this ticket.

**Acceptance:** compare scalar, compact-only, prefix-only, batched, and combined modes. Exercise ragged prompts, one-token answers, long answers, microbatch boundaries, reduced capacity, and different variant orders/wave sizes. Demonstrate bounded memory, isolation, and responsive cancellation. Report interactive-request latency as well as aggregate scoring throughput; a long diagnostic must not hide unacceptable contention behind a faster batch time.

### Ticket 7 — Product integration, evidence qualification, and rollout

**Status:** integration and guarded rollout are complete for the initial release. Broader
qualification and any future default-on decision are intentionally deferred.

**Depends on:** Tickets 1–6 for the combined path; individual optimizations may be qualified separately.

**Outcome:** use acceleration through existing investigation seams without changing what a receipt claims.

**Touchpoints:** `clozn/server/substrates.py`, `clozn/experiments/{multi_arm,scoring,runner,observations,persistence}.py`, context recipe/reader tests, engine client/capability tests.

**Work:** dispatch only uncached, compatible variants to the native scorer after control validation. Preserve ordering, deduplication, cancellation, partial completion, and durable evidence reuse. Carry execution metadata end to end; separate screening storage from trusted reusable observations. Keep answer-selection projections read-only. Do not conflate neutral-filler intervention with source deletion.

Qualify each supported model/backend/quantization/cache/execution configuration against the reference using the matrix below. Predeclare numerical and decision tolerances; do not widen them merely to make tests pass. Determinism within one batch shape is insufficient. A proof-grade flag must reflect verified capabilities, not the existence of a batch endpoint.

**Acceptance:** regression tests preserve current evidence behavior; cache tests prevent cross-regime collisions; unsupported configurations fall back transparently. Authoritative results continue to use scalar remeasurement unless the accelerated configuration is explicitly qualified for that use. Keep acceleration behind a default-off flag until its qualification and performance reports are reviewed. Include a simple rollback switch.

### Ticket 8 — Optional coarse-to-fine investigation scheduling

**Depends on:** Ticket 7. Separate product change; not required for the initial engine speedup.

**Status:** implemented as the opt-in `measure_context_effects_coarse_to_fine` recipe; see
`docs/CONTEXT_EFFECTS_COARSE_TO_FINE.md`.

**Outcome:** spend deeper analysis on promising or user-selected regions without pretending the rest were exhaustively measured.

**Work:** use screening to prioritize candidates, then obtain accepted evidence for claims. Support explicit budgets, progress/coverage, user-requested drilldown, and selective finer spans or joint interventions. Reuse existing observations. Keep this scheduler above the inference layer.

**Acceptance:** skipped and screening-only sources remain distinguishable from measured low-effect sources. Tests include redundant sources, opposing effects within a coarse span, and joint effects that individual deletions miss. A weak coarse result must not prohibit requested refinement or become an exhaustive irrelevance claim. Report end-to-end work saved, including verification costs.

### Ticket 9 — Automatic generation-state handoff

**Status:** deferred.

**Outcome:** allow ordinary generation to offer its compatible final prompt state to a subsequent
diagnostic without coupling evidence correctness to that state remaining resident.

**Work:** define ownership, expiry, memory pressure, worker restart, cancellation, and identity rules;
make diagnostic acquisition opportunistic; preserve request-local prefix scoring as the complete fallback.

**Acceptance:** an unavailable or evicted state changes only performance, never results or evidence
status. Tests cover sampled generation, worker restart, concurrent diagnostics, eviction, cancellation,
and a scalar comparison for every accepted state identity.

## Qualification and performance matrix

Use the same recorded token fixtures across modes. Include:

- Short and long prompts/answers; 1, 2, 4, and 8 variants where capacity allows.
- Early, middle, and late changes; substantial sharing, no sharing, duplicates, and unequal prompt lengths.
- KV and microbatch boundary cases; different wave shapes and input orders; supported model/cache/backend combinations.
- Cached versus uncached observations; cold versus warm runtime; failure/cancellation and competing interactive requests.
- Clear effects, near-floor effects, near-zero effects, opposing effects, and redundant/joint-source cases.

For each configuration report wall time and its breakdown, decoded work, transfer volume, total memory pressure, and fallback frequency. Compare per-token logprob error, baseline-minus-variant delta error, sign changes, ranking changes, threshold flips, and repeatability. Include aggregate/selected-answer statistics that the product actually uses.

One recorded CUDA fixture showed batched delta drift around 0.19 nats, while the reader's default measurement floor is 0.1 nats. This is a reason to qualify carefully—not a universal error bound. Fast weak results cannot safely rule out a source just because they appear below that floor.

Success means measured improvement on agreed representative local workloads, within accepted memory and interaction-latency budgets, while all claimed evidence retains its required fidelity. Do not promise a universal speedup or call a model-free test suite hardware qualification.

## Deferred work and explicit non-goals

- **Automatic original-generation state reuse:** the explicit worker-local
  `checkpoint_on_finish`/`generation_state` experiment is implemented and remains screening-only.
  Automatically retaining state from ordinary product generation, with an appropriate lifecycle and
  memory policy, is still deferred. Existing sampled confidences can reflect temperature, penalties,
  or constraints and are not automatically `/score` results.
- **Cross-request persistent prefix caches:** later; they add eviction, invalidation, ownership, and model-lifecycle complexity. Request-local sharing is the first milestone.
- **More independent contexts:** optional benchmark experiment, not the core architecture. Account for replicated KV/buffers and interactive contention before reconsidering current product restrictions.
- **Speculative shortcuts:** no attention-based substitute for causal scoring, no answer-KV sharing across different histories, no approximate result silently promoted to evidence, and no unrelated generation/reference-match rewrite.

## Coding-agent handoff template

> Implement Ticket N only. Read this document's shared requirements and the current code in the listed touchpoints. Preserve unrelated changes. Confirm the ticket's dependencies and contract before editing. Keep the existing scalar reference available; add focused tests and relevant benchmark evidence. Do not implement later tickets, change evidence semantics, broaden backend support claims, or enable the fast path by default. If the contract must change, report the proposed change before expanding scope.

Each agent returns: changed files; behavior added; tests run and results; measured versus unmeasured performance; numerical/fidelity status; limitations; and the next ticket's integration notes. If target hardware is unavailable, say so explicitly and leave its qualification incomplete.
