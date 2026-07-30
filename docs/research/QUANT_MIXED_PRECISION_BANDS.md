# Mixed-precision counterfactual: does protecting late-MLP layers pay off? — 2026-07-30

**Question:** `docs/research/QUANT_REGRESSION_POPULATION.md`'s n=30 population study found ffn
(MLP) restoration-beats-control share rising monotonically with depth (early 20.6% / mid 33.2% /
late 53.1% of tested SITES) and the recurring restoring layers (24/25/26) were task-independent
across categories and quant levels. The engineering claim that falls out: **quantization damage
concentrates in the last few MLP layers, so mixed-precision quantization should protect late-MLP
precision.** That claim is checkable two ways without new interpretability: (1) simulate the
counterfactual directly, and (2) check whether it agrees with published activation-aware
quantization / outlier-feature work and with the actual quantization scheme that produced our own
test models. This experiment does both.

**Model:** Qwen2.5-7B-Instruct. **Reference:** Q8_0. **Candidates:** Q2_K, Q4_K_M (both — GPU
throughput allowed it). **Instrument:** `clozn.analysis.causal_bisect._run_window`, called directly
against caller-chosen band site-lists rather than through the public `run_bisect()` search
entrypoint (see the script's own docstring for why: `run_bisect()` always tiles from layer 0 and
bisects retained windows in half — there is no way to ask it for three specific, pre-chosen,
never-subdivided bands). Every write, control arm, and gate below is still `_run_window`'s
unmodified code. **Script:** `scripts/tracer/quant_mixed_precision_bands.py`. **Report:**
`runs/experiments/quant_mixed_precision_bands.json` (`clozn.quant_mixed_precision_bands.v1`).
**Mine report reused as-is** (not regenerated): `runs/experiments/quant_regression_mine.json`,
identical to the one the population study used (298 prompts, 7 categories) — so this experiment's
30-disagreement stratified sample is drawn from the exact same pool, and reproducing
`select_stratified_sample` against it yields the population study's own 30-disagreement sample
byte-for-byte (16 Q2_K / 14 Q4_K_M, 5 arithmetic / 5 factual_recall / 4 each other category —
confirmed by re-running the selection before any engine call).

## What this design can and cannot show

This does **not** build a mixed-precision GGUF (a separate llama.cpp/gguf toolchain problem, out of
scope here). It simulates "what if the candidate's late-MLP layers were reference precision" by
**band-level broad transplant**: writing the reference's captured `ffn_out` state at every site in a
whole depth band into the candidate, jointly, in one forward, and checking whether that flips the
candidate's top-1 back to the reference's token. Transplanting the reference's *exact* activation is
strictly better information than any quantized-but-higher-bit-width approximation of it — **this is
an upper bound on what real mixed-precision quantization could recover, not a measurement of it.**
Nothing here demonstrates that re-quantizing those layers at higher bit-width would recover this
much; only that the information those layers are *currently missing* is this often sufficient, on
its own, to fix the candidate's answer.

## Method

**Bands.** Early/mid/late thirds use the same depth-bucket convention as the population study's own
table and `quant_regression_bisect.py`'s `_depth_bucket`: `frac = layer / (n_layer - 1)`, `<1/3`
early, `<2/3` mid, else late. Confirmed live (`pair_compatibility.assess_gguf_pair` + an
`ffn_capture` probe) that Qwen2.5-7B-Instruct has `n_layer=28`, `n_head=28`, identical across all
three GGUF files. At the write positions this experiment actually used (near the end of a
multi-token teacher-forced context, not position 0), **layer 27 never produced a capturable
`ffn_out` row** — confirmed in all 30 records, both candidates, `usable_layers` is 27, not 28, every
time. So in practice: early = layers 0–8 (9), mid = 9–17 (9), late = 18–26 (9, natively — 27 would
be a 10th member but was never usable). `matched_n = min(9,9,9) = 9`; no band needed trimming.

**The five arms** (per the experiment brief):
1. **LATE band** reference transplant (9 sites, layers 18–26)
2. **EARLY band** reference transplant, matched count (9 sites, layers 0–8)
3. **MID band** reference transplant, matched count (9 sites, layers 9–17)
4. **RANDOM equal-norm control at the LATE band** (bundled automatically inside every `_run_window`
   call — EARLY and MID get their own random controls too, for free, reported below as bonus
   context)
5. **No-write replay** — a bare repeat of the baseline call, no capture, no write. 30/30 stable
   (identical top-1 token id, logprob delta < 1e-6).

A fourth band, **late_natural** (the full, untrimmed late third — would have been 10 sites, 18–27),
was planned as a robustness check on the matched-count restriction. It was **skipped for all 30/30
disagreements**, not silently dropped: because layer 27 was never usable (above), the natural late
band and the matched late band are identical in both count and membership for every single
disagreement — there was never a genuinely different superset to test.

**Seed / random-control independence.** `seed = 9001 + sample_index` (matches
`quant_regression_bisect.py`'s own fix for the retracted seed confound — one base seed per
disagreement, not shared). One `random.Random(seed)` is threaded, by reference, across a
disagreement's four band calls (early → mid → late_matched → late_natural), never reconstructed per
band — mirrors `run_bisect()`'s own discipline of threading one continuously-advancing `rng` across
a hook search's tiles. Each band's random-control draw therefore comes from a different point in
that one advancing stream, never a frozen, reused direction.

**Sequential VRAM discipline, batched across disagreements.** Unlike `quant_regression_bisect.py`
(which boots a fresh process per disagreement via `run_bisect()`'s loader contract), this script
calls `_run_window` directly and is free of that contract: the reference is booted once and walks
all 30 disagreements' capture forwards before teardown; each candidate is booted once and walks
every disagreement assigned to it. The two 7B-class models were never resident together.

**Every bound:** `--sample-size 30`, `--seed 9001` (base), `--topk 5`,
`--primary-metric reference_token_logprob_recovery`, `store_tensors=False`, write target = first
disagreement position only per prompt (matches the population study's own convention). Recorded
verbatim in the JSON report's `bounds` field.

## Result — corrected denominator, read this before the headline number

**Audit finding, disclosed before any result is reported:** 5 of the 30 sampled disagreements
(`Q2_K__56`, `Q4_K_M__57`, `Q4_K_M__173`, `Q4_K_M__215`, `Q4_K_M__255`) turned out to be
**already-correct at fresh-baseline time** — the candidate's own top-1, re-checked live in this
run's own process, already matched the reference token, in *every* band, for that disagreement (not
band-specific: baseline is captured once per disagreement and reused across all 4 band tests). A
pilot run reproduced why directly: for `Q2_K__56` ("The capital of France is" → target `' Which'`
vs the mined `' Paris'`), the mine report's own recorded `candidate_logprob_of_reference_token`
(−2.006 nats) was already close to whatever logprob `' Paris'` had gotten — a near-tied top-1/top-2
pair. Re-querying the same model/prompt live, in a fresh engine process, consistently returned
`' Which'` as top-1 (verified 4 times, 2 call shapes × repeat, all agreeing with each other, all
disagreeing with the original mine). This is consistent with **cross-process GPU floating-point
non-associativity flipping the winner of a near-tied logit pair** — a real, quantifiable property of
this inference stack (5/30 ≈ 17% of a stratified real-disagreement sample here), not a bug in this
script. Mining and re-verification necessarily happen in different process launches (sequential VRAM
discipline forbids keeping every model resident at once across a multi-day experiment), so this
reproducibility gap is a standing, disclosed limitation of any experiment built on top of a cached
mine report, not just this one.

The first cut of this script's `_band_stats()` counted these 5 in the SAME denominator as genuine
restoration failures — silently treating "nothing was live to restore" as if it were "the transplant
tried and failed." Fixed before reporting (`scripts/tracer/quant_mixed_precision_bands.py`,
`_band_stats`): they are now excluded from `restoration_rate`'s denominator and tracked separately
(`n_already_correct_no_disagreement`). This is a **uniform** correction — the same 5 disagreements
are excluded from every band identically, so it rescales every band's rate by the same factor
(30/25 = 1.2×) and changes **no** relative comparison between bands. Both figures are in the JSON
(`characterization_audit_note`); the corrected ones (n=25 live disagreements) are reported below.

| band | n (live) | reference moved | random also moved | beat control | rate | Wilson 95% CI |
|---|---|---|---|---|---|---|
| early | 25 | 5 | 3 | 5 | 20.0% | [8.9%, 39.1%] |
| mid | 25 | 6 | 7 | 4 | 16.0% | [6.4%, 34.7%] |
| **late_matched** | 25 | 15 | **0** | 15 | **60.0%** | [40.7%, 76.6%] |
| late_natural | — | — | — | — | n/a | skipped 30/30 — see Method |

**The marginal rates alone do not decisively separate late from early**: 60.0% [40.7, 76.6] and
20.0% [8.9, 39.1] are non-overlapping, but late vs. mid [6.4, 34.7] and (using the uncorrected n=30
figures a careless read would compute) late vs. early [7.3, 33.6] come closer to overlapping. At
n=25 this is not the strongest evidence available here — the **paired** comparison is:

| | late beat_control=True | late beat_control=False |
|---|---|---|
| **early beat_control=True** | 5 | 0 |
| **early beat_control=False** | 10 | 10 |

| | late beat_control=True | late beat_control=False |
|---|---|---|
| **mid beat_control=True** | 3 | 1 |
| **mid beat_control=False** | 12 | 9 |

**10 disagreements where late restored and early did not, 0 the other way. 12 where late restored
and mid did not, 1 the other way.** This paired asymmetry is the evidence that actually carries the
claim here, not the marginal rate gap — it directly answers "does late beat early/mid," disagreement
by disagreement, on matched conditions (same prompt, same candidate, same write position, same seed
stream), which is a stronger design than comparing two independent proportions. It is also immune to
the denominator question above: an already-correct disagreement can only land in the "both no"
cell, never in the asymmetric cells the argument rests on.

**Both candidates independently show the identical late rate**: Q2_K 9/15 = 60.0%, Q4_K_M 6/10 =
60.0% (live disagreements only). Early/mid differ more by candidate (Q2_K early/mid 13.3%/13.3%;
Q4_K_M early/mid 30.0%/20.0%) but late does not — a second, independent echo of the population
study's own "task-independent, so read as a property of the quantizer, not a circuit" argument, here
across **candidates** rather than across **categories**.

By category (live only, late_matched beat_control): structured_json 4/4, arithmetic 4/5,
multilingual 2/3, factual_recall 2/3, code_completion 1/4, instruction_following 1/3,
multi_step_reasoning 1/3. Small per-category n; not independently powered, shown for completeness
only.

## Audit: does this beat the prior in a way that should trigger suspicion?

Per this project's own standing rule (an unexpectedly strong number is the best signal the
instrument is broken, not a reason to celebrate) — checked explicitly, before writing this section:
the population study's late-band rate was **53.1% of tested SITES** (a bisection search's per-site
share, 279/525, pooled over many single- and multi-site leaves). This experiment's late-band rate is
**60.0% of DISAGREEMENTS** (15/25, a single whole-band joint transplant per disagreement) — a
related but genuinely different unit of measurement (site-level search share vs. disagreement-level
broad-transplant success), so the two numbers are not directly comparable as the same statistic. They
are, however, in the same neighborhood, and this experiment's number is not the kind of prior-shattering
jump (the retracted seed-confound run was 12/30 = 40% localized_site against a ~25% prior — a hard
break) that triggered the earlier retraction. **Instrument sanity was 30/30 (every tested band, every
disagreement) and no-write replay was 30/30 stable** — both checked directly against the raw records,
not assumed. No further audit action was warranted.

**Is "random moved 0/25 at late, but 3/25 at early and 7/25 at mid" itself suspicious?** Considered
and judged plausible, not a red flag: restoring the candidate's answer requires landing in the
*specific* direction that encodes the missing reference information; a same-magnitude but
*randomly-oriented* joint perturbation across 9 sites at once is a large, high-dimensional
intervention that should, if anything, be MORE likely to disrupt an already-fragile prediction than
to accidentally restore the one correct one — and disruption (flipping the WRONG way, or flipping to
neither token) does not register as `random_moved`, only a flip specifically TO the target token
does. A clean 0/25 says the late band's specific reference direction is doing real work no
equal-magnitude noise reproduces; it would become suspicious only alongside evidence the write
mechanism itself is compromised, which instrument_sane=30/30 rules out. Note the mid band shows the
opposite texture — 7 random successes against only 4 beat-control (3 of mid's 6 reference-moved cases
were ALSO matched by random, i.e. `perturbation_sensitive`, not reference-specific) — mid reads as
the "noisiest" band: neither its own reference transplant nor a random one is a clean signal there,
while late is clean in both directions (every reference success at late is uncontaminated by a random
success).

## External check: does this agree with published quantization-sensitivity work?

This is the whole reason this experiment outranks the other pending research items — an internal
result that cannot be checked against outside work is worth much less than one that can.

**Direct, verified check against this experiment's own candidate models.** Verified live via
`WebFetch` of `github.com/ggml-org/llama.cpp`, `src/llama-quant.cpp` (fetched twice independently;
consistent both times) — the `use_more_bits(i_layer, n_layers)` heuristic
(`i < n/8 || i >= 7*n/8 || (i - n/8) % 3 == 2`) and the `FFN_DOWN` tensor-category branch. `ffn_down`
is the down-projection weight matrix whose output **is** the vector this experiment transplants
(clozn's `ffn` / `ffn_out` hook). For `LLAMA_FTYPE_MOSTLY_Q4_K_M` on a non-Falcon architecture
(Qwen2.5 included): `ffn_down` is upgraded from `Q4_K` to `Q6_K` exactly where `use_more_bits` is
true — for this model's `n_layer=28`, that is layers **{0,1,2, 5,8,11,14,17,20,23, 24,25,26,27}**
(both extremes plus periodic middle layers — not a late-only scheme). For
`LLAMA_FTYPE_MOSTLY_Q2_K`: `ffn_down` is **uniformly `Q3_K` at every layer, no depth variation at
all.**

This corroborates the direction of the claim: Q4_K_M already gives 4 of this experiment's own 9 late
layers (24, 25, 26, 27) extra bits and shows a much lower mined disagreement rate (42.6% of prompts,
`quant_regression_mine.json`) than Q2_K, which upgrades nothing anywhere and disagrees on 84.9% of
prompts. It does **not**, however, fully resolve in Q4_K_M's favor within the late band itself:
Q4_K_M's own late_matched restoration rate (60%, 6/10 live) is statistically indistinguishable from
Q2_K's (60%, 9/15 live), despite Q4_K_M having already protected nearly half its late layers and Q2_K
none. If llama.cpp's own upgrade fully fixed what this experiment measures, Q4_K_M's late band should
plausibly have LESS restorable damage left than Q2_K's — it does not, in this sample. Candidate,
un-adjudicated explanations: the un-upgraded `ffn_gate`/`ffn_up` tensors at those same late layers
still carry damage `Q6_K ffn_down` alone does not fix; `Q6_K` itself remains lossy enough to still
matter; or the depth-vs-leverage confound (below) means late transplants restore this well regardless
of how much the layer's own weights were already protected.

**Broader literature — orthogonal or partially disagreeing, not confirming.** None of the papers
checked directly measure "band-level transplant restorability," so none can straightforwardly
confirm this experiment; each bears on an adjacent question:
- **LLM.int8()** (Dettmers et al. 2022, arXiv:2208.07339): emergent outlier *features* (specific
  hidden dimensions) "occur in all layers" once they emerge at scale — a channel/dimension axis, not
  a depth-concentration claim. Orthogonal to this experiment's question.
- **AWQ** (Lin et al. 2023, arXiv:2306.00978): salience is identified per input *channel* via
  activation magnitude, independent of layer depth. Also orthogonal.
- **Massive Activations** (Sun et al. 2024, arXiv:2402.17762): a related but distinct outlier
  phenomenon that "emerges suddenly after a single layer of computation and diminishes at the last
  few layers" — if anything the opposite depth profile from this experiment's late-concentrated
  restorability, though it measures activation *magnitude*, not transplant-restorability, so it is
  not measuring the same thing.
- **Quantization Error Propagation** (arXiv:2504.09629): reports the FIRST and LAST transformer
  blocks both show the highest quantization sensitivity (a U-shape), with error compounding through
  depth. **Partial agreement** (late elevated, matches here) and **partial disagreement** (this
  experiment's early band was the LOWEST-restoring band, 20.0%, not elevated — no U-shape observed
  here).
- **SpQR** (Dettmers et al. 2023, arXiv:2306.03078): mixed precision via per-weight Hessian-based
  sensitivity; the material located does not frame its findings in depth-band terms, so it neither
  confirms nor contradicts this experiment.

**Could not verify, and not asserted:** a web search surfaced a claim ("early layers process
entangled, fine-grained feature manifolds and are highly sensitive to quantization noise, while
later layers manipulate more disentangled semantic features and tolerate lower precision")
attributed by the search engine's own summary to a specific LLM-quantization paper. Directly
fetching that paper's full text did **not** contain the passage, and the paper (confirmed via
WebFetch to evaluate Llama-2/3.1 and Qwen2.5) is LLM-focused — while the "entangled feature
manifold" framing is characteristic of CNN/vision-model quantization literature. This reads as a
search-summary conflation, not a real citation, and is disclosed here rather than asserted either
way, per this experiment's own instruction to say so when an external claim cannot be verified.

## Limits — these bound the claim

- **This is an upper bound, not a measurement of real mixed-precision quantization.** See "What this
  design can and cannot show" above — repeated here because it is the single most important caveat.
- **The depth-vs-leverage confound stands, unresolved.** Late layers both quantize worst (this
  project's own prior) AND have the most raw logit leverage (fewer remaining layers to wash out an
  intervention) — a joint 9-site late-band write is an even bigger, more direct lever than a single
  late site was in the population study. The same-depth random-equal-norm control fixes vector NORM
  at each site, which separates "any large perturbation would do this" from "the reference direction
  specifically does this" — but it does **not** separate "damage concentrates late" from "late writes
  move logits more per unit of correctly-aimed intervention," because a correctly-aimed intervention
  late in the network mechanically has more leverage regardless of whether quantization damage is
  actually worse there. See falsification note (3) below for a design that could separate them.
- **n=25 live disagreements** (30 sampled, 5 excluded as already-correct-at-recheck — see Result).
  Category-stratified from the SAME mine-report pool the population study used, not independently
  drawn; per-category counts are too small (3-5 each) to power a category-level claim.
- **Single model family.** Qwen2.5-7B-Instruct only.
- **Positions bisected are each prompt's FIRST disagreement only**, matching the population study's
  own convention.
- **No correction for multiple comparisons** across the 4 bands × 30 disagreements tested.

## What would falsify this

1. The late band's restoration rate failing to exceed early/mid's on a larger or differently-
   stratified sample — this run's paired asymmetry (10-0, 12-1) is the load-bearing evidence
   precisely because the marginal 95% CIs are not yet cleanly separated at n=25.
2. The late-band advantage collapsing under a design that controls the depth-vs-leverage confound
   this one cannot fully separate — e.g. amplifying an early-band transplant to the SAME empirical
   logit-movement magnitude late transplants achieve, and checking whether it then restores
   comparably.
3. Failure to replicate on a different model family or architecture.
4. The llama.cpp `use_more_bits()` correspondence failing under closer inspection — e.g. ablating
   just the layers Q4_K_M already upgrades to Q6_K and finding no disagreement-rate benefit
   attributable to that upgrade specifically.
5. A depth-stratified (not divergence-ranked) re-run of the population study's own attention-head
   search finding early/mid-band attention behaves like late-band ffn does here, which would
   undermine an ffn-specific late-concentration story.

## Reproduction

```
python scripts/tracer/quant_mixed_precision_bands.py --select-only   # verify the sample, no engine calls
python scripts/tracer/quant_mixed_precision_bands.py                  # full run, ~30-40 min on this box
```
Requires `runs/experiments/quant_regression_mine.json` (gitignored; regenerate with
`scripts/tracer/quant_regression_mine.py` if absent — see that script's own docstring for cost).
