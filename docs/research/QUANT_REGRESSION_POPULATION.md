# Quantization-regression population study — 2026-07-28

**Question:** across a population of real quantization regressions, what is the distribution of
causal verdicts? Not "can we find a localized one" — the distribution itself is the result.

**Reference:** Qwen2.5-7B-Instruct-Q8_0. **Candidates:** Q2_K, Q4_K_M.
**Instruments:** `clozn/analysis/{pair_compatibility,transplant,causal_bisect,restoration_metrics}.py`.
**Scripts:** `scripts/tracer/quant_regression_mine.py`, `scripts/tracer/quant_regression_bisect.py`.
**Artifacts (local, gitignored):** `runs/experiments/quant_regression_{mine,population_report}.json`.

**Note on this revision:** this branch was worked by two separate agent sessions concurrently, in the
same shared main working tree (no isolation), each unaware of the other. Both used the identical
Phase-1 corpus/mine output (298 prompts; deterministic teacher-forced greedy decode against fixed
weights, so identical inputs reproduce identical Phase-1 numbers regardless of which process ran it)
and the identical, since-fixed, per-disagreement seed scheme in `quant_regression_bisect.py`. One
session ran Phase 2 at the script's default `--sample-size 26`; this revision ran it at `--sample-size
30`, category-stratified with the SAME deterministic round-robin, so the 30-sample run's first 26
entries are literally the earlier session's 26 (confirmed: their verdict counts are a strict subset of
this run's). This is reproducibility evidence for the pipeline (same seeds → same verdicts across two
independent process invocations), not independent replication on a different sample — the two runs
share 26 of their 30 data points. Numbers below are the full n=30 run; the earlier n=26 figures are
superseded but were consistent with it wherever they overlap. Flagged here rather than silently
reconciled, since concurrent, un-coordinated writes to the same branch/working tree is itself worth
a maintainer's attention.

## Why this study exists

An earlier pass tried 15 prompts, found 4 Q2_K disagreements, bisected exactly ONE, got
`no_restoration`, and concluded quantization damage looked non-localizable. That was an
under-powered sample, not a finding. It also searched the MLP path only — the `ffn_out` hook and
composable attention sites did not exist yet.

## Phase 1 — mining (298 prompts, 7 categories, 2384 scored positions)

| category | Q2_K prompt | Q2_K pos | Q4_K_M prompt | Q4_K_M pos |
|---|---|---|---|---|
| arithmetic | 98.2% | 29.9% | 57.1% | 11.4% |
| multilingual | 93.3% | 29.4% | 55.6% | 11.1% |
| factual_recall | 91.5% | 26.1% | 40.4% | 6.4% |
| multi_step_reasoning | 90.0% | 26.2% | 45.0% | 7.8% |
| instruction_following | 82.5% | 23.1% | 47.5% | 8.1% |
| code_completion | 80.0% | 17.8% | 25.0% | 3.4% |
| **structured_json** | **40.0%** | **6.2%** | **13.3%** | **2.5%** |
| all | 84.9% | 23.8% | 42.6% | 7.7% |

**Constrained, low-entropy output survives aggressive quantization far better than free-form
generation.** An earlier 15-prompt pass had reported "Q4_K_M disagrees on 0/15" — pure small-sample
noise against the real 42.6%.

## Phase 2 — causal bisect (30 category-stratified disagreements, both hook paths)

30 disagreements (16 Q2_K, 14 Q4_K_M), stratified round-robin over the 7 categories (5 arithmetic, 5
factual_recall, 4 each of the other five) out of the pools mined in Phase 1 (sizes: arithmetic 87,
factual_recall 62, multilingual 67, instruction_following 52, multi_step_reasoning 54,
code_completion 42, structured_json 16). Each disagreement's FIRST disagreement position was bisected
over BOTH hooks independently: `ffn` (full writable range, window_size=max(4,n_layer//4)=7,
max_windows=4) and `head` (full `head_layers × head_indices` grid captured, narrowed to the top 16
sites by observational divergence, window_size=4, max_windows=4). `store_tensors=False`; `seed` varied
per disagreement (see below). 60 total bisect runs, 0 errors.

```
ffn  (MLP)        localized_site 12 | localized_window 3 | distributed 5 |
                  perturbation_sensitive 1 | no_restoration 9
head (attention)  localized_site 4  |                      distributed 2 |
                  perturbation_sensitive 1 | no_restoration 23
```

**20/30 (66.7%) on the MLP path beat the random-equal-norm control; 6/30 (20.0%) on attention.**
Every causal verdict cleared an independent random-equal-norm control — that is what "beat control"
means here, structurally, not just a label. Instrument sanity 482/482 (rate 1.0) — the candidate
self-transplant was a verified no-op in every observation, so the write path never confounded a
result.

Depth gradient (share of tested sites beating control), monotonic on ffn:

| hook | band | tested | moved | beat control | % |
|---|---|---|---|---|---|
| ffn | early | 335 | 86 | 69 | 20.6% |
| ffn | mid | 395 | 181 | 131 | 33.2% |
| ffn | late | 525 | 314 | 279 | 53.1% |
| head | late | 524 | 64 | 49 | 9.4% |

(head has NO early/mid rows: every one of its 524 tested sites, across all 30 disagreements, fell in
the late band — see Limits below, this is a coverage artifact of `max_head_sites`'s divergence
ranking, not a finding that early/mid attention heads are irrelevant.)

By category (hooks pooled, causal share): structured_json 5/8 (62.5%), code_completion 5/8 (62.5%),
multilingual 4/8 (50.0%), instruction_following 3/8 (37.5%), arithmetic 4/10 (40.0%), factual_recall
3/10 (30.0%), multi_step_reasoning 2/8 (25.0%). **The phase-1 inverse relationship holds:
structured_json has the lowest disagreement rate (Phase 1) and among the highest localization rates
(Phase 2).** Rare failures are more often localizable; common ones are more often distributed or
unrestored.

## The seed confound (found, retracted, corrected)

**Implementation follow-up (2026-07-29): the remaining within-disagreement confound described below is
now closed in `clozn.analysis.causal_bisect`.** `run_bisect(seed=N)` retains `N` as its reproducible base
seed, while every single-site confirmation derives an independent uint64 seed from SHA-256 over canonical
JSON keyed by `{base_seed, source, hook, layer, head?}`. The derivation strategy is recorded in every new
`clozn.causal-bisect.v1` artifact and the embedded `clozn.transplant.v1.random_seed` remains the actual
seed used at that leaf. The population figures in this document are historical measurements made before
that module-level correction; they are not silently recomputed here.

The first Phase-2 run (n=30, archived locally as
`_quant_regression_population_report_SEED_CONFOUND_v1.json`, gitignored) was **retracted**.
`transplant.run_site()` builds its RNG as `random.Random(seed)` fresh per call, and
`causal_bisect.run_bisect()` threads ONE `seed` to every leaf's `run_site()` call unconditionally;
the driver passed `seed=1` uniformly across all 30 disagreements. `random.Random(1)` reproduces an
identical float sequence on every construction, and every ffn site shares one vector width (n_embd) —
so **every single-site leaf confirmation in that run, across every disagreement, shared the same
frozen raw random direction, merely rescaled to each site's own reference-vector norm.** This is
`clozn/analysis/transplant.py` and `causal_bisect.py` working exactly as documented (a caller-supplied
seed IS supposed to be reproducible) — the gap was in this experiment's OWN calling convention, not
in those modules, and was fixed here (`seed = seed_base + sample_index`, varied per disagreement)
rather than by touching `clozn/analysis/`.

What exposed it: the retracted run returned 12 `localized_site`/30 on ffn, far above this project's
prior of ~3/12 reference-specific ([DISTRIBUTED_FUNCTION.md](DISTRIBUTED_FUNCTION.md)). **A result
that overturns a hard-won prior is the moment to audit the controls, not to report the news.**

Measured impact, same 30 disagreements re-run with the fix: of 60 comparable (id, hook) pairs, **3
changed verdict label** — `Q2_K__144/ffn: distributed_restoration → perturbation_sensitive`,
`Q2_K__213/ffn: localized_window → distributed_restoration` (both moves make the finding LESS
localized, i.e. the frozen weak control had been making these two look more precise than the
independent draw supports), and `Q4_K_M__57/head: distributed_restoration → localized_site` (this one
moves the OTHER way — MORE localized under the fix). The verdict histogram's shape survives
essentially unchanged (ffn localized_site 12→12, head localized_site 3→4). **This is a small,
non-systematically-biased impact, not a null one — and it was luck, not vindication:** had the
frozen seed-1 draw happened to be an unusually weak perturbation direction in this ~3584-dim residual
width, identical code would have produced a fabricated result with no outward sign, and the impact
would have been unknowable without re-running with independent seeds.

## Limits — these bound the claim

* **Historical second-order confound, now closed in code.** 17 of the 60 (disagreement, hook) documents
  (13 of the 30 disagreements) had multi-leaf searches sharing one seed draw across leaves WITHIN that
  single disagreement (rescaled per leaf's own reference norm, not independently redrawn) — over 40%
  of the sample. Those stored results retain that limitation. New `run_bisect()` artifacts use the
  site-derived seed contract described above; this report's historical measurements would need a new
  live run before their numeric results could be relabeled as free of the old confound.
* **The head search reached the LATE layer band almost exclusively** (`max_head_sites=16`, ranked by
  observational reference-vs-candidate divergence — which, empirically, concentrated there: all 524
  tested head sites across all 30 disagreements fell in the late third of the network). "Attention
  carries less signal" is qualified by this coverage gap, not established across the stack — a
  divergence-agnostic or depth-stratified head sampling strategy might find early/mid-layer attention
  effects this design structurally could not see.
* **Not population rates.** The sample is category-stratified, not randomly drawn from the 568 mined
  disagreements (253 Q2_K + 127 Q4_K_M prompts with ≥1 disagreement, more positions than that). n=30.
* **Depth leverage is a live alternative explanation for the depth gradient itself.** A late-layer ffn
  transplant is mechanically a bigger, more direct lever on the final logits (less remaining
  computation to alter or wash out the intervention) than an early-layer one, independent of where
  quantization damage actually originates. This design cannot cleanly separate "damage lives late" from
  "late interventions have more raw power to flip a decision" — both predict the same monotonic
  gradient measured above.
* Verdicts are per (disagreement, hook); no correction for multiple comparisons was applied.
* Positions bisected are each prompt's FIRST disagreement only (one flip per prompt, matching the
  original transplant-localization method) — later disagreements in the same continuation, when a
  prompt had more than one, were never bisected.

## Site-overlap analysis — is this a circuit? No.

Post-hoc analysis over the same n=30 artifact (no new run). For each disagreement, the ffn layers
whose single-site confirmation came back `reference_specific=True`:

```
L 0 (early)  2      L14 (mid )  3      L22 (late)  2      L25 (late)  6
L 7 (early)  1      L15 (mid )  1      L23 (late)  2      L26 (late)  4
L 8 (early)  1      L19 (late)  2      L24 (late)  5
L11 (mid )  1      L20 (late)  2
L13 (mid )  1      L21 (late)  1
```
12/30 records had >=1 ffn site beat control; 15 distinct layers ever did; 34 (record, layer) hits.

**The sites recur strongly.** The top three layers (24, 25, 26) hold **15/34 = 44%** of all hits,
where a uniform spread over 28 layers would give ~11%. Restoration is not scattered.

**But the recurrence is TASK-INDEPENDENT, which is what rules out a circuit reading:**

| layer | hits | categories | candidates |
|---|---|---|---|
| L24 | 5 | arithmetic, factual_recall, multilingual | Q2_K, Q4_K_M |
| L25 | 6 | arithmetic, structured_json, multilingual, multi_step_reasoning | Q2_K, Q4_K_M |
| L26 | 4 | arithmetic, multilingual | Q2_K, Q4_K_M |

A circuit would be task-SPECIFIC — arithmetic regressions restoring at one locus, multilingual at
another. Instead the same late layers restore everything regardless of task, and across both quant
levels. That is the signature of a **geometric property of the quantizer** (rounding damage
accumulates toward the output end) rather than of a reusable computational component.

**This reconciles with [DISTRIBUTED_FUNCTION.md](DISTRIBUTED_FUNCTION.md) rather than contradicting
it.** Those studies asked where a *capability* lives in one model (answer: nowhere localizable).
This asks where the *difference* between two models concentrates. A distributed function can have
concentrated damage — load rides every cable of a bridge, but a frayed cable localizes the failure,
not the load-bearing. Both hold simultaneously.

**Caveat that cannot be removed from this data:** the last layers are also where a write has the most
raw leverage on logits, so "late layers quantize worst" and "late writes move output most" predict
the same pattern. The same-depth random control shows it is not pure leverage (the reference
direction beats equal-norm noise 53% of the time in the late band), but the two are not cleanly
separated here. At n=30 (4-5 per category) task-independence is a strong hint, not a proof.

**What would flip this back toward circuits:** task-SPECIFIC recurrence — a layer restoring
arithmetic but not multilingual. Not observed here, but per-category n is too small to exclude. That
needs more bisects per category, i.e. a new GPU run.

**Engineering read, which needs no circuit claim:** quantization damage concentrates in the last ~3
MLP layers independent of task. That is a directly testable mixed-precision hypothesis (protect
late-MLP precision) and is externally checkable against activation-aware quantization / outlier-
feature literature.
