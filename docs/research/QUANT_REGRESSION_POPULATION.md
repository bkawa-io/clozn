# Quantization-regression population study — 2026-07-28

**Question:** across a population of real quantization regressions, what is the distribution of
causal verdicts? Not "can we find a localized one" — the distribution itself is the result.

**Reference:** Qwen2.5-7B-Instruct-Q8_0. **Candidates:** Q2_K, Q4_K_M.
**Instruments:** `clozn/analysis/{pair_compatibility,transplant,causal_bisect,restoration_metrics}.py`.
**Scripts:** `scripts/tracer/quant_regression_mine.py`, `scripts/tracer/quant_regression_bisect.py`.
**Artifacts (local, gitignored):** `runs/experiments/quant_regression_{mine,population_report}.json`.

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

## Phase 2 — causal bisect (26 category-stratified disagreements, both hook paths)

```
ffn  (MLP)        localized_site 9 | localized_window 3 | distributed 5 |
                  perturbation_sensitive 1 | no_restoration 8
head (attention)  localized_site 4 |                      distributed 2 |
                  no_restoration 20
```

**17/26 on the MLP path; 6/26 on attention.** Every causal verdict cleared an independent random
equal-norm control. Instrument sanity 388/388 (rate 1.0) — the candidate self-transplant was a
verified no-op in every observation, so the write path never confounded a result.

Depth gradient (share of tested sites beating control), monotonic:

| hook | band | tested | moved | beat control | |
|---|---|---|---|---|---|
| ffn | early | 273 | 57 | 41 | 15.0% |
| ffn | mid | 319 | 134 | 87 | 27.3% |
| ffn | late | 434 | 247 | 217 | 50.0% |
| head | late | 460 | 60 | 49 | 10.7% |

By category (hooks pooled, causal share): structured_json 6/8, code_completion 5/8, arithmetic 3/8,
factual_recall 3/8, multilingual 3/6, multi_step_reasoning 2/8, instruction_following 2/6.
**The phase-1 inverse relationship holds: structured_json has the lowest disagreement rate and the
highest localization rate.** Rare failures are localizable; common ones are distributed.

## The seed confound (found, retracted, corrected)

The first Phase-2 run was **retracted**. `transplant.run_site()` builds its RNG as
`random.Random(seed)` fresh per call and `causal_bisect.run_bisect()` threads ONE seed to every
leaf; the driver passed `seed=1` uniformly. `random.Random(1)` reproduces an identical float
sequence on every construction, and every ffn site shares one vector width — so **all 87 control
arms in that run used the same frozen direction, merely rescaled**, verified in the artifact
(1 distinct `random_seed`, against 12 in the corrected run).

What exposed it: the run returned 12 `localized_site`/30 on ffn, far above this project's prior of
~3/12 reference-specific ([DISTRIBUTED_FUNCTION.md](DISTRIBUTED_FUNCTION.md)). **A result that
overturns a hard-won prior is the moment to audit the controls, not to report the news.**

Measured impact, same cases re-run: 2 of 32 (id, hook) pairs changed verdict — one
`distributed → perturbation_sensitive`, one `localized_window → distributed`. Restricted to the 13
pairs where controls actually ran, 2/13. The seed-1 draw happened to behave typically. **That is
luck, not vindication:** had it been weak, identical code would have produced a fabricated result
with no outward sign, and the impact was unknowable without re-running.

## Limits — these bound the claim

* **The second-order confound is NOT closed.** 14 of 26 records had multi-leaf searches sharing one
  seed draw across leaves within a single disagreement — more than half the sample. `run_bisect()`
  threads one seed to every leaf; closing it means changing that seeding contract in
  `clozn/analysis/`, which this experiment was scoped not to touch.
* **The head search reached LATE layers only** (`max_head_sites=16`, ranked by observational
  divergence, which concentrated there). "Attention carries less signal" is qualified by coverage,
  not established across the stack.
* **Not population rates.** The sample is category-stratified, not random over the 568 mined
  disagreements. n=26.
* Verdicts are per (disagreement, hook); no correction for multiple comparisons was applied.
