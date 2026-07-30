# Task-specific recurrence: does any layer restore one capability but not another? — 2026-07-30

**Question:** [`QUANT_REGRESSION_POPULATION.md`](QUANT_REGRESSION_POPULATION.md) found ffn restoring
sites recur strongly (layers 24/25/26 held 44% of all hits, ~11% uniform) but the recurrence looked
**task-independent** — layer 25 alone restored arithmetic, structured_json, multilingual AND
multi_step_reasoning disagreements alike. That doc's own words: *"per-category n is too small to
exclude [task-specific structure]. That needs more bisects per category, i.e. a new GPU run."* This
is that run.

**Headline, power before p-value, because that ordering is the entire point of this document:** the
naive read of this run's own data — a bare p-value from the most inclusive site threshold — would be
**wrong to trust**, because that threshold turns out to have no demonstrated power at this sample
size (an injected, perfectly category-exclusive synthetic effect at the maximum testable size still
failed to reach significance there). Once that is diagnosed and corrected by choosing an
analytically-justified, better-powered threshold, this run **is** informative: it rules out any
single site where ≥36% of one category's disagreements concentrate exclusively, and finds none. That
is a real, bounded, useful negative — not the stronger "task-independence confirmed" claim, and not
"we learned nothing" either.

**Reference:** Qwen2.5-7B-Instruct-Q8_0. **Candidates:** Q2_K (24 disagreements), Q4_K_M (18).
**Categories:** `arithmetic`, `multilingual`, `structured_json` — 14 disagreements each, 42 total (the
suggested "maximally different" triad: two common high-disagreement-rate categories of different
kinds, plus `structured_json`, the outlier on every axis measured so far — lowest Phase-1 disagreement
rate, highest Phase-2 localization rate). **Instrument:** `clozn.analysis.causal_bisect.run_bisect()`,
unmodified, via `scripts/tracer/quant_regression_bisect.py`'s own `run_one_disagreement` (imported by
path, not reimplemented). **Script:** `scripts/tracer/quant_regression_task_specificity.py`. **Report:**
`runs/experiments/quant_task_specificity.json` (`clozn.quant_task_specificity.v1`, gitignored, ~5.5MB).

## Method

Both fixes the earlier seed-confound retraction produced are inherited automatically, never
re-specified: `head_site_selection` defaults to `"stratified_divergence"` in
`causal_bisect.run_bisect()` itself (the depth-banded head-site cap that closed the late-band-only
head coverage gap), and every single-site confirmation derives its own independent seed via
`sha256_canonical_json_uint64_be_v1`. This script's own `seed = 5001 + sample_index` (a base seed,
deliberately distinct from `quant_regression_bisect.py`'s default of 1 and
`quant_mixed_precision_bands.py`'s default of 9001) only varies the base, matching the fix for the
retracted confound.

**Sample:** `select_category_balanced_sample` — a deterministic round-robin across exactly the three
target categories, capped at 14 each (never padded; `structured_json`'s own mined pool is only 16, so
this uses 14 of 16 — 87.5% of everything available for that category). Pool sizes: arithmetic 87,
multilingual 67, structured_json 16.

**Bisect, per disagreement:** the SAME two-hook design as the population study — `ffn` (full writable
range, window_size=max(4,n_layer//4)=7, max_windows=4) and `head` (full grid, narrowed to 16 sites by
`stratified_divergence`, window_size=4, max_windows=4) — each an independent `run_bisect()` call, first
disagreement position only, `store_tensors=False`. 84 total bisect calls (42 disagreements × 2 hooks),
0 errors. **Instrument sanity 674/674 = 1.0** across every window and single-site observation — every
verdict this document reports cleared that gate.

**Checkpointing:** `runs/experiments/_quant_task_specificity_checkpoint.json` accumulated one completed
disagreement (both hooks' full documents) at a time, atomic write-then-replace, matching
`quant_regression_bisect.py`'s own discipline. A `--analyze-only` flag was added specifically so the
(expensive, GPU-bound) 42-disagreement bisect never had to be re-run while the statistical analysis
below was being debugged and corrected.

## Verdict histogram and depth gradient (context, not the main question)

```
localized_site           19    no_restoration          44
localized_window          9    perturbation_sensitive   4
distributed_restoration   8
```

Depth gradient (share of tested sites beating control) replicates the population study closely: ffn
early 104/478 = 21.8% (pop. study: 20.6%), mid 177/541 = 32.7% (33.2%), late 404/747 = 54.1% (53.1%).
**Bonus finding, not this experiment's own question but worth recording:** the `stratified_divergence`
fix visibly works here — head sites now actually cover early (119 tested) and mid (220 tested) bands,
not just late (389), closing the coverage artifact the population study's Limits section flagged
(`"every one of its 524 tested sites... fell in the late band"`). Head's beat-control rate is still
much lower than ffn's at every depth (early 4/119=3.4%, mid 19/220=8.6%, late 36/389=9.3%), but it is
no longer a coverage illusion.

Per-category localization share (either hook localizing, matching the population study's own
"hooks pooled, causal share" convention): **arithmetic 10/14 (71.4%), structured_json 10/14 (71.4%),
multilingual 9/14 (64.3%)**. Compare the population study's n=8-10-per-category figures: structured_json
was the clear high outlier there (62.5%) and arithmetic the low one (40.0%) — that ranking **does not
replicate** at this larger, independent per-category n. This is itself informative: it is a live
demonstration of exactly why that document called n=4-5/category "too small," on the very same
"which category localizes more often" axis, not just the task-specificity axis this run was built to
test.

## The site × category matrix

20 distinct `(hook, layer)` sites carried a `reference_specific=True` single-site hit; 56 hits total
across 42 records (17 distinct disagreements contributed at least one hit — most disagreements'
searches produced `no_restoration` or only window-level, non-attributable evidence).

| site | arithmetic | multilingual | structured_json |
|---|---|---|---|
| ffn:L24 | 3 | 1 | 1 |
| ffn:L25 | 3 | 2 | 2 |
| ffn:L26 | 5 | 3 | 0 |
| head:L9 | 0 | 0 | 1 |
| head:L26 | 0 | 0 | 2 |
| (14 more ffn sites, layers 0–23, each 1–2 hits) | | | |

Layers 24/25/26 hold 20/56 = 35.7% of all hits (uniform over 20 sites would give 15%) — the same
qualitative recurrence pattern the population study found, at a larger and category-focused sample.
`ffn:L26` is the single most visually suggestive cell (arithmetic 5, multilingual 3, **structured_json
0**) — the closest thing to a task-specific signal anywhere in this dataset. It does not clear
significance (below). `head:L9` and `head:L26` are structured_json-only, but at 1 and 2 hits
respectively they are far too thin to interpret alone — exactly the kind of cell a global,
multiple-comparisons-aware test exists to keep anyone from over-reading in isolation.

## Statistics: two permutation tests, a real bug found and fixed, and why a threshold choice was load-bearing

**Correction stated up front, per the experiment's own requirement:** this analysis uses a **permutation
null**, never a per-cell p-value — every reported significance test shuffles category labels across
disagreements and asks how often a statistic that extreme arises by relabeling alone. A naive
per-site-tested Bonferroni bound is also reported for comparison, never as the decision rule.

**Two global statistics**, chosen so no single-cell comparison is ever the basis for a claim:
1. **`chi2_sum`** — sum of per-site Pearson chi-square deviations from the observed overall category
   share, over every site with ≥2 hits (17 qualifying sites here). Detects diffuse, multi-site
   non-uniformity.
2. **`max_purity`** — the single largest (most-common-category count / total hits) over every
   qualifying site. Taking the **max** across every site searched is what makes this
   multiple-comparisons-correct: the permutation null itself takes the same max over the same
   qualifying sites every time, so a coincidentally-pure cell must beat what coincidence ALREADY
   typically produces, not a fixed nominal threshold.

**Real bug found and fixed before any result was trusted.** The first version of both tests built the
permutation label pool from `_id_category_map(hits)` — i.e., only the 17 disagreements that produced
≥1 hit, whose OWN category composition (arithmetic 9 / multilingual 5 / structured_json 3) is far
from the true 14/14/14 sample marginal. Shuffling within that skewed subset builds a null that has
quietly baked the very category-level hit-rate imbalance the test exists to interrogate into its own
reference distribution. It was caught by its own consequence, not by inspection: the minimum-detectable-
effect search (below) returned "not detected" even at k=14 — a fully category-exclusive synthetic site
covering every disagreement of one category. That is next to impossible under a correctly-specified
null and was the signal to stop and audit, the same discipline this project's seed-confound retraction
established, applied to a different kind of instrument. Fixed: the permutation pool is now the FULL
42-disagreement sample with its true category labels (`run_permutation_test`'s and `mde_search`'s own
docstrings carry the full account). The chi2_sum headline p-value moved only slightly under the fix
(0.330 → 0.352) — small, in this instance, but that was luck, not vindication, exactly as this
project's earlier seed-confound writeup put it about its own "small, non-systematic" impact.

**A second, independent issue the fix's own audit surfaced (not a bug — a real, quantifiable power
limit):** with only 3 categories at a 14/14/14 marginal, a qualifying site with exactly `min_hits=3`
hits has a **9.5% chance of looking perfectly category-pure by pure combinatorial luck alone**
(`comb(14,3)/comb(42,3) × 3` categories):

| `min_hits` | P(coincidentally pure, one specific category) | P(any of 3 categories) |
|---|---|---|
| 2 | 10.6% | 31.7% |
| 3 | 3.2% | 9.5% |
| 4 | 0.9% | 2.7% |
| 5 | 0.2% | 0.7% |
| 6 | 0.06% | 0.2% |

At `purity_min_hits=3` (this script's first default), 6 real sites qualify — several small enough
that their cumulative chance of a coincidental full-purity match is large. Confirmed directly by
running the minimum-detectable-effect search (inject a synthetic, previously-unused site hit
EXCLUSIVELY by `k` disagreements of one category, real background otherwise untouched, rerun the
max-purity test fresh for each `k`): at `min_hits=3` the p-value **plateaus around 0.15 and never
drops below 0.05, even at k=14 (100% of a category, the maximum possible effect)**. At `min_hits=4` it
plateaus around 0.06 — closer, still never crosses. **This threshold has no demonstrated power at
n=14/category and any "not significant" read from it would be uninformative, not evidence of anything.**

At `min_hits=5` (0.7%/site by-chance rate, 3 real qualifying sites: L24, L25, L26), the search **does**
cross significance:

| k (of 14) | fraction | observed max purity | p-value |
|---|---|---|---|
| 1–4 | ≤29% | 0.625 (unchanged — synthetic site not yet the max) | 0.33–0.34 |
| **5** | **36%** | **1.0** | **0.017** |
| 6 | 43% | 1.0 | 0.009 |
| 7–14 | 50–100% | 1.0 | 0.006–0.010 |

**This is the primary, well-powered result:** at `purity_min_hits=5`, this experiment could have
detected a single previously-unused site hit exclusively by ≥5/14 (≥36%) of one category's
disagreements, with zero contamination from the other two categories. It did not find one.

## Results (primary, at the well-powered threshold)

| statistic | observed | qualifying sites | permutations | p-value |
|---|---|---|---|---|
| `chi2_sum` (min_hits=2) | 30.71 | 17 | 100,000 | **0.352** |
| `max_purity` (min_hits=5) | 0.625 (`ffn:L26`, arithmetic 5 / multilingual 3 / structured_json 0) | 3 | 100,000 | **0.336** |

Naive Bonferroni context (not the decision rule used): 3 sites tested at `min_hits=5` → per-cell
alpha 0.0167 for a comparable claim; the observed p-values above are nowhere near that bar either, so
the conclusion does not depend on which correction philosophy is used here. `chi2_sum` is stable
across its own own threshold choice too (min_hits 2/3/4: p = 0.35 / 0.83 / 0.97 — same direction,
no sign of a hidden effect being thrown out by a stricter cut).

## Interpretation

**Do not read this as "task-independence confirmed."** That conclusion is not available from a
p-value alone, and it is the tempting misread precisely because it agrees with the population study's
prior. What this run actually establishes: at the analytically-justified, empirically-validated
`purity_min_hits=5` threshold — the one place in this whole analysis where the design demonstrably had
teeth — **no site concentrated ≥36% of one category's disagreements to the exclusion of the others.**
`ffn:L26`'s own 5/3/0 split is the closest thing to a counter-example the data contains, and it lands
at p=0.34: plausible under pure chance, not evidence of task-specific circuitry. Combined with the
global `chi2_sum` test (p=0.35, stable across its own threshold sweep) finding no diffuse
category-by-site structure either, the honest summary is: **no task-specific recurrence found, at a
power level that could have detected a moderately strong effect (≥36% single-category concentration)
had one existed.** Weaker, more diffuse task-specific effects (a handful of disagreements per category
softly favoring different layers, never near-exclusively) remain outside what this design could see.

## Correcting the record on the population study

[`QUANT_REGRESSION_POPULATION.md`](QUANT_REGRESSION_POPULATION.md)'s Site-overlap analysis section
reads task-independent recurrence as "the signature of a geometric property of the quantizer... rather
than of a reusable computational component," and flags in its own Limits section that per-category
n=4-5 there was "too small to exclude" task-specific structure. That reading remains **plausible** —
this run does not contradict it — but it was never, and still is not, **statistically established**.
This document is the requested follow-up at 14/category; it found no task-specific recurrence at a
power level bounded above (≥36% concentration, one site, one category, zero cross-contamination). It
does not raise that bound to "any effect, however small" — see Limits below. Readers of the population
study's quantizer-geometry framing should treat it as the leading, unfalsified hypothesis, not a
closed question.

## Limits — these bound the claim

* **The minimum-detectable-effect search characterizes ONE alternative shape: a single previously-
  unused site, hit exclusively by k disagreements of one category.** A more diffuse task-specific
  effect — several disagreements per category each mildly favoring a different existing site, never
  concentrating enough at any one site to be "the max" — is a structurally different alternative that
  `max_purity` is not built to detect and this MDE search did not characterize. `chi2_sum` is the
  better-suited statistic for that shape but was not given its own MDE search here (a disclosed scope
  boundary, not an oversight).
* **`structured_json`'s pool is nearly exhausted** (14 of 16 mined disagreements used) — this category
  cannot be grown much further without re-mining a larger corpus specifically for it.
* **Category-level localization RATE rankings are unstable between this run and the population
  study's** (structured_json highest there, tied-with-arithmetic here) — a second, independent
  demonstration that per-category n in the single digits to low teens is noisy on this axis, separate
  from the site-specificity question this document was built to answer.
* **Single model family, two quant levels.** Qwen2.5-7B-Instruct, Q2_K and Q4_K_M only.
* **No correction for multiple comparisons was applied to the site-level attribution done anywhere
  outside the two formal permutation tests** (e.g. the by-eye read of `ffn:L26`'s 5/3/0 split above is
  presented as illustrative context, not as an independently-tested claim).
* Positions bisected are each prompt's FIRST disagreement only, matching the population study's own
  convention.

## What would falsify this

A properly-powered (min_hits chosen the same principled way — by its own by-chance-purity rate, not
picked for a favorable p-value) `max_purity` or `chi2_sum` result crossing p<0.05 on a comparably-sized
or larger sample, especially if it replicated on an independent category triad or a different model
family.

## What would make this answerable at finer effect sizes

The MDE trace above is the concrete number: **≥5/14 disagreements (≥36%) per category** is what this
design and this run's own background noise could detect at `purity_min_hits=5`, 3 categories, α=0.05.
To resolve a smaller effect (say 20%, roughly 3/14) would need either (a) more disagreements per
category — the MDE trace's p-value gap between k=4 (0.34) and k=5 (0.017) is steep, suggesting the
detection boundary is closer to 30-35% than a smooth power curve might imply, so a materially smaller
detectable floor would likely need per-category n well into the 20s-30s, not merely mid-teens — or
(b) a statistic purpose-built for a diffuse alternative (`chi2_sum`'s own MDE, not run here — see
Limits) rather than the single-site-concentration shape `max_purity` targets.

## Reproduction

```
python scripts/tracer/quant_regression_task_specificity.py --select-only     # verify the sample, no engine calls
python scripts/tracer/quant_regression_task_specificity.py                    # full run, ~35-45 min on this box
python scripts/tracer/quant_regression_task_specificity.py --analyze-only     # rerun only the statistics,
                                                                                # from a completed checkpoint
                                                                                # (this document's own numbers
                                                                                # were finalized this way, after
                                                                                # fixing the permutation-pool bug,
                                                                                # with zero additional GPU calls)
```
Requires `runs/experiments/quant_regression_mine.json` (gitignored; regenerate with
`scripts/tracer/quant_regression_mine.py` if absent).
