# Depth vs. leverage: how much of the late-band advantage is content-specific? — 2026-07-30

**This is an analysis pass over existing artifacts. No new GPU work was run to produce this document.**

**Question, restated precisely:** three research docs share one caveat — a late-layer ffn write is
mechanically a bigger, more direct lever on final logits (fewer remaining layers to wash out the
intervention) *independent of* whether quantization damage actually concentrates there, so "damage
lives late" and "late writes move logits more" predict the same monotonic depth gradient. The
same-depth random-equal-norm control already in each study's design rules out "any large perturbation
works" (leverage of *magnitude* alone), but was explicitly flagged as not fully separating leverage from
content-specificity. This document re-mines the one artifact detailed enough to say more —
`runs/experiments/quant_mixed_precision_bands.json` — using fields the existing docs' published
`characterization` block does not surface, to find out how much further the existing data can actually
separate the two.

**Bottom line up front:** the record-level data shift the weight of evidence toward content-specificity
more than the existing docs' neutral "cannot cleanly separate" framing suggests, on **three independent
lines of evidence** below — but they do **not** close the question. No amplitude-matched controlled
experiment (the design the mixed-precision doc's own falsification note already specifies) has been run.
That remains the only design that would fully separate them. See "What remains unseparated" at the end.

## Artifacts used, and a provenance note that matters

- `runs/experiments/quant_regression_population_report.json` (`clozn.quant_regression_population.v1`,
  generated `2026-07-29T06:15:52Z`) — present in the repo's shared checkout, gitignored as expected.
- `runs/experiments/quant_task_specificity.json` (`clozn.quant_task_specificity.v1`) — present in the
  repo's shared checkout, gitignored as expected.
- `runs/experiments/quant_mixed_precision_bands.json` (`clozn.quant_mixed_precision_bands.v1`, generated
  `2026-07-30T10:12:47Z`) — **not present** in the shared checkout or in this analysis's own worktree
  (gitignored, so worktrees do not share it). It was located on disk in a **sibling agent worktree**
  (`agent-ab5a65e26a59ce9a0`), fully written (valid, complete JSON; the doc `docs/research/
  QUANT_MIXED_PRECISION_BANDS.md` at the current commit already cites the identical numbers verbatim, so
  this is the finished artifact behind the already-merged doc, not a stale or in-progress file). Copied
  read-only into this worktree's own `runs/experiments/` (gitignored, not committed) so the analysis
  script below has a stable local path. Flagged here explicitly per this project's standing rule about
  concurrent-session artifact discovery — this file was read, not generated; nothing in it was altered.

**A methodological note that bears on how much weight to put on population-report numbers below:** the
population report's `layer_depth_breakdown` (early 20.6% / mid 33.2% / late 53.1%) counts **bisection-tree
layer-occurrences**, not independent single trials — `quant_regression_bisect.py`'s `characterize()`
credits every layer inside a tested window with that window's outcome, and the search only re-tests
(subdivides) windows that already beat control, so layers inside repeatedly-retained windows accumulate
more tested/beat_control credit than layers whose enclosing window failed once and was never subdivided
further. This is a real property of an adaptive search (it is not wrong, and it is not new — it is simply
a different unit of analysis than "probability a single site beats control"), but it means the population
report's percentages are not a clean base rate to lean on for the movement-magnitude argument below. The
mixed-precision-bands report is not built this way: each of the three bands is tested **exactly once per
disagreement**, as a fixed, pre-chosen 9-site window decided before any outcome is known — no recursion,
no endogenous re-testing of winners. That is why this analysis is anchored in that artifact.

## Verified from source, not inferred from field names

**Is "restored" measured the same way for reference and random arms?** Yes — confirmed by reading
`clozn/analysis/causal_bisect.py` directly. `_run_window` computes `reference_moved =
_flipped_to_target(baseline_metrics, arms["reference_transplant"])` and `random_moved =
_flipped_to_target(baseline_metrics, arms["random_equal_norm"])` — the **same function**,
`_flipped_to_target(baseline, arm) = (not baseline_hit) and arm_hit`, applied to both arms' identically-
structured `top1_is_target` field. `beat_control = bool(reference_moved and not random_moved)` is a joint
rule over two identically-scored flags, not two differently-defined criteria compared after the fact.

**Are the site counts genuinely matched?** Yes, verified at record level (not from the bounds/method
prose): every one of the 30 sampled disagreements has exactly 9 sites in `early`, `mid`, and
`late_matched` — no record needed trimming (`early`/`mid` were already exactly 9; `late`'s natural 10th
site, layer 27, was never capturable at these write positions — disclosed in the report's own
`late_natural_skip_explanation` and independently confirmed here from `usable_layers` on every record).

**Is the "5 already-correct disagreements" exclusion uniform across bands, as the report's audit note
claims?** Yes, verified directly: the identical five IDs (`Q2_K__56`, `Q4_K_M__57`, `Q4_K_M__173`,
`Q4_K_M__215`, `Q4_K_M__255`) are excluded in `early`, `mid`, and `late_matched` alike — confirmed by
recomputing the exclusion from each band's own `reasons` text, not by trusting the note.

All of the above three checks matter because a "yes, but measured differently" or "no, not actually
matched" answer to any of them would have invalidated the comparison before it started. None did.

## Line 1: the random-arm control's own effect size does not scale with depth

The published `characterization` block only carries the *binary* flip outcome per arm. Every
`band_results[band]["movement_metrics"]` already carries a **continuous, pre-computed** score for both
arms — `reference_token_logprob_recovery`'s `movement` (signed logprob shift of the target token, and
`gap_closed_fraction` = movement normalized by that disagreement's own baseline-to-reference gap, which is
identical across bands for a given disagreement and therefore a fair cross-band comparison, unlike raw
movement). This analysis only reads that field; it computes no new score.

`gap_closed_fraction`, pooled over both candidates (n=25 disagreements/band):

| band | reference_transplant mean (median) | random_equal_norm mean (median) |
|---|---|---|
| early | 0.179 (0.173) | −18.33 (−5.87) |
| mid | 0.131 (0.162) | −5.27 (−4.09) |
| late_matched | **0.709 (0.777)** | −17.62 (−8.10) |

The reference (correctly-aimed) arm's normalized recovery is roughly flat between early and mid
(0.18 → 0.13), then jumps sharply at late (→ 0.71–0.78) — a step, not a smooth gradient. **This holds
independently in both candidates**, checked because this project's own history is that a clean pooled
result can hide a confound that stratification reveals: Q2_K goes 0.104 → 0.008 → **0.649**; Q4_K_M goes
0.292 → 0.315 → **0.799**. Both candidates independently show the jump concentrated specifically at late,
not a smooth early<mid<late staircase.

If that jump were mechanical leverage (closer to the unembedding, less remaining computation to wash out
*any* write), the content-blind random-equal-norm arm — norm-matched to the same reference vector, wrong
direction — should show the same depth pattern, since leverage as usually stated does not care about
direction. It does not: pooled random-arm magnitude is smallest at **mid** (6.35 gap-units, abs mean),
not smallest at early as a monotonic "more leverage deeper" story predicts. Checked per candidate: **both
candidates independently put mid as the trough** (Q2_K: early 6.5, mid **3.0**, late 8.3; Q4_K_M: early
36.4, mid **11.3**, late 31.6) — mid being the gentlest band for random noise is robust across
candidates. The early-vs-late *ordering* is **not** robust across candidates (Q2_K: late > early;
Q4_K_M: early > late) — disclosed rather than cherry-picked, and this analysis does not lean on that
specific ordering. What is robust is the absence of any monotonic early<mid<late increase for the
content-blind arm, in either candidate.

**Reading:** the random arm is an empirical proxy for what pure, content-blind positional leverage would
predict (same magnitude, wrong direction). Because that proxy does not track depth the way the reference
arm's recovery does, the late-specific jump in the reference arm is not well explained by a smoothly-
increasing positional mechanism alone — the content-blind channel's own effect size does not scale that
way in this data. This is an indirect, proxy-based argument, not a direct manipulation of depth holding
content fixed; see "What remains unseparated."

## Line 2: what a "diverted" random write at late actually looks like

Breaking the binary flip apart into `restored` (flipped to target) / `no_op` (write changed nothing at
top-1) / `diverted` (write changed top-1 to some third token, neither baseline's own answer nor the
target) — a partition the published `characterization` block does not carry, computed here from each
arm's own `top1_token_id`:

| band | arm | restored | no_op | diverted | n |
|---|---|---|---|---|---|
| early | random_equal_norm | 3 | 1 | 21 | 25 |
| mid | random_equal_norm | 7 | 2 | 16 | 25 |
| late_matched | random_equal_norm | **0** | 2 | **23** | 25 |

Concrete late examples (baseline top-1 → target vs. what the random arm actually produced):
`Q2_K__0` (arithmetic): baseline `' to'`, target `' $\'`, random top-1 `'1'`. `Q2_K__174`
(multi_step_reasoning): baseline `' calculated'`, target `' reduced'`, random top-1 `'*pow'`.
`Q2_K__253` (multilingual): baseline `' ?\n\n'`, target `' ?'`, random top-1 `'\n'`. These are not
near-misses — the random write at late genuinely produces unrelated tokens, consistent with "diverted."
But per Line 1, its *magnitude* is not the largest of the three bands, and per the table above, early's
diverted share (84%) is close to late's (92%) despite early's random magnitude being larger — so
"destroys the output more thoroughly at late" is not clearly true in raw destructiveness terms; what
changes at late is that essentially none of that destruction ever lands specifically on the target token
(0/25 vs 3/25 at early, despite early's noise being at least as large).

Mid tells an unrelated, independently useful story: of mid's 7 random restorations, **5** occurred on
disagreements where the **reference transplant itself failed to restore** (e.g. `Q2_K__144`: candidate
top-1 stayed `'name'` under the true reference-direction write, but flipped to the correct `'fruit'`
under random noise; same pattern in `Q2_K__213`, `Q2_K__176`, `Q2_K__254`, `Q4_K_M__112`). That is a
concrete illustration of the doc's own "mid is the noisiest band" reading (neither arm is clean there) —
at mid, top-1 for these disagreements looks fragile enough that even the *correct* information sometimes
fails to move it while unrelated noise sometimes does, consistent with near-tied logits rather than a
depth-gradient signal either way.

## Line 3: a destination-depth-controlled comparison, latent in the shuffled_window arm

`shuffled_window` was read carefully rather than assumed from its name. `causal_bisect._pick_shuffled_sites`
does **not** shuffle within a band — it returns the numerically-lowest disjoint usable sites, deterministically.
Recomputed from every record's own `usable_layers`/`bands` (not guessed): **`mid`'s and `late_matched`'s
`shuffled_window` arms both write their own band's real captured reference vectors into the `early`
band's site indices (layers 0–8)** — confirmed 30/30 records for each. `early`'s own `shuffled_window`
writes into `mid`'s indices (9–17) instead, and is not used in the comparison below.

This is an accidental but genuine control: it holds **destination depth fixed at early (layers 0–8)**
while varying **which depth the injected content was captured at**. Four arms land at the identical
destination:

| arm | content source | destination | restored / n | rate | Wilson 95% CI |
|---|---|---|---|---|---|
| early `reference_transplant` | early (native) | early | 5/25 | 20.0% | [8.9%, 39.1%] |
| early `random_equal_norm` | noise | early | 3/25 | 12.0% | [4.2%, 30.0%] |
| mid `shuffled_window` | mid | early | 4/25 | 16.0% | [6.4%, 34.7%] |
| late_matched `shuffled_window` | late | early | **10/25** | **40.0%** | [23.4%, 59.3%] |

Late-captured content, even forcibly relocated to early write positions — losing whatever destination-depth
leverage late naturally has — restores *more* often than any of: noise at early, mid-content relocated to
early, or even early's own native content written at its own native site. At n=25 the CIs are wide and the
late-shuffled vs. early-reference gap is only barely outside overlap (39.1% vs 40.0%) — **suggestive, not
decisive** at this sample size.

**This finding carries a real, disclosed confound this artifact cannot rule out.** `ffn_write` is a
transplant (substitution) of the captured `ffn_out` vector, not an additive nudge — writing "what layer 24's
MLP block produced" at layer 2's hook substitutes a vector whose raw norm may differ systematically from
what a layer-2-native vector would have (residual-stream / per-layer output norms commonly grow with depth
in transformer architectures — a plausible, unverified structural property here). `store_tensors=False`
means the raw captured vectors are not retained in this artifact, so their norms cannot be checked
post hoc. If late-captured vectors are simply larger in magnitude than mid-captured ones, part of this
result could be a residual leverage-by-magnitude effect smuggled back in through the content axis, not a
clean content-specificity result. Disclosed rather than asserted past what the data support.

## Cross-check against the other two artifacts

`quant_task_specificity.json`'s own depth-gradient replication (`docs/research/QUANT_TASK_SPECIFICITY.md`,
"Verdict histogram and depth gradient" section) closely reproduces the population study's ffn numbers
(early 21.8% / mid 32.7% / late 54.1% vs. 20.6/33.2/53.1) and extends the same gradient to the **head**
hook (early 3.4% / mid 8.6% / late 9.3%) — a much flatter, lower-magnitude version of the same monotonic
shape in a mechanistically different hook type. That a *different* computational surface shows a
qualitatively similar (if far weaker) depth trend is mildly more consistent with *some* shared positional
component across hook types than with a purely ffn-specific damage story — but head's own gradient is
far too shallow (3.4%→9.3%, vs ffn's 20.6%→53.1%) to explain ffn's much larger swing, so this is weak,
directionally-mixed evidence, not a tiebreaker either way. **`QUANT_TASK_SPECIFICITY.md`'s own Limits
section does not currently restate the depth-vs-leverage caveat at all** (checked directly — no match for
"leverage" in that file) even though its Method section reports the same depth gradient the confound
bears on; this document does not add one, since that doc's central claim (task-independent recurrence,
not a depth-vs-leverage claim) is not what this analysis bears on.

## Synthesis — what this does and does not establish

Three largely independent readings of the existing data (normalized reference-arm recovery jumping
specifically at late while the content-blind proxy does not track depth; the late band's random-arm
failures being total rather than merely magnitude-scaled; and a destination-depth-held-constant comparison
favoring late-sourced content) all point the same direction: **more of the late-band advantage looks
attributable to something specific about the information captured at late layers than to raw positional
leverage alone**, more than the existing docs' neutral "cannot cleanly separate" framing implies. This
analysis deliberately does not convert that into a percentage split (e.g. "70% content, 30% leverage") —
no measurement here isolates the two cleanly enough to support a number like that; doing so would be false
precision.

**What remains unseparated, stated plainly (this is not "fully resolved," and claiming so would be the
failure mode this project's own standing rule warns against):**

1. **No amplitude-matched controlled experiment has been run.** The mixed-precision doc's own
   falsification note already specifies the design that would close this: amplify an early-band transplant
   to the *same empirical logit-movement magnitude* late transplants achieve, and check whether it then
   restores comparably. Line 1 above is an indirect proxy argument (what does content-blind noise do at
   each depth), not that direct manipulation. This remains the one design that would actually settle it.
2. **The Line 3 finding carries an unverified magnitude confound** (residual/per-layer vector norm by
   source depth) that this artifact cannot check, because raw vectors were not retained
   (`store_tensors=False`). A cheap, well-specified next step: rerun the shuffled arm with an explicit
   norm-matching step (scale each relocated vector to the destination site's own typical norm before
   writing) — this reuses `_run_window`'s existing site-remapping machinery (the same mechanism
   `_pick_shuffled_sites`/`shuffled_vectors_by_dst` already implements) and would need a new GPU run, not
   attempted here per this analysis's scope.
3. **n=25 live disagreements per band**, already flagged as underpowered by the source doc's own marginal
   CIs; every further stratification in this analysis (by candidate, by outcome bucket) shrinks that
   further. The Line 3 table's widest gap (40.0% vs 20.0%) has CIs that only barely fail to overlap.
4. **Single model family** (Qwen2.5-7B-Instruct), ffn hook only, band-level joint transplant design.
   Nothing here bears on replication across architectures.
5. **This is orthogonal to the circuit-vs-quantizer-geometry question** `QUANT_TASK_SPECIFICITY.md`
   already investigated (task-independent recurrence). "Content-specific" here means the captured
   direction carries information beyond its magnitude and its write position — not a claim about a
   reusable computational component.

## Changes made to the shared caveat

`docs/research/QUANT_REGRESSION_POPULATION.md`'s and `docs/research/QUANT_MIXED_PRECISION_BANDS.md`'s
existing depth-vs-leverage caveats were **not deleted or reversed** — this analysis did not run the
controlled experiment that would justify that. Both received a short pointer to this document summarizing
that the evidence weight has shifted (not resolved) and citing the concrete numbers above.
`QUANT_TASK_SPECIFICITY.md` was left unmodified: it does not currently carry this caveat, and its central
claim is not about depth vs. leverage.

## Reproduction

```
python scripts/tracer/depth_vs_leverage_analysis.py [--report path/to/quant_mixed_precision_bands.json]
```
Analysis only — no engine calls, no new capture. Requires `quant_mixed_precision_bands.json`
(gitignored; not regenerated by this script — see `scripts/tracer/quant_mixed_precision_bands.py`'s own
docstring if it needs to be reproduced, which requires a live GPU run, not part of this analysis).
