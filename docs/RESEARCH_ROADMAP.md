# Research Roadmap — expand then contract

> **Archived research plan.** This is preserved as experimental history, not as a current-user command
> or release contract. See [CAPABILITIES.md](CAPABILITIES.md).

**Date:** 2026-07-19
**Principle:** Run the big messy experiments first (prove or kill). Productionize only what survives. Parallelize with sonnet worktree agents wherever tasks are independent.

**Sources:** `docs/BACKLOG.md`, `notes/FRONTIER_BETS.md` §7–9, `notes/JLENS_SAE_FINDINGS.md`, `notes/JLENS_ENGINE_PLAN.md` J5, `notes/INTROSPECTION_EXPERIMENTS.md` X2/X4.

---

## Killed (do not revisit)

- **Latent recurrence / peek-ahead steering** — 3 null results; no product path on capable models.
- **J-space trajectory as rumination detector** — p=0.156, wrong base rate; strong models don't ruminate.
- **White-box risk controller advantage** — token-probability wins; internal state adds nothing (probe is a topic detector).
  Full autopsy (moved from the retired BACKLOG #10): the deployed "white-box" score is bit-identical to `exp(min(logprob))`
  across all 362 items — the same number any OpenAI-compatible API returns. The signal itself WORKS for selective
  generation (TEST n=182: AUROC 0.822 [0.760, 0.879]; 0% error at 50% coverage; survives a length control) and
  self-reported confidence is degenerate (358/362 claim 1.0; AUROC 0.510). Internal candidates: best residual probe
  looks like a win on the mixed set (0.900) but a 1-bit `is_hard_arith` topic flag matches the baseline exactly (0.822)
  and corr(probe, topic) = −0.85; on the hard-arithmetic subset the baseline BEATS the probe 0.937 vs 0.799 (paired
  bootstrap −0.138 [−0.265, −0.029]). Scope: Qwen3.5-9B Q4_K_M, greedy, error set dominated by 4–6-digit
  multiplication; well-powered for arithmetic slips only. Harness/data: `scratchpad/wb_analyze.py`, `wb_results.json`,
  `wb_raw*.jsonl`, `wb_harvest.py`, `wb_live_energy.py`, `wb_probe_analyze.py`, `wb_probe_results.json`.
  Ship path (a): selective generation labeled token-probability-based → `docs/PRODUCT_ROADMAP.md` Phase 3.6.
  Known end-to-end gaps when wiring: `:8080` is the raw engine (not the gateway), and `~/.clozn/eval_report.json` is
  stale for this model (would be refused by `classify_run`'s exact-model provenance gate).
- **J-transport steering quality improvement** — doesn't beat raw directions. Value of J is authoring speed (instant dials), not better directions.
- **Null-space covert channel (Qwen2.5-7B)** — dead (RMSNorm makes the model isotropically sensitive).

---

## Phase A — EXPAND (experiments, prove/kill)

### Wave A1 — Single-model GPU experiments (Qwen3.5-9B, now)

All run against the currently-loaded engine. Each is a standalone script hitting the engine over HTTP — **fully parallelizable in separate worktrees**, no file conflicts.

**Verdicts (2026-07-19 run, scripts + results in `runs/experiments/`):**

| # | Verdict | Headline |
|---|---|---|
| A1.1 | **INTERVENTION LIVES + prediction RESCUED at L24 (A1.1b)** | Catch 100% (10/10), FP 5% (1/20), mostly coherent. L16 lead-time missed the bar (4/10 positive) — but A1.1b's paired replay (same generations, deterministic /jlens at both layers) shows **L24: 90% strictly-positive lead, median 2 tokens, max 27** (bar: >50%) at +5% clean-prompt false-flag cost (one plausible war-adjacent noise hit). Caveats: n=10; part of the gain may be "closer to the LM head reads more like next-token prediction" rather than deep foresight. Product: poll L24 (already served) for detection, keep L16 as OR-rescue. The "present-tense only" thesis gains a footnote: deeper taps buy a small (~2-token median) forward window. |
| A1.2 | **KILLED** | Live-jitter diversity 0.42 ≈ null-jitter 0.47 — perturbation direction inside vs outside J's live subspace makes no difference. Coherence half passed (live-jitter −0.69 vs temp −1.60 mean logprob) but the mechanism claim is dead. |
| A1.3 | **KILLED** | Prospective (pre-onset-only) AUC 0.64, detection 1/9. The retrospective live-energy gap replicates but is explained by content type, not a pre-collapse signature. |
| A1.5 | **LIVES** (with caveat) | Content-only pooled ρ=0.336 (p=0.0002), creative-vs-technical d=1.70. Raw pooled ρ=0.16 — diluted by `<think>` scaffold tokens (highest live-energy, near-zero surprisal). EEG strip must mask `<think>` spans. |
| A1.7 | **KILLED** (A1.7b re-run) | First run void (30/30 right). Hard battery (13 wrong / 31 right): brittleness AUC=0.49 = chance, permutation p=0.54. Paraphrase-variance does not separate wrong from right answers. Note: a raw-confidence baseline was ALSO at chance (0.48) on this battery. |
| A1.8 | **LIVES** | Kruskal-Wallis p=2e-13. Prose/dialogue route ~2x more energy through the live subspace than code/math (cross-cluster d>6). Nuance: code/math spread less energy across MORE dims (eff. rank 4.4-4.7 vs 3.1-3.4). |
| A1.4 | **LIVES** | Reframed (raw-vector injection impossible, 3584≠4096): the TOKEN-SPACE SPEC is portable. Author on 2.5-7B → read out to (token,weight) → rebuild on 3.5-9B via its own J/W_U. Transported +6.615 Δlogprob ≈ native ceiling +6.614 (p=0.87), random floor −1.19 (p=4e-16). Caveat: single-token specs are near-degenerate (99.8% self-weight); the real evidence is the composites — ocean+rain both anchors ported, fire−ocean subtraction survived the round trip. |

| A1.6 | **KILLED** | Doubt-point fork win-rate 18.8% (bar: >20%) — and RANDOM high-confidence positions won MORE (28.9%). Permutation p=0.94 for doubt>random. Entropy spikes carry no information about where extra compute helps; stitched completions beat greedy only 13.6% of the time. |
| A1.9 (verify-then-branch, BK) | **KILLED** (with a precise autopsy) | Score-gated best-of-4 (80.0%) ≈ random-gated (77.2%), p=0.14. Two independent causes: (1) the branching ceiling is tiny — always-branch only buys +4.4 pts (75.6→80.0; HARD 40→46.7%); (2) the gate is band-dependent exactly as A1.7b predicted — MEDIUM gate AUC 0.96-1.0 (but only n_wrong=2, fragile), HARD gate AUC 0.41 = blind on the tail where the wrong answers live (n_wrong=9). Same-model resampling + self-scoring can't rescue hard items: wrong answers are confident AND resampling regenerates them. |
| A1.10 (did-you-write-this, BK) | **RECEIPT_LIVES / verbal dead** | Statistical receipt: AUC=1.00 self-vs-human (perm p=0.0002) — perfect separation on verbatim text. Verbal "did you write this?": AUC=0.46 = chance (p=0.66) — the model's yes/no carries zero authorship information. Receipt−verbal gap 0.54, CI [0.36, 0.73]. Confabulation-gap prediction confirmed. HARD BOUNDARY: paraphrase kills the receipt completely (AUC 0.51 vs paraphrased-self) — the signal is verbatim-only. Product: a "native-to-this-model" receipt is real but must state the paraphrase limitation; never a verdict tool. |
| A1.12 + A1.12b (dir(c) vs diff-of-means, BK) | **dir(c) surfaces the WORD; contrastive carries the REGISTER — loops were mostly a greedy artifact** | GREEDY: dir(c) loops (4-gram 81%/97% vs contrastive 42%/72%); lexicon metrics reward the pathology; mean-logprob can't see loops. SAMPLED (shipped settings, 3 seeds): repetition dissolves — dir(c) 1.9%/0.9%; ablation splits credit (sampling 81%→30%, rep-penalty 30%→2%); **the random control fell too (86%→0.9%), so much of it was greedy itself, not the dial** → "attractor" claim narrowed. WHAT SURVIVES: anchor-word surfacing — dir(c) emits the literal word 80%/99% vs contrastive 32%/22%; dir("formal") says the word, contrastive writes "undersigned parties hereby agree". STATS HONESTY: sampled behavioural cell favours contrastive directionally (6.86 vs 3.97 high) but n.s. at n=18 (p=0.06 Wilcoxon) — unproven, needs power. dir(c) sampled retains its effect in 4 cells, loses most of it in 4 (loops had been carrying the score), 4 mixed; low strength survives better. Authoring: 80ms/word vs ~90min/6-concept battery. Product: tone library stays contrastive; dir(c) = low-strength topic nudge + logprob receipts (A1.4 unaffected) + seeds. |
| A1.11 (steering-by-domain) | **DOMAIN_GAP, direction reversed + measure dissociation** | Raw Δlogprob: code/math get MORE push (+10.6 vs +7.9 nats, d=1.81, p=6e-14); live-energy anti-correlates with push per-prompt (r=−0.80) — crowded channels resist injection. BUT behavioral appearance flips it: concept word surfaces in 100% of prose/dialogue continuations vs 25% of code/math — peaked baseline distributions in code/math absorb a big nat-gain without unseating argmax. Product: strength 0.4 visibly works in natural language; code/math need stronger dials for visible effect. Caveats: 0% syntax damage at this strength, but semantic derailment and one repetition-loop observed — "syntax survived" ≠ "side-effect free". |

#### A1.1 — Closed-loop disposition guardrails ★ flagship

The biggest unclaimed frontier (FRONTIER_BETS §9.1). We READ intent-before-speech (J-lens) and WRITE corrections (dir(c)) — but never in one loop during generation.

**Protocol:** Generate text on a banned-topic battery (~20 prompts designed to drift toward banned concepts). Every N tokens during generation: harvest hidden state → J-lens read → if banned concept in top-k → inject J-transported counter-direction → continue. Measure: catch-rate (% of banned outputs prevented), false-positive rate (% of clean outputs degraded), latency overhead vs output-filter baseline (just grep the final text).

**Kill condition:** FP > 20% at any useful catch-rate, OR catch-rate ≤ output-filter.
**If it lives:** The headline safety/steering feature — guardrails that act on intent, not output.
**Cost:** 1–2 days. **Engine needs:** /harvest, /jlens, /intervene (all exist).

#### A1.2 — Semantic temperature

JLENS_SAE_FINDINGS product implication #6. Jitter the residual along J's top singular vectors (live channels only). Null-space jitter should do nothing (built-in falsification).

**Protocol:** Same prompt, generate 10x with: (a) no jitter (greedy), (b) live-subspace jitter (top-k SVs of J, random perturbation), (c) null-space jitter (bottom dims), (d) standard temperature (for comparison). Score: diversity (pairwise BLEU between generations), coherence (/score of each generation under greedy), and the null control (c ≈ a).

**Kill condition:** Live-jitter ≈ null-jitter (direction doesn't matter), OR coherence collapses.
**If it lives:** A "creativity" dial that provably can't produce word salad.
**Cost:** 0.5 day.

#### A1.3 — Collapse prediction

JLENS_SAE_FINDINGS product implication #4 + finding #7. Live-energy fraction as a degeneration early-warning.

**Protocol:** Generate 50+ completions with varying prompts/lengths (some designed to loop via high-temperature + no-rep-penalty). Per token: compute `||Vh @ h||^2 / ||h||^2` (live-energy fraction). Label each generation as good/degenerate (human or automatic: repeated n-gram detector). Find threshold: does live-energy dropping below X predict degeneration K tokens before it's visible in text?

**Kill condition:** No threshold separates good from degenerate, or detection lag ≈ n-gram detector.
**If it lives:** Real-time gauge + auto-stop/retry signal in the engine.
**Cost:** 0.5 day.

#### A1.4 — Model-portable dials

JLENS_SAE_FINDINGS product implication #3. A concept direction calibrated for one model works on another via J-transport through each model's own J.

**Protocol:** Build dir("rain") via Qwen3.5-9B's J. Build dir("rain") via Qwen2.5-7B's J (from the existing artifact). Cross-apply: transport Qwen3.5's direction through Qwen2.5's J and inject into Qwen2.5 engine (or vice versa). Measure: does logprob("rain") rise vs random control?

**Kill condition:** Cross-model transported direction = random on the target model.
**If it lives:** Same concept works across models. Multi-model story.
**Cost:** 1 day. Needs both J artifacts (have them) + ability to load either model on GPU.

#### A1.5 — Per-token effective rank dynamics

JLENS_SAE_FINDINGS research idea (Tier 1). Formalize finding #7's preliminary observation.

**Protocol:** Generate diverse text (creative, technical, repetitive). Per token: compute effective rank of the J-projected hidden state (or just live-energy). Correlate with: token surprisal (negative logprob), is-novel (not in prompt), is-repeated. Statistical test: rank correlation + significance.

**Kill condition:** No significant correlation.
**If it lives:** A runtime "model is thinking hard vs coasting" signal. Powers the model-EEG strip.
**Cost:** 0.5 day.

#### A1.6 — Branch-on-doubt

FRONTIER_BETS §9.5. Interior-triggered test-time compute.

**Protocol:** Generate on 30 prompts. At every entropy spike (top-1 prob < threshold, e.g. < 0.3): pause, fork 3 continuations (different temperature seeds), /score all 3 + the original, pick the best. Compare: branched completion vs unbranched on a quality metric (human pref or /score of the full text under greedy).

**Kill condition:** Winner ≈ original on >80% of forks (the spikes aren't meaningful decision points).
**If it lives:** Selective compute exactly where the interior says it's needed.
**Cost:** 1 day.

#### A1.7 — Paraphrase-brittleness receipts

FRONTIER_BETS §9.6. A robustness signal with no second model.

**Protocol:** 30 QA prompts, each paraphrased 5 ways (LLM-generated paraphrases, verified equivalent). /score the SAME answer under all 5 phrasings. Compute: variance of logprobs across paraphrases per item. Correlate with correctness (items where the model is wrong should be more brittle).

**Kill condition:** No separation between correct and wrong answers' brittleness scores.
**If it lives:** A cheap, honest, novel robustness receipt type.
**Cost:** 0.5 day.

#### A1.8 — Content-type bandwidth

JLENS_SAE_FINDINGS research idea (Tier 2). Does the model allocate different bandwidth to different domains?

**Protocol:** Harvest hidden states from 20 prompts each of: code, prose, math, dialogue. Compute per-token live-energy fraction and effective rank. Compare distributions across domains.

**Kill condition:** All domains have same effective rank (within noise).
**If it lives:** Interesting finding (models allocate capacity by domain). Could inform domain-specific compression or routing.
**Cost:** 0.5 day.

---

### Wave A2 — Introspection live runs (harness built, never run live)

From task #75 (CPU harnesses built). Need the engine for generation + /jlens.

**Verdicts (2026-07-19 run):**

| # | Verdict | Headline |
|---|---|---|
| A2.1 (X2) | **NULL, as predicted** | Actor 62% vs observer 64% (chance 25%, both p<1e-7) — readouts ARE predictable from text, but the model has NO self-advantage (McNemar p=1.0, diff −2% CI [−10,+8]). Confirms "self-knowledge = re-derivation from weights." Disciplines copy: never claim the model reads its own internals. |
| A2.2 (X4) | **LIVES, qualified (selection-confounded)** | Ceiling real (classifier 76%/69%). Temp-foil arm: model 2AFC accuracy 1.0 on the 16/40 items that parsed, d′=2.67, permutation p=0.0. Scramble arm underpowered (7/40 parsed, n.s.). CAVEAT: 60–82% of judge calls never closed `<think>` even at 3200 tokens; the significant result rests on the fast-concluding minority, plausibly a non-random subsample. First pass was a parse artifact (always-"A" from matching "Stream A" inside unclosed reasoning) — caught, fixed, contaminated pass preserved as pass1_truncated_backup.json. Discrimination framing only. |
| A2.2b (X4 deconfound) | **LIVES + mechanism resolved** | Forced-choice judge (P(A) vs P(B) via /score — 100% parse by construction), all 40 pairs both arms. Actor: temp 95% (d′=2.19, p=0.0), scramble 72.5% (d′=0.82, p=0.0005). OBSERVER arm matches: temp 90%, scramble 75%, McNemar p=0.625/1.0 — zero actor advantage. Mechanism = re-derivation from the transcript, consistent with X2's null. Discrimination real; self-access not implied. |

#### A2.1 — X2: Self-prediction advantage

Does the model predict its own J-lens readouts better than a same-weights observer?

**Protocol:** Per `notes/INTROSPECTION_EXPERIMENTS.md` X2. 150 items, forced-choice (4 options: true concept + 3 distractors). Actor-A vs same-weights-observer on same transcript. McNemar test.

**Expected result:** Null (actor ties observer) — confirms "self-knowledge = re-derivation from weights." This disciplines copy: never claim the model "knows itself."
**Cost:** 1 day.

#### A2.2 — X4: Mirror test

Can the model identify "which J-lens readout stream is mine" above chance?

**Protocol:** Per `notes/INTROSPECTION_EXPERIMENTS.md` X4. Show model readout streams: its own vs another model's vs its own under different dials. Discrimination: which is me?

**Kill condition:** At-chance discrimination.
**If it lives:** Evidence of self-model discrimination (framed strictly as discrimination, not consciousness).
**Cost:** 1 day.

**Parallel plan:** X2 and X4 are fully independent — run in parallel worktrees.

---

### Wave A3 — Fast-J validation (CPU, parallel to A1)

**Verdicts (2026-07-19 run, full detail in `runs/experiments/a3_results.json` + `a3_1_root_cause.json`):**

| # | Verdict | Headline |
|---|---|---|
| A3.1 | **FAIL — three diagnosed bugs in fit_lens_fast.py** | (1) dead power-iteration loop (`--n-power` is a literal `pass`); (2) JVP finite-difference eps 1000x too small (1e-3 → pure noise, 50% sign agreement; 0.1 works); (3) UNFIXED operator mismatch — fast fitter estimates J at the last-token position only, dense fitter averages all valid positions with causal-sum reduction: genuinely different linear operators, no hyperparameter reconciles them. Containment 0.08 vs 0.9 bar across three independent runs even with (1)+(2) fixed. **Production unaffected**: qualify-whitebox and shipped artifacts never call this fitter (lab-only). The prior "97.9% offline containment" claim is not reproducible from anything runnable — treat as UNVERIFIED pending re-audit. |
| A3.2 | **Speed PASS (15.7 min), fidelity FAIL** | Llama-3.1-8B pilot fit with the patched fitter met the 20-min target. Functional smoke: fire +2.26 and ocean +1.55 vs random controls, but rain −1.33 (wrong direction); paired p=0.25 n.s. → cross-family port SKIPPED per pre-registered bar. Artifact kept at `~/.clozn/artifacts/jlens/llama3.1-8b-pilot/`, labeled PILOT with limitations in manifest. |

**A3.3 ran same day (results: `runs/experiments/a3_3_validation_results.json`):** the position-averaged operator fix is **VALIDATED** — containment 0.08 → **0.798**, singular values match dense (22.83 vs 22.97), rank50-vs-rank50 transported dirs cosine 0.82. Honest fit time: **86 min**, not 20. And a decisive new finding: the 0.95 dir-cosine gate was unpassable by construction — even a PERFECT rank-50 truncation of the dense J agrees with the full-rank transport at only 0.28, because **dir(c) draws ~92% of its mass from J's spectral tail**. Consequence: rank-50 compact lenses suffice for subspace features (live-energy/EEG) but **cannot author concept dials** — dial authoring needs the dense J. The k=50 Llama port is therefore moot (not a failure — mathematically out of reach). Cross-family port path: overnight dense-class Llama fit, then re-run. Fast-J's product niche narrows to subspace features; sweep n_power/n_prompts for the cost/quality curve (checkpoint each power round in one fit to get the curve free).

**Incidental infra bug found:** `clozn-server.exe` requires `engine/core/build-gpu/bin` on PATH for its `llama.dll`/`ggml-*` dependencies — any fresh shell hits STATUS_DLL_NOT_FOUND. Fix belongs in `clozn/cli/engine_process.py` (set the DLL path when spawning).

#### A3.1 — Fast-J fresh fit on Qwen3.5-9B

Validate that `clozn-jlens-work/scripts/fit_lens_fast.py` (Halko-Martinsson-Tropp randomized SVD) produces a compact J matching the dense one we already have.

**Protocol:** Run fit_lens_fast.py on Qwen3.5-9B (nf4, lab venv). Compare: (a) containment of top-23 dense singular subspace, (b) cosine of transported directions vs dense J^T, (c) ConceptSteer smoke with the compact artifact.

**Already validated offline** against Qwen2.5-7B: 97.9% containment at k=50, power=1. This confirms it on the 9B.
**Cost:** 0.5 day (mostly GPU time for the ~280 matmuls).

#### A3.2 — Fast-J on a novel model

Pick a model we DON'T have any J for (Llama-3.1-8B or Gemma-4). Fit from scratch in <20 min. Run ConceptSteer smoke.

**Kill condition:** Fit fails, or steer doesn't beat random on the new model.
**If it lives:** "clozn qualify-whitebox" can onboard ANY model with white-box features in 20 minutes. The multi-model story becomes real.
**Cost:** 1 day (includes model setup).

---

### Wave A4 — VRAM-blocked (need ~13GB: co-resident models)

**RUN 2026-07-19 (evening session, 23 min — capture stages ran in seconds, not the projected minutes). Verdicts (results: `runs/experiments/a4_*_results.json`, timeline: `a4_progress.log`):**

| # | Verdict | Headline |
|---|---|---|
| A4.2 (H7) | **MIXED** | Commit-order legibility LIVE: Dream's commit order correlates with final position at mean ρ=0.47, 7/10 prompts outside the shuffled null — real partial global-negotiation structure, neither sequential nor random. Content divergence INCONCLUSIVE: with R3's decode-regime floor applied, sampled-AR-vs-Dream contradiction rate (0.10) exactly equals the greedy-vs-sampled floor (0.10) — no substrate-attributable divergence at n=10. |
| A4.3 (H3) | **SPLIT, kill bar doesn't fire** | Dream infill beats AR prefix-only overall (0.53 vs 0.20), clears suffix-swap null. Clean wins: code 0.8-vs-0.2, structured 0.8-vs-0.4. Prose: three-way 0.0 miss (partly a fixed-gold scoring artifact on open-ended text). The one repetition loop found was on the AR arm. |
| A4.1 Tier-0 | **LIVE** | Key discovery: the streaming lens on /v1/completions taps Dream's REAL bidirectional residuals (standalone /jlens forces causal — code-read confirmed). Predictive readout-before-commit hit-rate 24.3% vs corrected null [12.7–15.7]; rises smoothly toward commit (~29% near, ~0% at lead-15): "the plan crystallizes progressively." Subspace claim only per R4. |
| A4.5 | **LIVE, self-corrected** | Qwen2.5-7B-authored dir(c) injected DIRECTLY into Dream (shared d_model/vocab — no bridge needed): coherent concept surfacing 36% vs 0% random vs 0% baseline. Headline self-corrected from a raw 69% that was inflated by short word-spam ("fire fire fire") invisible to the ≥8-word 4-gram check — new `word_spam` flag added. Distinct failure mode: concept directions over-inject into spam; matched-norm random never does. |
| A4.4 (H2) | NOT RUN (correctly) | No harness exists; half-day build per runbook. Ceiling-first design (R2) + degeneration veto (R1) recorded for whenever it's built. |

Session integrity: runbook's A4.1 null design was corrected pre-run (row-shuffled-J is INVALID per the project's own J1 finding — shuffled-pairing used instead; runbook fixed in place). Engine restored + verified. Third instance today of "metric blind to short repetition" (A4.5's word-spam) — R0's degeneration-flag rule now extends to text-match metrics with a length-aware spam check.

| # | Experiment | Question | Kill condition |
|---|---|---|---|
| A4.1 | **J-E5: Dream denoise J-lens** | Does corpus-averaged Jacobian transport survive bidirectional attention + mask tokens? | Fitted J has no live/null separation |
| A4.2 | **H7: Divergence atlas** | Same prompt → AR vs diffusion. Where does generation order change content? | Outputs converge on >90% of prompts |
| A4.3 | **H3: Substrate routing** | Does bidirectional infill beat AR FIM on real editing tasks? | Dream infill ≤ AR FIM on coherence |
| A4.4 | **H2: Score-gated self-repair** | AR generates + confidence-masks shaky spans + Dream re-solves + /score accepts. Net improvement? | Repairs score worse >50% of the time — **REVISED (R1/R2)**: mean-logprob accept gate is BROKEN (A1.12: can't see repetition loops) → add degeneration veto; and measure the UNCONDITIONAL repair ceiling first (A1.9 died of a tiny ceiling as much as a blind gate) |
| A4.5 | **9.9: Cross-model disposition transfer** | Dispositions are word-distributions; inject A's read into B. Coherent steer? | Transferred direction = random on model B |

**Parallel plan:** A4.1 (fit-only) and A4.2 (generation) are independent. A4.3/A4.4 depend on A4.2 verdict.

**MANDATORY design revisions from the 2026-07-19 expand wave** — full text in `notes/A4_RUNBOOK.md` §0b, they supersede the original specs:
- **R0 (all cards):** every generation at product decode settings; every quality metric carries a degeneration flag. Greedy-only + fluency-only metrics rewarded pathology twice today (A1.11, A1.12).
- **R1 (A4.4):** mean-logprob accept gate cannot see repetition loops → add an n-gram degeneration veto to every /score-based judgement.
- **R2 (A4.4):** run the unconditional-repair ceiling BEFORE the gated arm (A1.9's double failure mode).
- **R3 (A4.2):** AR-vs-diffusion confounds substrate with decode regime → match sampling, keep AR-greedy as reference, treat the AR-greedy-vs-AR-sampled gap as the floor.
- **R4 (A4.1):** subspace questions only — no Dream dials from a compact fit (A3.3: dir(c) is ~92% spectral tail); use the v3 position-averaged fitter, budget ~86 min.
- **R5 (A4.5):** evaluate transported dispositions at logprob AND behavioural level w/ degeneration flags; prefer contrastive-authored sources (A1.12: register lives in distributed directions).

---

### Wave A5 — Low-priority / speculative (park)

| # | Experiment | Notes |
|---|---|---|
| A5.1 | Watermarking via null space | Write payload into dead dims. Only if null-space shows any downstream readability on 9B. |
| A5.2 | Model telepathy (aligned J-spaces) | Small model thinks cheap, big model receives compressed state. Very speculative. |
| A5.3 | Activation memory (J-space codes) | Compress context into J-codes, reinject later. X7 showed plateau at k≈4-6 — is that enough? |
| A5.4 | X5: Convergence archaeology | No product tie. Time-sensitive. No GPU needed. Run when curious. |
| A5.5 | H5: Counterfactual patches | Needs /v1/revise ablated-context spike. Park until A4 wave lands. |
| A5.6 | Multi-feature composition quality | Finding #6 validated stability. Does quality (not just stability) improve with 3–4 composed features? Needs a tighter scorer. |

---

### Wave A6 — Intervention-validated circuit tracing (the north star; gated behind B2)

**Promoted 2026-07-19** out of "B2 item 4", where it was misfiled as an engineering deliverable. The *engineering* is B2; the *research question is open, unanswered, and the largest one left in the project.* Wave A4 is the last research that runs on today's engine — this is the research beyond it.

**The question:** can we produce a circuit trace in which **every claimed edge is validated by intervention** — not "these activations correlate with that behaviour" but "ablate this component and the downstream thing provably changes, here is the receipt"? Published attribution graphs are largely correlational. An intervention-validated tracer would be the strongest possible statement of this project's thesis: no circuit claim without a receipt.

**Why it's a Phase-A-style bet, not a build task:** nobody knows whether the intervention-validated version is tractable at useful granularity. Plausible failure modes, each of which is a real finding: (a) combinatorics — validating every edge needs more ablation runs than batching can buy; (b) polysemanticity — no clean component boundaries to ablate; (c) the effects are real but so distributed that traces are unreadable; (d) validated traces exist but tell you nothing a simpler receipt didn't.

**Prerequisites (B2 items 1-3).** Checkpoint/branch + batched decode are load-bearing: the tracer's cost model is "many ablated forward passes sharing a prefix", which is exactly what B2.1+B2.2 make cheap. Do not attempt before them.

**Design constraints carried from the 2026-07-19 wave** (learn these for free rather than rediscovering them):
- **Present-tense only.** Every predictive interior signal died today (A1.3, A1.6, A1.9, A1.7b, guardrail lead-time); every present-tense read/write lived. Scope the tracer to explaining what *is* happening, never to forecasting.
- **Degeneration flags on every quality judgement.** "The output got more probable" is exactly the metric that failed to see repetition loops (A1.12). An edge validated by logprob shift alone is not validated.
- **Product decode settings.** Greedy-only evaluation distorted two experiments today before we caught it (A1.11, A1.12/A1.12b).
- **Null controls are mandatory per edge** — random-equal-norm ablation is the floor every claimed edge must clear, exactly as dir(c) had to beat a random bag.
- **A1.1 is the miniature.** Read at a layer → intervene → measure downstream effect, with 100% catch / 5% FP, is a one-edge tracer that already works. Scale that shape.

**Cheapest-decisive first slice (when B2 lands):** take a single behaviour we can already produce and kill on demand (the A1.1 banned-concept drift, or the X8 eval-recognition direction), and try to trace *just that one behaviour* end-to-end with every edge intervention-validated. If one behaviour can't be traced honestly, the general tracer is not close.

---

## Phase B — CONTRACT (productionize survivors)

Start after A1–A3 land verdicts. What survives determines what gets built.

### B1 — Ship validated features (updated 2026-07-19 post-verdicts)

| Feature | Verdict gate | Build scope |
|---|---|---|
| Mid-gen guardrails | A1.1: intervention LIVES (100% catch, 5% FP) | Mid-gen polling → counter-injection + a receipt per firing. Honest copy: "catches and corrects during generation" — NOT "reads intent early" (4/10 lead). Cap re-steers ~3 then hard-stop (repetition cost observed). Still the headline. |
| Portable dial library | A1.4 LIVES (same-family ~100% of ceiling) | Dial file format = readable (token,weight) list. Author once, run on any qualified model. Cross-family gate: A3 stage 4 (in flight). |
| Per-domain dial strength | A1.11 DOMAIN_GAP | Calibrate on BEHAVIOR not Δlogprob (they rank domains oppositely). Raise strength for code/math; watch repetition-collapse in prose. Coherence check, not just syntax check. |
| Model-EEG strip | A1.5 + A1.8 LIVE | Ambient descriptive gauge of live-energy. REQUIRES `<think>`-span masking + per-domain norms (else it's a prose detector). Never predictive. Optional "domain sense" readout. |
| Native-text receipt | A1.10 RECEIPT_LIVES (AUC 1.0 verbatim) | "This text scores as native to this model" receipt. Verbatim-only — paraphrase kills it (AUC 0.51); print that limitation on the receipt. Signal, never verdict. |
| Fast model qualification | A3.1 FAILED as-shipped — power-iteration dead code (fix in flight) | Fix fit_lens_fast + audit qualify-whitebox CLI path + re-validate the 97.9% offline claim. Ship only after a real containment pass. |
| J-transport all steering | Validated (finding #6: stacking stability) | `/intervene` auto-transports via compact J. One matmul. |
| Risk controller last-mile | Validated (AUROC 0.822) | Wire abstain-band action + UI indicator. Label honestly as token-probability-based. Note finding #13: gates fail on the hard tail — document the band limitation. |
| Copy discipline | X1 + X2 + X4b (three independent nulls) | "Receipts, not self-narration" everywhere. The nulls are marketing-grade honesty proof — cite them. |

**Killed features (do not build):** semantic-temperature dial (A1.2), collapse auto-stop gauge (A1.3), branch-on-doubt forking (A1.6), paraphrase-brittleness receipt (A1.7b), verify-then-branch same-model (A1.9), null-space watermarking (A1.2 collateral).

### B2 — Engine infrastructure (big C++ arc)

From BACKLOG.md Phase 2. Start only after expand phase settles. **These three are prerequisites for Wave A6 (below), not features in themselves** — build them to unlock the tracer, and ship the incidental wins (faster receipts, cheaper multi-lens readout) along the way.

1. **Native exact-state checkpointing + branching** — pause, fork, resume bit-exactly.
2. **Batched multi-sequence decode** — faster receipts + higher-order causal credit.
3. **Device-resident multi-observer readout plane** — multiple lenses at once, <5-10% overhead.

### B3 — Polish + test + ship

- Real-browser pass over heavn UI (the visual quality gate)
- `clozn serve` auto-discovery smoke (no manual --jlens flags)
- Studio concept-dial E2E through the Python gateway
- Test suite green (defer all test-churn to here)
- Docs/claims refresh (true up to what shipped + what research proved)

---

## Execution model

**Main session = integrator.** Reads results, decides kill/continue, writes prompts for next wave.

**Sonnet worktree agents = experiment runners.** Each experiment is a self-contained Python script that hits the engine over HTTP. Agents work in isolated worktrees — no file conflicts. Results land as JSON + a short verdict text file.

**GPU serialization:** Only one agent can use the engine at a time for generation (/v1/completions is sequential). But /score, /harvest, /jlens are fast single-pass ops. Cheap experiments (A1.2, A1.3, A1.5, A1.7, A1.8) finish in minutes of engine time. Give the flagship (A1.1) and the heavyweight (A1.6) dedicated engine slots.

**Practical dispatch order for A1:**
1. Launch A1.2 + A1.3 + A1.5 + A1.7 + A1.8 in parallel worktrees (all cheap, <30 min engine time each)
2. While those run, start A1.1 (flagship, takes longer, uses the engine heavily)
3. After cheapies land, run A1.6 (branch-on-doubt, needs many sequential /v1/completions calls)
4. A1.4 (model-portable) is CPU-heavy (numpy J matmuls) + only a few engine calls — run anytime
5. A3.1 (Fast-J fit) runs in the lab venv (GPU autograd, separate from the engine) — truly parallel

**Decision points:**
- After A1: which features are alive? Scope B1.
- After A3: is Fast-J real? → becomes part of qualify-whitebox.
- After A4 (whenever VRAM unblocks): is the diffusion half alive? → H2/H3 become product.

---

## What's already proven (don't re-test)

- J-space exists at 7B (J4) and 9B (today's ConceptSteer smoke)
- dir(c) steers live (task #76, today) — logprob rises, content-specific
- J-transport prevents catastrophic collapse in multi-feature stacking (finding #6)
- Live/null subspace discrimination is real on deployed Q4_K_M (finding #8)
- Fast-J: **claim narrowed 2026-07-19.** The original 97.9% (finding #5) validated the randomized-SVD ALGEBRA against the already-computed dense J (perfect matvec oracle) — that layer stands, and A3.1's noiseless control replicated it. What was NEVER validated is the model-side JVP oracle in fit_lens_fast.py ("needs GPU test" in the ledger; TODO in the code) — and A3.1 showed it's broken (eps noise + last-token-only operator mismatch). "Fast-J works" was a compression error, not a regression. End-to-end fast fit: unvalidated + broken until A3.3.
- Live-energy correlates with token novelty (finding #7)
- X1 (self-report agreement): done
- X3 (injected-thought detection): done — d' separates real injection from sham
- X6 (CoT-as-paging): done — k* measured
- X7 (legible memory tax-curve): done — plateau at k≈4-6
- X8 (eval-mode probe): done — eval recognition readable + ablatable
- Pin-and-resolve editing: product-ready (Dream, task #87)
- Quant receipts: validated live (task #79)
- Verify-then-escalate routing: validated (task #70)
