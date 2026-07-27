# scripts/tracer — the causal-tracer validation + SAE feature studies

Reproduction scripts for the numbers quoted in `notes/CIRCUIT_TRACER_DESIGN.md` §5b–§5h (local
notes) and formerly tracked as `docs/BACKLOG.md` item 9 (retired — see `docs/PRODUCT_ROADMAP.md`).
All of them drive a **live clozn-server** over HTTP; none of them need
torch (the one exception is the W_dec export, which is a one-shot in `engine/core/tools/`).

| script | what it measures | needs |
|---|---|---|
| `causal_trace_battery.py` | 16 prompts × 7 categories through `clozn.analysis.tracer`; aggregate S4 predicted-vs-observed scorecard, verdict spread, node strength tiers, interaction gaps | any AR model + a J-lens sidecar |
| `sae_fidelity_vs_concentration.py` | per prompt: **causal fidelity** of the SAE reconstruction (substitute `h → h_hat`, re-score) vs **concentration** (do a few features carry the site?) + explained variance | Qwen2.5-7B + `~/.clozn/sae/andyrdt_l15` incl. `w_dec.f16.bin` |
| `sae_joint_vs_random.py` | joint ablation of the top-k features vs a matched **random-k** control, k = 1…all — the control that decides "sparse circuit" vs "distributed" | same |
| `layer_position_map.py` | **run this first.** (layer × position) mean-ablation map — where causal mass actually lives, before assuming any artifact can see it | any AR model |
| `edge_depth_profile.py` | routed fraction into the final position by *capture depth* — the profile that exposes when a single routed number is meaningless (§5f) | any AR model |
| `attn_knockout_scan.py` | per-position attention-knockout ranking (renormalized) into the final position — the cross-position measurement path patching couldn't give (§5g) | any AR model + `--no-flash-attn` |
| `attn_knockout_controls.py` | the knockout controls: self-cut null, sink behavior with/without renormalize, random-span floors | same |
| `span_selection_validation.py` | greedy span accumulation vs top-k singles vs matched random spans — why greedy is the search rule (§5h) | same |
| `loo_vs_knockout.py` | leave-one-out (delete text) vs knockout (sever reading) on the same target — the sign-flip receipt (deleting ' Japan' RAISES P(Tokyo)) | same |

## Running them

```bash
# engine, matched to the artifact you're testing
clozn-server ~/.clozn/models/Qwen2.5-7B-Instruct-Q4_K_M.gguf --port 8080 --gpu-layers 99 \
    --workers 2 --jlens ~/.clozn/artifacts/jlens/qwen2.5-7b-v1

python scripts/tracer/sae_fidelity_vs_concentration.py
```

Each writes a JSON alongside its console table.

## Four traps these scripts encode (each cost a wrong result first time)

1. **Position 0 is an attention sink.** Residual norm ~220× typical; the SAE emits 118 175 active
   features there and explained variance goes to −11 160. It is far out of distribution and will
   dominate any activation-ranked candidate list. `sae_fidelity_vs_concentration.py` excludes it
   explicitly — an earlier pass that didn't found nothing but the token `'In'`.
2. **Sum-of-singles ≠ joint ablation.** Interaction gaps on this stack run ≈ −60% (and −73% on the
   7B), so adding up single-feature deltas badly understates a set's joint effect. The first pass
   reported "top-12 features = 6.3% of the site" by summing; the joint arms in
   `sae_joint_vs_random.py` put the same quantity at 10–45%. Always run the joint arm.
3. **A multi-token entity's causal mass sits at its LAST tokens.** "Zorblax" spans positions 9–12;
   position 9 carries +0.12 and position 12 carries **+8.58**. Under causal attention only the
   final tokens have seen the whole name. Scan the whole span — `layer_position_map.py` does.
4. **A cross-position `routed_fraction` is a lower bound, not a measurement.** Single-site patching
   can't hold the path (the source re-supplies it downstream), and holding the whole destination
   column doesn't fix it either, because the last layer materializes only the logit rows
   (`inp_out_ids`) and is unpatchable. Late-layer sources read a flat 0.0% at every depth — which
   is physically impossible, and is the tell. Same-column edges are unaffected. See §5f.

## Headline results (Qwen2.5-7B, andyrdt_l15, 2026-07-20)

- Causal-trace S4 scorecard: **88/96 = 91.7%** correct flip predictions (9B, 16 prompts).
- SAE reconstruction preserves **~99.9% of causal content** while capturing only **57% of variance**
  — variance and function dissociate.
- **No single feature is load-bearing** (~0.5% of a site each, ~47 active), but **top-8/16 jointly
  carry 10–45%** vs 0–6% for matched random-k: real feature-level structure, not sparse.

Scope: one SAE, one layer, one model, a handful of prompts. These are measurements of *this
dictionary at layer 15 of this model*, not general claims about SAEs or circuits.

Knockout/provenance headline (9B unless noted): source-vs-competitor separation **159×**;
induction **1216×** and in-context k/v **2232×** span-vs-control ratios; renormalize is
mandatory (without it the position-0 sink ranks top at +0.717 — a pure amplitude artifact);
greedy beats best-single +7.60 vs +0.11 (redundancy makes single positions score ~0). The
scorecard caveat stands: 91.7% is accuracy at predicting *token flips*, not answer loss.

## Remaining work (moved here from docs/BACKLOG.md item 9 when the backlog was retired)

Ordering and gates live in `docs/PRODUCT_ROADMAP.md` (lanes R1/R5, Phase 3/4 UI). The concrete
items, closest to the code:

1. **Genuine screen-null** — replace the target concept rather than diluting it; the one control
   that could still invalidate the S0 screen. Cheap. (R1)
2. **Second model family** — everything above is Qwen; `FAILED_CONTROLS` has never fired on a
   real prompt, so discrimination is unproven. Llama-3.1-8B GGUF is already in
   `~/.clozn/models/`; no lens needed for the norms-only screen. (R1/R5)
3. **Attention-heatmap vs causal-rank head-to-head** — `kq_soft_max` flows through the same
   eval_cb the capture plane uses; needs the (existing) `--no-flash-attn` path. ~1 day. Turns
   "attention is correlational" from an answer into a number. (R1)
4. **Attention-head node units** — `kqv_out-<il>` is named per layer and materialized at all
   positions, so it dodges the `inp_out_ids` last-layer blocker. The honest path toward anything
   deserving the word "circuit". (R5)
5. **Run-journal input mode** — `clozn causal-trace <run-id>` instead of ad-hoc prompts. (product)
6. **Studio click-a-token panel** — the north-star surface; gate any "why" copy on the measured
   ~24% legibility. (product)
