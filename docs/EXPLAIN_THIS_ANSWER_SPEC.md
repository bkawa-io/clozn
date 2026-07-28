# SPEC — "Explain this answer": inspect any reply (the core surface)

> **Historical feature specification (2026-07-04).** It records the earlier memory-card product shape
> and old Studio terminology. Do not use it as current-user documentation; see
> [CAPABILITIES.md](CAPABILITIES.md).

*2026-07-04. The mainstream front door: a normal user, chatting normally, taps "explain" on any reply
and sees — measured, never self-reported — how confident the model was, what influenced the answer and
where, and what it'd look like with the dials changed. The studio becomes the power-user basement; this
is the feature everyone actually uses. Built almost entirely from parts that already exist (the Run
Inspector is ~70% of it). Read the memory files + FINDINGS.md first; house rules bind.*

## The principle (do not violate — it's the whole point)

Every panel answers a question the model **cannot be trusted to answer about itself** (blindness, 4
scales). So every signal is MEASURED, never narrated:

- **confidence** = read the token distribution (how *decided* each word was), NOT an asked-for "0–100"
  (that scalar is dead — it saturates). And label it honestly: **confident ≠ correct** (hallucinations
  are high-confidence). The panel shows commitment, not truth.
- **influence** = toggle it off and measure (ablation), NOT "which memory do you think you used?"
- **concepts** = read the SAE features that fired, NOT "what were you thinking about?"
- **the narration** (v2) = the model composes a *why* CONSTRAINED to the receipts, and its unconstrained
  instinct is run in parallel and DIFFED — the confabulation-gap experiment, shipped live.

Invariants on every field: `causal_verified` (active vs. patched-and-measured) travels with each claim;
"no receipt for that" is a first-class answer when the record is thin; reproduce **turn-N's** influence
manifest, not today's; use the **actual** (post-truncation) input that was sent, not the raw transcript.

## What already exists (assemble, don't rebuild)

`runlog` logs per reply: `messages` (the recipe), `response`, `memory{mode,cards_applied,applied_ids,
gate}`, `behavior.active_dials`, `trace{tokens,confidence,alternatives}`, `timing`. `replay.py` does the
greedy ablation (`/runs/<id>/replay` with `memory_off|disabled_memory_ids|behavior_off|behavior_overrides
|greedy`), state restored in `finally`. heavn Replay renders the token timeline, influence column, per-card /
per-dial receipt buttons + delta strips. `memory_cards` carries provenance quotes. The engine emits
`sae:<id>` concept readouts on the stream. **The missing work is assembly + a lightweight surface + the
narration, not new primitives.**

## Milestones (ship in order; each is independently useful)

### M1 — `/runs/<id>/explain` (assemble the FREE signals; zero generation) — Sonnet, ~1 day
One endpoint that reads the run and returns a structured `explanation` with no model calls:
- **confidence**: from `trace` — the K "uncertain moments" (tokens below a threshold), each with its
  alternatives; a one-line "N hesitations" summary. NEVER a single aggregate % (that's the dead scalar).
- **influences_active**: the manifest — cards that fired (each with its provenance quote + gate value),
  dials that were on. Free — already logged. Each tagged `causal_verified:null` (active, not yet proven).
- **concepts**: if the run has `sae:*` readouts (engine path), the top features per span; else omit with
  an honest "concept readout needs the qwen/PyTorch substrate (SAE) — not available on this run."
Done: a captured run returns a complete explanation object; model-free tests over a fixture run.

### M2 — on-demand causal receipts (the honest ablation) — Sonnet, ~1 day
Per active influence, a "prove it" that runs the **rigorous** receipt (fixing the sampled-baseline seam):
regenerate BOTH arms greedy — greedy-WITH the influence AND greedy-WITHOUT — and diff *those two* (the
stored sampled reply is context, never a term in the subtraction). Uses `replay.py`; marks
`causal_verified:true`. A "prove all" batches the leave-one-out arms into one forward pass (the
batched-receipts win — free at bf16; re-verify at 7B nf4 before relying on it there). **Redundancy
guard:** if dropping A alone and B alone each show ~no effect but dropping both together does, report
"A+B are redundant; together they drive this" instead of "neither mattered" (leave-one-out's blind spot).
Cost note honestly surfaced: a front-of-context card ablation re-prefills the whole context (KV can't be
reused); a dial toggles at decode time (prompt KV reusable) — cheap. Done: rigorous per-influence deltas
with the redundancy case tested.

### M3 — counterfactual dials (interactive) — Sonnet, hours
Sliders in the panel that re-run the reply greedily via `replay.py {behavior_overrides,greedy}`, live.
Each dial shows its measured per-model dose (the dose-receipt rule — a 7B-calibrated dial derails a 1.5B).
Done: "make it warmer" re-rolls the reply and shows the delta.

### M4 — the accountable-self narration (the crown; honesty-critical) — Opus, ~2 days
Compose a natural-language *why*, and guard it:
1. **constrained**: feed the M1/M2 explanation object as context, ask the model to explain the answer
   USING ONLY those facts. Every clause traces to a receipt.
2. **unconstrained**: separately ask "why did you say that?" with NO receipts (the confabulation sample).
3. **diff + flag**: where the unconstrained claim isn't supported by the receipts, flag it inline:
   "⚠ it was about to credit X; no receipt for that." Show the constrained narration with the flags.
Done: on a run where the model confabulates an influence, the diff catches it (a gated `-m model` test
seeding a known divergence). This is `self_audit_*`'s finding as a permanent feature.

### M5 — the surfaces (the "not in the studio" path) — shipped
- **TUI**: `clozn explain <run>` renders the endpoint and `clozn inspect <run>` now prefers direct local-
  journal assembly (so a stopped gateway/model is fine), with `--last` and exact `--json` output. The inline
  hotkey during `clozn run` chat remains optional polish; the terminal-native view already carries the
  reply — confidence sparkline, influence list with quotes, "prove"/"what-if" prompts. Terminal-native.
- **the bridge for any client**: the `/v1/chat/completions` response already lands in `runlog`; return
  the `run_id` (response field or header) so a companion `clozn inspect` / side panel shows the
  explanation for the reply the user just got in *their* client. Chat anywhere, inspect on demand.
- **web**: add an "Explain" summary tab to the Run Inspector that pre-assembles M1 + the M4 narration, so
  a non-power-user gets the story without clicking each receipt (the studio already has the raw pieces).
Done: a user chatting through a normal OpenAI client gets `clozn_run_id` + `X-Clozn-Run-Id` and can run
`clozn inspect <id>` without generation or a live model; non-local ids fall back to the chosen gateway.

## Cost model (why it's feasible on a normal chat)
M1 free (read the log). M2 O(influences-that-fired), lazy (only on drill-in), batched. M3 one greedy
re-gen per drag. M4 two generations, on demand. Nothing eager per-turn except the already-cheap manifest
+ trace logging. The expensive case (front-of-context ablation re-prefills) is real — surface it, and
prefer batching / KV-snapshot reuse where the change doesn't touch the shared prefix.

## The trap (stated so no one builds it)
Do NOT add a plain "explain this" that asks the model to explain itself and prints the answer. That is the
confabulation machine with an icon. Any natural-language why MUST be receipt-constrained + confabulation-
diffed (M4). If M4 isn't built yet, M1–M3 show raw measured signals with NO narration — which is honest
and already better than anything shipping.
