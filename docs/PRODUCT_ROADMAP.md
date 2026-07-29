# Clozn Product Roadmap — 2026-07-20

> **Historical planning snapshot.** Status statements below describe the repository at their recorded
> dates and may name surfaces later removed. Use [CAPABILITIES.md](CAPABILITIES.md) for current-user
> capability and release status.

**The ordering authority for product work.** Synthesized from four inputs: the two independent
positioning audits of 2026-07-20 (`notes/PRODUCT_POSITIONING_2026-07.md` = market research per
persona; `notes/POSITIONING_AUDIT_B_2026-07.md` = repo/capability audit + feature matrices), the
open-work tracker (`docs/BACKLOG.md`), and the research ledger (`docs/RESEARCH_ROADMAP.md` +
`notes/FRONTIER_BETS.md`). Where the audits disagreed on sequencing, §2 records the resolution.
`docs/BACKLOG.md` is retired to a stub (its working detail moved to `scripts/tracer/README.md`,
`docs/RESEARCH_ROADMAP.md`, and §8 here); §12 maps its items and corrects two stale entries.
The forward-looking half of `docs/ROADMAP.md` is superseded by this file.
(§ numbers cited in the body refer to this file's sections unless a filename is given.)

Effort bands: **S** ≤ ~2 days · **M** ≤ ~2 weeks · **L** weeks-to-months · **R** gated on a
research result. Bands are planning aids, not commitments.

### Delivery status — 2026-07-23

**2026-07-23 run addendum (BK-directed autonomous run):**
- **R1 / provenance — WIRED.** The 41/41-validated `trace_provenance` now has a product surface:
  `clozn provenance [last]` + `POST /runs/<id>/provenance` (938601b). The R1 chip *label language*
  remains BK's product decision; the verb ships with the module's honest verdict strings.
- **Phase 3.6 — abstain/ask ACTION added (opt-in).** `clozn_selective`, default off, fail-closed
  without a calibrated profile, token-probability caveat on every firing (ff3d06f).
- **R5 / tracer — screen-null CLOSED + contrastive scoring shipped.** The pre-registered screen-null
  came back MIXED (absolute screen nominates answer-CATEGORY sites; a wrong token PASSes); the fix,
  contrastive+directional foil scoring, makes it answer-SELECTIVE (wrong token → 0 strong nodes 6/6)
  (1edcecf). `causal-trace` gained `--from-run`/`--contrast`/`--screen-mode` + an auto-downgrade
  robustness fix (0984ffa).
- **R6 / facts tier — MEASURED (Q3).** Null-controlled injection battery: value injection is
  answer-SPECIFIC (real 5/12 vs null 0/12) but efficacy moderate (42%, untuned) — real mechanism,
  promotion gated on an efficacy-tuning pass (0bc5d33). Not shipped at 42%.
- **R3 / guardrails — PRODUCTIZED (opt-in), honest.** `clozn_guard` mid-gen detect+correct, per-firing
  receipt, re-steer cap, fail-closed refuse (df01235). Framed PRESENT-TENSE (A1.1's lead-time thesis
  FAILED, median 0 tokens); firing threshold is an uncalibrated placeholder → promotion gated on a
  threshold-calibration pass, same shape as facts.
- **Molecules program — MEASURED NEGATIVE (distributed-function arc complete).** Greedily-optimized
  position coalitions do NOT beat matched random coalitions (0.93x); distributed function holds at a
  5th level. Localized structure lives on the input edges (attention severance), not residual sites
  (6efda7a).
- **Doc-drift hygiene + per-model concept-dial calibration** (6745334, cf40711).

**2026-07-23 PHASE 2 addendum ("all of it" — surface + promote + research):**
- **R3 guardrails — CALIBRATED + LIVE-FIRING.** Guard-signal calibration found the shipped defaults
  were 9B-specific (layer 16 invalid on 7B; concept-word signal doesn't surface — use trigger-token
  SETS). Fixed: catch 100% / fp 0% (6+6 battery), artifact written, guard promoted to read it
  (69ce0f1) — live check: violent text fires (12.45≥10.07), clean quiet. Promotion gate cleared;
  small-battery caveat stands.
- **R1/R5 Studio surfacing — DONE.** Plain-language provenance chip (d5e30a7) + POST
  /runs/<id>/causal-trace route (300c8e5) + the on-demand click-a-token causal panel (per-position,
  contrastive, honest verdict/nodes/FAILED_CONTROLS/distributed-function caveat).
- **Distributed-function writeup** (docs/research/DISTRIBUTED_FUNCTION.md, 18c62ab) — internal; its
  receipt audit caught that "41/41" reproduces only under CURRENT grading (stored summaries stale);
  SCOPE_NOTE + coalition.py corrected (8fac4ff, 0cf87bb).
- **R6 facts efficacy — engine path RULED OUT (measured, 0cf87bb).** Engine steering can't inject
  rare stored facts (loses to priors); needs the step-targeted torch mechanism. LAB-GATED.
- **LAB-GATED / not doable this environment (CPU-only torch, external jlens-fit pipeline):** 4.3
  jlens artifact rebuild + SAE import; facts full efficacy tuning. Deferred follow-up: refresh the
  stale provenance_battery summary blocks by re-running under --no-flash-attn.

**2026-07-28 STALENESS WARNING — read before trusting the table below.** This file was last edited
2026-07-24 (`4132323`). As of 2026-07-28, `main` is **135 commits ahead** of that edit and the
delivery-status table has NOT been re-audited against them. Treat every DONE / IN PROGRESS row as
accurate *as of 2026-07-24*, not as current. Two drifts are confirmed by direct code reading rather
than inference, and are named here as representative, not exhaustive:

- **§7 item 2 (versioned hook/intervention contracts)** is marked IN PROGRESS in the table below.
  Since then `clozn.hook_vocabulary.v1` and `clozn.intervention_manifest.v1` have SHIPPED
  (`clozn/receipts/hook_vocabulary.py`, `clozn/receipts/intervention_manifest.py`), the former served
  verbatim at `GET /contracts/hooks`, the latter hash-compatible with clozn-client's own manifest
  builder. Scope limit worth recording: the manifest's arms are `attention_knockout` / `steer` /
  `steer_vec` only — **residual writes are not covered by it**, so any feature needing multi-write
  replay extends it as `.v2` (schemas are immutable once released; see `clozn/schemas/__init__.py`).
- **The Studio rewrite is undocumented here.** `studio-frontend/` (React 19 + Vite + TypeScript,
  built into `studio/next/`, served at `/`) does not appear anywhere in this file.

Re-auditing the full table is its own task and has not been done. Do not cite a row below as current
status without checking it against `main` first.

Status meanings: **DONE** means the acceptance language in this roadmap is implemented and tested;
**IN PROGRESS** means a useful slice is shipped but named acceptance work remains. Items not listed
below are still queued/not started. Commit IDs are included so this snapshot can be audited against
`main` rather than inferred from filenames or older planning notes.

| Roadmap item | Status | Evidence and remaining work |
|---|---|---|
| Gate 0.1 — one instrumented request path | **DONE** | OpenAI Chat, legacy text completions, Ollama chat/generate, and `clozn run` share the instrumented substrate and finalize coherent journal runs with the exact delivered prompt. CLI turns retain the readable user message separately from the rendered engine prompt (`c56e320`, `fd4f68e`, `fc7e28d`; `tests/test_gate0_request_paths.py`). |
| Gate 0.2 — no silent field ignoring | **DONE** for the current OpenAI/Ollama shims | Central OpenAI validation and Ollama explicit-or-rejected field policy are tested and documented (`fd4f68e`). Unsupported behavior-bearing values now receive named 400s; accepted neutral values are documented. |
| Gate 0.4 — artifact-qualified white-box features | **DONE** | `clozn qualify-whitebox` is the model/artifact capability gate; unqualified or mismatched artifacts fail closed. |
| Phase 1.1 — `clozn diff-model` | **DONE** | Command, same-tokenizer preflight, template policy, paired token receipts, and heuristic verdict shipped (`0ee66f2`). A real Qwen2.5-0.5B-Instruct → Reasoning-0.5b SFT run verified 8/8 ladders in both directions and produced the worked case study (`5d6439f`); the live run also exposed and fixed a capped-detail denominator bug (`1971fe5`). |
| Phase 1.2 — Experiment object v0 | **DONE** | `clozn experiment run/show` executes target + guard cases × base/tuned/quant/prompt/dial variants × seeds, retains instrumented run evidence, and supports per-cell drill-down (`64d0f20`). |
| Phase 1.3 — reproduction receipt | **DONE** | Runs and exported receipt bundles carry model SHA-256, tokenizer/template rendering fingerprint, sampler/seed metadata, engine build when exposed, and Clozn version; CLI runs use the same identity producer (`0d62101`, `04af391`). Missing upstream identity remains visibly omitted rather than fabricated. |
| Phase 1.4 — headless CI gate | **DONE** | `clozn ci baseline/check` has deterministic exit codes, budgets, identity policy, and JSON reports (`64d5c8e`). `clozn ci check --experiment` validates a complete Phase-1.2 artifact, recomputes paired target/guard changes from raw cells, applies per-candidate budgets, and can require stable model identity (`fc7e28d`). |
| Phase 1.5 — deployment equivalence | **IN PROGRESS; local acceptance MET** | `clozn validate-export` now performs the LoRA closeout's three-arm base/base-plus-adapter/merged check, fails on static or effective identity drift, gates teacher-forced deltas, and writes `clozn.adapter-export-receipt.v1`. Model-free known-good and deliberately bad merges are covered; a live known merged GGUF qualification remains required. |
| Phase 1.6 — positioning collateral | **DONE** | README now leads with Model CI + an inspectable no-switch runtime and links the real Qwen reasoning-SFT case study (`5d6439f`). The worked experiment found one target gain and one structured-output guard regression; the strict identity-qualified CI policy rejected it. |
| Phase 2.1 — Ollama NDJSON streaming | **DONE** | Default-stream semantics, NDJSON framing, cancellation, finish reasons, and one instrumented final run are implemented and tested (`fd4f68e`). |
| Phase 2.2 — honest Ollama fields/tags | **DONE** | Unsupported top-level/options fields are rejected, supported sampler options are forwarded, and `/api/tags` uses the real digest or omits it (`fd4f68e`). |
| Phase 2.3 — legacy completions + CLI journal unification | **DONE** | Legacy streaming/non-streaming completions use the shared instrumented substrate and capture memory, dials, trace, raw and rendered prompts, decode metadata, finish/error state, and one journal run. CLI journals keep the user message plus the exact rendered prompt and immutable identity (`04af391`, `fc7e28d`; `tests/test_gate0_request_paths.py`). |
| Phase 2.4 — truncation/context receipts | **DONE** | Every new run carries `clozn.context_receipt.v1`: gateway-delivered messages remain distinct from the assembled messages/exact rendered prompt that survived into generation. OpenAI/Ollama bodies and terminal stream frames emit structured `output_truncated` warnings on a proven `length` stop; non-stream responses also carry `X-Clozn-Warning`. Replay children retain their own post-change prompt and show a loud cutoff alert. `clozn context last [--json]` reads the latest organic receipt (`tests/test_context_receipt.py`). Overlong inputs remain rejected, never silently described as truncated. |
| Phase 2.5 — think-tag hygiene | **DONE** | A shared batch/stream policy removes model-emitted `<think>` blocks (including prompt-prefilled and unclosed blocks) from OpenAI content, CLI/Studio history, replay/branch inputs, and the public token timeline. The local journal retains `clozn.reasoning_trace.v1` blocks plus separated reasoning token evidence; Replay exposes them only in a collapsed evidence drawer. Ollama places captured reasoning in its separate `thinking` field (`tests/test_think_tags.py` and protocol integration tests). |
| Phase 2.6 — stable run-ID side-channel | **DONE** | OpenAI SSE, legacy completion SSE, and Ollama NDJSON terminal frames expose the finalized run ID; opt-in client/session headers support privacy-preserving exact lookup; `/runs/latest`, insertion-ordered `/runs/watch`, `clozn watch`, and Studio exact-run adoption close concurrent-client races. |
| Phase 2.7 — real-client conformance | **IN PROGRESS** | Pinned released clients now have executable lanes: OpenAI Python 2.46.0, Ollama Python 0.6.2, Ollama JS 0.6.3, and Aider 0.86.2. SDK discovery, non-stream/stream, cancellation, stable IDs, journaling, and typed unsupported-field cases are covered. `docs/CLIENT_CONFORMANCE.md` publishes the honest matrix. Open WebUI 0.10.2 has a pinned scheduled provider-path lane, but remains pending until that external lane runs successfully; its full native-tool loop remains unqualified. |
| Phase 2.8 — tools/function calls + structured output | **IN PROGRESS; local acceptance MET** | OpenAI Chat Completions has a fail-closed native slice: up to 32 strict function definitions with `auto`/`none` and at most one returned call, assistant tool-call + matching tool-result continuation, buffered validated SSE deltas, `json_object`, and restricted strict `json_schema`. The private AR worker now atomically renders with llama-common, enforces the emitted grammar during sampling, and parses with llama-common; the public gateway independently validates the parsed message. Qualification registry v2 binds the exact active `model_sha256`, `template_fingerprint`, native pipeline IDs, schema subset, and passing evidence. Model-free C++/Python tests and the pinned OpenAI SDK exercise the path, typed failures, and atomic `output_contract` evidence. A live CPU smoke on exact Qwen2.5-0.5B Q4_K_M passed tool call, tool-result continuation, `json_object`, and strict `json_schema`; this is not yet an installed qualification artifact. The scheduled Open WebUI lane now carries a deterministic two-request tool proxy probe, but its released-client job and a qualified real-model gateway pass remain open. |
| Phase 3.3 — memory receipts | **DONE** | Run records capture selected/injected/omitted card evidence and the causal receipt backend exists. `memory used last`, token-cost/scoping UX, and Markdown card import/export remain open. |
| Phase 3.5 — trust/privacy plumbing | **DONE** | ef24e85 shipped verify-offline/ledger/redact-delete/OTel; f174b31 adds the loopback-bind check with by-design outbound disclosure, child-aware cascade delete (typed refusal), literal-scoped redaction (shared blobs untouched, stated), persisted age retention applied by GC, and the OTel field-mapping table. |
| Phase 3.6 — calibrated ask/abstain | **DONE** | 9c02ce8 shipped the wizard/profile-store/live wiring; d11baa7 adds `clozn eval policy show` (live-model detection, fails closed), the token-probability + hard-tail honesty notes on every calibration surface, and the end-to-end test proving the wizard's own fit drives a live reply. |
| Phase 3.7 / R1 — context ↔ answer influence map + provenance | **IN PROGRESS; automatic map and product labels remain open** | Arbitrary context-span receipts, exact-answer teacher-forced token deltas, matched filler controls, attention knockout, and a self-contained HTML receipt renderer already exist. A 30-case/9-category battery, Qwen + Llama second-family validation, focus-span dependence, trimmed controls, and a null rank test also shipped (`985c961`, `04af391`, `b5a089b`). Automatic source-aware segmentation, baseline-reusing multi-span scoring, linked highlighting in Studio and the saved HTML receipt, persistence/export of the versioned influence matrix, the attention-vs-causal head-to-head, and the final provenance-label gate remain open. |
| Phase 4.2 — hook/intervention contracts | **IN PROGRESS** | Engine capture/write seams, checkpoints, batched branching, multi-observer readouts, and attention knockout exist. A stable public hook vocabulary plus a versioned replayable intervention manifest remain open. |
| R5 — tracer credibility/granularity | **IN PROGRESS** | S0–S4 causal tracing, controls, attention knockout, and the location-level CLI exist. The R5 second-family battery, reliable `FAILED_CONTROLS` exercise, and head-level node units remain open. |
| Phase 3.7 / R1 — context ↔ answer influence map + provenance | **DONE except the R1-gated tier** | Coarse-to-fine refinement, strongest-redundant-pair joint check (interaction nats, never percentages), blob persistence + GC/redact/delete integration, GET export, and both linked-highlighting surfaces shipped (5cf2cf0 + 2d844f1). R1 measurements complete: 41/41 two-family provenance battery, focus-span + trimmed null, attention-vs-causal head-to-head (mean Spearman 0.218, top-1 3/8 — 2d685a2). The internally-confirmed badge + summary chip remain gated on the R1 label-language decision. |
| Phase 4.1 — pip client | **DONE** | clozn-client 0.11.0 committed (26b564a): typed engine/gateway clients, zero mandatory deps, manifests, 42 tests; worked examples for patch sweep, knockout scan, provenance. |
| Phase 4.2 — hook/intervention contracts | **DONE (server v1)** | GET /contracts/hooks serves clozn.hook_vocabulary.v1 (code-derived semantics, UNSPECIFIED where unverified); POST /contracts/replay executes client manifests with typed capability refusals; sha256 cross-checked byte-identical with clozn-client. Engine-side C++ hook additions remain future breadth. |
| Phase 4.4 — statistical rigor | **DONE** | Paired bootstrap CIs (cases, not cells), comparison-count honesty with Bonferroni CIs beside raw (no badges), replay-class labels, opt-in primary_metric preserving old manifest hashes; `clozn experiment stats` (d11baa7). |
| Phase 4.5 — open export | **DONE** | `clozn runs export-bundle`: hash-verified manifest, tensors (.npz or documented raw f32), generated nbformat-v4 reproduction notebook with ONE clearly-optional live cell; honesty block separates offline-proven from live claims (dbd596d, live-proven on a real run). |
| R5 — tracer credibility | **battery + STOP-check DONE; head units open** | Any-GGUF ablation screen (9bc6cd6) + sidecar-vs-engine qualification (22fe9c4). Llama-3.1-8B battery: 16/16, S4 91.1% (216/237) — cross-family replication of 91.7%; FAILED_CONTROLS fired on a real prompt for the first time; interaction gap −82% (worse than Qwen's −60%). 9B regression re-run post-refactor: 91.7% bit-for-bit. Head-level node units remain (C++). |
| Coalition/Shapley credit | **DONE (sequential; batching blocked on C++)** | Opt-in coalition reports: exact Shapley N≤4, Shapley-Taylor + bootstrap CIs N≥5, interaction-gap caveat printed; new `clozn prove` CLI verb. Engine finding: /v1/branch shares one SampleConfig across branches — heterogeneous-arm batching needs per-branch steering in C++; the seam ships with bit-exact cross-check / explicit approximate opt-in. |
| Engine debt tail | **CLOSED (built or measured-blocked)** | Batched multi-arm /score shipped APPROXIMATE with the label on-wire, consumed only by the tracer's nomination screen (fc51f23). KV-blob fast restore bit-exact 3 ways (bad111e). Sampled resume bit-exact via sampler provenance + shape-true rebuild -- the batch-shape landmine's 3rd documented firing (26ce1e4). Steering-in-checkpoints + checkpoint_on_finish live-KV save: 6/6 exact across steered/sampled/both regimes (2109030). Per-branch GENERATION steering: measured-blocked -- per-branch cvec is structurally impossible and per-row batching is provably non-bit-exact (twice), while batched coalition credit requires bit-exactness by its own cross-check; the seam stays sequential, honestly. Head units: engine primitives shipped (5ae7174); the scientific unit measured NEGATIVE two-family (fa6e259) -- tracer does not grow head nodes. | Checkpoints, batched branch execution, and the readout plane shipped. Coalition/Shapley causal credit, KV-blob fast restore, and sampler/RNG + intervention checkpoint state remain open. |

---

## 0. Positioning

> **Clozn makes local-model behavior inspectable, correctable, and reproducible — inside the
> tools people already use.**

| Persona | Promise | The wedge, per the market research |
|---|---|---|
| Model developers | "What improved, what broke, and can I prove it?" | **Model CI** — their #1 documented need (regression/forgetting detection) has *no incumbent*, and clozn's primitives already cover most of it |
| Model users | "What happened, and what should I try?" | The trustworthy runtime **under** their current apps, at the exact moment the Ollama trust crisis has them shopping |
| Model researchers | "What mechanism does the evidence support, under what controls?" | Causal interventions over HTTP on any GGUF, on the GPU they own — a verifiably unclaimed niche (category-creation bet) |

Both audits independently concluded: the gap is **not capabilities** — it is product composition,
compatibility, and discoverability. Do not lead with "AI MRI"; it is the deepest drill-down, not
the front door.

## 1. What the market research changed (decisions, not vibes)

1. **Uncertainty display is demoted everywhere.** No named demand in any persona, and our own gate
   result proved the deployed signal is bit-identical to black-box logprobs. Chips stay as
   texture; never the pitch. (Demotes BACKLOG #18 ambient channel-3 — see §8.)
2. **Model CI is promoted to the lead wedge.** "Evaluation is vibes" is a *named* failure mode;
   silent no-op LoRAs and undetected forgetting are the top developer pains; nobody owns the gate.
3. **The J-lens runs a published method, not our invention** — a Jacobian lens (Neuronpedia shipped
   one 2026-07-17, verified; the technique traces to Anthropic's own interpretability work). What we
   can claim is *where it runs* (inside the llama.cpp serving path, <10% overhead, on the quantized
   model you actually deploy) — not the method, and not that quantized transfer is novel; that
   fidelity is measured on one model family so far, unverified beyond it.
4. **Steering-vector sharing is a bet, not a plan** — strong adoption-gap evidence, weak voiced
   demand. Test cheaply (§8 R4) before building any UI.
5. **The GGUF-interp niche is real but inferred** — two serious teams built "interp bolted onto a
   fast inference engine" on vLLM this year; nobody has claimed the consumer/GGUF substrate. Treat
   as credibility moat, not revenue.

## 2. Sequencing rationale (where the two audits disagreed)

Audit B ordered: compatibility → receipts/repair → Model CI → researcher lab, on the logic that
compatibility is the adoption contract. The market research ranked Model CI the strongest wedge.
Both are right; the resolution is a dependency observation: **Model CI v1 does not need the
compatibility chain** — `test-model`/`quant-check`/model-diff run on GGUFs and fixture suites with
no client attached — while Model CI v2's killer feature (promote real captured runs into
regression suites) *does* need real apps flowing through clozn. So:

- **Phase 1 = Model CI v1**: cheapest path to a public, differentiated, receipts-backed story.
- **Phase 2 = the no-switch runtime**: streaming + conformance; the adoption contract.
- **Phase 3 = daily-trust loop**: receipts/repair during normal use, and Model CI v2 where the
  two tracks join.
- **Phase 4 = the qualified researcher lab**: smallest audience, deepest moat, feeds credibility
  to everything above.

Research lanes (§8) run alongside, GPU-serialized. Phases are sequential in *focus*, not strictly
in time — an agent can carry a Phase-2 M-item while Phase 1 integrates.

## 3. Gate 0 — the standing product contract (binding on all phases)

1. Every accepted request runs on **one instrumented path** and gets a stable run ID + receipt.
   **Status: DONE (2026-07-20).** OpenAI Chat, legacy OpenAI Completions, Ollama chat/generate,
   and CLI turns now converge on the instrumented substrate/journal contract. Streaming protocols
   persist a run but still need the Phase-2.6 lookup side channel to expose its ID after headers commit.
2. **No silent field-ignoring.** Unsupported behavior-bearing fields are rejected with typed
   errors (the knockout-vs-flash-attn refusal is the house pattern). The Ollama shim's currently
   ignored fields must become explicit before any compatibility claim.
3. **Label vocabulary is fixed**: delivered / survived / influenced / supported / probable /
   calibrated are distinct and never conflated in UI or copy.
4. **White-box features are artifact-qualified** — never implied by model family. `qualify-whitebox`
   is the gate.
5. **Every steering action is reversible** and records a before/after comparison.
6. **Research visuals carry method, artifact, controls, and limits** beside the picture.
7. **Banned claims** (measured, not stylistic): uncertainty as a white-box advantage; "explains
   WHY in words" (24% legibility); model self-report as ground truth (X1/X3); "silent influence"
   badges (filler-null); circuit discovery (it's causal tracing over locations); perf promises
   built on KV-prefix reuse.
8. **Receipt integrity outranks everything.** One overclaiming receipt burns the trust the whole
   product is about. The killed-ideas list (§11) is part of this contract.

## 4. Phase 1 — Model CI v1 (the developer wedge)

*Public story when it lands: "CI for your fine-tune — did it break something? Here's the
per-token receipt." Shippable as an HF/HN post backed by a real LoRA case study.*

1. **`clozn diff-model` (base vs fine-tune/merge receipts)** — generalize the quant-check
   machinery: teacher-force the SAME answers under both models, per-token diff, honesty-labeled.
   Same-tokenizer constraint stated plainly. **S–M.**
   *Why:* catches the two named disasters — silent no-op LoRA (diff ≈ 0 when it shouldn't be) and
   forgetting (diff ≫ 0 where it shouldn't be). *Payoff:* the wedge feature, from shipped code.
   *Gate:* validate on one real LoRA pair before the public story (needs a download + GPU smoke).
   **Status: DONE (2026-07-20).** The Qwen2.5-0.5B-Instruct → Reasoning-0.5b SFT case passed
   tokenizer/template preflight, verified both eight-run directions, and was followed by a real
   target/guard experiment. See `docs/MODEL_DIFF_CASE_STUDY_QWEN_REASONING.md`.
2. **Experiment object v0** — one versioned manifest: named cases × variants (base / tuned /
   quant / prompt / dial) × seeds, target suite + guard suite, per-case drill-down; subsumes the
   currently separate `test` / `eval` / `test-model` / `quant-check` outputs. **M.**
   **Status: DONE (2026-07-20).** Shipped as `clozn experiment run/show`; result artifacts retain each
   instrumented run and its immutable identity. Multi-model variants use explicit gateway URLs.
   *Why:* audit B's core diagnosis — primitives exist, the composition doesn't. *Payoff:* one
   command answers "what improved, what regressed" with paired evidence.
3. **Reproduction receipt completion** — immutable identity on every run: model SHA-256,
   tokenizer/template hash, sampling, seeds, build; bundle export. **M.**
   *Why:* HF's own analysis: 96.5% of 50k+ eval records lack minimal reproduction fields.
   *Payoff:* "reproducible" becomes checkable, and it's the substrate for #4 and later HF export.
4. **Headless CI gate** — `clozn ci check <experiment>`: deterministic exit code, allowed-delta
   budgets, baseline artifact, machine-readable report. **M.**
   **Status: DONE (2026-07-20).** `clozn ci check --experiment RESULT.json` validates artifact
   integrity and complete case × variant × seed coverage, recomputes paired changes from cells,
   and gates target gains/regressions, guard regressions, execution errors, and optional identity.
   *Why:* "CI" isn't CI until a pipeline can fail on it. *Payoff:* GitHub-Actions-ready gate.
5. **Deployment-equivalence check v0** — template/tokenizer/BOS-EOS/vocab + known-answer diff
   across an HF-trainer export → GGUF. **M.**
   *Why:* garbled-GGUF export bugs are a documented, recurring developer disaster. *Payoff:* the
   trainer-to-runtime gap gets a gate; also fixes our own CLI-vs-gateway template divergence.
6. **Positioning collateral** — README/story refresh for the wedge, one worked case study. **S.**
   (Docs polish stays cycle-end per standing preference, but the wedge story is product, not polish.)
   **Status: DONE (2026-07-20).** README leads with Model CI, and the worked case demonstrates a
   genuine target gain, a structured-output guard regression, and an identity-qualified CI rejection.

Deferred within this wedge: adapter hot-swap in the C++ engine (**L**, after the loop is
coherent); LightEval/Inspect/Promptfoo adapters + HF Community-Evals/EEE export (**M**, Phase 3);
merge-recipe/registry anything (non-goal §10).

## 5. Phase 2 — the no-switch runtime (persona-1 adoption contract)

*Public story: "Point Open WebUI (or your Ollama app) at clozn. Everything works — and every run
becomes inspectable." No broad compatibility claim until the conformance matrix is green.*

1. **Ollama NDJSON streaming** through the instrumented path, with correct default-stream
   semantics, finish reasons, cancellation, and single coherent run finalization. **M.**
   *Why:* release-blocker; clients stream by default. *Payoff:* the drop-in story becomes true.
2. **Explicit-or-rejected shim fields** — stop silently ignoring `raw`/`format`/`keep_alive`/
   `options`/`think`/etc.; honest `/api/tags` metadata (no placeholder digests). **S–M.** (Gate-0.)
3. **Instrument or retire legacy `/v1/completions`**; unify `clozn run` onto the rendered-template
   journal record. **S–M.** (Gate-0 violations, found by audit B.)
   **Status: DONE (2026-07-20).** Both completion modes use the shared substrate and create one
   honest run record; CLI stores the raw message and exact rendered engine input separately.
4. **Truncation + context receipts** — loud warning on context capping/truncation in API + Replay;
   `clozn context last` with delivered/survived sections. **S–M.**
   *Why:* silent context mishandling is persona-1's top named pain; we already record the truth.
   **Status: DONE (2026-07-20).** The journal persists delivered/survived evidence and worker-reported
   context counts when available; API/stream and Replay surfaces warn on output cutoffs without
   mislabeling them as prompt truncation, and `clozn context last` renders the receipt locally.
5. **Think-tag hygiene** — strip/manage per client so think-blocks never corrupt history or tool
   parsing; journal the stripped reasoning as inspectable trace material. **S–M.**
   **Status: DONE (2026-07-20).** OpenAI, legacy completions, CLI, Studio, replay/branch, and stateful
   lab history now consume only the public answer. Ollama carries reasoning separately as `thinking`;
   the journal and Replay retain it as explicitly labeled, non-privileged evidence.
6. **Stable run-ID side-channel** — `X-Clozn-Run-Id` header + latest-by-client/session lookup +
   `clozn watch`. **S–M.**
   *Why:* third-party clients drop custom body fields; the sidecar needs a reliable hook.
   **Status: DONE (2026-07-20).** Non-stream replies carry `X-Clozn-Run-Id` and body IDs; OpenAI SSE,
   legacy completion SSE, and Ollama NDJSON carry the ID on their ordinary terminal frame. Callers can
   opt into exact cross-protocol correlation with `X-Clozn-Client-Id` / `X-Clozn-Session-Id`; only
   install-local HMAC fingerprints are journaled, and portable receipts omit them. `/runs/latest`,
   cursor-based `/runs/watch`, `clozn watch`, and Studio use journal insertion order so overlapping slow
   requests cannot be mistaken for the newest run.
7. **Real-client conformance matrix** — Ollama Python/JS SDKs, OpenAI SDK, Open WebUI, one coding
   agent; streaming/cancel/tools cases; published as a compatibility report. **M, recurring.**
   **Status: IN PROGRESS (2026-07-20).** Official OpenAI Python, Ollama Python/JS, and the released
   Aider CLI are pinned and executed model-free against the real gateway in CI. The Ollama Python client
   also closes a live stream and proves a partial cancellation run is journaled. Open WebUI has a pinned
   weekly/manual released-client lane covering model discovery and proxied non-stream/stream chat, but it
   is not marked green before that external workflow succeeds. The Phase 2.8 gateway contract is exercised
   through the OpenAI SDK, but Open WebUI's complete two-request native-tool loop remains unqualified.
8. **Tools/function calls + structured output** for a deliberately small qualified model set —
   parser/renderer qualification per model, malformed-output recovery explicit. **L.**
   *Why:* agent clients are the growth segment; tool failures are routinely misblamed on models.
   **Status: IN PROGRESS (2026-07-21).** A fail-closed OpenAI Chat Completions slice now supports up to
   32 strict function definitions (`auto` or text-bypass `none`) with at most one returned call,
   assistant tool-call/tool-result continuation, buffer-then-validate SSE, `json_object`, and a bounded
   strict `json_schema`. The private AR worker keeps one prepared descriptor across llama-common template
   rendering, grammar-constrained generation, and llama-common parsing, so a client cannot substitute stale
   or modified parser/grammar state between those stages. The public gateway uses that atomic path only after
   qualification registry v2 matches the active model SHA-256, template fingerprint, exact native worker
   pipeline, schema subset, and passing evidence; it then strictly validates the native message and records
   raw output, native parser result/error, validator result, contract, qualification, and outcome in one
   journal run. The request model label cannot qualify the worker, and no real model is prequalified.
   Model-free native/gateway tests are green. A manual CPU smoke on
   `qwen2.5-0.5b-instruct-q4_k_m.gguf` (SHA-256
   `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`, template fingerprint
   `b96c223e2aa0e18a`) passed a native tool call, matching tool-result continuation, `json_object`, and
   strict `json_schema`; Llama-3.2-1B failed closed because its template emitted no structured grammar.
   The scheduled Open WebUI 0.10.2 lane now includes the complete caller-managed two-request tool proxy
   sequence. The installed qualification artifact exists for the exact 0.5B (runner refuses overwrite), and the
   caller-managed two-request tool loop + strict json_schema now pass 10/10 through the qualified
   real-model public gateway on the kv_unified build (94668a9 -- the ~20% llama_decode failures were a
   silent 256-token per-sequence KV cap, armed since 2.2). Remaining: the scheduled Open WebUI
   released-client lane (external).
9. **`clozn connect <app>`** setup helper with config backup. **M.**
10. **Real-browser pass over Studio** (BACKLOG #2, still open) as the phase quality gate. **S.**

Explicitly later: Anthropic-Messages compat, vision, embeddings sidecar, broad lifecycle API —
sequencing deferrals, not non-goals.

## 6. Phase 3 — the daily-trust loop (receipts, repair, and Model CI v2)

*Public story: "The runtime that shows you what your model actually saw — and lets you fix it."*

**Phase-3 priority override (2026-07-21):** after the current trust-plumbing checkpoint (#5), ship
the #7 context ↔ answer influence-map MVP before the remaining #6 wizard and #8/#9 composition work.
The broadly available forced-score tier does not wait on R1; R1 gates the stronger internally-confirmed
labels and the provenance summary chip.

1. **Corrective retries** — `retry last --less-verbose / --more-concrete / --use-context /
   --ask-before-guessing`; prompt/sampling interventions first, dial-backed only where qualified;
   scope once/session/profile; compare + undo mandatory. **M.**
2. **Why-slow / why-cut-off diagnosis** — plain-language decomposition (load, prefill, generation,
   context/KV allocation, CPU spill, client auxiliary calls). **M.**
3. **Memory receipts** — `memory used last` (selected/injected/omitted + token cost), app/project
   scoping, and **markdown import/export of cards** (meet the folk 12KB-memory-file practice). **S–M.**
4. **Model CI v2: run promotion** — `suite create --from-runs` with redact/edit/freeze; captured
   real-app runs become regression cases. **M.** *(The join point of the two tracks.)*
5. **Trust plumbing** — local-only flag + outbound ledger + `doctor --verify-offline`; run-journal
   redact/retention/delete UX; OTel/OpenInference export with prompt-privacy defaults. **M.**
6. **Calibrated ask/abstain** — ship the selective-generation last-mile honestly labeled as
   token-probability-based (BACKLOG #10 path (a)); per-model/task calibration wizard; band
   limitations printed. **M.**
7. **Context ↔ answer influence map + provenance summary** — make the exact survived prompt and the
   recorded answer one linked evidence surface. Hover or focus an answer word/clause and the context
   spans that most influenced it light up; hover a prompt/document/memory span and the answer spans most
   dependent on it light up in return. Click pins the selection for touch, keyboard use, and inspection.
   The same signed influence matrix drives both directions; it never invents percentages or implies that
   interacting/redundant causes add to 100%. Ship the interaction in Studio and in the existing
   self-contained, offline HTML receipt so the evidence remains explorable after the model and gateway
   stop. The HTML is only a rendering of stored evidence and should preserve the receipt renderer's
   no-network, injection-safe contract. **M; Phase-3 headline surface.**

   The fast broadly available tier teacher-forces the run's exact answer and reuses one baseline while
   ablating/replacing source-aware context spans; it reports per-answer-token signed deltas against
   matched controls, automatically works coarse-to-fine, checks the strongest redundant pairs, and says
   "no clear source" when nothing clears the floor. Persist the complete versioned
   `clozn.context_answer_influence.v1` evidence object with prompt/answer span boundaries, method,
   model/template identity, raw deltas, controls, thresholds, timing, and artifact hash. The interactive
   receipt may render a sparse top-link view for size, but the portable evidence must retain the complete
   measured matrix.

   A stricter **internally confirmed** tier uses qualified attention/residual interventions to show that
   preventing answer positions from accessing a context region also reduced support, without editing the
   visible prompt. That badge and the CONTEXT_CARRIED / MIXED / PARAMETRIC summary remain **R → M**, gated
   on lane R1 (attention-vs-causal head-to-head plus the no-flash-attn mode/performance story). Never call
   either tier a circuit explanation. *Performance gate:* benchmark an 8-source coarse map over a normal
   recorded answer on the reference 7B/GPU; first useful links must feel interactive and the receipt must
   print its measured latency. If the controlled map cannot complete within a documented low-single-digit
   second budget, stream progressive coarse results rather than making a blanket "instant" claim.

   *Why:* this is the one surface that makes Clozn's value legible to all three personas at once: users
   see which supplied context supported which words, developers get a debuggable prompt/RAG receipt, and
   researchers can drill from a behavioral link into qualified internal mediation evidence. The summary
   chip is subordinate to this map, not the product by itself.
8. **Studio IA: three home views** over one object model (run / experiment / evidence), per
   audit B §6 — Replay is the run view; the experiment matrix becomes the developer home; evidence
   view carries method/control labels. Console re-skin (`notes/CLOZN_UX.md`) folds in here only if
   pursued. **L.**
9. **HF Community-Evals/EEE export + eval-tool adapters** (from Phase 1 deferral). **M.**

## 7. Phase 4 — the qualified researcher lab (credibility moat)

*Public story: "Causal experiments at llama.cpp speed, on the GPU you own — pip install, three
worked examples, an honest benchmark." Smallest audience; shapes architecture and credibility,
not revenue.*

1. **`pip install clozn-client`** + three worked notebook examples (patch sweep, knockout scan,
   provenance) + an honest speed benchmark vs TransformerLens on the same experiments. **M.**
   *Why:* researchers script; the HTTP API isn't real to them without this. The field's loudest
   complaint is patching speed — we should measure ours in their terms, honestly.
2. **Versioned hook/capture + intervention contracts** — named hook vocabulary with exact
   semantics (pre/post norm, pre/post residual add, position, step), capture budgets + stats,
   a serialized intervention manifest replayable across an experiment. **L.**
3. **One rigorous qualified path before any breadth** — Qwen3.5-9B reference checkpoint + the
   shipped GGUF: quantized-vs-reference activation qualification; import + validate one external
   SAE (audit B reports a Qwen-Scope Qwen3.5-9B artifact — verify) and a pre-fitted J-lens;
   publish our J-fitting recipe so researchers can produce sidecars. **L.**
   *Why:* "can you trust interp on a quantized model?" is the objection that decides this persona
   — and quant receipts make the objection itself a use case.
4. **Statistical rigor + evidence labels as product** — replicates/CIs/controls in one experiment
   report; replay honesty labels (bit-identical vs re-prefilled vs stochastic); every panel names
   method/artifact/controls. **M.**
5. **Open export** — manifest + tensor bundles + a generated notebook that reproduces a receipt. **M.**

**Added 2026-07-28 (BK-directed).** Three features, to be built in this order because 6 supplies the
action substrate for 8, and 7 supplies its deepest comparison evidence. **Sequencing rule, binding:**
do not build the polished workbench (8) before 6 and 7 emit versioned artifacts — otherwise the
frontend hard-codes another generation of temporary API shapes. This restates roadmap rule 7
(schema-first) for a case where the temptation to invert it is strong.

6. **Exact execution forks** — restore a KV checkpoint, truncate it to an arbitrary token position,
   change one declared variable, continue. **M.**
   *Claim boundary, explicit:* this is an **exactness** claim, NOT a performance claim. §11 killed
   "KV-prefix-reuse perf promises" and that kill stands — nothing here may be framed as a speed
   feature. The claim is: *continuing unchanged from a fork point reproduces the original suffix
   token-for-token.* Exactness is **regime-scoped from the start**: truncating above the prompt
   boundary re-decodes a generated token at batch shape 1, matching its original decode, and is
   bit-exact; forking at the prompt/generation boundary re-prefills instead, because the last prompt
   token was originally decoded inside the full prefill batch. That is the batch-shape landmine,
   which has already fired three times in this repo (see the Engine debt tail row above) — it is
   designed around here, not discovered by testing. The `reprefill` path must never report itself as
   `live_kv` exactness.
   *Substrate already shipped:* `GgmlAdapter::evict_from` (KV truncation), checkpoint save/restore,
   sampler provenance (re-seed + `rng_discard`, not a serialized RNG), steering-in-checkpoints.
   *Known defects this work fixes:* `checkpoint_on_finish` with `stream:true` saves a checkpoint and
   never returns its id; `/health` does not advertise checkpoint capability; `/v1/branch` re-prefills
   from `ckpt.tokens` and drops both `has_steer` and `has_sampler`, so batched branching is **not**
   an exact-fork substrate for steered runs today.
7. **Cross-model mechanistic diff and causal bisect** — where two compatible executions diverge
   internally, and which divergences causally matter. **L.** Serves the Model CI wedge (§0) most
   directly of the three: "why did my quant regress" is a model-developer question.
   *Hard v1 scope gate:* same architecture, tokenizer, layer count, and **residual width** — the
   width limit is not conservatism, it is mechanical: `/score` writes require
   `values.size() == positions.size() * n_embd` of the target engine and there is no projection
   layer. Writable layers are `[1, n_layer)`; the final layer cannot be captured whole-sequence.
   *Method warning, load-bearing:* a residual write **overwrites** the row, so transplanting a layer
   *window* `[a..b]` is mathematically identical to transplanting at `b` alone. A coarse-to-fine
   window bisect over the residual stream therefore cannot distinguish localized from distributed —
   it can only find the shallowest layer at which a full overwrite suffices. v1 ships that honestly
   as a **sufficiency frontier** and omits `distributed_restoration` from the verdict vocabulary
   until necessity testing (transplant all-but-L) or composable sub-layer writes (`head_write` at
   `kqv_out-<il>`) exist. The verdict taxonomy otherwise stands, and encodes the transplant study's
   own correction: FP transplant fixed 5/12 quant flips but the random equal-norm control fixed
   3/12, leaving 3/12 genuinely FP-specific.
   *Prerequisite:* `EngineClient.score()` exposes neither `capture` nor `write` nor `arms`; existing
   callers bypass the SDK with raw `urllib`. Batched `arms` is `batched_approximate` (~0.19 nats) —
   screening only, never receipts.
8. **Unified token workbench** — refactor Scope/Observatory so one selected token anchors all
   evidence (`run × output token × optional reference run`). **M.** Least aligned of the three with
   the §0 Model CI wedge; sequence it last on those grounds as well as the dependency ones.
   *Two unlisted prerequisites, both real:* (a) there is **no async job system** anywhere in this
   codebase — `POST /runs/<id>/influence-map` computes synchronously in the request handler; the
   only "cancel" is a client-side `AbortSignal` that drops the connection while the server keeps
   computing. Any progress/cancellation surface is new work, not reuse. (b) `studio-frontend/` has
   **no test framework** — the standing checks are a copy-linter, `tsc -b`, and an SSR smoke render
   that explicitly cannot run effects.
   *Constraint:* the UI's evidence vocabularies are deliberately separate closed unions per artifact
   type, guarded by `never`-exhaustiveness switches so they can never be cross-rendered. Do not
   collapse them into one flat status enum.

Then breadth, in this order: head/source-position tracing (lane R5), TransformerLens/NNsight
bridges, Neuronpedia/HF publish flows. Deprioritized: J-lens fitting inside the fast runtime, SAE
training, native circuit-tracer reimplementation.

## 8. Research lanes (parallel, GPU-serialized; each has a gate and a product hook)

- **R1 — Provenance hardening.** Multi-prompt battery (≥30 across categories), second model
  family (Llama-3.1-8B GGUF already on disk — no lens needed), a genuine screen-null (replace the
  target concept, don't dilute), attention-heatmap-vs-causal-rank head-to-head (~1 day; needs the
  no-flash-attn path that already exists). *Gates:* Phase-3 provenance chip; any RAG-receipt
  marketing. *Also resolves:* FAILED_CONTROLS has never fired on a real prompt.
- **R2 — Model-diff transplants** (BACKLOG #11). Cross-model residual alignment + A→B transplant
  at token+layer. *Hook:* upgrades Model CI from "shows the regression" to "localizes it, proven
  by transplant." **L/R.**
- **R3 — Mid-gen guardrails productization** (A1.1 was INCONCLUSIVE: catch 100% / FP 5% PASSED, but
  the intent-before-speech lead-time thesis FAILED at median 0 tokens — so this is a PRESENT-TENSE
  detect-and-correct guard, never precognitive. Opt-in `clozn_guard` built + tested + honesty-clean
  (df01235); firing threshold uncalibrated, promotion gated on a calibration pass). Receipt per firing,
  re-steer cap, honest copy ("catches and corrects during generation," never "reads intent
  early"). *Hook:* the headline steering feature for developers. **M.**
- **R4 — Steering-pack demand test.** Publish ~3 verified dial packs (contrastive for register,
  dir(c) low-strength topic nudges) with swap receipts + A1.4 portable specs on HF; watch 60–90
  days. *Gate:* any sharing UI. Cost: days, mostly authoring. **S–M.**
- **R5 — Tracer credibility + granularity.** Second-family battery; exercise FAILED_CONTROLS;
  then head-level node units (`kqv_out` is named and materialized at all positions — dodges the
  last-layer `inp_out_ids` blocker). *Hook:* the researcher story graduates from "causal trace"
  toward something deserving "circuit." **L/R.**
- **R6 — Memory that changes the answer.** Two-tier legible memory scale-up (X7, n=6 today) +
  native fast-weight fact memory (BACKLOG #12) with with/without/null receipts. *Hook:* the
  persona-1/2 memory story beyond prompt cards. **L/R.**
- **R7 — AR×diffusion. DORMANT** — re-enter on a chat-quality open dLLM (today's LLaDA/Dream are
  viz-only research substrates, not product-grade); pin-and-resolve editing (Studio's Edit view) is
  the one piece kept visible meanwhile. Also VRAM-gated regardless (co-residency; park until ~10 GB
  frees). H2 ceiling-first with degeneration veto (per the A4 runbook §0b revisions — not this
  file's R1/R2 lanes), H5 counterfactual patches, Route C free-text edit instructions remain
  scoped, paused. **R.**
- **R8 — Cross-family dials.** Dense Llama-3.1-8B J fit (deferred #109, overnight) → A1.4 spec
  ports with floors/ceilings; Fast-J stays scoped to subspace features (dial authoring needs the
  dense J — measured, not a preference). *Hook:* "author a dial once, run it on any qualified
  model." **R.**
- **Engine debt, scheduled opportunistically:** batched causal credit (coalition/Shapley over
  teacher-forced arms — the seam is in `clozn/receipts/core.py`, on top of the `/v1/branch`
  batched-decode primitive); KV-blob fast restore (restore currently re-prefills from saved
  tokens — correct, just slower); sampler/RNG + intervention state in checkpoints.

*Execution constraint (from the retired BACKLOG header): VRAM, not compute, is the live limit —
one 0.5B engine ≈ 2.7 GB fits the current ~3 GB headroom; anything needing Qwen-7B + Dream
co-resident (~13 GB) queues until ~10 GB frees. GPU work serializes; lanes are parallel only in
the CPU/desk portions.*

## 9. Demotions and parked items (explicit changes vs BACKLOG.md)

*(References below: §6.8 = Phase-3 Studio IA item.)*

- **Ambient channel-3** (inline confidence shading in Cursor/ChatGPT web; BACKLOG #18): demoted
  from "endgame" to **parked**. Both audits found no demand for uncertainty display; the honest
  signal isn't white-box; highest effort of the ambient tier. Revisit only behind a run-ID
  sidecar with real users.
- **J5 lens extensions** (Dream lens, chat-vs-web lens, stream top-k): research lane, low.
- **Design-agent mock pack (D1–D5)**: only if the §6.8 Studio IA work is pursued.
- **Killed features stay killed** (§11) — including branch-on-doubt and paraphrase-brittleness
  from BACKLOG #17's "assembled-but-unconnected" list.
- **SAE consumer features**: research surface only (our own study: no sparse load-bearing
  features at product granularity).

## 10. Non-goals (merged from both audits; stable)

No model hosting/registry/marketplace; no training stack (trainers, labeling, synthetic data); no
cloud offering or multi-tenant serving; no general chat/RAG/agent application (we sit *under*
those); no proxying an external Ollama as the primary architecture (deep evidence requires running
the model); no replacing HF/W&B/LightEval/Inspect/TransformerLens/NNsight/SAELens/Neuronpedia —
integrate; no frontier-scale SAE/CLT training or NDIF-like remote GPU fabric; no closed-model
internals; Studio never mandatory for core value; no benchmark leaderboard.

## 11. Killed — do not revive without new evidence

Semantic temperature · prospective collapse gauge · branch-on-doubt · paraphrase-brittleness
receipt · same-model verify-then-branch · null-space watermarking · scalar self-reported
confidence · internal probe as general correctness detector · J-transport as steering-*quality*
(it's authoring/stability infrastructure) · "model authorship" as a verdict (verbatim-only
receipt survives) · white-box uncertainty advantage · silent-influence badges · KV-prefix reuse
perf promises. Full autopsies: `docs/RESEARCH_ROADMAP.md` (Killed + wave verdicts),
`notes/POSITIONING_AUDIT_B_2026-07.md` §1.10/§7.4.

## 12. Reconciliation with docs/BACKLOG.md

| BACKLOG item | Status / new home |
|---|---|
| #2 real-browser Studio pass | open → Phase 2 quality gate (§5.10) |
| #5 H7+H3 captures "blocked on VRAM" | **stale** — ran 2026-07-19 (A4.2 MIXED, A4.3 SPLIT); remaining diffusion work → lane R7 |
| #7 batched causal credit | engine debt (§8 tail) |
| #9 tracer REMAINING list | split: journal input mode + click-a-token → Phase 3/4 UI; screen-null + 2nd family + attention head-to-head → R1/R5; head units → R5 |
| #10 risk controller last-mile | Phase 3.6 (path (a), honest label) |
| #11 model diffing/transplants | lane R2 (v1 wrapper ships in Phase 1.1) |
| #12 fact memory · #13 guardrails | R6 · R3 |
| #14 H2/H5 · #15 Route C | R7 (VRAM-gated) |
| #16 J5 · #17 leftovers | demoted (§9) |
| #18 ambient channel-3 | **demoted/parked** (§9) |
| #20 design mocks | conditional (§9) |
| Parked "Ollama drop-in? NOT registered" | **stale** — shim registered + live (BK's merge c56e320); full contract → Phase 2 |
| Task #44 docs polish | cycle-end, unchanged; Phase-1.6 wedge story is exempt (product, not polish) |
| Task #109 Llama dense J fit | lane R8 |

## 13. Success signals (trimmed to what one person can actually watch)

- **Phase 1: MET LOCALLY (2026-07-20).** The real Qwen reasoning-SFT case produced a `CHANGED`
  model diff, one target gain, one guard regression, and an exit-1 CI rejection with stable identity.
- **Phase 2:** conformance matrix green for Open WebUI + both Ollama SDKs + one coding agent;
  zero silent-field incidents; a stranger's app works by changing one base URL.
- **Phase 3:** time from "odd response" to diagnosis measured in one command; retries kept vs
  undone; run promotion used on real captured traffic; a saved offline HTML receipt lets a person hover
  either side of a real context ↔ answer link and inspect the measured reciprocal highlighting, controls,
  method, and latency without a running model.
- **Phase 4 / lanes:** a pip-client notebook reproduces a receipt end-to-end; R1 battery passes
  (or honestly fails and the chip stays gated); R4 gives a real adoption number for dial packs.
