# Clozn product positioning and user-needs audit

- **Date:** July 20, 2026
- **Repository snapshot:** working branch agent/instrument-ollama-runtime at b6190b8; origin/main at 1e7ccb5
- **Research window:** primarily January–July 2026, with older primary sources used where they still define a current product capability

## Executive recommendation

Clozn should not position itself as another local-model runner, another chat application, or a prettier interpretability dashboard.

It should position itself as:

> **The behavioral observability and control layer for local models. Keep your app; Clozn shows what the model actually received, where a run deserves scrutiny, what changed its behavior, and whether a proposed fix made things better or worse.**

The product should have one shared technical spine and three depths of use:

| Audience | Product promise | Primary surface | What earns adoption |
|---|---|---|---|
| Model users | Keep using the app you already like; understand and repair a bad local-model run | Compatible endpoint and CLI; Studio only on demand | Compatibility, exact context receipts, plain-language diagnosis, quick repair |
| Model developers | Prove whether a tune, adapter, quantization, prompt, or steering change helped without breaking something else | Experiment CLI and Studio comparison matrix | Reproducible paired evaluation, deployment-equivalence checks, shareable evidence |
| Model researchers | Run controlled interventions against a real local inference runtime and inspect the resulting state | Lab API/CLI, notebooks, data exports, MRI as a navigator | Scientific controls, reproducibility, extensibility, honest limits |

This ordering should also drive implementation. Mainstream compatibility and low-friction receipts are prerequisites. The developer workflow is the most credible near-term differentiated product. The researcher workflow is the deepest moat, but it is the smallest audience and should not dictate the first-run experience.

The key product pattern is progressive disclosure:

~~~text
existing app
    ↓  Ollama / OpenAI / later Anthropic-compatible request
Clozn instrumented runtime
    ├─ clean response back to the existing app
    ├─ local run receipt and stable run ID
    ├─ CLI diagnosis and corrective retry
    ├─ Studio comparison and forensics
    └─ research-grade interventions when explicitly enabled
~~~

The MRI is therefore not the front door. It is the deepest drill-down after a concrete behavioral question has been identified.

## The strategic answer in one page

### Where the market is already strong

Ollama and LM Studio are rapidly broadening the definition of a local runtime. Ollama reported 8.9 million developers in July 2026 and now combines local and cloud models, an application, OpenAI and Anthropic compatibility, model lifecycle commands, tools, structured outputs, multimodality, and one-command configuration for popular coding agents. [Ollama, July 9, 2026](https://ollama.com/blog/all-aboard-open-models), [Ollama integrations](https://docs.ollama.com/integrations), [ollama launch, January 23, 2026](https://ollama.com/blog/launch)

LM Studio spans a desktop app, headless daemon, CLI, model discovery and download, resource estimation, local document chat, native SDKs, OpenAI and Anthropic compatibility, MCP, stateful chat, and model lifecycle management. [LM Studio product surfaces](https://lmstudio.ai/docs/app/basics/lmstudio-vs-llmster-vs-lms), [LM Studio server](https://lmstudio.ai/docs/developer/core/server), [LM Studio developer overview](https://lmstudio.ai/docs/developer)

Open WebUI already offers a broad application layer over local runtimes: chat, memory, notes, RAG/knowledge, tools, web search, model profiles, feedback, and lightweight evaluation. [Open WebUI features](https://docs.openwebui.com/features/), [Open WebUI memory](https://docs.openwebui.com/features/chat-conversations/memory/), [Open WebUI evaluation](https://docs.openwebui.com/features/administration/evaluation/)

Trying to beat those products on catalog breadth, general chat UX, RAG, agent orchestration, or raw serving performance would spread Clozn across mature categories with larger communities.

### Where the gap remains

When a local model behaves badly, users still struggle to determine whether the cause was:

- The model or quantization
- A too-small or unexpectedly large context allocation
- Prompt-template, role, tool-call, or protocol incompatibility
- Conversation history or retrieved context being truncated
- A memory being selected but not used
- Sampling settings or a frontend-added system prompt
- CPU fallback, model reload, prompt processing, or hidden auxiliary calls
- An adapter, fine-tune, steering vector, or export changing behavior

Existing observability products can record prompts, responses, latency, tokens, and application traces. Langfuse, for example, offers OpenTelemetry-based LLM traces and evaluation. Promptfoo offers provider-agnostic assertions, red teaming, repeat runs, tracing, and a web UI. [Langfuse observability](https://langfuse.com/docs/observability/sdk/overview), [Promptfoo configuration](https://www.promptfoo.dev/docs/configuration/reference/)

Those tools generally observe the request boundary. Clozn can combine boundary-level observability with the runtime’s exact rendered prompt, token distribution, replayable state, interventions, and artifact-qualified internal readouts. That combination—not merely “we show activations”—is the defensible product.

### Recommended category

The clearest category is:

> **Local model behavior tools**

Supporting descriptions:

- For model users: **A debugger and repair layer under the local AI apps you already use.**
- For model developers: **Model CI: compare, diagnose, and prove every behavioral change.**
- For researchers: **A controlled intervention runtime for GGUF models.**

Avoid leading with “AI MRI.” It is a memorable metaphor for a view, but it invites stronger scientific claims than the current readouts can support and does not state a user outcome.

## Method and confidence

This report combines:

1. A fresh source audit of the current Clozn repository, including the public CLI, server routes, run journal, receipt and memory systems, steering and artifact contracts, Studio modules, test coverage, causal-tracing scripts, and current research/backlog documents.
2. Recent official product documentation and release material for Ollama, LM Studio, Open WebUI, Hugging Face, PEFT/TRL, LightEval, TransformerLens, NNsight, Neuronpedia, and adjacent tools.
3. Recent GitHub issues and community discussions from approximately the last six months. These are useful evidence of concrete failure modes, but they are directional anecdotes, not prevalence estimates.
4. Clozn’s own measured research findings. Negative results are treated as product decisions rather than buried experiments.

No user interviews, product telemetry, or representative survey were available. The audience sizing and “users want” statements below are desk-research hypotheses to validate, not measured market prevalence.

Effort estimates are relative:

| Label | Meaning |
|---|---|
| S | Several focused engineering days; little architectural uncertainty |
| M | Roughly two to six engineering weeks; multiple components or substantial product work |
| L | Roughly one to three engineering months; architecture, compatibility, or broad UX work |
| XL / research | Unknown until an experiment resolves a scientific or substrate-level blocker |

These are planning bands, not delivery commitments.

# 1. Fresh Clozn capability audit

## 1.1 Runtime and model operations

The current product runtime is a Torch-free Python gateway supervising a private C++ GGUF worker. It supports autoregressive text generation, streaming, CPU/GPU execution, sampling, context selection, one model per gateway, and local model download/inspection. Gateway chat uses the template embedded in the loaded GGUF. The separate clozn run path still hand-renders family-specific Qwen/Mistral/Llama/Gemma templates, so CLI and gateway prompt rendering can diverge; this is an important deployment-equivalence gap.

The serving architecture is intentionally local and single-user today: loopback binding, no remote auth/TLS or multi-tenant control plane, one admitted active generation, and no general continuous batching. The newer batched-branch primitive accelerates controlled counterfactual work; it is not production request batching.

The public CLI currently includes:

- run, serve, models, pull, plan, studio, lab, smoke, ps, stop, doctor
- trace, branch, explain, inspect
- preferences
- test, eval, test-model, quant-check
- qualify-whitebox
- causal-trace
- migrations and run-store garbage collection

The fit planner can inspect a GGUF before download/load and estimate fit and throughput. Automatic empirical calibration remains incomplete.

Core autoregressive support is broader than white-box support. The repository qualification ledger covers Llama 3.1, Qwen 3.5, Gemma 4, Ministral 3, and Qwen 2.5 at the runtime level, while steering, J-lenses, and SAEs remain model- and artifact-specific. This separation is correctly documented in [MODEL_SUPPORT.md](./MODEL_SUPPORT.md) and [RUNTIME_SPLIT.md](./RUNTIME_SPLIT.md).

**Product interpretation:** the runtime is credible enough to be the instrumented execution path, but it is not yet a drop-in replacement for the breadth and performance engineering of Ollama.

## 1.2 Compatibility surface

The documented OpenAI surface is intentionally strict:

- GET /v1/models
- POST /v1/chat/completions, streaming and non-streaming
- POST /v1/completions

It is text-only and currently does not support the OpenAI Responses API, embeddings, tools/function calls, multimodal content, structured JSON, stop sequences, or multiple choices. Unsupported behavior-bearing fields on the strict Chat Completions path return explicit typed errors rather than being silently ignored. This is a good correctness policy, but the missing features are material compatibility gaps. OpenAI Chat Completions is instrumented; the legacy /v1/completions route still forwards directly to the worker and produces no Clozn run ID, journal entry, memory/dial application, or receipt. See [OPENAI_COMPATIBILITY.md](./OPENAI_COMPATIBILITY.md).

The current working branch adds an Ollama-shaped surface:

- GET /api/tags
- GET /api/version
- POST /api/chat
- POST /api/generate

This is a minimal client-connectivity shim rather than general Ollama compatibility. Non-streaming chat/generate calls now pass through Clozn’s instrumented Chat Completions path, so those accepted calls produce normal gateway run records. Important differences remain:

- Ollama streaming requests return 501, and omitting stream produces a non-stream response even though upstream Ollama streams by default.
- /api/generate turns the prompt into chat messages and applies the GGUF chat template; it does not preserve raw-generation semantics.
- The requested model name is not validated or routed; the currently loaded worker answers.
- /api/tags describes only the loaded worker and uses placeholder size, digest, and timestamp fields.
- raw, template, format, suffix, keep_alive, context, think, images, and many options are currently ignored rather than rejected.
- Responses omit much of Ollama’s normal timing, token-count, model, and lifecycle metadata.
- model show/ps/pull/create/copy/delete, embeddings, tools, vision, structured output, thinking fields, and the wider option/lifecycle surface are absent.

This matters because upstream Ollama now supports default NDJSON streaming, streaming tool calls, thinking output, logprobs, structured JSON/schema output, vision, embeddings, OpenAI Responses, an Anthropic-compatible Messages surface, and a broad lifecycle API. [Ollama streaming](https://docs.ollama.com/api/streaming), [Ollama chat API](https://docs.ollama.com/api/chat), [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility), [Ollama structured outputs](https://docs.ollama.com/capabilities/structured-outputs)

**Product interpretation:** chat/generate are now on the correct instrumented architecture, but the branch is a connectivity proof, not yet an Ollama replacement contract. The streaming 501 and silently ignored Ollama fields are especially damaging because they turn protocol mismatch into apparent model behavior.

## 1.3 Run journal, trace, and replay

An engine-backed gateway Chat Completions run can record:

- A stable run ID and lineage
- The exact rendered prompt
- Partial model/runtime identity and sampling parameters: filename, inferred quant/family/model ID, context/device/offload, and sampling, but not a model SHA-256
- Token pieces, emitted-token probabilities, top alternatives, entropy, and finish reason
- Timings and performance metadata
- Active memory cards and behavior dials
- Confidence spans and close-call locations
- Immutable trace blobs referenced from a local SQLite journal

Clozn run has a narrower journal record: it stores the raw user prompt rather than the exact rendered template and omits much of the runtime/sampling identity. /v1/completions currently journals nothing. The caller-supplied top-level model name is not itself immutable identity.

Gateway runs can be inspected through CLI commands and the Replay UI. Current conversation branch/replay preserves lineage and regenerates from a truncated transcript. Snapshot descriptors exist, but the stateless chat path does not restore/reuse live KV state, so this should not be described as exact runtime time travel.

The instrumented Chat Completions API exposes an additive Clozn run ID and opt-in trust, receipt, and lens extensions. Third-party clients will usually ignore these extras, and streaming clients may discard custom response metadata. The stable side-channel therefore needs to be headers, server-side latest-run lookup, CLI watch mode, and optional standard trace export rather than colored or proprietary response text.

**Product interpretation:** this is already a meaningful moat. The remaining work is not inventing more telemetry; it is presenting the existing evidence around user questions and exporting it through standard seams.

## 1.4 Uncertainty and trust

Clozn has several distinct signals that must remain distinct:

- Raw emitted-token probability and top alternatives
- Token/span entropy or surprise
- Journal-derived acceptance calibration: replies historically kept at a given signal level
- Optional outcome-grounded calibration from labeled probes
- Optional local NLI-style support checks
- Selective-generation policies that can answer, ask, or abstain

The internal experiments found that emitted-token probability was useful for selective generation on the tested mixed set: AUROC 0.822. The documented setup was Qwen 3.5 9B Q4_K_M under greedy decoding, 362 items split TRAIN 180 / TEST 182, with 67 of 75 errors coming from a constructed hard-multiplication stress set. Self-reported confidence was effectively at chance because nearly every answer claimed certainty. The best internal probe appeared stronger on the mixed set but underperformed token probability on the fixed hard-arithmetic subset, showing that it had learned topic difficulty rather than privileged correctness. This is evidence for arithmetic-slip risk selection in that setup, not general factual uncertainty. Details and caveats are in [BACKLOG.md](./BACKLOG.md) and [RESEARCH_ROADMAP.md](./RESEARCH_ROADMAP.md).

Recent external work also warns that answer-token logprobs can be anti-calibrated or practically unhelpful on some tasks and models. [VERDI, May 2026](https://arxiv.org/abs/2605.11334), [Beyond Logprobs, June 2026](https://arxiv.org/abs/2606.24420)

**Product interpretation:** Clozn may say “token surprise,” “model confidence at generation,” or “calibrated risk on this task.” It must not say that raw probability is truth, factuality, or general confidence. The useful product is an evidence bundle and a risk policy, not a magical scalar.

## 1.5 Context, memory, and personalization

Product memory currently consists of readable prompt cards with:

- CRUD, review states, and provenance
- Topic gating and exact prompt assembly
- Manual activation and approval/dismissal
- Proposals derived from prior feedback/runs
- Profiles that snapshot a live setup
- A facts surface and memory-influence receipts

Quick feedback such as “too verbose” can be logged and turned into reviewable preference/dial proposals. This is a good trust model because persistent changes remain visible and reversible.

Important limits:

- A receipt can prove that a card was selected and injected. Proving that it governed the answer requires a counterfactual or causal test.
- The current facts read path is not yet a reliable native behavior-changing memory.
- “Internalized” soft-prefix training is a lab mode, not the product memory contract.
- Anchored memory is a model-specific Qwen 2.5 J-lens mechanism for narrow content traits; it is not universal factual memory and intentionally refuses rules/style.

Open WebUI’s own memory documentation makes the ecosystem difficulty explicit: small local models can store or retrieve memory inconsistently. [Open WebUI memory](https://docs.openwebui.com/features/chat-conversations/memory/)

**Product interpretation:** Clozn’s differentiated mainstream memory feature is not automatic memory creation. It is an exact, user-controlled memory receipt: what was selected, injected, omitted, scoped, and influential enough to survive a counterfactual check.

## 1.6 Steering and repair

Current steering surfaces include:

- Sampling and prompt-level changes
- Tone/behavior dials
- Contrastive direction derivation and calibration
- Direct concept-direction nudges
- Quick repair and preference workflows
- Optional J-transport for more stable stacking

The research findings sharply constrain the claims:

- Direct concept directions tend to surface literal anchor words more than a general behavioral register.
- Contrastive directions are better for style.
- J-transport improves authoring speed and multi-direction stability; it did not establish better steering direction quality.
- Some calibrated dials collapse or become incoherent above a narrow strength range.
- Per-domain behavioral calibration is necessary; logprob effect alone is insufficient.
- White-box steering remains artifact-qualified, not implied by GGUF runtime support.

**Product interpretation:** mainstream corrective actions should begin with prompt/sampling/profile interventions and compare before/after behavior. Activation steering should be labeled model-qualified and experimental until it passes a held-out target suite and a side-effect suite.

## 1.7 Developer evaluation and causal receipts

Clozn already contains unusually useful, but narrow, building blocks:

- Teacher-forced token scoring
- Tiny assertions over an already stored run; live mode adds the leans_on causal assertion rather than generating a fresh suite
- A small built-in greedy factual/arithmetic golden probe fixture; it does not accept user-defined cases/variants or enforce stored model/tokenizer/template identity
- Outcome-grounded Brier/ECE/risk-coverage evaluation
- A teacher-forced two-GGUF quant receipt over ten hand-picked prompts, measuring one-step argmax/dependence shifts rather than task score, full KL, latency/memory, or divergent end-to-end generations; its live two-engine smoke remains deferred
- Causal influence receipts scoped to supported memory cards, behavior dials, selected spans, swaps, and anchored-memory arms—not arbitrary retrieved documents or all context sources
- A run-scoped “change one thing” experiment envelope against one stored run; it has no named cases, variants, repetitions, metrics, suites, or multi-model orchestration
- Replay/fork/lineage and observational model diffs
- Model/artifact qualification contracts

The weakness is product composition. These are commands and panels, not yet one coherent experiment object that compares named variants, runs repeated cases, stores all reproduction metadata, shows target gains and regressions together, and exports standard evaluation records.

**Product interpretation:** Clozn is much closer to a distinctive “model CI” workflow than the current product story suggests.

## 1.8 Research and interpretability substrate

The runtime supports:

- Activation reads and layer sweeps
- Activation/steering writes
- Optional J-lens forward readouts
- An optional Qwen-specific SAE load/readout path; no SAE artifact-contract type is currently qualified
- A device-resident multi-observer plane
- Bit-exact token-state save/restore by re-prefilling saved tokens; direct KV-blob restoration and sampler/intervention snapshotting remain deferred
- Batched branch evaluation
- Causal tracing over layer-position sites with controls
- Intervention arms, joint/sub-additivity tests, path patching, and generation arms

Measured observer overhead was approximately 4.8% for one stream and 9.4% for two concurrent streams on 96-token Qwen 3.5 9B Q4 runs, with J-lens at layers 16 and 24 plus norms at every token, full coverage, and no observed drops. The current causal-trace battery predicted next-token flips at 88/96, or 91.7%, across 16 prompts on the documented 9B setup.

The caveats are just as important:

- A token flip is not the loss of an answer; the model may produce the answer one token later.
- Current nodes are layer-position sites, not discovered circuits or semantic components.
- Median legibility was only about 24%; most measured causal mass was not nameable.
- Only roughly 45% of nodes were strong in the reported battery.
- Interaction effects were large, so isolated-feature narratives are unsafe.
- Final-prompt-token sites often reflect depth-wise commitment, not necessarily the source of knowledge.
- The current SAE study is specifically Qwen 2.5 7B, the andyrdt layer-15 dictionary, and a small prompt set. Its reconstruction preserved approximately 99.9% of measured causal effect while explaining 57% of variance; individual active features carried about 0.5% of a site each, while top-8/16 groups carried 10–45% versus 0–6% for matched random groups.

The next scientific blockers are head/source-position granularity, a second model family, stronger nulls and controls, attention-versus-causal comparison, journal-driven input batches, external artifact contracts, and reproducible data export. See [scripts/tracer/README.md](../scripts/tracer/README.md) and the current causal-tracing section of [BACKLOG.md](./BACKLOG.md).

**Product interpretation:** this is a legitimate research substrate, but the honest name is “causal trace” rather than “circuit discovery.” MRI visuals should navigate measured data and controls, never imply that a beautiful activation picture is an explanation.

## 1.9 Studio and UX state

Studio currently has modules for Replay, Read, Memory, Facts, Patch, Scope, Experiment, Edit, Atlas, and Settings. The refreshed Replay view in the working branch is a substantial improvement: a light-theme operational surface with linked signal ribbon, token readout, trust channels, quick repair/steer, explain, receipts, span forensics, and memory influence.

The remaining product issue is information architecture:

- The interface is organized around backend capabilities and metaphors rather than user jobs.
- Several panels expose a signal without clearly stating what decision it supports.
- Model users have no reason to open Studio until a run fails; the runtime must surface value before that point.
- Model developers need a case-by-variant experiment matrix as their home, not a runtime dashboard.
- Researchers need raw values, controls, artifact identity, exports, and notebook/API links beside the visual.
- The repository still identifies a full real-browser/human-click pass as open work.

**Product interpretation:** the visual direction is not the core problem. The problem is that each audience needs a different default question, while the current UI presents one broad control room.

## 1.10 Research findings that should change the roadmap

The following internal findings should be treated as closed product decisions unless new evidence directly overturns them:

| Idea | Result | Product decision |
|---|---|---|
| Semantic-temperature dial | Killed | Do not build or position |
| Automatic collapse predictor from live energy | Killed prospectively | Do not show as predictive safety gauge |
| Branch at doubt/entropy spike | Killed; random positions did better | Do not spend compute based on token doubt alone |
| Paraphrase brittleness as wrong-answer signal | Killed at chance | Do not ship a brittleness receipt |
| Same-model verify-then-branch | Killed on hard tail | Do not market resampling/self-scoring as a reliability fix |
| Null-space watermark | Killed with parent mechanism | Do not build |
| Scalar self-reported confidence | Degenerate | Never use as trust evidence |
| Internal probe as general correctness detector | Did not survive hard-subset comparison | Treat as task/topic-specific unless outcome-calibrated |
| J-transport as steering-quality improvement | Unsupported | Position as stability/authoring infrastructure only |
| Direct concept direction as style control | Weak/literal | Use contrastive directions for style; qualify per model |
| “Model authorship” receipt | Works for verbatim, dies under paraphrase | Do not present as an authorship verdict |

Validated or promising directions:

- Exact run/context receipts
- Selective generation after task/model calibration
- Prompt/card counterfactual receipts
- Reproducible variant comparison
- Conservative, behaviorally calibrated dials
- J-transport for stable composition
- Causal tracing with explicit controls and narrow claims
- SAE reconstruction studies that measure causal fidelity rather than variance alone

## 1.11 Audit ledger and verification limits

The audit distinguishes shipped source from working-branch changes and research evidence. “Reviewed” below means code, tests, and documentation were inspected. I ran the local CLI help/registration check, but did **not** rerun the model-dependent benchmarks, launch a live model, exercise real Ollama/OpenAI clients, or perform a full browser click-through for this report.

| Area | Status at audited commit | Primary repository evidence | Verification in this audit | Confidence |
|---|---|---|---|---|
| Core gateway/worker and public CLI | Present on origin/main and working branch | ../clozn/cli/, ../clozn/server/, ../engine/, runtime tests | Source/docs reviewed; python3 -m clozn --help executed | High for structure; live behavior not rerun |
| Strict OpenAI Chat Completions | Present on origin/main and branch | OPENAI_COMPATIBILITY.md, server routes, OpenAI tests | Source/tests reviewed; no live SDK/model call | High for declared surface |
| Legacy /v1/completions | Present but bypasses Clozn run instrumentation | server route/worker forwarding code | Source reviewed | High |
| Ollama-shaped shim | Working branch b6190b8 only; not origin/main | tests/test_ollama_instrumented.py and server compatibility routes | Source/tests reviewed; no live Ollama SDK/app call | High for route behavior; medium for real-client compatibility |
| Run journal/receipts | Present for instrumented gateway chat; narrower or absent on other paths | ../clozn/runs/, receipt modules, run tests | Source/tests reviewed; no new live run | High for code path, medium for operational completeness |
| Memory/preferences/profiles | Present, with facts/anchored/internalized limits | memory, behavior, server routes, corresponding tests | Source/tests reviewed | High for implemented primitives |
| Replay light-theme refresh | Working branch b6190b8 | ../studio/heavn/modules/replay.mjs and UI tests | Source/model-free tests reviewed; no full browser pass | Medium |
| Quant, eval, golden tests, one-change experiments | Present but narrow, as described in section 1.7 | CLI commands, eval/experiment/receipt modules and tests | Source/tests reviewed; model-dependent runs not repeated | High for scope, no new outcome evidence |
| J-lens/SAE/causal-trace substrate | Research/qualified-model scope, not general product support | BACKLOG.md, RESEARCH_ROADMAP.md, ../scripts/tracer/, qualification code/tests | Source/docs reviewed; benchmark artifacts not rerun | Medium; conditional on documented model/artifact setup |
| AUROC 0.822 selective-generation result | Documented experiment: Qwen 3.5 9B Q4_K_M, greedy, 362 items, TEST 182, arithmetic-heavy error set | BACKLOG.md and referenced scratchpad result files | Documentation reviewed; result files/experiment not independently rerun here | Treat as scoped evidence, not a product-wide estimate |
| 4.8%/9.4% observer overhead | Documented 96-token 9B Q4 runs, J-lens L16/L24 plus norms every token | BACKLOG.md | Documentation reviewed; not benchmarked here | Scoped performance measurement |
| 88/96 causal token-flip prediction | Documented 16-prompt 9B battery | BACKLOG.md and ../scripts/tracer/README.md | Documentation/scripts reviewed; not rerun here | Scoped method result, not answer-loss prediction |
| SAE causal-fidelity result | Documented Qwen 2.5 7B andyrdt layer-15 dictionary study | BACKLOG.md and ../scripts/tracer/README.md | Documentation/scripts reviewed; not rerun here | One dictionary/layer/model only |

# 2. Audience research and jobs to be done

## 2.1 Model users

### What they are doing now

Common workflows combine Ollama or LM Studio with a preferred frontend such as Open WebUI, the Ollama app, LM Studio Chat, a coding assistant, an IDE extension, or a document/RAG tool. The runtime increasingly behaves as infrastructure rather than the place where the user chats.

Ollama now makes this explicit through direct integrations and one-command setup for Codex, Claude Code, OpenCode, and other tools. Coding agents are advised to use at least 64K context, which raises the memory/performance stakes for local hardware. [Ollama Codex integration](https://docs.ollama.com/integrations/codex), [Ollama Claude Code integration](https://docs.ollama.com/integrations/claude-code)

Users also compare models, quantizations, context sizes, thinking modes, and GPU offload. They usually diagnose failures by triangulating frontend behavior, runtime logs, task manager/GPU tools, and trial-and-error prompts.

### What they lack

**Exact context visibility.** Ollama’s automatic context policy can trade capacity for CPU fallback or severe slowdown, while frontend RAG documentation warns that retrieved material may be partially processed or not used when the context budget is insufficient. [Ollama context documentation](https://docs.ollama.com/context-length), [Ollama issue #14073](https://github.com/ollama/ollama/issues/14073), [Open WebUI RAG](https://docs.openwebui.com/features/chat-conversations/rag/)

**Compatibility diagnosis.** Tool formats, thinking fields, chat templates, stop behavior, and endpoint conventions differ. Recent Qwen/Ollama issues show that a response that looks like “the model is bad at tools” can actually be a renderer/parser or sampling-parameter defect. [Ollama Qwen 3.5 issue #14493](https://github.com/ollama/ollama/issues/14493), [Ollama Qwen tool issue #14601](https://github.com/ollama/ollama/issues/14601)

**Performance explanations.** Users can see raw metrics, but not a plain-language decomposition of load, prompt processing, generation, frontend auxiliary calls, context allocation, and CPU/GPU spill.

**Trustworthy uncertainty.** Runtimes expose logprobs, but raw probability is not factual correctness. Users need honest signal names, task calibration where possible, and recommended actions.

**Memory transparency.** Existing memory systems are useful but model-dependent. Users want to know what was saved, selected, injected, omitted, and scoped—not merely that “memory is on.”

**Corrective action in the moment.** Modelfiles, presets, and system prompts help users preconfigure behavior, but do not give an immediate, measured “less verbose,” “more concrete,” “use the supplied evidence,” or “ask before guessing” repair with before/after comparison.

### Recent model-user evidence

| Observation | Recent direct source | Evidence type / strength | Product inference |
|---|---|---|---|
| Local runtimes are increasingly infrastructure under existing coding, IDE, assistant, and RAG tools | [Ollama integrations, current](https://docs.ollama.com/integrations), [ollama launch, January 23, 2026](https://ollama.com/blog/launch) | Official capability evidence; strong for workflow availability, not usage share | Compatibility and setup helpers are an adoption contract |
| Automatic large context can spill to CPU and make a machine appear hung | [Ollama issue #14073, February 2026](https://github.com/ollama/ollama/issues/14073) | Concrete issue; directional | Show actual allocation, spill, and why-slow diagnosis |
| Requested context and the runtime’s reported/allocated context can diverge | [Ollama issue #15944, May 2026](https://github.com/ollama/ollama/issues/15944) | Concrete issue; directional | Receipt the requested, capped, and actual context separately |
| Tool and thinking failures can come from renderer/parser/sampling defects rather than model capability | [Ollama issue #14493, February 2026](https://github.com/ollama/ollama/issues/14493), [issue #14601, March 2026](https://github.com/ollama/ollama/issues/14601) | Source-level bug reports; strong failure examples | Protocol/template qualification must precede behavioral diagnosis |
| Users struggle to make local tool calls reliable across Open WebUI, LM Studio, Ollama, and Qwen models | [LocalLLaMA discussion, spring 2026](https://www.reddit.com/r/LocalLLaMA/comments/1sp631h/are_you_guys_actually_using_local_tool_calling_or/), [Open WebUI/Qwen discussion, March 2026](https://www.reddit.com/r/LocalLLaMA/comments/1rhmwfn/cant_get_qwen_models_to_work_with_tool_calls/) | Community anecdotes; low prevalence confidence | A conformance report is more useful than blaming “small models” generically |
| A frontend can turn a fast direct Ollama reply into a much slower perceived workflow through context/reloads/auxiliary requests | [Ollama discussion, July 16, 2026](https://www.reddit.com/r/ollama/comments/1uy50kf/slow_response_time_when_using_webui_and_ollama/) | Recent anecdote | Group calls by client/session and explain load, prompt, generation, and auxiliary work |
| Open WebUI explicitly warns that memory quality depends on model capability and that small local models may be inconsistent | [Open WebUI memory, current](https://docs.openwebui.com/features/chat-conversations/memory/) | Official limitation; strong | Make memory explicit, scoped, and reviewable rather than magical |
| Quantization labels themselves are a setup/choice barrier | [LocalLLM discussion, July 17, 2026](https://www.reddit.com/r/LocalLLM/comments/1uz5uv2/quantization_without_the_jargon_what_q4_q5_q8/) | Community anecdote | Present fit/quality/speed tradeoffs in plain language |

### Model-user product contract

**Must be excellent**

1. Existing Ollama/OpenAI applications continue to work, including streaming and tools.
2. Every run receives a stable local receipt without polluting response text.
3. “What entered the model?” and “what was truncated?” are exact and easy to answer.
4. Slow, cut-off, looping, and tool-failure runs receive a plain-language diagnosis.
5. Probability, support, and calibrated risk are visibly different concepts.
6. A targeted correction is one command/action away and fully reversible.
7. Local-only operation is visible and auditable.
8. The core value works without opening Studio.

**Useful but secondary**

- Compare/replay and side-by-side repair
- Explicit memory management and profiles
- Run search and export
- Lightweight notification or tray/watch surface
- Fit and context recommendations

**Low importance for this audience**

- Neuron/feature visualization
- J-lens training
- Full experiment design
- Fine-tuning pipelines and model publishing
- Dense research controls

### Recommended model-user flow

**Proposed target flow; these commands are not the current CLI contract.**

~~~bash
clozn serve qwen3.5 --compat ollama --local-only
clozn watch

# After something odd happens in Open WebUI, Codex, or another app:
clozn explain last
clozn context last
clozn retry last --less-verbose
clozn compare last
clozn studio open last
~~~

The response in the third-party application remains clean. Clozn provides a run ID header where possible and keeps a local “latest request from this client/session” index for clients that drop custom metadata.

## 2.2 Model developers

### What they are doing now

In this report, **model developer** means a model author, fine-tuner, adapter/steering author, quantizer, or optimizer—not the much larger group of application developers who merely call an LLM API. Application developers overlap the model-user compatibility needs and the trace/evaluation integrations, but deserve separate validation if they become a primary audience.

The common loop is:

~~~text
collect/clean data
    → train an adapter or checkpoint
    → inspect loss
    → manually chat or run a benchmark
    → merge/convert/quantize
    → serve through Ollama, llama.cpp, or vLLM
    → discover behavioral or template regressions
~~~

Training is already well served by Transformers, TRL, PEFT, Unsloth, Axolotl, and LLaMA-Factory. Training tracking is served by W&B, MLflow, TensorBoard, and Trackio. Evaluation catalogs and execution are served by LightEval, Inspect, lm-evaluation-harness, and Promptfoo.

Hugging Face’s 2026 Community Evals work explicitly identifies saturated benchmarks, disagreement between nominally identical results, and scattered evaluation records. Its Evaluation Cards analysis found that 96.5% of more than 50,000 evaluation records lacked at least one minimal reproduction field; max_tokens and temperature were absent from the overwhelming majority. [Hugging Face Community Evals, February 4, 2026](https://huggingface.co/blog/community-evals), [HF and Every Eval Ever, June 30, 2026](https://huggingface.co/blog/eee-community-evals), [Evaluation Cards, June 2026](https://huggingface.co/blog/evaleval/evaluation-cards-launch)

### What they lack

**A closed behavioral loop.** Training loss does not answer whether a target behavior improved, general ability regressed, tool output broke, or style became brittle.

**Deployment equivalence.** Chat templates, BOS/EOS/pad tokens, tokenizer revisions, adapter/base identity, quantization paths, and conversion tools can silently change behavior. Recent Unsloth, TRL, llama.cpp, vLLM, and Ollama documentation/issues all show concrete forms of this failure. [Unsloth issue #3899](https://github.com/unslothai/unsloth/issues/3899), [TRL issue #5138](https://github.com/huggingface/trl/issues/5138), [llama.cpp issue #19626](https://github.com/ggml-org/llama.cpp/issues/19626), [vLLM LoRA documentation](https://docs.vllm.ai/en/stable/features/lora/), [Ollama model import](https://docs.ollama.com/import)

**Paired case-level comparison.** Aggregate scores hide whether an intervention fixed the intended behavior while creating weak regressions elsewhere.

**Reproducible shareable evidence.** An adapter or personalized model is usually shared as weights plus an incomplete model card, not a manifest tying immutable revisions, prompt template, decoding configuration, test suite, and per-case results together.

**Steering safety.** Activation steering is becoming an explicit toolkit/artifact category, but effects remain layer-, task-, model-, and data-dependent. Shared vectors need provenance and side-effect tests. [AI Steerability 360, March 2026](https://arxiv.org/abs/2603.07837), [Steering Vectors Are an Adversarial Attack Surface, June 2026](https://arxiv.org/abs/2606.05958)

### Recent model-developer evidence

| Observation | Recent direct source | Evidence type / strength | Product inference |
|---|---|---|---|
| Evaluation results are scattered and frequently omit reproduction-critical fields | [HF Community Evals, February 4, 2026](https://huggingface.co/blog/community-evals), [Evaluation Cards, June 2026](https://huggingface.co/blog/evaleval/evaluation-cards-launch) | Official ecosystem analysis with a 50,000+ record sample; strong | Complete receipts and standard export are a real wedge |
| Practitioners recommend defining a small real-work eval before tuning and separating changing knowledge from behavioral tuning | [LocalLLaMA discussion, July 14, 2026](https://www.reddit.com/r/LocalLLaMA/comments/1uw9tom/post_training_custom_datasets/) | Recent community advice; directional | Promote real captured runs into target and guard suites |
| Data cleaning/formatting/deduplication is often harder than invoking the trainer | [LocalLLM discussion, June 18, 2026](https://www.reddit.com/r/LocalLLM/comments/1u92lvs/one_way_to_make_data_preparation_easier_when/) | Community anecdote | Integrate with data/training tools; do not build another trainer |
| Fine-tunes can show seed instability, length overfitting, or loss of general ability | [Regression discussion, January 7, 2026](https://www.reddit.com/r/LocalLLaMA/comments/1q691wa/do_you_see_instability_or_weird_regressions_when/), [fine-tuning discussion, April 19, 2026](https://www.reddit.com/r/LocalLLaMA/comments/1sq816a/question_regarding_fine_tuning/), [small-model regression, May 15, 2026](https://www.reddit.com/r/LocalLLaMA/comments/1tcihvl/i_taught_my_1b_to_follow_instructions_it_got/) | Multiple community anecdotes; directional, not prevalence | Show target gain and general guard-suite regressions together |
| Chat-template/EOS/export mismatches can produce garbled GGUF behavior | [Unsloth issue #3899, January 16, 2026](https://github.com/unslothai/unsloth/issues/3899), [TRL issue #5138, February 20, 2026](https://github.com/huggingface/trl/issues/5138), [llama.cpp issue #19626, February 14, 2026](https://github.com/ggml-org/llama.cpp/issues/19626) | Concrete tool issues; strong failure examples | Deployment-equivalence validation should be first-class |
| PEFT method choice should be compared on task, resource cost, and forgetting rather than defaulting to LoRA | [Hugging Face Beyond LoRA, June 18, 2026](https://huggingface.co/blog/peft-beyond-lora) | Official controlled comparison | Clozn should compare artifacts, not prescribe a trainer |
| Quantization authors increasingly examine distribution drift and same-top-token behavior, not only perplexity | [Qwen 3.5 9B comparison, March 11, 2026](https://www.reddit.com/r/LocalLLaMA/comments/1rr72lr/qwen359b_quantization_comparison/), [Qwen 3.6 27B comparison, May 29, 2026](https://www.reddit.com/r/LocalLLaMA/comments/1tr9vzn/qwen3627b_quantization_benchmark/) | Community benchmark work; directional | Extend quant-check toward paired behavioral and distribution evidence |

### Model-developer product contract

**Must be excellent**

1. Reproducible run receipts capture every behavior-bearing parameter and immutable artifact identity.
2. A suite runs the same real cases against base, adapter, checkpoint, merged model, quantization, prompt, and steering variants.
3. Target gain, capability retention, format/tool validity, robustness, latency, and memory are shown together.
4. Every aggregate result drills down to paired outputs, exact rendered context, and trace evidence.
5. Deployment-equivalence validation catches template/tokenizer/adapter/export mismatches.
6. Captured real-app runs can be promoted into regression cases.
7. Results export to Hugging Face Community Evals/EEE and common test formats.
8. Shared steering/personalization artifacts include provenance and held-out side-effect evidence.

**Useful but secondary**

- Adapter import, aliases, safe enable/disable/hotswap
- HF model/dataset revision pinning
- LightEval/Inspect/lm-eval adapters
- Training-log links from Trackio/W&B/TRL
- Quantization quality/size/speed Pareto views
- Checkpoint sweep orchestration

**Do not compete on**

- Distributed/full-model training
- General dataset labeling or synthetic-data generation
- Another trainer GUI
- High-throughput multi-tenant serving
- A new benchmark catalog or model marketplace

### Recommended model-developer flow

**Proposed target flow; most commands below do not exist today.**

~~~bash
clozn model add hf://org/base@revision
clozn adapter add hf://user/adapter@revision
clozn validate adapter:user/adapter --base org/base

clozn suite create my-real-tasks --from-runs last-50
clozn experiment run my-real-tasks \
  --variants base,adapter,gguf-q4,steer:grounded \
  --seeds 3

clozn compare EXPERIMENT_ID
clozn studio open EXPERIMENT_ID
clozn receipt export EXPERIMENT_ID --format eee
~~~

Studio’s default view for this audience should be a case-by-variant matrix. The MRI appears after a user clicks a regression cell.

## 2.3 Model researchers

### What they are doing now

A typical mechanistic-interpretability workflow is:

1. Define a behavior and clean/corrupt or positive/negative contrasts.
2. Capture residual, attention, MLP, logit, and sometimes gradient data.
3. Locate candidates with logit/tuned/J-lenses, probes, attribution patching, SAEs, transcoders, or automated discovery.
4. Interpret candidates through top activations, labels, attention views, or graphs.
5. Patch, ablate, clamp, swap, or steer a direction across layer/position/strength sweeps.
6. Measure intended effect, collateral behavior, robustness, and simple baselines.
7. Share notebooks, model/artifact revisions, tensors, and visual links.

TransformerLens remains a standard local notebook library for caching and changing activations. Its version 3 TransformerBridge expands Hugging Face architecture and quantization support while also illustrating why exact backend/numerical provenance matters: the project documents migration differences and historical behavioral drift between implementations. [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), [TransformerLens v3 migration](https://transformerlensorg.github.io/TransformerLens/content/migrating_to_v3.html)

NNsight and NDIF provide Pythonic local/remote interventions, gradients, batching, generation-step editing, a no-code Workbench, and remote large-model execution. Version 0.6 added streaming vLLM interventions, remote SAEs/adapters, and an open-source NDIF server. [NNsight](https://github.com/ndif-team/nnsight), [NNsight 0.6, February 26, 2026](https://nnsight.net/blog/2026/02/26/introducing-nnsight-06/)

Neuronpedia already provides the strongest broad visual/share layer: SAE features, explanations, steering, probes/vectors, attention-head views, J-lenses, natural-language autoencoders, and circuit graphs over a large public artifact collection. [Neuronpedia](https://www.neuronpedia.org/), [Neuronpedia open source](https://www.neuronpedia.org/blog/neuronpedia-is-now-open-source)

SAELens handles SAE training/loading/evaluation; circuit-tracer produces prompt-specific attribution graphs; and Hugging Face has become the distribution layer for model, SAE, lens, and steering artifacts. [SAELens](https://github.com/decoderesearch/SAELens), [circuit-tracer](https://github.com/decoderesearch/circuit-tracer)

### What changed recently

The researcher gap is not “there is no polished interpretability UI.” The durable gap is the experiment loop joining a real behavior to provenance, localization, a causal intervention, collateral evaluation, and a shareable result. A July 2026 practical survey describes actionable interpretability as “Locate, Steer, Improve,” which closely matches Clozn’s strongest substrate. [Actionable Mechanistic Interpretability, ACL Findings 2026](https://aclanthology.org/2026.findings-acl.502/)

The July 2026 J-space/J-lens release is particularly relevant. A pre-fitted Jacobian lens can project a residual vector into the final-layer vocabulary basis, giving a layer-by-position view of concepts the model is disposed to verbalize later. Neuronpedia quickly added interactive support and pre-fitted lenses across multiple Gemma, Llama, GPT-OSS, and Qwen models. The method’s own documentation says it is a bag-of-concepts, mostly single-token view with uninterpretable regions—not a literal transcript of thought. [Anthropic J-space](https://www.anthropic.com/research/global-workspace), [J-lens repository](https://github.com/anthropics/jacobian-lens), [Neuronpedia J-lens update, July 10, 2026](https://www.neuronpedia.org/blog/jacobian-lens)

Qwen-Scope released SAEs across Qwen 3/3.5 variants, including a Qwen 3.5 9B residual-stream artifact. This is unusually aligned with Clozn’s current model work and should be consumed and validated before Clozn trains a replacement. [Qwen-Scope, May 12, 2026](https://arxiv.org/abs/2605.11887), [Qwen 3.5 9B SAE artifact](https://huggingface.co/Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50)

At the same time, recent work increases the burden of proof:

- Nominally identical SAEs can learn materially different dictionaries across random seeds. [ICLR 2026 SAE consistency study](https://iclr.cc/virtual/2026/poster/10010650), [ACL 2026 consistency study](https://aclanthology.org/2026.acl-long.99/)
- Feature absorption and abstraction mismatch remain real limitations.
- A May 2026 audit found reliability failures in canonical SAE benchmark metrics. [SAE metric audit, May 2026](https://arxiv.org/abs/2605.18229)
- AxBench found prompting beat the tested steering methods in its study, while later work found supervised SAE selection more competitive. The field does not support a universal steering winner. [AxBench](https://proceedings.mlr.press/v267/wu25a.html), [supervised SAE selection study, May 2026](https://arxiv.org/abs/2605.31183)
- Natural-language activation explanations can be useful for hypothesis generation but can confabulate. [Anthropic Natural Language Autoencoders](https://www.anthropic.com/research/natural-language-autoencoders), [Neuronpedia NLA discussion](https://www.neuronpedia.org/blog/nlas)

### Model-researcher product contract

**Must be excellent**

1. Exact provenance: model revision, quantization, dtype, backend/build, tokenizer/template, seed/sampler, hook semantics, artifact hashes, prompt/context, and intervention.
2. A versioned selective activation-capture and intervention contract.
3. Deterministic replay where possible and explicit nondeterminism where not.
4. One experiment runner for baseline, null/control, layer/position/strength sweeps, target metrics, and collateral metrics.
5. Trace/run comparison aligned at prompt, token, layer, intervention, and outcome levels.
6. Open export: a readable manifest plus common tensor/data formats and generated Python/notebook code.
7. Every visual identifies evidence type, method, hook point, artifact, fidelity/coverage, thresholds, and causal-validation status.
8. Automation through CLI/Python; Studio is optional.

**High-value integrations**

- Apply pre-fitted J-lenses in live runs.
- Apply and validate Qwen-Scope/SAELens artifacts.
- Fit/evaluate linear probes from labeled receipts.
- Export/import TransformerLens and NNsight studies.
- Export selected artifacts/results to Hugging Face and Neuronpedia.
- Attach Clozn receipts and validation arms to circuit-tracer graphs.

**Lower priority**

- Fitting J-lenses inside the fast runtime
- Native full circuit-tracer reimplementation
- Training SAEs or cross-layer transcoders
- A public feature/model catalog
- Remote frontier-scale compute infrastructure

### Recommended researcher architecture

Trying to make the production C++ runtime perform every research job would be expensive and constraining. Use two backends with one receipt schema:

~~~text
Fast Clozn runtime
  ordinary local inference
  token distributions and timings
  selective forward hooks
  application of pre-fitted probes, SAEs, J-lenses, and steering vectors

Research backend
  Hugging Face / PyTorch / TransformerBridge or NNsight sidecar
  gradients and Jacobians
  fitting
  rich activation patching
  circuit tracing

Shared layer
  run IDs, artifact identity, experiment manifests, metrics, exports, and Studio views
~~~

The fast runtime remains the place where an actual user run is captured. A “send to lab” action creates a reproducible sidecar study against an equivalent checkpoint and clearly reports any quantization/backend mismatch.

### Hard research blockers

**Activation volume.** A 32-layer, width-4096 residual stream over 2,048 tokens is roughly 512 MiB in fp16 before attention, MLP, gradients, turns, or generation steps. Full capture must be opt-in and budgeted. Prefer layer/position filters, last-token capture, online projection into a lens/SAE/probe, sparse top-k storage, and a preflight size estimate.

**Quantized artifact validity.** Most lenses and SAEs are fitted against a specific fp16/bf16 Hugging Face checkpoint. A matching family name is not sufficient for a GGUF quantization. Qualification should test activation alignment, lens-rank stability, SAE firing/reconstruction, steering direction, collateral behavior, prompt template, and distribution.

**Hook semantics.** Hybrid/nonstandard architectures make “layer 12 residual” ambiguous. The hook contract must record pre/post norm, pre/post residual add, attention/MLP/delta block, token position, generation step, and architecture mapping.

**Gradient boundary.** Full Jacobian fitting and many circuit methods require autograd. This is the primary reason for the research-backend split.

### Recommended researcher flow

**Proposed target flow; this is a product specification, not the current CLI.**

~~~bash
clozn trace capture RUN_ID --hook resid_post --layers 8,12,16 --positions last
clozn lens apply RUN_ID --artifact qwen3.5-9b-jlens
clozn sae inspect RUN_ID --artifact Qwen/SAE-Res-Qwen3.5-9B-Base-W64K-L0_50
clozn intervene sweep RUN_ID --feature 6159 --layers 8:20 --strength -4:4
clozn compare BASELINE VARIANTS --eval evals.yaml
clozn receipt export EXPERIMENT_ID --format bundle
clozn studio open EXPERIMENT_ID
~~~

# 3. Feature portfolio and gap analysis

The canonical priority vocabulary is:

- **Must excel:** adoption or scientific-credibility gate for at least one target audience.
- **Adequate:** reliable integration/support is required, but Clozn does not need category leadership.
- **Deprioritize:** useful later or only after a narrower path works.
- **Non-goal:** deliberately outside the product boundary.

Roadmap phase numbers describe sequencing, not quality. “Must excel within a qualified scope” means a small supported model/artifact set must be rigorous; broad artifact/model coverage can still be merely adequate or deferred. Likewise, memory **visibility** is a must while broad automatic memory management is adequate, and research evidence **correctness/provenance** is a must while visual polish is adequate.

## 3.1 Foundation and model-user features

| Feature | Audience / excellence bar | Existing solutions | Current backend/support | Target experience | Backend effort | UX effort | Blocker or research need |
|---|---|---|---|---|---:|---:|---|
| Ollama chat/generate compatibility | Users: must excel; developers: must | Ollama native API; LM Studio compatibility | Minimal current-branch connectivity shim: placeholder tags plus non-stream chat/generate through instrumented chat; model names are not routed, generate is templated as chat, default stream semantics differ, many fields are ignored, and streaming is 501 | Change base URL and keep the existing app; compatibility test report names any missing behavior | L, ongoing | M | Model-specific tool/thinking parsers, raw-generate semantics, metadata/lifecycle fidelity, and real-client conformance |
| OpenAI compatibility | Users: must excel | Ollama, LM Studio, llama.cpp, LocalAI | Partial: instrumented text-only Chat Completions; models; legacy completions bypasses the run journal/receipts | Make Chat Completions first-class; instrument or explicitly retire legacy completions; add Responses, tools, structured output, stop, usage, cancellation, capability metadata | L | S–M | Responses event semantics; tool round trips; strict no-silent-ignore policy |
| Anthropic compatibility | Important if coding agents are a target | Ollama and LM Studio expose Messages-style compatibility | Absent | Explicit later milestone or adapter; do not imply support | L | S | Broad protocol and agent-harness behavior; decide after OpenAI/Ollama parity |
| Streaming | Users: release-blocking | Default in Ollama REST; OpenAI SSE elsewhere | OpenAI stream exists; Ollama stream 501 | Correct NDJSON/SSE chunks, cancellation, finish reason, thinking/tool fields, final receipt mapping | M | S | Streaming must still write one coherent instrumented run without double-finalization |
| Tools/function calls | Users: release-blocking for agents | Ollama/OpenAI/Anthropic and current local frontends | Absent on public compatibility path | Tool schema and results preserved exactly; tool calls visible in receipt; model-specific parser qualification | L | M | Renderer/parser correctness varies by model; malformed output recovery must be explicit |
| Structured output / JSON schema | Users/developers: high | Ollama, LM Studio, llama.cpp grammars | Absent | Accept native/OAI schemas, constrain decoding, validate response, record schema/hash/failures | M–L | S | Grammar/constrained decoding support in engine |
| Thinking/reasoning fields | Users: high for current models | Ollama and current OpenAI-compatible runtimes | Not public compatibility behavior | Preserve separate thinking field when model/template supports it; never leak/merge accidentally | M | M | Per-model templates and protocol mappings |
| Vision/multimodal | Users: useful, not initial differentiator | Ollama/LM Studio/Open WebUI | Absent | Clearly capability-gated; add after text/tool parity | L–XL | M | Engine architecture/projectors and image tokenization; broad scope |
| Embeddings | Useful for Open WebUI/RAG compatibility | Ollama/LM Studio and dedicated embedding servers | Absent; one generative model per gateway | Either a small managed embedding sidecar or explicit external embedding config | L | M | Multi-model lifecycle and memory footprint; avoid pretending generative and embedding model are one |
| Model lifecycle/API metadata | Users: compatibility support | Ollama tags/show/ps/pull/create/copy/delete; LM Studio model API | CLI pull/models/ps exists; API surface mostly absent | Familiar aliases and honest capability/model metadata; lifecycle actions may call Clozn’s own store | M–L | M | Names/digests/manifests differ from Ollama; define compatible semantics rather than fake fields |
| One-command app setup | Users: high adoption lever | ollama launch configures Codex/Claude Code/OpenCode | Absent | clozn connect APP or generated copy/paste config with backup/restore and health check | M | M | Do not overwrite user configs without backup; each app changes |
| Real-client conformance | All: must excel | Upstream test suites are incomplete; apps reveal edge cases | Unit/route tests exist; no broad app certification | Matrix of Open WebUI, Ollama SDKs, OpenAI SDKs, Codex/OpenCode, cancellation/tools/streaming | M, recurring | S | CI hardware/model cost; maintain golden recordings and small live tests |
| Stable run ID side-channel | All: must excel | Generic trace IDs/OTel | Additive response ID/header in some paths; local journal | X-Clozn-Run-ID where possible, latest-by-client/session fallback, inspect/watch commands | S–M | S | Browsers/SDKs may hide headers; streaming final metadata differs |
| Exact input/context receipt | Users: differentiator; developers/researchers: must | LM Studio raw formatted-input logs; generic traces; RAG citations | Partial gateway primitives: rendered prompt, memory/dials, sampling, trace; CLI run stores raw input and legacy completions stores no run; truncation UX incomplete | Delivered / survived / influenced separated; exact token budget, retained/dropped turns, memories, docs, tools, template overhead | M | M | Unify all accepted execution paths; add engine/gateway truncation events and source boundaries |
| Context influence receipt | All: valuable but narrow | RAG attribution tools and research methods | Partial and scoped: supported cards, dials, selected spans, swaps, and anchored-memory arms | Async “test influence” on an explicitly supported source/span with baseline/null and measured effect | M–L | M | Counterfactual cost; correlated sources; arbitrary retrieved-document/context attribution is absent |
| Token probability/surprise | Users: useful if honest | Ollama/LM Studio expose logprobs | Shipped in CLI heat and trace/Studio | Clean third-party text; sidecar heatmap; surprising spans and alternatives; clear label | S | S–M | Tokenization/UI alignment; never call raw value factual confidence |
| Calibrated risk / ask / abstain | Users: potential differentiator | Research systems and custom guardrails; no universal local-runtime answer | Partial: outcome calibration, journal proxy, policy metadata | Per-model/task calibration wizard and policy; scope/coverage/error curve shown | M | M | Requires labeled outcomes; may fail on hard-tail distribution shifts |
| Context/source support | Users/developers: useful | RAG evaluators, NLI/checkers | Optional local support signal | Distinct “source support” channel with cited span and checker identity | M | M | A local checker is not external truth; small models may be weak judges |
| Quick corrective retry | Users: must excel | Regenerate/edit/system prompts/presets exist elsewhere | Partial: quick repair, dials, preferences, replay | retry last --less-verbose / --more-concrete / --use-context / --ask-before-guessing; one-shot/session/profile; compare and undo | M | M | Begin with prompts/sampling; activation paths only when qualified |
| Persistent behavior profile | Users: high; developers: useful | Modelfiles, LM Studio presets, Open WebUI custom models | Partial profiles and preferences | Named, diffable profile bound to model/app/project; exact receipt; exportable manifest | M | M | Conflict order among app system prompt, profile, memory, and request |
| Memory visibility and scope | Users: must; developers: useful | Open WebUI, Letta, Mem0 | Partial cards, topic gating, CRUD/review/provenance | memory used last; selected/injected/omitted and token cost; app/model/project scope; retry without | M | M | Reliable automatic selection on small models; facts path not yet native/causal |
| Automatic memory extraction | Users: secondary | Open WebUI, Mem0, Letta | Proposals/review exist | Opt-in proposal inbox; never silently rewrite identity/preferences | M–L | M | Conflict/freshness and small-model extraction quality; recent memory audits show this remains hard |
| Performance diagnosis | Users: differentiator; developers: must | ollama ps/logs; LM Studio estimates/log stream | Timings, plan, doctor, runtime metadata exist | Explain load, prompt, generation, hidden calls, context/KV allocation, CPU/GPU spill, and safe next action | M | M | Hardware/backend-specific resource telemetry and grouped app requests |
| Instrumentation performance envelope | All: must trust | Profiler/observability tools report overhead; few local runtimes qualify white-box cost | Observer-plane overhead/coverage/drop measurements exist for one documented 9B setup; no general user-facing budget | Preflight and receipt show each observer’s expected/measured overhead, coverage, drops, storage, and an uninstrumented baseline | M | M | Repeat across models/devices/modes; instrumentation must not silently change behavior |
| Local-only audit | Users: trust requirement | Ollama no-cloud; LM Studio offline | Loopback/local architecture; product messaging/policy incomplete | local-only flag, no silent cloud fallback, outbound/tool ledger, doctor --verify-offline | S–M | M | OS-level verification differs; be precise about process/tool network activity |
| Run-journal privacy and lifecycle | All: trust requirement | Langfuse/Phoenix retention controls; app history deletion | Local SQLite plus content-addressed blobs, migrations, export, and unreferenced-blob GC; no coherent per-run redaction/retention/delete policy UX | Inspect storage cost; redact/export/delete a run or client/session; retention policy; secure purge semantics; private fields off by default in external export | M | M | Referential integrity across lineage/blobs; recovery versus irreversible deletion; sensitive prompt handling |
| Standard trace export | Developers/users: support | OpenTelemetry/OpenInference, Langfuse, Phoenix | Local proprietary run journal | Optional OTel/OpenInference export with prompt privacy controls; run ID correlation | M | M | Map rich local evidence without leaking prompt content by default |

### Current foundation UX and canonical priority

This companion table separates user-facing access **today** from backend primitives in the main matrix.

| Feature | Current UX today | Canonical priority |
|---|---|---|
| Ollama chat/generate compatibility | A user can manually point a non-streaming client at the server; common default streaming returns 501 and ignored fields are not explained in-product | Must excel |
| OpenAI compatibility | clozn serve supports text Chat Completions; strict errors are useful, but legacy completions silently lacks Clozn receipts | Must excel |
| Anthropic compatibility | No user-facing path | Adequate after Ollama/OpenAI parity; must only if coding-agent users become the primary wedge |
| Streaming | OpenAI clients can stream; Ollama clients receive 501 | Must excel |
| Tools/function calls | No public compatibility UX | Must excel for agent/tool clients |
| Structured output / JSON schema | Strict OpenAI calls are rejected; Ollama-shaped fields may be ignored | Adequate initially; must for developer/agent certification |
| Thinking/reasoning fields | No separate compatibility/UI treatment | Adequate |
| Vision/multimodal | No supported UX | Deprioritize until text/tools are excellent |
| Embeddings | No supported UX; users must configure another service outside Clozn | Adequate for RAG integrations, not differentiation |
| Model lifecycle/API metadata | CLI models/pull/ps/stop exists; Ollama tags is a placeholder rather than a familiar lifecycle UX | Adequate |
| One-command app setup | Manual base-URL/config editing only | Adequate, high adoption leverage |
| Real-client conformance | Route/unit tests are invisible to users; no compatibility badge/report | Must excel |
| Stable run ID side-channel | Some instrumented non-stream responses expose an ID/header and inspect works if the user retains it; no client/session latest lookup | Must excel |
| Exact input/context receipt | CLI explain/inspect/trace and Replay expose pieces; no single delivered/survived view, and coverage differs by entry path | Must excel |
| Context influence receipt | Selected card/span/swap receipts exist through specialized CLI/API/Studio paths; no arbitrary-source action | Adequate, rigorous only where offered |
| Token probability/surprise | clozn run --heat, trace, and Replay render token-level evidence | Must excel as an honest differentiator |
| Calibrated risk / ask / abstain | eval/trust/policy primitives exist, but there is no end-to-end calibration wizard or reliable ask/abstain action in normal use | Adequate/research until outcome calibration is supplied |
| Context/source support | Optional explain/receipt signal; not a general evidence workflow | Adequate |
| Quick corrective retry | Studio has quick-repair/preference surfaces and branch/replay primitives; proposed retry-last CLI actions do not exist | Must excel |
| Persistent behavior profile | Profile/preference panels and CLI preference review exist, but no simple app/project-scoped diffable CLI workflow | Adequate |
| Memory visibility and scope | Studio exposes memory cards/review/provenance; no concise memory-used-last receipt or complete app/project scoping | Must excel for visibility; CRUD breadth adequate |
| Automatic memory extraction | Reviewable proposals exist; no dependable automatic cross-model memory UX | Adequate and opt-in |
| Performance diagnosis | plan and doctor plus timings exist; no one-screen “why this run was slow” synthesis | Must excel |
| Instrumentation performance envelope | No normal user view of observer overhead/coverage/drop/storage; measurements live in research docs/frames | Must excel for trust when instrumentation is enabled |
| Local-only audit | Loopback/local architecture is implicit; no local-only switch, outbound ledger, or verification command | Must excel |
| Run-journal privacy and lifecycle | Migrate/export/GC primitives exist; no coherent per-run redact/retention/delete UX | Must excel |
| Standard trace export | Single-run proprietary bundle only; no OTel/OpenInference settings or privacy preview | Adequate |

## 3.2 Model-developer features

| Feature | Excellence bar | Existing solutions | Current backend/support | Target experience | Backend effort | UX effort | Blocker or research need |
|---|---|---|---|---|---:|---:|---|
| Experiment object and suite runner | Must excel | Promptfoo, Inspect, LightEval, lm-eval | Narrow pieces: assertions over a stored run, outcome eval, a fixed greedy probe fixture, and a one-run/one-change experiment envelope | One manifest names cases, variants, repetitions, metrics, target suite, guard suite, and immutable outputs | M | M | Build real orchestration and a metric contract without inventing a closed benchmark catalog |
| Real-run-to-test promotion | Must excel | Observability products can create datasets from traces | Runs/journal exist; no polished suite promotion | suite create --from-runs; redact/edit; freeze expected behavior; retain original receipt | M | M | Privacy/redaction and non-deterministic expected outputs |
| Paired variant comparison | Must excel | Eval tools compare scores; chat UIs compare responses | Partial single-run replay/model diff plus a constrained one-step two-GGUF quant receipt | Case × variant matrix across base, prompt, adapter, checkpoint, quant, dial; aligned output and trace diff | M–L | L | Token alignment after divergence; one-model-per-gateway orchestration |
| Target gain plus regression guard | Must excel | Inspect/LightEval/custom CI | Tiny assertions and eval metrics exist | Every experiment shows intended gains, retained capabilities, format/tool validity, latency/memory, and worst regressions | M | M | Need metric plugin contract and guard-suite defaults |
| Reproduction receipt | Must excel | HF Evaluation Cards, W&B/Trackio, generic manifests | Partial gateway run metadata and a single-run receipt_bundle.v1; immutable model/tokenizer/template/adapter/tool/dependency identity is incomplete | Capture immutable model/tokenizer/template/adapter hashes, build, hardware, schema/tools, context, decoding, seeds | M | M | Some runtimes do not provide deterministic seeds; hash user-owned artifacts safely |
| Deployment-equivalence validation | Must excel | Fragmented checks in training/export tools | Absent end to end; white-box artifact qualification and runtime smoke tests are adjacent primitives, not a base/adapter/export equivalence check | validate base/adapter/GGUF: template/tokenizer/BOS/EOS/vocab/revision/tensor layout/known-answer diff | M–L | M | Adapter support/load path; canonical model metadata across HF and GGUF |
| Adapter import and controlled switching | Important support | PEFT, vLLM, Ollama adapters | Absent; existing runtime “adapter” classes are backend adapters, not PEFT/LoRA support | Add immutable adapter alias, validate base, enable/disable, compare; no throughput promises | L | M | C++ engine adapter support, MoE layouts, quantized merge/application semantics |
| Fine-tune integration | Integrate, do not own | TRL, PEFT, Unsloth, Axolotl, LLaMA-Factory | Lab code only, not a product trainer | Import artifacts/logs; export feedback/eval dataset; optional callbacks that run Clozn suites | M–L | M | Multiple schemas and training frameworks; avoid trainer scope creep |
| Quantization experiment | High | llama.cpp perplexity/KL scripts and community tools | quant-check implements a teacher-forced two-GGUF, ten-prompt, one-step argmax/dependence receipt; live two-engine smoke is deferred and from-log does not reconstruct active dials | Named quant variants; task score, top-token changes, approximate/exact drift, size, load, speed, context | M | M | Exact KLD needs full logits or engine divergence path; BF16 reference may not fit locally |
| Steering authoring and safety sweep | Differentiator if rigorous | RepE/steering toolkits; prompt baselines | Dial derivation/calibration scripts and qualified runtime steering | Sweep layer/strength; compare prompt-only baseline; target and guard suites; publish safe envelope | M–L | L | Model-specific artifacts, collapse regions, adversarial source data, domain differences |
| HF artifact manifest and publishing | Important | HF Hub/model cards/Community Evals/EEE | Pull support; no complete publish flow | Personalized model = immutable manifest plus eval/receipt bundle; export EEE/evaluation card; link weights on HF | M | M | License/private-data lineage and schema mapping |
| LightEval/Inspect/Promptfoo integration | Important | These tools already own task libraries/runners | Text-only cases may connect indirectly through OpenAI Chat Completions; there is no explicit provider/import adapter, per-sample result ingestion, or tools/structured-output parity | Clozn as provider and receipt source; import their per-sample results; deep-link failures | M | M | Match provider/tool semantics and avoid duplicate execution |
| Training tracker links | Secondary | Trackio, W&B, MLflow | Absent | Link checkpoint/run IDs and metrics to behavioral experiment, not duplicate training charts | S–M | S–M | External IDs/auth optional; local-first default |
| Headless CI gate | Must excel for “model CI” | Promptfoo/Inspect/pytest/GitHub Actions | test and test-model can return pass/fail for narrow existing checks; no experiment-level baseline budgets, regression thresholds, or portable gate artifact | ci check EXPERIMENT: deterministic exit code, allowed deltas/budgets, baseline artifact, machine-readable report, worst-case links | M | S–M | Stochastic tests, baseline updates, flaky hardware/performance thresholds, secrets/private cases |
| Shareable evidence bundle | Must excel | Model cards and observability exports are fragmented | A proprietary receipt_bundle.v1 JSON/Markdown export exists for one run; it is not an experiment/HF/EEE/OTel/Inspect bundle and omits several immutable artifact/tool/dependency fields | Self-contained manifest, summaries, per-case records, redacted prompts, artifact hashes, optional tensors | M | M | Size/privacy policy; stable schema versioning |

### Current model-developer UX and canonical priority

| Feature | Current UX today | Canonical priority |
|---|---|---|
| Experiment object and suite runner | Separate test, eval, test-model, quant-check, and one-run Experiment surfaces; no shared suite/variant object | Must excel |
| Real-run-to-test promotion | Runs can be inspected/exported, but cannot be promoted through a guided redact/edit/freeze flow | Must excel |
| Paired variant comparison | Replay/model diff/quant-check provide isolated comparisons; no case × variant matrix | Must excel |
| Target gain plus regression guard | Narrow assertions and outcome metrics are separate commands; no target-plus-guard summary | Must excel |
| Reproduction receipt | inspect/export provides a single-run bundle with incomplete immutable artifact identity | Must excel |
| Deployment-equivalence validation | qualify-whitebox, smoke, and doctor are adjacent commands; there is no trainer-to-GGUF equivalence report | Must excel |
| Adapter import and controlled switching | No model-adapter UX | Adequate after the core experiment loop |
| Fine-tune integration | Lab code exists, but there is no supported trainer/log/artifact import workflow | Adequate; integrate, do not own |
| Quantization experiment | quant-check is a specialized CLI receipt, not a task/quality/speed Pareto experiment | Adequate, with strong developer value |
| Steering authoring and safety sweep | Internal calibration scripts and Studio dial/patch surfaces exist; no held-out target/guard sweep workflow | Adequate initially; must be rigorous for any artifact Clozn publishes |
| HF artifact manifest and publishing | pull works; no manifest authoring/publish/evaluation-card UX | Must excel for evidence export, adequate for marketplace breadth |
| LightEval/Inspect/Promptfoo integration | Text-only tools can call the OpenAI endpoint; no explicit adapter/import/deep link | Adequate |
| Training tracker links | No UX | Adequate |
| Headless CI gate | Narrow commands can exit pass/fail; no experiment gate, baseline budget, or portable report | Must excel to claim “model CI” |
| Shareable evidence bundle | One-run JSON/Markdown receipt bundle only | Must excel |

## 3.3 Researcher features

| Feature | Excellence bar | Existing solutions | Current backend/support | Target experience | Backend effort | UX effort | Blocker or research need |
|---|---|---|---|---|---:|---:|---|
| Versioned hook/capture ABI | Must excel | TransformerLens and NNsight hooks | Engine activation read/layer sweeps; no broad public stable ABI | Named architecture-aware hooks, layer/position/step filters, capture budget and stats | L | M | Hook semantics across model families; storage and observer overhead |
| Versioned intervention ABI | Must excel | TransformerLens, NNsight, pyvene | State/intervene routes and steering writes | Serialized patch/ablate/clamp/add/swap manifest replayable across an experiment | L | M | Backend support by tensor location and batch branch semantics |
| Baseline/control/sweep runner | Must excel | Bespoke notebooks/circuit-tracer | Causal-trace arms and scripts exist | One spec runs baseline, random/null, layer/position/strength, target/collateral measures | M after ABI | M | Controls must be method-specific; cost planning |
| Deterministic/qualified replay | Must excel | Notebook scripts vary | Checkpoint save/restore and lineage exist | Report bit-identical, statistically equivalent, or non-reproducible with reason | M | S–M | GPU kernels, sampler and cross-backend differences |
| Open tensors and manifest export | Must excel | PT/NPZ/HF/Neuronpedia fragmented | JSON/trace exports in pieces | JSON manifest plus Arrow/NPZ/Zarr where appropriate; generated Python/Colab | M | M | Versioning, size, privacy, dependency choices |
| J-lens application | High P1 | Anthropic repo, Neuronpedia | J-lens runtime path exists for qualified sidecar; Qwen 2.5 scope | Import pre-fitted artifact; layer × token ranks; pin/swap/steer; clear “disposed to verbalize” label | M | M | Artifact format, Qwen 3.5 qualification, quantized activation alignment |
| J-lens fitting | Lower P2/P3 | Anthropic reference, PyTorch | Lab/research work, not product | Send-to-lab job; fit from 100–1,000 prompts; import artifact and validation report | L–XL | M | Backward Jacobians/autograd; compute and reference implementation maintenance |
| SAE artifact application | High P1 | SAELens, Qwen-Scope, Neuronpedia | Optional Qwen-specific SAE readout; no general artifact contract | Import HF/SAELens/Qwen-Scope artifact; activation, reconstruction, top examples, causal validation | M–L | L | Artifact schemas, dictionary scale, quantized alignment, metric instability |
| SAE/CLT training | Deprioritize | SAELens, Qwen-Scope, CLT-Forge | Bespoke lab scripts | External workflow only; import resulting artifact | XL | L | Compute, data, evolving practices; no product advantage |
| Probe training/evaluation | High | sklearn, TransformerLens, NNsight | Probe/readout pieces | Fit from labeled receipts with CV, calibration, layer scan, baselines, causal follow-up | M | M | Observational probes can be topic detectors; enforce held-out/causal labels |
| Current layer-position causal trace | Valuable, narrow | Activation patching tools | Shipped CLI/scripts; 91.7% token-flip prediction in measured battery | Journal-driven batch trace, stronger controls, Studio token panel, costs and caveats visible | M | M | Second model family; answer-vs-token evaluation; attention comparison |
| Head/source-position trace | Must for stronger claims | TransformerLens/NNsight research scripts | Absent | Trace attention head, source position, MLP/component sites and validate route hypotheses | L–XL | L | Engine hook granularity, combinatorial search, causal controls |
| Circuit-tracer import/export | Integrate | circuit-tracer/Neuronpedia | Absent | Attach a graph to original run and its validation arms; export study to PyTorch backend | L | M | Mapping quantized fast run to reference checkpoint/hook graph |
| TransformerLens/NNsight bridge | High P2 | Both are mature ecosystems | Absent | Export exact prompt/artifacts; generated study code; import resulting interventions/metrics | L | M | Cross-backend numerical equivalence and hook mapping |
| Neuronpedia/HF share | Medium | Existing public artifact catalogs | Some data fetch scripts; no publish flow | Deep-link/import artifacts; publish selected result bundle rather than rebuild catalog | M | M | API/schema and private/self-hosted modes |
| Python SDK / notebook API | Must excel | TransformerLens, NNsight, SAELens expose Python-first workflows | Internal Python clients/modules exist but no stable typed researcher SDK or supported notebook contract | Every capture/intervention/experiment action available through a versioned Python API; generated notebook reproduces a receipt | M–L | S–M | API versioning, async/streaming ergonomics, large tensor transport, local/sidecar parity |
| Statistical rigor layer | Must excel | Scientific Python, Inspect task repeats, bespoke notebooks | Some causal scripts include controls, batteries, bootstrap CIs, and strength tiers; no general experiment-level statistics contract | Replicates/seeds, effect sizes, intervals, multiple-comparison handling, power/cost estimate, predeclared primary metric and null | M | M | Method-specific assumptions; avoid a one-size-fits-all significance badge |
| Research evidence UI | Must excel | Neuronpedia/NNterp/Workbench | MRI/Scope/Atlas prototypes | Every panel exposes evidence type, artifact, method, fidelity, thresholds, control and validation state | M | L | Requires disciplined design system, not just visualization polish |

### Current researcher UX and canonical priority

| Feature | Current UX today | Canonical priority |
|---|---|---|
| Versioned hook/capture ABI | Internal engine routes and clients expose activation reads; no stable supported hook vocabulary, capture budget, or public compatibility report | Must excel |
| Versioned intervention ABI | Internal state/intervene routes and steering code exist; no single serialized public intervention manifest | Must excel |
| Baseline/control/sweep runner | causal-trace and research scripts run specific controlled arms; no general experiment spec across methods | Must excel |
| Deterministic/qualified replay | Branch/restore primitives and lineage exist; UX does not label bit-identical versus re-prefilled/statistically equivalent replay | Must excel |
| Open tensors and manifest export | JSON/trace pieces exist; no standard large-tensor bundle or generated notebook | Must excel |
| J-lens application | Qualified sidecar/readout paths and Studio/Explain concepts exist for a narrow model setup; no current general artifact-import flow | Adequate breadth; must be rigorous for each qualified artifact |
| J-lens fitting | Research/lab work only; no supported job UX | Deprioritize |
| SAE artifact application | A Qwen-specific --sae/readout path and concepts UI exist without a qualified general SAE contract | Adequate breadth; must be rigorous for each qualified artifact |
| SAE/CLT training | Bespoke research scripts only | Deprioritize / external integration |
| Probe training/evaluation | Probe/readout code exists; no supported fit/CV/calibration/causal-follow-up flow | Adequate |
| Current layer-position causal trace | clozn causal-trace exposes a narrow CLI result; batteries are scripts and the Studio token panel is still open | Must excel within its stated layer-position/token-flip scope |
| Head/source-position trace | No UX | Deprioritize until hook granularity research lands |
| Circuit-tracer import/export | No UX | Adequate integration; native reimplementation is a non-goal near term |
| TransformerLens/NNsight bridge | No UX | Adequate |
| Neuronpedia/HF share | Data-fetch scripts only; no publish/deep-link flow | Adequate |
| Python SDK / notebook API | Internal clients/modules are importable but unsupported as a stable researcher contract | Must excel |
| Statistical rigor layer | Some scripts show controls/CIs/strength tiers; no common experiment report | Must excel |
| Research evidence UI | Scope/Atlas/Read/MRI prototypes exist; method/artifact/control/validation metadata is not uniformly foregrounded | Must excel for correctness/provenance; visual polish adequate |

# 4. Priorities by audience

## 4.1 Model users

### Be excellent at

- Drop-in compatibility and behavioral parity
- Exact run/context/memory receipts
- Plain-language performance and protocol diagnosis
- Honest uncertainty channels and task-calibrated risk where available
- One-step corrective retry, compare, undo, and save-as-profile
- Verifiable local-only operation
- CLI/API value without Studio

### Be adequate at

- Model lifecycle breadth
- Lightweight run history/search
- Basic multi-model/quant comparison
- Explicit memory CRUD and scoping
- Integrations/setup helpers

### Deprioritize

- Internal activation views
- Artifact training
- Fine-tuning and publication
- Dense experiment controls
- General chat/RAG/agent-app features

## 4.2 Model developers

### Be excellent at

- Reproducible receipts
- Real-case behavioral suites
- Paired variant comparison
- Target improvement and side-effect/regression evaluation
- Deployment-equivalence validation
- Failure drill-down from outcome to context/token/internal evidence
- Shareable HF-compatible evidence

### Be adequate at

- Adapter import/switching
- Tracker and eval-tool integration
- Quantization comparison
- Steering sweeps
- Checkpoint orchestration

### Deprioritize

- Owning the training stack
- Dataset annotation/generation
- High-throughput serving
- A proprietary benchmark marketplace
- Full experiment-tracking replacement

## 4.3 Model researchers

### Be excellent at

- Provenance, replay qualification, controls, and open exports
- Stable hook/intervention contracts
- Batch sweep and comparison
- Applying external artifacts to real runs
- Visible epistemic labels and method limitations
- CLI/Python automation

### Be adequate at

- Studio navigation and visual polish
- J-lens/SAE/probe application for a small qualified set
- TransformerLens/NNsight/Neuronpedia bridges
- Research-backend job handoff

### Deprioritize

- Training frontier-scale dictionaries
- Replacing TransformerLens/NNsight
- Hosting a public feature catalog
- Remote cluster infrastructure
- Supporting every model architecture before one path is rigorous

# 5. Recommended delivery sequence

## Gate 0: define the product contract

Before adding more panels, adopt these invariants:

1. All public compatibility requests use one instrumented execution path.
2. Unsupported behavior is rejected explicitly; no important field is silently ignored.
3. Every accepted run has a stable ID and receipt.
4. Delivered, survived, influenced, supported, probable, and calibrated are separate labels.
5. White-box features are artifact-qualified, never inferred from model family alone.
6. Every steering action is reversible and records a before/after comparison.
7. Research visuals identify method, artifact, controls, and limits.

## Phase 1: no-switch foundation

**Goal:** an Ollama/OpenAI user can point a real app at Clozn without losing core behavior.

Ship:

- Ollama NDJSON streaming through the instrumented path
- Correct cancellation/finalization/finish reasons
- Tool calls and tool results for a deliberately small qualified model set
- Structured JSON/schema output
- Stop/options/context semantics and thinking fields
- More complete models/show/ps capability metadata
- Stable run-ID mapping for streaming and non-streaming
- Real-client conformance tests with Ollama Python/JS, OpenAI SDK, Open WebUI, and at least one coding harness
- A connect/config helper with backups and a compatibility health report

Do not claim Ollama compatibility broadly until the conformance matrix is green. The current Ollama streaming 501 is a release-blocker for this promise.

## Phase 2: receipts and repair without Studio

**Goal:** Clozn provides value during normal use and resolves the first concrete pain.

Ship:

- clozn watch and latest-by-client/session
- clozn context/explain last with delivered/survived/influenced sections
- Exact truncation, memory, tool-result, template, context, and timing accounting
- Plain-language “why slow / cut off / looped / ignored context” diagnosis
- Honest token-surprise spans and alternatives
- Corrective retries: less verbose, more concrete, grounded, ask-before-guessing, lower creativity
- Once/session/profile scope, diff, undo, and compare
- Explicit memory used/omitted/token-cost/scoping controls
- local-only/outbound ledger
- Optional OpenTelemetry/OpenInference export with prompt-content controls

This is the first version with a strong mainstream positioning story.

## Phase 3: model CI

**Goal:** turn existing backend pieces into the developer wedge.

Ship:

- A versioned experiment/suite manifest
- Promote captured runs to editable/redactable cases
- Paired case × variant execution and Studio matrix
- Target suite plus guard suite
- Repeated seeds/samples, caching, metric plugins, and failure drill-down
- Deployment-equivalence validation for GGUF/template/tokenizer first
- Quantization comparison with clear approximate-versus-exact divergence labels
- LightEval/Inspect/Promptfoo provider/import adapters
- HF Community Evals/EEE/evaluation-card export
- Shareable evidence bundle

Then add adapter loading/validation and tracker integration when the base experiment loop is coherent.

## Phase 4: qualified white-box lab

**Goal:** make one researcher path rigorous before expanding breadth.

Recommended first path:

1. Qwen 3.5 9B reference checkpoint plus the current GGUF quantization.
2. Versioned residual hook and intervention ABI.
3. Quantized-versus-reference activation qualification.
4. Import and validate a Qwen-Scope SAE.
5. Import and validate a pre-fitted J-lens.
6. One baseline/control/sweep experiment spec.
7. Export tensors/manifest/generated Python.
8. Research evidence panel with method/coverage/validation labels.

Then pursue head/source-position tracing and a second model family. Defer SAE training and native full circuit tracing.

# 6. UX architecture

## 6.1 One data model, three home views

Do not make three disconnected products. Use a shared object model:

~~~text
run
  context sources
  output/token trace
  timing/resources
  behavior profile
  receipt
  lineage

experiment
  cases
  variants
  repetitions
  metrics
  target and guard suites
  run IDs

artifact
  model/revision compatibility
  hook/layer/dimensions
  provenance
  qualification
  evidence envelope
~~~

Then provide three entry views:

- **Run view** for users: What happened? What deserves scrutiny? What can I try?
- **Experiment view** for developers: Which variant won, what regressed, and can I reproduce it?
- **Evidence view** for researchers: Which measurement/intervention supports the hypothesis, under what controls?

## 6.2 Where probability colors render

ANSI heat colors are a terminal rendering feature. Arbitrary OpenAI/Ollama clients will normally render only clean response text, and should not receive escape codes or embedded markup.

Target rendering surfaces:

- Terminal color in the existing clozn run --heat path and a proposed clozn watch command
- A linked token ribbon in Studio
- Optional browser/editor extensions that explicitly consume a Clozn run ID
- Structured trace APIs for custom clients
- No modified answer text by default

This answers the earlier product question: compatibility captures the run; a sidecar renders Clozn’s additional evidence.

## 6.3 Studio design principles

- Lead with a question and decision, not a sensor.
- Show the smallest useful summary first.
- Make “not available” and “not qualified” normal, designed states.
- Put evidence provenance beside the visual, not in a hidden settings panel.
- Use the MRI metaphor only where the underlying measurement is real.
- Let every Studio action copy its CLI/API equivalent.
- Make compare/undo mandatory for steering.
- Do not require users to understand layers, residuals, or entropy to resolve a mainstream run.

# 7. Explicit boundaries, deferrals, and claim guardrails

## 7.1 Permanent product scope boundaries

- Do not replace Ollama’s model catalog, cloud, hardware breadth, or raw inference-optimization race.
- Do not replace Open WebUI or LM Studio as a general chat, document, RAG, voice, image, or agent application.
- Do not build a universal agent orchestrator, hosted model marketplace, or high-throughput multi-tenant serving platform.
- Do not make proxying an external Ollama server the primary architecture. It may be a migration/observability bridge; deep Clozn evidence requires Clozn to run the model.
- Do not own distributed/full-model training, general data labeling/scraping/synthetic-data generation, or another comprehensive trainer UI.
- Do not replace Hugging Face Hub, W&B, Trackio, LightEval, Inspect, TransformerLens, NNsight/NDIF, SAELens, or Neuronpedia. Integrate with them.
- Do not create a proprietary universal benchmark leaderboard or public feature/model catalog.
- Do not train frontier-scale SAEs/CLTs or build an NDIF-like remote GPU fabric as a core product.
- Do not support closed-model internals where weights and activations are unavailable.
- Do not make Studio a mandatory chat app or notebook replacement.

## 7.2 Sequencing deferrals, not permanent non-goals

- Enterprise remote auth, billing, collaboration, and shared tenancy follow excellent local single-user value.
- Anthropic compatibility follows credible Ollama/OpenAI parity unless coding-agent evidence makes it urgent.
- Broad vision/multimodal and managed embeddings follow excellent text, streaming, tools, and context receipts.
- Broad adapter/model-family coverage follows one rigorous developer artifact path.
- Head/source-position tracing follows a stable residual hook/intervention ABI and current trace hardening.
- J-lens fitting, SAE training, and native circuit-tracer breadth follow reliable application/import/export of existing artifacts.

## 7.3 Epistemic and product-claim guardrails

- Never guarantee factual correctness or equate token probability with truth.
- Never claim that supplied context governed an answer without an appropriately controlled influence measurement.
- Never call readouts “thoughts,” “beliefs,” factual knowledge, or automatic circuit discovery.
- Never treat attention, an SAE label, a natural-language activation explanation, or a high-accuracy observational probe as causal attribution by itself.
- Never imply universal white-box support from a model-family name; require exact artifact/runtime qualification.
- Never promise that an adapter, personalized model, memory, or steering action has no regressions.
- Never apply or publish automatic steering without a held-out target and side-effect suite.
- Never silently fall back to cloud or silently persist a supposedly personal memory.
- Never require Studio for normal model-user value.
- Never inject terminal colors or proprietary markup into arbitrary third-party answers.

## 7.4 Research ideas explicitly not to revive without new evidence

- Semantic temperature
- Prospective collapse gauge from live energy
- Branch-on-doubt
- Paraphrase-brittleness receipt
- Same-model verify-then-branch as a hard-tail reliability fix
- Null-space watermarking
- Scalar self-reported confidence
- J-transport as evidence of improved steering direction quality

# 8. Decision rubric for future backend capabilities

Every proposed capability should answer:

| Question | Why it matters |
|---|---|
| Who experiences the problem: user, developer, researcher? | Prevents a researcher-only feature from hijacking mainstream UX |
| What decision does the signal/action enable? | Prevents decorative telemetry |
| Can the user receive value without changing apps? | Protects the adoption contract |
| Is the claim exact, observational, calibrated, or causal? | Determines language, controls, and UI treatment |
| What is the simplest existing external tool to integrate? | Avoids rebuilding mature ecosystems |
| Does it require Clozn to run the model? | Separates real differentiation from proxy-level telemetry |
| Which model/artifact/runtime identities qualify it? | Prevents false universality |
| What is the null/control and the held-out side-effect test? | Makes steering and interpretability scientifically credible |
| How is it reproduced and exported? | Makes developer/research value portable |
| What is the failure state? | Ensures unavailable/unsupported results are honest and useful |

# 9. Product success metrics

## Model users

- Percentage of target apps passing compatibility conformance
- Time from “odd response” to a useful diagnosis
- Percentage of runs with exact truncation/context accounting
- Corrective retries kept versus undone
- Studio-open rate after an issue, not as a vanity engagement target
- Zero silent fallback/unsupported-field incidents

## Model developers

- Time to create a suite from real runs
- Percentage of experiments with complete reproduction fields
- Regressions found before artifact publication
- Percentage of aggregate failures with a useful case-level explanation
- Evidence bundles exported to HF/standard formats
- Deployment mismatch catches

## Researchers

- Reproduction success from exported bundle
- Percentage of claims with an explicit null/control
- Artifact qualification coverage and failure transparency
- Agreement between fast GGUF and reference-backend readouts
- Time from a real run to a completed intervention sweep
- External notebooks/tools successfully round-tripped

# 10. Bottom line

Clozn already contains three unusually valuable assets:

1. A real instrumented local execution path.
2. Receipts, replay, counterfactuals, and experiment primitives that can answer behavioral questions.
3. A serious research substrate with enough negative results to support honest product boundaries.

What is missing is not another large backend invention. It is a product hierarchy:

- **Compatibility earns access to the user’s real workflow.**
- **Receipts and repair earn daily trust.**
- **Model CI earns a differentiated developer use case.**
- **Qualified causal tools create the long-term research moat.**

The recommended positioning is therefore:

> **Clozn makes local-model behavior inspectable, correctable, and reproducible—inside the tools people already use.**

For model users, that means “what happened and what should I try?”

For model developers, it means “what improved, what broke, and can I prove it?”

For model researchers, it means “what mechanism is the evidence consistent with, and does an intervention validate it?”

That is a coherent product rather than three unrelated audiences. The depth changes; the run, receipt, experiment, and evidence spine stays the same.

# Appendix A: repository evidence reviewed

- [README.md](../README.md)
- [ARCHITECTURE.md](./ARCHITECTURE.md)
- [TECHNICAL.md](./TECHNICAL.md)
- [RUNTIME_SPLIT.md](./RUNTIME_SPLIT.md)
- [MODEL_SUPPORT.md](./MODEL_SUPPORT.md)
- [OPENAI_COMPATIBILITY.md](./OPENAI_COMPATIBILITY.md)
- [EXPLAIN_THIS_ANSWER_SPEC.md](./EXPLAIN_THIS_ANSWER_SPEC.md)
- [STUDIO.md](./STUDIO.md)
- [BACKLOG.md](./BACKLOG.md)
- [RESEARCH_ROADMAP.md](./RESEARCH_ROADMAP.md)
- [scripts/tracer/README.md](../scripts/tracer/README.md)
- Public CLI parser and command implementations under ../clozn/cli/
- Gateway, route, substrate, receipt, run, memory, steering, analysis, and artifact modules under ../clozn/
- Studio implementation under ../studio/heavn/
- Current test suite under ../tests/

# Appendix B: evidence cautions

- Official documentation establishes current product capabilities, not how frequently users need each one.
- GitHub issues and Reddit discussions establish concrete failure modes, not prevalence.
- Clozn research numbers describe the documented model/task/artifact setups and must not be generalized without replication.
- External research is moving quickly. In particular, J-lens, SAE evaluation, activation steering, and circuit-tracing practices should be re-audited before implementation milestones.
- The current working branch includes unmerged compatibility/UI work. Statements about the Ollama route refer to b6190b8, not origin/main at 1e7ccb5.
