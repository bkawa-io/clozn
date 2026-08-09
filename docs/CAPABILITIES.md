# Clozn capability and release matrix

This is the single current-user capability summary. Feature documents should link here instead of
copying status tables.

The status words are deliberately distinct:

- **Merged** means the implementation is in this source tree and has automated evidence.
- **Released** means users can obtain it from a tagged public package and, where required, matching
  engine artifacts. A version string in source is not a release.
- **Qualified** means the named model, artifact, or integration passed its applicable evidence gate.
  Qualification never generalizes to an untested model or artifact.

There is no tagged public release or published engine artifact matrix in this snapshot. Consequently,
the release column is **Unreleased** even where support is merged and tested. See
[native distribution status](NATIVE_DISTRIBUTION.md).

Managed-runtime note: the Python gateway now runs in the `clozn serve` supervisor process. The only
remaining runtime subprocesses are the private C++ model workers; the legacy gateway subprocess and
projection-file handoff are retired.

| Capability | Core support | CLI / API | Studio | Qualification | Released version | Limitations |
|---|---|---|---|---|---|---|
| Autoregressive GGUF runtime | Merged: supervised C++ worker, strict gateway, journaling, trace, scoring, steering | `clozn run`, `serve`, `smoke`; OpenAI subset and native stream | Runtime state, runs, Lens, Model | Wave 1 records deep CPU core passes for five exact model rows; see [model support](MODEL_SUPPORT.md) | Unreleased | A loadable GGUF and usable embedded chat template are still required; optional white-box artifacts have separate gates |
| Managed engine setup | Merged: manifest selection, verified download, transactional promotion and rollback | `clozn setup status|upgrade|rollback`; `clozn doctor` | None | Model-free installer and clean-room source-snapshot tests | Unreleased | No public manifest/archive matrix exists yet, so this is not a working public install path |
| Base, fine-tune, and quant comparison | Merged: teacher-forced token-difference ladder and target/guard experiment artifacts | `clozn diff-model`, `quant-check`, `experiment`, `ci` | Experiment matrix and cell detail | Worked Qwen reasoning-SFT case plus model-free contract tests | Unreleased | A sampled screen is not proof of quality; tokenizer/template mismatch fails or is labeled |
| GitHub Action model gate | Merged source package: engine-free verify path, trusted run orchestration, SHA-keyed/cache-qualified downloads, metadata-only indexed receipt bundle, job summary/JUnit, exact exit propagation, and read-only PR degradation | Publication-ready `integrations/github-action/action.yml`; core `clozn ci check`, `model-lock fetch` | None | Model-free Action/security/cache/receipt tests in this tree | Unreleased | `bkawa-io/clozn-action`, immutable semver tags, moving `v1`, Marketplace publication, and public Clozn/engine compatibility pins do not exist yet; run mode is restricted to trusted non-PR events |
| LoRA loading and comparison | Merged: one fail-closed LoRA attachment, run identity, and three-arm merged-export receipt | `clozn serve --adapter`; `clozn diff-adapter`; `clozn adapter validate`; `clozn validate-export` | Adapter identity appears in comparison provenance | Attachment and scale-zero proof recorded in [adapter support](ENGINE_ADAPTER_SUPPORT.md); merged-export gate has model-free bad-merge coverage but still needs a known live merged export | Unreleased | LoRA GGUF only; PEFT/Safetensors conversion stays external; one adapter at a time; no hot swap |
| Context receipt | Merged: versioned delivered/assembled/omitted segments, exact rendered hash/size/token facts, raw termination, stable client source IDs, and privacy tiers | `clozn context last|show|export`; `GET/POST /runs/<id>/context-receipt` | Compact and detailed Lens views | Schema, endpoint, CLI, and model-free Studio tests | Unreleased | Describes delivered context; it does not by itself prove that a source caused output |
| Measured source support | Merged: bounded source influence measurement, receipt-linked source identity, identity-bound cache, privacy-safe portable export, and explicit absence/evidence states | `clozn provenance`; influence-map routes including `/export` | Sources lens renders measured, below-floor, omitted, and unavailable states | Research batteries cover exact Qwen2.5 and Llama3.1 rows; not a universal model qualification | Unreleased | The dedicated attention-knockout provenance command needs a worker started without flash attention; the Sources lens does not silently change worker configuration |
| Run investigation synthesis | Merged: read-only, versioned composition of context delivery, stored influence, deterministic diagnosis, structural comparison, and corrective-action availability | `GET /runs/<id>/investigation` | Lens received-context view plus Ask Another Question routing to real evidence/actions | Model-free schema, route, and Studio tests | Unreleased | The GET never starts model work; unavailable measurements are returned as typed action descriptors, and artifact-native evidence states remain distinct |
| "Why this?" influence query | Merged: read-only projection over an already-persisted context<->answer influence map, scoped to a caller-selected half-open Unicode-code-point range of the recorded answer -- overlapping measured links only, `effect`/`delta_nats`/`abs_delta_nats`/`clears_floor`/`evidence_state` preserved exactly, never recomputed. Backend contract only; no Studio surface yet | `GET /runs/<id>/influence-query?start=&end=&limit=` | None yet -- this PR ends at the backend contract | Model-free domain and route tests (answer drift, redaction, below-floor-only, ranking/limit, no-engine-access) | Unreleased | Never starts a measurement -- a run with no persisted influence map returns a typed `not_measured` result, not an empty "nothing mattered" list; `privacy=full` is not supported yet |
| Context tension detector | Merged: read-only detector over the same persisted influence map, built on the shared `clozn.runs.influence_geometry` gate/geometry primitives "Why this?" also uses. A tension record requires two DISTINCT context spans with `causally_supported` links to the SAME answer span and opposite `effect` values -- never inferred from semantic similarity, source order, or textual overlap, and never labeled "conflict"/"contradiction" (that is measured opposing pressure, not a claim the source texts disagree). Backend contract only; no Studio surface yet | `GET /runs/<id>/context-tension[?start=&end=][&limit=]` (whole-answer when start/end are omitted) | None yet -- this PR ends at the backend contract | Model-free domain and route tests (Cartesian opposing pairs, same-answer-span requirement, stale/redacted fail-closed, deterministic `tension_id`, no-engine-access) | Unreleased | High-precision by design: below-floor `observed` links and `neutral` links never produce a tension record, and an empty `tensions` list is never narrated as "the context was consistent" |
| Context utilization coverage | Merged: read-only, source-level coverage view over the same persisted influence map -- for every prompt source, whether it was `clear_measured_effect`, `below_measured_floor`, or `not_measured` (present in the assembled prompt but outside the bounded attribution selection). Classification uses ONLY coarse prompt spans (never fine/refined descendants, which exist only for the strongest sources and would otherwise bias the comparison); `below_measured_floor` additionally requires the measurement to be structurally complete (`matrix_complete` and `selection.complete_for_selected_spans` both true), and any selection/`selected`-flag inconsistency fails closed with a contract error rather than guessing. Named "utilization", not "dead context": `not_measured` is never described as low-effect. Backend contract only; no Studio surface yet | `GET /runs/<id>/context-utilization` (no query params -- returns full coverage, never paginated) | None yet -- this PR ends at the backend contract | Model-free domain and route tests (fine-span exclusion, refined-source no-double-count, incomplete-matrix fail-closed, selection-consistency fail-closed, deterministic ordering, no-engine-access) | Unreleased | No `utilization_percent`/`importance_score`/token-savings claim -- `max_abs_delta_nats` is descriptive sort metadata only; a source below the measurement floor on this one recorded answer is not automatically safe to remove from a future prompt |
| Run comparison and replay planning | Merged: versioned run diff across identity, context, settings, and output; `--replay --execute` runs available swaps as bounded controlled tests | `clozn compare-runs`; `GET /runs/compare`; `POST /runs/compare/test` | Compare workspace | Schema and model-free route/CLI tests | Unreleased | `--replay` remains model-free planning; `--replay --execute` requires a running gateway and preserves child-run evidence |
| Durable checkpoint snapshots | Merged: content-addressed pin store, preview-before-write pinning, restart-safe export/import, explicit pin hydration for exact-fork capture, and typed unpin dependency checks | `clozn snapshot pin|list|unpin`; `POST /runs/<id>/snapshot/pin`; `POST /runs/<id>/execution-fork/checkpoint` with explicit `{"pinned":true}`; `GET /snapshots`; `POST /snapshots/<run-id>/unpin` | Snapshots panel with storage preview, explicit pin confirmation, manifest facts, and two-step unpin | Schema, route, store, capture, and Studio tests; live engine 22/22 plus gateway pin/restart battery 17/17 | Unreleased | Studio exposes manifest metadata only; checkpoint bytes stay in the local blob store; pin hydration is fail-closed on blob, identity, generation, and token-history mismatch; child-dependent pins require an explicit cascade decision |
| Answer Time Machine eligibility | Merged: read-only, versioned replay-fidelity and per-turn branch-eligibility receipt; explicit latest-turn and exact matching session-turn prompt-boundary verification | `GET /runs/<id>/time-machine`; `POST /runs/<id>/branch`; `POST /runs/<id>/time-machine/verify`; `POST /runs/<id>/time-machine/branch` | Lens card shows structural replay, exact same-prompt child replay, turn selection, optional replacement question, exact verification, and historical source-run provenance | Model-free route, branch schema/fixtures, Studio tests, live three-turn session proof, and durable-pin hydration coverage | Unreleased | The legacy branch action remains `structurally_reproducible` transcript replay. Exact child replay restores a verified prompt-boundary checkpoint and requires an unchanged control; alternate-question branching remains structural |
| Reversible corrective retry | Merged: request-local, counterfactual debugging tool -- a matched greedy baseline/corrected comparison and a keep that only ever selects the corrected child as that one run's own revision. No session/profile scope, no persisted policy, no undo transaction that reaches beyond the run it was generated from | `clozn retry`; `/runs/<id>/retry`; `/runs/<id>/corrective-actions` preview/confirm/keep | Behavior's one-shot retries module and run comparison surfaces | Model-free comparison and revision-selection tests | Unreleased | Requires a ready worker to generate the comparison; unavailable dial backends are labeled; leaves no standing behavior change for a later, unrelated request |
| Ollama model adoption | Merged: exact manifest-layer resolution, fidelity classification, dry-run/guided try, hard-link/copy, optional core smoke, and undo -- scoped entirely to Clozn's own model directory | `clozn adopt ollama --try --yes [--qualify]` | None | Model-free resolver/adoption/qualification tests; a live adopted model still determines whether `--qualify` passes | Unreleased | Does not mutate or stop Ollama; hard links require a compatible filesystem; does not configure any other application (see Removed/Retired) |
| Performance diagnosis | Merged: versioned worker timings for load/startup, template, tokenize, context/KV creation, prefill, and decode; gateway timings for queue, dispatch, serialization, and stream flush; separate clock owners; known/unaccounted aggregation; evidence-gated regression attribution | `clozn diagnose --performance`; `GET /runs/<id>/performance` | Measured phase breakdown, known/unaccounted time, throughput provenance, and rule evidence | Model-free protocol/old-worker/cold-warm/long-prompt/cancellation/CPU-vs-GPU tests plus a clean CPU worker compile | Unreleased | Cross-process offsets are never aligned; overlapping and process-startup spans are shown but excluded from known in-request time. Live GGUF CPU/GPU runs are still required to qualify absolute timing accuracy and model/backend-specific thresholds |
| J-lens readout | Merged apply path with artifact identity and checksum validation | `/jlens`, `/runs/<id>/jlens` | Lens layer view with provenance caption | Exact Qwen2.5-7B Q4_K_M row only | Unreleased | Fit per model; a linear readout is not a transcript of thought or causal proof |
| OpenAI-compatible text API | Merged strict subset of models and Chat Completions | `/v1/models`, `/v1/chat/completions` | Runs recorded through the same gateway | Model-free SDK conformance plus real-runtime smoke gates | Unreleased | Text-only subset; legacy `/v1/completions` returns a typed HTTP 410 retirement response |
| Explicit concept guard intervention | Merged: request-local, opt-in closed-loop disposition guardrail -- J-lens-based concept polling, per-model threshold calibration, fail-closed on an unresolvable calibrated concept, mid-generation counter-injection, and a per-request receipt. No persisted server-wide default; a request that omits `clozn_guard` is byte-identical to the ordinary path | `clozn_guard` request field on `/v1/chat/completions` | None (no persistent guard configuration surface) | Model-free control-loop, calibration-load, and server-wiring tests | Unreleased | Present-tense detect-and-correct only, never predictive/lead-time; incompatible with `stream: true` and with the other `clozn_*` extensions in v1; an uncalibrated concept degrades to annotate-only rather than firing |
| Calibration-evidence annotation | Merged: always-on, metadata-only verdict (`clozn_policy`) reporting the calibrated risk band, score, and answer/ask/abstain thresholds for a completed reply against a saved `clozn eval --save` profile. Read-only -- never rewrites the reply text and never decides production behavior on the caller's behalf | `clozn_policy` response field; `clozn eval`, `clozn eval policy` | None (evidence only; no configuration surface) | Model-free verdict/signal/attachment tests | Unreleased | A calibrated band is a fitted threshold from a labeled set, not a per-answer correctness guarantee; absent when no calibration is saved for the exact model/task |
| Managed multi-model runtime | Merged: qualified preload manifest, exact-identity worker registry/router, per-worker generation concurrency, in-process cold loading with single-flight coalescing, and verified-idle LRU eviction | `clozn serve --models-config`; `GET /readyz`, `GET /runtime/models` | None | Model-free routing/registry/cold-load/soak tests; live two-GGUF default battery plus twenty-cycle cold-load soak in `scripts/smoke/managed_runtime_smoke.py` | Unreleased | The Python gateway runs in the supervisor process; C++ workers remain private subprocesses. Receipt/replay/influence/legacy-fork routes fail closed under a managed gateway; SAE/J-lens cannot be enabled on a managed worker in v1; the resident-worker limit is a hard cap, not advisory |
| Self-serve model qualification | Merged Q1/Q2 model-free plan plus Q3 core receipt and Q4-Q8 fail-closed orchestration primitives | `clozn qualify MODEL --plan`; `clozn qualify MODEL --run` | None | Versioned plan/run/lab-step schemas; exact identity and live Context Receipt smoke are opt-in; lab adapters, resumable batteries, and transactional artifact install/rollback are model-free tested | Unreleased | `--run` proves only core runtime evidence. Dial/J-lens fitting still runs in an explicit external lab command; a successful command is not accepted until its model-bound manifest validates. No Torch/Transformers import in product code |

## Removed and research-only surfaces

- **Durable scoped corrections / Teach Once** were retired: a correction could be drafted, confirmed,
  scoped to a session/client/model/project, and then auto-applied to every future matching request
  until explicitly disabled or deleted (F5/F6: `POST /corrections`, `POST /corrections/<id>/verify`,
  `clozn corrections ...`), and the prompt-first retry route additionally supported persisting a preset
  as a standing session/profile policy (`--scope session|profile`, `/corrective-retries/<id>/undo`).
  Both were removed because Clozn does not own persistent behavioral policy that silently reshapes
  unrelated future requests -- see the causal-debugger positioning in [README.md](../README.md).
  `GET`/`POST /corrections` and `/corrections/*` now return a typed HTTP 410
  (`durable_corrections_retired`) instead of a 404, so a caller can tell "this used to work and no
  longer applies." Studio no longer exposes a Corrections/Teach Once surface; the one-shot corrective
  retry (row above) is unaffected. Runs recorded before the retirement may still carry
  `applied_corrections`/`correction_conflicts` receipt fields or a `corrective_retry.scope` of
  `session`/`profile`; those remain readable as historical evidence and validate against their
  original schemas, but nothing in the product reads or re-applies them anymore.
- **Named behavior profiles** were retired: CLOZN used to let a user bundle steering state into a
  saved persona (`work`, `friend`, ...) with save/switch/export/import/delete, and the switched-to
  name was written onto every subsequent run as `meta.active_profile`. That whole persona lifecycle
  was removed -- CLOZN is a causal debugger, not a persona manager; see the positioning in
  [README.md](../README.md). The primitive underneath a profile switch was always steering itself
  (`/steer/axes`, `/steer/set`, `/steer/check`, custom dials, concept steering), which is completely
  untouched and persists exactly as it did before. `GET`/`POST /profiles/*` now return a typed HTTP
  410 (`profiles_retired`). Existing files under `~/.clozn/profiles/` are left on disk untouched and
  are never read by the product; a stale `active_profile` key in `studio_settings.json` is likewise
  never read or written and has no effect on any generation. Runs recorded before the retirement may
  still carry `meta.active_profile`; that remains readable as historical evidence, but no new run
  writes it.
- **Downstream client configuration management** was retired: `clozn connect` used to safely patch a
  third-party application's config file -- Aider's YAML, Open WebUI's environment, a generic OpenAI
  client's `.env`, an Ollama SDK environment -- with a pre-write backup, sha256-based drift detection,
  and a compare-and-swap undo. `clozn adopt ollama --connect APP` additionally offered to run that same
  mutation as a step after adopting a model, and its dry-run preview scanned for which such apps were
  installed. All of it was removed: Clozn is the debugger; other applications own their own
  configuration. It never wrote a value into an application's file, only ever told the user what value
  to set. Removed: `clozn connect`, the generic `Connector`/`AiderConnector`/`EnvFileConnector`/
  `GenericOpenAIConnector`/`OpenWebUIConnector`/`OllamaSDKConnector` framework, `clozn.connect.
  transaction.v1` transaction state, and `--connect`/`--url`/`--api-key`/`--client-model-label` from
  `clozn adopt ollama`. The two file-integrity primitives that framework also used for handling a
  model blob (`sha256_path`, `atomic_copy_file`) moved to
  [`clozn/cli/commands/_fileops.py`](../clozn/cli/commands/_fileops.py), since `clozn adopt ollama`
  has a genuine, unrelated need for them. Existing `~/.clozn/connect/*.json` transaction files left by
  an older Clozn version are never read again and have zero runtime effect. Old `clozn.adopt-ollama.v1`
  adoption documents that still carry the retired `client_transactions` field remain readable (the
  schema no longer declares that field, but validation stays permissive about unknown fields). Point an
  existing client at Clozn's OpenAI-compatible endpoint yourself -- see
  [CLIENT_CONFORMANCE.md](CLIENT_CONFORMANCE.md) for exact Aider/Open WebUI/SDK values.
- **Broad assistant-behavior policy management** was retired: Clozn used to let a caller persist a
  server-wide concept-guard default (`GET`/`POST /guard/mode`, a `generation_guard` setting read when a
  request omitted `clozn_guard`) and a server-wide or per-request selective-generation ACTION
  (`clozn_selective`, a `selective_generation` setting) that could silently replace a reply's actual
  text with a clarify/abstain message when a calibrated band said not to just answer. Both were removed
  -- Clozn is a causal debugger, not a production policy engine; see the positioning in
  [README.md](../README.md). "Measurement and explicit intervention stay. Ambient policy goes." Kept,
  unchanged: the request-local `clozn_guard` intervention and the always-on `clozn_policy` calibration
  evidence (both rows above), and all guard-calibration/eval research tooling
  (`scripts/calibration/guard_signal_calibrate.py`, `clozn/eval/policy.py`, `clozn eval`). `GET`/`POST
  /guard/mode` no longer exist as routes at all (the module was deleted, not stubbed to 410) -- GET
  falls through to the server's ordinary unknown-route 404, POST to the same generic
  no-active-substrate 409 any other made-up write path gets. A stale `generation_guard` or
  `selective_generation` key left in `studio_settings.json` by an older Clozn build is never read or
  written and has no effect on any generation. A request that still sends `clozn_selective` gets an
  honest HTTP 400 (`unsupported_parameter`) rather than a silent no-op. Runs recorded before the
  retirement may still carry `meta.clozn_selective_action`; that remains readable as historical
  evidence, but no new run writes it, and no new response carries the field.
- Prompt-card and learned-prefix memory were removed from the product on 2026-07-27. Existing run
  readers may still tolerate old `memory` fields, but current runs do not apply or record cards.
- The user-facing PyTorch workbench was retired with that memory path. Offline calibration and research
  scripts are not an alternate serving product.
- Diffusion-language-model work is historical research, not a current product substrate or command.
  [DESIGN.md](DESIGN.md) and [TECHNICAL.md](TECHNICAL.md) preserve that lineage as archives.
