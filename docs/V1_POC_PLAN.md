# Clozn v1 POC plan

Status: implemented on `main` for the scoped code, package, Studio, native worker, live backend, and
manual browser gates. Current qualification evidence is recorded in `docs/V1_LIVE_QUALIFICATION.md`.

Originally prepared 2026-10-12 against `c66cedbd715fe5899a5b2b0977b84902e1793f21`.

**Product promise:** See how context affects an answer, then find a smaller context that preserves it.

**Recommendation:** selectively rebuild the product in this repository. Replace the broad public surface with a small, explicit one; retain and simplify the existing inference, recording, source-mapping, scoring, and exact-verification machinery. Finish with one implementation and one supported product. Restore removed features from Git only when a later product need justifies them.

This is a scoped rebuild, with working checkpoints between changes. A new inference engine, storage technology, frontend framework, plugin system, or general experiment platform is outside this effort.

**1. The complete v1 experience**

1. Start one local model and open Studio through one documented launch path.
2. Create a run from a text prompt, load a working example, or open a run recorded through the local text API.
3. See input and output side by side in Context Effects. Select answer text or a context source to explore measured relationships in either direction, expand either side into a heatmap, summarize influence extremes, or search for Lean Context under fixed probability-loss tolerances.
4. Use Lean Context for a bounded probability-loss reduction, or Minimal Context for exact answer and stop preservation. See honest progress, cancel if needed, and review the best result found within the selected search contract.
5. Save an exact Minimal Context result, compare it with its original, and reopen both after restarting Clozn.

Studio has a run list with a New run action and one run workspace with two modes: **Context Effects** and **Minimal Context**. Comparison belongs to the reduction result. Received context, provenance, and runtime details become expandable information in that workspace. Session navigation, a separate runtime dashboard, and a general comparison workbench leave the product.

The composer is deliberately small: one Markdown/plain-text prompt and optional instructions. Clozn derives a bounded flat partition across every non-empty input message using basic Markdown structure; clients do not submit source blocks or source ranges. The example populates these fields and executes a real run. Document ingestion, retrieval, website fetching, project indexing, and a full chat application are outside v1.

**2. Keep/cut decisions**

| Area | v1 decision | Boundary |
|---|---|---|
| Local generation | Keep | One loaded GGUF, one qualified generation configuration, text streaming, cancellation, durable recording. |
| Context Effects visual | Keep as the default | Linked input/output text, pane-expanded heatmaps, influence extremes, bidirectional selection, stable text ranges, directional measured effects, keyboard access, explicit unmeasured/unavailable states. |
| Lean Context | Add as a bounded Context Effects view | Reuse compatible single-source score observations; search combined removals under fixed balanced teacher-forced loss tolerances; report best found within 16 new executions and never claim a proven minimum. |
| Context Effects orchestration | Simplify | One measurement action and a bounded policy. Remove execution-strategy selectors, joint-effect exploration, and the public coarse/fine tuning interface. Keep an internal optimization only when the qualified demo needs it. |
| Minimal Context | Keep | Exact preservation, fewer rendered input tokens, bounded search across all input text, cancellation, durable result, verified child creation. |
| Run comparison | Narrow | Original versus its reduced child: retained/removed context, rendered token counts, answer match, and generation identity. |
| Run storage and evidence | Keep and simplify | Immutable run/evidence records, integrity checks, source ranges, basic deletion, and necessary blob cleanup. |
| Local text API | Keep a documented subset | Chat completions and model identity/listing; text-only messages and supported generation fields. Unsupported behavior-bearing fields receive clear errors. |
| Claims, diagnoses, corrective actions/retries | Cut | UI, routes, CLI, feature-specific recipes, schemas, fixtures, and docs. Keep actionable operational errors. |
| Generation guards, concept steering, calibrated trust | Cut | Remove alternate generation paths and policy decoration from ordinary requests. |
| Time Travel, token workbench, branch fans, sampler sensitivity | Cut | Remove exposed research workflows. Preserve internal replay/state reuse needed by the two retained analyses. |
| User snapshots and continuation workbench | Cut | Remove pin/restore/continuation product surfaces. Internal checkpoint primitives are assessed by dependency. |
| Model CI, experiment matrices, regression suites | Cut | Remove model/adapter comparison products, suite manifests, statistical reports, and the packaged GitHub Action. |
| Mechanistic tracing, transplant, J-lens, SAE | Cut | Remove exposed capabilities, qualification assets, sidecar loading, and unused custom native code. Preserve shared scoring and template functions. |
| Multiple loaded models | Cut as a supported mode | Remove pool configuration, preload/eviction workflows, and multi-model product claims. Simplify the existing router rather than adding a second single-model runtime. |
| Tools, structured output, Ollama/Responses compatibility | Cut | One text API contract. Keep reuse of a compatible local GGUF independent of protocol emulation. |
| Research SDK, notebook/OTel/evidence exports | Cut | Remove the separate research product and export workflows. Retain the private worker client used by the gateway. |
| Historical journal migration | Defer | Preserve old data; v1 uses an explicit, isolated data root. Import is future work. |
| Automatic updater/rollback, adoption wizard, broad qualification UI | Defer | Reuse necessary install/model acquisition helpers through one supported path; keep qualification as a development/release check. |

Every feature deletion includes its public registration, implementation used only by that feature, schemas, package assets, tests, examples, and claims in documentation. Feature flags and an in-tree legacy application do not count as removal. Shared code stays only with an identified v1 caller and appropriate coverage.

**3. Scientific and product contracts to preserve**

Context Effects and Minimal Context use the same recorded source identities and range mappings. The reduction universe is a declared set of legal, non-overlapping removable units derived from that catalogue. Nested display spans may remain visible without becoming independently deletable overlapping units. Every non-empty message body is eligible, including instructions and the latest user message; only structural chat-template syntax stays outside the source universe.

Context Effects removes a source and scores the recorded continuation. Links mean a measured change in the likelihood of selected output tokens under that intervention. Preserve the sign and measurement coverage. A whole-block measurement must be shown at whole-block granularity; precise text selection must not manufacture finer evidence. Low measured effect, unmeasured, failed, and unavailable remain distinct.

Minimal Context separately verifies the recorded output under the qualified model, template, tokenizer, engine identity, and decoding contract. Preservation includes completion/stop semantics. Teacher-forced likelihood is not its success criterion. Removing several individually weak sources still needs verification of their combined removal.

The objective is fewer **rendered prompt tokens**, including template effects. Retained source count is secondary. The normal result is **Best verified within budget**. A stronger minimum claim appears only when its actual proof conditions hold; global-minimum certification is not required to ship.

Saving a result retains the verified winner, creates an immutable child with matching evidence, and leaves the original intact. Replay drift, stale identity, failed recording, or an unavailable control prevents a verified-child claim. Repeated save requests must not accidentally duplicate children.

**4. Target implementation boundaries**

| Responsibility | Starting points in the current code | Target |
|---|---|---|
| Runtime and inference | `clozn/cli/runtime_process.py`, `clozn/server/model_routing.py`, `clozn/server/substrates.py`, `engine/client/clozn_engine/`, `engine/core/` | One supervisor and private worker. Internally expose generation, recorded-answer scoring, and exact reference matching. |
| Recorded runs and context | `clozn/runs/`, `clozn/receipts/`, `clozn/replay/span_bridge.py` | One canonical record/source mapping, with immutable evidence and a versioned storage contract. |
| Shared analysis execution | `clozn/experiments/execution.py`, `execution_facts.py`, `scoring.py`, `persistence.py`, `selections.py`, `observations.py` | Keep the execution and evidence needed by the two analyses. Package imports should not load unrelated products. |
| Effects | `clozn/recipes/context_effects.py`, `clozn/experiments/context_investigation.py`, `clozn/server/routes/context_investigation_v1.py` | A linked reader/query and one effects job family. |
| Reduction | `clozn/recipes/minimal_context.py`, `minimal_context_result_store.py`, `clozn/experiments/context_search.py`, `search.py`, `clozn/server/routes/minimal_context.py` | One exact-preservation workflow and result format. |
| Child creation and comparison | `clozn/experiments/materialize.py`, `clozn/replay/replay.py`, relevant run-diff helpers | Narrow entry point for saving the verified reduction and comparing it with the parent. |
| Background work | `clozn/server/influence_jobs.py` | One job lifecycle for measurement/reduction: queued, running, completed, cancelled, failed. Durable finished results; interrupted jobs are explicit on restart. |
| Studio | `studio-frontend/src/app/`, `features/inspect/Inspect.tsx`, `MinimalContext.tsx`, and their data modules | Run list/composer plus the two-mode workspace; reuse accurate text selection, highlighting, and range geometry. |

These are reuse boundaries, not an instruction to preserve entire directories. Avoid renaming shared code merely to make the directory tree look new. Split a mixed module where the split enables a concrete feature deletion or removes an unwanted dependency.

Particular dependency traps:

- `clozn/recipes/__init__.py` eagerly imports Time Travel; `clozn/experiments/__init__.py` imports model-suite code. Narrow these exports before removing feature modules.
- `clozn/experiments/materialize.py` uses replay and `clozn.analysis.model_diff.diff_runs`. Extract the small retained comparison operation before deleting model-comparison workflows.
- Single-model serving uses the same router as multi-model serving. Keep the selection/identity contract while reducing it to one worker.
- Native reference matching and variant scoring use checkpoint code. User-facing snapshots can disappear while these internal facilities remain.
- `engine/core/serve/routes_whitebox.cpp` contains ordinary scoring and template preparation alongside research operations. Deleting the file wholesale would break v1.
- Context Effects currently calls the coarse-to-fine effects-search API. Moving the UI to the simple direct-measurement API requires an explicit client/server change; removing the scheduler first would break the retained screen.
- `clozn-client/` is the research SDK. `engine/client/clozn_engine/` is the gateway's worker client and remains required.

The public server gets an explicit route allowlist. CLI commands get an explicit command list. Replace source-file discovery in the two `_autoload.py` registries for the v1 product, so dropping an old file into a directory cannot expose a feature again. Unknown run subroutes must return an error instead of falling through to a generic run response.

Proposed CLI scope: `serve`, `run`, `studio`, `pull`, `models`, `ps`, `stop`, `doctor`, and `version`. Basic run deletion belongs in Studio. Smoke tests and qualification remain development scripts. Reuse current command names and helpers where practical; the primary documented launch is `serve` with an option to open Studio.

Proposed API families: health/readiness; the text generation API; run list/read/delete; the canonical context/effects reader and query; effects jobs; reduction jobs/results; save reduced run; parent/child comparison; and static Studio assets. Keep existing URLs where their semantics still fit. Do not add a second alias API merely to label the product v1.

**5. Implementation sequence**

Each step is a reviewable change or small series. Do not begin with a mass deletion that removes the runnable reference implementation.

| Step | Work | Completion gate |
|---|---|---|
| 0 — Preserve and define | Record the exact starting revision and preserve it with a tag when implementation starts. Work on a `codex/` branch. Capture current failures, current command/route manifests, and a small real-model demo input. Pin the reference GGUF digest and engine configuration. | Rollback reference exists; v1 scope, baseline failures, and demo input are committed. |
| 1 — Make the retained path reliable | Fix runtime identity normalization, explicit recording failures, storage configuration, and smoke-test capability discovery. Establish a successful real-model path through both analyses and child creation. | A fresh run has a durable ID; effects produce usable evidence; reduction verifies and saves a child; both reopen after restart. |
| 2 — Establish product boundaries | Add explicit command/route lists, slim eager imports, remove guard/calibration branches from ordinary generation, restrict text/API inputs, and support one worker. Keep necessary helpers until callers have moved. | Ordinary and streamed text generation still record correctly. Removed surfaces are absent and unsupported inputs fail visibly. |
| 3 — Remove independent products | Remove Claims/Actions, model CI/suites/matrices, model and adapter validation, triage, research SDK/export workflows, and their unique assets/tests/docs. Extract shared comparison helpers where needed. | Core imports and the real-model path work without any deleted product. Packaged assets no longer contain those features. |
| 4 — Narrow the two analyses | Keep the linked reader and direct effects measurement. Remove source-test/counterfactual/branch controls and public scheduling policies. Keep one bounded reduction policy and one shared job lifecycle. | Exact input/output selections still map to recorded evidence; effects and reduction remain distinct; cancel/restart/budget exhaustion behave honestly. |
| 5 — Rebuild the small Studio | Add run list/composer/example; replace the broad shell with the two-mode workspace; embed reduction comparison; fold context/runtime details into disclosure panels; retain basic deletion. | A new user can complete the full demo in the browser without a pre-existing journal or manually crafted API request. |
| 6 — Remove workbench and native leftovers | Delete Time Travel/token/branch/snapshot surfaces; remove unused native research route registrations, sidecar loaders, custom assets, and feature-only build targets. Retain only required replay/checkpoint/scoring paths. | Fresh native build and relevant C++ tests pass; neither the package nor the worker advertises removed capabilities; both analyses still pass the live gate. |
| 7 — Package and qualify | Regenerate Studio, trim schemas/protocol assets, update metadata/docs, simplify CI around retained contracts, and verify one supported installation route outside the checkout. | A clean install on the reference platform runs the entire browser/API demo, including restart, cancellation, failures, and data deletion. |

Steps 1, 4, 5, and 7 are the critical product path. Step 6 has the most dependency uncertainty; split it by native feature family and preserve a working checkpoint after each removal. Correctness and semantic parity come before an additional execution optimization.

**6. Concrete removal map**

The following files identify feature ownership. Mixed/shared files require extraction first; these lists are not wildcard deletion commands.

| Removal group | Current entry points |
|---|---|
| Claims and repair | `features/inspect/ClaimSupport.tsx`, `DiagnosisRepair.tsx` under `studio-frontend/src/`; `clozn/server/routes/claim_support.py`, `corrective_actions.py`, `corrective_retries.py`, `diagnosis_findings.py`; associated data/CSS modules. |
| Generation controls | `clozn/server/generation_guard.py`; guard and calibrated-policy calls in `routes/openai.py` and `generation_gateway.py`; feature-only uses of `clozn/eval/`, `clozn/behavior/`, and `clozn/runs/calibrated_trust.py`. |
| Structured/tool compatibility | `clozn/server/structured_io.py`, structured qualification in `app.py`/substrates, tool/JSON branches in generation, `routes/ollama.py`, and the Responses adapter. Preserve shared chat-template preparation. |
| Time and token workbench | `clozn/recipes/time_travel.py`; routes `time_travel_v1.py`, `timetravel.py`, `time_machine_continuation.py`, `token_workbench.py`, `token_workbench_actions.py`, `branch_fan.py`, `sampler_sensitivity.py`, `snapshot.py`, `snapshots.py`, `rewind_fidelity.py`, `suggested_breakpoints.py`, `universal_test_this.py`; corresponding CLI registrations. |
| Evaluation products | `clozn/experiments/suite.py`; CLI modules `experiment_suite.py`, `regression_suite.py`, `ci_check.py`, `diff_model.py`, `diff_adapter.py`, `validate_export.py`, `quant_check.py`, `test_model.py`, `eval.py`, `investigate_experiment.py`, `triage.py`; `features/compare/ExperimentMatrix.tsx`; `integrations/github-action/`. |
| Mechanistic products | Routes `causal_trace.py`, `context_bisect.py`, `mechanistic_diff.py`, `readouts.py`; mechanistic CLI registrations; feature-only analysis/readout modules; `engine/core/serve/routes_jlens.cpp` and research portions of `routes_whitebox.cpp`/`server_main.cpp`. |
| Other influence interfaces | `section_influence.py`, `section_drill.py`, `influence_map.py`, `influence_query.py`, `influence_counterfactual.py`, `context_tension.py`, `context_utilization.py` under `clozn/server/routes/`. Retain any source-mapping/scoring helper still required by canonical Context Effects; expose one effects contract. |
| Expanded Effects actions | Remove-test, generated-counterfactual, and materialization branches in `context_investigation_v1.py`; corresponding methods in `contextInvestigationClient.ts`/`contextInvestigationActions.ts`; orchestration controls in `Inspect.tsx`. Keep Minimal Context's own materialization path. |
| Multi-model operations | Multi-model configuration and pool behavior in `managed_models.py`, `worker_registry.py`, and `model_routing.py`. Preserve worker supervision and the one-model identity/selection seam. |
| Extra applications/assets | `clozn-client/`, obsolete Studio extension assets, feature-only examples/research scripts, schemas, protocol fixtures, and export assets. Inspect `setup.py` package-data discovery so deleted products cannot remain in a stale wheel. |

Resolve less obvious commands such as `explain`, `prove`, `trace`, `test`, `watch`, `retry`, and `plan` against the v1 command allowlist. Retain a shared helper only when a kept command or analysis needs it; remove the independent workflow.

**7. Data, compatibility, and release limits**

Use a dedicated root such as `~/.clozn/v1/` for the v1 run database, blobs, evidence, and results. Make the root injectable for tests and portable installs. Reuse compatible storage implementation and schema machinery; a product version label does not require arbitrarily renumbering every artifact schema.

Leave `~/.clozn/runs/` and other historical user data untouched. Reuse model files by explicit path or the existing model catalogue. Do not auto-import historical records or maintain a legacy journal reader in the v1 application. Startup should clearly identify the active data location, and the preserved old checkout remains available for historical inspection.

Readiness must cover both the worker and the ability to record a run. An incompatible/unwritable journal is an actionable startup/readiness error. If recording fails after generation begins, expose a recording failure; streamed text cannot be presented as a successfully saved run simply because tokens arrived.

Initial qualification target: macOS Apple Silicon, one pinned small GGUF and one backend configuration. The existing CPU run is a starting smoke fixture, not a finished qualification. Select CPU or Metal using real effects/reduction timings in step 1. Broader model and operating-system support may continue to work, but is outside the v1 release claim.

Starting analysis limits for qualification: 16 removable context units, 128 generated answer tokens, 32 new reduction trials, and one active analysis job. These are proposed bounds to test, not measured capacity claims. Reserve answer space inside the model context window, define what consumes each budget, and count unchanged controls separately. Reject over-limit analysis before expensive work; never silently truncate context or label unmeasured sources irrelevant. Runs that exceed analysis limits remain readable.

Measure the example's effects/reduction latency and memory on the reference machine, including cold and warm runs. Adjust the example, limits, or necessary internal optimization before finalizing the support contract. Do not promise fast execution solely from a mockup or a model-free test.

**8. Validation and definition of done**

The earlier audit at the starting revision passed 5,344 Python tests, 228 frontend tests, a frontend production build, 12 tests in the existing native build, and seven clean-install checks. It also found that fresh real-run analysis failed on identity normalization and an existing journal could fail recording silently. These results establish a useful regression baseline; they do not establish a working v1 or replace a fresh native build.

Preserve relevant coverage in `tests/test_context_effects_kernel.py`, `test_context_investigation_kernel.py`, `test_context_investigation_routes.py`, `test_context_units.py`, `test_context_search_universe.py`, `test_minimal_context_kernel.py`, `test_minimal_context_v1_result.py`, plus recording, identity, replay, materialization, privacy, and range-mapping tests. Move or rewrite tests to match retained contracts where needed. Remove tests only with their deleted feature; a lower test count is neither a goal nor evidence of correctness.

| Gate | Required evidence |
|---|---|
| Source mapping | Unicode, repeated text, template boundaries, nested ranges, and bidirectional selections resolve to the actual recorded spans. |
| Effects | A known relevant-source intervention changes recorded-answer scores; low-effect, unmeasured, unavailable, stale, and failed cases render distinctly. Tested scoring agrees with its trusted reference policy. |
| Reduction | Full output and stop behavior match under the recorded contract. Joint removal is verified. Token objective, zero improvement, exhausted budget, and stronger-proof labels are correct. |
| Saving | Reduced child has valid evidence and parent linkage; retries are idempotent; original is unchanged; stale or drifting replay cannot produce a verified result. |
| Lifecycle | Cancellation during each phase, worker loss, restart, recording failure, and deleted runs leave honest terminal states and no stranded native sessions. Results reopen without rerunning the model. |
| Product boundary | Exact CLI/route manifests; removed fields rejected; no deleted-feature imports, exports, package assets, default flags, or capability advertisements. |
| Browser journey | Fresh start → create real example run → select linked text → measure → reduce → save → compare → restart/reopen → delete. Check keyboard use and narrow layouts. |
| Installation | Fresh native build, rebuilt committed Studio assets, retained Python/frontend/native checks, and `scripts/release/clean_room_install_test.py` from a clean build directory outside the checkout. |

The known blockers to resolve first are the `engine_build`/artifact-digest mismatch between `clozn/runs/identity.py` and `clozn/experiments/execution_facts.py`, swallowed recording errors in `clozn/server/app.py`, and the context-investigation smoke script's stale capability lookup.

v1 is done when the full journey passes on the published configuration, both signature features work from a fresh installation, old data remains intact, and the removed products are absent from the shipped application. README, package metadata, help output, capabilities, and release checks must describe that same product.

**First implementation change:** preserve the current revision, add the isolated v1 data-root plumbing, fix recording/identity failures, and establish the live two-feature acceptance test. That gives the subsequent cuts a reliable foundation.
