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

| Capability | Core support | CLI / API | Studio | Qualification | Released version | Limitations |
|---|---|---|---|---|---|---|
| Autoregressive GGUF runtime | Merged: supervised C++ worker, strict gateway, journaling, trace, scoring, steering | `clozn run`, `serve`, `smoke`; OpenAI subset and native stream | Runtime state, runs, Lens, Model | Wave 1 records deep CPU core passes for five exact model rows; see [model support](MODEL_SUPPORT.md) | Unreleased | A loadable GGUF and usable embedded chat template are still required; optional white-box artifacts have separate gates |
| Managed engine setup | Merged: manifest selection, verified download, transactional promotion and rollback | `clozn setup status|upgrade|rollback`; `clozn doctor` | None | Model-free installer and clean-room source-snapshot tests | Unreleased | No public manifest/archive matrix exists yet, so this is not a working public install path |
| Base, fine-tune, and quant comparison | Merged: teacher-forced token-difference ladder and target/guard experiment artifacts | `clozn diff-model`, `quant-check`, `experiment`, `ci` | Experiment matrix and cell detail | Worked Qwen reasoning-SFT case plus model-free contract tests | Unreleased | A sampled screen is not proof of quality; tokenizer/template mismatch fails or is labeled |
| GitHub Action model gate | Merged source package: engine-free verify path, trusted run orchestration, SHA-keyed/cache-qualified downloads, metadata-only indexed receipt bundle, job summary/JUnit, exact exit propagation, and read-only PR degradation | Publication-ready `integrations/github-action/action.yml`; core `clozn ci check`, `model-lock fetch` | None | Model-free Action/security/cache/receipt tests in this tree | Unreleased | `bkawa-io/clozn-action`, immutable semver tags, moving `v1`, Marketplace publication, and public Clozn/engine compatibility pins do not exist yet; run mode is restricted to trusted non-PR events |
| LoRA loading and comparison | Merged: one fail-closed LoRA attachment, run identity, and three-arm merged-export receipt | `clozn serve --adapter`; `clozn diff-adapter`; `clozn adapter validate`; `clozn validate-export` | Adapter identity appears in comparison provenance | Attachment and scale-zero proof recorded in [adapter support](ENGINE_ADAPTER_SUPPORT.md); merged-export gate has model-free bad-merge coverage but still needs a known live merged export | Unreleased | LoRA GGUF only; PEFT/Safetensors conversion stays external; one adapter at a time; no hot swap |
| Context receipt | Merged: versioned delivered/assembled/omitted segments, exact rendered hash/size/token facts, raw termination, stable client source IDs, and privacy tiers | `clozn context last|show|export`; `GET/POST /runs/<id>/context-receipt` | Compact and detailed Lens views | Schema, endpoint, CLI, and model-free Studio tests | Unreleased | Describes delivered context; it does not by itself prove that a source caused output |
| Measured source support | Merged: bounded source influence measurement, receipt-linked source identity, identity-bound cache, privacy-safe portable export, and explicit absence/evidence states | `clozn provenance`; influence-map routes including `/export` | Sources lens renders measured, below-floor, omitted, and unavailable states | Research batteries cover exact Qwen2.5 and Llama3.1 rows; not a universal model qualification | Unreleased | The dedicated attention-knockout provenance command needs a worker started without flash attention; the Sources lens does not silently change worker configuration |
| Run investigation synthesis | Merged: read-only, versioned composition of context delivery, stored influence, deterministic diagnosis, structural comparison, and corrective-action availability | `GET /runs/<id>/investigation` | Backend contract ready; primary investigation entry point not yet wired | Model-free schema and route tests | Unreleased | The GET never starts model work; unavailable measurements are returned as typed action descriptors, and artifact-native evidence states remain distinct |
| Run comparison and replay planning | Merged: versioned run diff across identity, context, settings, and output | `clozn compare-runs`; `GET /runs/compare` | Compare workspace | Schema and model-free route/CLI tests | Unreleased | `--replay` plans eligible swaps; it does not execute them |
| Reversible corrective retry | Merged: prompt-first retry, scoped activation, transaction record, proven undo | `clozn retry last|undo`; retry/undo routes | Behavior and run comparison surfaces | Model-free comparison, conflict, and state-restoration tests | Unreleased | Requires a ready worker to generate the comparison; unavailable dial backends are labeled |
| Ollama model adoption | Merged: exact manifest-layer resolution, fidelity classification, dry-run/guided try, hard-link/copy, optional core smoke, transactional Aider/OpenAI/Open WebUI/Ollama-SDK connectors, and undo | `clozn adopt ollama --try --yes [--qualify] [--connect APP]` | None | Model-free resolver/adoption/qualification/connector tests; a live adopted model still determines whether `--qualify` passes | Unreleased | Does not mutate or stop Ollama; hard links require a compatible filesystem; no desktop-specific connector is claimed |
| Performance diagnosis | Merged: versioned worker timings for load/startup, template, tokenize, context/KV creation, prefill, and decode; gateway timings for queue, dispatch, serialization, and stream flush; separate clock owners; known/unaccounted aggregation; evidence-gated regression attribution | `clozn diagnose --performance`; `GET /runs/<id>/performance` | Measured phase breakdown, known/unaccounted time, throughput provenance, and rule evidence | Model-free protocol/old-worker/cold-warm/long-prompt/cancellation/CPU-vs-GPU tests plus a clean CPU worker compile | Unreleased | Cross-process offsets are never aligned; overlapping and process-startup spans are shown but excluded from known in-request time. Live GGUF CPU/GPU runs are still required to qualify absolute timing accuracy and model/backend-specific thresholds |
| J-lens readout | Merged apply path with artifact identity and checksum validation | `/jlens`, `/runs/<id>/jlens` | Lens layer view with provenance caption | Exact Qwen2.5-7B Q4_K_M row only | Unreleased | Fit per model; a linear readout is not a transcript of thought or causal proof |
| OpenAI-compatible text API | Merged strict subset of models, chat completions, and text completions | `/v1/models`, `/v1/chat/completions`, `/v1/completions` | Runs recorded through the same gateway | Model-free SDK conformance plus real-runtime smoke gates | Unreleased | Text-only subset; unsupported behavior-bearing fields return typed errors |

## Removed and research-only surfaces

- Prompt-card and learned-prefix memory were removed from the product on 2026-07-27. Existing run
  readers may still tolerate old `memory` fields, but current runs do not apply or record cards.
- The user-facing PyTorch workbench was retired with that memory path. Offline calibration and research
  scripts are not an alternate serving product.
- Diffusion-language-model work is historical research, not a current product substrate or command.
  [DESIGN.md](DESIGN.md) and [TECHNICAL.md](TECHNICAL.md) preserve that lineage as archives.
