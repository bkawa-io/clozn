# Clozn

**Model CI and an inspectable local runtime for the GGUFs you already use.** Compare a base model with a
fine-tune or LoRA adapter, run target + guard experiments, and fail CI on regressions with per-token
receipts. Then serve the model through familiar OpenAI/Ollama-compatible APIs: prompts, responses,
steering, timings, and exact rendered context become inspectable runs without requiring every user to
live in Studio.

For everyday debugging, Clozn shows what context was delivered and survived, compares any two recorded
runs, measures source support when the required evidence is available, and offers one-shot corrective
retries and controlled comparisons — try a correction, generate a matched candidate, and see whether it
actually changed the output. Nothing persists past the run it was generated from. For deeper work, it
can teacher-force a stored answer, capture token
alternatives, apply qualified interventions, and attach model-specific J-lens readouts. These are
evidence tools—not a claim to decode literal thought—and white-box capabilities fail closed unless the
exact artifact is qualified. See the [capability matrix](docs/CAPABILITIES.md) and
[model support](docs/MODEL_SUPPORT.md) for the boundaries.

`clozn` = `cloze` (the engine inside) + *cozen* (to deceive — the illusion it reveals).

→ **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the design, the layers, the state-stream protocol.
→ **[docs/CAPABILITIES.md](docs/CAPABILITIES.md)** — what is merged, released, qualified, and limited.
→ **[docs/ROADMAP.md](docs/ROADMAP.md)** — the consolidated map: what's done, the v1 cut, what's next.

## Quickstart

> **Distribution status:** the managed setup contract is implemented, but no public engine release
> matrix is published yet. This source snapshot still requires a developer build; see
> [native distribution status](docs/NATIVE_DISTRIBUTION.md) and
> [development bring-up](docs/DEVELOPMENT.md).

Run a local model in one command. `clozn` starts one public, Torch-free product gateway and one private
C++ model worker. It finds your build (GPU if present), streams tokens, reports honestly what it is
running on, and fails with one clear line instead of a stack trace.

```bash
clozn pull llama-1b                       # download a model (qwen / mistral / gemma-2b / owner/repo/file.gguf)
clozn models                              # discover local GGUFs + the backend that would run them
clozn run llama-1b "Explain entropy."     # one-shot, streams tokens to the terminal
clozn serve qwen --port 8080              # gateway + private worker; API and Studio on :8080
clozn studio --open                       # attach a browser to that already-running gateway
clozn ps                                  # what's running    ·    clozn stop qwen   to stop it
```

If the GGUF already exists in Ollama, inspect the adoption plan before copying or hard-linking anything:

```bash
clozn adopt ollama                        # list discoverable Ollama models
clozn adopt ollama --model MODEL --dry-run
clozn adopt ollama --model MODEL --try --yes --qualify --connect open-webui
```

Managed Ollama manifests resolve the exact model layer by digest and size; Clozn never guesses by
choosing the largest blob. `--try` previews disk, fidelity, capability, and connector changes until
`--yes` is supplied. Adoption, qualification, and client configuration remain independently undoable.

Before changing the runtime, run its managed acceptance gate:

```bash
clozn smoke qwen --preflight              # report every missing build/model/asset prerequisite
clozn smoke qwen                          # test APIs + SQLite, restart the worker, then clean up
clozn smoke qwen --deep                   # also exercise forced receipts and replay
```

`clozn smoke --url http://127.0.0.1:8080` attaches non-destructively to an existing gateway. Managed
smoke owns the stack it starts, verifies that the private worker can be replaced without changing the
public gateway, and stops the complete process tree when finished.

## Model CI

Start with the question fine-tune authors actually have: did the candidate change, what improved, and
what regressed?

```bash
clozn diff-model base.gguf tuned.gguf --runs 8 --both
clozn diff-adapter base.gguf tune.gguf
clozn validate-export base.gguf --adapter tune.gguf --merged merged.gguf \
  --suite examples/adapter-export-suite.v1.json --out export-receipt.json
clozn experiment run examples/experiment.v0.json --out result.json
clozn ci check --experiment result.json --min-target-gains 1 --max-guard-regressions 0
```

`diff-model` refuses mismatched tokenizers and labels its verdict as a sample-based screen, not proof of
quality. `validate-export` uses base, base-plus-adapter, and merged arms; it gates the adapted-versus-
merged teacher-forced deltas while keeping base as the control and never calls a differently quantized
merge byte-equivalent. Experiment artifacts retain each instrumented run; the CI gate validates their
matrix and identity evidence, recomputes comparisons from raw cells, and returns a deterministic exit
code. See the
[worked Qwen reasoning-SFT case study](docs/MODEL_DIFF_CASE_STUDY_QWEN_REASONING.md) for a real two-GGUF
run on a 16 GB Mac.

Chat templates come from each model's own GGUF (Qwen / Llama-3 / Mistral / Gemma / …), applied
engine-side, so pulled models chat coherently — not just Qwen. Drop the prompt — `clozn run llama-1b` —
for an interactive chat (multi-turn; `/reset` clears, `/bye` quits).

`clozn run` reuses a running `serve` for that model (warm, no reload); otherwise it spawns a temporary
gateway/worker pair and tears it down after. Product commands never bypass the gateway to call a warm
worker directly. `clozn serve` supervises the private worker and restarts it after an unexpected exit.

OpenAI clients use the documented subset of `/v1/chat/completions` and `/v1/models`; the retired
`/v1/completions` path returns a typed HTTP 410 migration response. Unsupported behavior-bearing fields
return a typed 400 instead of being silently ignored. See the exact
[endpoint/field matrix](docs/OPENAI_COMPATIBILITY.md). Clozn's CLI and Studio instrumentation use
`/api/clozn/generate`, which preserves the native state-event stream. Native event frames never leak into
an OpenAI completion stream.

Every run is debuggable after the fact. The engine streams per-token confidence and the alternatives it
weighed; open a run in Studio for the delivered-context receipt, measured source links when available,
the **"Disposed to say · J-lens"** panel (per-token, per-layer, with an unskippable provenance caption),
the branch lineage tree, and the exact rendered prompt the model saw:

```bash
clozn trace                               # last runlog entry: confidence timeline + almost-said tokens
clozn inspect <clozn_run_id>              # explain any API reply from the local journal; no model needed
clozn branch                              # re-run from the most uncertain token on the alternative
clozn test cases.json                     # run-level assertions over the receipt/replay seams
clozn context export <run_id> --out receipt.json --privacy metadata_only
```

`clozn trace`, `clozn inspect`, and Studio read the same SQLite journal. Non-streaming OpenAI chat
exposes the exact id as `clozn_run_id` and `X-Clozn-Run-Id`. Streaming requests are journaled but cannot
expose a post-generation run id in already-committed headers; the terminal chat chunk carries the id.
`inspect` assembles confidence, active influences, and captured
concepts locally, falling back to a running gateway only when the id is not in this journal.
Queryable run
metadata lives in `~/.clozn/runs/runs.sqlite3`; large traces are immutable, content-addressed blobs under
`~/.clozn/runs/blobs/sha256`. Schema migrations apply automatically whenever the store is opened.
`clozn test` runs user-authored checks against a stored run; model-free assertions such as `contains`,
`finish_reason`, and `min_confidence` read the run alone, while live checks honestly skip when their
measurement prerequisite is unavailable. Point any OpenAI client at `clozn serve`; pass
`"clozn_trust": true` in a chat request to get per-claim confidence spans back on the wire (labeled
uncalibrated).

`clozn run …` works once the repo root is on PATH; otherwise `python -m clozn run …`. Put GGUFs in
`~/.clozn/models`, set `CLOZN_MODELS=<dir>`, or list dirs in `~/.clozn/config.json`. For this unreleased
source snapshot, build the engine first. On Windows: `cd engine/core && build_gpu.bat` (GPU, CUDA) or
`build_serve.bat` (CPU). On Linux/macOS: `./engine/core/build_gpu.sh` (GPU: CUDA on Linux, Metal on
macOS) or `./engine/core/build_serve.sh` (CPU). See [Platform support](#platform-support) below for what
is and is not independently verified on each.

## Platform support

The Python side (`clozn/`, stdlib-only) was never Windows-only: `clozn/setup/platform_detect.py`
already detects Windows/Linux/macOS and x86_64/arm64 and treats every Apple Silicon Mac as Metal-capable
by design, and both `clozn.sh` (POSIX) and `clozn.cmd` (Windows) are documented launchers.

What genuinely had no POSIX path until this change: `engine/core/`'s build convenience scripts
(`build_serve.bat`, `build_gpu.bat`, …) had no `.sh` equivalent, and `_env_with_dlls()` — the code that
puts the engine's own build directory on a spawned worker's library search path — only ever wrote
`PATH`, the Windows DLL mechanism. That made it inert rather than broken on Linux/macOS: a CMake-built
binary normally also carries an rpath into its own build tree, which is almost certainly why a
locally-built engine has run on a Mac before despite this. `engine/core/build_serve.sh` / `build_gpu.sh`
now mirror the `.bat` scripts (`build_core.sh`, `build_cuda.sh`, `build_sae.sh` too — see
[docs/DEVELOPMENT.md](docs/DEVELOPMENT.md)), and `_env_with_dlls()` now also sets `LD_LIBRARY_PATH`
(Linux) / `DYLD_LIBRARY_PATH` (macOS) as an additional robustness path, not a fix for a prior failure.

What is actually verified, per platform, as of this snapshot:

| Platform | Python CLI/product | Engine build (CMake) | How that is known |
|---|---|---|---|
| Windows x64 | yes | CPU + CUDA GPU | The primary dev machine for this repo (RTX 5080); built and run directly, repeatedly. |
| Linux x64, CPU | yes (stdlib-only) | yes | `.github/workflows/real-runtime-smoke.yml` builds `clozn-server`, downloads a real GGUF, and runs `clozn smoke --deep` against it on `ubuntu-24.04` every night — the strongest non-Windows evidence in this repo. |
| Linux x64, CUDA GPU | yes | scripted (`build_gpu.sh` detects `nvcc` on `PATH`) | Not built or run anywhere; no GPU CI runner exists for this repo. |
| macOS, Metal | yes | scripted (`build_gpu.sh` detects Darwin) | The repo owner reports having run clozn on a Mac. No CI run or build artifact for it exists in this repository: `.github/workflows/native-engine-release.yml` has a `macos-14`/Metal release-matrix cell, but that workflow has never been dispatched (0 recorded runs). Treat this row as owner-reported, not reproducible from this repo alone. |

"Scripted" means the CMake invocation and platform-detection logic exist and are exercised by unit tests
that simulate the target platform (`tests/test_env_with_dlls_platform.py`) — not that a build has
actually run there. None of this is a claim that Linux CUDA or macOS are unsupported in principle; the
codebase (standard CMake, `if(WIN32)`/`if(MSVC)`-gated C++, POSIX-aware Python process management) was
already structurally cross-platform. It is a claim about which cells have and have not actually been
exercised, so nobody trusts a row this document doesn't back up.

## Layout

| Dir | What |
|---|---|
| `clozn/`    | the product Python package — server/API, run journal, context receipts, replay, steering, readouts, the J-lens proxy, and Model CI (`python -m clozn`) |
| `engine/`   | the C++/ggml runtime plus optional CUDA kernels — runs GGUF models, emits the state stream, harvests activations, applies steers, and serves `/jlens` |
| `studio/`   | the white-box UI — runs, comparisons, experiments, context/source evidence, performance, tone dials, and J-lens readouts, served by the backend |
| `protocol/` | the one state-stream contract the engine emits and the studio consumes |
| `docs/`     | architecture, the consolidated roadmap, and the honest technical account |
| `tests/`    | the model-free product suite · `scripts/` dev tooling |

The legibility-science spikes and findings (the interpretability-tax thread) live in a separate
local-only sibling repo: `../clozn-research`.

The product runtime is the **autoregressive GGUF** path. The earlier diffusion runtime and PyTorch
workbench are research history, not alternate current-user commands; the old design and measurements are
kept in clearly labeled archival documents. The AR core contract includes trace, harvest, steer, and
teacher-forced `/score`; the checked-in qualification ledger records how far each exact model/quant has
passed. **J-lens runs a published Jacobian-lens method** (not a Clozn invention), fit per model offline
(nf4 + autograd) and applied forward on the engine's own GGUF head; today's qualified fit covers
Qwen2.5-7B. A second-family fit and targeted cross-family write checks remain open.
