# Clozn (wip dont use this yet please)

**Model CI and an inspectable local runtime for the GGUFs you already use.** Compare a base model with a
fine-tune, run target + guard experiments, and fail CI on regressions with per-token receipts. Then serve
the model through familiar OpenAI/Ollama-compatible APIs: prompts, responses, memory, steering, timings,
and exact rendered context become inspectable runs without requiring every user to live in Studio.

For deeper work, Clozn can teacher-force a stored answer to measure which supplied memory it depended on,
capture token alternatives, apply qualified interventions, and attach model-specific J-lens readouts.
These are evidence tools—not a claim to decode literal thought—and white-box capabilities fail closed
unless the exact artifact is qualified. See [model support](docs/MODEL_SUPPORT.md) for the boundary.

`clozn` = `cloze` (the engine inside) + *cozen* (to deceive — the illusion it reveals).

→ **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — the design, the layers, the state-stream protocol.
→ **[docs/ROADMAP.md](docs/ROADMAP.md)** — the consolidated map: what's done, the v1 cut, what's next.

## Quickstart

Run a local model in one command. `clozn` starts one public, Torch-free product gateway and one private
C++ model worker. It finds your build (GPU if present), streams tokens, reports honestly what it is
running on, and fails with one clear line instead of a stack trace.

```bash
clozn pull llama-1b                       # download a model (qwen / mistral / gemma-2b / owner/repo/file.gguf)
clozn models                              # discover local GGUFs + the backend that would run them
clozn run llama-1b "Explain entropy."     # one-shot, streams tokens to the terminal
clozn serve qwen --port 8080              # gateway + private worker; API and Studio on :8080
clozn studio --open                       # attach a browser to that already-running gateway
clozn lab qwen --open                     # optional PyTorch workbench; deliberately no /v1 API
clozn ps                                  # what's running    ·    clozn stop qwen   to stop it
```

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
clozn experiment run examples/experiment.v0.json --out result.json
clozn ci check --experiment result.json --min-target-gains 1 --max-guard-regressions 0
```

`diff-model` refuses mismatched tokenizers and labels its verdict as a sample-based screen, not proof of
quality. Experiment artifacts retain each instrumented run; the CI gate validates their matrix and
identity evidence, recomputes comparisons from raw cells, and returns a deterministic exit code. See the
[worked Qwen reasoning-SFT case study](docs/MODEL_DIFF_CASE_STUDY_QWEN_REASONING.md) for a real two-GGUF
run on a 16 GB Mac.

Chat templates come from each model's own GGUF (Qwen / Llama-3 / Mistral / Gemma / …), applied
engine-side, so pulled models chat coherently — not just Qwen. Drop the prompt — `clozn run llama-1b` —
for an interactive chat (multi-turn; `/reset` clears, `/bye` quits).

`clozn run` reuses a running `serve` for that model (warm, no reload); otherwise it spawns a temporary
gateway/worker pair and tears it down after. Product commands never bypass the gateway to call a warm
worker directly. `clozn serve` supervises the private worker and restarts it after an unexpected exit.

OpenAI clients use the documented subset of `/v1/chat/completions`, `/v1/completions`, and `/v1/models`;
unsupported behavior-bearing fields return a typed 400 instead of being silently ignored. See the exact
[endpoint/field matrix](docs/OPENAI_COMPATIBILITY.md). Clozn's CLI and Studio instrumentation use
`/api/clozn/generate`, which preserves the native state-event stream. Native event frames never leak into
an OpenAI completion stream.

Every run is debuggable — and provable — after the fact. The engine streams per-token confidence + the
alternatives it weighed; open a run in Studio's **Run Inspector** for **causal receipts** (which memory
the answer leaned on, per token), the **"Disposed to say · J-lens"** panel (per-token, per-layer, with an
unskippable provenance caption), the branch lineage tree, and the exact rendered prompt the model saw:

```bash
clozn trace                               # last runlog entry: confidence timeline + almost-said tokens
clozn inspect <clozn_run_id>              # explain any API reply from the local journal; no model needed
clozn branch                              # re-run from the most uncertain token on the alternative
clozn test cases.json                     # run-level assertions over the receipt/replay seams
```

`clozn trace` and `clozn inspect` read the same SQLite journal that heavn uses. Non-streaming OpenAI chat
exposes the exact id as `clozn_run_id` and `X-Clozn-Run-Id`; legacy text completions use the header while
keeping their standard body shape. Streaming requests are journaled but cannot expose a post-generation
run id in already-committed headers yet. `inspect` assembles confidence, active influences, and captured
concepts locally, falling back to a running gateway only when the id is not in this journal.
Queryable run
metadata lives in `~/.clozn/runs/runs.sqlite3`; large traces are immutable, content-addressed blobs under
`~/.clozn/runs/blobs/sha256`. To import an old beta JSON journal once, run `clozn migrate-runs`.
`clozn test` runs user-authored checks against a stored run: static ones (`contains` / `finish_reason` /
`min_confidence` / `card_applied` / …) read the run alone; the causal `leans_on` check re-runs the real
ablation and honestly **skips** (never a silent pass) unless you pass `--live`. Point any OpenAI client
at `clozn serve`; pass `"clozn_trust": true` in a chat request to get per-claim confidence
spans back on the wire (labeled uncalibrated).

`clozn run …` works once the repo root is on PATH; otherwise `python -m clozn run …`. Put GGUFs in
`~/.clozn/models`, set `CLOZN_MODELS=<dir>`, or list dirs in `~/.clozn/config.json`. Build the engine
first: `cd engine/core && build_gpu.bat` (GPU, CUDA) or `build_serve.bat` (CPU).

## Layout

| Dir | What |
|---|---|
| `clozn/`    | the product Python package — server/API, runlog, memory, receipts, replay, steering, readouts, the J-lens proxy, the tiny-test harness (`python -m clozn`) |
| `engine/`   | the runtime ("cloze"): C++/ggml core + `kernels/` (CUDA) + the Python `lab/` reference — runs models, emits the state-stream, harvests activations, applies steers, serves `/jlens` |
| `studio/`   | the white-box UI — the Run Inspector (receipts, trace, lineage, memory, tone dials, J-lens readouts), served by the backend |
| `protocol/` | the one state-stream contract the engine emits and the studio consumes |
| `docs/`     | architecture, the consolidated roadmap, and the honest technical account |
| `tests/`    | the model-free product suite · `scripts/` dev tooling |

The legibility-science spikes and findings (the interpretability-tax thread) live in a separate
local-only sibling repo: `../clozn-research`.

Two model substrates sit behind one spine: **autoregressive** GGUF (live) and **diffusion** LLaDA/Dream
— **dormant** (re-enter on a chat-quality open dLLM; pin-and-resolve editing, Studio's Edit view, is the
one piece kept visible meanwhile). The AR core contract includes trace, harvest, steer, and teacher-forced
`/score`; the checked-in qualification ledger records how far each exact model/quant has passed. **J-lens
runs a published Jacobian-lens method** (not a clozn invention), fit per model offline (nf4 + autograd)
and applied forward on the engine's own GGUF head; today's qualified fit covers Qwen2.5-7B, and the
quantized-transfer fidelity itself isn't claimed as novel. A second-family fit and targeted cross-family
write checks remain open.
