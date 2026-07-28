# Runtime split — what lives where

> **Current status:** the online product is one stdlib-only Python gateway supervising one private C++
> autoregressive GGUF worker. Offline research and calibration code is not a selectable product
> substrate. See [CAPABILITIES.md](CAPABILITIES.md) for release and qualification status.

## The decision

**The product is forward-only; gradient work is offline.** The installed gateway never imports Torch,
Transformers, or a live research model. Offline jobs may produce model-scoped dials, J-lenses, or other
artifacts, and the product applies them only after their identity and dimensions validate.

The earlier prompt-card/learned-prefix memory system and its user-facing workbench were removed on
2026-07-27. A few compatibility readers and archived research modules retain old field names; they do
not make those features current.

## Deployed topology

```text
OpenAI/Ollama client · CLI · Studio
                    │
                    ▼
public loopback Python gateway (`clozn serve`, stdlib-only)
API compatibility · run journal · context receipts · comparisons · actions
                    │ protocol 1.0
                    ▼
private loopback C++ worker
template · generate · sample · tap · score · steer · optional J-lens
```

`clozn serve` launches and supervises the pair; `clozn studio` attaches to the existing gateway.
`/v1/*` is the compatibility surface and `/api/clozn/generate` is the native instrumented stream.
`clozn smoke MODEL` is the repeatable topology gate.

## Ownership

### Product worker — `engine/core`

| Capability | Status | Evidence |
|---|---|---|
| Autoregressive GGUF generation and streaming | merged | engine tests plus managed real-runtime smoke |
| Embedded per-model chat templates | merged | renderer tests and Wave 1 full-Jinja evidence |
| Temperature/top-k/top-p/repetition/seed sampling | merged | worker/gateway handoff tests |
| Activation reads and steering writes | merged, qualification-scoped | `/harvest`, `/state`, `/intervene`, targeted tests |
| Teacher-forced scoring | merged | `/score`, deep smoke, receipt tests |
| LoRA attachment | merged, one adapter | fail-closed engine tests and three-arm live proof |
| J-lens apply/read | merged when a qualified sidecar is loaded | contract tests; exact Qwen2.5 row qualified |
| Continuous batching | not built | one active generation path |

### Product gateway — `clozn/server`

- Strict OpenAI and Ollama envelopes, request admission, cancellation state, and worker supervision.
- Run journaling, content-addressed trace blobs, migrations, privacy controls, and export.
- Context receipts, source measurement, comparisons, experiments, replay planning, and corrective
  action/undo.
- The one `EngineSubstrate` adapter used by product routes.

### Offline jobs and research

- Dial derivation/calibration and model qualification.
- J-lens fitting and artifact export.
- Research scripts and retained historical modules, with no public serving command.

## Artifact boundary

| Artifact | Producer | Consumer | Validation |
|---|---|---|---|
| model-scoped dial bundle | offline calibration | engine steering | checkpoint/substrate identity and safe ranges |
| J-lens manifest and matrices | offline fit/export | engine `/jlens` | manifest, hashes, dimensions, exact qualified GGUF |
| SAE bundle | external or offline export | engine readout | model/layer/dimension identity |
| behavior profile | product import/export | gateway steering | versioned schema; legacy cards are carried but never applied |
| run rows and trace blobs | gateway | CLI, Studio, receipts | transactional migrations and digest verification |
| managed engine archive | release pipeline | setup installer | versioned manifest, checksum, embedded build identity |

## Remaining boundaries

1. A client disconnect stops gateway consumption, but engine-loop cooperative cancellation remains
   limited.
2. Generation/steering mutations are serialized; there is no vLLM-style continuous batching.
3. Loopback is the supported deployment. Remote exposure requires an explicit auth/TLS design.
4. Core-qualified does not mean dial-, lens-, SAE-, adapter-, or structured-I/O-qualified.
5. Managed setup is merged but not released until the public archive matrix exists.
