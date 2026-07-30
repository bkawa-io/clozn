# Execution-fork planning contract

`clozn.replay.execution_fork.plan_execution_fork()` is the model-free gate in front of the private
worker's `/v1/execution-fork` primitive. It emits `clozn.execution-fork.v1`; it does not generate,
persist a child, or mutate a checkpoint.

The classifier has exactly three outcomes:

- `exact_execution_fork`: a compound checkpoint reference, token boundaries, prompt boundary,
  worker process generation, complete runtime key, and one supported intervention all passed.
- `reconstructed_replay`: no checkpoint reference was supplied, and the final rendered prompt plus
  complete response token boundaries support the legacy text reconstruction path.
- `unavailable`: a requested path cannot be executed honestly.

The checkpoint reference is the pair `checkpoint_id` + `worker_generation_id`, with its parent run,
availability state, prompt-token boundary, and token-history length. Supplying an exact reference is
a binding choice: missing, expired, stale, wrong-parent, or incompatible references return
`unavailable`. They never fall through to reconstructed replay.

The runtime key is a digest over the exact GGUF hash, template fingerprint, engine build, context
size, backend, full adapter identity/artifact/scale, and white-box flags. An attached adapter whose
artifact identity is unavailable makes the plan unavailable; it is not treated as the base model.

`position` is a zero-based parent response-token index. The worker truncation point is
`checkpoint.prompt_tokens + position`. Position zero therefore uses
`prompt_boundary_reprefill`; later response positions use `generated_token_live_kv`. Both remain
planned claims until execution confirms the worker receipt and the required unchanged control.

Every eligible plan reserves immutable child lineage (`parent_run_id`, `source: fork`, and the
change hash). A later executor must create a child rather than rewrite the parent, record the
unchanged-control result, and advance the artifact phase and child receipt status.

## Exact gateway execution

`POST /runs/<id>/execution-fork/plan` creates a current, identity-qualified plan without generation.
`POST /runs/<id>/execution-fork` accepts that plan and rechecks the parent fingerprint, runtime key,
worker generation, checkpoint, and exactness regime before touching the worker. These endpoints are
separate from the legacy `/runs/<id>/fork` text-splice path.

Execution first calls the worker with `intervention: {"type": "none"}` for exactly the parent suffix
horizon. Both token IDs and decoded text must match the stored suffix. Divergence or control failure
is terminal and the requested intervention is not called.

Only a successful intervention is recorded as a `source: fork` child run. Its
`execution_fork` field contains the completed artifact with the allocated child ID. Failed controls,
stale plans, intervention failures, cancellation, and persistence failures are not fabricated as
generation runs; their immutable terminal artifacts live in the execution-fork result store and are
read with `GET /execution-forks/<execution_id>`.

The child journals the runtime and worker that actually executed the fork, not a wholesale copy of
the parent's process receipt. A managed child rewrites `meta.model_routing` to the current qualified
runtime key and worker generation. Sampling interventions record the worker's fully resolved
checkpoint-plus-override decode regime. A raw-vector steering intervention clears inherited dial
names and records the exact vector/layer/coefficient provenance; clearing steering removes the
parent's steering claim.

## Recorded-parent checkpoint capture

`POST /runs/<id>/execution-fork/checkpoint` prepares the exact reference required by the planner. It
returns `clozn.checkpoint-reference.v1` with one of three statuses:

- `available`: the checkpoint was captured on the matching worker generation and its unchanged
  prompt-boundary exact-fork control matched the parent token IDs and decoded text.
- `unavailable`: required parent evidence was never sufficient to attempt capture.
- `failed`: a worker operation failed, returned inconsistent evidence, or the unchanged control
  diverged.

The capture path uses the matching worker's `/score` response to obtain prompt token IDs. It sends
the exact recorded output IDs separately as `continuation_ids`; it never tokenizes prompt and output
as one string across the BPE boundary. The worker-returned prompt count must match the original
`meta.prompt_tokens` receipt. The complete board is then checkpointed with `prefill_to` equal to that
recorded prompt boundary, so the prompt is rebuilt in one batch and generated tokens are decoded
one at a time.

Sampled parents require their complete fixed-seed decode configuration. `rng_draws` is the number
of committed recorded output tokens. Greedy parents store no sampler state. A parent with active
steering is eligible only when `meta.execution_fork_steering` retains the exact raw vector, layer,
coefficient, and a digest of the recorded active dials. Dial names are not re-derived from the live
library because calibration drift could change the vector.

The returned reference reports the worker's actual `size_bytes` and always states
`storage: worker_memory`, `durability: ephemeral`, `pinned: false`, and
`eviction_policy: bounded_fifo`. It expires on worker restart, FIFO eviction, or gateway shutdown.
This endpoint does not persist, pin, export, import, truncate, or delete checkpoints. It also does
not alter the parent or create a proof child run; the reusable unchanged-control seam records only
hash-based proof inside the checkpoint-reference receipt.
