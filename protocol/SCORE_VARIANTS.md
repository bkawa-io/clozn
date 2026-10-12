# Related-prompt scoring wire contract

Status: worker protocol v1, experimental and screening-only.

This document specifies the direct worker operation used to score one recorded continuation under
several related tokenized prompts. The operation is generic: the worker receives token IDs and does
not interpret sources, RAG, deletions, or counterfactual meaning.

The normative implementation identifiers are:

- contract and conformance fixture: `protocol/fixtures/score_variants.json`;
- endpoint: `POST /v1/score-variants`;
- advertised capability: `score_variants: true` in `GET /health`;
- response schema: `clozn.score-variants-worker.v1`;
- model-free validator: `engine/core/include/clozn/score_variants_protocol.hpp`.

The C++ and Python constants are generated from the fixture. Both validators consume the fixture's
same conformance request and response and fail closed when the required v1 fields below disagree.
Additive response telemetry may be retained by clients, but it may not alter the v1 evidence status.

## Request

```json
{
  "variants": [
    {"id": "baseline", "prompt_ids": [1, 2, 3]},
    {"id": "without-source:a", "prompt_ids": [1, 2, 8]}
  ],
  "continuation_ids": [20, 21],
  "generation_state": {
    "checkpoint_id": "ckpt-worker-0",
    "worker_generation_id": "worker-generation",
    "baseline_variant_id": "baseline"
  }
}
```

The top-level object is closed. It contains exactly `variants`, `continuation_ids`, and optionally
`generation_state`. Each variant object is closed and contains exactly:

- `id`: 1–128 ASCII bytes from `[A-Za-z0-9_.:-]`, unique within the request;
- `prompt_ids`: a non-empty array of JSON integers in `[0, vocab_size)`.

`continuation_ids` is a non-empty array of JSON integers in `[0, vocab_size)`. Booleans, numeric
strings, and integral-looking floats such as `1.0` are not token IDs. There are 1–64 variants. Every
`prompt_ids + continuation_ids` sequence must fit in `n_ctx`, and the sum of logical prompt rows may
not exceed `16 * n_ctx`.

`generation_state` is an optional acceleration hint, not an evidence input. When present it is a
closed object with exactly three non-empty strings:

- `checkpoint_id`, at most 512 bytes;
- `worker_generation_id`, at most 128 bytes;
- `baseline_variant_id`, which must name one request variant.

Checkpoint reuse is admitted only under the stricter identity, prefix, steering, and capacity rules
in [SCORE_VARIANTS_GENERATION_STATE_REUSE.md](../docs/SCORE_VARIANTS_GENERATION_STATE_REUSE.md).
An unavailable or ineligible checkpoint falls back to request-local prefix reuse and is reported in
metadata; it does not change the continuation or scores being requested.

## Successful response envelope

```json
{
  "schema_version": "clozn.score-variants-worker.v1",
  "status": "completed",
  "continuation_ids": [20, 21],
  "results": [
    {
      "id": "baseline",
      "status": "completed",
      "prompt_token_count": 3,
      "target_logprobs": [-1.25, -0.75]
    }
  ],
  "execution": {
    "strategy": "native_prefix_reuse_step_batch",
    "numerical_regime": "screening_unqualified",
    "evidence_use": "screening",
    "qualified": false,
    "execution_identity": {
      "algorithm_id": "clozn.score-variants.native-prefix-trie-step-batch-target-reduce-generation-reuse.v3",
      "worker_generation_id": "worker-generation",
      "model_sha256": "<optional lowercase SHA-256>",
      "tokenizer_sha256": "<optional lowercase SHA-256>"
    },
    "waves": []
  },
  "metrics": {}
}
```

The worker returns result rows in caller order and echoes `continuation_ids` exactly. A result row has
the same `id` and exact `prompt_token_count` as its request row. Row status is one of `completed`,
`failed`, `cancelled`, `not_started`, or `unsupported`.

- A `completed` row carries `target_logprobs`, with one finite number per continuation token, and no
  failure reason.
- Any other row carries a non-empty `reason_code` and no partial `target_logprobs`. Failed rows also
  carry a non-empty human-readable `error`.

The top-level status must agree with the rows:

- `completed`: every row completed;
- `partial`: at least one but not every row completed;
- `failed` or `cancelled`: no row completed.

## Execution metadata

V1 is always an unqualified native screen:

- `strategy` starts with `native_`;
- `numerical_regime` is `screening_unqualified`;
- `evidence_use` is `screening`;
- `qualified` is `false`;
- `execution_identity` includes non-empty `algorithm_id` and `worker_generation_id`; model and
  tokenizer hashes are optional but, when present, are lowercase SHA-256 digests.

`waves` is a non-empty ordered array. Each request variant appears in exactly one wave. Every wave
contains:

- zero-based `wave_index` and execution `status`;
- non-empty caller-ordered `variant_ids`;
- `unique_prompt_count`;
- non-negative `logical_prompt_rows`, `deduplicated_prompt_rows`, `physical_prompt_rows`,
  `logical_continuation_kv_rows`, `continuation_kv_rows`, and `required_kv_rows`;
- `accounting: planned_capacity`;
- `prefill_strategy`, either request prefix-trie or generation-checkpoint prefix reuse;
- a `generation_state_reuse` record containing applied status, reason, prefix depths, decoded prompt
  rows, and required KV rows.

Physical prompt rows cannot exceed logical prompt rows. Required KV rows cover physical prompt plus
retained continuation KV. Duplicate complete prompts may share one physical result but must fan back
out into distinct caller-ordered result rows.

The execution envelope also reports:

- `full_vocab_host_reduction`: whether any observed score used host log-softmax;
- `target_logprob_reduction`: `backend_softmax_target_gather` or `host_log_softmax`;
- aggregate `generation_state_reuse` request/status/applied-wave/lifecycle facts.

`metrics` reports counts and measured durations. Its current fields include variant, unique-prompt,
duplicate, and wave counts; planned logical/deduplicated/physical prompt and continuation rows;
successful decode rows/calls; host-output floats; and prefill, continuation, and wall seconds. Metrics
describe execution and capacity; they are not evidence qualification.

## Errors

Errors are JSON objects:

```json
{
  "error": "human-readable detail",
  "code": "request_invalid",
  "qualified": false,
  "evidence_use": "screening"
}
```

Malformed or over-limit requests return HTTP 400. A non-autoregressive worker returns HTTP 409 with
`unsupported_model_mode`. An internal response-contract violation returns HTTP 500. Stable validation
codes include `request_invalid`, `variant_count_invalid`, `variant_id_invalid`,
`duplicate_variant_id`, `prompt_token_ids_invalid`, `continuation_token_ids_invalid`, and
`request_limit_exceeded`. Callers must use `code`, not parse the human-readable `error` string.

## Evidence and lifecycle invariants

1. These scores are screening candidates, never trusted observations. Proof-critical, selected, and
   borderline effects are scalar-remeasured before publication.
2. Prefix reuse is token-exact. Branches never rejoin after divergence.
3. The recorded continuation is teacher-forced identically for every variant; no sampling occurs.
4. Only the target-token log probability is returned. Full-vocabulary normalization still occurs.
5. KV and checkpoints are request/worker-generation scoped. Success and failure clean request state
   before returning the context lease.
6. Unsupported or unprofitable product workloads may use the scalar path, but must label that path
   `scalar_reference`; they must not fabricate a native response.

Qualification fixtures and rollout policy are recorded in
[CONTEXT_SCORING_EFFICIENCY_PLAN.md](../docs/CONTEXT_SCORING_EFFICIENCY_PLAN.md).
