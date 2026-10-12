# Generation-state reuse for variant scoring

Status: experimental, screening-only.

## Purpose

`/v1/score-variants` can optionally borrow the prompt KV captured by an earlier generation instead
of rebuilding the identical baseline prefix. This is an acceleration hint only. It does not change
the continuation being scored, promote native scores to trusted evidence, or create a persistent KV
cache.

The producer is an ordinary autoregressive generation with `checkpoint_on_finish: true`. Its
response returns both `checkpoint_id` and `worker_generation_id`. A later score-variants request may
send those fields together with the ID of the variant whose `prompt_ids` are the original generation
prompt:

```json
{
  "variants": [
    {"id": "baseline", "prompt_ids": [1, 2, 3, 4]},
    {"id": "without-source:a", "prompt_ids": [1, 2, 3, 8]}
  ],
  "continuation_ids": [20, 21],
  "generation_state": {
    "checkpoint_id": "ckpt-<worker-generation>-0",
    "worker_generation_id": "<worker-generation>",
    "baseline_variant_id": "baseline"
  }
}
```

## Ownership and lifetime

- A worker owns at most 16 in-memory checkpoints in insertion-order FIFO storage. Reuse does not
  increase that capacity or keep an entry alive.
- Checkpoint references are valid only for the worker generation that issued them. A process restart
  changes `worker_generation_id`.
- Lookup copies the checkpoint record before the request acquires a model context. Eviction after
  lookup cannot invalidate an in-flight request, and the copy is destroyed when the request ends.
- Restored KV lives only in the scoring request's ordinary pooled context lease. Every success and
  failure path clears request KV before releasing that lease.
- Unknown, evicted, stale, incompatible, or unhelpful state falls back to the ordinary request-local
  prefix-trie prefill and is reported in execution metadata. There is no hidden cross-request cache.

## Admission rules

Reuse is applied only when all of these checks pass:

1. The checkpoint is causal, has a valid declared `prompt_tokens` boundary, and belongs to the named
   worker generation.
2. The named baseline variant is token-for-token equal to
   `checkpoint.tokens[0:checkpoint.prompt_tokens]`.
3. The checkpoint does not contain steering. The score-variants v1 request has no intervention field,
   so mixing a steered prefix with unsteered suffix rows would be invalid.
4. Every prompt in a wave shares at least 75% of the original generation prompt. This deliberately
   restricts v1 to late-diverging variants; shallower reuse has weak compute value and showed larger
   numerical drift in qualification.
5. The checkpoint-specific physical KV accounting fits the worker context.

Serialized KV contains no logits row. If a prompt ends exactly at the common-prefix boundary, the
executor retains one fewer KV token and re-decodes that final shared token. This produces the prompt-
final logits needed to score continuation token 0. At a divergence boundary every prompt already has
a suffix, so the entire common prefix can remain resident.

The restored prefix is byte-identical to the original generation state. Re-decoded suffix and
continuation work can still differ numerically from scalar `/score` because the batch shape differs;
the response therefore remains `screening_unqualified` and `evidence_use: screening`.

## Execution shape

For each resident wave, the executor restores the checkpoint into sequence 0, evicts the checkpoint
tail at the planned retained depth, associates that retained prefix with the other sequence IDs, and
decodes only each variant's remaining prompt suffix. It then teacher-forces the shared continuation in
the existing one-token-per-sequence batches. Between waves the checkpoint is restored again; after the
last wave all sequence state is cleared.

Prompt-final and continuation output rows use the same backend softmax + target-gather path as an
ordinary native variant score. On a supported backend, exactly one float per variant/continuation token
is copied to the host; an unsupported backend falls back explicitly and reports full-vocabulary host
reduction in the response metadata.

The first implementation intentionally uses sequential suffix decodes after the original prefix. It
targets the main lifecycle win without introducing another trie executor. Request-local trie prefill
remains the fallback and is often better for very shallow divergence or wave shapes whose suffix rows
would exceed capacity.
