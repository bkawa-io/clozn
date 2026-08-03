# ADR 010 — Exact appended-turn continuation

Status: Implemented and live-accepted on 2026-08-02.
Date: 2026-08-02

## Decision

Clozn will add a distinct Answer Time Machine operation that continues an immutable historical
conversation state with one newly appended user turn. The public action is not an extension of the
existing structural `POST /runs/<id>/branch` route and is not the same operation as the exact
same-prompt `POST /runs/<id>/time-machine/branch` replay.

The operation is exact only when it:

1. resolves exactly one immutable organic source run for the requested turn;
2. imports the source run's durable pinned checkpoint after verifying its content, capture regime, and
   complete runtime identity;
3. proves that the source model, tokenizer, template, adapter, generation settings, and checkpoint
   identity match the selected runtime;
4. renders and tokenizes the requested conversation only to derive and validate the append suffix;
5. sends only that validated suffix to the worker;
6. leaves the historical KV state untouched, decodes appended tokens sequentially at batch shape one,
   and generates from the resulting state under one worker lease;
7. persists a new immutable child and a closed `clozn.time-machine-continuation.v1` receipt.

If any precondition fails, the action returns a typed `unavailable`, `failed`, or `cancelled` receipt.
It never falls back to transcript reconstruction, full-prefix re-prefill, or the structural branch
route.

## What “exact” means

The exactness claim is deliberately narrow: the selected identity-qualified checkpoint state is
restored byte-for-byte and the worker executes only the validated newly appended token suffix. The
checkpoint may be organically retained live KV or the result of Clozn's separate verified
prompt-boundary reprefill plus unchanged-control proof; the receipt records which. This is not a claim
that the new response will equal a response produced by rebuilding the full prompt in a fresh request.

Those executions have different batch histories. In particular, an ordinary fresh prompt may prefill
many tokens in one batch, while this operation preserves the historical state and appends each new token
at batch shape one. Clozn records that regime and sets `fresh_full_prompt_equivalence_claimed: false`.
Sampling and backend nondeterminism remain ordinary properties of generating a new suffix; they do not
weaken the claim about the restored historical prefix.

“Exact continuation” therefore requires all of the following receipt facts:

- `historical_prefix_recomputed: false`;
- `historical_prefix_retokenized_for_execution: false`;
- `append_only_execution: true`;
- `append_decode_regime: sequential_single_token`;
- `structural_fallback_used: false`;
- verified checkpoint, worker generation, runtime, tokenizer, template, adapter, and settings identity.

The exactness claim is absent unless the worker finishes, the child is durably recorded, and all of
those fields are confirmed.

## Public request and terminal receipt

The gateway route is:

```text
POST /runs/<requested-run-id>/time-machine/continue
```

Its v1 JSON request is closed:

```json
{
  "turn": 1,
  "user": {"content": "What changed after the restart?"},
  "max_tokens": 256
}
```

`turn`, `user.content`, and `max_tokens` are the only v1 fields. Sampling, adapter, template, model,
and other generation settings are inherited exactly from the source run. A later product decision to
allow overrides requires a new receipt regime; an override must not be smuggled into v1.

The terminal response and stored artifact use `clozn.time-machine-continuation.v1`. The receipt is
closed and records:

- requested run, source run, source turn, and resolution method;
- checkpoint reference or durable pin, original and executing worker generations, state hashes, and
  restart safety;
- model, tokenizer, template, adapter, engine, context, backend, and generation-settings identity;
- the appended token IDs and their boundary proof;
- sampler state provenance;
- the worker result, cancellation state, exactness claim, unavoidable differences, and typed failure;
- requested-parent/source-checkpoint/child lineage and parent immutability.

Raw user content belongs in the immutable child run. The receipt stores its SHA-256 and byte length,
plus the concrete token IDs that crossed the worker boundary.

## Source resolution and pin eligibility

The latest completed conversational turn may resolve to the requested run itself. An earlier turn must
resolve through the existing fail-closed organic session-prefix rule: exactly one non-derived session
run whose sanitized completed messages match the requested prefix byte-for-byte.

A completed public v1 continuation requires `durable_pin_import`: the exact source run's
content-addressed pin is imported into a compatible worker. The pin survives restart; its source
worker generation remains audit provenance and is not compared to the new process generation. A
process-local checkpoint can still be a useful Time Machine cache candidate, but it is not sufficient
readiness for this restart-safe public action. The private worker primitive accepts a live checkpoint
because every successful import materializes one; that implementation detail is not a public fallback.

Each source also records `capture_regime`:

- `organic_live_kv`: checkpoint bytes retained from the source execution;
- `verified_prompt_boundary_reprefill`: the existing checkpoint-capture path rebuilt the source with
  the recorded prompt batch shape and one-token generated decode shape, and its unchanged control
  matched before the checkpoint was exposed or pinned.

The latter is an allowed checkpoint-creation step, not an execution fallback. The continuation worker
still receives only checkpoint bytes and append tokens; it never rebuilds the prefix. An unverified
text/token reconstruction is not an allowed source.

Unpinning remains child-aware. A pin that backs a continuation child cannot be silently removed;
callers must receive the existing dependency refusal or make an explicit cascade choice.

## Append derivation and boundary proof

The gateway uses the exact source template and tokenizer to render the historical conversation plus the
new user turn and assistant generation prefix. This full render is validation evidence only; its
historical prefix is never submitted to the worker.

The derivation succeeds only when the source token history is an exact prefix of the fully rendered token
history. The receipt records both token-history hashes and counts, the concrete suffix token IDs, the
suffix hash, the rendered-suffix hash, and `prefix_match: true`. A BPE boundary mismatch, template drift,
missing generation prefix, malformed token ID, or empty suffix is terminal.

The private worker receives the already validated suffix. It does not tokenize user text and does not
re-render chat templates.

## Private worker primitive

The C++ worker exposes `POST /v1/time-machine/continue`, a private append-and-generate primitive under the gateway-only worker
protocol. Its request carries:

- checkpoint ID and executing worker generation ID;
- expected restored position and historical token-history hash;
- expected checkpoint-payload hash;
- append token IDs and append hash;
- generation limit, request/cancellation ID, and an optional checkpoint-on-finish flag.

Sampler and steering state are checkpoint-owned and are returned as preservation evidence; the request
cannot override them. Adapter state is worker-load identity, verified by gateway routing and the
checkpoint import codec, and likewise cannot be supplied or overridden on this closed private request.

The primitive performs restore, identity/position validation, sequential append, and generation under
one context lease. No intermediate restored or partially appended state is observable by another
request. Cooperative cancellation uses the existing `/cancel` registry. Cancellation before child
persistence creates no run and returns a terminal cancelled receipt.

The worker rejects stale generations, unknown checkpoints, position/history/payload mismatch, invalid
token IDs, unsupported execution regimes, and cancellation with stable machine-readable codes. Corrupt
imports and model/engine/backend/adapter mismatches fail earlier in the existing checkpoint-import
codec. It may optionally return a checkpoint for the new child, but failure to persist that optional
checkpoint does not permit rewriting the parent.

## Restart behavior

An ephemeral checkpoint expires on worker restart, FIFO eviction, and gateway shutdown. If it expires,
the continuation is unavailable.

A durable pin may be imported after restart only when the checkpoint codec verifies the blob and payload
digests and the current worker matches the recorded engine/model/tokenizer/template/context/backend and
adapter identity. The executing worker generation is expected to differ after restart and is recorded as
an unavoidable process difference. It is never treated as a model-state mismatch.

There is no restart fallback to re-prefill.

## Failure and cancellation semantics

The receipt enumerates failure stages (`request`, `source_resolution`, `checkpoint`, `identity`,
`append_derivation`, `worker_restore`, `worker_append`, `generation`, `persistence`) and stable codes.
Unknown schema versions fail closed. Unknown worker codes are translated to `worker_protocol_error` with
the original detail retained only in logs, not promoted into an unversioned artifact field.

The gateway checks cancellation before worker invocation, the worker checks cooperatively during append
and generation, and the gateway checks again before persistence. A cancellation never creates a partial
child. If persistence succeeds before a late cancellation arrives, the terminal result is completed and
the immutable child remains authoritative.

## Implemented slices and acceptance

1. The closed schema and valid/invalid fixtures are registered as
   `clozn.time-machine-continuation.v1`.
2. The C++ primitive and model-free protocol test cover stale generation, malformed append,
   cancellation, and position/history/payload checks; the existing checkpoint codec owns durable
   import identity validation.
3. Python owns source resolution, durable-pin hydration, append derivation, typed worker invocation,
   terminal-result persistence, and atomic immutable child creation.
4. Studio presents exact continuation separately from exact same-prompt replay and structural
   alternate-question branching, and exposes source-turn pin lifecycle/readiness.
5. `scripts/smoke/time_machine_continuation_gateway_smoke.py` is the repeatable live acceptance battery.
   Its 2026-08-02 Qwen2.5-0.5B Metal run completed 18 checks with 0 failures and 0 skips, including
   latest and earlier turns, restart between pin and continuation, stale generation, cancellation,
   missing-pin refusal, lineage, parent immutability, and cleanup. Identity/import mismatch remains
   covered by the checkpoint codec's focused unit and live batteries; this run had no compatible LoRA
   artifact with which to construct a meaningful live adapter-mismatch arm.

## Rejected alternatives

- Concatenate text and submit the full conversation again: structural replay, not exact continuation.
- Retokenize and re-prefill the full historical prompt inside the continuation operation: useful only
  in the separate checkpoint-capture/control stage; it is never an append-execution fallback.
- Reuse `/v1/execution-fork` with an invented response-token intervention: that primitive branches inside
  an existing response and does not represent a chat-template append boundary.
- Accept arbitrary generation overrides in v1: this obscures whether sampler/settings state was
  preserved.
- Treat a matching model filename as identity: exact continuation requires the full runtime and artifact
  identity already used by execution-fork and checkpoint import.
