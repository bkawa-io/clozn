# Clozn v1 architecture

Clozn is one public Python gateway supervising one private native GGUF worker.

```text
CLI / OpenAI client / Studio
              |
        public gateway
  validation · recording · jobs
              |
       private worker HTTP
 generation · template · score
 exact match · cancellation
```

The gateway is stdlib-only. It owns the public API, Studio assets, run journal, source mapping,
Context Effects, Minimal Context orchestration, background job state, saved-child creation, and reduced
comparison. The worker owns model loading, tokenization, model-specific chat-template rendering,
autoregressive generation, scalar teacher-forced scoring, exact-reference probes, and cancellation.

## Public boundary

The only public model API is the text subset documented in
[OPENAI_COMPATIBILITY.md](OPENAI_COMPATIBILITY.md). Studio additionally uses explicit run, context,
effects, reduction, comparison, and cancellation routes. The worker port is loopback-private and is not
a second product API.

Both CLI commands and HTTP route modules use checked-in allowlists. Restoring a removed file cannot
restore a public surface.

## Recording boundary

A successful generation response represents a durably recorded run. The gateway records the delivered
messages, rendered prompt when available, answer, finish reason, token trace, source catalogue, model and
engine identity, and generation settings. If persistence fails after generation, the public request fails
with `journal_persistence_failed`.

v1 data is isolated under `CLOZN_DATA_ROOT` or `~/.clozn/v1/`. SQLite is authoritative for run metadata;
large traces are immutable SHA-256 blobs. Saved reduction children are new immutable runs linked to an
unchanged parent.

## Evidence boundary

Context Effects removes a recorded source and scores fixed output tokens. Its links mean measured change
under that intervention, not general attribution. Minimal Context separately asks the worker to reproduce
the recorded output and stop behavior under a reduced prompt. It reports the best verified result within
its bounded budget unless stronger proof conditions are actually met.

Missing, unmeasured, unavailable, failed, low-effect, and exact-preserved states remain distinct. Source
and answer selections always resolve through recorded ranges; the UI must not imply finer evidence than
was measured.

## Lifecycle

`clozn serve MODEL` starts the gateway and one worker. Generation and analysis use the same selected
worker identity. Context Effects and Minimal Context run as cancellable jobs with explicit queued,
running, completed, cancelled, and failed states. Finished analysis results are durable; interrupted work
is not presented as complete.
