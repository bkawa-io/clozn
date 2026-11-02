# Clozn v1 architecture

Clozn is one public Python gateway supervising one private native GGUF worker.

```text
CLI / OpenAI client / Studio
              |
        public gateway
 validation · recording · jobs
              |
       private worker HTTP
 generation · template · scoring
```

The stdlib-only gateway owns the public API, Studio assets, durable runs, source mapping, Context
Effects, Lean Context, and job state. The private worker owns model loading, tokenization,
model-specific template rendering, generation, and teacher-forced scoring.

## Public boundary

The only public model API is the text subset in
[OPENAI_COMPATIBILITY.md](OPENAI_COMPATIBILITY.md). Studio additionally uses explicit run,
context-receipt, and context-investigation routes. The worker listens on loopback and is not a second
product API.

Both CLI commands and gateway route modules use checked-in allowlists. Restoring an old source file
does not restore a public feature.

## Recording boundary

A successful generation response represents a durably recorded run. The gateway records delivered
messages, rendered prompt when available, answer, finish reason, token trace, source catalogue,
runtime identity, and generation settings. If persistence fails after generation, the public request
fails with `journal_persistence_failed`.

v1 data lives under `CLOZN_DATA_ROOT` or `~/.clozn/v1/`. SQLite is authoritative for run metadata;
large traces are immutable SHA-256 blobs.

## Evidence boundary

Context Effects removes a recorded source and scores the fixed recorded-answer tokens. Its links mean
measured change under that intervention, not truth or general attribution. Lean Context uses the same
source universe to search for a smaller prompt within fixed probability-loss tolerances, then freshly
generates both full-context and reduced-context answers.

Missing, unmeasured, unavailable, failed, and low-effect states remain distinct. Source and answer
selections resolve through recorded ranges; the UI never claims finer evidence than was measured.

## Lifecycle

`clozn serve MODEL` starts the gateway and one worker. Generation and analysis use the same selected
worker identity. Context Effects and Lean Context run as cancellable jobs with explicit queued,
running, completed, cancelled, and failed states. Context Effects evidence is durable; active job state
is process-local.
