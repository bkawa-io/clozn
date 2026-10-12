# Context Investigation live qualification

Status date: 2026-10-09

This records the real-GGUF, public-gateway, and browser evidence for Studio Context Investigation.
It complements the deterministic unit/contract suite; it is not a substitute for those tests.

## Runtime

- Model: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
- Model SHA-256: `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653`
- Backend: CPU, autoregressive, 2,048-token context
- Engine build: `development`
- Worker protocol: `1.1`
- Worker capabilities used: `score_variants`, `reference_match_arms`
- Studio path: `/app/index.html`

The generic managed-runtime smoke passed first: public gateway startup/readiness, OpenAI and native
generation streams, Run and trace persistence, content-addressed trace integrity, and supervised process
cleanup all passed. Startup was 3.271 seconds; the non-streaming chat request was 0.132 seconds on this
host.

## Repeatable gateway smoke

The backend lifecycle is captured by:

```sh
python scripts/smoke/context_investigation_gateway_smoke.py \
  --base-url http://127.0.0.1:61873 \
  --native \
  --exercise-cancellation
```

The harness creates a fresh source-annotated Run on every invocation so prior trusted observations cannot
turn the qualification into a cache-only pass. It then verifies:

1. the canonical reader preserves all exact source identities;
2. `effects-search/jobs` starts and reaches a terminal completed resource;
3. the explicit native policy produces `screening_unqualified` rows and selected
   `scalar_reference` evidence in the same result;
4. only trusted scalar singleton evidence is published to the reader;
5. the locked answer query preserves the recorded answer and truthfully reports partial coverage;
6. the remove-source test returns typed direct evidence;
7. counterfactual generation returns a durable GeneratedObservation identity;
8. repeating materialization returns the same immutable child Run;
9. the child Run is readable and the response contains the parent/child Compare link.
10. a second, deliberately expensive Run enters coarse scoring, accepts cancellation only after the
    scoring phase is observable, reports real executed and skipped work, and leaves the canonical
    reader projection unchanged.

Passing report facts from the final run:

- parent Run: `run_001a121b17f0d_f27b1d`
- selected source: `src_166fabd4eab3cf859daac5cd`
- recorded answer: `ORCHID-731`
- preservation result: `diverged`
- child Run: `run_001a121b1859e_a0889d`
- observed numerical regimes: `screening_unqualified`, `scalar_reference`
- elapsed gateway workflow: 1.724 seconds

The cancellation fixture used 24 exact sources and a 384-token recorded continuation. Cancellation
was acknowledged while scalar control scoring was in flight. Its terminal typed job resource reported
1 real model measurement executed, 24 measurements skipped, and
`reader_projection.status = not_published_cancelled`; the canonical reader publication snapshot before
and after cancellation was identical. The cancellation sub-workflow completed in 6.099 seconds on the
same CPU worker.

The generated counterfactual text is not a quality assertion. This gate checks execution, evidence
identity, lifecycle, and lineage semantics only.

## Real browser pass

The served Studio bundle was exercised in a real in-app browser against the same public gateway.

- An initially unmeasured Run (`run_001a121b2586e_fb83cc`) opened at a URL-stable answer/source
  coordinate.
- Ordinary navigation showed `Not measured` and did not dispatch model work.
- Clicking **Measure context** showed the live `Coarse Scalar Scoring` phase and a cancellable state.
- Poll completion refreshed the reader in place to `Measured`, with a `+33.62 Δ nats` support
  relationship and no URL-coordinate change.
- On the native qualification Run, **Test removal** rendered `Diverged` with `0 recorded tokens matched`.
- **Generate preview** rendered the generated observation without creating a child.
- **Materialize…** opened a second explicit confirmation stating that the source Run would not change.
- **Create child Run** reused `run_001a121b1859e_a0889d` and rendered both **Inspect child** and
  **Compare parent and child** links.
- Browser console warning/error log: empty.

## Defects found by qualification

The first live pass found and fixed two integration defects that fixture-only paths did not expose:

- an explicit native-screen request was accepted by the route but short-circuited through the default
  scalar strategy before the requirement reached `EngineSubstrate.score_tokens_many`;
- `EngineSubstrate.probe_reference_match` read an uninitialized optional `steer` field, causing every
  live preservation control to fail before its deletion arm ran.

Both fixes have focused regression tests. Automatic native admission remains default-off; explicit
screen-then-refine now bypasses only the performance rollout gate, never capability checks or exact
refinement.

## Remaining live gate

No Context Investigation job-lifecycle gate remains open. The remaining product gates are listed in
`STUDIO_RELEASE_CHECKLIST.md`.
