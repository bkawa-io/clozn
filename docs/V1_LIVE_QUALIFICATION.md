# Clozn v1 live qualification

Status date: 2026-10-23
Status: exact-reduction reference gate and long-conversation Context Effects/Lean Context browser gate passed

This gate covers the v1 promise: measure how context affects a recorded answer,
then find and save a smaller context that preserves it exactly.

## Long-conversation POC qualification

The current Context Effects UI was exercised as a real user journey with a useful local model, not a
fixture. Clozn served a 12-turn essay-writing conversation through the OpenAI-compatible endpoint. Each
request resent the full message history, so the final run contained 24 non-empty messages and 20,719
characters of analyzable context.

- Model: `Qwen3.5-9B-Q4_K_M.gguf`
- Model SHA-256: `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`
- Engine SHA-256: `f8214539cc9e8fda5556c264e2fc9352287bda58ae2f541506a2515803cfdfa7`
- Host: Apple Silicon, 16 GiB unified memory
- Backend: Metal, autoregressive, 8,192-token context
- Context-arm cap for this qualification: 32
- Final Run: `run_001a160f7c88e_46b1c7`

Across the conversation, rendered prompt length grew from 165 to 4,271 tokens. Time to first token grew
from 0.623 to 8.733 seconds; observed generation throughput was about 15.5–19.7 tokens/second; and complete
turn latency was about 14.8–19.4 seconds. Those numbers exclude suspended laptop time: the qualification
was rerun under macOS `caffeinate` after sleep was identified as a source of false latency and stalled Metal
command-buffer observations.

The final recorded answer was 119 tokens. It was coherent and grounded in the conversation, but it missed
two explicit requirements: it produced four sentences instead of exactly three and did not end with the
required phrase. That made it a useful investigation target rather than a hand-picked success.

### Context Effects evidence

All 24 leave-one-source-out arms completed with trusted recorded-continuation scores. The awake-only job
took 615.4 seconds (10 minutes 15.4 seconds), which is functionally successful but not a release-ready
latency target for this model and conversation size.

- Highest total context effect: the final user instruction, `+103.55` total delta nats.
- Lowest absolute total context effect: the early assignment-rubric turn, `+0.38` total delta nats.
- Highest answer-span effect: `ordinary access can`, `15.51` absolute delta nats from one source.
- Reader coverage: 24 measured sources, 0 unavailable, 0 failed, and 124 above-floor answer ranges.

These rankings matched the conversation: the final formatting/content request dominated the final prose,
while several older planning turns had little marginal effect after their content had been repeated later.
The measurements describe influence on the recorded answer, not correctness or factual quality.

Follow-up profiling separated the unavoidable multi-arm work from implementation overhead. Replaying the
baseline plus 24 modified contexts directly through one awake 8,192-token worker took 325.3 seconds. A
target-logprob-only path was slower on this model (14.90 versus 8.92 seconds for the same single score),
and two concurrent workers doubled individual latency without improving throughput. A related-prompt
batch prototype improved the 25-score replay by only 1.4% at the qualified 8,192-token context size. At a
16,384-token context size it was 20.6% faster, but floating-point differences changed 11 of 2,832
measurement-floor classifications (0.388%). It caused no above-floor sign flips, but Context Effects is a
trusted evidence surface, so that approximate batch path was not shipped as the default. The exact scalar
path remains the reference; the practical v1 controls are bounded arm count, an honest indeterminate
progress state, and clear guidance that local analysis can take several minutes.

### Lean Context evidence

Lean Context reused 8 compatible observations and used its full budget of 16 new model executions. The
best tested candidate reduced the rendered prompt from 4,271 to 3,776 tokens, retained 22 of 24 spans
(88%), and stayed inside the balanced tolerance at 0.044 mean positive support loss and 0.508 maximum
single-token loss nats. It correctly reports “best found within this search budget,” not a minimum proof.

### Browser signoff for the new views

The bundled Studio was exercised against the completed real run. The following interactions passed:

1. selecting answer text and following bidirectional source relationships in Linked view;
2. expanding Context into a full-width heatmap, pinning a span, and reading its linked answer text;
3. collapsing Context and expanding Answer into the corresponding full-width heatmap;
4. opening Efficiency, reading highest/lowest context and answer rankings, and navigating a ranking back
   into the linked evidence view;
5. starting Lean Context, observing its job state, and reading the completed reduction and tolerance result.

The pass exposed two usability issues now covered by frontend tests: a long model batch looked frozen at
`0%`, and “16-execution budget” did not say whether reused evidence consumed the budget. Active zero-unit
jobs now use an indeterminate `Working…` state with local-model guidance, and Lean Context explicitly says
that the limit is 16 **new** executions while compatible prior measurements are free to reuse.

The pass also found three native/runtime integration defects that now have focused coverage: Qwen3.5's
template could spend the entire answer budget in hidden thinking, worker SSE error frames could be mistaken
for an empty success, and hybrid/recurrent KV caches cannot always roll back to an arbitrary prefix. Ordinary
chat rendering now disables thinking by default, worker errors propagate, and failed recurrent rollback
falls back to an exact cold pass.

One bounded-search limitation remains explicit: receipt sources do not cross message boundaries. If the
number of non-empty messages exceeds `CLOZN_MAX_CONTEXT_ARMS`, Clozn refuses bounded Lean/Minimal Context
search rather than dropping context. This qualification used a cap of 32 for its 24-message conversation.

## Exact-preservation reference configuration (2026-10-17)

- Model: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
- Model SHA-256: `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653`
- Engine artifact SHA-256: `8a3eb93d34d15565b27cc4938a03f7b25c6763089ca3b9462fec07ce8c53f741`
- Backend: CPU, autoregressive, 4,096-token context
- Worker protocol: `1.1`
- Data: isolated v1 root supplied through `CLOZN_DATA_ROOT`

The fixture supplies one Markdown context with sections for launch code, launch site,
launch window, and badge color. Clozn derives the four analysis chunks. The question asks for the
launch code only.

## Repeatable gate

Start the gateway with the reference model, then run:

```sh
python scripts/smoke/v1_acceptance.py \
  --base-url http://127.0.0.1:8080
```

The harness verifies that:

1. generation returns a durable parent Run;
2. Context Effects uses all four canonical source identities and publishes trusted evidence;
3. every Context Effects measurement uses the trusted scalar scoring contract;
4. Minimal Context uses the same source universe and returns an `exact_preserved` winner;
5. the winner reduces rendered prompt tokens;
6. the result is durably listed and readable;
7. saving creates a child whose answer exactly matches the parent;
8. repeating save returns the same child rather than creating a sibling.

## Passing evidence

- Parent Run: `run_001a14cac9ea0_c203f1`
- Minimal Context result: `mcres_d285c39c372e26db910f33d5`
- Reduced child Run: `run_001a14caca594_9d1d71`
- Recorded answer: `ORCHID-731`
- Context source count: 4
- Retained sources: 1
- Removed sources: 3
- Rendered prompt tokens: 100 → 54 (46% reduction)
- Certificate: `INCLUSION_MINIMUM`
- End-to-end elapsed time: 1.941 seconds

The gateway was then stopped and restarted against the same v1 data root.
The parent, result, and child all reopened; the result binding remained
`current`; and another save request returned the same child ID.

## Browser signoff

The bundled production Studio was exercised from a fresh Runs page against the
same reference runtime. The built-in example created a real run, Context Effects
measured the canonical source set, Minimal Context found a one-source inclusion
minimum, and the explicit save action opened the persisted parent-versus-child
comparison. The child then reopened from the Runs list after navigation away.

- Parent Run: `run_001a14cb0e982_eb55d6`
- Minimal Context result: `mcres_56474c0940ec1fbaa11169a0`
- Reduced child Run: `run_001a14cb16de6_e7f02f`
- Recorded answer: `ORCHID-731`
- Rendered prompt tokens: 96 → 75 (21.9% reduction)
- Comparison: exact answer, stop-behavior, and generation-identity matches
- Repeated save: returned the existing child rather than creating a sibling

The Runs page identified the child as a reduced replay, and deletion of a
disposable run reached the explicit irreversible-action confirmation.

## Defects found by this gate

The live pass found four integration defects not exposed by the model-free
suite:

- a friendly top-level engine build label was still compared directly with
  the canonical executable digest in the parent-runtime projection;
- exact caller source ranges separated by newline bytes fell back to one
  whole-message Minimal Context unit;
- winner source order was incorrectly treated as intervention identity during
  save;
- repeated Minimal Context save requests created duplicate children.

All four now have focused regression coverage. Browser signoff additionally
found and fixed two integration defects: zero-width end-of-sequence tokens were
being projected as empty text ranges, and the Studio compared deleted-source
array order instead of source-set membership when validating materialization.
Both browser findings also have focused regression coverage.
