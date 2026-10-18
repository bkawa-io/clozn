# Clozn v1 live qualification

Status date: 2026-10-17  
Status: backend and browser two-feature paths passed on the reference local configuration

This gate covers the v1 promise: measure how context affects a recorded answer,
then find and save a smaller context that preserves it exactly.

## Reference configuration

- Model: `Qwen2.5-0.5B-Instruct-Q4_K_M.gguf`
- Model SHA-256: `6eb923e7d26e9cea28811e1a8e852009b21242fb157b26149d3b188f3a8c8653`
- Engine artifact SHA-256: `8a3eb93d34d15565b27cc4938a03f7b25c6763089ca3b9462fec07ce8c53f741`
- Backend: CPU, autoregressive, 4,096-token context
- Worker protocol: `1.1`
- Data: isolated v1 root supplied through `CLOZN_DATA_ROOT`

The fixture supplies four exact context blocks: launch code, launch site,
launch window, and badge color. The question asks for the launch code only.

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
