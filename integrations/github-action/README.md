# Clozn model gate Action source

This directory is the publication-ready source tree for the future dedicated
`bkawa-io/clozn-action` repository. Copy its contents to that repository root;
do not publish this directory itself as though the Action already existed.

The source pins an exact Clozn release, produces a GitHub job summary and JUnit
report, uploads evidence, and propagates Clozn's exit code after artifact
handling. It has two intentionally separate modes.

## Verify mode

`mode: verify` is the default and is safe for an ordinary GitHub-hosted CPU
runner. It accepts an existing `clozn.experiment.result.v0` file and runs:

```text
python -m clozn ci check --experiment RESULT.json \
  --report clozn-ci-report.json \
  --github-summary clozn-summary.md \
  --junit-report clozn-junit.xml \
  [declared budget flags]
```

The Action rejects model locks, manifests, engine versions, adapter files, and
producer commands in verify mode. This path never imports Clozn's engine setup
or model fetch modules, never starts a worker, and never downloads a model.
The gate subprocess additionally sets `CLOZN_LOCAL_ONLY=1`; installing the
exact Python package release is the only package-network step.

The resulting job summary contains the target/guard matrix, budgets, identity
drift, evidence availability, artifact checksums, and a local reproduction
command. JUnit has one test case per gate check. Exit codes are unchanged:

- `0`: gate passed
- `1`: a declared budget failed
- `2`: the artifact or orchestration was invalid
- `3`: identity-policy refusal, when supported by the selected gate

Invalid input never gets a fabricated `clozn.ci-report.v1`; the Action writes a
small action-result record plus an error JUnit/summary and propagates `2`.

## Trusted run mode

`mode: run` is a different security boundary. It is accepted only on `push`,
`workflow_dispatch`, or `schedule`, with `trusted-run: true`. Pull-request and
`pull_request_target` events are refused.

Before any model network access, run mode:

1. validates the experiment manifest and model lock;
2. computes a redacted cache identity from all model SHAs, sizes,
   quantizations, chat-template SHAs, adapter bytes, engine release, Clozn
   release, and complete suite manifest bytes;
3. restores the cache under that complete run identity;
4. fetches each lock role through `clozn model-lock fetch`, whose HTTPS,
   redirect, size, mandatory SHA, atomic-promotion, and SHA-keyed-cache rules
   are owned by Clozn;
5. installs the exact released engine;
6. invokes one trusted producer as a direct JSON argv array;
7. gates the produced evidence through the same verify operation;
8. requests `clozn stop all` in `finally`, including producer/setup failures.

`producer-argv` is not a shell string. Supported substitutions are
`{manifest}`, `{evidence}`, `{model_cache}`, and `{model:ROLE}`. Shell
interpreters are rejected. The producer remains trusted code and must come
from the protected base branch.

Example:

```yaml
- uses: bkawa-io/clozn-action@v1
  with:
    mode: run
    trusted-run: true
    clozn-version: "0.1.0"
    engine-version: "0.1.0"
    manifest: experiments/tiny-cpu.json
    model-lock: models/clozn.lock.json
    evidence: artifacts/result.json
    producer-argv: >-
      ["python", "ci/produce_experiment.py",
       "--manifest", "{manifest}",
       "--baseline", "{model:baseline}",
       "--candidate", "{model:candidate}",
       "--out", "{evidence}"]
```

## Receipt evidence

If enabled, `clozn-receipts.zip` is built only from the report-level
`receipt_index`. An indexed run is bundled only when its ID exactly matches the
embedded experiment cell and run record. Missing or mismatched evidence remains
an explicit `evidence_unavailable` entry.

The v1 bundle supports only `metadata_only`. It includes filtered identity,
generation settings, context segment/hash facts, finish status, and
input/output fingerprints. It never copies prompts, messages, responses,
rendered prompts, source text, raw tool output, arbitrary extension payloads,
or local paths.

## PR-native behavior

The failing final Action step is the useful Check. No source annotation is
created unless Clozn has an actual repository file and line; semantic model
regressions are not mapped to fake code locations or SARIF findings.

`comment: auto` attempts one concise hidden-marker PR comment. Reruns update
that comment. A read-only token or fork permission failure degrades to the job
summary and uploaded artifacts without changing the gate result.

See [SECURITY.md](SECURITY.md), the [verify example](examples/verify.yml), the
[hosted CPU example](examples/hosted-cpu.yml), and the
[split GPU/CPU example](examples/split-gpu-producer.yml).

## Publication state

No external repository, release tag, Marketplace listing, immutable release
commit, or moving `v1` tag is created by this source-tree closeout. Follow
[RELEASE.md](RELEASE.md) only after a compatible Clozn package and engine
release are publicly available.
