# Security model

## Verify mode and untrusted pull requests

Verify mode may inspect an artifact produced by untrusted code because it:

- invokes only `clozn ci check --experiment`;
- rejects every run-mode input;
- does not parse a model lock, fetch a model, install an engine, or start a
  worker;
- needs no repository write permission;
- emits no source annotations without real file/line provenance;
- exports metadata-only receipt bundles.

Use minimum permissions:

```yaml
permissions:
  contents: read
```

Adding `pull-requests: write` is optional and only enables the redacted
hidden-marker comment. If GitHub withholds write permission on a fork, comment
publication returns `read_only`; the summary, JUnit, artifacts, and gate exit
remain available.

Do not put secrets in experiment prompts, responses, case names, artifact
names, producer argv, lockfile URLs, or Action inputs. The receipt ZIP excludes
verbatim prompts and responses, but the full experiment evidence artifact is
uploaded because it is the gate input. Set repository-appropriate retention
and access controls.

## Run mode and privileged runners

Never execute run mode on `pull_request_target`, and never combine
`pull_request_target` privileges with a checkout of pull-request contents.
Run mode refuses all pull-request event names, even with `trusted-run: true`.

Safe patterns are:

- a protected-branch `push`;
- a reviewed `workflow_dispatch` whose workflow checks out a protected commit;
- a scheduled run pinned to the default branch;
- a self-hosted GPU producer on trusted code, followed by an ordinary
  GitHub-hosted verify job receiving only the evidence artifact.

`workflow_call` is intentionally refused because the called workflow cannot
prove from the event name alone that its checkout is a protected revision.

Run mode's `producer-argv` is direct argv, never shell text. That prevents shell
interpolation but does not make the producer untrusted-safe: Python programs
and other executables can execute arbitrary trusted repository code.

## Remote artifacts and caches

Model lock entries require HTTPS and SHA-256. Redirects are bounded and cannot
downgrade HTTPS. Optional size is enforced. Downloads are streamed to a
temporary file, verified, fsynced, and atomically promoted to a filename keyed
by SHA. Cache hits are rehashed.

The Action cache identity changes with:

- model SHA/size/quantization;
- chat-template SHA;
- adapter file bytes;
- engine release;
- Clozn release;
- experiment suite, variant, prompt/template, seed, and scoring material.

URLs and prompt material are hashed or excluded from outputs, not logged as
cache-key components.

Pin every third-party Action by full commit SHA. The examples do this for
checkout, cache, upload, and download. Review those upstream commits before
updating them.

## Cancellation and cleanup

Run mode invokes `clozn stop all` in a `finally` path after setup, fetch,
producer, or gate failure. GitHub can still terminate a job without allowing
user-space cleanup. Use ephemeral self-hosted runners or an out-of-band runner
reaper; do not treat a workflow `finally` as the sole isolation boundary.

## Reporting

Report vulnerabilities privately through the repository's security advisory
channel after the dedicated repository exists. Do not place model URLs,
credentials, signed query parameters, private prompts, or full evidence in a
public issue.
