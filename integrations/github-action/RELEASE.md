# Dedicated Action release checklist

This checklist is documentation, not authorization to publish.

1. Create `bkawa-io/clozn-action` and copy this directory's contents to its
   repository root.
2. Pin `clozn-version` to a public release containing
   `clozn.ci-report.v1`, secure `model-lock fetch`, report receipt indexes, and
   `clozn.receipts.ci_bundle`.
3. Confirm the released engine version and manifest are public and
   cryptographically pinned before enabling run mode.
4. Run the package's unit tests, actionlint, a forked verify workflow, a
   read-only comment workflow, hosted CPU run mode, and split GPU/CPU workflow.
5. Review all third-party Action commit SHAs.
6. Create an immutable semver tag such as `v1.0.0` at the reviewed commit.
7. Move `v1` to that exact commit only after the immutable tag is public.
8. Record the Clozn/engine compatibility matrix in the release notes.
9. Publish Marketplace metadata only after the tagged workflow succeeds.

The current monorepo closeout performs none of steps 1, 6, 7, or 9.
