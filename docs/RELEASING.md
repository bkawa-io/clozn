# Releasing Clozn

The release version has one source of truth: `clozn/__init__.py`. The first public version is `0.1.0`.

## Before tagging

1. Update `CHANGELOG.md` and confirm the version.
2. Run the local gates:

   ```sh
   .venv/bin/python -m pytest -q tests clozn
   .venv/bin/python scripts/docs/check_drift.py .
   cd studio-frontend && pnpm check && pnpm build && pnpm build:demo
   ctest --test-dir engine/core/build-cpu --output-on-failure
   ```

3. Push the release commit to `main` and wait for CI. The recorded-demo workflow deploys GitHub Pages
   from the same commit.
4. Verify the [recorded demo](https://bkawa-io.github.io/clozn/) in a fresh browser session.

## Publish 0.1.0

Create and push the immutable version tag:

```sh
git tag -a v0.1.0 -m "Clozn 0.1.0"
git push origin v0.1.0
```

Then dispatch `native-engine-release.yml` against `v0.1.0` with `publish=true`:

```sh
gh workflow run native-engine-release.yml --ref v0.1.0 -f publish=true
```

The protected workflow validates the Python package, builds and qualifies the complete native matrix,
creates the GitHub release, and only then publishes the Python distributions to PyPI. Do not recreate or
move a published version tag.

## After publishing

- Confirm the GitHub release contains three native archives and `clozn-engine-manifest.json`.
- Install the published Python distribution in a clean Python 3.11+ environment and run `clozn doctor`.
- Keep `main` on the next unreleased version before accumulating post-release changes.
