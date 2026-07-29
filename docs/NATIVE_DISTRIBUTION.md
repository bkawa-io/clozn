# Native distribution status and build identity

## Current release status

The managed setup implementation is merged in this source snapshot, but no public engine release matrix
has been published yet. Do not describe `clozn setup` as a working public install path until a
`bkawa-io/clozn` GitHub Release contains the manifest and every advertised archive has passed its
clean-machine lane.

Source contributors should continue to use the build instructions in
[DEVELOPMENT.md](DEVELOPMENT.md). `CLOZN_ENGINE_MANIFEST_URL` is a development/test override, not an
alternate release channel.

## Release manifest locations

Ordinary setup resolves manifests from the canonical repository:

- latest: `https://github.com/bkawa-io/clozn/releases/latest/download/clozn-engine-manifest.json`
- exact version: `https://github.com/bkawa-io/clozn/releases/download/v<VERSION>/clozn-engine-manifest.json`

`clozn setup --version X` fetches the immutable `vX` asset and also verifies that the document declares
`clozn_version: X`. A missing, mistagged, or replaced asset fails without changing the active engine.

## Embedded engine identity

Every `clozn-server` build has a model-free identity endpoint:

```bash
clozn-server --version
clozn-server --version --json
```

The JSON contract contains:

- engine version;
- immutable build ID;
- worker protocol version;
- compiled backend (`cpu`, `cuda`, or `metal`);
- full pinned llama.cpp commit;
- compile-time feature flags.

CMake derives the engine version from `clozn.__version__` and the llama.cpp revision from
`engine/core/third_party/bootstrap_llama.py`. Release automation must set a unique
`-DCLOZN_ENGINE_BUILD_ID=<id>`; local builds default to `development`.

The release manifest repeats the per-artifact build ID, llama.cpp commit, backend, and feature flags.
Setup downloads and hashes the archive, extracts it into staging, runs `--version --json`, and compares
the embedded identity with the selected manifest entry. Nonzero exit, malformed JSON, incompatible
protocol, unsupported backend, or any identity disagreement refuses promotion. The prior active engine
and registry remain untouched.

## Remaining release work

This contract closes build identification and immutable manifest resolution. Public availability still
requires reproducible CPU/Metal archive workflows, generated hashes/manifests, clean-machine
qualification for each advertised cell, an atomic GitHub Release, and only then publication of the
matching Python wheel and sdist.
