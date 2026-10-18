# Native distribution status

This source snapshot requires a locally built `clozn-server`. No public native archive matrix or
automatic setup/rollback command is part of v1.

Build instructions are in [DEVELOPMENT.md](DEVELOPMENT.md). Every worker build exposes model-free
identity through:

```sh
clozn-server --version
clozn-server --version --json
```

The identity includes the worker protocol, build ID, backend, pinned llama.cpp revision, and compiled
features. A future packaged release must publish immutable archives and hashes, validate this embedded
identity on a clean machine, and pass the complete v1 live journey before the README can advertise it as
an installation path.
