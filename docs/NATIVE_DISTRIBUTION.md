# Native distribution

Each tagged release publishes three qualified `clozn-server` archives:

- Linux x86_64 CPU;
- Windows x86_64 CPU; and
- macOS arm64 Metal.

The release also includes `clozn-engine-manifest.json` with immutable URLs, sizes, hashes, runtime
identities, and protocol compatibility. The protected release workflow builds and tests every cell
before making the GitHub release visible, then publishes the Python sdist and wheel.

The v1 README uses the source installer because it is the single cross-platform onboarding path and
can build for configurations outside that initial matrix. Native build instructions are in
[DEVELOPMENT.md](DEVELOPMENT.md).
