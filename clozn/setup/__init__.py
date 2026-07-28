"""Native distribution and managed setup (roadmap feature 01).

``clozn setup`` downloads a version-matched, checksummed native ``clozn-server`` engine artifact into
``~/.clozn/engines/<version>/<platform-key>/`` and records it in a local registry so
``clozn/cli/engine_process.py`` can find it without a source checkout or a local C++ toolchain. This
package is the client-side half of that: manifest parsing/selection, platform detection, download +
safe extraction, and the on-disk install registry. It never imports Torch or anything outside the
stdlib (pyproject.toml's ``dependencies = []`` is load-bearing -- see docs/SEAMS.md).

    clozn/setup/errors.py            SetupError and its subclasses -- no dependency on the CLI layer
    clozn/setup/platform_detect.py   detect_platform() -- os/arch/gpu backend, best-effort
    clozn/setup/manifest.py          parse_manifest() + select_artifact() -- pure, no I/O
    clozn/setup/transport.py         fetch_bytes()/download_to_file() -- the only place a URL is opened
    clozn/setup/archive.py           safe_extract() -- path-traversal/symlink-safe zip/tar extraction
    clozn/setup/lock.py              SetupLock -- advisory single-writer lock for ~/.clozn/engines/
    clozn/setup/registry.py          read/write ~/.clozn/engines/registry.json (clozn.engine-registry.v1)
    clozn/setup/install.py           orchestrates all of the above for install/upgrade/rollback/status

``clozn/cli/commands/setup_engine.py`` is the only caller that should exist outside this package; it
translates SetupError into ``clozn.cli.main.CloznError`` and adds CLI-shaped (human + --json) output.
"""
