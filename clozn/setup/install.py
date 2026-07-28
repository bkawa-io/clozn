"""Orchestrates clozn/setup's other modules into the four user-facing operations:
plan_install(), run_install() (also serves "upgrade" -- see its docstring), run_rollback(), read_status().

Transaction shape (roadmap setup-flow steps 3-11), all of it inside one SetupLock:

    1. fetch + parse the manifest, select the narrowest matching artifact           (no disk writes)
    2. skip straight to step 6 if this exact version+platform is already installed  (no disk writes)
    3. download to ~/.clozn/engines/.download/ , hashing as it streams
    4. verify sha256 (and size) against the manifest BEFORE extraction              (clozn/setup/transport.py)
    5. extract into a throwaway ~/.clozn/engines/.staging/<uuid>/ directory         (clozn/setup/archive.py)
    6. qualify the extracted entrypoint (best-effort process-start check)
    7. atomically promote staging -> the real ~/.clozn/engines/<version>/<platform>/ (os.replace)
    8. atomically update registry.json's `installed`/`active`/`previous`            (clozn/setup/registry.py)

A failure at any step before 7 leaves the CURRENT active engine and registry.json completely untouched
-- nothing is promoted or recorded until the artifact has already passed its own qualification. This is
what "on failed install/upgrade, leave the prior active engine untouched" means in practice: the write
that would change what's active is the LAST thing this module does, not the first.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone

from clozn.setup import manifest as manifest_mod
from clozn.setup import platform_detect
from clozn.setup import registry as registry_mod
from clozn.setup import transport
from clozn.setup.archive import safe_extract
from clozn.setup.errors import ManifestError, SelectionError, SetupError
from clozn.setup.lock import SetupLock

DEFAULT_MANIFEST_URL = "https://github.com/brigittekawaguchi/clozn/releases/latest/download/clozn-engine-manifest.json"
MANIFEST_URL_ENV = "CLOZN_ENGINE_MANIFEST_URL"

_QUALIFY_TIMEOUT_S = 5.0


def resolve_manifest_url(explicit: "str | None" = None) -> str:
    """`explicit` (a CLI --manifest-url, if this ever grows one) wins, then CLOZN_ENGINE_MANIFEST_URL
    (roadmap: 'Add CLOZN_ENGINE_MANIFEST_URL only as an explicit developer/testing override'), then the
    real default. There is no per-version URL template today -- no engine release has ever been
    published (see this feature's plan, Slice E deferred) -- so `--version` is enforced by checking the
    fetched manifest's own clozn_version, not by pointing at a different URL per version. A future
    release pipeline can add that without changing this function's contract."""
    return explicit or os.environ.get(MANIFEST_URL_ENV) or DEFAULT_MANIFEST_URL


def fetch_manifest(url: str) -> dict:
    """Fetch, JSON-parse, and validate (clozn/setup/manifest.py) the document at `url`. Raises
    TransportError/ManifestError -- both SetupError subclasses."""
    raw = transport.fetch_bytes(url)
    try:
        document = json.loads(raw)
    except Exception as error:
        raise ManifestError(f"manifest at {url!r} is not valid JSON: {error}") from None
    return manifest_mod.parse_manifest(document)


def plan_install(*, manifest_url: "str | None" = None, backend_pref: str = "auto",
                  version: "str | None" = None, home: str, platform: "dict | None" = None) -> dict:
    """Compute (never execute) what `clozn setup`/`upgrade` would do. Side-effect-free: the only I/O is
    fetching the manifest and reading the on-disk registry, neither of which mutates anything -- this is
    exactly what `--dry-run --json` returns, and what run_install() computes first before doing
    anything else. Raises ManifestError/SelectionError; never returns None (roadmap rule 3)."""
    resolved_url = resolve_manifest_url(manifest_url)
    doc = fetch_manifest(resolved_url)
    engine_version = doc["clozn_version"]
    if version is not None and version != engine_version:
        raise SelectionError(
            f"manifest at {resolved_url!r} publishes clozn_version {engine_version!r}, not the "
            f"requested --version {version!r}. clozn setup has no per-version manifest URL scheme yet "
            f"(see docs/agent_roadmap/01-native-distribution-and-managed-setup.md, Slice E); point "
            f"{MANIFEST_URL_ENV} at the manifest for the exact version you want.")
    resolved_platform = platform if platform is not None else platform_detect.detect_platform()
    artifact = manifest_mod.select_artifact(doc, resolved_platform, backend_pref=backend_pref)
    key = manifest_mod.install_key(engine_version, artifact)

    reg = registry_mod.load(home)
    already_installed = key in (reg.get("installed") or {})
    return {
        "manifest_url": resolved_url,
        "protocol_version": doc["protocol_version"],
        "engine_version": engine_version,
        "platform": resolved_platform,
        "artifact": artifact,
        "install_key": key,
        "target_dir": os.path.join(registry_mod.engines_dir(home), *key.split("/")),
        "currently_active": reg.get("active"),
        "already_installed": already_installed,
        "would_change_active": reg.get("active") != key,
    }


def qualify_entrypoint(argv: list, *, timeout: float = _QUALIFY_TIMEOUT_S) -> dict:
    """Best-effort 'is this actually launchable' check: run `argv + ['--version']` and report whether
    the OS could even exec it, plus its exit code/output when it could.

    KNOWN GAP, stated rather than hidden: as of this writing engine/core/serve/server_main.cpp does not
    implement --version at all, so a real clozn-server build exiting non-zero or printing a usage error
    here is EXPECTED, not a sign of a bad download -- this function and its caller both treat `ran: True`
    with any returncode as a pass. What this DOES catch reliably, and DOES fail the install over, is
    `ran: False`: a missing file, a wrong-architecture binary, or a permissions problem -- anything that
    stops the OS from launching the process at all. Never raises."""
    try:
        result = subprocess.run(
            list(argv) + ["--version"], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as error:
        return {"ran": False, "error": f"not found or not executable: {error}"}
    except OSError as error:
        return {"ran": False, "error": f"could not launch: {error}"}
    except subprocess.TimeoutExpired:
        return {"ran": False, "error": f"did not exit within {timeout}s"}
    return {
        "ran": True,
        "returncode": result.returncode,
        "stdout": (result.stdout or "").strip()[:500],
        "stderr": (result.stderr or "").strip()[:500],
    }


def _rmtree_quietly(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _install_artifact(plan: dict, home: str, *, argv_prefix: "list | None" = None) -> dict:
    """Download, verify, extract, and qualify `plan['artifact']` into a throwaway staging directory,
    then atomically promote it to `plan['target_dir']`. Returns the installed-artifact record
    (clozn.engine-registry.v1's `installed_artifact` shape). Raises on any failure, having touched
    nothing under the real `target_dir` -- the staging directory is removed on the way out either way.
    `argv_prefix` is a test-only seam (e.g. [sys.executable] to qualify a .py fixture "engine" instead of
    a real binary); production callers never pass it."""
    artifact = plan["artifact"]
    engines_root = registry_mod.engines_dir(home)
    download_dir = os.path.join(engines_root, ".download")
    staging_dir = os.path.join(engines_root, ".staging", uuid.uuid4().hex)
    os.makedirs(download_dir, exist_ok=True)
    os.makedirs(staging_dir, exist_ok=True)
    archive_path = os.path.join(download_dir, uuid.uuid4().hex + "-" + os.path.basename(artifact["url"]))
    try:
        transport.download_to_file(
            artifact["url"], archive_path,
            expected_sha256=artifact["sha256"], expected_size=artifact.get("size_bytes"))
        safe_extract(archive_path, staging_dir)

        entrypoint = os.path.join(staging_dir, *artifact["entrypoint"].replace("\\", "/").split("/"))
        if not os.path.isfile(entrypoint):
            raise SetupError(
                f"manifest entrypoint {artifact['entrypoint']!r} was not found in the extracted archive")
        qualification = qualify_entrypoint(list(argv_prefix or []) + [entrypoint])
        if not qualification["ran"]:
            raise SetupError(
                f"the extracted engine could not be launched: {qualification['error']} -- refusing to "
                f"install it (the prior active engine, if any, is untouched)")

        target_dir = plan["target_dir"]
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        if os.path.isdir(target_dir):
            _rmtree_quietly(target_dir)   # only reached on --force re-install of an already-present key
        os.replace(staging_dir, target_dir)
        staging_dir = None   # promoted -- do not clean it up in the finally below

        final_entrypoint = os.path.join(target_dir, *artifact["entrypoint"].replace("\\", "/").split("/"))
        return {
            "version": plan["engine_version"],
            "os": artifact["os"], "arch": artifact["arch"], "backend": artifact["backend"],
            **({"cuda_major": artifact["cuda_major"]} if artifact.get("cuda_major") else {}),
            "sha256": artifact["sha256"],
            "protocol_version": plan["protocol_version"],
            "entrypoint": final_entrypoint,
            "installed_at": datetime.now(timezone.utc).isoformat(),
            "qualification": qualification,
        }
    finally:
        try:
            os.remove(archive_path)   # the downloaded archive is never kept once extracted
        except OSError:
            pass
        if staging_dir is not None:   # None only once promotion succeeded -- otherwise clean up
            _rmtree_quietly(staging_dir)


def run_install(*, manifest_url: "str | None" = None, backend_pref: str = "auto",
                 version: "str | None" = None, home: str, force: bool = False,
                 platform: "dict | None" = None, argv_prefix: "list | None" = None) -> dict:
    """Install (or, if already installed, just activate) the manifest-selected artifact, inside a
    SetupLock. Serves both `clozn setup` and `clozn setup upgrade` -- the two differ only in CLI framing
    (see clozn/cli/commands/setup_engine.py), never in mechanics: both must retain whatever was
    previously active as `previous` for rollback, and both are safe to run when nothing has ever been
    installed. Returns a result dict every state clozn doctor's 4-state contract needs; see
    clozn/cli/commands/setup_engine.py for how it is rendered."""
    with SetupLock(registry_mod.lock_path(home)):
        plan = plan_install(manifest_url=manifest_url, backend_pref=backend_pref, version=version,
                             home=home, platform=platform)
        reg = registry_mod.load(home)

        if plan["already_installed"] and not force:
            if not plan["would_change_active"]:
                return {**plan, "action": "noop_already_active", "record": reg["installed"][plan["install_key"]]}
            reg = registry_mod.record_install(
                reg, plan["install_key"], reg["installed"][plan["install_key"]], make_active=True)
            registry_mod.save(home, reg)
            return {**plan, "action": "activated_existing_install",
                    "record": reg["installed"][plan["install_key"]]}

        record = _install_artifact(plan, home, argv_prefix=argv_prefix)
        reg = registry_mod.record_install(reg, plan["install_key"], record, make_active=True)
        registry_mod.save(home, reg)
        return {**plan, "action": "installed", "record": record}


def run_rollback(*, home: str) -> dict:
    """Swap `active`/`previous` in the registry -- no download, both directories already exist on disk.
    Raises RegistryError (a SetupError) if there is nothing to roll back to, or if the previous engine's
    directory was removed out of band since it was recorded."""
    with SetupLock(registry_mod.lock_path(home)):
        reg = registry_mod.load(home)
        before_active = reg.get("active")
        reg = registry_mod.prune_missing(reg)
        new_reg = registry_mod.rollback(reg)
        registry_mod.save(home, new_reg)
        return {
            "action": "rolled_back",
            "rolled_back_from": before_active,
            "active": new_reg.get("active"),
            "record": (new_reg.get("installed") or {}).get(new_reg.get("active")),
        }


def read_status(*, home: str) -> dict:
    """The current managed-engine state: active/previous records (each self-healed against the
    filesystem -- see registry.prune_missing) plus every other installed version. Read-only from the
    caller's point of view; a pruned registry IS written back (best-effort) so a stale entry heals once,
    matching clozn/cli/engine_process.py's daemons.json `_find_warm` prune-then-write pattern, rather
    than re-discovering the same staleness on every future call."""
    reg = registry_mod.load(home)
    pruned = registry_mod.prune_missing(reg)
    if pruned != reg:
        try:
            registry_mod.save(home, pruned)
        except Exception:
            pass
    installed = pruned.get("installed") or {}
    return {
        "active": installed.get(pruned.get("active")) if pruned.get("active") else None,
        "active_key": pruned.get("active"),
        "previous": installed.get(pruned.get("previous")) if pruned.get("previous") else None,
        "previous_key": pruned.get("previous"),
        "installed": [{"key": key, **record} for key, record in sorted(installed.items())],
    }


_NO_FIXTURE_MODEL = (
    "no bundled fixture model ships with clozn (roadmap non-goal: 'claiming white-box qualification "
    "merely because core inference works'); run `clozn run <model> \"hello\"` against a real model to "
    "verify inference, or the relevant `clozn trace-circuit`/explain command for white-box capability."
)


def four_state_report(*, engine_exe: "str | None", discovery_source: "str | None" = None,
                       backend: "str | None" = None, qualification: "dict | None" = None,
                       deep: bool = False, argv_prefix: "list | None" = None) -> dict:
    """The 4 states `clozn setup`/`clozn doctor` must report SEPARATELY (roadmap: 'Never compress these
    into a single "installed" status'): Python package installed, compatible engine installed, core
    inference qualification, white-box qualification. Takes plain strings rather than an EngineDiscovery
    object on purpose -- clozn/setup never imports clozn.cli.engine_process (see this package's __init__
    docstring on the direction of that dependency); both clozn/cli/commands/setup_engine.py and
    clozn/cli/commands/doctor.py pass fields pulled out of whatever discovery/install-record object they
    already have.

    States 3 and 4 are honestly reported as 'skipped', never 'passed': no bundled fixture GGUF ships with
    clozn today (checked -- there is none in this tree), and actually qualifying inference needs a real
    model, which violates the model-free contract these commands make. `qualification` lets a caller that
    already ran qualify_entrypoint() (e.g. a fresh `clozn setup` install) pass its result through instead
    of this function re-running the process-start check a second time; `deep=True` with no pre-computed
    `qualification` runs it here (this is `clozn doctor --deep`'s only extra cost over the default sweep).
    """
    if engine_exe is None:
        engine_state = {
            "status": "missing",
            "reason": "no engine found by any discovery tier; run `clozn setup` to install one",
        }
    else:
        engine_state = {"status": "found", "exe": engine_exe}
        if discovery_source:
            engine_state["discovery_source"] = discovery_source
        if backend:
            engine_state["backend"] = backend
        if qualification is None and deep:
            qualification = qualify_entrypoint(list(argv_prefix or []) + [engine_exe])
        if qualification is not None:
            engine_state["process_start_check"] = qualification
            if not qualification.get("ran"):
                engine_state["status"] = "found_but_not_launchable"

    return {
        "python_package_installed": {"status": "passed"},
        "compatible_engine_installed": engine_state,
        "core_inference_qualification": {"status": "skipped", "reason": _NO_FIXTURE_MODEL},
        "white_box_qualification": {"status": "skipped", "reason": _NO_FIXTURE_MODEL},
    }
