"""Orchestrates clozn/setup's other modules into the four user-facing operations:
plan_install(), run_install() (also serves "upgrade" -- see its docstring), run_rollback(), read_status().

Transaction shape (roadmap setup-flow steps 3-11), all of it inside one SetupLock:

    1. fetch + parse the manifest, select the narrowest matching artifact           (no disk writes)
    2. skip straight to step 6 if this exact version+platform is already installed  (no disk writes)
    3. download to ~/.clozn/engines/.download/ , hashing as it streams
    4. verify sha256 (and size) against the manifest BEFORE extraction              (clozn/setup/transport.py)
    5. extract into a throwaway ~/.clozn/engines/.staging/<uuid>/ directory         (clozn/setup/archive.py)
    6. qualify the extracted entrypoint against the manifest's embedded build identity
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
import re
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from urllib.parse import quote

from clozn.protocol import check_worker_protocol
from clozn.setup import manifest as manifest_mod
from clozn.setup import platform_detect
from clozn.setup import registry as registry_mod
from clozn.setup import transport
from clozn.setup.archive import safe_extract
from clozn.setup.errors import ManifestError, SelectionError, SetupError
from clozn.setup.lock import SetupLock

RELEASE_REPOSITORY = "bkawa-io/clozn"
DEFAULT_MANIFEST_URL = (
    f"https://github.com/{RELEASE_REPOSITORY}/releases/latest/download/clozn-engine-manifest.json"
)
VERSIONED_MANIFEST_URL = (
    f"https://github.com/{RELEASE_REPOSITORY}/releases/download/v{{version}}/"
    "clozn-engine-manifest.json"
)
MANIFEST_URL_ENV = "CLOZN_ENGINE_MANIFEST_URL"

_QUALIFY_TIMEOUT_S = 5.0
_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


def resolve_manifest_url(explicit: "str | None" = None, *, version: "str | None" = None) -> str:
    """Resolve the release manifest without silently changing release authority.

    An explicit URL (the internal test seam) wins, followed by CLOZN_ENGINE_MANIFEST_URL (the documented
    development override). Ordinary installs use the latest release; ``--version X`` uses the immutable
    ``vX`` release asset. The fetched document is still checked against ``version`` by plan_install(), so
    a mistagged or replaced asset fails closed instead of installing a different build.
    """
    override = explicit or os.environ.get(MANIFEST_URL_ENV)
    if override:
        return override
    if version is None:
        return DEFAULT_MANIFEST_URL
    if not isinstance(version, str) or not version.strip():
        raise SelectionError("--version must be a non-empty release version")
    return VERSIONED_MANIFEST_URL.format(version=quote(version.strip(), safe=""))


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
    resolved_url = resolve_manifest_url(manifest_url, version=version)
    doc = fetch_manifest(resolved_url)
    engine_version = doc["clozn_version"]
    if version is not None and version != engine_version:
        raise SelectionError(
            f"manifest at {resolved_url!r} publishes clozn_version {engine_version!r}, not the "
            f"requested --version {version!r}; refusing a mistagged or replaced release asset.")
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


def _build_info_error(build_info) -> "str | None":
    if not isinstance(build_info, dict):
        return "build-info output must be one JSON object"
    required_strings = ("engine_version", "build_id", "protocol_version", "backend",
                        "llama_cpp_commit")
    for field in required_strings:
        if not isinstance(build_info.get(field), str) or not build_info[field]:
            return f"build-info field {field!r} must be a non-empty string"
    if build_info["backend"] not in ("cpu", "cuda", "metal"):
        return f"build-info backend {build_info['backend']!r} is unsupported"
    if not _SHA1_RE.fullmatch(build_info["llama_cpp_commit"]):
        return "build-info llama_cpp_commit must be a full lowercase 40-character Git SHA"
    feature_flags = build_info.get("feature_flags")
    if not isinstance(feature_flags, dict):
        return "build-info field 'feature_flags' must be an object"
    if any(not isinstance(name, str) or not name or not isinstance(enabled, bool)
           for name, enabled in feature_flags.items()):
        return "build-info feature_flags must map non-empty names to booleans"
    compatible, reason = check_worker_protocol(build_info["protocol_version"])
    if not compatible:
        return f"build-info protocol_version is incompatible: {reason}"
    return None


def qualify_entrypoint(argv: list, *, timeout: float = _QUALIFY_TIMEOUT_S,
                       expected: "dict | None" = None) -> dict:
    """Run ``clozn-server --version --json`` and strictly validate its embedded build identity.

    ``expected`` is the selected manifest identity. Any nonzero exit, malformed output, incompatible
    protocol, unsupported backend, or manifest/build disagreement returns ``qualified: False``. The
    function never raises so doctor/status callers can render the exact failure; installers must refuse
    promotion unless ``qualified`` is true.
    """
    try:
        result = subprocess.run(
            list(argv) + ["--version", "--json"], capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError as error:
        return {"ran": False, "qualified": False, "error": f"not found or not executable: {error}"}
    except OSError as error:
        return {"ran": False, "qualified": False, "error": f"could not launch: {error}"}
    except subprocess.TimeoutExpired:
        return {"ran": False, "qualified": False, "error": f"did not exit within {timeout}s"}
    report = {
        "ran": True,
        "returncode": result.returncode,
        "stdout": (result.stdout or "").strip()[:500],
        "stderr": (result.stderr or "").strip()[:500],
    }
    if result.returncode != 0:
        return {**report, "qualified": False,
                "error": f"build-info command exited with status {result.returncode}"}
    try:
        build_info = json.loads(result.stdout)
    except (TypeError, json.JSONDecodeError) as error:
        return {**report, "qualified": False, "error": f"malformed build-info JSON: {error}"}
    error = _build_info_error(build_info)
    if error:
        return {**report, "qualified": False, "error": error}
    for field, wanted in (expected or {}).items():
        if wanted is not None and build_info.get(field) != wanted:
            return {
                **report,
                "qualified": False,
                "build_info": build_info,
                "error": (
                    f"manifest/build disagreement for {field}: manifest declares {wanted!r}, "
                    f"binary reports {build_info.get(field)!r}"
                ),
            }
    return {**report, "qualified": True, "build_info": build_info}


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
        expected_build_info = {
            "engine_version": plan["engine_version"],
            "protocol_version": plan["protocol_version"],
            "backend": artifact["backend"],
            "build_id": artifact.get("build_id"),
            "llama_cpp_commit": artifact.get("llama_cpp_commit"),
            "feature_flags": artifact.get("feature_flags"),
        }
        qualification = qualify_entrypoint(
            list(argv_prefix or []) + [entrypoint], expected=expected_build_info)
        if not qualification["qualified"]:
            raise SetupError(
                f"the extracted engine failed build identity qualification: {qualification['error']} "
                f"-- refusing to "
                f"install it (the prior active engine, if any, is untouched)")

        target_dir = plan["target_dir"]
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)
        if os.path.isdir(target_dir):
            _rmtree_quietly(target_dir)   # only reached on --force re-install of an already-present key
        os.replace(staging_dir, target_dir)
        staging_dir = None   # promoted -- do not clean it up in the finally below

        final_entrypoint = os.path.join(target_dir, *artifact["entrypoint"].replace("\\", "/").split("/"))
        build_info = qualification["build_info"]
        return {
            "version": plan["engine_version"],
            "os": artifact["os"], "arch": artifact["arch"], "backend": artifact["backend"],
            **({"cuda_major": artifact["cuda_major"]} if artifact.get("cuda_major") else {}),
            "sha256": artifact["sha256"],
            "protocol_version": plan["protocol_version"],
            "build_id": build_info["build_id"],
            "llama_cpp_commit": build_info["llama_cpp_commit"],
            "feature_flags": build_info["feature_flags"],
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
    of this function re-running the build-identity check a second time; `deep=True` with no pre-computed
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
            engine_state["build_identity_check"] = qualification
            if not qualification.get("ran"):
                engine_state["status"] = "found_but_not_launchable"
            elif not qualification.get("qualified"):
                engine_state["status"] = "found_but_not_qualified"

    return {
        "python_package_installed": {"status": "passed"},
        "compatible_engine_installed": engine_state,
        "core_inference_qualification": {"status": "skipped", "reason": _NO_FIXTURE_MODEL},
        "white_box_qualification": {"status": "skipped", "reason": _NO_FIXTURE_MODEL},
    }
