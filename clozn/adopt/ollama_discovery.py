"""Read-only discovery of a real Ollama install: its API/executable, and its models.

FOLLOWS THE SPEC'S DETECTION ORDER (notes/agent_roadmap/11-adopt-ollama.md "Discovery flow"), because
the API is the stable, documented surface and the on-disk layout is not (unversioned, undocumented
outside Ollama's own source -- see this module's own risk note below):

    1. Configured OLLAMA_HOST (or an explicit override) -- an HTTP GET /api/version round trip.
    2. The `ollama` executable on PATH -- `ollama --version` as a subprocess probe.
    3. The known default local endpoint (127.0.0.1:11434) -- the same /api/version round trip, tried
       only if step 1 didn't already try that exact host.
    4. Known on-disk model storage locations, ONLY as a last-resort fallback, and always flagged with a
       warning: unlike the other three, this source cannot see the daemon's actual view of what's
       installed, cannot distinguish a model from garbage-collected blobs, and its exact layout is a
       version-sensitive assumption (see known_storage_paths()'s docstring).

Whichever source answers first is recorded as `source` in discover()'s result and is the one piece of
provenance every later adoption-receipt field should point back to (spec: "Record discovery source and
version where available").

NETWORK POLICY: every HTTP call here goes through plain `urllib.request.urlopen`, which
`clozn/__init__.py` has already wrapped with `clozn.network_policy`'s process-wide local-only guard and
privacy-safe outbound ledger by the time this module is ever imported (clozn.network_policy.install_
urllib_guard() runs at package import). Ollama's default host is loopback, so it is allowed even under
local-only mode; a custom OLLAMA_HOST pointing off-box is not, and the resulting
`network_policy.LocalOnlyViolation` is deliberately NOT swallowed here (see probe_endpoint) -- "the
network policy blocked this" and "nothing is listening" are different facts and must not collapse into
the same silent "not found".

MODEL-FREE TESTING: `list_models`/`show_model`/`probe_endpoint`/`executable_version` all reach urllib or
subprocess exactly once per call and take no other global state, so tests/test_ollama_discovery.py
exercises every branch by monkeypatching `urllib.request.urlopen` and `subprocess.run` -- no real Ollama
process is ever started by the unit suite.

WHAT THIS MODULE DELIBERATELY DOES NOT DO: it does not parse `ollama list`'s plain-text table output.
That table's column layout is not a documented contract and has changed across Ollama releases; treating
it as a data source would be exactly the "guessing at an undocumented format" this module's own risk
note warns against for the storage fallback. If the daemon isn't reachable over HTTP, model listing is
unavailable and callers should say so plainly (roadmap rule 3: no silent fallback) rather than scrape a
table.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.error
import urllib.request

from clozn import network_policy

OLLAMA_DEFAULT_HOST = "http://127.0.0.1:11434"


def _normalize_host(host: str) -> str:
    host = str(host).strip()
    if not host:
        raise ValueError("empty Ollama host")
    if "://" not in host:
        host = f"http://{host}"
    return host.rstrip("/")


def probe_endpoint(host: str, timeout: float = 2.0) -> dict | None:
    """GET `{host}/api/version`. Returns `{"version": "..."}` (version may itself be None if the
    response didn't carry one) on any reachable, JSON-object response; returns None for every ordinary
    "nothing is there" failure (connection refused, timeout, non-JSON body, HTTP error, DNS failure).

    A `network_policy.LocalOnlyViolation` is the one exception NOT swallowed here: local-only mode
    blocking a non-loopback host is a policy decision, not evidence that Ollama isn't running, and the
    caller (discover()) turns it into an explicit warning rather than a silent miss.
    """
    try:
        request = urllib.request.Request(f"{host}/api/version", headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except network_policy.LocalOnlyViolation:
        raise
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    return {"version": payload.get("version")}


def executable_version(exe_path: str | None, timeout: float = 5.0) -> str | None:
    """Best-effort `<exe> --version` -- mirrors clozn.cli.commands.models._detect_vram_gb's own
    "Returns None if the tool isn't there or times out" contract. Never raises: an absent/broken
    executable is the common case, not a bug in this probe."""
    if not exe_path:
        return None
    try:
        out = subprocess.run([exe_path, "--version"], capture_output=True, text=True, timeout=timeout)
    except Exception:
        return None
    if out.returncode != 0:
        return None
    text = (out.stdout or "").strip() or (out.stderr or "").strip()
    return text or None


def known_storage_paths() -> list[str]:
    """Every plausible Ollama model-storage directory, most-likely-correct first, de-duplicated.

    THESE PATHS ARE AN ASSUMPTION, NOT A VERIFIED CONTRACT. `OLLAMA_MODELS` is Ollama's own documented
    override (checked first). `~/.ollama/models` is Ollama's documented POSIX/macOS default. The
    Windows default is the least certain of the three -- Ollama for Windows is newer than the POSIX
    build and this module was written without a Windows Ollama install to check against, so both a
    `~/.ollama/models` guess (os.path.expanduser resolves `~` to the user profile dir on Windows too)
    and a `%LOCALAPPDATA%\\Ollama\\models` guess are included. Treat any storage-fallback discovery
    result as lower-confidence than an API or executable answer (see discover()'s docstring) until this
    has been checked against a real Windows Ollama install.
    """
    candidates = []
    env_dir = os.environ.get("OLLAMA_MODELS")
    if env_dir:
        candidates.append(env_dir)
    candidates.append(os.path.join(os.path.expanduser("~"), ".ollama", "models"))
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            candidates.append(os.path.join(local_app_data, "Ollama", "models"))
    seen: set[str] = set()
    out = []
    for candidate in candidates:
        resolved = os.path.abspath(os.path.expanduser(candidate))
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def known_storage_path() -> str | None:
    """The first known_storage_paths() entry that actually exists on disk, or None."""
    for path in known_storage_paths():
        if os.path.isdir(path):
            return path
    return None


def _result(*, found: bool, source: str | None, warnings: list[str], host: str | None = None,
           version: str | None = None, executable_path: str | None = None,
           storage_path: str | None = None) -> dict:
    return {
        "found": found,
        "source": source,
        "host": host,
        "version": version,
        "executable_path": executable_path,
        "storage_path": storage_path,
        "warnings": list(warnings),
    }


def discover(*, host_override: str | None = None, timeout: float = 2.0,
            exe_path: str | None = None) -> dict:
    """Run the spec's 4-step detection order and return the first source that answers.

    `host_override`/`exe_path` exist purely so tests and callers can pin the probed host/executable
    without mutating process environment or PATH; production callers normally omit both and let this
    read `OLLAMA_HOST` / `shutil.which("ollama")` itself.
    """
    warnings: list[str] = []

    configured = host_override or os.environ.get("OLLAMA_HOST")
    configured_host = None
    if configured:
        configured_host = _normalize_host(configured)
        try:
            info = probe_endpoint(configured_host, timeout=timeout)
        except network_policy.LocalOnlyViolation as exc:
            info = None
            warnings.append(
                f"local-only network policy blocked probing configured OLLAMA_HOST "
                f"({exc.host or configured_host}); this means the policy said no, not that Ollama "
                f"isn't running -- disable local-only mode or point OLLAMA_HOST at a loopback address "
                f"to check.")
        if info is not None:
            return _result(found=True, source="env", host=configured_host, version=info.get("version"),
                           warnings=warnings)
        warnings.append(f"OLLAMA_HOST={configured!r} did not answer /api/version; trying other sources")

    exe = exe_path if exe_path is not None else shutil.which("ollama")
    exe_version = executable_version(exe, timeout=timeout) if exe else None

    default_host = _normalize_host(OLLAMA_DEFAULT_HOST)
    if configured_host != default_host:
        try:
            info = probe_endpoint(default_host, timeout=timeout)
        except network_policy.LocalOnlyViolation:
            info = None   # the default host is loopback; local-only mode never blocks it in practice
        if info is not None:
            return _result(found=True, source="endpoint", host=default_host, version=info.get("version"),
                           executable_path=exe, warnings=warnings)

    if exe_version is not None:
        return _result(found=True, source="executable", host=None, version=exe_version,
                       executable_path=exe, warnings=warnings)
    if exe:
        warnings.append(f"found an 'ollama' executable at {exe} but `ollama --version` failed or timed "
                        f"out")

    storage = known_storage_path()
    if storage is not None:
        warnings.append(
            "no live Ollama API or working executable found; falling back to on-disk storage detection "
            "only. This confirms a storage directory exists but cannot list models, distinguish a "
            "model from a garbage-collected blob, or confirm a version -- start Ollama and retry for a "
            "reliable model list.")
        return _result(found=True, source="storage_fallback", host=None, version=None,
                       executable_path=exe, storage_path=storage, warnings=warnings)

    return _result(found=False, source=None, warnings=warnings, executable_path=exe)


def list_models(host: str, timeout: float = 5.0) -> list[dict]:
    """GET `{host}/api/tags` -> the raw list of Ollama model dicts (name/model, size, digest,
    modified_at, details{format,family,parameter_size,quantization_level}), unmodified.

    Only callable when discover() found a live API (`source` in {"env", "endpoint"}); raises the
    underlying urllib error rather than returning an empty list on failure -- a caller who explicitly
    asked to list a daemon's models should see a clear network error, not a silent "no models".
    """
    request = urllib.request.Request(f"{host}/api/tags", headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected /api/tags response shape from {host}: {type(payload).__name__}")
    return payload.get("models") or []


def show_model(host: str, name: str, timeout: float = 5.0) -> dict:
    """POST `{host}/api/show` for one model's full definition: modelfile text, parsed `parameters`,
    `template`, `details`, and (on newer Ollama) structured `model_info`. Sends `{"name": name}` --
    the field Ollama's `/api/show` has accepted the longest; newer server versions also accept `model`,
    but `name` round-trips on every version this module was written against.

    Raises the underlying urllib/json error on failure, same reasoning as list_models().
    """
    body = json.dumps({"name": name}).encode("utf-8")
    request = urllib.request.Request(
        f"{host}/api/show", data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"unexpected /api/show response shape from {host}: {type(payload).__name__}")
    return payload
