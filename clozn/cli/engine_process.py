"""engine_process -- find the engine build, launch it as a subprocess with the right DLLs on PATH, wait for
/health, and track the `clozn serve` <-> `clozn run` warm-daemon registry (~/.clozn/daemons.json).

HOME/CloznError live on `clozn.cli.main` (the CLI's shared-state owner, mirroring the server's app.py);
every function here that needs either does `from clozn.cli import main as ctx` INSIDE the function body
(never at module level) and reads `ctx.HOME` / raises `ctx.CloznError(...)` at call time. This is
deliberately lazy, not just a style preference: main.py imports THIS module (for _free_port etc.) at its
own module level, so a module-level `from clozn.cli import main as ctx` here would deadlock the first time
anything imports clozn.cli.engine_process before clozn.cli.main has been touched (a real circular import,
not a theoretical one -- caught by directly `import clozn.cli.engine_process` in isolation). Deferring the
import to call time sidesteps it entirely: by the time any of these functions actually run, module loading
has long finished.
"""
from __future__ import annotations

import dataclasses
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from clozn.cli import process_guard

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root (parent of clozn/)
ENGINE_CORE = os.path.join(REPO, "engine", "core")

# Engine builds, most-preferred first. (subdir, is_gpu); the exe sits at <subdir>/ or <subdir>/Release/.
BUILDS = [("build-gpu", True), ("build-cuda", True),
          ("build-ggml-cpu", False), ("build-serve", False), ("build-cpu", False)]

# clozn-server's own shared libraries (llama.cpp's split build: ggml*.{dll,so,dylib} + llama*.{dll,so,dylib}).
# Used as existence markers below -- a candidate directory only counts as a "DLL dir" if one of these is
# actually in it, not merely because a plausibly-named subfolder happens to exist (see _dll_dirs_for).
# Windows is the platform this repo builds and qualifies continuously (this dev box, RTX 5080); Linux CPU
# also has a real, CI-proven build (.github/workflows/real-runtime-smoke.yml builds+runs clozn-server on
# ubuntu-24.04 nightly). Neither of those emits a loadable `.so` plugin next to the exe today (the Linux
# CPU build links statically), so `.dll` remains the only marker that has ever actually matched in
# practice. The `.so`/`.dylib` names are here so a future build that DOES emit loadable backend plugins
# (e.g. a dynamically-loaded ggml-cuda/ggml-metal backend -- see engine/core/build_gpu.sh) is found the
# same way, instead of silently limiting dll_dirs to just the exe's own directory. Checking a few extra
# never-matching names on Windows is a no-op there.
_ENGINE_DLL_MARKERS = ("llama.dll", "ggml.dll", "libllama.so", "libggml.so", "libllama.dylib", "libggml.dylib")


def _dll_dirs_for(exe: str) -> list[str]:
    """Directories to prepend to a spawned clozn-server's PATH so Windows can resolve its llama.dll /
    ggml-*.dll imports (STATUS_DLL_NOT_FOUND otherwise -- these DLLs live in a `bin` sibling, not next to
    the exe, so the OS's automatic "search the app directory first" behavior never finds them; PATH is
    the mechanism that does).

    Derived from `exe`'s OWN location on disk -- never a hardcoded absolute path -- so this keeps working
    whichever build layout produced it: single-config CMake (DLLs in ``<build>/bin``, the exe directly in
    ``<build>/``) and multi-config/Visual-Studio-style generators (exe in ``<build>/Release/``, DLLs in
    ``<build>/bin`` or ``<build>/bin/Release``). Every candidate except the exe's own directory is checked
    for an ACTUAL marker DLL before being trusted -- "check where they actually are relative to the binary
    before hardcoding," not "assume a subfolder name is right because it exists." The exe's own directory
    is always included even when empty: harmless, and preserves the pre-existing behavior of never handing
    back zero directories for an otherwise-found exe.
    """
    exe_dir = os.path.dirname(exe)
    build_root = os.path.dirname(exe_dir) if os.path.basename(exe_dir).lower() == "release" else exe_dir
    candidates = [os.path.join(exe_dir, "bin"), os.path.join(build_root, "bin"),
                  os.path.join(build_root, "bin", "Release"), os.path.join(build_root, "Release")]
    dirs = [exe_dir]
    for d in candidates:
        d = os.path.normpath(d)
        if d not in dirs and os.path.isdir(d) and any(
                os.path.isfile(os.path.join(d, marker)) for marker in _ENGINE_DLL_MARKERS):
            dirs.append(d)
    return dirs


@dataclasses.dataclass(frozen=True)
class EngineDiscovery:
    """The full result of one find_engine_ex() resolution -- what find_engine() trims to its historical
    3-tuple, plus everything roadmap feature 01 wants recorded on every run: which of the four
    precedence tiers produced this engine, and (when knowable) its backend and managed-install identity.

    `backend` is deliberately coarse ("cpu"/"gpu") for every tier except "managed": tiers 1/3/4 only ever
    know GPU-or-not (a directory name, or an env var), never the exact accelerator, so claiming "cuda"
    there would be a guess this module has no evidence for. Only the managed tier's registry record (see
    clozn/setup/registry.py) carries a manifest-declared exact backend, so only it reports one.
    """

    exe: str
    dll_dirs: list
    gpu: bool
    discovery_source: str                    # "env_override" | "managed" | "repo_dev_build" | "legacy"
    backend: "str | None" = None
    artifact_sha256: "str | None" = None
    engine_version: "str | None" = None
    build_id: "str | None" = None
    llama_cpp_commit: "str | None" = None


def _env_override_candidate() -> "EngineDiscovery | None":
    """Tier 1. CLOZN_ENGINE is the spec's own name for this override; CLOZN_ENGINE_BIN is kept working
    indefinitely as the pre-existing name (renaming it would break any script/CI already setting it) --
    CLOZN_ENGINE wins when both are set. CLOZN_ENGINE_GPU marks a CLOZN_ENGINE(_BIN)-pointed build as a
    GPU worker; there is no way to detect that from the file itself."""
    from clozn.cli import main as ctx

    override = os.environ.get("CLOZN_ENGINE") or os.environ.get("CLOZN_ENGINE_BIN")
    if not override:
        return None
    exe = os.path.abspath(os.path.expanduser(override))
    if not os.path.isfile(exe):
        raise ctx.CloznError(
            f"CLOZN_ENGINE{'' if os.environ.get('CLOZN_ENGINE') else '_BIN'} does not point to a file: {exe}")
    gpu = os.environ.get("CLOZN_ENGINE_GPU", "").strip().lower() in ("1", "true", "yes", "on")
    return EngineDiscovery(exe=exe, dll_dirs=_dll_dirs_for(exe), gpu=gpu,
                           discovery_source="env_override", backend="gpu" if gpu else "cpu")


def _managed_candidate() -> "EngineDiscovery | None":
    """Tier 2. The active engine `clozn setup` installed, per ~/.clozn/engines/registry.json. Absent
    entirely (returns None, not an error) when nothing has ever been installed, or when the registry's
    `active` entry is stale (its entrypoint file was removed out of band) -- clozn.setup.registry's own
    self-heal (prune_missing) is what a `clozn setup status` call reconciles; discovery here just moves
    on to tier 3 rather than failing the whole lookup over one missing managed install."""
    from clozn.cli import main as ctx
    from clozn.setup import registry as setup_registry

    doc = setup_registry.load(ctx.HOME)
    active_key = doc.get("active")
    if not active_key:
        return None
    record = (doc.get("installed") or {}).get(active_key)
    if not isinstance(record, dict):
        return None
    exe = record.get("entrypoint")
    if not exe or not os.path.isfile(exe):
        return None
    backend = record.get("backend")
    gpu = backend not in (None, "cpu")
    return EngineDiscovery(exe=exe, dll_dirs=_dll_dirs_for(exe), gpu=gpu,
                           discovery_source="managed", backend=backend,
                           artifact_sha256=record.get("sha256"), engine_version=record.get("version"),
                           build_id=record.get("build_id"),
                           llama_cpp_commit=record.get("llama_cpp_commit"))


def _repo_dev_build_candidates() -> list:
    """Tier 3. Today's original (pre-feature-01) discovery logic, unchanged: every BUILDS subdirectory
    under engine/core/ that contains a clozn-server binary, most-preferred first."""
    found = []
    for sub, gpu in BUILDS:
        root = os.path.join(ENGINE_CORE, sub)
        for exe in (os.path.join(root, "clozn-server.exe"),
                    os.path.join(root, "Release", "clozn-server.exe"),
                    os.path.join(root, "clozn-server")):       # posix
            if os.path.isfile(exe):
                found.append(EngineDiscovery(exe=exe, dll_dirs=_dll_dirs_for(exe), gpu=gpu,
                                             discovery_source="repo_dev_build",
                                             backend="gpu" if gpu else "cpu"))
                break
    return found


def _legacy_candidates() -> list:
    """Tier 4, named and kept last so a future migration path (e.g. a pre-feature-01 install layout, or
    a deprecated managed-engine directory shape) has somewhere to land without another precedence-order
    edit. No such layout has ever shipped -- this always returns [] today, honestly, rather than
    inventing a filesystem location no version of clozn has ever written to."""
    return []


def find_engine_ex(prefer_gpu=True) -> EngineDiscovery:
    """The full discovery result -- see EngineDiscovery. Precedence: CLOZN_ENGINE(_BIN) override ->
    active managed engine (clozn setup) -> repository-local dev build -> legacy search paths (currently
    always empty). ``prefer_gpu=False`` is the CLI's documented ``--cpu`` contract: it skips any
    candidate this function otherwise cannot prove is a CPU build, at every tier -- including refusing a
    GPU env override and falling through past a GPU managed install to whatever CPU candidate a later
    tier offers, exactly as it already did for tier 3 before this refactor."""
    from clozn.cli import main as ctx

    override = _env_override_candidate()
    if override is not None:
        if not prefer_gpu and override.gpu:
            raise ctx.CloznError(
                "--cpu was requested, but CLOZN_ENGINE(_BIN) is marked as a GPU worker; "
                "point it at a CPU build or unset CLOZN_ENGINE_GPU"
            )
        return override

    managed = _managed_candidate()
    if managed is not None and (prefer_gpu or not managed.gpu):
        return managed

    candidates = _repo_dev_build_candidates() + _legacy_candidates()
    if not prefer_gpu:
        candidates = [c for c in candidates if not c.gpu]
        if not candidates:
            raise ctx.CloznError(
                "--cpu was requested, but no CPU engine build was found. "
                "Build engine/core/build-serve as described in docs/DEVELOPMENT.md, or run `clozn "
                "setup --backend cpu`."
            )
        return candidates[0]

    if not candidates:
        raise ctx.CloznError(
            "no engine found. Run `clozn setup` to install one, see docs/DEVELOPMENT.md to build one, "
            "or set CLOZN_ENGINE."
        )
    candidates.sort(key=lambda c: 0 if c.gpu else 1)
    return candidates[0]


def find_engine(prefer_gpu=True) -> tuple[str, list[str], bool]:
    """-> (exe_path, dll_dirs, is_gpu). The historical 3-tuple contract every existing caller (doctor,
    smoke, models, spawn_engine, ...) already unpacks; find_engine_ex() above is the same lookup with
    the discovery-source/backend/artifact identity feature 01's run-journal recording needs."""
    d = find_engine_ex(prefer_gpu)
    return d.exe, d.dll_dirs, d.gpu


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _env_with_dlls(dll_dirs: list[str], gpu: bool) -> dict:
    """Build the subprocess environment for the engine binary: put its own build directory (dll_dirs) --
    and, for a GPU build, the CUDA runtime -- wherever THIS platform's dynamic linker actually looks.
    Windows resolves DLLs via PATH, which is why this function existed at all (task #103: a spawned
    clozn-server.exe hit STATUS_DLL_NOT_FOUND without it). Linux/macOS were previously inert here, not
    broken by it: this only ever wrote PATH, which those platforms' dynamic linkers don't consult for
    shared-library resolution -- but a CMake-built binary typically also carries an rpath pointing at its
    own build tree, so it resolves its libs without any help from this function regardless. That is the
    likely reason a locally-built engine has run on a Mac before despite this function never having
    touched DYLD_LIBRARY_PATH. Setting LD_LIBRARY_PATH/DYLD_LIBRARY_PATH here is a genuine ADDITIONAL
    robustness path for layouts where rpath isn't enough (a GPU backend loaded as a separate plugin at
    runtime, e.g. a future dynamically-loaded ggml-cuda.so -- see engine/core/build_gpu.sh), not a fix
    for something that was failing before.

    Windows behavior is unchanged byte-for-byte from before this function became platform-aware: same
    hardcoded CUDA v13.3 bin dirs, same existence checks, same PATH construction, same key. See
    tests/test_env_with_dlls_platform.py for the pinning tests and tests/test_runtime_architecture.py's
    test_env_with_dlls_prepends_the_dll_dir_without_mutating_os_environ for the pre-existing contract
    this must keep holding on whichever platform actually runs it.
    """
    env = dict(os.environ)
    extra = list(dll_dirs)
    if gpu and sys.platform == "win32":
        for c in (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
                  r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"):
            if os.path.isdir(c):
                extra.append(c)
    elif gpu and sys.platform.startswith("linux"):
        # No fixed install path to guess here (Windows ships one CUDA installer to one well-known
        # location; Linux CUDA installs are apt/conda/manual and vary). CUDA_HOME is the toolkit's own
        # documented env var; when set, its lib64 is the runtime's usual home. When unset, assume the
        # CUDA runtime is already reachable via LD_LIBRARY_PATH (a system or conda CUDA install) --
        # exactly the assumption engine/core/build_gpu.sh's nvcc-on-PATH detection already relies on.
        cuda_home = os.environ.get("CUDA_HOME")
        if cuda_home:
            lib64 = os.path.join(cuda_home, "lib64")
            if os.path.isdir(lib64):
                extra.append(lib64)

    if sys.platform == "win32":
        env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    elif sys.platform == "darwin":
        env["DYLD_LIBRARY_PATH"] = os.pathsep.join(extra + [env.get("DYLD_LIBRARY_PATH", "")])
    else:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(extra + [env.get("LD_LIBRARY_PATH", "")])
    return env


def _launch_args(exe: str, model: str, port: int, flags: dict, gpu: bool) -> list[str]:
    args = [exe, model, "--port", str(port), "--host", "127.0.0.1"]
    if "gpu_layers" in flags:
        args += ["--gpu-layers", str(flags["gpu_layers"])]
    elif gpu:
        # Preserve Clozn's historical implicit GPU behavior when the compatibility flag is omitted.
        args += ["--gpu-layers", "99"]
    if flags.get("ctx") is not None:
        args += ["--ctx", str(flags["ctx"])]
    if "mask" in flags:
        args += ["--diffusion", "--mask-token", str(flags["mask"])]
    if "eos" in flags:
        args += ["--eos", str(flags["eos"])]
    if "sae" in flags:                        # passthrough only: dims must match, server refuses politely
        args += ["--sae", flags["sae"]]
        if "sae_k" in flags:
            args += ["--sae-k", str(flags["sae_k"])]
    if flags.get("jlens"):
        args += ["--jlens", str(flags["jlens"])]
    if flags.get("_model_sha256"):
        args += ["--model-sha256", str(flags["_model_sha256"])]
    # The complete GGUF and its tokenizer metadata are distinct identity facets.  The latter is
    # what Time Machine uses to prove the worker launched with the tokenizer that rendered a
    # candidate append-only turn.
    if flags.get("_tokenizer_sha256"):
        args += ["--tokenizer-sha256", str(flags["_tokenizer_sha256"])]
    # A fine-tune adapter (LoRA). Given a first-class key rather than riding extra_args because the
    # adapter is part of the run's REPRODUCTION IDENTITY -- what weights actually answered -- not a
    # tuning knob, so callers that record identity need to read it back structurally. The engine
    # ABORTS rather than serving the base model if the adapter will not attach, so there is no path
    # where this silently does nothing.
    if flags.get("adapter"):
        args += ["--lora", str(flags["adapter"])]
        if flags.get("adapter_scale") is not None:
            args += ["--lora-scale", str(flags["adapter_scale"])]
    # Generic passthrough for engine flags this mapping has no key for (first user:
    # --no-flash-attn, which the provenance/attn-knockout mode requires -- flash attention fuses
    # the softmax, so the weights never materialize). A list of literal argv tokens, appended
    # verbatim; the engine itself rejects anything it doesn't understand.
    if flags.get("extra_args"):
        args += [str(a) for a in flags["extra_args"]]
    return args


def _health(port: int, timeout=3.0):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=timeout) as response:
            return json.loads(response.read())
    except Exception:
        return None


def _terminate_process(proc, timeout: float = 5.0) -> None:
    """Best-effort child cleanup, including interrupted startup."""
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except Exception:
        try:
            proc.kill()
            proc.wait(timeout=timeout)
        except Exception:
            pass


def spawn_engine(
    model: str,
    port: int,
    flags: dict,
    *,
    prefer_gpu=True,
    logf=None,
    boot_timeout=180,
    engine_discovery=None,
):
    """Start an engine on `port`, wait until /health is ok. Returns (proc, health, is_gpu)."""
    from clozn.cli import main as ctx
    if engine_discovery is None:
        exe, dll_dirs, gpu = find_engine(prefer_gpu)
    else:
        exe = engine_discovery.exe
        dll_dirs = list(engine_discovery.dll_dirs)
        gpu = bool(engine_discovery.gpu)
    launch_flags = dict(flags)
    disable_auto_jlens = (
        launch_flags.pop("_disable_auto_jlens", False) is True
    )
    if os.path.isfile(model):
        from clozn.artifacts.contracts import (ArtifactContractError, find_compatible_artifact,
                                               gguf_identity)
        identity = gguf_identity(model)
        launch_flags["_model_sha256"] = identity["sha256"]
        launch_flags["_tokenizer_sha256"] = identity["tokenizer_sha256"]
        if disable_auto_jlens:
            # Managed routing v1 cannot key a value-bearing J-lens artifact.
            # Suppress legacy auto-discovery explicitly rather than launching
            # evidence behavior absent from the immutable runtime key.
            launch_flags.pop("jlens", None)
        else:
            try:
                artifact_root = os.environ.get("CLOZN_ARTIFACTS_DIR") or os.path.join(ctx.HOME, "artifacts")
                jlens_dir = find_compatible_artifact(
                    "jlens", identity, artifact_root,
                    explicit_dir=os.environ.get("CLOZN_JLENS_DIR") or launch_flags.get("jlens"),
                )
            except ArtifactContractError as error:
                raise ctx.CloznError(f"J-lens artifact refused: {error}") from None
            if jlens_dir:
                launch_flags["jlens"] = jlens_dir
            else:
                launch_flags.pop("jlens", None)
    args = _launch_args(exe, model, port, launch_flags, gpu)
    proc = subprocess.Popen(args, env=_env_with_dlls(dll_dirs, gpu),
                            stdout=logf or subprocess.DEVNULL, stderr=subprocess.STDOUT,
                            **process_guard.subprocess_kwargs())
    # Parent-death guard (ADR 008 Stage 0): best-effort, never raises, degrades to an unguarded child
    # on any failure or on a platform (macOS) with no kernel primitive for this. Covers every worker
    # spawn -- initial launch, a managed-registry preload, and every WorkerHandle.restart() -- because
    # this is the one call site all of them funnel through.
    process_guard.guard(proc)
    started = time.monotonic()
    try:
        while time.monotonic() - started < boot_timeout:
            if proc.poll() is not None:                        # died before healthy
                raise ctx.CloznError(f"engine exited (code {proc.returncode}). {_log_tail(logf)}")
            h = _health(port)
            if h and h.get("status") == "ok":
                # Handshake: refuse a worker whose protocol MAJOR this supervisor can't drive, rather than
                # proxy a stream it may no longer parse. The usual cause is a stale clozn-server binary that
                # predates the handshake -- the message says to rebuild. (A compatible worker proceeds.)
                from clozn.protocol import check_worker_protocol
                ok, reason = check_worker_protocol(h.get("protocol_version"))
                if not ok:
                    _terminate_process(proc)
                    raise ctx.CloznError(f"engine protocol handshake failed: {reason}")
                return proc, h, gpu
            time.sleep(0.3)
        raise ctx.CloznError(f"engine did not become healthy within {boot_timeout}s. {_log_tail(logf)}")
    except BaseException:
        _terminate_process(proc)
        raise


def _log_tail(logf, n=400):
    if not logf:
        return ""
    try:
        logf.flush()
        with open(logf.name, "r", errors="replace") as f:
            return "last output: " + f.read()[-n:].strip().replace("\n", " ")
    except Exception:
        return ""


# --------------------------------------------------------------- warm-daemon registry (clozn serve <-> run)
# `clozn serve` records {port -> model/gpu/mode} here; `clozn run` reuses a live one instead of reloading.
# Stale entries self-heal: a dead gateway fails /readyz in _find_warm and is ignored (then pruned).

def _reg_path() -> str:
    from clozn.cli import main as ctx
    return os.path.join(ctx.HOME, "daemons.json")


def _reg_read() -> dict:
    try:
        with open(_reg_path(), encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _reg_write(d: dict):
    from clozn.cli import main as ctx
    from clozn._io import atomic_write_json
    os.makedirs(ctx.HOME, exist_ok=True)
    try:
        atomic_write_json(_reg_path(), d)
    except Exception:
        pass


def _register(model: str, port: int, gpu: bool, mode: str, pid: int, **runtime_fields):
    d = _reg_read()
    d[str(port)] = {"model": model, "gpu": gpu, "mode": mode, "pid": pid, **runtime_fields}
    _reg_write(d)


def _kill(pid: int):
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.kill(pid, 15)
    except Exception:
        pass


def _pid_alive(pid) -> bool:
    """True iff the process is still running -- READ-ONLY, never signals it (os.kill(pid, 0) tries to
    terminate on Windows, so use tasklist there instead)."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True, text=True, timeout=5)
            return str(pid) in out.stdout
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:
        return False


def _await_dead(pids, timeout: float = 5.0) -> None:
    """Block up to `timeout` for every pid to exit, so a caller can prune the registry with no live
    supervisor left to race its write (best-effort; returns at the deadline regardless)."""
    remaining = {int(p) for p in pids if str(p).isdigit()}
    deadline = time.monotonic() + max(0.0, timeout)
    while remaining and time.monotonic() < deadline:
        remaining = {p for p in remaining if _pid_alive(p)}
        if remaining:
            time.sleep(0.15)


def _unregister(port: int):
    d = _reg_read()
    if d.pop(str(port), None) is not None:
        _reg_write(d)


def _find_warm(model: str, n_ctx: int | None = None):
    """A live product gateway for this exact model/context -> (public_port, gpu, mode), else None."""
    from clozn.cli.runtime_process import gateway_health

    d = _reg_read(); hit = None; dirty = False
    for port, ent in list(d.items()):
        h = gateway_health(int(port), timeout=1.0)
        if not h:
            d.pop(port, None); dirty = True; continue           # prune the dead
        worker_ctx = ((h.get("worker") or {}).get("n_ctx") if isinstance(h, dict) else None)
        context_matches = n_ctx is None or worker_ctx == n_ctx
        if ent.get("model") == model and context_matches and hit is None:
            hit = (int(port), bool(ent.get("gpu")), ent.get("mode", h.get("mode", "?")))
    if dirty:
        _reg_write(d)
    return hit
