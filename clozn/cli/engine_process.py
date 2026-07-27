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

import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root (parent of clozn/)
ENGINE_CORE = os.path.join(REPO, "engine", "core")

# Engine builds, most-preferred first. (subdir, is_gpu); the exe sits at <subdir>/ or <subdir>/Release/.
BUILDS = [("build-gpu", True), ("build-cuda", True),
          ("build-ggml-cpu", False), ("build-serve", False), ("build-cpu", False)]

# clozn-server.exe's own shared libraries (llama.cpp's split build: ggml*.dll + llama*.dll). Used as
# existence markers below -- a candidate directory only counts as a "DLL dir" if one of these is actually
# in it, not merely because a plausibly-named subfolder happens to exist (see _dll_dirs_for).
_ENGINE_DLL_MARKERS = ("llama.dll", "ggml.dll")


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


def find_engine(prefer_gpu=True) -> tuple[str, list[str], bool]:
    """-> (exe_path, dll_dirs, is_gpu).

    ``prefer_gpu=True`` prefers a GPU build but may fall back to CPU.  ``False`` is
    the implementation of the CLI's documented ``--cpu`` contract and therefore
    refuses to return a GPU worker.
    """
    from clozn.cli import main as ctx
    override = os.environ.get("CLOZN_ENGINE_BIN")
    if override:
        exe = os.path.abspath(os.path.expanduser(override))
        if not os.path.isfile(exe):
            raise ctx.CloznError(f"CLOZN_ENGINE_BIN does not point to a file: {exe}")
        bins = _dll_dirs_for(exe)
        gpu = os.environ.get("CLOZN_ENGINE_GPU", "").strip().lower() in ("1", "true", "yes", "on")
        if not prefer_gpu and gpu:
            raise ctx.CloznError(
                "--cpu was requested, but CLOZN_ENGINE_BIN is marked as a GPU worker; "
                "point it at a CPU build or unset CLOZN_ENGINE_GPU"
            )
        return exe, bins, gpu
    cands = []
    for sub, gpu in BUILDS:
        root = os.path.join(ENGINE_CORE, sub)
        for exe in (os.path.join(root, "clozn-server.exe"),
                    os.path.join(root, "Release", "clozn-server.exe"),
                    os.path.join(root, "clozn-server")):       # posix
            if os.path.isfile(exe):
                cands.append((exe, _dll_dirs_for(exe), gpu))
                break
    if not cands:
        raise ctx.CloznError("no engine found. See docs/DEVELOPMENT.md, or set CLOZN_ENGINE_BIN.")
    if not prefer_gpu:
        cands = [candidate for candidate in cands if not candidate[2]]
        if not cands:
            raise ctx.CloznError(
                "--cpu was requested, but no CPU engine build was found. "
                "Build engine/core/build-serve as described in docs/DEVELOPMENT.md."
            )
    else:
        cands.sort(key=lambda candidate: 0 if candidate[2] else 1)
    return cands[0]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _env_with_dlls(dll_dirs: list[str], gpu: bool) -> dict:
    env = dict(os.environ)
    extra = list(dll_dirs)
    if gpu:
        for c in (r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64",
                  r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"):
            if os.path.isdir(c):
                extra.append(c)
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def _launch_args(exe: str, model: str, port: int, flags: dict, gpu: bool) -> list[str]:
    args = [exe, model, "--port", str(port), "--host", "127.0.0.1"]
    if gpu:
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


def spawn_engine(model: str, port: int, flags: dict, *, prefer_gpu=True, logf=None, boot_timeout=180):
    """Start an engine on `port`, wait until /health is ok. Returns (proc, health, is_gpu)."""
    from clozn.cli import main as ctx
    exe, dll_dirs, gpu = find_engine(prefer_gpu)
    launch_flags = dict(flags)
    if os.path.isfile(model):
        from clozn.artifacts.contracts import (ArtifactContractError, find_compatible_artifact,
                                               gguf_identity)
        identity = gguf_identity(model)
        launch_flags["_model_sha256"] = identity["sha256"]
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
                            stdout=logf or subprocess.DEVNULL, stderr=subprocess.STDOUT)
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
