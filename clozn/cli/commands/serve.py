"""The product process boundary: ``clozn serve``, ``ps``, and ``stop``."""
from __future__ import annotations

import os
import signal
import sys
import time
import ipaddress

from clozn.cli import formatting as fmt
from clozn.cli.commands.models import _flags_for, _friendly, resolve_model
from clozn.cli.engine_process import _kill, _reg_read, _reg_write, _register, _unregister, _await_dead
from clozn.cli.runtime_process import (
    RuntimeConfig,
    gateway_health,
    gateway_liveness,
    spawn_runtime,
)


def cmd_serve(args):
    from clozn.cli import main as ctx

    positional_model = getattr(args, "model", None)
    flagged_model = getattr(args, "model_flag", None)
    if positional_model and flagged_model:
        raise ctx.CloznError("model was specified both positionally and with -m/--model")
    requested_model = positional_model or flagged_model
    parallel = getattr(args, "parallel", 1)
    if parallel != 1:
        raise ctx.CloznError(
            "Clozn currently serializes generation per model to preserve debugger evidence; "
            "--parallel values above 1 are not supported"
        )
    gpu_layers = getattr(args, "gpu_layers", None)
    if getattr(args, "cpu", False) and gpu_layers is not None and gpu_layers > 0:
        raise ctx.CloznError("--cpu cannot be combined with a positive GPU-layer count")

    if not requested_model:
        raise ctx.CloznError("give one GGUF MODEL")
    model = resolve_model(requested_model)
    print(f"{fmt.DIM}- model: {model}{fmt.RST}", file=sys.stderr, flush=True)
    flags = _flags_for(model)
    if getattr(args, "mask", None) is not None:
        flags["mask"] = args.mask
    if getattr(args, "eos", None) is not None:
        flags["eos"] = args.eos
    if getattr(args, "ctx", None) is not None:
        flags["ctx"] = args.ctx
    if getattr(args, "batch", None) is not None:
        flags["batch"] = args.batch
    if getattr(args, "ubatch", None) is not None:
        flags["ubatch"] = args.ubatch
    if gpu_layers is not None:
        flags["gpu_layers"] = gpu_layers

    port = args.port or 8080
    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    alias = getattr(args, "alias", None)
    try:
        loopback = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = False
    if not loopback:
        print(
            "warning: Clozn is listening beyond localhost.\n"
            "The gateway includes debugging/run APIs and has no network authentication.",
            file=sys.stderr,
            flush=True,
        )
    os.makedirs(ctx.HOME, exist_ok=True)
    worker_log = open(os.path.join(ctx.HOME, "worker.log"), "w", encoding="utf-8")
    stack = None
    registered = False
    previous_sigterm = None

    def interrupt_for_shutdown(_signum, _frame):
        raise KeyboardInterrupt

    started = time.time()
    try:
        try:
            previous_sigterm = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, interrupt_for_shutdown)
        except (AttributeError, ValueError):
            previous_sigterm = None  # unavailable, or cmd_serve was invoked outside the main thread
        print(f"{fmt.DIM}- starting {_friendly(model)} …{fmt.RST}", file=sys.stderr, flush=True)
        from clozn.paths import data_root
        print(f"{fmt.DIM}- data root: {data_root()}{fmt.RST}", file=sys.stderr, flush=True)
        try:
            runtime_config = RuntimeConfig(
                model=model,
                public_port=port,
                host=host,
                public_model_id=alias,
                flags=flags,
                prefer_gpu=not args.cpu,
            )
        except Exception as error:
            from clozn.cli.runtime_identity import WorkerRegistryConfigError
            if isinstance(error, WorkerRegistryConfigError):
                raise ctx.CloznError(str(error)) from None
            raise
        stack = spawn_runtime(runtime_config, worker_log=worker_log)
        health = stack.worker_health
        _register(
            model,
            port,
            stack.gpu,
            health.get("mode", "?"),
            os.getpid(),
            **stack.registry_fields(),
        )
        registered = True

        base = f"http://127.0.0.1:{port}"
        listening = f"{host}:{port}"
        print(f"  Listening:                 {listening}")
        if alias:
            print(f"  Public model:              {alias}")
        print(
            f"\n  {fmt.BOLD}{alias or _friendly(model)}{fmt.RST} ready on "
            f"{'GPU' if stack.gpu else 'CPU'} ({health.get('mode')}) "
            f"in {time.time()-started:.1f}s"
        )
        print(f"  Studio:                    {fmt.BOLD}{base}/{fmt.RST}")
        print(f"  OpenAI chat:               POST {base}/v1/chat/completions")
        print(f"  Readiness:                 GET  {base}/readyz")
        print(
            f"\n  {fmt.DIM}one public gateway; the model worker is private"
            f"   -   Ctrl-C to stop{fmt.RST}\n"
        )

        def restarted(current):
            print(
                f"{fmt.DIM}- model worker lifecycle changed after an "
                f"unexpected exit{fmt.RST}",
                file=sys.stderr,
            )
            _register(
                model,
                port,
                current.gpu,
                current.worker_health.get("mode", "?"),
                os.getpid(),
                **current.registry_fields(),
            )

        gateway_code = stack.wait(on_worker_restart=restarted)
        raise ctx.CloznError(f"public gateway exited unexpectedly (code {gateway_code})")
    except KeyboardInterrupt:
        print(f"\n{fmt.DIM}- stopping{fmt.RST}", file=sys.stderr)
    finally:
        if stack is not None:
            stack.stop()
        if registered:
            _unregister(port)
        worker_log.close()
        if previous_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, previous_sigterm)
            except (AttributeError, ValueError):
                pass


def cmd_ps(_args):
    registry = _reg_read()
    live = []
    changed = False
    for port, entry in list(registry.items()):
        if gateway_liveness(int(port), timeout=1.0):
            live.append((port, entry))
        else:
            registry.pop(port, None)
            changed = True
    if changed:
        _reg_write(registry)
    if not live:
        print("no Clozn runtimes running.")
        return
    print(f"{'MODEL':<14} {'PORT':>6}  {'BACKEND':<8} {'MODE':<16} WORKER")
    for port, entry in live:
        print(
            f"{_friendly(entry.get('model', '?')):<14} {port:>6}  "
            f"{('GPU' if entry.get('gpu') else 'CPU'):<8} {entry.get('mode', '?'):<16} "
            f"{_worker_url(int(port))}"
        )


def _worker_url(port: int) -> str:
    """The gateway's raw C++ worker base URL, read from /engine/health's worker_url.

    ``-`` means the field is unavailable while the worker is down; this probe must never make
    ``clozn ps`` slow or crash.
    """
    try:
        import json as _json
        import urllib.request as _rq
        with _rq.urlopen(f"http://127.0.0.1:{port}/engine/health", timeout=1.0) as r:
            info = (_json.load(r) or {}).get("engine") or {}
        return str(info.get("worker_url") or "-")
    except Exception:
        return "-"


def _runtime_pids(entry: dict) -> list[int]:
    """All valid supervisor/gateway/worker PIDs in one local registry row."""
    candidates = [entry.get(key) for key in ("pid", "gateway_pid", "worker_pid")]
    models = entry.get("models")
    if isinstance(models, list):
        candidates.extend(
            model.get("worker_pid")
            for model in models
            if isinstance(model, dict)
        )
    result = []
    seen = set()
    for candidate in candidates:
        try:
            pid = int(candidate)
        except (TypeError, ValueError):
            continue
        if pid > 0 and pid not in seen:
            seen.add(pid)
            result.append(pid)
    return result


def cmd_stop(args):
    from clozn.cli import main as ctx

    registry = _reg_read()
    targets = [
        (port, entry)
        for port, entry in registry.items()
        if args.which in ("all", str(port)) or _friendly(entry.get("model", "")) == args.which
    ]
    if not targets:
        raise ctx.CloznError(f"no running runtime matches '{args.which}'. See: clozn ps")
    killed_pids: set[int] = set()
    for port, entry in targets:
        # Ask the supervisor to stop first so its finally block owns the normal shutdown. Children are
        # still signalled explicitly as a fallback for a wedged/dead supervisor and for old registry rows.
        pids = _runtime_pids(entry)
        try:
            supervisor_pid = int(entry.get("pid"))
        except (TypeError, ValueError):
            supervisor_pid = None
        for index, pid in enumerate(pids):
            _kill(pid)
            if index == 0 and pid == supervisor_pid:
                time.sleep(0.2)
        killed_pids.update(pids)
        print(f"stopped {_friendly(entry.get('model', '?'))} on port {port}")
    # Prune AUTHORITATIVELY, after the processes are actually gone. A force-killed supervisor (Windows
    # taskkill /F) can run its own dying unregister / worker-restart write and resurrect a now-dead row
    # AFTER our write, leaving a stale entry that only self-heals lazily on the next `clozn ps`. Wait
    # them out, then re-read + drop the stopped ports so no live writer can lose the update.
    _await_dead(killed_pids, timeout=5.0)
    registry = _reg_read()
    for port, _ in targets:
        registry.pop(port, None)
    _reg_write(registry)
