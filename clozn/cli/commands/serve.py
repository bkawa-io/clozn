"""The product process boundary: ``clozn serve``, ``ps``, and ``stop``."""
from __future__ import annotations

import os
import signal
import sys
import time

from clozn.cli import formatting as fmt
from clozn.cli.commands.models import _flags_for, _friendly, resolve_model
from clozn.cli.engine_process import _kill, _reg_read, _reg_write, _register, _unregister, _await_dead
from clozn.cli.runtime_process import RuntimeConfig, gateway_health, spawn_runtime


def cmd_serve(args):
    from clozn.cli import main as ctx

    model = resolve_model(args.model)
    print(f"{fmt.DIM}- model: {model}{fmt.RST}", file=sys.stderr, flush=True)
    flags = _flags_for(model)
    if args.mask is not None:
        flags["mask"] = args.mask
    if args.eos is not None:
        flags["eos"] = args.eos
    if args.ctx is not None:
        flags["ctx"] = args.ctx
    if args.sae is not None:
        flags["sae"] = args.sae
        if args.sae_k is not None:
            flags["sae_k"] = args.sae_k

    port = args.port or 8080
    os.makedirs(ctx.HOME, exist_ok=True)
    worker_log = open(os.path.join(ctx.HOME, "worker.log"), "w", encoding="utf-8")
    gateway_log = open(os.path.join(ctx.HOME, "gateway.log"), "w", encoding="utf-8")
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
        stack = spawn_runtime(
            RuntimeConfig(model=model, public_port=port, flags=flags, prefer_gpu=not args.cpu),
            worker_log=worker_log,
            gateway_log=gateway_log,
        )
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
        print(
            f"\n  {fmt.BOLD}{_friendly(model)}{fmt.RST} ready on "
            f"{'GPU' if stack.gpu else 'CPU'} ({health.get('mode')}) in {time.time()-started:.1f}s"
        )
        print(f"  Studio:                    {fmt.BOLD}{base}/{fmt.RST}")
        print(f"  OpenAI chat:               POST {base}/v1/chat/completions")
        print(f"  OpenAI text completions:   POST {base}/v1/completions")
        print(f"  Clozn event stream:        POST {base}/api/clozn/generate")
        print(f"  Readiness:                 GET  {base}/readyz")
        print(f"\n  {fmt.DIM}one public gateway; the model worker is private   -   Ctrl-C to stop{fmt.RST}\n")

        def restarted(current):
            print(f"{fmt.DIM}- model worker restarted after an unexpected exit{fmt.RST}", file=sys.stderr)
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
        gateway_log.close()
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
        if gateway_health(int(port), timeout=1.0):
            live.append((port, entry))
        else:
            registry.pop(port, None)
            changed = True
    if changed:
        _reg_write(registry)
    if not live:
        print("no Clozn runtimes running.")
        return
    print(f"{'MODEL':<14} {'PORT':>6}  {'BACKEND':<8} MODE")
    for port, entry in live:
        print(
            f"{_friendly(entry.get('model', '?')):<14} {port:>6}  "
            f"{('GPU' if entry.get('gpu') else 'CPU'):<8} {entry.get('mode', '?')}"
        )


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
        seen = set()
        for key in ("pid", "gateway_pid", "worker_pid"):
            pid = entry.get(key)
            if pid and int(pid) not in seen:
                seen.add(int(pid))
                _kill(int(pid))
                if key == "pid":
                    time.sleep(0.2)
        killed_pids |= seen
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
