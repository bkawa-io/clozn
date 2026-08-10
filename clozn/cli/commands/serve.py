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

    models_config_path = getattr(args, "models_config", None)
    managed = None
    if models_config_path:
        if requested_model:
            raise ctx.CloznError(
                "MODEL and --models-config are mutually exclusive"
            )
        if getattr(args, "alias", None):
            raise ctx.CloznError("--alias is supported only for single-model serving")
        if gpu_layers is not None:
            raise ctx.CloznError("GPU-layer flags are per-model in --models-config")
        incompatible = [
            name for name, active in (
                ("--ctx", getattr(args, "ctx", None) is not None),
                ("--cpu", bool(getattr(args, "cpu", False))),
                ("--mask", getattr(args, "mask", None) is not None),
                ("--eos", getattr(args, "eos", None) is not None),
                ("--sae", getattr(args, "sae", None) is not None),
                ("--sae-k", getattr(args, "sae_k", None) is not None),
                (
                    "--no-flash-attn",
                    bool(getattr(args, "no_flash_attn", False)),
                ),
                ("--adapter", getattr(args, "adapter", None) is not None),
                (
                    "--adapter-scale",
                    getattr(args, "adapter_scale", None) is not None,
                ),
            )
            if active
        ]
        if incompatible:
            raise ctx.CloznError(
                f"{incompatible[0]} is per-model in --models-config"
            )
        from clozn.cli.managed_models import (
            ManagedModelsConfigError,
            load_managed_models,
        )
        try:
            managed = load_managed_models(
                models_config_path,
                default_model_id=getattr(args, "default_model", None),
                preload_model_ids=getattr(args, "preload", None),
                max_loaded_models=getattr(args, "max_loaded_models", None),
            )
        except ManagedModelsConfigError as error:
            raise ctx.CloznError(str(error)) from None
        default_definition = managed.definition(managed.default_model_id)
        model = default_definition.model
        flags = {}
        print(
            f"{fmt.DIM}- models: "
            f"{', '.join(item.model_id for item in managed.definitions)} "
            f"(default {managed.default_model_id}){fmt.RST}",
            file=sys.stderr,
            flush=True,
        )
    else:
        if not requested_model:
            raise ctx.CloznError(
                "give MODEL, or use --models-config with a qualified manifest"
            )
        if any(
            getattr(args, name, None) is not None
            for name in ("default_model", "preload", "max_loaded_models")
        ):
            raise ctx.CloznError(
                "--default-model/--preload/--max-loaded-models require "
                "--models-config"
            )
        model = resolve_model(requested_model)
        print(
            f"{fmt.DIM}- model: {model}{fmt.RST}",
            file=sys.stderr,
            flush=True,
        )
        flags = _flags_for(model)
        if args.mask is not None:
            flags["mask"] = args.mask
        if args.eos is not None:
            flags["eos"] = args.eos
        if args.ctx is not None:
            flags["ctx"] = args.ctx
        if gpu_layers is not None:
            flags["gpu_layers"] = gpu_layers
        if args.sae is not None:
            flags["sae"] = args.sae
            if args.sae_k is not None:
                flags["sae_k"] = args.sae_k
        if args.no_flash_attn:
            # extra_args is the generic engine-argv passthrough _launch_args already documents (see
            # engine_process.py) -- attention-edge provenance (`clozn provenance`) needs the engine started
            # this way so kq_soft_max materializes for /score's attn_knockout. NOT the Studio Sources lens,
            # which runs on /runs/<id>/influence-map's forced-scoring path and never touches attn_knockout.
            flags.setdefault("extra_args", []).append("--no-flash-attn")
        if getattr(args, "adapter", None):
            # Fail here rather than letting the engine abort mid-boot: a missing adapter file is a typo, and
            # the actionable message is the path the user typed, not a worker exit code.
            adapter = os.path.abspath(os.path.expanduser(args.adapter))
            if not os.path.isfile(adapter):
                raise ctx.CloznError(f"adapter not found: {adapter}")
            flags["adapter"] = adapter
            if getattr(args, "adapter_scale", None) is not None:
                flags["adapter_scale"] = args.adapter_scale
            scale = flags.get("adapter_scale", 1.0)
            print(
                f"{fmt.DIM}- adapter: {adapter} (scale {scale}){fmt.RST}",
                file=sys.stderr,
                flush=True,
            )

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
        try:
            runtime_config = RuntimeConfig(
                model=model,
                public_port=port,
                host=host,
                public_model_id=alias,
                structured_mode=getattr(args, "structured", "auto"),
                flags=flags,
                prefer_gpu=not args.cpu,
                worker_definitions=managed.definitions if managed else (),
                default_model_id=managed.default_model_id if managed else None,
                preload_model_ids=managed.preload_model_ids if managed else (),
                max_loaded_models=(
                    managed.max_loaded_models if managed else None
                ),
            )
        except Exception as error:
            from clozn.cli.worker_registry import WorkerRegistryConfigError
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
        if stack.worker_registry is not None:
            runtime_status = stack.worker_registry.status()
            resident = sum(
                worker["state"] == "ready"
                for worker in runtime_status["workers"]
            )
            print(
                f"\n  {fmt.BOLD}{resident}/{len(runtime_status['workers'])} "
                f"models resident{fmt.RST} "
                f"(default {runtime_status['default_model_id']}) "
                f"in {time.time()-started:.1f}s"
            )
        else:
            print(
                f"\n  {fmt.BOLD}{alias or _friendly(model)}{fmt.RST} ready on "
                f"{'GPU' if stack.gpu else 'CPU'} ({health.get('mode')}) "
                f"in {time.time()-started:.1f}s"
        )
        print(f"  Studio:                    {fmt.BOLD}{base}/{fmt.RST}")
        print(f"  OpenAI chat:               POST {base}/v1/chat/completions")
        print(f"  Clozn event stream:        POST {base}/api/clozn/generate")
        print(f"  Readiness:                 GET  {base}/readyz")
        worker_phrase = (
            "model workers are private"
            if stack.worker_registry is not None
            else "the model worker is private"
        )
        print(
            f"\n  {fmt.DIM}one public gateway; {worker_phrase}"
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
    """The gateway's raw C++ worker base URL (serves /score etc. -- what export-bundle's live check
    needs), read from /engine/health's worker_url. '-' when unavailable: the field is honest-absent on
    older gateways and while the worker is down, and this must never make `clozn ps` slow or crash."""
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
