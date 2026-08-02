"""Lifecycle for the product runtime: one public gateway and one private model worker.

The C++ worker is deliberately not a second product server.  It binds a random loopback
port and is reachable only by the Python gateway.  ``clozn serve`` owns both children,
monitors them, and restarts the worker after an unexpected exit.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import hashlib
import os
import socket
import subprocess
import threading
import time
import urllib.request
from types import MappingProxyType

from clozn.cli.engine_process import _free_port, find_engine_ex, spawn_engine
from clozn.cli.worker_handle import (
    WorkerHandle,
    WorkerRestartLimitError,
    terminate_process as _terminate,
)
from clozn.cli.worker_registry import (
    WorkerDefinition,
    WorkerRegistry,
    WorkerRegistryConfigError,
)

def gateway_health(port: int, timeout: float = 2.0) -> dict | None:
    """Return the public gateway's readiness document, or ``None`` when not ready."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/readyz", timeout=timeout) as response:
            data = json.loads(response.read())
        return data if isinstance(data, dict) and data.get("status") == "ok" else None
    except Exception:
        return None


def gateway_liveness(port: int, timeout: float = 2.0) -> dict | None:
    """Return the public gateway's liveness document, independent of workers."""
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{int(port)}/healthz", timeout=timeout
        ) as response:
            data = json.loads(response.read())
        return (
            data
            if isinstance(data, dict) and data.get("status") == "ok"
            else None
        )
    except Exception:
        return None


def port_is_open(port: int, timeout: float = 0.2) -> bool:
    """True when any process is listening on the loopback port."""
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _worker_template_fingerprint(port: int) -> str:
    """Canonical live template rendering fingerprint for one private worker."""
    from clozn.runs.identity import CANONICAL_CONVERSATION

    body = json.dumps({
        "messages": list(CANONICAL_CONVERSATION),
        "add_assistant": True,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{int(port)}/apply_template",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        payload = json.loads(response.read())
    rendered = payload.get("prompt") if isinstance(payload, dict) else None
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("worker returned no canonical rendered prompt")
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()[:16]


def _qualified_engine_discoveries(
    definitions: tuple[WorkerDefinition, ...],
) -> dict[str, object]:
    """Resolve and verify the exact executable/backend before any worker spawn."""
    from clozn.artifacts.contracts import sha256_file

    discoveries = {}
    digest_by_exe = {}
    for definition in definitions:
        discovery = find_engine_ex(prefer_gpu=definition.prefer_gpu)
        backend = discovery.backend or ("gpu" if discovery.gpu else "cpu")
        if backend != definition.runtime_key.backend:
            raise WorkerRegistryConfigError(
                f"{definition.model_id!r} configured backend "
                f"{definition.runtime_key.backend!r} does not match selected "
                f"engine backend {backend!r}"
            )
        exe = os.path.abspath(discovery.exe)
        digest = digest_by_exe.get(exe)
        if digest is None:
            digest = sha256_file(exe)
            digest_by_exe[exe] = digest
        observed_build = f"sha256:{digest}"
        if definition.runtime_key.engine_build != observed_build:
            raise WorkerRegistryConfigError(
                f"{definition.model_id!r} configured engine_build does not "
                "match the selected executable SHA-256"
            )
        discoveries[definition.model_id] = discovery
    return discoveries


def _selected_engine_discovery(prefer_gpu: bool) -> tuple[object, str]:
    """Resolve once and hash the executable bytes used by this runtime."""
    from clozn.artifacts.contracts import sha256_file

    discovery = find_engine_ex(prefer_gpu=prefer_gpu)
    return discovery, sha256_file(os.path.abspath(discovery.exe))


@dataclass(frozen=True)
class RuntimeConfig:
    """The complete immutable launch specification for one product runtime."""

    model: str
    public_port: int
    flags: Mapping[str, object] = field(default_factory=dict)
    prefer_gpu: bool = True
    host: str = "127.0.0.1"
    worker_port: int | None = None
    gateway_boot_timeout: float = 45.0
    worker_boot_timeout: float = 180.0
    restart_limit: int = 3
    restart_window: float = 60.0
    worker_definitions: tuple[WorkerDefinition, ...] = ()
    default_model_id: str | None = None
    preload_model_ids: tuple[str, ...] = ()
    max_loaded_models: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "flags", MappingProxyType(dict(self.flags)))
        definitions = tuple(self.worker_definitions)
        object.__setattr__(self, "worker_definitions", definitions)
        object.__setattr__(
            self, "preload_model_ids", tuple(self.preload_model_ids)
        )
        if definitions:
            if not self.default_model_id:
                raise WorkerRegistryConfigError(
                    "managed runtime requires default_model_id"
                )
            # Configuration validation only; no process starts here.
            WorkerRegistry(
                definitions,
                default_model_id=self.default_model_id,
                preload_model_ids=self.preload_model_ids,
                max_loaded_workers=self.max_loaded_models,
            )

    @property
    def managed_models(self) -> bool:
        return bool(self.worker_definitions)


@dataclass
class RuntimeStack:
    """The supervised pair created from a :class:`RuntimeConfig`."""

    config: RuntimeConfig
    worker_port: int
    worker: subprocess.Popen
    gateway: object
    worker_health: dict
    gpu: bool
    worker_log: object | None = None
    _restart_times: list[float] = field(default_factory=list)
    _stopping: bool = False
    _worker_handle: WorkerHandle | None = field(default=None, repr=False)
    worker_registry: WorkerRegistry | None = field(default=None, repr=False)
    _maintenance_stop: threading.Event | None = field(default=None, repr=False)
    _maintenance_thread: threading.Thread | None = field(default=None, repr=False)
    _maintenance_callback: object | None = field(default=None, repr=False)
    _maintenance_error: BaseException | None = field(default=None, repr=False)

    def __post_init__(self):
        if self.worker_registry is not None:
            self._sync_registry_primary()
            return
        if self._worker_handle is None:
            self._worker_handle = WorkerHandle(
                model=self.config.model,
                port=self.worker_port,
                flags=self.config.flags,
                prefer_gpu=self.config.prefer_gpu,
                boot_timeout=self.config.worker_boot_timeout,
                restart_limit=self.config.restart_limit,
                restart_window=self.config.restart_window,
                process=self.worker,
                health=self.worker_health,
                gpu=self.gpu,
                log=self.worker_log,
                # Resolve runtime_process.spawn_engine at call time.  Besides preserving the existing
                # test seam, this keeps RuntimeStack independent of worker_handle's module default.
                spawn=lambda *args, **kwargs: spawn_engine(*args, **kwargs),
                restart_times=self._restart_times,
            )
        else:
            self._worker_handle.restart_times = self._restart_times

    @property
    def public_port(self) -> int:
        return self.config.public_port

    def registry_fields(self) -> dict:
        """Stable process metadata consumed by ``clozn ps/run/stop/studio``."""
        if self.worker_registry is not None:
            status = self.worker_registry.status()
            workers = [{
                "model_id": worker["model_id"],
                "state": worker["state"],
                "default": worker["default"],
                "preloaded": worker["preloaded"],
                "worker_pid": worker["worker_pid"],
                "worker_port": worker["worker_port"],
                "failure_code": worker["failure_code"],
            } for worker in status["workers"]]
            return {
                "kind": "runtime",
                "gateway_pid": self.gateway.pid,
                "worker_pid": self.worker.pid,
                "worker_port": self.worker_port,
                "default_model_id": status["default_model_id"],
                "preload_model_ids": status["preload_model_ids"],
                "max_loaded_models": status["max_loaded_workers"],
                "models": workers,
            }
        return {
            "kind": "runtime",
            "gateway_pid": self.gateway.pid,
            **self._worker_handle.registry_fields(),
        }

    def stop(self) -> None:
        self._stopping = True
        if self._maintenance_stop is not None:
            self._maintenance_stop.set()
        if (
            self._maintenance_thread is not None
            and self._maintenance_thread is not threading.current_thread()
        ):
            self._maintenance_thread.join(timeout=5.0)
        _terminate(self.gateway)
        if self.worker_registry is not None:
            self.worker_registry.stop_all()
        else:
            self._worker_handle.stop()

    def _sync_registry_primary(self) -> None:
        if self.worker_registry is None:
            return
        status = self.worker_registry.status()
        ordered = [status["default_model_id"]] + [
            worker["model_id"] for worker in status["workers"]
            if worker["model_id"] != status["default_model_id"]
        ]
        for model_id in ordered:
            handle = self.worker_registry.worker_handle(model_id)
            if handle is not None and handle.process.poll() is None:
                self.worker_port = handle.port
                self.worker = handle.process
                self.worker_health = handle.health
                self.gpu = handle.gpu
                return

    def recover_worker(self, model_id: str) -> bool:
        """Recover one failed preload in the in-process registry."""
        if self.worker_registry is None:
            raise RuntimeError("single-worker runtime has no managed registry")
        recovered = self.worker_registry.recover_failed(model_id)
        self._sync_registry_primary()
        return recovered

    def _restart_worker(self) -> None:
        from clozn.cli import main as ctx

        try:
            self._worker_handle.restart()
        except WorkerRestartLimitError as exc:
            self._restart_times = self._worker_handle.restart_times
            raise ctx.CloznError(str(exc)) from exc
        self.worker = self._worker_handle.process
        self.worker_health = self._worker_handle.health
        self.gpu = self._worker_handle.gpu
        self._restart_times = self._worker_handle.restart_times

    def wait(self, on_worker_restart=None, poll_interval: float = 0.25) -> int:
        """Monitor both children until the gateway exits; restart only the private worker."""
        self._maintenance_callback = on_worker_restart
        while True:
            gateway_code = self.gateway.poll()
            if gateway_code is not None:
                return int(gateway_code)
            if self._maintenance_error is not None:
                raise RuntimeError("worker maintenance loop failed") from self._maintenance_error
            if self._maintenance_thread is not None:
                time.sleep(poll_interval)
                continue
            if self.worker_registry is not None:
                before = self.worker_registry.routing_projection()
                self.worker_registry.maintain()
                after = self.worker_registry.routing_projection()
                if after != before:
                    self._sync_registry_primary()
                    if on_worker_restart is not None:
                        on_worker_restart(self)
                time.sleep(poll_interval)
                continue
            if self.worker.poll() is not None:
                self._restart_worker()
                if on_worker_restart is not None:
                    on_worker_restart(self)
            time.sleep(poll_interval)


class InProcessGateway:
    """Popen-shaped lifetime handle for a background in-process HTTP server."""

    def __init__(self, server):
        self.server = server
        self.pid = os.getpid()
        self.returncode = None
        self._thread = threading.Thread(
            target=self._serve,
            name="clozn-gateway",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def _serve(self) -> None:
        try:
            self.server.serve_forever()
        except BaseException:
            self.returncode = 1
        finally:
            try:
                self.server.server_close()
            except Exception:
                pass
            if self.returncode is None:
                self.returncode = 0

    def poll(self):
        return None if self._thread.is_alive() else self.returncode

    def terminate(self) -> None:
        if self.poll() is None:
            self.server.shutdown()

    def kill(self) -> None:
        self.terminate()

    def wait(self, timeout=None):
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise subprocess.TimeoutExpired("in-process gateway", timeout)
        return self.returncode


def _start_registry_maintenance(stack: RuntimeStack, *, poll_interval: float = 0.25) -> None:
    """Run the supervisor-owned registry poll independently of HTTP dispatch."""
    if stack.worker_registry is None or stack._maintenance_thread is not None:
        return
    stop_event = threading.Event()
    stack._maintenance_stop = stop_event

    def maintain():
        try:
            while not stop_event.wait(poll_interval):
                before = stack.worker_registry.routing_projection()
                stack.worker_registry.maintain()
                after = stack.worker_registry.routing_projection()
                if after != before:
                    stack._sync_registry_primary()
                    callback = stack._maintenance_callback
                    if callback is not None:
                        callback(stack)
        except BaseException as error:
            if not stop_event.is_set():
                stack._maintenance_error = error

    thread = threading.Thread(
        target=maintain,
        name="clozn-worker-maintenance",
        daemon=True,
    )
    stack._maintenance_thread = thread
    thread.start()


def _spawn_managed_runtime_inprocess(
    config: RuntimeConfig,
    *,
    worker_log=None,
) -> RuntimeStack:
    """ADR 008 Stage 2/3: managed workers plus the in-process gateway and cold loader.

    The registry remains supervisor-owned and the gateway receives an in-memory
    projection source plus the live cold-loader and busy-call adapters.
    """
    from clozn.cli import main as ctx
    from clozn.server import app as server_app
    from clozn.server.model_routing import (
        ColdLoadOutcome,
        InMemoryProjectionRouter,
    )

    definitions = tuple(
        replace(
            definition,
            log=definition.log if definition.log is not None else worker_log,
        )
        for definition in config.worker_definitions
    )
    discoveries = _qualified_engine_discoveries(definitions)

    def managed_spawn(model, port, flags, *, model_id=None, **kwargs):
        if model_id not in discoveries:
            raise WorkerRegistryConfigError(
                f"spawn requested unknown managed model {model_id!r}"
            )
        managed_flags = dict(flags)
        managed_flags["_disable_auto_jlens"] = True
        return spawn_engine(
            model,
            port,
            managed_flags,
            engine_discovery=discoveries[model_id],
            **kwargs,
        )

    registry = WorkerRegistry(
        definitions,
        default_model_id=config.default_model_id,
        preload_model_ids=config.preload_model_ids,
        max_loaded_workers=config.max_loaded_models,
        spawn=managed_spawn,
        template_probe=_worker_template_fingerprint,
        port_factory=lambda: _free_port(),
        busy_tracking_wired=True,
    )
    gateway = None
    try:
        status = registry.start_preloaded()
        ready = [worker for worker in status["workers"] if worker["state"] == "ready"]
        if not ready:
            failures = ", ".join(
                f"{worker['model_id']}={worker['failure_code'] or worker['state']}"
                for worker in status["workers"] if worker["preloaded"]
            )
            raise ctx.CloznError(
                "no configured preload became ready"
                + (f" ({failures})" if failures else "")
            )

        def engine_factory(port):
            client_type = getattr(server_app, "EngineClient", None)
            if client_type is None:
                raise RuntimeError("the private worker client is unavailable")
            return client_type(port=port)

        def registry_cold_loader(model_id: str, timeout: float) -> ColdLoadOutcome:
            result = registry.ensure_loaded(model_id, timeout=timeout)
            worker = next(
                item for item in registry.status()["workers"]
                if item["model_id"] == model_id
            )
            return ColdLoadOutcome(
                state=result.state_after.value,
                kind=result.kind,
                outcome=result.outcome,
                coalesced=result.coalesced,
                wait_ms=result.wait_ms,
                worker_port=worker["worker_port"],
                worker_identity=worker["worker_identity"],
                failure_code=result.failure_code,
                message=result.error,
                event_id=result.event_id,
            )

        router = InMemoryProjectionRouter(
            registry.routing_projection,
            engine_factory=engine_factory,
            substrate_factory=lambda engine: server_app.EngineSubstrate(engine=engine),
            loader=registry_cold_loader,
            worker_call_tracker=registry.track_call,
        )
        _control_sub, control_engine = router.control_pair()
        if control_engine is None:
            raise ctx.CloznError("no configured preload became reachable")
        server = server_app.build_server(
            host=config.host,
            port=config.public_port,
            engine=control_engine,
            model_router=router,
        )
        gateway = InProcessGateway(server)
        gateway.start()

        started = time.monotonic()
        while time.monotonic() - started < config.gateway_boot_timeout:
            if gateway.poll() is not None:
                raise ctx.CloznError(
                    f"in-process gateway exited during startup (code {gateway.returncode})"
                )
            registry.maintain()
            if gateway_health(config.public_port):
                current_ready = [
                    worker for worker in registry.status()["workers"]
                    if worker["state"] == "ready"
                ]
                if not current_ready:
                    raise ctx.CloznError(
                        "all configured preloads failed during gateway startup"
                    )
                current_preferred = next(
                    (worker for worker in current_ready
                     if worker["model_id"] == config.default_model_id),
                    current_ready[0],
                )
                control_handle = registry.worker_handle(current_preferred["model_id"])
                if control_handle is None:
                    raise RuntimeError("ready preload has no worker handle")
                stack = RuntimeStack(
                    config=config,
                    worker_port=control_handle.port,
                    worker=control_handle.process,
                    gateway=gateway,
                    worker_health=control_handle.health,
                    gpu=control_handle.gpu,
                    worker_log=worker_log,
                    worker_registry=registry,
                )
                _start_registry_maintenance(stack)
                return stack
            time.sleep(0.2)
        raise ctx.CloznError(
            f"in-process gateway did not become ready within "
            f"{config.gateway_boot_timeout:g}s"
        )
    except BaseException:
        _terminate(gateway)
        registry.stop_all()
        raise


def _spawn_legacy_runtime_inprocess(
    config: RuntimeConfig,
    *,
    worker_log=None,
) -> RuntimeStack:
    """Run a single configured model in the merged gateway process."""
    from clozn.cli import main as ctx
    from clozn.server import app as server_app

    worker_port = config.worker_port or _free_port()
    worker_handle = None
    gateway = None
    try:
        discovery, _executable_sha256 = _selected_engine_discovery(config.prefer_gpu)
        legacy_spawn = lambda *args, **kwargs: spawn_engine(
            *args, engine_discovery=discovery, **kwargs
        )
        worker_handle = WorkerHandle.start(
            model=config.model,
            port=worker_port,
            flags=config.flags,
            prefer_gpu=config.prefer_gpu,
            log=worker_log,
            boot_timeout=config.worker_boot_timeout,
            restart_limit=config.restart_limit,
            restart_window=config.restart_window,
            spawn=legacy_spawn,
        )
        client_type = getattr(server_app, "EngineClient", None)
        if client_type is None:
            raise ctx.CloznError("the private worker client is unavailable")
        engine = client_type(port=worker_port)
        sub = server_app.EngineSubstrate(engine=engine)
        server = server_app.build_server(
            host=config.host,
            port=config.public_port,
            engine=engine,
            sub=sub,
        )
        gateway = InProcessGateway(server)
        gateway.start()
        started = time.monotonic()
        while time.monotonic() - started < config.gateway_boot_timeout:
            if gateway.poll() is not None:
                raise ctx.CloznError(
                    f"in-process gateway exited during startup (code {gateway.returncode})"
                )
            if worker_handle.process.poll() is not None:
                raise ctx.CloznError(
                    f"model worker exited during gateway startup (code {worker_handle.process.returncode})"
                )
            if gateway_health(config.public_port):
                return RuntimeStack(
                    config=config,
                    worker_port=worker_port,
                    worker=worker_handle.process,
                    gateway=gateway,
                    worker_health=worker_handle.health,
                    gpu=worker_handle.gpu,
                    worker_log=worker_log,
                    _worker_handle=worker_handle,
                )
            time.sleep(0.2)
        raise ctx.CloznError(
            f"in-process gateway did not become ready within "
            f"{config.gateway_boot_timeout:g}s"
        )
    except BaseException:
        _terminate(gateway)
        if worker_handle is not None:
            worker_handle.stop()
        raise


def _spawn_runtime_inprocess(
    config: RuntimeConfig,
    *,
    worker_log=None,
) -> RuntimeStack:
    if config.managed_models:
        return _spawn_managed_runtime_inprocess(
            config, worker_log=worker_log
        )
    return _spawn_legacy_runtime_inprocess(
        config, worker_log=worker_log
    )


def spawn_runtime_inprocess(
    config: RuntimeConfig, *, worker_log=None
) -> RuntimeStack:
    """Explicit entry point for the merged in-process runtime."""
    return _spawn_runtime_inprocess(
        config, worker_log=worker_log
    )


def spawn_runtime(config: RuntimeConfig, *, worker_log=None) -> RuntimeStack:
    """Launch the private worker(s) and serve the public API in this process."""
    from clozn.cli import main as ctx

    if port_is_open(config.public_port):
        raise ctx.CloznError(
            f"port {config.public_port} is already in use. Pick another with --port."
        )
    try:
        return spawn_runtime_inprocess(config, worker_log=worker_log)
    except WorkerRegistryConfigError as error:
        raise ctx.CloznError(str(error)) from None
