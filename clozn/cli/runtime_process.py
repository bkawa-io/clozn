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
import sys
import tempfile
import time
import urllib.request
from types import MappingProxyType
from typing import Mapping

from clozn.cli import process_guard
from clozn.cli.engine_process import REPO, _free_port, _log_tail, find_engine_ex, spawn_engine
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

_ENGINE_IDENTITY_ENV = (
    "CLOZN_ENGINE_DISCOVERY_SOURCE",
    "CLOZN_ENGINE_BACKEND",
    "CLOZN_ENGINE_ARTIFACT_SHA256",
    "CLOZN_ENGINE_VERSION",
    "CLOZN_ENGINE_BUILD_ID",
    "CLOZN_ENGINE_LLAMA_CPP_COMMIT",
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


class RoutingProjectionTransport:
    """Private atomic supervisor-to-gateway routing projection."""

    def __init__(self, directory: str, path: str):
        self.directory = os.path.abspath(directory)
        self.path = os.path.abspath(path)
        self._closed = False

    @classmethod
    def create(cls, projection: Mapping) -> "RoutingProjectionTransport":
        directory = tempfile.mkdtemp(prefix="clozn-routing-")
        transport = cls(directory, os.path.join(directory, "projection.json"))
        try:
            transport.publish(projection)
        except BaseException:
            transport.close()
            raise
        return transport

    def publish(self, projection: Mapping) -> None:
        if self._closed:
            raise RuntimeError("routing projection transport is closed")
        if not isinstance(projection, Mapping):
            raise TypeError("routing projection must be an object")
        raw = json.dumps(
            dict(projection),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        temporary = os.path.join(self.directory, "projection.next")
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            try:
                os.remove(temporary)
            except OSError:
                pass
            raise
        os.replace(temporary, self.path)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for path in (
            os.path.join(self.directory, "projection.next"),
            self.path,
        ):
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass
        try:
            os.rmdir(self.directory)
        except OSError:
            pass


@dataclass(frozen=True)
class RuntimeConfig:
    """The complete immutable launch specification for one product runtime."""

    model: str
    public_port: int
    flags: Mapping[str, object] = field(default_factory=dict)
    prefer_gpu: bool = True
    host: str = "127.0.0.1"
    worker_port: int | None = None
    gateway_python: str = field(default_factory=lambda: sys.executable)
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
    gateway: subprocess.Popen
    worker_health: dict
    gpu: bool
    worker_log: object | None = None
    gateway_log: object | None = None
    _restart_times: list[float] = field(default_factory=list)
    _stopping: bool = False
    _worker_handle: WorkerHandle | None = field(default=None, repr=False)
    worker_registry: WorkerRegistry | None = field(default=None, repr=False)
    routing_transport: RoutingProjectionTransport | None = field(
        default=None, repr=False
    )

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
        _terminate(self.gateway)
        if self.worker_registry is not None:
            self.worker_registry.stop_all()
            if self.routing_transport is not None:
                self.routing_transport.close()
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

    def _publish_registry(self) -> None:
        if self.worker_registry is None or self.routing_transport is None:
            return
        self.routing_transport.publish(
            self.worker_registry.routing_projection()
        )

    def recover_worker(self, model_id: str) -> bool:
        """Recover one failed preload and refresh the gateway projection."""
        if self.worker_registry is None:
            raise RuntimeError("single-worker runtime has no managed registry")
        recovered = self.worker_registry.recover_failed(model_id)
        self._sync_registry_primary()
        self._publish_registry()
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
        while True:
            gateway_code = self.gateway.poll()
            if gateway_code is not None:
                return int(gateway_code)
            if self.worker_registry is not None:
                before = self.worker_registry.routing_projection()
                self.worker_registry.maintain()
                after = self.worker_registry.routing_projection()
                if after != before:
                    self._sync_registry_primary()
                    self.routing_transport.publish(after)
                    if on_worker_restart is not None:
                        on_worker_restart(self)
                time.sleep(poll_interval)
                continue
            if self.worker.poll() is not None:
                self._restart_worker()
                if on_worker_restart is not None:
                    on_worker_restart(self)
            time.sleep(poll_interval)


def _spawn_managed_runtime(
    config: RuntimeConfig,
    *,
    worker_log=None,
    gateway_log=None,
) -> RuntimeStack:
    """Launch qualified preloads, then one public file-backed gateway."""
    from clozn.cli import main as ctx

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
    )
    gateway = None
    transport = None
    try:
        status = registry.start_preloaded()
        ready = [
            worker for worker in status["workers"]
            if worker["state"] == "ready"
        ]
        if not ready:
            failures = ", ".join(
                f"{worker['model_id']}={worker['failure_code'] or worker['state']}"
                for worker in status["workers"]
                if worker["preloaded"]
            )
            raise ctx.CloznError(
                "no configured preload became ready"
                + (f" ({failures})" if failures else "")
            )
        transport = RoutingProjectionTransport.create(
            registry.routing_projection()
        )
        preferred = next(
            (
                worker for worker in ready
                if worker["model_id"] == status["default_model_id"]
            ),
            ready[0],
        )
        control_handle = registry.worker_handle(preferred["model_id"])
        if control_handle is None:  # status/handle internal invariant
            raise RuntimeError("ready preload has no worker handle")

        env = dict(os.environ)
        # These variables describe one process-wide engine and are therefore
        # false in a gateway serving multiple exact runtime keys. Per-worker
        # routing receipts are the only managed identity authority.
        for name in _ENGINE_IDENTITY_ENV:
            env.pop(name, None)
        env["CLOZN_ENGINE_PORT"] = str(control_handle.port)
        env["CLOZN_MODEL_ROUTING_FILE"] = transport.path
        env["CLOZN_RUNTIME_KIND"] = "product"
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            config.gateway_python,
            "-m",
            "clozn.server.app",
            "--host",
            config.host,
            "--port",
            str(config.public_port),
        ]
        gateway = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=gateway_log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            **process_guard.subprocess_kwargs(),
        )
        # Parent-death guard (ADR 008 Stage 0): see engine_process.spawn_engine's identical call for
        # the full rationale. Best-effort, never raises.
        process_guard.guard(gateway)

        started = time.monotonic()
        while time.monotonic() - started < config.gateway_boot_timeout:
            if gateway.poll() is not None:
                raise ctx.CloznError(
                    f"gateway exited during startup (code {gateway.returncode}). "
                    f"{_log_tail(gateway_log)}"
                )
            before = registry.routing_projection()
            registry.maintain()
            after = registry.routing_projection()
            if after != before:
                transport.publish(after)
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
                    (
                        worker for worker in current_ready
                        if worker["model_id"] == config.default_model_id
                    ),
                    current_ready[0],
                )
                control_handle = registry.worker_handle(
                    current_preferred["model_id"]
                )
                if control_handle is None:
                    raise RuntimeError("ready preload has no worker handle")
                return RuntimeStack(
                    config=config,
                    worker_port=control_handle.port,
                    worker=control_handle.process,
                    gateway=gateway,
                    worker_health=control_handle.health,
                    gpu=control_handle.gpu,
                    worker_log=worker_log,
                    gateway_log=gateway_log,
                    worker_registry=registry,
                    routing_transport=transport,
                )
            time.sleep(0.2)
        raise ctx.CloznError(
            f"gateway did not become ready within "
            f"{config.gateway_boot_timeout:g}s. {_log_tail(gateway_log)}"
        )
    except BaseException:
        _terminate(gateway)
        registry.stop_all()
        if transport is not None:
            transport.close()
        raise


def spawn_runtime(config: RuntimeConfig, *, worker_log=None, gateway_log=None) -> RuntimeStack:
    """Launch worker, then gateway, and return only after the public API is ready."""
    from clozn.cli import main as ctx

    if port_is_open(config.public_port):
        raise ctx.CloznError(f"port {config.public_port} is already in use. Pick another with --port.")
    if config.managed_models:
        try:
            return _spawn_managed_runtime(
                config,
                worker_log=worker_log,
                gateway_log=gateway_log,
            )
        except WorkerRegistryConfigError as error:
            raise ctx.CloznError(str(error)) from None

    worker_port = config.worker_port or _free_port()
    worker_handle = None
    worker = gateway = None
    try:
        # Resolve exactly once.  The same executable selection is reused by
        # initial launch and every WorkerHandle restart, while its bytes'
        # digest crosses to the gateway as immutable engine-build evidence.
        discovery, executable_sha256 = _selected_engine_discovery(
            config.prefer_gpu
        )
        legacy_spawn = lambda *args, **kwargs: spawn_engine(
            *args,
            engine_discovery=discovery,
            **kwargs,
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
            # Resolve the compatibility seam in this module, not worker_handle's import-time default.
            spawn=legacy_spawn,
        )
        worker, health, gpu = (
            worker_handle.process,
            worker_handle.health,
            worker_handle.gpu,
        )
        env = dict(os.environ)
        # A user/shell-stale managed projection must never hijack the legacy
        # compatibility path.
        env.pop("CLOZN_MODEL_ROUTING_FILE", None)
        for name in _ENGINE_IDENTITY_ENV:
            env.pop(name, None)
        env["CLOZN_ENGINE_PORT"] = str(worker_port)
        env["CLOZN_RUNTIME_KIND"] = "product"
        env["PYTHONUNBUFFERED"] = "1"
        # These facts describe the exact selection handed to spawn_engine
        # above.  artifact_sha256 is deliberately the selected executable's
        # bytes, including repository development builds.
        env["CLOZN_ENGINE_DISCOVERY_SOURCE"] = discovery.discovery_source
        if discovery.backend:
            env["CLOZN_ENGINE_BACKEND"] = discovery.backend
        env["CLOZN_ENGINE_ARTIFACT_SHA256"] = executable_sha256
        if discovery.engine_version:
            env["CLOZN_ENGINE_VERSION"] = discovery.engine_version
        if discovery.build_id:
            env["CLOZN_ENGINE_BUILD_ID"] = discovery.build_id
        if discovery.llama_cpp_commit:
            env["CLOZN_ENGINE_LLAMA_CPP_COMMIT"] = discovery.llama_cpp_commit
        command = [
            config.gateway_python,
            "-m",
            "clozn.server.app",
            "--host",
            config.host,
            "--port",
            str(config.public_port),
        ]
        gateway = subprocess.Popen(
            command,
            cwd=REPO,
            env=env,
            stdout=gateway_log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            **process_guard.subprocess_kwargs(),
        )
        # Parent-death guard (ADR 008 Stage 0): see engine_process.spawn_engine's identical call for
        # the full rationale. Best-effort, never raises.
        process_guard.guard(gateway)

        started = time.monotonic()
        while time.monotonic() - started < config.gateway_boot_timeout:
            if gateway.poll() is not None:
                raise ctx.CloznError(
                    f"gateway exited during startup (code {gateway.returncode}). {_log_tail(gateway_log)}"
                )
            if worker.poll() is not None:
                raise ctx.CloznError(
                    f"model worker exited during gateway startup (code {worker.returncode}). "
                    f"{_log_tail(worker_log)}"
                )
            if gateway_health(config.public_port):
                return RuntimeStack(
                    config=config,
                    worker_port=worker_port,
                    worker=worker,
                    gateway=gateway,
                    worker_health=health,
                    gpu=gpu,
                    worker_log=worker_log,
                    gateway_log=gateway_log,
                    _worker_handle=worker_handle,
                )
            time.sleep(0.2)
        raise ctx.CloznError(
            f"gateway did not become ready within {config.gateway_boot_timeout:g}s. "
            f"{_log_tail(gateway_log)}"
        )
    except BaseException:
        _terminate(gateway)
        if worker_handle is not None:
            worker_handle.stop()
        else:
            _terminate(worker)
        raise
