"""Lifecycle for the product runtime: one public gateway and one private model worker.

The C++ worker is deliberately not a second product server.  It binds a random loopback
port and is reachable only by the Python gateway.  ``clozn serve`` owns both children,
monitors them, and restarts the worker after an unexpected exit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
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
from clozn.cli.runtime_identity import (
    RuntimeKey,
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


def _selected_engine_discovery(prefer_gpu: bool) -> tuple[object, str]:
    """Resolve once and hash the executable bytes used by this runtime."""
    from clozn.artifacts.contracts import sha256_file

    discovery = find_engine_ex(prefer_gpu=prefer_gpu)
    return discovery, sha256_file(os.path.abspath(discovery.exe))


def _engine_artifact_context(discovery, executable_sha256: str) -> dict:
    """Identity facts owned by executable selection rather than the worker."""
    return {
        key: value
        for key, value in {
            "discovery_source": getattr(discovery, "discovery_source", None),
            "backend": getattr(discovery, "backend", None),
            "artifact_sha256": executable_sha256,
            "engine_version": getattr(discovery, "engine_version", None),
            "build_id": getattr(discovery, "build_id", None),
            "llama_cpp_commit": getattr(discovery, "llama_cpp_commit", None),
        }.items()
        if value not in (None, "")
    }


def _single_model_router(config, *, engine, sub, health, discovery, executable_sha256, worker_port):
    """Build the strict router for the one configured model."""
    from clozn.server.model_routing import PreloadedModelBinding, PreloadedModelRouter
    from clozn.server.request_gate import WorkerGateRegistry

    model_sha256 = health.get("model_sha256")
    context_size = health.get("n_ctx")
    protocol_version = health.get("protocol_version")
    worker_id = health.get("worker_generation_id")
    if not isinstance(model_sha256, str) or len(model_sha256) != 64:
        raise WorkerRegistryConfigError("worker health omitted the model SHA-256")
    if isinstance(context_size, bool) or not isinstance(context_size, int) or context_size < 1:
        raise WorkerRegistryConfigError("worker health omitted the context size")
    if not isinstance(protocol_version, str) or not protocol_version:
        raise WorkerRegistryConfigError("worker health omitted the protocol version")
    if not isinstance(worker_id, str) or not worker_id:
        raise WorkerRegistryConfigError("worker health omitted the process generation")
    template_fingerprint = _worker_template_fingerprint(worker_port)
    backend = discovery.backend or ("gpu" if discovery.gpu else "cpu")
    runtime_key = RuntimeKey(
        gguf_artifact_sha256=model_sha256,
        context_size=context_size,
        backend=backend,
        template_fingerprint=template_fingerprint,
        engine_build=f"sha256:{executable_sha256}",
    )
    health_model = str(health.get("model") or config.model)
    model_id = config.public_model_id or os.path.basename(health_model).removesuffix(".gguf")
    model_id = model_id or "clozn-local"
    worker_identity = {
        "worker_id": worker_id,
        "worker_generation_id": worker_id,
        "worker_generation": 1,
        "runtime_key_sha256": runtime_key.key_sha256,
        "protocol_version": protocol_version,
        "engine_build": runtime_key.engine_build,
        "backend": backend,
    }
    binding = PreloadedModelBinding(
        model_id=model_id,
        resolved_artifact={
            "model_id": model_id,
            "format": "gguf",
            "artifact_sha256": model_sha256,
        },
        runtime_key=runtime_key.as_dict(),
        state="ready",
        worker_identity=worker_identity,
        sub=sub,
        engine=engine,
        preloaded=True,
    )
    return PreloadedModelRouter(
        (binding,),
        default_model_id=model_id,
        preload_model_ids=(model_id,),
        max_loaded_workers=1,
        gate=WorkerGateRegistry((model_id,)),
    )


@dataclass(frozen=True)
class RuntimeConfig:
    """The complete immutable launch specification for one product runtime."""

    model: str
    public_port: int
    flags: Mapping[str, object] = field(default_factory=dict)
    prefer_gpu: bool = True
    host: str = "127.0.0.1"
    public_model_id: str | None = None
    worker_port: int | None = None
    gateway_boot_timeout: float = 45.0
    worker_boot_timeout: float = 180.0
    restart_limit: int = 3
    restart_window: float = 60.0
    def __post_init__(self):
        object.__setattr__(self, "flags", MappingProxyType(dict(self.flags)))


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

    def __post_init__(self):
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
        return {
            "kind": "runtime",
            "gateway_pid": self.gateway.pid,
            **self._worker_handle.registry_fields(),
        }

    def stop(self) -> None:
        self._stopping = True
        _terminate(self.gateway)
        self._worker_handle.stop()

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


def _spawn_single_model_runtime(
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
        discovery, executable_sha256 = _selected_engine_discovery(config.prefer_gpu)
        selected_spawn = lambda *args, **kwargs: spawn_engine(
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
            spawn=selected_spawn,
        )
        client_type = getattr(server_app, "EngineClient", None)
        if client_type is None:
            raise ctx.CloznError("the private worker client is unavailable")
        engine = client_type(port=worker_port)
        sub = server_app.EngineSubstrate(engine=engine)
        sub._engine_artifact_context = _engine_artifact_context(
            discovery, executable_sha256
        )
        router = _single_model_router(
            config,
            engine=engine,
            sub=sub,
            health=worker_handle.health,
            discovery=discovery,
            executable_sha256=executable_sha256,
            worker_port=worker_port,
        )
        server = server_app.build_server(
            host=config.host,
            port=config.public_port,
            engine=engine,
            model_router=router,
            public_model_id=config.public_model_id,
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


def spawn_runtime(config: RuntimeConfig, *, worker_log=None) -> RuntimeStack:
    """Launch the private worker and serve the public API in this process."""
    from clozn.cli import main as ctx

    if port_is_open(config.public_port):
        raise ctx.CloznError(
            f"port {config.public_port} is already in use. Pick another with --port."
        )
    try:
        return _spawn_single_model_runtime(config, worker_log=worker_log)
    except WorkerRegistryConfigError as error:
        raise ctx.CloznError(str(error)) from None
