"""Lifecycle for the product runtime: one public gateway and one private model worker.

The C++ worker is deliberately not a second product server.  It binds a random loopback
port and is reachable only by the Python gateway.  ``clozn serve`` owns both children,
monitors them, and restarts the worker after an unexpected exit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from types import MappingProxyType
from typing import Mapping

from clozn.cli.engine_process import REPO, _free_port, _log_tail, find_engine_ex, spawn_engine


def gateway_health(port: int, timeout: float = 2.0) -> dict | None:
    """Return the public gateway's readiness document, or ``None`` when not ready."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/readyz", timeout=timeout) as response:
            data = json.loads(response.read())
        return data if isinstance(data, dict) and data.get("status") == "ok" else None
    except Exception:
        return None


def port_is_open(port: int, timeout: float = 0.2) -> bool:
    """True when any process is listening on the loopback port."""
    try:
        with socket.create_connection(("127.0.0.1", int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _terminate(proc, timeout: float = 5.0) -> None:
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

    def __post_init__(self):
        object.__setattr__(self, "flags", MappingProxyType(dict(self.flags)))


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

    @property
    def public_port(self) -> int:
        return self.config.public_port

    def registry_fields(self) -> dict:
        """Stable process metadata consumed by ``clozn ps/run/stop/studio``."""
        return {
            "kind": "runtime",
            "gateway_pid": self.gateway.pid,
            "worker_pid": self.worker.pid,
            "worker_port": self.worker_port,
        }

    def stop(self) -> None:
        self._stopping = True
        _terminate(self.gateway)
        _terminate(self.worker)

    def _restart_worker(self) -> None:
        from clozn.cli import main as ctx

        now = time.monotonic()
        self._restart_times = [t for t in self._restart_times if now - t <= self.config.restart_window]
        if len(self._restart_times) >= self.config.restart_limit:
            raise ctx.CloznError(
                f"model worker exited {self.config.restart_limit} times within "
                f"{int(self.config.restart_window)}s; giving up. {_log_tail(self.worker_log)}"
            )
        self._restart_times.append(now)
        self.worker, self.worker_health, self.gpu = spawn_engine(
            self.config.model,
            self.worker_port,
            self.config.flags,
            prefer_gpu=self.config.prefer_gpu,
            logf=self.worker_log,
            boot_timeout=self.config.worker_boot_timeout,
        )

    def wait(self, on_worker_restart=None, poll_interval: float = 0.25) -> int:
        """Monitor both children until the gateway exits; restart only the private worker."""
        while True:
            gateway_code = self.gateway.poll()
            if gateway_code is not None:
                return int(gateway_code)
            if self.worker.poll() is not None:
                self._restart_worker()
                if on_worker_restart is not None:
                    on_worker_restart(self)
            time.sleep(poll_interval)


def spawn_runtime(config: RuntimeConfig, *, worker_log=None, gateway_log=None) -> RuntimeStack:
    """Launch worker, then gateway, and return only after the public API is ready."""
    from clozn.cli import main as ctx

    if port_is_open(config.public_port):
        raise ctx.CloznError(f"port {config.public_port} is already in use. Pick another with --port.")

    worker_port = config.worker_port or _free_port()
    worker = gateway = None
    try:
        worker, health, gpu = spawn_engine(
            config.model,
            worker_port,
            config.flags,
            prefer_gpu=config.prefer_gpu,
            logf=worker_log,
            boot_timeout=config.worker_boot_timeout,
        )
        env = dict(os.environ)
        env["CLOZN_ENGINE_PORT"] = str(worker_port)
        env["CLOZN_RUNTIME_KIND"] = "product"
        env["PYTHONUNBUFFERED"] = "1"
        # roadmap feature 01: the gateway is a SEPARATE subprocess (see the command below), so the
        # discovery-tier/backend/artifact facts spawn_engine's own find_engine() call just established
        # do not exist in that process unless handed across the boundary -- env vars are the same
        # mechanism CLOZN_ENGINE_PORT already uses. This is a second, cheap, purely-filesystem lookup
        # (never a re-download or re-extraction); its result is deterministic against the same on-disk
        # state spawn_engine() just read, so it is never allowed to fail the actual server start --
        # missing discovery metadata just means clozn.runs.identity_providers.engine_artifact reports
        # less, per the omit-don't-invent rule the rest of clozn.runs follows.
        try:
            discovery = find_engine_ex(prefer_gpu=config.prefer_gpu)
            env["CLOZN_ENGINE_DISCOVERY_SOURCE"] = discovery.discovery_source
            if discovery.backend:
                env["CLOZN_ENGINE_BACKEND"] = discovery.backend
            if discovery.artifact_sha256:
                env["CLOZN_ENGINE_ARTIFACT_SHA256"] = discovery.artifact_sha256
            if discovery.engine_version:
                env["CLOZN_ENGINE_VERSION"] = discovery.engine_version
            if discovery.build_id:
                env["CLOZN_ENGINE_BUILD_ID"] = discovery.build_id
            if discovery.llama_cpp_commit:
                env["CLOZN_ENGINE_LLAMA_CPP_COMMIT"] = discovery.llama_cpp_commit
        except Exception:
            pass
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
        )

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
                )
            time.sleep(0.2)
        raise ctx.CloznError(
            f"gateway did not become ready within {config.gateway_boot_timeout:g}s. "
            f"{_log_tail(gateway_log)}"
        )
    except BaseException:
        _terminate(gateway)
        _terminate(worker)
        raise
