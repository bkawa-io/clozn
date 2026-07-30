"""Reusable lifecycle handle for one private GGUF worker.

This module deliberately knows nothing about public model routing.  It extracts
the lifecycle that the current single-model ``clozn serve`` supervisor already
owns: start, health/identity handshake result, bounded restart accounting,
graceful shutdown, and hard-termination fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import subprocess
import time
from typing import Callable, Mapping

from clozn.cli.engine_process import _log_tail, spawn_engine


SpawnWorker = Callable[..., tuple[subprocess.Popen, dict, bool]]


def terminate_process(proc, timeout: float = 5.0) -> None:
    """Stop one child, escalating to ``kill`` only when termination fails."""
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


class WorkerRestartLimitError(RuntimeError):
    """The worker exceeded its configured restart budget."""


@dataclass
class WorkerHandle:
    """The complete mutable lifecycle state for one private model worker."""

    model: str
    port: int
    flags: Mapping[str, object]
    prefer_gpu: bool
    boot_timeout: float
    restart_limit: int
    restart_window: float
    process: subprocess.Popen
    health: dict
    gpu: bool
    log: object | None = None
    spawn: SpawnWorker = spawn_engine
    restart_times: list[float] = field(default_factory=list)
    stopping: bool = False
    clock: Callable[[], float] = time.monotonic

    @classmethod
    def start(
        cls,
        *,
        model: str,
        port: int,
        flags: Mapping[str, object],
        prefer_gpu: bool,
        boot_timeout: float,
        restart_limit: int,
        restart_window: float,
        log=None,
        spawn: SpawnWorker = spawn_engine,
    ) -> "WorkerHandle":
        """Spawn and handshake one worker, returning its owned lifecycle state."""
        process, health, gpu = spawn(
            model,
            port,
            flags,
            prefer_gpu=prefer_gpu,
            logf=log,
            boot_timeout=boot_timeout,
        )
        return cls(
            model=model,
            port=port,
            flags=flags,
            prefer_gpu=prefer_gpu,
            boot_timeout=boot_timeout,
            restart_limit=restart_limit,
            restart_window=restart_window,
            process=process,
            health=health,
            gpu=gpu,
            log=log,
            spawn=spawn,
        )

    def registry_fields(self) -> dict:
        return {"worker_pid": self.process.pid, "worker_port": self.port}

    def stop(self) -> None:
        self.stopping = True
        terminate_process(self.process)

    def restart(self) -> None:
        """Restart after an unexpected exit, enforcing the existing time window."""
        now = self.clock()
        self.restart_times = [
            started for started in self.restart_times
            if now - started <= self.restart_window
        ]
        if len(self.restart_times) >= self.restart_limit:
            raise WorkerRestartLimitError(
                f"model worker exited {self.restart_limit} times within "
                f"{int(self.restart_window)}s; giving up. {_log_tail(self.log)}"
            )
        self.restart_times.append(now)
        self.process, self.health, self.gpu = self.spawn(
            self.model,
            self.port,
            self.flags,
            prefer_gpu=self.prefer_gpu,
            logf=self.log,
            boot_timeout=self.boot_timeout,
        )
