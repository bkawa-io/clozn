"""Reusable lifecycle handle for one private GGUF worker.

This module deliberately knows nothing about public model routing.  It extracts
the lifecycle that the current single-model ``clozn serve`` supervisor already
owns: start, health/identity handshake result, bounded restart accounting,
graceful shutdown, and hard-termination fallback.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import subprocess
import threading
import time
from typing import Callable, Iterator, Mapping

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
    active_calls: int = field(default=0, compare=False, repr=False)
    _call_cond: threading.Condition = field(
        default_factory=threading.Condition, compare=False, repr=False
    )

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

    @property
    def busy(self) -> bool:
        """True while at least one private-worker call is in flight.

        RT-04 eviction must consult this, not a last-used timestamp: a worker
        mid generation or mid mutation is never idle even if nothing *started*
        recently, and a worker that finished five minutes ago is idle even if
        it was hot a moment before that.

        Caveat this property cannot express by itself: ``False`` means "zero calls
        currently tracked," which is only the same fact as "genuinely idle" if
        something actually calls :meth:`track_call` around every real call this
        handle makes. In production today nothing does -- real generation traffic
        flows gateway<->worker directly and never touches the supervisor process,
        so this reads permanently False regardless of what the worker is actually
        doing (see docs/design/006-cross-process-cold-load-protocol.md's Context
        section). ``WorkerRegistry`` is responsible for not trusting this value
        until it was constructed with ``busy_tracking_wired=True``; this property
        itself has no way to know whether it is being fed real traffic.
        """
        with self._call_cond:
            return self.active_calls > 0

    @contextmanager
    def track_call(self) -> Iterator[None]:
        """Mark one in-flight private-worker call for eviction to respect.

        Cooperative cancellation cannot interrupt an already in-flight private
        worker call -- protocol 1.1 carries no request ID for it -- so this
        only ever counts calls in and out. It never attempts to cancel one.
        """
        with self._call_cond:
            self.active_calls += 1
        try:
            yield
        finally:
            with self._call_cond:
                self.active_calls -= 1
                self._call_cond.notify_all()

    def wait_until_idle(self, timeout: float | None = None) -> bool:
        """Block for every call ``track_call`` currently knows about to finish.

        Returns False on timeout without cancelling anything. There is no
        protocol-level cancellation to fall back on, so a caller that gets
        False back must say so honestly (typically ``EvictionTimeoutError``)
        rather than proceed as though the worker had gone idle.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._call_cond:
            while self.active_calls > 0:
                remaining = None if deadline is None else deadline - time.monotonic()
                if deadline is not None and remaining <= 0:
                    return False
                self._call_cond.wait(remaining)
            return True

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
