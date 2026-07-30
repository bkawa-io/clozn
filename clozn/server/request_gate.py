"""Bounded serialization for stateful product operations.

The current product adapter keeps generation metadata, steering, and memory state on one
shared object.  Until those become request-local, concurrent POST dispatch would let two
runs overwrite each other's evidence.  This gate admits a bounded queue and executes one
POST at a time; GET health, Studio assets, and run inspection remain concurrent.
"""
from __future__ import annotations

import os
import threading
import time


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


def _positive_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, default))
        return value if value > 0 else default
    except (TypeError, ValueError):
        return default


class RequestGate:
    def __init__(self, capacity: int = 32, wait_timeout: float = 600.0):
        self.capacity = max(1, int(capacity))
        self.wait_timeout = max(0.001, float(wait_timeout))
        self._slots = threading.BoundedSemaphore(self.capacity)
        self._turn = threading.Lock()
        self._state_lock = threading.Lock()
        self._active = 0
        self._waiting = 0

    @classmethod
    def from_env(cls):
        return cls(
            capacity=_positive_int("CLOZN_MAX_PENDING_REQUESTS", 32),
            wait_timeout=_positive_float("CLOZN_QUEUE_TIMEOUT", 600.0),
        )

    def acquire(self, cancel_check=None, poll_interval: float = 0.2) -> str | None:
        """Admit one request, or explain why not. Returns ``None`` on admission, else one of ``"full"`` |
        ``"timeout"`` | ``"cancelled"``.

        `cancel_check`, when given, is polled every `poll_interval` seconds while this request is QUEUED
        waiting for its turn (never while merely checking the bounded `_slots` semaphore just below, which
        is already non-blocking) -- a callable returning True means the caller's own liveness signal (in
        production: "has the requesting client's TCP connection already closed", see http_policy.
        client_gone) says to abandon the wait rather than occupy a queue slot for the full `wait_timeout`
        (600s by default) after nobody is left to serve. `cancel_check=None` preserves the exact original
        single blocking wait -- every existing caller/test that has no liveness probe to offer keeps the
        pre-cancellation behavior byte-for-byte (one `Lock.acquire(timeout=...)` call, not a poll loop)."""
        if not self._slots.acquire(blocking=False):
            return "full"
        with self._state_lock:
            self._waiting += 1
        acquired = False
        cancelled = False
        try:
            if cancel_check is None:
                acquired = self._turn.acquire(timeout=self.wait_timeout)
            else:
                deadline = time.monotonic() + self.wait_timeout
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    acquired = self._turn.acquire(timeout=min(poll_interval, remaining))
                    if acquired:
                        break
                    if cancel_check():
                        cancelled = True
                        break
        finally:
            with self._state_lock:
                self._waiting -= 1
            if not acquired:
                self._slots.release()
        if not acquired:
            return "cancelled" if cancelled else "timeout"
        with self._state_lock:
            self._active = 1
        return None

    def release(self) -> None:
        with self._state_lock:
            self._active = 0
        self._turn.release()
        self._slots.release()

    def snapshot(self) -> dict:
        with self._state_lock:
            return {
                "active": self._active,
                "waiting": self._waiting,
                "capacity": self.capacity,
                "wait_timeout_seconds": self.wait_timeout,
            }


_REJECTION_STATUS = {"full": 429, "timeout": 503, "cancelled": 499}
_REJECTION_MESSAGE = {
    "full": "request queue is full",
    "timeout": "request timed out while waiting for the model",
    "cancelled": "client disconnected while queued",
}


def rejection_response(outcome: str) -> tuple:
    """The one HTTP shape for a RequestGate rejection: (status, json_body, extra_headers).

    Shared by every gate in the gateway -- the catch-all POST_GATE (app.py's
    do_POST) and any per-worker WorkerGateRegistry gate (model_routing.py's
    select_for_handler, legacy no-router branch) -- so a caller sees an
    identical envelope no matter which gate actually said no.
    """
    return (
        _REJECTION_STATUS[outcome],
        {"error": {"message": _REJECTION_MESSAGE[outcome], "type": "server_busy"}},
        {"Retry-After": "1"},
    )


class WorkerGateRegistry:
    """One independent :class:`RequestGate` per configured worker key.

    RT-05: replaces the single global ``POST_GATE`` turn for GENERATION
    specifically.  Two different workers' generation requests no longer
    contend for one lock -- each key gets its own bounded queue + one-turn
    gate, so worker A being busy (or slow) never delays a request to worker
    B.  Two requests naming the SAME worker still serialize: each gate's
    "turn" is exactly ``RequestGate``'s existing one-at-a-time admission,
    matching the private engine's own documented one-active-generation-path
    limit (docs/RUNTIME_SPLIT.md: "Continuous batching: not built -- one
    active generation path"). That same same-worker serialization is also
    what keeps ``EngineSubstrate``'s per-worker (not shared) ``self._request``
    / ``self.steer.strength`` state safe under this change -- see
    ``clozn/server/request_context.py``'s module docstring for the race this
    prevents; it is real, not theoretical, and this registry deliberately
    does not attempt to raise per-worker concurrency above 1 by default.

    ``acquire_all`` is the other half of the safety story: any POST whose
    worker scope this registry cannot precisely resolve (fork, checkpoint,
    replay, steer/memory mutation, ...) must still be excluded from EVERY
    worker's generation turn, not just one -- see ``clozn/server/app.py``'s
    ``do_POST``. It acquires every gate in one fixed (sorted) order, so two
    callers racing to drain the whole registry can never deadlock against
    each other, and rolls back cleanly on a partial failure.
    """

    def __init__(self, worker_keys, *, capacity: int | None = None,
                 wait_timeout: float | None = None):
        keys = sorted({str(key) for key in worker_keys})
        if not keys:
            raise ValueError("a worker gate registry needs at least one worker key")
        resolved_capacity = (
            capacity if capacity is not None
            else _positive_int("CLOZN_MAX_PENDING_REQUESTS", 32)
        )
        resolved_wait_timeout = (
            wait_timeout if wait_timeout is not None
            else _positive_float("CLOZN_QUEUE_TIMEOUT", 600.0)
        )
        self._keys = keys
        self._gates = {
            key: RequestGate(capacity=resolved_capacity, wait_timeout=resolved_wait_timeout)
            for key in keys
        }

    def worker_keys(self) -> list:
        return list(self._keys)

    def acquire_generation(self, worker_key, *, cancel_check=None):
        """Admit one generation request onto ``worker_key``'s own turn.

        Returns ``(None, release)`` on admission -- call ``release()``
        exactly once, always (see ``model_routing.ModelSelection.gate_release``)
        -- or ``(outcome, None)`` on refusal, where ``outcome`` is
        ``RequestGate``'s own ``"full"``/``"timeout"``/``"cancelled"``
        vocabulary.  Raises ``KeyError`` for a worker key this registry was
        never configured with: fail closed on identity, never silently
        borrow a different worker's gate.
        """
        gate = self._gates[worker_key]
        outcome = gate.acquire(cancel_check=cancel_check)
        if outcome is not None:
            return outcome, None
        return None, gate.release

    def acquire_all(self, *, cancel_check=None):
        """Exclude every configured worker's generation turn at once.

        For a POST this registry cannot attribute to one specific worker.
        Returns ``(None, release)`` on success (call ``release()`` exactly
        once) or ``(outcome, None)`` on refusal.  Rolls back cleanly --
        releases whatever it already grabbed -- on any rejection partway
        through, so a "full" on worker 3 of 5 never leaks the first two.
        """
        acquired = []
        for key in self._keys:
            outcome = self._gates[key].acquire(cancel_check=cancel_check)
            if outcome is not None:
                for gate in reversed(acquired):
                    gate.release()
                return outcome, None
            acquired.append(self._gates[key])

        def _release():
            for gate in reversed(acquired):
                gate.release()

        return None, _release

    def snapshot(self) -> dict:
        return {key: gate.snapshot() for key, gate in self._gates.items()}
