"""SetupLock -- an advisory, single-writer lock for ~/.clozn/engines/, so two concurrent `clozn setup`
invocations do not race the same install directory or registry.json.

`os.O_CREAT | os.O_EXCL` is the only cross-platform-atomic "create if absent" primitive the stdlib
offers without a third-party dependency (pyproject.toml's `dependencies = []` rules out a package like
filelock/portalocker). A lock file left behind by a crashed process is treated as stale after
LOCK_STALE_SECONDS -- not because the PID inside it is verified alive (Windows PID reuse makes that
unreliable to check from a different process), but because a `clozn setup` run genuinely in progress for
longer than that is itself the anomaly worth surfacing, and the alternative -- one crashed run's lock
file blocking every future `clozn setup` forever -- is worse.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from clozn.setup.errors import LockError

LOCK_STALE_SECONDS = 15 * 60   # a real download + extract + qualify should never take this long


class SetupLock:
    """`with SetupLock(path):` raises LockError immediately if another invocation holds it -- no
    blocking or retry loop, since `clozn setup` is an interactive/CI command, not a work queue."""

    def __init__(self, path: str):
        self.path = path

    def __enter__(self) -> "SetupLock":
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        try:
            self._acquire()
        except FileExistsError:
            if self._steal_if_stale():
                self._acquire()
            else:
                raise LockError(
                    f"another `clozn setup` appears to be running ({self.path} is held). If you are "
                    f"certain none is (e.g. a previous run crashed), delete that file and retry."
                ) from None
        return self

    def _acquire(self) -> None:
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, datetime.now(timezone.utc).isoformat().encode("utf-8"))
        finally:
            os.close(fd)

    def _steal_if_stale(self) -> bool:
        try:
            age = time.time() - os.path.getmtime(self.path)
        except OSError:
            return True   # vanished between our EEXIST and this stat -- safe to (re)create it
        if age < LOCK_STALE_SECONDS:
            return False
        try:
            os.remove(self.path)
        except OSError:
            return False
        return True

    def __exit__(self, exc_type, exc, tb) -> None:
        try:
            os.remove(self.path)
        except OSError:
            pass
