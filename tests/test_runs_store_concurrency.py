"""Concurrency contract tests for clozn/runs/store.py::_ensure() (data-integrity fix).

Background: `_ensure()` is called cheaply on EVERY store entry point and, before this fix, had NO
process- or thread-level protection around its one-time WAL-mode switch (`_connect()`'s
`PRAGMA journal_mode=WAL`) plus schema migration. RT-05's global `POST_GATE` in clozn/server/app.py used
to accidentally mask this by serializing every POST; RT-05 replaces that gate with per-worker gating,
which makes the race reachable in production, and RT-05's own fix (warming the store once before the
server accepts connections) only covers the server's own boot path -- a second entry point (the CLI, a
test harness, a concurrent `clozn` invocation against the same `~/.clozn/runs/runs.sqlite3`) can still
race `_ensure()` directly.

Root cause, isolated empirically (ad hoc reproducer, not part of this file, run against UNMODIFIED
pre-fix code): `PRAGMA journal_mode=WAL` is a genuine, one-time on-disk file-format conversion the first
time it runs against a non-WAL file. SQLite requires exclusive access to perform that conversion, and --
unlike ordinary write-lock contention -- it does NOT retry through the busy handler / `busy_timeout` if
another connection holds so much as a pending write transaction. Every observed failure's traceback
pointed at exactly this line, in `_connect()`. Measured failure rate: 20 threads, barrier-synchronized,
racing a FRESH store, 30 trials (600 total `_ensure()` calls) -> 30-60 failures (5-12%), every one
`sqlite3.OperationalError: database is locked`. A companion cross-process reproducer (real OS processes,
not threads) also reproduced it, at a lower rate given a busy-wait synchronization primitive is looser
than an in-process Barrier (120 calls -> 2 failures, 1.7%), confirming the race is not merely a
same-process/GIL artifact.

The fix (`_ensure_schema_locked` in store.py) serializes the WAL switch + `migrations.migrate()` behind a
SEPARATE lock-only SQLite file that is deliberately never itself switched to WAL, plus a per-process
short-circuit cache once a given db path is confirmed current. See store.py's own docstrings for the
full design rationale and the additional isolated measurements (0/300 ordinary BEGIN IMMEDIATE
contention failures, 0/150 WAL-pragma-reissue-on-an-already-WAL-file failures under concurrent write
load, 0/500 dedicated-lock-primitive failures) that ruled out a bare `threading.Lock` and pinned down
exactly what needed protecting.

These tests exercise the REAL store.py/migrations.py code end-to-end (no mocking) and are expected to
run at 0 failures against the fixed code; reverting the fix in `_ensure_schema_locked`/`_ensure` (e.g.
`git stash` this file's sibling changes to store.py) reproduces failures in the thread-based tests below
reliably.
"""
from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clozn.runs.migrations as migrations  # noqa: E402
import clozn.runs.store as store            # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Redirect the run store to a temp dir and clear the per-process schema-verified cache, so this
    test's fresh, never-touched store actually exercises the slow (lock-protected) init path instead of
    short-circuiting on a cache entry a PRIOR test happened to leave behind for a colliding path."""
    runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(store, "RUNS_DIR", runs_dir)
    store._schema_verified.clear()
    yield runs_dir
    store._schema_verified.clear()


def _assert_schema_current(db_path: str) -> None:
    """Not merely 'no exception' -- the schema must actually be fully, correctly migrated."""
    db = sqlite3.connect(db_path)
    try:
        assert migrations.current_version(db) == migrations.TARGET_VERSION
        assert migrations.pending(db) == []
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"schema_meta", "runs", "pinned_checkpoints"}.issubset(tables)
    finally:
        db.close()


def _run_thread_race(n_threads: int) -> list[str]:
    barrier = threading.Barrier(n_threads)
    results: list[str] = [None] * n_threads

    def worker(i):
        barrier.wait()
        try:
            store._ensure()
            results[i] = "OK"
        except BaseException as exc:  # noqa: BLE001 -- we want to see literally everything
            results[i] = f"{type(exc).__name__}: {exc}"

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


# ============================================================================== cross-thread concurrency

def test_ensure_survives_concurrent_threads_on_fresh_store(isolated):
    """25 threads, released simultaneously via a Barrier, all call _ensure() against a store that has
    NEVER been touched. Every call must succeed, and the resulting schema must be fully migrated."""
    results = _run_thread_race(25)
    failures = [r for r in results if r != "OK"]
    assert failures == [], f"{len(failures)}/{len(results)} _ensure() calls failed: {failures[:5]}"
    _assert_schema_current(store._db_path())


def test_ensure_concurrent_threads_do_not_corrupt_the_migration_ledger(isolated):
    """Beyond 'no exception': the migration ledger must show each migration applied EXACTLY once, never
    partially and never twice -- a lost race that silently reapplied or skipped a step would pass a bare
    no-exception check but corrupt the ledger."""
    results = _run_thread_race(20)
    assert [r for r in results if r != "OK"] == []

    db = sqlite3.connect(store._db_path())
    try:
        rows = db.execute("SELECT key FROM schema_meta WHERE key LIKE 'migration:%'").fetchall()
        versions = sorted(int(k[0].split(":")[1]) for k in rows)
        assert versions == [migration.version for migration in migrations.MIGRATIONS], versions
    finally:
        db.close()


def test_ensure_concurrency_flake_hunt(isolated, tmp_path):
    """25 trials x 20 threads = 500 `_ensure()` calls against 25 SEPARATE fresh stores in one assertion --
    a single green race proves nothing (the pre-fix rate above was only 5-12%; most individual small
    trials passed clean even on unmodified code). RT-05's own concurrency test ran 175 executions for
    itself; this exceeds that bar in-process. A separate, real cross-process flake-hunt (30 trials x 20
    threads = 600 executions, plus a 15 trials x 12 real-OS-process run = 180 executions) was run
    standalone during development; see the commit body for those numbers."""
    n_trials = 25
    n_threads = 20
    total = 0
    failures: list[str] = []
    for trial in range(n_trials):
        store.RUNS_DIR = str(tmp_path / f"trial_{trial}" / "runs")
        store._schema_verified.clear()
        results = _run_thread_race(n_threads)
        total += len(results)
        failures.extend(r for r in results if r != "OK")

    assert failures == [], (
        f"{len(failures)}/{total} calls failed across {n_trials} trials: {failures[:5]}"
    )


# ================================================================================= cross-process concurrency

_WORKER_SCRIPT = """
import sys
import time

sys.path.insert(0, sys.argv[1])
import clozn.runs.store as store

store.RUNS_DIR = sys.argv[2]
start_at = float(sys.argv[3])
while time.time() < start_at:
    pass
try:
    store._ensure()
    print("OK")
except BaseException as exc:  # noqa: BLE001
    print(f"FAIL:{type(exc).__name__}:{exc}")
"""


def test_ensure_survives_concurrent_processes_on_fresh_store(tmp_path):
    """Real OS processes (not threads, not multiprocessing.Pool -- which pickles test-module state in a
    way that's fragile under pytest + Windows 'spawn') racing a fresh store, synchronized via a shared
    target wall-clock timestamp. Proves the fix is cross-PROCESS safe, not merely cross-thread -- a bare
    `threading.Lock` would pass every test above but NOT this one."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runs_dir = str(tmp_path / "runs")
    script_path = tmp_path / "_ensure_race_worker.py"
    script_path.write_text(_WORKER_SCRIPT, encoding="utf-8")

    n = 10
    start_at = time.time() + 2.0  # generous head start covers process-spawn/import latency
    procs = [
        subprocess.Popen(
            [sys.executable, str(script_path), repo_root, runs_dir, str(start_at)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        for _ in range(n)
    ]
    outputs = [p.communicate(timeout=30) for p in procs]
    results = [out.strip() for out, _err in outputs]
    errs = [err for _out, err in outputs if err.strip()]

    failures = [r for r in results if r != "OK"]
    assert failures == [], (
        f"{len(failures)}/{n} cross-process _ensure() calls failed: {failures}\nstderr: {errs}"
    )
    _assert_schema_current(os.path.join(runs_dir, "runs.sqlite3"))


# ========================================================================================= typed lock error

def test_schema_lock_timeout_is_typed_not_silent(isolated, monkeypatch):
    """If the cross-process lock cannot be acquired within the timeout, _ensure() must raise the typed
    SchemaLockTimeout -- never silently proceed against a possibly half-initialized schema, and never
    hang past the configured timeout. Simulated by holding the lock file open in an uncommitted
    transaction from this same test process."""
    monkeypatch.setattr(store, "_SCHEMA_LOCK_TIMEOUT_S", 0.3)
    os.makedirs(store.RUNS_DIR, exist_ok=True)
    blocker = sqlite3.connect(store._schema_lock_path(), timeout=30.0)
    blocker.isolation_level = None
    blocker.execute("BEGIN IMMEDIATE")
    try:
        started = time.monotonic()
        with pytest.raises(store.SchemaLockTimeout):
            store._ensure()
        elapsed = time.monotonic() - started
        assert elapsed < 5.0, f"took {elapsed:.1f}s to time out -- not honoring the shortened timeout"
    finally:
        blocker.execute("COMMIT")
        blocker.close()

    # the lock is released now -- a subsequent _ensure() must succeed normally, proving the timed-out
    # attempt above left nothing wedged or half-applied.
    store._ensure()
    _assert_schema_current(store._db_path())


# ================================================================================== idempotency (sanity)

def test_ensure_is_idempotent_after_concurrent_init(isolated):
    """Once the race has settled, repeated sequential _ensure() calls (the normal per-request path) must
    remain cheap no-ops that never re-touch migrations."""
    _run_thread_race(10)
    _assert_schema_current(store._db_path())
    for _ in range(5):
        store._ensure()          # must not raise, must not reapply anything
    _assert_schema_current(store._db_path())
