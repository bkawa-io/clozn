"""Stage 0 of ADR 008 (docs/design/008-single-process-runtime.md): the parent-death guard.

The graceful shutdown path -- ``RuntimeStack.stop()`` calling ``worker_registry.stop_all()`` /
terminating the gateway from ``cmd_serve``'s ``finally`` -- already works today and is proven by
``tests/test_runtime_architecture.py``. A test that only exercises THAT path proves nothing about this
guard, because the guard exists exclusively for the UNGRACEFUL case: SIGKILL, a crash, ``taskkill /F``,
an IDE stop button, the supervisor itself being reaped. Every live test below kills a process externally
and ungracefully -- never ``.stop()``, never a signal the victim could catch -- and checks what the OS
actually did.

Three tiers of coverage:

  1. ``ProcessGuardUnitTests`` -- pure-logic tests of ``clozn.cli.process_guard`` itself (unsupported
     platform reports honestly; an install failure degrades instead of raising). Safe and meaningful on
     every platform/CI lane; no live subprocess beyond one trivial smoke check.
  2. ``WindowsParentDeathGuardLiveTests`` -- the real end-to-end proof on Windows (the primary dev box):
     spawn a throwaway "supervisor" subprocess that spawns a guarded worker child, ``taskkill /F`` the
     supervisor (not its tree -- only the guard, never taskkill's own cascade, may be what kills the
     worker), and assert the OS reaped the worker. Includes a negative control (no guard) proving the
     harness actually detects orphaning, and a same-process check that the guard does NOT kill children
     during normal operation (the "job handle held only in a local" bug from ADR 008's hazard list).
  3. ``LinuxParentDeathGuardLiveTests`` -- the identical shape using SIGKILL + PR_SET_PDEATHSIG. Skipped
     everywhere except a real Linux host (runs for real on ``ubuntu-latest`` in ``ci.yml``'s ``python``
     job); the skip states why rather than silently doing nothing.
"""
from __future__ import annotations

import gc
import os
import subprocess
import sys
import time
import unittest

from clozn.cli import process_guard

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tasklist_has_pid(pid: int) -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
        capture_output=True, text=True, timeout=5,
    )
    return str(pid) in out.stdout


def _posix_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class ProcessGuardUnitTests(unittest.TestCase):
    """Pure-logic tests: safe and meaningful on every platform, every CI lane."""

    def setUp(self):
        process_guard._WARNED.clear()

    def test_unsupported_platform_reports_honestly_and_degrades(self):
        """macOS (or any platform with no kernel primitive) must return False, never claim success."""
        class _FakeProc:
            pid = 999999

        real_platform = sys.platform
        process_guard.sys.platform = "darwin"
        try:
            result = process_guard.guard(_FakeProc())
        finally:
            process_guard.sys.platform = real_platform
        self.assertFalse(result, "an unsupported platform must never report a guard as installed")

    def test_unsupported_platform_warns_once_not_never_and_not_repeatedly(self):
        class _FakeProc:
            pid = 999999

        real_platform = sys.platform
        real_stderr = sys.stderr
        import io

        captured = io.StringIO()
        process_guard.sys.platform = "darwin"
        sys.stderr = captured
        try:
            process_guard.guard(_FakeProc())
            process_guard.guard(_FakeProc())
            process_guard.guard(_FakeProc())
        finally:
            process_guard.sys.platform = real_platform
            sys.stderr = real_stderr
        output = captured.getvalue()
        self.assertIn("no parent-death guard exists", output, "an honest report must be printed")
        self.assertEqual(
            output.count("no parent-death guard exists"), 1,
            "the warning must be emitted once, not once per call (would spam logs on every spawn)",
        )

    def test_windows_job_object_creation_failure_degrades_not_raises(self):
        if sys.platform != "win32":
            self.skipTest("Windows-only code path")

        class _FakeProc:
            pid = 999999

        original_create = process_guard._create_job_object
        original_handle = process_guard._JOB_HANDLE
        original_failed = process_guard._JOB_INIT_FAILED
        process_guard._create_job_object = lambda: (_ for _ in ()).throw(OSError("simulated"))
        # Force a re-attempt for THIS test only -- an earlier test in this file may already have
        # created a real job and cached it. Restored to the exact prior values below, not hardcoded
        # back to None, so a real handle another test is relying on isn't dropped uncontrolled.
        process_guard._JOB_HANDLE = None
        process_guard._JOB_INIT_FAILED = False
        try:
            result = process_guard.guard(_FakeProc())
        finally:
            process_guard._create_job_object = original_create
            process_guard._JOB_HANDLE = original_handle
            process_guard._JOB_INIT_FAILED = original_failed
        self.assertFalse(result, "a Job Object creation failure must degrade to unguarded, never raise")

    def test_windows_guard_handles_missing_popen_handle_attribute(self):
        """Hazard 3: `Popen._handle` is semi-private; a future CPython removing it must degrade."""
        if sys.platform != "win32":
            self.skipTest("Windows-only code path")

        class _FakeProcNoHandle:
            pid = 999999
            # deliberately no _handle attribute at all

        result = process_guard.guard(_FakeProcNoHandle())
        self.assertFalse(result, "a missing _handle must degrade to unguarded, never raise")

    def test_linux_libc_load_failure_degrades_not_raises(self):
        """Simulated on whichever host runs this: proves the DEGRADE PATH itself, independent of the
        live end-to-end proof in LinuxParentDeathGuardLiveTests below (which needs a real Linux host)."""
        real_platform = sys.platform
        process_guard.sys.platform = "linux"
        original_attempted = process_guard._LIBC_LOAD_ATTEMPTED
        original_libc = process_guard._LIBC
        process_guard._LIBC_LOAD_ATTEMPTED = True
        process_guard._LIBC = None  # simulate "we already tried, and it failed"
        try:
            kwargs = process_guard.subprocess_kwargs()
            result = process_guard.guard(object())
        finally:
            process_guard.sys.platform = real_platform
            process_guard._LIBC_LOAD_ATTEMPTED = original_attempted
            process_guard._LIBC = original_libc
        self.assertEqual(kwargs, {}, "no libc means no preexec_fn -- must not hand Popen a broken hook")
        self.assertFalse(result, "guard() must report False when libc never loaded")

    def test_subprocess_kwargs_and_guard_never_raise_on_this_real_host(self):
        """Smoke check on the ACTUAL host platform: both halves must survive a real spawn/guard cycle
        without raising, whatever the outcome. Not a proof of protection by itself -- see the live
        per-platform test classes below for that."""
        proc = subprocess.Popen(
            [sys.executable, "-c", "pass"],
            **process_guard.subprocess_kwargs(),
        )
        try:
            process_guard.guard(proc)  # must not raise regardless of platform
        finally:
            proc.wait(timeout=5)


@unittest.skipUnless(
    sys.platform == "win32",
    "the Job Object guard is Windows-only; this is the real end-to-end proof on the primary dev box",
)
class WindowsParentDeathGuardLiveTests(unittest.TestCase):
    """Spawns a real 'supervisor' process, kills it ungracefully, checks what the OS actually did."""

    _GUARDED_SUPERVISOR = (
        "import subprocess, sys, time\n"
        f"sys.path.insert(0, {REPO_ROOT!r})\n"
        "from clozn.cli import process_guard\n"
        "worker = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(120)'],\n"
        "    **process_guard.subprocess_kwargs(),\n"
        ")\n"
        "ok = process_guard.guard(worker)\n"
        "print(worker.pid, int(bool(ok)), flush=True)\n"
        "time.sleep(120)\n"
    )

    _UNGUARDED_SUPERVISOR = (
        "import subprocess, sys, time\n"
        "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "print(worker.pid, 0, flush=True)\n"
        "time.sleep(120)\n"
    )

    def _spawn_supervisor(self, script: str):
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        line = proc.stdout.readline()
        if not line.strip():
            remainder = proc.stdout.read()
            proc.wait(timeout=5)
            self.fail(f"supervisor produced no output (exit={proc.returncode}): {remainder!r}")
        parts = line.strip().split()
        self.assertEqual(len(parts), 2, f"malformed supervisor output: {line!r}")
        worker_pid = int(parts[0])
        guard_installed = parts[1] == "1"
        return proc, worker_pid, guard_installed

    @staticmethod
    def _kill_ungracefully(pid: int) -> None:
        # /F, no /T: an external, forced kill of ONLY this pid -- no tree-kill, so any subsequent
        # cleanup must come from the OS-level guard, never from taskkill doing the guard's job for it.
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, timeout=10)

    @staticmethod
    def _wait_gone(pid: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _tasklist_has_pid(pid):
                return True
            time.sleep(0.25)
        return not _tasklist_has_pid(pid)

    def test_guarded_worker_dies_when_supervisor_is_killed_ungracefully(self):
        supervisor, worker_pid, guard_installed = self._spawn_supervisor(self._GUARDED_SUPERVISOR)
        try:
            self.assertTrue(guard_installed,
                             "process_guard.guard() reported the guard was NOT installed on this "
                             "host (e.g. AssignProcessToJobObject failed) -- this test cannot prove "
                             "anything about kill behavior without a real guard in place")
            self.assertTrue(_tasklist_has_pid(worker_pid), "worker never started")
            self._kill_ungracefully(supervisor.pid)
            gone = self._wait_gone(worker_pid, timeout=10.0)
            self.assertTrue(
                gone,
                "guarded worker SURVIVED an ungraceful ('taskkill /F', no stack.stop()) supervisor "
                "kill -- the Job Object guard did not fire",
            )
        finally:
            supervisor.wait(timeout=5)
            if _tasklist_has_pid(worker_pid):
                subprocess.run(["taskkill", "/PID", str(worker_pid), "/F"], capture_output=True)

    def test_unguarded_worker_survives_supervisor_kill_negative_control(self):
        """Without the guard, today's known bug reproduces: the worker orphans and keeps running. This
        proves the harness genuinely detects orphaning -- the positive test above isn't passing because
        `taskkill` itself, or some other mechanism, cleans up regardless of the guard."""
        supervisor, worker_pid, _ = self._spawn_supervisor(self._UNGUARDED_SUPERVISOR)
        try:
            self.assertTrue(_tasklist_has_pid(worker_pid), "worker never started")
            self._kill_ungracefully(supervisor.pid)
            time.sleep(3.0)
            self.assertTrue(
                _tasklist_has_pid(worker_pid),
                "expected the UNGUARDED worker to orphan and keep running -- if it died anyway, "
                "something other than the guard is reaping children on this host, and the positive "
                "test above proves nothing",
            )
        finally:
            supervisor.wait(timeout=5)
            subprocess.run(["taskkill", "/PID", str(worker_pid), "/F"], capture_output=True)

    def test_guard_does_not_kill_child_during_normal_operation(self):
        """Hazard 1: a job handle held only in a local (rather than a process-lifetime reference) fires
        KILL_ON_JOB_CLOSE the instant that local is garbage-collected -- i.e. immediately, not on
        parent death. Prove children survive ordinary operation, including after a forced gc.collect()."""
        worker = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            **process_guard.subprocess_kwargs(),
        )
        try:
            installed = process_guard.guard(worker)
            self.assertTrue(installed, "guard did not install; the rest of this test would be vacuous")
            gc.collect()
            time.sleep(2.0)
            self.assertIsNone(
                worker.poll(),
                "the guard killed its own child during NORMAL operation -- the Job Object handle was "
                "likely garbage-collected instead of held for the process's lifetime",
            )
        finally:
            worker.terminate()
            worker.wait(timeout=5)


@unittest.skipUnless(
    sys.platform.startswith("linux"),
    "PR_SET_PDEATHSIG is Linux-only; this live-kill test needs a real POSIX host and runs for real on "
    "ubuntu-latest in ci.yml's `python` job -- skipped (not silently passed) everywhere else",
)
class LinuxParentDeathGuardLiveTests(unittest.TestCase):
    """The Linux mirror of the Windows live tests above: SIGKILL the supervisor, never a signal it
    could catch or a graceful stop, and check whether the OS actually reaped the worker."""

    _GUARDED_SUPERVISOR = (
        "import subprocess, sys, time\n"
        f"sys.path.insert(0, {REPO_ROOT!r})\n"
        "from clozn.cli import process_guard\n"
        "worker = subprocess.Popen(\n"
        "    [sys.executable, '-c', 'import time; time.sleep(120)'],\n"
        "    **process_guard.subprocess_kwargs(),\n"
        ")\n"
        "ok = process_guard.guard(worker)\n"
        "print(worker.pid, int(bool(ok)), flush=True)\n"
        "time.sleep(120)\n"
    )

    _UNGUARDED_SUPERVISOR = (
        "import subprocess, sys, time\n"
        "worker = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])\n"
        "print(worker.pid, 0, flush=True)\n"
        "time.sleep(120)\n"
    )

    def _spawn_supervisor(self, script: str):
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        line = proc.stdout.readline()
        if not line.strip():
            remainder = proc.stdout.read()
            proc.wait(timeout=5)
            self.fail(f"supervisor produced no output (exit={proc.returncode}): {remainder!r}")
        parts = line.strip().split()
        self.assertEqual(len(parts), 2, f"malformed supervisor output: {line!r}")
        worker_pid = int(parts[0])
        guard_installed = parts[1] == "1"
        return proc, worker_pid, guard_installed

    @staticmethod
    def _kill_ungracefully(pid: int) -> None:
        import signal

        os.kill(pid, signal.SIGKILL)  # uncatchable; the honest "ungraceful" case

    @staticmethod
    def _wait_gone(pid: int, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not _posix_pid_alive(pid):
                return True
            time.sleep(0.1)
        return not _posix_pid_alive(pid)

    def test_guarded_worker_dies_when_supervisor_is_sigkilled(self):
        supervisor, worker_pid, guard_installed = self._spawn_supervisor(self._GUARDED_SUPERVISOR)
        try:
            self.assertTrue(guard_installed,
                             "process_guard.guard() reported the guard was NOT installed (libc failed "
                             "to load?) -- this test cannot prove anything without a real guard")
            self.assertTrue(_posix_pid_alive(worker_pid), "worker never started")
            self._kill_ungracefully(supervisor.pid)
            gone = self._wait_gone(worker_pid, timeout=10.0)
            self.assertTrue(
                gone,
                "guarded worker SURVIVED a SIGKILL to its supervisor -- PR_SET_PDEATHSIG did not fire",
            )
        finally:
            try:
                supervisor.wait(timeout=5)
            except Exception:
                pass
            if _posix_pid_alive(worker_pid):
                import signal

                try:
                    os.kill(worker_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_unguarded_worker_survives_supervisor_sigkill_negative_control(self):
        supervisor, worker_pid, _ = self._spawn_supervisor(self._UNGUARDED_SUPERVISOR)
        try:
            self.assertTrue(_posix_pid_alive(worker_pid), "worker never started")
            self._kill_ungracefully(supervisor.pid)
            time.sleep(1.0)
            self.assertTrue(
                _posix_pid_alive(worker_pid),
                "expected the UNGUARDED worker to orphan (reparented to init) and keep running -- if "
                "it died anyway, the positive test above proves nothing on this host",
            )
        finally:
            try:
                supervisor.wait(timeout=5)
            except Exception:
                pass
            import signal

            try:
                os.kill(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_guard_does_not_kill_child_during_normal_operation(self):
        worker = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(5)"],
            **process_guard.subprocess_kwargs(),
        )
        try:
            gc.collect()
            time.sleep(2.0)
            self.assertIsNone(
                worker.poll(),
                "the guard killed its own child during NORMAL operation on Linux -- unexpected, since "
                "PDEATHSIG only fires on parent-thread exit, not on gc",
            )
        finally:
            worker.terminate()
            worker.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
