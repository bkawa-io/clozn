"""process_guard -- OS-level parent-death guard for spawned child processes (gateway + worker).

Stage 0 of ``docs/design/008-single-process-runtime.md`` (ADR 008 section 3.3). Today there is NO
parent-death guard of any kind: kill ``clozn serve``'s supervisor process ungracefully (SIGKILL, a
crash, ``taskkill /F``, an IDE stop button) and every child it spawned -- the gateway subprocess and
every C++ worker -- orphans. A worker holds GPU VRAM and binds a random loopback port reachable only
by the gateway, so an orphaned worker is simultaneously invisible and useless; on a 16GB box that is
stranded VRAM with nothing left alive to reclaim it. This has happened four separate times on this
project (see ``docs/HANDOFF_2026-07-30.md``'s process-lessons section).

This module is exclusively about that UNGRACEFUL case. The graceful path -- ``RuntimeStack.stop()``
calling ``worker_registry.stop_all()`` / terminating the gateway from ``cmd_serve``'s ``finally``
(``clozn/cli/runtime_process.py``, ``clozn/cli/commands/serve.py``) -- already works today and is
untouched by anything here.

Public API. Both call sites (the worker spawn in ``clozn/cli/engine_process.py::spawn_engine`` and the
two gateway spawns in ``clozn/cli/runtime_process.py``) use the same two functions, in this order::

    proc = subprocess.Popen(args, ..., **process_guard.subprocess_kwargs())   # BEFORE spawn (Linux hook)
    process_guard.guard(proc)                                                  # AFTER spawn (Windows hook)

Platform coverage (see ADR 008 section 3.3 for the full reasoning):

  * Linux   -- ``prctl(PR_SET_PDEATHSIG)`` via ``preexec_fn``. Real, OS-enforced: the kernel signals the
              child the moment the thread that forked it exits, for any reason.
  * Windows -- a Job Object with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE``. Real, OS-enforced: the OS kills
              every assigned process the instant the job's last handle closes, which happens
              automatically when this process exits for any reason (the handle is never closed
              deliberately -- see ``_JOB_HANDLE``'s docstring below).
  * Everything else (macOS, *BSD, ...) -- genuinely NOT guarded. There is no kernel primitive
              equivalent to either of the above; the only real fix is a ``getppid()`` poll (or a
              heartbeat) inside the worker's own C++ code, which is out of scope for this module (a
              pure-Python, zero-``engine/core``-change guard) and is reported honestly rather than
              faked.

Every public function here degrades instead of raising: an install failure on any platform logs once
(best-effort, to stderr) and continues with an unguarded child. A runtime that refuses to start because
a defensive process-hygiene mechanism threw an exception is a strictly worse outcome than one whose
children might leak on an abnormal kill -- that trade is the entire point of this module existing as a
side channel around ``subprocess.Popen``, never a gate in front of it.
"""
from __future__ import annotations

import ctypes
import sys

_WARNED: set = set()


def _warn_once(key: str, message: str) -> None:
    """Emit a one-time, best-effort diagnostic. Never raises -- a broken stderr must not break spawning."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    try:
        print(f"clozn: {message}", file=sys.stderr)
    except Exception:
        pass


# ------------------------------------------------------------------------------------------------ linux

_LIBC = None
_LIBC_LOAD_ATTEMPTED = False
_PR_SET_PDEATHSIG = 1
# SIGKILL, hardcoded as the POSIX signal number 9 rather than `signal.SIGKILL` at module scope: this
# module is imported unconditionally on every platform (including Windows, where `signal.SIGKILL` does
# not exist), and this constant is only ever handed to a syscall on Linux. SIGKILL specifically (not
# SIGTERM) because the point is a worker holding VRAM that MUST go away -- a signal the worker could
# catch, ignore, or hang while handling defeats the guard's whole purpose. Matches ADR 008 section 3.3.
_LINUX_PDEATHSIG = 9


def _load_libc():
    """Resolve libc exactly once, IN THE PARENT process, well before any fork happens.

    ``_linux_preexec`` (below) runs *inside* the freshly forked child, after ``fork()`` and before
    ``exec()`` -- at that point only the forking thread exists, so anything that could touch a lock
    held by a sibling thread at fork time (import machinery, ``dlopen``, memory allocation) can
    deadlock the child forever with no way to recover. Loading libc is exactly that kind of work, so it
    must already be done by the time the child needs it; the preexec function itself just makes one
    ctypes call against an already-resolved handle.
    """
    global _LIBC, _LIBC_LOAD_ATTEMPTED
    if _LIBC_LOAD_ATTEMPTED:
        return _LIBC
    _LIBC_LOAD_ATTEMPTED = True
    try:
        import ctypes.util

        name = ctypes.util.find_library("c") or "libc.so.6"
        _LIBC = ctypes.CDLL(name, use_errno=True)
    except Exception as exc:
        _LIBC = None
        _warn_once(
            "linux-libc-load-failed",
            f"could not load libc for PR_SET_PDEATHSIG ({exc!r}); this worker/gateway will NOT be "
            "guarded against an ungraceful parent death on this host.",
        )
    return _LIBC


def _linux_preexec() -> None:
    """Runs INSIDE the forked child, after ``fork()`` and before ``exec()`` (the POSIX ``preexec_fn``
    contract). Deliberately does nothing but one ctypes call against the ALREADY-loaded ``_LIBC`` --
    see :func:`_load_libc`'s docstring for why nothing heavier is safe here. Must never raise: an
    exception in ``preexec_fn`` surfaces to the parent as an opaque, hard-to-diagnose ``Popen`` failure
    with no useful traceback (Stage 0's hazard list), so every failure mode here is swallowed silently
    -- the child simply proceeds unguarded rather than never launching at all.
    """
    try:
        if _LIBC is not None:
            _LIBC.prctl(_PR_SET_PDEATHSIG, _LINUX_PDEATHSIG, 0, 0, 0)
    except Exception:
        pass


# ---------------------------------------------------------------------------------------------- windows

_JOB_LOCK = None  # lazily a threading.Lock(); see _job_lock()
# Process-lifetime reference to the ONE guard Job Object's raw HANDLE (a plain ctypes int, never a
# wrapper with a __del__/Close() finalizer). This MUST stay reachable for as long as this interpreter is
# alive: JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE fires the instant the job's LAST handle closes. A handle
# stashed only in a local that goes out of scope (or any wrapper object relying on garbage collection to
# release OS resources, e.g. `_winapi.Handle`) would close early and kill every assigned child
# IMMEDIATELY instead of on parent death -- see the module docstring and Stage 0's hazard list. Storing
# a bare int at module scope sidesteps that risk entirely: nothing about normal Python GC ever closes a
# plain integer, and the OS only reclaims the handle when this whole process exits (gracefully or not),
# which is exactly the lifetime the guard needs.
_JOB_HANDLE = None
_JOB_INIT_FAILED = False
_KERNEL32 = None

_JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000
_JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS = 9  # JobObjectExtendedLimitInformation


def _job_lock():
    global _JOB_LOCK
    if _JOB_LOCK is None:
        import threading

        _JOB_LOCK = threading.Lock()
    return _JOB_LOCK


def _win_kernel32():
    global _KERNEL32
    if _KERNEL32 is None:
        _KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)
    return _KERNEL32


def _win_extended_limit_info_type():
    """Build the ``JOBOBJECT_EXTENDED_LIMIT_INFORMATION`` ctypes struct lazily, inside a Windows-only
    function -- importing ``ctypes.wintypes`` at module scope is harmless on other platforms, but
    building this here keeps every Windows-specific symbol local to code that only ever runs on win32.
    """
    from ctypes import wintypes

    class _BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IO_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class _EXTENDED(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BASIC),
            ("IoInfo", _IO_COUNTERS),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    return _EXTENDED


def _create_job_object():
    """Create and configure ONE Job Object with ``KILL_ON_JOB_CLOSE``. Returns a raw handle (plain
    ctypes/py int) or ``None`` on any failure -- never raises (callers still wrap this, belt and
    suspenders, since Stage 0's hazard list calls out install failures explicitly).
    """
    kernel32 = _win_kernel32()
    kernel32.CreateJobObjectW.restype = ctypes.c_void_p
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return None

    info_type = _win_extended_limit_info_type()
    info = info_type()
    info.BasicLimitInformation.LimitFlags = _JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE

    kernel32.SetInformationJobObject.restype = ctypes.c_int
    kernel32.SetInformationJobObject.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32,
    ]
    ok = kernel32.SetInformationJobObject(
        job, _JOBOBJECT_EXTENDED_LIMIT_INFORMATION_CLASS, ctypes.byref(info), ctypes.sizeof(info)
    )
    if not ok:
        kernel32.CloseHandle.restype = ctypes.c_int
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle(job)
        return None
    return job


def _get_or_create_job():
    """Lazily create the one guard Job Object for this process's lifetime (see ``_JOB_HANDLE``'s
    docstring for why it is cached at module scope rather than returned fresh each call). Nested jobs
    (CI runners / IDEs that already run inside a job object) need Windows 8+ for
    ``AssignProcessToJobObject`` to succeed against an already-jobbed process; that is an OS version
    floor this repo already assumes elsewhere, but assignment can still legitimately fail on some hosts
    (a restrictive pre-existing job) -- that failure is handled where assignment happens, not here.
    """
    global _JOB_HANDLE, _JOB_INIT_FAILED
    if _JOB_HANDLE is not None:
        return _JOB_HANDLE
    if _JOB_INIT_FAILED:
        return None
    with _job_lock():
        if _JOB_HANDLE is not None:
            return _JOB_HANDLE
        if _JOB_INIT_FAILED:
            return None
        try:
            job = _create_job_object()
        except Exception:
            job = None
        if job is None:
            _JOB_INIT_FAILED = True
            _warn_once(
                "windows-job-create-failed",
                "could not create/configure a parent-death guard Job Object "
                "(CreateJobObjectW/SetInformationJobObject failed); children will NOT be guarded "
                "against an ungraceful parent death on this host.",
            )
            return None
        _JOB_HANDLE = job
        return job


def _windows_guard(proc) -> bool:
    """Assign an already-spawned child to the guard Job Object. Best-effort only: any failure (job
    creation, a missing/invalid handle, a nested-job assignment refusal) degrades to an unguarded
    child and returns ``False`` -- never raises.
    """
    job = _get_or_create_job()
    if not job:
        return False

    # `Popen._handle` is semi-private -- there is no public `subprocess` API that exposes a raw Windows
    # process HANDLE, and every stdlib-only Job Object recipe (this one included) relies on the same
    # attribute (ADR 008 section 3.3 names this as a known, small, accepted risk; a `pywin32` dependency
    # would avoid it but this project has never taken one -- see `pyproject.toml`'s stdlib-only rule).
    # `getattr` with a default so a future CPython that renames/removes it degrades instead of raising.
    #
    # Deliberately NOT falling back to re-opening the process by PID (`OpenProcess(pid)`) when
    # `_handle` is unavailable: a PID can be reused by an unrelated process between spawn and this call
    # (vanishingly unlikely for a process this code just spawned itself, but not zero), and in tests
    # that monkeypatch `subprocess.Popen` with a fake process object carrying a fabricated `pid`, a
    # PID-based fallback could coincidentally collide with a REAL, unrelated live process on the host
    # and assign it into a job that kills it when this interpreter exits. Degrading to unguarded is
    # strictly safer than that failure mode.
    handle = getattr(proc, "_handle", None)
    if not handle:
        return False
    try:
        h = int(handle)
    except Exception:
        return False
    if not h:
        return False

    kernel32 = _win_kernel32()
    kernel32.AssignProcessToJobObject.restype = ctypes.c_int
    kernel32.AssignProcessToJobObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    try:
        ok = kernel32.AssignProcessToJobObject(job, h)
    except Exception:
        ok = 0
    if not ok:
        _warn_once(
            "windows-assign-failed",
            "could not assign one or more child processes to the parent-death guard Job Object "
            "(AssignProcessToJobObject failed -- possibly a restrictive pre-existing job on this "
            "host); those children will NOT be guarded against an ungraceful parent death.",
        )
        return False
    return True


# ----------------------------------------------------------------------------------------------- public

def subprocess_kwargs() -> dict:
    """Extra kwargs to merge into a ``subprocess.Popen(...)`` call, for platforms that need pre-spawn
    wiring (Linux's ``preexec_fn``). Call this BEFORE ``Popen`` -- unlike :func:`guard`, this cannot be
    applied after the fact. Returns ``{}`` (safely, doing nothing) on every other platform. Never
    raises.
    """
    try:
        if sys.platform.startswith("linux") and _load_libc() is not None:
            return {"preexec_fn": _linux_preexec}
    except Exception:
        pass
    return {}


def guard(proc) -> bool:
    """Best-effort, POST-spawn half of the parent-death guard. Call immediately after
    ``subprocess.Popen(...)`` returns, on both the gateway and every worker.

    Returns ``True`` iff a real, OS-enforced guard is now believed to be protecting this child;
    ``False`` means the child is unguarded (unsupported platform, or an install failure). Either way
    the child process itself is completely unaffected by this call, and this function NEVER raises --
    a spawn must succeed even when this defensive mechanism cannot be installed.
    """
    try:
        if sys.platform == "win32":
            return _windows_guard(proc)
        if sys.platform.startswith("linux"):
            # The actual guard (prctl) was installed inside the child via the `preexec_fn` that
            # `subprocess_kwargs()` contributed to the `Popen(...)` call that already happened; there
            # is nothing left to do post-spawn. Report whether libc was even available to attempt it --
            # this process cannot observe from the parent side whether the syscall inside the (now
            # separate) child actually succeeded.
            return _LIBC is not None
        _warn_once(
            "platform-unguarded",
            f"no parent-death guard exists on this platform ({sys.platform!r}) -- there is no kernel "
            "primitive equivalent to Linux's PR_SET_PDEATHSIG or a Windows Job Object here (see "
            "docs/design/008-single-process-runtime.md section 3.3). If this process is killed "
            "ungracefully (SIGKILL, a crash), its children will orphan and must be reaped manually "
            "(`clozn stop`, or a process manager).",
        )
        return False
    except Exception as exc:
        _warn_once(
            "guard-crashed",
            f"parent-death guard raised unexpectedly (pid={getattr(proc, 'pid', '?')}): {exc!r}; "
            "continuing WITHOUT a guard for this child rather than fail the spawn.",
        )
        return False
