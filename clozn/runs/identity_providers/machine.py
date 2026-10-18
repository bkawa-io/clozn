"""Machine/runtime identity facts recorded with runs.

The provider lands at identity["ext"]["machine"] and records inexpensive process-global details that
help explain why otherwise identical local runs may differ.

CACHING IS A HARD REQUIREMENT, NOT A NICETY
----------------------------------------------
identity(context) runs on the path that records every real run. This module probes once per process
and caches the result (including a negative result), because these facts do not change mid-process.
"""
from __future__ import annotations

import os
import platform
import subprocess

NAME = "machine"

# None = not yet probed this process. {} (falsy but not None) is a valid, cached "nothing measurable"
# result -- distinct from "haven't looked yet" so a GPU-less machine doesn't retry nvidia-smi every run.
_CACHE: dict | None = None


def _probe_vram_gb() -> float | None:
    """Best-effort total VRAM via `nvidia-smi` -- a driver metadata query, not a CUDA context: it
    allocates nothing and runs no compute. None if nvidia-smi is missing, errors, or times out."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            return round(float(out.stdout.strip().splitlines()[0]) / 1024, 1)
    except Exception:
        pass
    return None


def _probe() -> dict:
    """Everything this process can honestly establish about the machine it's running on. Never raises;
    each fact is independently guarded so one failure (e.g. os.cpu_count() returning None on an odd
    platform) does not cost the others."""
    out: dict = {}
    try:
        system = platform.system()
        if system:
            out["os"] = system
    except Exception:
        pass
    try:
        release = platform.release()
        if release:
            out["os_release"] = release
    except Exception:
        pass
    try:
        machine = platform.machine()
        if machine:
            out["arch"] = machine
    except Exception:
        pass
    try:
        cpu_count = os.cpu_count()
        if cpu_count:
            out["cpu_count"] = cpu_count
    except Exception:
        pass
    vram_gb = _probe_vram_gb()
    if vram_gb is not None:
        out["gpu_vram_gb"] = vram_gb
    return out


def identity(context) -> dict:
    """The machine facet, probed once per process and cached (see module docstring) -- `context` is
    accepted for the seam's contract but unused: every fact here is process-global, not per-request."""
    global _CACHE
    if _CACHE is None:
        _CACHE = _probe()
    return dict(_CACHE) if _CACHE else {}


def reset_cache() -> None:
    """Drop the per-process probe cache. For tests only -- a real process never needs to re-probe."""
    global _CACHE
    _CACHE = None
