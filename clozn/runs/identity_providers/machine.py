"""clozn/runs/identity_providers/machine.py -- the machine/runtime identity facet performance diagnosis
owns. clozn/runs/identity_ext.py's own module docstring names "machine/runtime identity by performance
diagnosis" as one of the facets this seam exists for, and clozn/schemas/defs/clozn.run-identity.v1.json's
`ext` description literally lists 'machine' as its worked example namespace -- this file is that facet.

Lands at identity["ext"]["machine"], which is also what clozn.runs.perf_trace.build_trace() reads for the
clozn.performance-trace.v1 artifact's `machine_identity` block (see that module).

CACHING IS A HARD REQUIREMENT, NOT A NICETY
----------------------------------------------
identity(context) runs on the path that records EVERY real run -- clozn/runs/identity_ext.py's own
docstring: "keep them cheap ... never make a network call." clozn.cli.commands.models._detect_vram_gb()
shells out to `nvidia-smi` on every call, which is fine for `clozn plan` (one human-triggered invocation)
but would be a real regression here: an uncached subprocess spawn on every chat turn taxes the one hot
path this entire seam exists to protect. This module probes ONCE per process and caches the result
(including a negative result -- "no nvidia-smi found" is itself worth not re-discovering every request)
for the rest of the process's life, mirroring clozn.runs.identity's own model_hashes.json cache
discipline for the identical reason: a fact that does not change mid-process should only be measured
once.
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
