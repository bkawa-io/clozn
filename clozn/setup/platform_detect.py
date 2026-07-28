"""detect_platform() -- best-effort, model-free, network-free description of the running machine.

Used by clozn/setup/manifest.py's select_artifact() to pick the narrowest compatible engine artifact.
Every field beyond os/arch is a HINT, never a hard requirement: a probe that fails (no nvidia-smi on
PATH, an unrecognized `platform.machine()` string) returns None for that field rather than raising or
guessing, so a caller can always fall back to an explicit `--backend` choice. Nothing here downloads,
imports Torch, or touches the network -- see clozn/setup/transport.py for the one module in this
package that is allowed to open a URL.
"""
from __future__ import annotations

import platform as _platform
import re
import shutil
import subprocess

_OS_MAP = {"windows": "windows", "darwin": "macos", "linux": "linux"}
_ARCH_MAP = {
    "amd64": "x86_64", "x86_64": "x86_64", "x64": "x86_64",
    "arm64": "arm64", "aarch64": "arm64",
}

_NVIDIA_SMI_TIMEOUT_S = 3.0
_CUDA_VERSION_RE = re.compile(r"CUDA Version:\s*([0-9]+)\.")


def detect_os() -> str:
    """One of 'windows' | 'linux' | 'macos', or platform.system()'s own lowercase name for anything else
    (an honest passthrough -- selection simply finds no matching artifact rather than this function
    inventing a supported-looking value)."""
    return _OS_MAP.get(_platform.system().lower(), _platform.system().lower())


def detect_arch() -> str:
    """One of 'x86_64' | 'arm64', or platform.machine()'s own lowercase name for anything else."""
    return _ARCH_MAP.get(_platform.machine().lower(), _platform.machine().lower())


def _probe_nvidia_cuda_major() -> "int | None":
    """The CUDA major version an installed NVIDIA driver advertises via `nvidia-smi`'s own header line
    ("CUDA Version: 12.6"), or None on ANY failure: no nvidia-smi on PATH, it exits non-zero, it times
    out, or its output doesn't parse. This is the driver's MAXIMUM supported CUDA runtime, not proof an
    artifact built against exactly that major will run -- clozn/setup/manifest.py's selection treats it
    as a hint (prefer an exact match, otherwise the highest artifact major <= this), not a guarantee.
    Never raises; a probe that can't run is exactly as informative as no GPU at all."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        result = subprocess.run(
            [exe], capture_output=True, text=True, timeout=_NVIDIA_SMI_TIMEOUT_S,
        )
    except Exception:
        return None
    match = _CUDA_VERSION_RE.search(result.stdout or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def detect_platform(*, probe_gpu: bool = True) -> dict:
    """{'os', 'arch', 'gpu_backend', 'cuda_major'}. `gpu_backend` is None | 'cuda' | 'metal';
    `cuda_major` is only ever set alongside gpu_backend == 'cuda'.

    `probe_gpu=False` skips the nvidia-smi subprocess call entirely (gpu_backend/cuda_major both None) --
    the test suite always passes this, or better, calls select_artifact() directly with a synthetic
    platform dict, since a subprocess probe result is inherently machine-dependent and cannot be a
    meaningful assertion in CI. macOS/arm64 is reported as 'metal' with no probe: every Apple Silicon Mac
    clozn's v1 support matrix targets ships a Metal-capable GPU, so there is nothing to detect."""
    osname = detect_os()
    arch = detect_arch()
    gpu_backend = None
    cuda_major = None
    if osname == "macos" and arch == "arm64":
        gpu_backend = "metal"
    elif probe_gpu:
        cuda_major = _probe_nvidia_cuda_major()
        if cuda_major is not None:
            gpu_backend = "cuda"
    return {"os": osname, "arch": arch, "gpu_backend": gpu_backend, "cuda_major": cuda_major}
