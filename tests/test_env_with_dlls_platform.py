"""_env_with_dlls()'s platform-aware shared-library env var (engine/core POSIX build support,
feat/platform-port-2026-07). Windows is this repo's primary, continuously-built dev platform (this box,
RTX 5080) -- a regression in its branch would break the most-exercised setup, so it is pinned
byte-for-byte here (same hardcoded CUDA v13.3 bin dirs, same PATH construction the function has always
used). Linux and macOS are unit-checked by simulating sys.platform / os.path.isdir; nothing here is
evidence of a real Linux/macOS engine build by itself -- see README.md's platform section for what
independent evidence actually exists (a nightly-green Linux CPU build+smoke in CI; a macOS run reported
by the repo owner but with no CI record found in this repository).
"""
from __future__ import annotations

import os
import sys

from clozn.cli import engine_process

CUDA_X64 = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64"
CUDA_BIN = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin"


def _no_dirs_exist(monkeypatch):
    monkeypatch.setattr(os.path, "isdir", lambda p: False)


# ------------------------------------------------------------------------------------- windows (pinned)
# These four pin the exact pre-existing Windows behavior: PATH, in this order, these two hardcoded CUDA
# dirs (only when they exist on disk), never touched for a CPU (gpu=False) launch.

def test_windows_cpu_uses_path_and_is_unchanged(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("PATH", r"C:\already\here")
    # `_env_with_dlls` copies os.environ, so the "not in env" assertions below only mean
    # "this branch did not ADD it" once anything inherited is cleared. GitHub's Linux runners
    # really do export LD_LIBRARY_PATH (/opt/hostedtoolcache/Python/*/x64/lib), which failed
    # this test on CI while passing on a Windows dev box that has neither variable set.
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    _no_dirs_exist(monkeypatch)

    env = engine_process._env_with_dlls([r"C:\dev\build-serve"], gpu=False)

    assert env["PATH"] == os.pathsep.join([r"C:\dev\build-serve", r"C:\already\here"])
    assert "LD_LIBRARY_PATH" not in env
    assert "DYLD_LIBRARY_PATH" not in env


def test_windows_gpu_appends_cuda_dirs_that_exist_in_the_documented_order(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("PATH", r"C:\already\here")
    monkeypatch.setattr(os.path, "isdir", lambda p: p in (CUDA_X64, CUDA_BIN))

    env = engine_process._env_with_dlls([r"C:\dev\build-gpu"], gpu=True)

    assert env["PATH"] == os.pathsep.join(
        [r"C:\dev\build-gpu", CUDA_X64, CUDA_BIN, r"C:\already\here"]
    )


def test_windows_gpu_skips_cuda_dirs_that_do_not_exist(monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("PATH", r"C:\already\here")
    _no_dirs_exist(monkeypatch)

    env = engine_process._env_with_dlls([r"C:\dev\build-gpu"], gpu=True)

    assert env["PATH"] == os.pathsep.join([r"C:\dev\build-gpu", r"C:\already\here"])


def test_windows_cpu_never_probes_the_cuda_dirs_at_all(monkeypatch):
    # gpu=False must never even ask whether the CUDA dirs exist -- probing (regardless of the answer)
    # would itself be a behavior change from the pre-existing function.
    probed = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(os.path, "isdir", lambda p: probed.append(p) or False)

    engine_process._env_with_dlls([r"C:\dev\build-serve"], gpu=False)

    assert probed == []


# --------------------------------------------------------------------------------------------- linux

def test_linux_uses_ld_library_path_and_leaves_path_untouched(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing/lib")
    monkeypatch.delenv("CUDA_HOME", raising=False)
    # Twin of the LD_LIBRARY_PATH problem in the Windows/macOS tests, latent rather than observed:
    # CI is Linux and never sets DYLD_LIBRARY_PATH, but a developer running this suite ON a Mac
    # very well might, and would then see this fail for a reason that has nothing to do with the
    # Linux branch it is testing. Cleared for the same reason, before anyone hits it.
    monkeypatch.delenv("DYLD_LIBRARY_PATH", raising=False)
    path_before = os.environ.get("PATH", "")

    env = engine_process._env_with_dlls(["/repo/engine/core/build-serve"], gpu=False)

    assert env["LD_LIBRARY_PATH"] == os.pathsep.join(
        ["/repo/engine/core/build-serve", "/existing/lib"]
    )
    assert env["PATH"] == path_before
    assert "DYLD_LIBRARY_PATH" not in env


def test_linux_gpu_appends_cuda_home_lib64_when_set_and_present(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    # os.path.join is the HOST interpreter's real path module (ntpath here, since sys.platform is only
    # monkeypatched, not the os module) -- os.path.join("/usr/local/cuda", "lib64") on a Windows host
    # yields a mixed-separator string, not the POSIX-joined one. Normalize before comparing so this test
    # exercises the CUDA_HOME branch itself rather than a Windows-vs-POSIX os.path.join artifact.
    monkeypatch.setattr(os.path, "isdir", lambda p: p.replace("\\", "/") == "/usr/local/cuda/lib64")

    env = engine_process._env_with_dlls(["/repo/engine/core/build-gpu"], gpu=True)

    parts = [p.replace("\\", "/") for p in env["LD_LIBRARY_PATH"].split(os.pathsep)]
    assert parts[0] == "/repo/engine/core/build-gpu"
    assert "/usr/local/cuda/lib64" in parts


def test_linux_gpu_skips_cuda_home_lib64_when_it_does_not_exist(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
    _no_dirs_exist(monkeypatch)

    env = engine_process._env_with_dlls(["/repo/engine/core/build-gpu"], gpu=True)

    assert "/usr/local/cuda/lib64" not in env["LD_LIBRARY_PATH"]


def test_linux_gpu_without_cuda_home_assumes_ld_library_path_already_has_it(monkeypatch):
    # No CUDA_HOME set: the function must not guess a path -- it leaves LD_LIBRARY_PATH to whatever a
    # system/conda CUDA install already put there (matches build_gpu.sh's nvcc-on-PATH assumption).
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/conda/env/lib")

    env = engine_process._env_with_dlls(["/repo/engine/core/build-gpu"], gpu=True)

    assert env["LD_LIBRARY_PATH"] == os.pathsep.join(
        ["/repo/engine/core/build-gpu", "/conda/env/lib"]
    )


def test_linux_never_touches_windows_cuda_paths(monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("LD_LIBRARY_PATH", "")
    monkeypatch.delenv("CUDA_HOME", raising=False)

    env = engine_process._env_with_dlls(["/repo/engine/core/build-gpu"], gpu=True)

    assert CUDA_X64 not in env["LD_LIBRARY_PATH"]
    assert CUDA_BIN not in env["LD_LIBRARY_PATH"]


# --------------------------------------------------------------------------------------------- macos

def test_macos_uses_dyld_library_path_and_leaves_path_untouched(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "/existing")
    # See the note in test_windows_cpu_uses_path_and_is_unchanged: the LD_LIBRARY_PATH assertion
    # below is about what this branch ADDS, and GitHub's Linux runners export it already.
    monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    path_before = os.environ.get("PATH", "")

    env = engine_process._env_with_dlls(["/repo/engine/core/build-serve"], gpu=False)

    assert env["DYLD_LIBRARY_PATH"] == os.pathsep.join(
        ["/repo/engine/core/build-serve", "/existing"]
    )
    assert env["PATH"] == path_before
    assert "LD_LIBRARY_PATH" not in env


def test_macos_gpu_never_consults_cuda_home_or_any_cuda_path(monkeypatch):
    # Metal has no CUDA-equivalent runtime search path (see build_gpu.sh: Metal on Darwin never sets
    # GGML_CUDA). gpu=True on darwin must add nothing beyond the dll dir itself.
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setenv("DYLD_LIBRARY_PATH", "")
    monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")

    env = engine_process._env_with_dlls(["/repo/engine/core/build-gpu"], gpu=True)

    assert env["DYLD_LIBRARY_PATH"] == "/repo/engine/core/build-gpu" + os.pathsep
    assert "/usr/local/cuda" not in env["DYLD_LIBRARY_PATH"]
