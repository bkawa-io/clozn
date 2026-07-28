"""detect_platform() is inherently machine-dependent for its GPU fields, so this file only asserts what
is true on ANY runner: os/arch resolve to a known label, and probe_gpu=False never touches nvidia-smi
(model-free/network-free per the baseline rules -- selection-logic tests in test_setup_manifest.py use
synthetic platform dicts instead of a real probe)."""
from __future__ import annotations

from clozn.setup import platform_detect as pd


def test_detect_os_is_a_known_label_on_this_runner():
    assert pd.detect_os() in ("windows", "linux", "macos")


def test_detect_arch_is_a_known_label_on_this_runner():
    assert pd.detect_arch() in ("x86_64", "arm64")


def test_detect_platform_with_probe_gpu_false_never_reports_a_gpu_backend_off_macos():
    result = pd.detect_platform(probe_gpu=False)
    assert set(result) == {"os", "arch", "gpu_backend", "cuda_major"}
    if result["os"] != "macos":
        assert result["gpu_backend"] is None
        assert result["cuda_major"] is None


def test_detect_platform_macos_arm64_always_reports_metal_with_no_probe(monkeypatch):
    monkeypatch.setattr(pd, "detect_os", lambda: "macos")
    monkeypatch.setattr(pd, "detect_arch", lambda: "arm64")
    calls = []
    monkeypatch.setattr(pd, "_probe_nvidia_cuda_major", lambda: calls.append(1) or 999)
    result = pd.detect_platform(probe_gpu=True)
    assert result == {"os": "macos", "arch": "arm64", "gpu_backend": "metal", "cuda_major": None}
    assert calls == []   # never probed -- metal needs no nvidia-smi


def test_probe_nvidia_cuda_major_returns_none_when_nvidia_smi_is_absent(monkeypatch):
    monkeypatch.setattr(pd.shutil, "which", lambda name: None)
    assert pd._probe_nvidia_cuda_major() is None


def test_probe_nvidia_cuda_major_parses_the_header_line(monkeypatch):
    class _FakeResult:
        stdout = "NVIDIA-SMI 550.54  Driver Version: 550.54    CUDA Version: 12.4     \n"

    monkeypatch.setattr(pd.shutil, "which", lambda name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(pd.subprocess, "run", lambda *a, **k: _FakeResult())
    assert pd._probe_nvidia_cuda_major() == 12


def test_probe_nvidia_cuda_major_never_raises_on_a_broken_subprocess(monkeypatch):
    monkeypatch.setattr(pd.shutil, "which", lambda name: "/usr/bin/nvidia-smi")

    def _boom(*a, **k):
        raise OSError("no such file")

    monkeypatch.setattr(pd.subprocess, "run", _boom)
    assert pd._probe_nvidia_cuda_major() is None
