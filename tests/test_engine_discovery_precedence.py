"""find_engine_ex()'s 4-tier precedence (roadmap feature 01): CLOZN_ENGINE(_BIN) override -> active
managed engine (clozn setup) -> repository-local dev build -> legacy search paths. Every tier is
exercised in isolation via tmp_path + monkeypatch -- no real engine build or GPU is ever required.
"""
from __future__ import annotations

import os

import pytest

from clozn.cli import engine_process
from clozn.cli import main as cli
from clozn.cli.main import CloznError
from clozn.setup import registry as setup_registry


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_process, "ENGINE_CORE", str(tmp_path / "engine_core"))
    monkeypatch.setattr(cli, "HOME", str(tmp_path / ".clozn"))
    for var in ("CLOZN_ENGINE", "CLOZN_ENGINE_BIN", "CLOZN_ENGINE_GPU"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("fake")
    return path


def _dev_build(tmp_path, subdir, gpu_dirname):
    exe = str(tmp_path / "engine_core" / subdir / "clozn-server.exe")
    return _touch(exe)


def _install_managed(home, *, backend="cuda", cuda_major=12, sha256="a" * 64, version="1.0.0"):
    exe = _touch(os.path.join(home, "engines", version, f"windows-x86_64-{backend}", "bin", "clozn-server.exe"))
    record = {
        "version": version, "os": "windows", "arch": "x86_64", "backend": backend,
        **({"cuda_major": cuda_major} if backend == "cuda" else {}),
        "sha256": sha256, "protocol_version": "1.0", "entrypoint": exe,
        "installed_at": "2026-07-27T00:00:00+00:00",
    }
    key = f"{version}/windows-x86_64-{backend}{cuda_major if backend == 'cuda' else ''}"
    doc = setup_registry.record_install({}, key, record, make_active=True)
    setup_registry.save(home, doc)
    return exe


# ------------------------------------------------------------------------------------------------- none

def test_no_candidates_raises_and_mentions_setup(isolated):
    with pytest.raises(CloznError, match="clozn setup"):
        engine_process.find_engine_ex()


# --------------------------------------------------------------------------------------- tier 3: repo dev

def test_repo_dev_build_only(isolated):
    exe = _dev_build(isolated, "build-gpu", "gpu")
    result = engine_process.find_engine_ex()
    assert result.exe == exe
    assert result.discovery_source == "repo_dev_build"
    assert result.backend == "gpu"
    assert result.gpu is True


def test_repo_dev_build_prefers_gpu_build_when_multiple_present(isolated):
    _dev_build(isolated, "build-cpu", "cpu")
    gpu_exe = _dev_build(isolated, "build-gpu", "gpu")
    result = engine_process.find_engine_ex(prefer_gpu=True)
    assert result.exe == gpu_exe


def test_repo_dev_build_prefer_gpu_false_skips_gpu_build(isolated):
    cpu_exe = _dev_build(isolated, "build-cpu", "cpu")
    _dev_build(isolated, "build-gpu", "gpu")
    result = engine_process.find_engine_ex(prefer_gpu=False)
    assert result.exe == cpu_exe
    assert result.gpu is False


def test_repo_dev_build_prefer_gpu_false_with_only_gpu_available_raises(isolated):
    _dev_build(isolated, "build-gpu", "gpu")
    with pytest.raises(CloznError, match="no CPU engine build"):
        engine_process.find_engine_ex(prefer_gpu=False)


# ----------------------------------------------------------------------------------------- tier 2: managed

def test_managed_engine_only(isolated):
    exe = _install_managed(cli.HOME, backend="cuda", cuda_major=12, sha256="b" * 64, version="2.0.0")
    result = engine_process.find_engine_ex()
    assert result.exe == exe
    assert result.discovery_source == "managed"
    assert result.backend == "cuda"
    assert result.artifact_sha256 == "b" * 64
    assert result.engine_version == "2.0.0"
    assert result.gpu is True


def test_managed_beats_repo_dev_build(isolated):
    _dev_build(isolated, "build-gpu", "gpu")
    managed_exe = _install_managed(cli.HOME)
    result = engine_process.find_engine_ex()
    assert result.exe == managed_exe
    assert result.discovery_source == "managed"


def test_managed_cpu_engine_reports_cpu_backend(isolated):
    exe = _install_managed(cli.HOME, backend="cpu", cuda_major=None, version="2.0.0")
    result = engine_process.find_engine_ex()
    assert result.exe == exe
    assert result.backend == "cpu"
    assert result.gpu is False


def test_managed_gpu_engine_skipped_when_prefer_gpu_false_falls_through_to_dev_cpu_build(isolated):
    cpu_dev = _dev_build(isolated, "build-cpu", "cpu")
    _install_managed(cli.HOME, backend="cuda")
    result = engine_process.find_engine_ex(prefer_gpu=False)
    assert result.exe == cpu_dev
    assert result.discovery_source == "repo_dev_build"


def test_managed_engine_with_missing_entrypoint_self_heals_to_next_tier(isolated):
    exe = _install_managed(cli.HOME)
    os.remove(exe)   # simulate the file having been removed out of band
    dev_exe = _dev_build(isolated, "build-cpu", "cpu")
    result = engine_process.find_engine_ex()
    assert result.exe == dev_exe
    assert result.discovery_source == "repo_dev_build"


# ------------------------------------------------------------------------------------- tier 1: env override

def test_clozn_engine_env_wins_over_everything(isolated, monkeypatch):
    _install_managed(cli.HOME)
    _dev_build(isolated, "build-cpu", "cpu")
    override_exe = _touch(str(isolated / "custom" / "clozn-server.exe"))
    monkeypatch.setenv("CLOZN_ENGINE", override_exe)
    result = engine_process.find_engine_ex()
    assert result.exe == override_exe
    assert result.discovery_source == "env_override"
    assert result.backend == "cpu"


def test_legacy_clozn_engine_bin_still_works(isolated, monkeypatch):
    override_exe = _touch(str(isolated / "custom" / "clozn-server.exe"))
    monkeypatch.setenv("CLOZN_ENGINE_BIN", override_exe)
    result = engine_process.find_engine_ex()
    assert result.exe == override_exe
    assert result.discovery_source == "env_override"


def test_clozn_engine_wins_over_clozn_engine_bin_when_both_set(isolated, monkeypatch):
    new_exe = _touch(str(isolated / "new" / "clozn-server.exe"))
    old_exe = _touch(str(isolated / "old" / "clozn-server.exe"))
    monkeypatch.setenv("CLOZN_ENGINE", new_exe)
    monkeypatch.setenv("CLOZN_ENGINE_BIN", old_exe)
    result = engine_process.find_engine_ex()
    assert result.exe == new_exe


def test_clozn_engine_gpu_marks_the_override_as_a_gpu_worker(isolated, monkeypatch):
    override_exe = _touch(str(isolated / "custom" / "clozn-server.exe"))
    monkeypatch.setenv("CLOZN_ENGINE", override_exe)
    monkeypatch.setenv("CLOZN_ENGINE_GPU", "1")
    result = engine_process.find_engine_ex()
    assert result.gpu is True
    assert result.backend == "gpu"


def test_prefer_gpu_false_refuses_a_gpu_env_override(isolated, monkeypatch):
    override_exe = _touch(str(isolated / "custom" / "clozn-server.exe"))
    monkeypatch.setenv("CLOZN_ENGINE", override_exe)
    monkeypatch.setenv("CLOZN_ENGINE_GPU", "1")
    with pytest.raises(CloznError, match="CLOZN_ENGINE"):
        engine_process.find_engine_ex(prefer_gpu=False)


def test_clozn_engine_pointing_at_a_missing_file_raises(isolated, monkeypatch):
    monkeypatch.setenv("CLOZN_ENGINE", str(isolated / "does-not-exist.exe"))
    with pytest.raises(CloznError, match="does not point to a file"):
        engine_process.find_engine_ex()


# --------------------------------------------------------------------------------------- back-compat shape

def test_find_engine_still_returns_the_historical_3_tuple(isolated):
    exe = _dev_build(isolated, "build-cpu", "cpu")
    result = engine_process.find_engine()
    assert result == (exe, engine_process._dll_dirs_for(exe), False)


def test_find_engine_and_find_engine_ex_agree(isolated):
    _install_managed(cli.HOME)
    exe, dll_dirs, gpu = engine_process.find_engine()
    ex = engine_process.find_engine_ex()
    assert (exe, dll_dirs, gpu) == (ex.exe, ex.dll_dirs, ex.gpu)
