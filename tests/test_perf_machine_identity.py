"""Model-free coverage for clozn/runs/identity_providers/machine.py -- the machine/runtime identity facet
this feature owns. Complements tests/test_identity_ext.py's blanket "no shipped provider raises" check
with this provider's own logic: caching, nvidia-smi absence/failure, and the seam's namespace contract.
"""
from __future__ import annotations

import subprocess

import pytest

from clozn.runs import identity_ext
from clozn.runs.identity_providers import machine


@pytest.fixture(autouse=True)
def _reset():
    machine.reset_cache()
    identity_ext.reset_cache()
    yield
    machine.reset_cache()
    identity_ext.reset_cache()


def test_name_is_the_documented_namespace():
    assert machine.NAME == "machine"


def test_identity_never_raises_even_if_every_probe_fails(monkeypatch):
    monkeypatch.setattr(machine.platform, "system", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(machine.platform, "release", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(machine.platform, "machine", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(machine.os, "cpu_count", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(machine.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError))
    assert machine.identity(None) == {}


def test_nvidia_smi_absent_omits_gpu_vram_but_keeps_other_facts(monkeypatch):
    def _raise(*a, **k):
        raise FileNotFoundError("no nvidia-smi")
    monkeypatch.setattr(machine.subprocess, "run", _raise)
    result = machine.identity(None)
    assert "gpu_vram_gb" not in result
    # os/arch/cpu_count are real platform facts on whatever machine runs this test -- at least one of the
    # cheap facts should still come through even with no GPU probe available.
    assert "os" in result


def test_nvidia_smi_present_parses_vram_gb(monkeypatch):
    def _fake_run(args, **kwargs):
        return subprocess.CompletedProcess(args, 0, stdout="16384\n", stderr="")
    monkeypatch.setattr(machine.subprocess, "run", _fake_run)
    result = machine.identity(None)
    assert result["gpu_vram_gb"] == 16.0


def test_probe_runs_at_most_once_per_process_even_across_many_calls(monkeypatch):
    calls = {"n": 0}

    def _counting_run(args, **kwargs):
        calls["n"] += 1
        return subprocess.CompletedProcess(args, 0, stdout="8192\n", stderr="")
    monkeypatch.setattr(machine.subprocess, "run", _counting_run)

    for _ in range(5):
        machine.identity({"anything": "ignored"})
    assert calls["n"] == 1, "nvidia-smi must be probed once per process, not once per recorded run"


def test_a_negative_result_is_cached_too_and_does_not_retry(monkeypatch):
    """No GPU found is itself worth remembering -- a GPU-less machine must not re-spawn nvidia-smi on
    every single chat turn just to learn the same negative answer again."""
    calls = {"n": 0}

    def _raise(*a, **k):
        calls["n"] += 1
        raise FileNotFoundError
    monkeypatch.setattr(machine.subprocess, "run", _raise)
    monkeypatch.setattr(machine.platform, "system", lambda: "")
    monkeypatch.setattr(machine.platform, "release", lambda: "")
    monkeypatch.setattr(machine.platform, "machine", lambda: "")
    monkeypatch.setattr(machine.os, "cpu_count", lambda: None)

    assert machine.identity(None) == {}
    assert machine.identity(None) == {}
    assert calls["n"] == 1


def test_returned_dict_is_a_copy_not_the_live_cache(monkeypatch):
    monkeypatch.setattr(machine.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError))
    monkeypatch.setattr(machine.platform, "system", lambda: "Windows")
    result = machine.identity(None)
    result["os"] = "mutated"
    assert machine.identity(None)["os"] == "Windows"


def test_lands_under_ext_machine_via_the_real_identity_ext_seam(monkeypatch):
    """End-to-end through the actual identity_ext discovery (not a shim), matching what
    clozn.runs.identity.runtime_identity() will produce on a real run."""
    monkeypatch.setattr(machine.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError))
    collected = identity_ext.collect({})
    assert "machine" in collected
    assert isinstance(collected["machine"], dict)
