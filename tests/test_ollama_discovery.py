"""Model-free tests for clozn.adopt.ollama_discovery -- no real Ollama process is ever started.

Every HTTP call is intercepted by monkeypatching `urllib.request.urlopen`; every executable probe is
intercepted by monkeypatching `subprocess.run`. Response fixtures under tests/fixtures/ollama/ simulate
two different Ollama releases so list_models()/show_model() are proven tolerant of both shapes.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request

import pytest

from clozn import network_policy
from clozn.adopt import ollama_discovery as od

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "ollama")


def _load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as handle:
        return json.load(handle)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False

    def read(self):
        return self._body


def _urlopen_returning(payload: dict):
    def _fake(_request, timeout=None):
        return _FakeResponse(payload)
    return _fake


def _urlopen_raising(exc: BaseException):
    def _fake(_request, timeout=None):
        raise exc
    return _fake


def _urlopen_only_for_host(host: str, payload: dict):
    """Answers only requests whose full_url starts with `host`; anything else looks like nothing's
    listening there, matching real socket behavior when only one of several hosts has a server."""
    def _fake(request, timeout=None):
        url = request.full_url if isinstance(request, urllib.request.Request) else str(request)
        if url.startswith(host):
            return _FakeResponse(payload)
        raise urllib.error.URLError("connection refused")
    return _fake


# --------------------------------------------------------------------------------------- probe_endpoint

def test_probe_endpoint_returns_version_on_success(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_returning({"version": "0.6.2"}))
    assert od.probe_endpoint("http://127.0.0.1:11434") == {"version": "0.6.2"}


def test_probe_endpoint_returns_none_on_connection_refused(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_raising(urllib.error.URLError("refused")))
    assert od.probe_endpoint("http://127.0.0.1:11434") is None


def test_probe_endpoint_returns_none_on_non_json_body(monkeypatch):
    def _fake(_request, timeout=None):
        class _Bad:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *_exc):
                return False

            def read(self_inner):
                return b"not json"
        return _Bad()
    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    assert od.probe_endpoint("http://127.0.0.1:11434") is None


def test_probe_endpoint_propagates_local_only_violation(monkeypatch):
    violation = network_policy.LocalOnlyViolation("10.0.0.5", "private_network")
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_raising(violation))
    with pytest.raises(network_policy.LocalOnlyViolation):
        od.probe_endpoint("http://10.0.0.5:11434")


# ---------------------------------------------------------------------------------- executable_version

def test_executable_version_returns_none_when_no_path_given():
    assert od.executable_version(None) is None


def test_executable_version_parses_successful_output(monkeypatch):
    def _fake_run(cmd, capture_output, text, timeout):
        assert cmd == ["/usr/bin/ollama", "--version"]
        return subprocess.CompletedProcess(cmd, 0, stdout="ollama version is 0.6.2\n", stderr="")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert od.executable_version("/usr/bin/ollama") == "ollama version is 0.6.2"


def test_executable_version_returns_none_on_nonzero_exit(monkeypatch):
    def _fake_run(cmd, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="boom")
    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert od.executable_version("/usr/bin/ollama") is None


def test_executable_version_returns_none_on_timeout(monkeypatch):
    def _fake_run(cmd, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)
    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert od.executable_version("/usr/bin/ollama") is None


# ------------------------------------------------------------------------------------- known_storage_*

def test_known_storage_path_prefers_env_override(tmp_path, monkeypatch):
    override = tmp_path / "custom-ollama-models"
    override.mkdir()
    monkeypatch.setenv("OLLAMA_MODELS", str(override))
    assert od.known_storage_path() == str(override.resolve())


def test_known_storage_path_none_when_nothing_exists(tmp_path, monkeypatch):
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / "nonexistent-home") if p == "~" else p)
    assert od.known_storage_path() is None


# ------------------------------------------------------------------------------------------- discover()

def test_discover_finds_configured_env_host_first(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9999")
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _urlopen_only_for_host("http://127.0.0.1:9999", {"version": "0.6.2"}))
    result = od.discover(exe_path=None)
    assert result["found"] is True
    assert result["source"] == "env"
    assert result["host"] == "http://127.0.0.1:9999"
    assert result["version"] == "0.6.2"
    assert result["warnings"] == []


def test_discover_falls_back_to_executable_when_configured_host_unreachable(monkeypatch, tmp_path):
    fake_exe = tmp_path / "ollama"
    fake_exe.write_text("")
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9999")
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_raising(urllib.error.URLError("refused")))

    def _fake_run(cmd, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 0, stdout="ollama version is 0.6.2", stderr="")
    monkeypatch.setattr(subprocess, "run", _fake_run)

    result = od.discover(exe_path=str(fake_exe))
    assert result["found"] is True
    assert result["source"] == "executable"
    assert result["version"] == "ollama version is 0.6.2"
    assert any("did not answer" in w for w in result["warnings"])


def test_discover_falls_back_to_default_endpoint_when_no_env_or_executable(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setattr(
        urllib.request, "urlopen",
        _urlopen_only_for_host(od.OLLAMA_DEFAULT_HOST, {"version": "0.5.1"}))
    result = od.discover(exe_path=None)
    assert result["found"] is True
    assert result["source"] == "endpoint"
    assert result["host"] == od.OLLAMA_DEFAULT_HOST
    assert result["version"] == "0.5.1"


def test_discover_falls_back_to_storage_when_nothing_reachable(monkeypatch, tmp_path):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_raising(urllib.error.URLError("refused")))
    storage = tmp_path / "models"
    storage.mkdir()
    monkeypatch.setenv("OLLAMA_MODELS", str(storage))

    result = od.discover(exe_path=None)
    assert result["found"] is True
    assert result["source"] == "storage_fallback"
    assert result["storage_path"] == str(storage.resolve())
    assert result["host"] is None
    assert result["version"] is None
    assert any("storage detection only" in w for w in result["warnings"])


def test_discover_reports_not_found_when_nothing_exists(monkeypatch, tmp_path):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_raising(urllib.error.URLError("refused")))
    monkeypatch.setattr(od, "known_storage_path", lambda: None)

    result = od.discover(exe_path=None)
    assert result["found"] is False
    assert result["source"] is None


def test_discover_treats_local_only_block_as_a_warning_not_a_silent_miss(monkeypatch, tmp_path):
    monkeypatch.setenv("OLLAMA_HOST", "http://10.0.0.5:11434")
    violation = network_policy.LocalOnlyViolation("10.0.0.5", "private_network")

    def _fake(request, timeout=None):
        url = request.full_url
        if url.startswith("http://10.0.0.5:11434"):
            raise violation
        raise urllib.error.URLError("refused")
    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    monkeypatch.setattr(od, "known_storage_path", lambda: None)

    result = od.discover(exe_path=None)
    assert result["found"] is False
    assert any("local-only network policy blocked" in w for w in result["warnings"])


# ---------------------------------------------------------------------------- list_models / show_model

@pytest.mark.parametrize("fixture_name", ["tags_v0_1_29.json", "tags_v0_6_2.json"])
def test_list_models_tolerates_both_fixture_versions(monkeypatch, fixture_name):
    payload = _load_fixture(fixture_name)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_returning(payload))
    models = od.list_models("http://127.0.0.1:11434")
    assert isinstance(models, list) and len(models) >= 1
    assert all("name" in m for m in models)


def test_list_models_raises_on_transport_failure(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_raising(urllib.error.URLError("refused")))
    with pytest.raises(urllib.error.URLError):
        od.list_models("http://127.0.0.1:11434")


@pytest.mark.parametrize("fixture_name", ["show_v0_1_29.json", "show_v0_6_2.json"])
def test_show_model_tolerates_both_fixture_versions(monkeypatch, fixture_name):
    payload = _load_fixture(fixture_name)
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_returning(payload))
    shown = od.show_model("http://127.0.0.1:11434", "qwen2.5:7b-instruct")
    assert "modelfile" in shown
    assert "template" in shown


def test_show_model_raises_on_transport_failure(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", _urlopen_raising(urllib.error.URLError("refused")))
    with pytest.raises(urllib.error.URLError):
        od.show_model("http://127.0.0.1:11434", "missing-model")
