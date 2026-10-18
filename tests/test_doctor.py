"""Focused tests for the Phase 3.5 `clozn doctor --verify-offline` additions: the static loopback-bind
check and the always-present "by design" outbound-capable command disclosure. `_check_offline`'s pass/fail
logic itself is exercised end-to-end in tests/test_privacy_cli.py; this file covers what changed here.
"""
from __future__ import annotations

import json

import pytest

from clozn import network_policy
from clozn.cli.commands import doctor


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(network_policy, "POLICY_PATH", str(tmp_path / "network_policy.json"))
    monkeypatch.setattr(network_policy, "LEDGER_PATH", str(tmp_path / "outbound_attempts.jsonl"))
    monkeypatch.delenv(network_policy.POLICY_ENV, raising=False)
    monkeypatch.delenv(network_policy.LEDGER_ENV, raising=False)
    monkeypatch.delenv(network_policy.LOCAL_ONLY_ENV, raising=False)
    return tmp_path


def test_check_bind_loopback_reports_ok_on_this_checkout():
    result = doctor._check_bind_loopback()
    assert result["label"] == "gateway/engine bind"
    assert result["status"] == "OK"
    assert "127.0.0.1" in result["detail"]
    assert "not a live-process probe" in result["detail"]


def test_check_bind_loopback_warns_never_fails_on_a_broken_import(monkeypatch):
    import clozn.cli.commands.doctor as doctor_module

    def _broken_build_parser():
        raise RuntimeError("boom")

    monkeypatch.setattr("clozn.cli.main.build_parser", _broken_build_parser)
    result = doctor_module._check_bind_loopback()
    assert result["status"] == "WARN"
    assert "boom" in result["detail"]


def test_check_offline_always_lists_known_outbound_capable_commands(isolated):
    result = doctor._check_offline()
    assert "known_outbound_capable_commands" in result
    commands = result["known_outbound_capable_commands"]
    assert any("clozn pull" in c for c in commands)
    assert "clozn pull" in result["detail"]


def test_check_offline_lists_known_commands_even_when_verified(isolated):
    network_policy.set_local_only(True)
    result = doctor._check_offline()
    assert result["status"] == "OK"
    assert "clozn pull" in result["detail"]


def test_run_all_verify_offline_includes_bind_and_offline_checks(isolated):
    checks = doctor._run_all(verify_offline=True)
    labels = [c["label"] for c in checks]
    assert "gateway/engine bind" in labels
    assert "offline enforcement" in labels
    # Both requested checks are additive to the always-on baseline.
    assert len(labels) == len(doctor._run_all(verify_offline=False)) + 2


def test_run_all_without_verify_offline_omits_both_new_checks():
    checks = doctor._run_all(verify_offline=False)
    labels = [c["label"] for c in checks]
    assert "gateway/engine bind" not in labels
    assert "offline enforcement" not in labels


def test_cmd_doctor_json_includes_new_checks(isolated, capsys):
    import argparse
    args = argparse.Namespace(json=True, verify_offline=True)
    doctor.cmd_doctor(args)
    out = json.loads(capsys.readouterr().out)
    labels = [c["label"] for c in out["checks"]]
    assert "gateway/engine bind" in labels
