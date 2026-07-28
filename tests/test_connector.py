"""Tests for clozn.cli.commands._connector: the extracted primitives (proven equivalent to
connect.py's pre-refactor behavior by tests/test_connect_cli.py continuing to pass unchanged) plus the
generic Connector interface and its first implementation, AiderConnector.
"""
from __future__ import annotations

import shutil

import pytest

from clozn.cli.commands import _connector as conn


# --------------------------------------------------------------------------------------- primitives

def test_atomic_write_text_then_sha256_path_round_trips(tmp_path):
    target = tmp_path / "file.txt"
    conn.atomic_write_text(target, "hello\n", prior_mode=None)
    assert target.read_text(encoding="utf-8") == "hello\n"
    assert conn.sha256_path(target) == conn.sha256_bytes(b"hello\n")


def test_atomic_restore_replaces_target_with_source_bytes(tmp_path):
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("original\n", encoding="utf-8")
    target.write_text("current\n", encoding="utf-8")
    conn.atomic_restore(source, target)
    assert target.read_text(encoding="utf-8") == "original\n"


# ------------------------------------------------------------------------------------------ Connector

def test_connector_base_class_methods_raise_not_implemented():
    base = conn.Connector()
    with pytest.raises(NotImplementedError):
        base.detect()
    with pytest.raises(NotImplementedError):
        base.plan()
    with pytest.raises(NotImplementedError):
        base.apply(None)
    with pytest.raises(NotImplementedError):
        base.undo()


def test_aider_connector_id_is_aider():
    assert conn.AiderConnector.id == "aider"


def test_aider_connector_detect_reports_absent_executable(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda _name: None)
    connector = conn.AiderConnector(config_path=tmp_path / ".aider.conf.yml")
    detection = connector.detect()
    assert detection.installed is False
    assert detection.app == "aider"
    assert "no 'aider' executable" in detection.note


def test_aider_connector_detect_reports_present_executable(monkeypatch, tmp_path):
    monkeypatch.setattr(shutil, "which", lambda _name: "/usr/bin/aider")
    connector = conn.AiderConnector(config_path=tmp_path / ".aider.conf.yml")
    detection = connector.detect()
    assert detection.installed is True
    assert detection.executable_path == "/usr/bin/aider"


def test_aider_connector_plan_is_dry_run_and_writes_nothing(tmp_path):
    config = tmp_path / ".aider.conf.yml"
    state = tmp_path / "state.json"
    connector = conn.AiderConnector(config_path=config)
    plan = connector.plan(base_url="http://127.0.0.1:8080", model="clozn", api_key="local",
                          state_path=state)
    assert plan.app == "aider"
    assert plan.status == "dry_run"
    assert not config.exists()
    assert not state.exists()


def test_aider_connector_apply_then_undo_round_trips(tmp_path):
    config = tmp_path / ".aider.conf.yml"
    state = tmp_path / "state.json"
    config.write_text("dark-mode: true\n", encoding="utf-8")
    original = config.read_bytes()
    connector = conn.AiderConnector(config_path=config)

    transaction = connector.apply(base_url="http://127.0.0.1:8080", model="clozn", api_key="local",
                                  state_path=state)
    assert transaction.app == "aider"
    assert transaction.report["status"] == "updated"
    assert config.read_bytes() != original

    undone = connector.undo(state_path=state)
    assert undone.app == "aider"
    assert undone.status == "restored"
    assert config.read_bytes() == original
    assert not state.exists()
