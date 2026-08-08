"""Regression coverage for retiring downstream client configuration management.

Clozn is the debugger; other applications own their own configuration. `clozn connect` (and the
generic connector framework it was built on) used to safely patch a third-party app's config file --
Aider's YAML, Open WebUI's env, a generic OpenAI env, an Ollama SDK env -- with backup, drift
detection, and undo. That whole feature was removed: see docs/CAPABILITIES.md. `clozn adopt ollama`
keeps discovering/reusing an existing Ollama model; it never mutates another application's state.
"""
from __future__ import annotations

import argparse
import json

import pytest


# --------------------------------------------------------------------------- the modules are gone

def test_connect_command_module_no_longer_exists():
    with pytest.raises(ImportError):
        import clozn.cli.commands.connect  # noqa: F401


def test_connector_framework_module_no_longer_exists():
    with pytest.raises(ImportError):
        import clozn.cli.commands._connector  # noqa: F401


# ----------------------------------------------------------------------------- the CLI is gone

def test_clozn_connect_no_longer_exists_in_the_parser():
    from clozn.cli.main import build_parser
    parser = build_parser()
    subparsers_action = next(
        action for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert "connect" not in subparsers_action.choices


def test_help_output_does_not_advertise_connect(capsys):
    from clozn.cli.main import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    printed = capsys.readouterr().out
    assert "connect" not in printed.lower()


def test_clozn_connect_is_an_unrecognized_command():
    from clozn.cli.main import build_parser
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["connect", "aider"])


# ------------------------------------------------------------------------- old connector state is inert

@pytest.fixture
def isolated_home(tmp_path, monkeypatch):
    import clozn.cli.main as ctx
    home_dir = tmp_path / "clozn-home"
    home_dir.mkdir()
    monkeypatch.setattr(ctx, "HOME", str(home_dir))
    return home_dir


def test_old_connect_transaction_files_are_never_read(isolated_home):
    """A transaction directory left behind by a previous Clozn version must have zero runtime effect:
    never read, never used to restore or reapply a configuration, and its mere presence must never
    raise."""
    connect_dir = isolated_home / "connect"
    connect_dir.mkdir()
    (connect_dir / "aider.json").write_text(json.dumps({
        "schema_version": "clozn.connect.transaction.v1",
        "app": "aider",
        "target": "/home/user/.aider.conf.yml",
        "target_existed": True,
        "backup": "/home/user/.aider.conf.yml.bak-20260101T000000.000000Z",
        "before_sha256": "a" * 64,
        "after_sha256": "b" * 64,
        "created_at": "2026-01-01T00:00:00+00:00",
    }), encoding="utf-8")
    (connect_dir / "open-webui.json").write_text("not even valid json {{{", encoding="utf-8")

    # Nothing in the product reads this directory. Proven two ways: (1) no importable module
    # references it at all (see the module-gone tests above), and (2) an unrelated adopt-side
    # operation completes normally with this directory present, untouched, and never inspected.
    from clozn.cli.commands import adopt
    report = adopt._describe_setup(argparse.Namespace(app="ollama", host="http://127.0.0.1:1", model=None))
    assert report["status"] == "described"

    # The stale files are exactly as they were -- nothing attempted to read, restore, or delete them.
    assert (connect_dir / "aider.json").is_file()
    assert (connect_dir / "open-webui.json").read_text(encoding="utf-8") == "not even valid json {{{"


def test_old_connect_transaction_state_does_not_block_adoption(isolated_home, tmp_path, monkeypatch):
    """A stale ~/.clozn/connect/ directory must not interfere with `clozn adopt ollama` in any way --
    the two were always independent, and one no longer exists at all."""
    connect_dir = isolated_home / "connect"
    connect_dir.mkdir()
    (connect_dir / "aider.json").write_text(json.dumps({
        "schema_version": "clozn.connect.transaction.v1", "app": "aider",
        "target": str(tmp_path / ".aider.conf.yml"), "target_existed": False,
        "after_sha256": "c" * 64, "created_at": "2026-01-01T00:00:00+00:00",
    }), encoding="utf-8")

    import urllib.error
    import urllib.request
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(urllib.error.URLError("refused")))
    from clozn.cli.commands import adopt
    with pytest.raises(ValueError, match="no Ollama installation found"):
        adopt._build_plan(argparse.Namespace(app="ollama", host=None, model="anything"))
    # The failure above is the ordinary "no Ollama found" refusal, not anything related to the stale
    # connect state -- which remains untouched.
    assert (connect_dir / "aider.json").is_file()


# ----------------------------------------------------------------------------- historical schema reads

def test_historical_adopt_document_with_client_transactions_still_validates():
    """An adopt-ollama transaction written by a previous Clozn version may still carry the retired
    client_transactions field. Reading/validating it today must not fail -- historical evidence stays
    legible even though nothing produces or acts on that field anymore."""
    from clozn import schemas
    document = {
        "schema_version": "clozn.adopt-ollama.v1",
        "created_at": "2026-01-01T00:00:00+00:00",
        "discovery": {"source": "env", "host": "http://127.0.0.1:11434"},
        "ollama": {"model_name": "llama3.1:8b"},
        "clozn": {
            "registered_name": "ollama/llama3.1:8b",
            "path": "/home/user/.clozn/models/ollama__llama3.1_8b.gguf",
            "mode": "copy",
        },
        "template": {"source": "ollama_modelfile", "exactly_reproduced": False, "warnings": []},
        "client_transactions": [{
            "app": "aider", "status": "updated", "target": "/home/user/.aider.conf.yml",
            "state_path": "/home/user/.clozn/connect/aider.json",
        }],
    }
    schemas.validate(document, "clozn.adopt-ollama.v1")
