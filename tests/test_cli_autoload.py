"""Contract tests for the explicit CLI registration manifest."""
from __future__ import annotations

import argparse

import pytest

from clozn.cli import main as cli_main
from clozn.cli.commands import _autoload


def test_no_command_module_failed_to_load():
    """A broken manifested command remains reachable as a hard CI failure."""
    cli_main.build_parser()
    assert _autoload.LOAD_FAILURES == [], (
        "command modules opted into autoload but failed: "
        + "; ".join(f"{name}: {type(exc).__name__}: {exc}" for name, exc in _autoload.LOAD_FAILURES))


def test_build_parser_is_idempotent():
    """build_parser() is called by tests and by main(); autoload must not accumulate state across calls
    or register a command onto a stale parser."""
    first = {a.dest for a in cli_main.build_parser()._actions}
    second = {a.dest for a in cli_main.build_parser()._actions}
    assert first == second


def test_public_commands_match_the_v1_allowlist():
    parser = cli_main.build_parser()
    sub = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert sub, "no subparsers on the root parser"
    names = list(sub[0].choices)
    assert len(names) == len(set(names)), f"duplicate subcommand names: {names}"
    assert tuple(names) == cli_main.PUBLIC_COMMANDS


def test_production_command_manifest_is_explicit_and_stable():
    assert _autoload.COMMAND_MODULES == ()
    assert _autoload._candidates() == sorted(_autoload.COMMAND_MODULES)


def test_a_module_that_opts_in_is_registered(tmp_path, monkeypatch):
    """End-to-end: a file with CLOZN_AUTOLOAD = True and add_subparser() lands in the parser."""
    module = tmp_path / "zz_probe_command.py"
    module.write_text(
        "CLOZN_AUTOLOAD = True\n"
        "def cmd_probe(args):\n"
        "    return 0\n"
        "def add_subparser(sub):\n"
        "    p = sub.add_parser('zz-probe', help='autoload probe')\n"
        "    p.set_defaults(fn=cmd_probe)\n",
        encoding="utf-8")

    # A throwaway package rooted at tmp_path, so importlib resolves the probe without touching the real
    # clozn.cli.commands tree.
    import sys
    import types
    shim = types.ModuleType("zz_probe_pkg_shim")
    shim.__path__ = [str(tmp_path)]
    sys.modules["zz_probe_pkg_shim"] = shim
    monkeypatch.setattr(_autoload, "_PACKAGE", "zz_probe_pkg_shim")
    try:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        count = _autoload.register_all(sub, ["zz_probe_command"])
        assert count == 1, f"expected 1 registration, got {count}; failures={_autoload.LOAD_FAILURES}"
        assert "zz-probe" in sub.choices
    finally:
        sys.modules.pop("zz_probe_pkg_shim", None)
        sys.modules.pop("zz_probe_pkg_shim.zz_probe_command", None)


def test_an_unlisted_module_is_not_imported(tmp_path, monkeypatch):
    """A source file outside the manifest cannot expose a command or even run import-time code."""
    (tmp_path / "zz_inert.py").write_text("raise AssertionError('must not be imported')\n",
                                          encoding="utf-8")
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    before = len(_autoload.LOAD_FAILURES)
    assert _autoload.register_all(sub, []) == 0
    assert len(_autoload.LOAD_FAILURES) == before, "an inert module was imported"


def test_a_broken_module_is_recorded_not_raised(tmp_path, monkeypatch, capsys):
    """Roadmap rule 3 (no silent fallback): the failure is announced and recorded, but the CLI survives."""
    (tmp_path / "zz_broken.py").write_text("CLOZN_AUTOLOAD = True\nimport nonexistent_module_xyz\n",
                                           encoding="utf-8")
    monkeypatch.setattr(_autoload, "_PACKAGE", "zz_broken_shim")
    import sys
    import types
    shim = types.ModuleType("zz_broken_shim")
    shim.__path__ = [str(tmp_path)]
    sys.modules["zz_broken_shim"] = shim
    before = len(_autoload.LOAD_FAILURES)
    try:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        assert _autoload.register_all(sub, ["zz_broken"]) == 0  # did not raise
        assert len(_autoload.LOAD_FAILURES) == before + 1
        assert "zz_broken" in capsys.readouterr().err
    finally:
        sys.modules.pop("zz_broken_shim", None)
        del _autoload.LOAD_FAILURES[before:]


def test_marker_constant_is_the_documented_name():
    """The marker is the module-local assertion paired with the central manifest."""
    assert _autoload.MARKER == "CLOZN_AUTOLOAD"


def test_manifest_rejects_invalid_or_duplicate_module_names():
    with pytest.raises(ValueError, match="invalid command"):
        _autoload._candidates(["not-a-module"])
    with pytest.raises(ValueError, match="duplicate command"):
        _autoload._candidates(["adapter", "adapter"])
