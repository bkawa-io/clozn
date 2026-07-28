"""Contract tests for the opt-in subcommand autoloader (clozn/cli/commands/_autoload.py)."""
from __future__ import annotations

import argparse

import pytest

from clozn.cli import main as cli_main
from clozn.cli.commands import _autoload


def test_no_command_module_failed_to_load():
    """The teeth behind _autoload's deliberately-broad except: a module that opts in and then fails to
    import degrades to a stderr warning at runtime (so a broken command cannot take down `clozn doctor`),
    which would be easy to scroll past. This turns it into a hard CI failure instead."""
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


def test_hand_wired_commands_are_not_double_registered():
    """Existing modules define add_subparser but do NOT set the marker. If autoload ever picked one up,
    argparse would raise on the duplicate name -- this asserts the whole tree still builds."""
    parser = cli_main.build_parser()
    sub = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    assert sub, "no subparsers on the root parser"
    names = list(sub[0].choices)
    assert len(names) == len(set(names)), f"duplicate subcommand names: {names}"
    # A representative sample of the hand-wired tree, to catch an autoload change that shadows one.
    for expected in ("run", "serve", "models", "doctor", "inspect", "provenance", "version"):
        assert expected in names, f"`clozn {expected}` disappeared from the command tree"


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
    monkeypatch.setattr(_autoload, "_DIR", str(tmp_path))
    monkeypatch.setattr(_autoload, "_PACKAGE", "zz_probe_pkg_shim")
    try:
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="cmd")
        count = _autoload.register_all(sub)
        assert count == 1, f"expected 1 registration, got {count}; failures={_autoload.LOAD_FAILURES}"
        assert "zz-probe" in sub.choices
    finally:
        sys.modules.pop("zz_probe_pkg_shim", None)
        sys.modules.pop("zz_probe_pkg_shim.zz_probe_command", None)


def test_a_module_without_the_marker_is_not_imported(tmp_path, monkeypatch):
    """The text scan must skip non-participating modules outright -- importing every module in the
    package on every `clozn --help` is the cost this design exists to avoid."""
    (tmp_path / "zz_inert.py").write_text("raise AssertionError('must not be imported')\n",
                                          encoding="utf-8")
    monkeypatch.setattr(_autoload, "_DIR", str(tmp_path))
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd")
    before = len(_autoload.LOAD_FAILURES)
    assert _autoload.register_all(sub) == 0
    assert len(_autoload.LOAD_FAILURES) == before, "an inert module was imported"


def test_a_broken_module_is_recorded_not_raised(tmp_path, monkeypatch, capsys):
    """Roadmap rule 3 (no silent fallback): the failure is announced and recorded, but the CLI survives."""
    (tmp_path / "zz_broken.py").write_text("CLOZN_AUTOLOAD = True\nimport nonexistent_module_xyz\n",
                                           encoding="utf-8")
    monkeypatch.setattr(_autoload, "_DIR", str(tmp_path))
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
        assert _autoload.register_all(sub) == 0          # did not raise
        assert len(_autoload.LOAD_FAILURES) == before + 1
        assert "zz_broken" in capsys.readouterr().err
    finally:
        sys.modules.pop("zz_broken_shim", None)
        del _autoload.LOAD_FAILURES[before:]


def test_marker_constant_is_the_documented_name():
    """Agents are told to write `CLOZN_AUTOLOAD = True` verbatim; renaming the constant silently breaks
    every command module written against the documented contract."""
    assert _autoload.MARKER == "CLOZN_AUTOLOAD"
