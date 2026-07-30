"""tests/test_teaching_loop_cli.py -- `clozn corrections verify ...` (F6's CLI exposure, added to the
existing `clozn/cli/commands/corrections.py`). Mirrors tests/test_corrections_cli.py's pattern: parse via
build_parser(), then call ns.fn(ns) directly -- no subprocess, no live model.
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clozn.runs.store as store                            # noqa: E402
from clozn.cli.main import CloznError, build_parser           # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(store, "RUNS_DIR", runs_dir)
    store._schema_verified.clear()
    yield runs_dir
    store._schema_verified.clear()


def _run(argv):
    ns = build_parser().parse_args(argv)
    return ns, ns.fn(ns)


def _record(response: str) -> str:
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response=response)
    assert rid is not None
    return rid


def test_verify_subcommand_is_registered():
    parser = build_parser()
    for action in parser._actions:
        if getattr(action, "choices", None) and "corrections" in action.choices:
            corrections_parser = action.choices["corrections"]
            break
    else:
        pytest.fail("corrections subcommand not found")
    for action in corrections_parser._actions:
        if getattr(action, "choices", None) and "verify" in action.choices:
            return
    pytest.fail("corrections verify subcommand not registered")


def test_verify_promotes_via_cli(isolated, capsys):
    ns, code = _run(["corrections", "draft", "--scope-kind", "session", "--scope-value", "sess-1",
                     "--type", "style", "--content", "Be terse.", "--json"])
    assert code == 0
    correction_id = json.loads(capsys.readouterr().out)["id"]

    target = _record("long rambling bad answer")
    child = _record("short good answer")

    ns, code = _run(["corrections", "verify", correction_id, "--target", target, "--child", child,
                     "--json"])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["verification"] == "passed"
    assert result["promoted"] is True
    assert result["target_run_id"] == target
    assert result["child_run_id"] == child

    ns, code = _run(["corrections", "resolve", "--session", "sess-1", "--json"])
    resolution = json.loads(capsys.readouterr().out)
    assert [a["correction_id"] for a in resolution["applied"]] == [correction_id]


def test_verify_leaves_draft_via_cli_on_failed_verification(isolated, capsys):
    ns, code = _run(["corrections", "draft", "--scope-kind", "session", "--scope-value", "sess-1",
                     "--type", "style", "--content", "Be terse.", "--json"])
    correction_id = json.loads(capsys.readouterr().out)["id"]

    target = _record("same output")
    child = _record("same output")

    ns, code = _run(["corrections", "verify", correction_id, "--target", target, "--child", child,
                     "--json"])
    assert code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["verification"] == "failed"
    assert result["promoted"] is False

    ns, code = _run(["corrections", "show", correction_id, "--json"])
    doc = json.loads(capsys.readouterr().out)
    assert "confirmed_ts" not in doc


def test_verify_unknown_run_id_raises_cloznerror(isolated, capsys):
    ns, code = _run(["corrections", "draft", "--scope-kind", "session", "--scope-value", "sess-1",
                     "--type", "style", "--content", "Be terse.", "--json"])
    correction_id = json.loads(capsys.readouterr().out)["id"]
    child = _record("good")

    ns = build_parser().parse_args(
        ["corrections", "verify", correction_id, "--target", "run_missing", "--child", child])
    with pytest.raises(CloznError):
        ns.fn(ns)


def test_verify_rejects_unknown_match_criterion_at_argparse_level():
    with pytest.raises(SystemExit):
        build_parser().parse_args(
            ["corrections", "verify", "corr_" + "0" * 24, "--target", "run_a", "--child", "run_b",
             "--match", "vibes"])
