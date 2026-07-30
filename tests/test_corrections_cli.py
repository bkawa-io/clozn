"""tests/test_corrections_cli.py -- `clozn corrections ...` (clozn/cli/commands/corrections.py).

Registered via CLOZN_AUTOLOAD; mirrors tests/test_investigate_experiment_cli.py's pattern (parse via
build_parser(), then call ns.fn(ns) directly -- no subprocess, no live model).
"""
from __future__ import annotations

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clozn.cli.commands.corrections as cmd_mod          # noqa: E402
import clozn.runs.store as store                            # noqa: E402
from clozn.cli.main import CloznError, build_parser         # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(store, "RUNS_DIR", runs_dir)
    store._schema_verified.clear()
    yield runs_dir
    store._schema_verified.clear()


def _subparser_choices(parser):
    for action in parser._actions:
        if getattr(action, "choices", None) and "corrections" in action.choices:
            return action.choices
    return {}


def _run(argv):
    ns = build_parser().parse_args(argv)
    return ns, ns.fn(ns)


def test_corrections_is_registered():
    assert "corrections" in _subparser_choices(build_parser())


def test_autoload_marker_is_set():
    assert cmd_mod.CLOZN_AUTOLOAD is True


def test_draft_confirm_list_resolve_roundtrip(isolated, capsys):
    ns, code = _run(["corrections", "draft", "--scope-kind", "session", "--scope-value", "sess-1",
                     "--type", "style", "--content", "Be terse.", "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    correction_id = out["id"]
    assert out["enabled"] is False

    ns, code = _run(["corrections", "confirm", correction_id, "--json"])
    assert code == 0
    confirmed = json.loads(capsys.readouterr().out)
    assert confirmed["correction"]["enabled"] is True

    ns, code = _run(["corrections", "list", "--json"])
    assert code == 0
    listed = json.loads(capsys.readouterr().out)
    assert [d["id"] for d in listed] == [correction_id]

    ns, code = _run(["corrections", "resolve", "--session", "sess-1", "--json"])
    assert code == 0
    resolution = json.loads(capsys.readouterr().out)
    assert [a["correction_id"] for a in resolution["applied"]] == [correction_id]


def test_disable_enable_undo(isolated, capsys):
    ns, _ = _run(["corrections", "draft", "--scope-kind", "session", "--scope-value", "sess-1",
                 "--type", "style", "--content", "Be terse.", "--json"])
    correction_id = json.loads(capsys.readouterr().out)["id"]
    _run(["corrections", "confirm", correction_id])
    capsys.readouterr()

    _run(["corrections", "disable", correction_id, "--json"])
    disabled = json.loads(capsys.readouterr().out)
    assert disabled["enabled"] is False

    _run(["corrections", "undo", correction_id, "--json"])
    undone = json.loads(capsys.readouterr().out)
    assert undone["enabled"] is True


def test_delete_requires_yes(isolated, capsys):
    ns, _ = _run(["corrections", "draft", "--scope-kind", "session", "--scope-value", "sess-1",
                 "--type", "style", "--content", "Be terse.", "--json"])
    correction_id = json.loads(capsys.readouterr().out)["id"]
    ns = build_parser().parse_args(["corrections", "delete", correction_id])
    with pytest.raises(CloznError):
        ns.fn(ns)


def test_delete_with_yes_scrubs_content(isolated, capsys):
    ns, _ = _run(["corrections", "draft", "--scope-kind", "session", "--scope-value", "sess-1",
                 "--type", "style", "--content", "Be terse.", "--json"])
    correction_id = json.loads(capsys.readouterr().out)["id"]
    _run(["corrections", "delete", correction_id, "--yes", "--json"])
    deleted = json.loads(capsys.readouterr().out)
    assert "content" not in deleted
    assert deleted["deleted_ts"] is not None


def test_export_round_trips_through_schema(isolated, capsys):
    ns, _ = _run(["corrections", "draft", "--scope-kind", "global_local",
                 "--type", "forbidden_behavior", "--content", "Never do X.", "--json"])
    correction_id = json.loads(capsys.readouterr().out)["id"]
    _run(["corrections", "confirm", correction_id])
    capsys.readouterr()
    _run(["corrections", "export", correction_id])
    exported = json.loads(capsys.readouterr().out)
    assert exported["correction_id"] == correction_id
    assert len(exported["events"]) == 2


def test_show_unknown_id_raises(isolated):
    ns = build_parser().parse_args(["corrections", "show", "corr_" + "0" * 24])
    with pytest.raises(CloznError):
        ns.fn(ns)


def test_draft_rejects_unknown_scope_kind_at_argparse_level():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["corrections", "draft", "--scope-kind", "topic_relevance",
                                   "--type", "style", "--content", "x"])
