"""tests/test_investigate_experiment_cli.py -- clozn/cli/commands/investigate_experiment.py
(`clozn investigate-experiment RUN_ID ...`).

Registered via CLOZN_AUTOLOAD (docs/SEAMS.md Seam 1); `test_investigate_experiment_is_registered` proves
the marker actually wires it into `build_parser()` end to end. Plan-only throughout: this command never
touches a substrate, so every test here is model-free.
"""
from __future__ import annotations

import json

import pytest

import clozn.cli.commands.investigate_experiment as cmd_mod
from clozn.cli.main import CloznError, build_parser


def _subparser_choices(parser):
    for action in parser._actions:
        if getattr(action, "choices", None) and "investigate-experiment" in action.choices:
            return action.choices
    return {}


RUN = {
    "id": "run_cli_1",
    "messages": [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello there", "source_id": "doc-1"},
    ],
}


def test_investigate_experiment_is_registered():
    assert "investigate-experiment" in _subparser_choices(build_parser())


def test_investigate_experiment_autoload_marker_is_set():
    assert cmd_mod.CLOZN_AUTOLOAD is True


def test_parses_remove_span():
    ns = build_parser().parse_args(["investigate-experiment", "run_1", "--remove-span", "span_abc"])
    assert ns.run_id == "run_1"
    assert ns.remove_span == "span_abc"
    assert ns.fn is cmd_mod.cmd_investigate_experiment


def test_parses_sampler():
    ns = build_parser().parse_args(
        ["investigate-experiment", "run_1", "--sampler", "temperature=0.5,top_k=40"])
    assert ns.sampler == "temperature=0.5,top_k=40"


def test_kind_flags_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "investigate-experiment", "run_1", "--remove-span", "a", "--omit-source", "b",
        ])


def test_at_least_one_kind_flag_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["investigate-experiment", "run_1"])


def test_parse_sampler_rejects_unknown_field():
    with pytest.raises(CloznError, match="unknown sampler field"):
        cmd_mod._parse_sampler("bogus=1")


def test_parse_sampler_rejects_non_numeric_value():
    with pytest.raises(CloznError, match="must be a number"):
        cmd_mod._parse_sampler("temperature=hot")


def test_parse_sampler_requires_at_least_one_pair():
    with pytest.raises(CloznError, match="at least one"):
        cmd_mod._parse_sampler("")


def test_build_intervention_from_each_flag_group():
    ns = build_parser().parse_args(["investigate-experiment", "r", "--remove-span", "span_x"])
    assert cmd_mod._build_intervention(ns) == {"kind": "remove_span", "span_address_id": "span_x"}

    ns = build_parser().parse_args(["investigate-experiment", "r", "--adapter-scale", "0"])
    assert cmd_mod._build_intervention(ns) == {"kind": "adapter_scale", "scale": 0.0}


def test_cmd_investigate_experiment_prints_planned_json(monkeypatch, capsys):
    import clozn.runs.store as runlog
    monkeypatch.setattr(runlog, "get_run", lambda rid: RUN if rid == "run_cli_1" else None)

    ns = build_parser().parse_args([
        "investigate-experiment", "run_cli_1", "--omit-source", "doc-1", "--json",
    ])
    rc = cmd_mod.cmd_investigate_experiment(ns)
    assert rc == 0
    document = json.loads(capsys.readouterr().out)
    assert document["schema_version"] == "clozn.investigation-experiment.v1"
    assert document["phase"] == "planned"
    assert document["plan"]["resolved"]["kind"] == "omit_source"


def test_cmd_investigate_experiment_prints_human_readable_refusal(monkeypatch, capsys):
    import clozn.runs.store as runlog
    monkeypatch.setattr(runlog, "get_run", lambda rid: RUN if rid == "run_cli_1" else None)

    ns = build_parser().parse_args(["investigate-experiment", "run_cli_1", "--adapter-scale", "0"])
    rc = cmd_mod.cmd_investigate_experiment(ns)
    assert rc == 0
    out = capsys.readouterr().out
    assert "phase:        refused" in out
    assert "adapter_rescale_unavailable_in_planner" in out


def test_cmd_investigate_experiment_run_not_found_raises(tmp_path, monkeypatch):
    import clozn.runs.store as runlog
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))

    ns = build_parser().parse_args(["investigate-experiment", "does-not-exist", "--adapter-scale", "0"])
    with pytest.raises(CloznError, match="run not found"):
        cmd_mod.cmd_investigate_experiment(ns)
