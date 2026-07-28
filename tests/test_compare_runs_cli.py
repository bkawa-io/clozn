"""test_compare_runs_cli.py -- model-free unit tests for `clozn/cli/commands/compare_runs.py` (`clozn
compare-runs`, agent roadmap feature 10). No server, no model, no GPU: `clozn.runs.store.get_run` is
monkeypatched with synthetic run records, mirroring tests/test_diff_model.py's own argparse-wiring
discipline (`_build_parser`) and clozn/cli/commands/explain.py's "local journal first" pattern this
command follows.
"""
from __future__ import annotations

import argparse
import json

import pytest

import clozn.cli.commands.compare_runs as cr
import clozn.runs.store as runlog
from clozn.cli.main import CloznError


def _run(rid, **kw):
    rec = {"id": rid, "identity": {}, "meta": {}, "messages": [], "response": "", "context_receipt": {},
          "output_contract": {}, "trace": {}}
    rec.update(kw)
    return rec


# ==================================================================================== add_subparser / argparse

def _build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    cr.add_subparser(sub)
    return p


def test_autoload_marker_is_set():
    assert cr.CLOZN_AUTOLOAD is True


def test_add_subparser_defaults():
    p = _build_parser()
    args = p.parse_args(["compare-runs", "run_a", "run_b"])
    assert args.run_a == "run_a"
    assert args.run_b == "run_b"
    assert args.json is False
    assert args.replay is False
    assert args.fn is cr.cmd_compare_runs


def test_add_subparser_parses_flags():
    p = _build_parser()
    args = p.parse_args(["compare-runs", "run_a", "run_b", "--json", "--replay"])
    assert args.json is True
    assert args.replay is True


def test_add_subparser_requires_both_run_ids():
    import contextlib
    import io
    p = _build_parser()
    with contextlib.redirect_stderr(io.StringIO()):
        with pytest.raises(SystemExit):
            p.parse_args(["compare-runs", "only_one"])


# ============================================================================================= cmd_compare_runs

def test_missing_run_raises_clean_error(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: None)
    args = argparse.Namespace(run_a="run_a", run_b="run_b", json=False, replay=False)
    with pytest.raises(CloznError) as exc:
        cr.cmd_compare_runs(args)
    assert "run_a" in str(exc.value) and "run_b" in str(exc.value)


def test_one_missing_run_is_named_not_the_other(monkeypatch):
    def fake_get_run(rid):
        return _run(rid) if rid == "run_a" else None
    monkeypatch.setattr(runlog, "get_run", fake_get_run)
    args = argparse.Namespace(run_a="run_a", run_b="run_b", json=False, replay=False)
    with pytest.raises(CloznError) as exc:
        cr.cmd_compare_runs(args)
    assert "run_b" in str(exc.value)
    assert "run_a" not in str(exc.value).replace("run_ab", "")   # only the actually-missing id is named


def test_success_prints_human_table(monkeypatch, capsys):
    a = _run("run_a", identity={"model_sha256": "a" * 64})
    b = _run("run_b", identity={"model_sha256": "b" * 64})
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"run_a": a, "run_b": b}[rid])
    args = argparse.Namespace(run_a="run_a", run_b="run_b", json=False, replay=False)
    cr.cmd_compare_runs(args)
    out = capsys.readouterr().out
    assert "compare-runs" in out
    assert "identity.model_sha256" in out
    assert "primary findings" in out


def test_success_json_mode_emits_valid_run_diff_document(monkeypatch, capsys):
    a = _run("run_a", identity={"model_sha256": "a" * 64})
    b = _run("run_b", identity={"model_sha256": "b" * 64})
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"run_a": a, "run_b": b}[rid])
    args = argparse.Namespace(run_a="run_a", run_b="run_b", json=True, replay=False)
    cr.cmd_compare_runs(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema_version"] == "clozn.run-diff.v1"
    from clozn import schemas
    schemas.validate(payload, "clozn.run-diff.v1")   # additionalProperties permissive -> "ok" is fine too


def test_replay_flag_adds_plan_to_json(monkeypatch, capsys):
    a = _run("run_a", meta={"temperature": 0.2})
    b = _run("run_b", meta={"temperature": 0.9})
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"run_a": a, "run_b": b}[rid])
    args = argparse.Namespace(run_a="run_a", run_b="run_b", json=True, replay=True)
    cr.cmd_compare_runs(args)
    payload = json.loads(capsys.readouterr().out)
    assert "replay_plan" in payload
    assert payload["replay_plan"]["run_a"] == "run_a"


def test_replay_flag_adds_section_to_human_output(monkeypatch, capsys):
    a = _run("run_a", meta={"temperature": 0.2})
    b = _run("run_b", meta={"temperature": 0.9})
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"run_a": a, "run_b": b}[rid])
    args = argparse.Namespace(run_a="run_a", run_b="run_b", json=False, replay=True)
    cr.cmd_compare_runs(args)
    out = capsys.readouterr().out
    assert "replay (planned, not executed)" in out


def test_no_replay_flag_omits_replay_section(monkeypatch, capsys):
    a = _run("run_a")
    b = _run("run_b")
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"run_a": a, "run_b": b}[rid])
    args = argparse.Namespace(run_a="run_a", run_b="run_b", json=False, replay=False)
    cr.cmd_compare_runs(args)
    out = capsys.readouterr().out
    assert "replay" not in out.lower()


# =================================================================================================== labels

def test_label_mapping_matches_the_spec_vocabulary():
    assert cr._label_for("identity.model_sha256") == "Model"
    assert cr._label_for("identity.ext.adapter.strength") == "Model"
    assert cr._label_for("generation.temperature") == "Settings"
    assert cr._label_for("context.delivered.messages.count") == "History"
    assert cr._label_for("output.tool_call_status") == "Tools"
    assert cr._label_for("output.finish_reason") == "Output"


# ============================================================================================= format_compare_runs

def test_format_handles_empty_result_without_raising():
    assert "compare-runs" in cr.format_compare_runs({})


def test_format_shows_no_findings_notice_when_findings_are_empty():
    out = cr.format_compare_runs({"run_a": "a", "run_b": "b", "differences": [], "findings": []})
    assert "no classified findings" in out
    assert "none -- these two runs match" in out


def test_format_flags_privacy_limited_runs():
    result = {
        "run_a": "a", "run_b": "b",
        "differences": [{"dimension": "context.delivered.messages.count", "kind": "unavailable",
                         "rank": 6, "evidence": [], "value_a": 4}],
        "findings": [], "privacy_limited": True,
    }
    out = cr.format_compare_runs(result)
    assert "content wasn't" in out or "unavailable" in out
