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
    assert args.execute is False


def test_add_subparser_parses_replay_execution_flag():
    args = _build_parser().parse_args(["compare-runs", "run_a", "run_b", "--replay", "--execute"])
    assert args.replay is True
    assert args.execute is True


def test_add_subparser_accepts_one_candidate_for_automatic_selection():
    p = _build_parser()
    args = p.parse_args(["compare-runs", "candidate", "--against", "same_session"])
    assert args.run_a == "candidate"
    assert args.run_b is None
    assert args.against == "same_session"


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


def test_replay_execute_uses_available_planner_swaps(monkeypatch, capsys):
    sampling_a = {"sampling": "sample", "temperature": 0.2, "top_p": 0.9, "top_k": 40,
                  "repetition_penalty": 1.1, "seed": 1}
    sampling_b = {**sampling_a, "temperature": 0.8, "seed": 2}
    a = _run("run_a", response="good", messages=[{"role": "system", "content": "setup"},
                                                   {"role": "user", "content": "full"}],
             meta=sampling_a, context_receipt={"delivered": [{"segment_id": "one"},
                                                              {"segment_id": "two"}]})
    b = _run("run_b", response="bad", messages=[{"role": "user", "content": "short"}],
             meta=sampling_b, context_receipt={"delivered": [{"segment_id": "one"}]})
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"run_a": a, "run_b": b}[rid])
    calls = []

    def fake_request(run_a, run_b, **kwargs):
        calls.append((run_a, run_b, kwargs))
        return {
            "schema_version": "clozn.run-change-test.v1",
            "status": "completed",
            "budget": {"runs_used": 2, "max_runs": 4},
            "tests": [{"kind": "context", "status": "observed", "runs_used": 2,
                       "reason": "controlled"}],
        }

    monkeypatch.setattr(cr, "_request_tests", fake_request)
    args = _build_parser().parse_args([
        "compare-runs", "run_a", "run_b", "--replay", "--execute", "--json",
    ])
    cr.cmd_compare_runs(args)
    payload = json.loads(capsys.readouterr().out)
    assert calls and calls[0][0:2] == ("run_a", "run_b")
    assert calls[0][2]["tests"] == ["context", "sampling"]
    assert payload["replay_execution"]["status"] == "completed"


def test_execute_requires_replay(monkeypatch):
    a = _run("run_a")
    b = _run("run_b")
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"run_a": a, "run_b": b}[rid])
    args = _build_parser().parse_args(["compare-runs", "run_a", "run_b", "--execute"])
    with pytest.raises(CloznError, match="requires --replay"):
        cr.cmd_compare_runs(args)


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
    assert cr._label_for("identity.ext.adapter.strength") == "Adapter"
    assert cr._label_for("identity.ext.engine_artifact.build_id") == "Engine"
    assert cr._label_for("identity.ext.machine.cpu") == "Machine"
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


def test_automatic_selection_uses_full_store_and_shows_why(monkeypatch, capsys):
    older = _run("older", identity={"model_sha256": "m" * 64}, recorded_ts=1)
    child = _run("child", identity={"model_sha256": "m" * 64}, recorded_ts=2,
                 parent_run_id="older", source="replay")
    candidate = _run("candidate", identity={"model_sha256": "m" * 64}, recorded_ts=3)
    monkeypatch.setattr(runlog, "get_run", lambda rid: candidate if rid == "candidate" else None)
    monkeypatch.setattr(runlog, "iter_runs", lambda: [candidate, child, older])
    args = _build_parser().parse_args([
        "compare-runs", "candidate", "--against", "previous_compatible",
    ])
    cr.cmd_compare_runs(args)
    out = capsys.readouterr().out
    assert "previous_compatible" in out
    assert "older -> candidate" in out
    assert "child runs were excluded" in out


def test_plan_adds_separate_controlled_artifact_without_calling_gateway(monkeypatch, capsys):
    meta_a = {"sampling": "sample", "temperature": 0.2, "top_p": 0.9, "top_k": 40,
              "repetition_penalty": 1.1, "seed": 1}
    meta_b = {**meta_a, "temperature": 0.8, "seed": 2}
    a = _run("run_a", response="good", messages=[{"role": "user", "content": "full"}], meta=meta_a)
    b = _run("run_b", response="bad", messages=[{"role": "user", "content": "short"}], meta=meta_b)
    monkeypatch.setattr(runlog, "get_run", lambda rid: {"run_a": a, "run_b": b}[rid])
    args = _build_parser().parse_args([
        "compare-runs", "run_a", "run_b", "--test", "context,sampling", "--plan", "--json",
    ])
    cr.cmd_compare_runs(args)
    payload = json.loads(capsys.readouterr().out)
    assert payload["controlled_tests"]["status"] == "planned"
    assert payload["controlled_tests"]["budget"]["runs_used"] == 0
