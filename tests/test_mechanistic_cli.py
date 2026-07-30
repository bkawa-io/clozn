"""test_mechanistic_cli -- MECH-CLI-01: `clozn diff-model --mechanistic --case ...` and
`clozn experiment explain-cell ...` (clozn/cli/commands/mechanistic.py, plus the small additive edits in
clozn/cli/commands/diff_model.py and clozn/cli/commands/experiment_suite.py that wire them in).

Model-free / GPU-free / real-file-free throughout: `pair_compatibility.assess_gguf_pair` and
`clozn.cli.commands.models.resolve_model` are the only two calls anywhere in this path that would touch a
real file, and both are monkeypatched out here exactly like tests/test_pair_compatibility.py monkeypatches
`gguf_identity` -- no real GGUF, no engine, no GPU, ever exercised by this file's own suite. Every target
artifact this suite writes goes to `tmp_path` via `--out`, never `~/.clozn` (see tests/conftest.py's
autouse tripwire).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

import clozn.cli.commands.diff_model as dm  # noqa: E402
import clozn.cli.commands.mechanistic as mech  # noqa: E402
from clozn.cli.main import CloznError, build_parser  # noqa: E402
from clozn.experiments import suite  # noqa: E402

REF_SHA = "a" * 64
CAND_SHA = "b" * 64


def _fake_assess_gguf_pair(path_a, path_b, *, label_a=None, label_b=None, **_kw):
    """Stands in for a real GGUF-header read: always reports the two fixed test shas as compatible,
    regardless of the paths given (the paths are only ever fixture strings in this suite, never real
    files)."""
    return {
        "schema_version": "clozn.pair-compatibility.v1", "generated_at": "2026-07-29T00:00:00Z",
        "model_a": {"sha256": REF_SHA, "label": label_a}, "model_b": {"sha256": CAND_SHA, "label": label_b},
        "tokenizer": {"state": "exact", "method": "hash"}, "template": {"state": "same", "method": "hash"},
        "architecture": {"state": "same"}, "layer_count": {"state": "same"},
        "hidden_size": {"state": "same"}, "vocab_size": {"state": "same"}, "writable_layers": {},
        "verdict": {"overall": "compatible", "reasons": [],
                   "operations": {"per_token_comparison": {"permitted": True, "reason": "ok"},
                                  "residual_transplant": {"permitted": True, "reason": "ok"}}},
    }


@pytest.fixture(autouse=True)
def _no_real_gguf_reads(monkeypatch):
    monkeypatch.setattr(mech.pair_compatibility, "assess_gguf_pair", _fake_assess_gguf_pair)


def _write_result(tmp_path, *, candidate_status="fail", model_path_a="/models/a.gguf",
                  model_path_b="/models/b.gguf"):
    manifest = suite.validate_manifest({
        "schema_version": suite.MANIFEST_SCHEMA, "name": "cli-mech-check", "seeds": [0],
        "defaults": {}, "baseline_variant": "base",
        "variants": [{"name": "base", "kind": "base"}, {"name": "candidate", "kind": "tuned"}],
        "suites": {
            "target": {"cases": [{"name": "c1", "prompt": "p1"}]},
            "guard": {"cases": [{"name": "g1", "prompt": "p2"}]},
        },
    })

    def _run(run_id, response, sha, path):
        return {"id": run_id, "model": "x", "response": response,
               "identity": {"model_sha256": sha, "model_path": path, "captured_at": "now"},
               "trace": {"tokens": list(response), "token_ids": list(range(len(response)))},
               "messages": [{"role": "user", "content": "p1"}]}

    cells = []
    for suite_name, case in (("target", "c1"), ("guard", "g1")):
        cells.append({"suite": suite_name, "case": case, "variant": "base", "variant_kind": "base",
                     "seed": 0, "status": "pass", "run_id": "run-base", "response": "abc",
                     "assertions": [], "min_confidence": None, "receipts": None, "error": None,
                     "run": _run("run-base", "abc", REF_SHA, model_path_a)})
        cells.append({"suite": suite_name, "case": case, "variant": "candidate", "variant_kind": "tuned",
                     "seed": 0, "status": candidate_status, "run_id": "run-candidate", "response": "abx",
                     "assertions": [{"name": "t", "check": "equals", "target": "text", "expected": "abc",
                                    "actual": "abx", "status": "fail", "note": None}]
                     if candidate_status == "fail" else [],
                     "min_confidence": None, "receipts": None, "error": None,
                     "run": _run("run-candidate", "abx", CAND_SHA, model_path_b)})

    result = suite.validate_result({
        "schema_version": suite.RESULT_SCHEMA, "experiment_id": "exp_cli_mech", "name": manifest["name"],
        "created_at": "2026-07-29T00:00:00Z", "manifest_sha256": suite._manifest_digest(manifest),
        "manifest": manifest, "seeds": [0], "cells": cells,
        "summary": suite._summarize(cells, "base", ["base", "candidate"]),
    })
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return str(path)


# ==================================================================================== format_target_report

def _experiment_cell_target():
    return {
        "schema_version": "clozn.mechanistic-target.v1", "target_id": "mechtarget_deadbeef00000000",
        "generated_at": "2026-07-29T00:00:00Z",
        "origin": {"kind": "experiment_cell", "experiment_id": "exp_x", "suite": "target", "case": "c1",
                  "seed": 0, "reference_variant": "base", "candidate_variant": "candidate",
                  "reference_run_id": "run-base", "candidate_run_id": "run-candidate"},
        "behavioral_delta": {"summary": "1 assertion(s) did not pass."},
        "answer_position": {"kind": "token_index", "index": 2},
        "reference_token": {"id": 2, "piece": "c"}, "candidate_token": {"id": 2, "piece": "x"},
        "reference_model": {"filename": "a.gguf", "sha256": REF_SHA},
        "candidate_model": {"filename": "b.gguf", "sha256": CAND_SHA},
        "pair_compatibility": {"schema_version": "clozn.pair-compatibility.v1",
                              "verdict": {"overall": "compatible", "reasons": [], "operations": {}}},
    }


def test_format_target_report_experiment_cell():
    text = mech.format_target_report(_experiment_cell_target())
    assert "mechtarget_deadbeef00000000" in text
    assert "experiment_cell" in text
    assert "target/c1" in text
    assert "'base' (reference)" in text
    assert "'candidate' (candidate)" in text
    assert "1 assertion(s) did not pass." in text
    assert "token index 2" in text


def test_format_target_report_diff_model_position():
    target = dict(_experiment_cell_target())
    target["origin"] = {"kind": "diff_model_position", "run_id": "run-1", "anchor": "reference",
                        "position_index": 7, "label_a": "reference", "label_b": "candidate"}
    text = mech.format_target_report(target)
    assert "diff_model_position" in text
    assert "run=run-1" in text
    assert "position=7" in text


def test_format_target_report_final_response_position_has_no_token_lines():
    target = dict(_experiment_cell_target())
    target = {k: v for k, v in target.items() if k not in ("reference_token", "candidate_token")}
    target["answer_position"] = {"kind": "final_response", "note": "responses are identical"}
    text = mech.format_target_report(target)
    assert "final response" in text
    assert "reference token" not in text


# ==================================================================================== explain-cell (CLI)

def _explain_args(tmp_path, result_path, **overrides):
    p = build_parser()
    argv = ["experiment", "explain-cell", result_path, "--case", "c1", "--variant", "candidate", "--seed", "0",
           "--out", str(tmp_path / "target.json")]
    for key, value in overrides.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return p.parse_args(argv)


def test_explain_cell_is_registered_under_experiment():
    p = build_parser()
    ns = p.parse_args(["experiment", "explain-cell", "result.json", "--case", "c1", "--variant", "candidate"])
    assert ns.fn is mech.cmd_explain_cell
    assert ns.suite == "target"
    assert ns.seed == 0
    assert ns.reference_variant is None
    assert ns.json is False


def test_cmd_explain_cell_writes_target_and_prints_report(tmp_path, capsys):
    result_path = _write_result(tmp_path)
    args = _explain_args(tmp_path, result_path)
    rc = mech.cmd_explain_cell(args)
    assert rc == 0
    out_path = tmp_path / "target.json"
    assert out_path.exists()
    target = json.loads(out_path.read_text(encoding="utf-8"))
    assert target["schema_version"] == "clozn.mechanistic-target.v1"
    assert target["origin"]["case"] == "c1"
    printed = capsys.readouterr().out
    assert str(out_path) in printed
    assert "clozn mechanistic target" in printed


def test_cmd_explain_cell_json_flag_prints_the_target(tmp_path, capsys):
    result_path = _write_result(tmp_path)
    p = build_parser()
    args = p.parse_args(["experiment", "explain-cell", result_path, "--case", "c1", "--variant", "candidate",
                         "--seed", "0", "--out", str(tmp_path / "target.json"), "--json"])
    rc = mech.cmd_explain_cell(args)
    assert rc == 0
    printed = capsys.readouterr().out
    printed_target = json.loads(printed)
    assert printed_target["schema_version"] == "clozn.mechanistic-target.v1"


def test_cmd_explain_cell_refuses_when_candidate_did_not_fail(tmp_path):
    result_path = _write_result(tmp_path, candidate_status="pass")
    args = _explain_args(tmp_path, result_path)
    with pytest.raises(CloznError, match="candidate_not_failing"):
        mech.cmd_explain_cell(args)


def test_cmd_explain_cell_reports_a_missing_result_file(tmp_path):
    p = build_parser()
    args = p.parse_args(["experiment", "explain-cell", str(tmp_path / "nope.json"), "--case", "c1",
                         "--variant", "candidate"])
    with pytest.raises(CloznError, match="could not read experiment result"):
        mech.cmd_explain_cell(args)


def test_cmd_explain_cell_reports_missing_cell(tmp_path):
    result_path = _write_result(tmp_path)
    p = build_parser()
    args = p.parse_args(["experiment", "explain-cell", result_path, "--case", "does-not-exist",
                         "--variant", "candidate", "--out", str(tmp_path / "target.json")])
    with pytest.raises(CloznError, match="expected exactly one cell"):
        mech.cmd_explain_cell(args)


# ============================================================================= diff-model --mechanistic

def _diff_model_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    dm.add_subparser(sub)
    return p


def test_diff_model_mechanistic_flags_are_registered_with_defaults():
    p = _diff_model_parser()
    ns = p.parse_args(["diff-model", "A.gguf", "B.gguf"])
    assert ns.mechanistic is False
    assert ns.case is None
    assert ns.variant == "candidate"
    assert ns.reference_variant is None
    assert ns.seed == 0
    assert ns.out is None


def test_cmd_diff_model_dispatches_to_mechanistic_path_without_booting_an_engine(tmp_path, monkeypatch):
    result_path = _write_result(tmp_path, model_path_a="A.gguf", model_path_b="B.gguf")
    monkeypatch.setattr(mech, "resolve_model", lambda arg: arg)   # no real file resolution needed
    p = _diff_model_parser()
    out_path = tmp_path / "target.json"
    ns = p.parse_args(["diff-model", "A.gguf", "B.gguf", "--mechanistic", "--case",
                       f"{result_path}:target/c1", "--out", str(out_path)])
    rc = dm.cmd_diff_model(ns)
    assert rc == 0
    assert out_path.exists()
    target = json.loads(out_path.read_text(encoding="utf-8"))
    assert target["origin"]["kind"] == "experiment_cell"
    assert target["origin"]["case"] == "c1"


def test_cmd_diff_model_mechanistic_json_output(tmp_path, monkeypatch, capsys):
    result_path = _write_result(tmp_path, model_path_a="A.gguf", model_path_b="B.gguf")
    monkeypatch.setattr(mech, "resolve_model", lambda arg: arg)
    p = _diff_model_parser()
    out_path = tmp_path / "target.json"
    ns = p.parse_args(["diff-model", "A.gguf", "B.gguf", "--mechanistic", "--case",
                       f"{result_path}:target/c1", "--out", str(out_path), "--json"])
    rc = dm.cmd_diff_model(ns)
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["schema_version"] == "clozn.mechanistic-target.v1"


def test_cmd_diff_model_mechanistic_requires_case():
    p = _diff_model_parser()
    ns = p.parse_args(["diff-model", "A.gguf", "B.gguf", "--mechanistic"])
    with pytest.raises(CloznError, match="requires --case"):
        dm.cmd_diff_model(ns)


def test_cmd_diff_model_mechanistic_rejects_a_malformed_case_argument():
    p = _diff_model_parser()
    ns = p.parse_args(["diff-model", "A.gguf", "B.gguf", "--mechanistic", "--case", "result.json-no-colon"])
    with pytest.raises(CloznError, match="RESULT.json:SUITE/CASE"):
        dm.cmd_diff_model(ns)


def test_cmd_diff_model_mechanistic_reports_refusal_from_the_resolver(tmp_path, monkeypatch):
    result_path = _write_result(tmp_path, candidate_status="pass", model_path_a="A.gguf",
                                model_path_b="B.gguf")
    monkeypatch.setattr(mech, "resolve_model", lambda arg: arg)
    p = _diff_model_parser()
    ns = p.parse_args(["diff-model", "A.gguf", "B.gguf", "--mechanistic", "--case",
                       f"{result_path}:target/c1"])
    with pytest.raises(CloznError, match="candidate_not_failing"):
        dm.cmd_diff_model(ns)


def test_cmd_diff_model_without_mechanistic_flag_takes_the_ordinary_path(monkeypatch):
    """A guard against the branch swallowing the ordinary (engine-booting) path: with --mechanistic
    absent, cmd_diff_model must still reach its normal body -- proven here by making the normal body's
    first real call (_import_engine_client) raise something recognizable rather than actually booting."""
    p = _diff_model_parser()
    ns = p.parse_args(["diff-model", "A.gguf", "B.gguf"])

    def _boom():
        raise RuntimeError("ordinary-path-reached")
    monkeypatch.setattr(dm.qc, "_import_engine_client", _boom)
    with pytest.raises(RuntimeError, match="ordinary-path-reached"):
        dm.cmd_diff_model(ns)
