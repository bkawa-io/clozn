"""tests/test_ci_check.py -- clozn/cli/commands/ci_check.py (`clozn ci baseline` / `clozn ci check`,
Phase-1 §4.4 "Headless CI gate").

Model-free / GPU-free throughout, mirroring tests/test_diff_model.py's own discipline: `identity_policy_
check` is pure and tested directly; `run_golden_check` is tested against FIXTURE `clozn.eval.golden.
run_and_grade`/`.engine_health` outputs (mirrors tests/test_cli_test_model.py's own monkeypatch target);
`run_tiny_check` is tested against a REAL tiny-test spec file + an isolated run store (mirrors tests/
test_testkit_cli.py's `iso`/`_make_run` fixtures) -- no monkeypatching needed there, since the tiny-test
harness's static checks are already model-free; `build_baseline`/`run_gate`'s handling of all three checks
(including "diff", the one LIVE/GPU primitive) is exercised by monkeypatching the three runner functions
directly (`ci.run_golden_check`/`run_tiny_check`/`run_diff_check`), exactly how tests/test_diff_model.py
monkeypatches `dm.run_direction`. `cmd_diff_model`-equivalent LIVE boot (`run_diff_check` itself actually
booting two engines) is DEFERRED, same as quant_check/diff_model -- never invoked here.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

import clozn.cli.commands.ci_check as ci  # noqa: E402
import clozn.cli.formatting as fmt  # noqa: E402
import clozn.runs.identity as identity  # noqa: E402
import clozn.runs.store as runlog  # noqa: E402
from clozn.cli.main import build_parser  # noqa: E402
from clozn.eval import golden  # noqa: E402


# ==================================================================================================== fixtures

@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate every bit of real machine state this module could otherwise touch: the model-hash cache
    (clozn/runs/identity.py), the run store (clozn/runs/store.py), and color codes in printed output."""
    monkeypatch.setattr(identity, "_CACHE_PATH", str(tmp_path / "model_hashes.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(fmt, "COLOR", False)
    monkeypatch.setattr(fmt, "DIM", "")
    monkeypatch.setattr(fmt, "BOLD", "")
    monkeypatch.setattr(fmt, "RST", "")
    return tmp_path


def _model_file(tmp_path, name="model.gguf", content=b"fake-gguf-bytes-A"):
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _write_spec(tmp_path, spec, name="spec.json"):
    p = tmp_path / name
    p.write_text(json.dumps(spec), encoding="utf-8")
    return str(p)


def _make_run(**overrides):
    """Records a run via clozn.runs.store.record and returns its id (record()'s own return shape --
    a plain str, not a dict; mirrors tests/test_testkit_cli.py's own `_make_run` helper)."""
    rec = dict(source="test", client="pytest", model="m",
              messages=[{"role": "user", "content": "capital of France?"}],
              response="The capital of France is Paris.", finish_reason="stop", started=1.0, ended=1.1)
    rec.update(overrides)
    return runlog.record(**rec)


# ==================================================================================================== argparse

def _subparser_choices(p):
    for a in p._actions:
        if getattr(a, "choices", None) and "ci" in a.choices:
            return a.choices
    return {}


def test_ci_is_registered():
    assert "ci" in _subparser_choices(build_parser())


def test_ci_baseline_defaults():
    ns = build_parser().parse_args(["ci", "baseline", "out.json", "model.gguf"])
    assert ns.out == "out.json" and ns.model == "model.gguf"
    assert ns.url == ci._DEFAULT_URL
    assert ns.which == "all"
    assert ns.no_golden is False
    assert ns.min_pass_rate is None
    assert ns.tiny == []
    assert ns.reference is None
    assert ns.diff_runs == 8
    assert ns.max_argmax_flips_total is None
    assert ns.max_mean_abs_delta_nats is None
    assert ns.pin_model is False
    assert ns.cpu is False
    assert ns.fn is ci.cmd_ci_baseline


def test_ci_baseline_parses_overrides():
    ns = build_parser().parse_args([
        "ci", "baseline", "out.json", "model.gguf", "--url", "http://x:1", "--set", "hard",
        "--no-golden", "--min-pass-rate", "0.8", "--tiny", "a.json", "--tiny", "b.json",
        "--reference", "ref.gguf", "--diff-runs", "4", "--max-argmax-flips-total", "3",
        "--max-mean-abs-delta-nats", "0.05", "--pin-model", "--cpu",
    ])
    assert ns.url == "http://x:1" and ns.which == "hard"
    assert ns.no_golden is True and ns.min_pass_rate == 0.8
    assert ns.tiny == ["a.json", "b.json"]
    assert ns.reference == "ref.gguf" and ns.diff_runs == 4
    assert ns.max_argmax_flips_total == 3 and ns.max_mean_abs_delta_nats == 0.05
    assert ns.pin_model is True and ns.cpu is True


def test_ci_check_defaults():
    ns = build_parser().parse_args(["ci", "check", "--baseline", "b.json", "model.gguf"])
    assert ns.baseline == "b.json" and ns.model == "model.gguf"
    assert ns.experiment is None
    assert ns.url == ci._DEFAULT_URL
    assert ns.allow_model_change is False
    assert ns.report is None
    assert ns.json is False
    assert ns.cpu is False
    assert ns.max_execution_errors == 0
    assert ns.max_target_regressions == 0
    assert ns.max_guard_regressions == 0
    assert ns.min_target_gains == 0
    assert ns.require_run_identity is False
    assert ns.fn is ci.cmd_ci_check


def test_ci_check_experiment_parses_without_model():
    ns = build_parser().parse_args([
        "ci", "check", "--experiment", "result.json", "--max-execution-errors", "1",
        "--max-target-regressions", "2", "--max-guard-regressions", "3",
        "--min-target-gains", "4", "--require-run-identity",
    ])
    assert ns.experiment == "result.json" and ns.baseline is None and ns.model is None
    assert (ns.max_execution_errors, ns.max_target_regressions,
            ns.max_guard_regressions, ns.min_target_gains) == (1, 2, 3, 4)
    assert ns.require_run_identity is True


def test_ci_check_rejects_both_artifact_sources():
    with pytest.raises(SystemExit):
        build_parser().parse_args([
            "ci", "check", "--baseline", "baseline.json", "--experiment", "result.json", "model.gguf",
        ])


def test_ci_check_requires_baseline_flag():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ci", "check", "model.gguf"])


def test_ci_no_subcommand_returns_2(capsys):
    ns = build_parser().parse_args(["ci"])
    rc = ns.fn(ns)
    assert rc == 2
    assert "clozn ci baseline" in capsys.readouterr().out


# =========================================================================================== identity_policy_check

def test_identity_policy_unpinned_mismatch_is_ok():
    out = ci.identity_policy_check({"model_sha256": "aaa"}, False, {"model_sha256": "bbb"}, False)
    assert out["ok"] is True and out["match"] is False


def test_identity_policy_pinned_match_is_ok():
    out = ci.identity_policy_check({"model_sha256": "aaa"}, True, {"model_sha256": "aaa"}, False)
    assert out["ok"] is True and out["match"] is True


def test_identity_policy_pinned_mismatch_refuses():
    out = ci.identity_policy_check({"model_sha256": "aaa"}, True, {"model_sha256": "bbb"}, False)
    assert out["ok"] is False
    assert "aaa" in out["reason"] and "bbb" in out["reason"]
    assert "allow-model-change" in out["reason"]


def test_identity_policy_pinned_mismatch_allowed_with_flag():
    out = ci.identity_policy_check({"model_sha256": "aaa"}, True, {"model_sha256": "bbb"}, True)
    assert out["ok"] is True


def test_identity_policy_pinned_missing_sha_refuses_with_unverifiable_wording():
    out = ci.identity_policy_check({"model_sha256": None}, True, {"model_sha256": "bbb"}, False)
    assert out["ok"] is False
    assert "could not be verified" in out["reason"]


# ====================================================================================== experiment artifact

def _experiment_result(statuses=None, *, identities=True):
    statuses = statuses or {
        ("target", "base"): "fail", ("target", "candidate"): "pass",
        ("guard", "base"): "pass", ("guard", "candidate"): "pass",
    }
    manifest = {
        "schema_version": "clozn.experiment.v0", "name": "ci fixture", "seeds": [0],
        "defaults": {"model": "clozn"}, "baseline_variant": "base",
        "variants": [{"name": "base", "kind": "base"},
                     {"name": "candidate", "kind": "tuned"}],
        "suites": {
            "target": {"cases": [{"name": "target-case", "prompt": "target", "expect": {}}]},
            "guard": {"cases": [{"name": "guard-case", "prompt": "guard", "expect": {}}]},
        },
    }
    cells = []
    for suite_name, case_name in (("target", "target-case"), ("guard", "guard-case")):
        for variant in ("base", "candidate"):
            status = statuses[(suite_name, variant)]
            run_id = f"run_{suite_name}_{variant}"
            run = None if status == "error" else {"id": run_id, "model": variant}
            if run is not None and identities:
                run["identity"] = {"model_sha256": f"sha-{variant}"}
            cells.append({"suite": suite_name, "case": case_name, "variant": variant,
                          "variant_kind": "base" if variant == "base" else "tuned", "seed": 0,
                          "status": status, "run_id": None if status == "error" else run_id,
                          "response": "reply", "assertions": [], "error": "boom" if status == "error" else None,
                          "run": run})
    return {
        "schema_version": ci.EXPERIMENT_RESULT_SCHEMA, "experiment_id": "exp_fixture", "name": "ci fixture",
        "manifest_sha256": ci._experiment_manifest_digest(manifest), "manifest": manifest,
        "seeds": [0], "cells": cells,
        # Deliberately false: the gate must derive comparisons from cells, not trust this summary.
        "summary": {"comparisons": [{"variant": "candidate", "target_gains": [],
                                      "target_regressions": [], "guard_regressions": []}]},
    }


def test_gate_experiment_result_passes_and_recomputes_target_gain():
    report = ci.gate_experiment_result(result=_experiment_result(), min_target_gains=1,
                                       require_run_identity=True)
    assert report["overall"] == "pass"
    assert report["artifact"]["baseline_variant"] == "base"
    assert report["checks"]["target_gains"]["observed"]["by_variant"] == {"candidate": 1}
    assert report["checks"]["artifact_integrity"]["passed"] is True


def test_gate_experiment_result_target_regression_fails_per_candidate():
    statuses = {("target", "base"): "pass", ("target", "candidate"): "fail",
                ("guard", "base"): "pass", ("guard", "candidate"): "pass"}
    report = ci.gate_experiment_result(result=_experiment_result(statuses))
    check = report["checks"]["target_regressions"]
    assert report["overall"] == "fail" and check["passed"] is False
    assert check["observed"]["by_variant"] == {"candidate": 1}
    assert check["worst_offenders"][0]["case"] == "target-case"


def test_gate_experiment_result_guard_regression_includes_unscored():
    statuses = {("target", "base"): "fail", ("target", "candidate"): "pass",
                ("guard", "base"): "pass", ("guard", "candidate"): "unscored"}
    report = ci.gate_experiment_result(result=_experiment_result(statuses))
    assert report["checks"]["guard_regressions"]["passed"] is False
    assert report["checks"]["guard_regressions"]["observed"]["by_variant"] == {"candidate": 1}


def test_gate_experiment_result_budgets_can_allow_regression():
    statuses = {("target", "base"): "pass", ("target", "candidate"): "fail",
                ("guard", "base"): "pass", ("guard", "candidate"): "fail"}
    report = ci.gate_experiment_result(result=_experiment_result(statuses),
                                       max_target_regressions=1, max_guard_regressions=1)
    assert report["overall"] == "pass"


def test_gate_experiment_result_execution_errors_are_bounded():
    statuses = {("target", "base"): "fail", ("target", "candidate"): "error",
                ("guard", "base"): "pass", ("guard", "candidate"): "pass"}
    result = _experiment_result(statuses)
    failed = ci.gate_experiment_result(result=result)
    allowed = ci.gate_experiment_result(result=result, max_execution_errors=1)
    assert failed["checks"]["execution_errors"]["passed"] is False
    assert allowed["checks"]["execution_errors"]["passed"] is True


def test_gate_experiment_result_minimum_target_gain_is_enforced():
    report = ci.gate_experiment_result(result=_experiment_result(), min_target_gains=2)
    assert report["overall"] == "fail"
    assert report["checks"]["target_gains"]["passed"] is False


def test_gate_experiment_result_requires_identity_only_when_requested():
    result = _experiment_result(identities=False)
    permissive = ci.gate_experiment_result(result=result)
    strict = ci.gate_experiment_result(result=result, require_run_identity=True)
    assert permissive["checks"]["artifact_integrity"]["passed"] is True
    assert strict["checks"]["artifact_integrity"]["passed"] is False
    assert strict["checks"]["artifact_integrity"]["worst_offenders"][0]["kind"] == "missing_model_sha256"


def test_gate_experiment_result_rejects_identity_change_within_variant():
    result = _experiment_result()
    candidate_cells = [cell for cell in result["cells"] if cell["variant"] == "candidate"]
    candidate_cells[1]["run"]["identity"]["model_sha256"] = "different-sha"
    report = ci.gate_experiment_result(result=result)
    findings = report["checks"]["artifact_integrity"]["worst_offenders"]
    assert any(item["kind"] == "variant_identity_changed" for item in findings)


def test_gate_experiment_result_manifest_digest_mismatch_fails_integrity():
    result = _experiment_result()
    result["manifest_sha256"] = "tampered"
    report = ci.gate_experiment_result(result=result)
    assert report["overall"] == "fail"
    assert report["checks"]["artifact_integrity"]["worst_offenders"][0]["kind"] == \
        "manifest_sha256_mismatch"


@pytest.mark.parametrize("mutation", ["wrong_schema", "missing_cell", "duplicate_cell", "bad_status"])
def test_gate_experiment_result_rejects_malformed_artifact(mutation):
    result = _experiment_result()
    if mutation == "wrong_schema":
        result["schema_version"] = "something.else"
    elif mutation == "missing_cell":
        result["cells"].pop()
    elif mutation == "duplicate_cell":
        result["cells"].append(dict(result["cells"][0]))
    else:
        result["cells"][0]["status"] = "maybe"
    with pytest.raises(ci.CIExperimentArtifactError):
        ci.gate_experiment_result(result=result)


def test_gate_experiment_result_rejects_negative_budget():
    with pytest.raises(ci.CIExperimentArtifactError, match="non-negative integer"):
        ci.gate_experiment_result(result=_experiment_result(), max_execution_errors=-1)


# ================================================================================================= run_golden_check

def test_run_golden_check_computes_pass_rate_and_wrong_list(monkeypatch):
    rows = [
        {"q": "q1", "gold": "Paris", "reply": "Paris", "correct": True},
        {"q": "q2", "gold": "42", "reply": "41", "correct": False},
        {"q": "q3", "gold": "x", "reply": "x", "correct": True},
    ]
    monkeypatch.setattr(golden, "run_and_grade", lambda url, which, model="clozn": rows)
    monkeypatch.setattr(golden, "engine_health", lambda url, timeout=10.0: {"model": "m.gguf", "model_sha256": "s"})

    out = ci.run_golden_check("http://x", "all")
    assert out["n"] == 3 and out["n_correct"] == 2
    assert out["pass_rate"] == pytest.approx(2 / 3)
    assert out["wrong"] == [{"q": "q2", "gold": "42", "reply": "41"}]
    assert out["model"] == "m.gguf" and out["model_sha256"] == "s"


def test_run_golden_check_zero_probes_pass_rate_none(monkeypatch):
    monkeypatch.setattr(golden, "run_and_grade", lambda url, which, model="clozn": [])
    monkeypatch.setattr(golden, "engine_health", lambda url, timeout=10.0: {})
    out = ci.run_golden_check("http://x", "all")
    assert out["n"] == 0 and out["pass_rate"] is None


# ================================================================================================== run_tiny_check

def test_run_tiny_check_real_spec_and_run_store(iso):
    run_id = _make_run(response="The capital of France is Paris.")
    spec = {"tests": [{"name": "capital-is-paris", "run": run_id,
                       "assert": [{"check": "contains", "value": "Paris"}]}]}
    path = _write_spec(iso, spec)

    out = ci.run_tiny_check(path)
    assert out["status"] == "pass"
    assert out["by_test"] == {"capital-is-paris": "pass"}
    assert out["error"] is None
    assert out["suite"]["tests"][0]["run_id"] == run_id


def test_run_tiny_check_missing_file_is_error():
    out = ci.run_tiny_check("/no/such/file.json")
    assert out["status"] == "error"
    assert "could not read" in out["error"]
    assert "by_test" not in out


def test_run_tiny_check_invalid_json_is_error(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    out = ci.run_tiny_check(str(p))
    assert out["status"] == "error"
    assert "not valid JSON" in out["error"]


def test_run_tiny_check_missing_tests_key_is_error(tmp_path):
    path = _write_spec(tmp_path, {"nope": []})
    out = ci.run_tiny_check(path)
    assert out["status"] == "error"
    assert "'tests' list" in out["error"]


def test_run_tiny_check_failing_assertion(iso):
    run_id = _make_run(response="wrong answer")
    spec = {"tests": [{"name": "t1", "run": run_id,
                       "assert": [{"check": "contains", "value": "Paris"}]}]}
    path = _write_spec(iso, spec)
    out = ci.run_tiny_check(path)
    assert out["status"] == "fail"
    assert out["by_test"] == {"t1": "fail"}


# =============================================================================================== build_baseline

def test_build_baseline_golden_only(iso, monkeypatch):
    model_path = _model_file(iso)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 10, "n_correct": 9, "pass_rate": 0.9, "wrong": [{"q": "q1", "gold": "a", "reply": "b"}],
        "model": "m", "model_sha256": "s",
    })

    baseline = ci.build_baseline(model_path=model_path, which="all", tiny_files=[], reference=None)

    assert baseline["schema_version"] == ci.SCHEMA_VERSION
    assert baseline["pin_model"] is False
    assert baseline["identity"]["model_sha256"] == _sha(b"fake-gguf-bytes-A")
    assert baseline["identity"]["model_path"] == os.path.abspath(model_path)
    g = baseline["checks"]["golden"]
    assert g["enabled"] is True and g["which"] == "all"
    assert g["min_pass_rate"] == pytest.approx(0.9)          # default: this run's own measured rate
    assert g["measured"] == {"n": 10, "n_correct": 9, "pass_rate": 0.9}
    assert baseline["checks"]["tiny"] == {"enabled": False, "files": []}
    assert baseline["checks"]["diff"] == {"enabled": False}


def test_build_baseline_golden_explicit_min_pass_rate_override(iso, monkeypatch):
    model_path = _model_file(iso)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 10, "n_correct": 9, "pass_rate": 0.9, "wrong": [], "model": None, "model_sha256": None,
    })
    baseline = ci.build_baseline(model_path=model_path, min_pass_rate=0.5, tiny_files=[])
    assert baseline["checks"]["golden"]["min_pass_rate"] == 0.5


def test_build_baseline_golden_disabled(iso, monkeypatch):
    model_path = _model_file(iso)
    called = []
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: called.append(1) or {})
    baseline = ci.build_baseline(model_path=model_path, golden_enabled=False, tiny_files=[])
    assert baseline["checks"]["golden"] == {"enabled": False}
    assert not called


def test_build_baseline_tiny_records_passing_test_names(iso, monkeypatch):
    model_path = _model_file(iso)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 0, "n_correct": 0, "pass_rate": None, "wrong": [], "model": None, "model_sha256": None,
    })

    def fake_tiny(path):
        return {"file": path, "status": "fail", "counts": {"pass": 1, "fail": 1},
                "error": None, "by_test": {"t1": "pass", "t2": "fail"}, "suite": {}}

    monkeypatch.setattr(ci, "run_tiny_check", fake_tiny)
    baseline = ci.build_baseline(model_path=model_path, golden_enabled=False, tiny_files=["spec1.json"])
    t = baseline["checks"]["tiny"]
    assert t["enabled"] is True
    assert t["files"][0]["path"] == "spec1.json"
    assert t["files"][0]["baseline_passing_tests"] == ["t1"]


def test_build_baseline_diff_defaults_budgets_to_measured(iso, monkeypatch):
    model_path = _model_file(iso)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 0, "n_correct": 0, "pass_rate": None, "wrong": [], "model": None, "model_sha256": None,
    })
    monkeypatch.setattr(ci, "run_diff_check", lambda reference, candidate, *, runs, cpu: {
        "total_tokens": 500, "total_flipped": 3, "mean_abs_delta_nats_all_mean": 0.01,
        "verdict": "CHANGED", "top_flips": [{"run_id": "r1", "delta_nats": 0.5}],
        "caveat": "some caveat", "topk_note": "some topk note", "label_a": "ref", "label_b": "cand",
    })
    baseline = ci.build_baseline(model_path=model_path, golden_enabled=False, tiny_files=[],
                                 reference="ref.gguf", diff_runs=4)
    d = baseline["checks"]["diff"]
    assert d["enabled"] is True and d["reference"] == "ref.gguf" and d["runs"] == 4
    assert d["max_argmax_flips_total"] == 3
    assert d["max_mean_abs_delta_nats"] == pytest.approx(0.01)
    assert d["measured"]["verdict"] == "CHANGED"


def test_build_baseline_diff_explicit_budget_overrides(iso, monkeypatch):
    model_path = _model_file(iso)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 0, "n_correct": 0, "pass_rate": None, "wrong": [], "model": None, "model_sha256": None,
    })
    monkeypatch.setattr(ci, "run_diff_check", lambda reference, candidate, *, runs, cpu: {
        "total_tokens": 500, "total_flipped": 3, "mean_abs_delta_nats_all_mean": 0.01,
        "verdict": "CHANGED", "top_flips": [], "caveat": None, "topk_note": None,
    })
    baseline = ci.build_baseline(model_path=model_path, golden_enabled=False, tiny_files=[],
                                 reference="ref.gguf", max_argmax_flips_total=10,
                                 max_mean_abs_delta_nats=0.2)
    d = baseline["checks"]["diff"]
    assert d["max_argmax_flips_total"] == 10
    assert d["max_mean_abs_delta_nats"] == pytest.approx(0.2)


# ===================================================================================================== run_gate

def _passing_golden_baseline(model_path, pin_model=False):
    return {"schema_version": 1, "pin_model": pin_model,
           "identity": {"model_sha256": _sha(b"fake-gguf-bytes-A")},
           "checks": {"golden": {"enabled": True, "which": "all", "min_pass_rate": 0.9},
                     "tiny": {"enabled": False, "files": []}, "diff": {"enabled": False}}}


def test_run_gate_golden_passes(iso, monkeypatch):
    model_path = _model_file(iso)
    baseline = _passing_golden_baseline(model_path)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 10, "n_correct": 9, "pass_rate": 0.9, "wrong": [], "model": None, "model_sha256": None,
    })
    report = ci.run_gate(baseline=baseline, model_path=model_path)
    assert report["overall"] == "pass"
    assert report["checks"]["golden"]["passed"] is True


def test_run_gate_golden_budget_violation_fails(iso, monkeypatch):
    model_path = _model_file(iso)
    baseline = _passing_golden_baseline(model_path)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 10, "n_correct": 5, "pass_rate": 0.5, "wrong": [{"q": "q1", "gold": "a", "reply": "b"}],
        "model": None, "model_sha256": None,
    })
    report = ci.run_gate(baseline=baseline, model_path=model_path)
    assert report["overall"] == "fail"
    assert report["checks"]["golden"]["passed"] is False
    assert "0.5" in report["checks"]["golden"]["reason"]
    assert report["checks"]["golden"]["worst_offenders"] == [{"q": "q1", "gold": "a", "reply": "b"}]


def test_run_gate_golden_execution_error_is_failed_not_raised(iso, monkeypatch):
    model_path = _model_file(iso)
    baseline = _passing_golden_baseline(model_path)

    def boom(url, which):
        raise RuntimeError("gateway unreachable")

    monkeypatch.setattr(ci, "run_golden_check", boom)
    report = ci.run_gate(baseline=baseline, model_path=model_path)
    assert report["overall"] == "fail"
    assert report["checks"]["golden"]["ran"] is False
    assert "gateway unreachable" in report["checks"]["golden"]["reason"]


def test_run_gate_no_enabled_checks_fails_not_vacuous_pass(iso):
    model_path = _model_file(iso)
    baseline = {"pin_model": False, "identity": {},
               "checks": {"golden": {"enabled": False}, "tiny": {"enabled": False, "files": []},
                         "diff": {"enabled": False}}}
    report = ci.run_gate(baseline=baseline, model_path=model_path)
    assert report["overall"] == "fail"
    assert "no enabled checks" in report["reason"]


def test_run_gate_identity_refusal_pinned_mismatch(iso, tmp_path):
    model_path = _model_file(iso, content=b"model-B-bytes")
    baseline = _passing_golden_baseline(model_path, pin_model=True)  # baseline sha is for "fake-gguf-bytes-A"
    with pytest.raises(ci.CIIdentityRefusal):
        ci.run_gate(baseline=baseline, model_path=model_path)


def test_run_gate_identity_refusal_allowed_with_flag(iso, monkeypatch):
    model_path = _model_file(iso, content=b"model-B-bytes")
    baseline = _passing_golden_baseline(model_path, pin_model=True)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 10, "n_correct": 9, "pass_rate": 0.9, "wrong": [], "model": None, "model_sha256": None,
    })
    report = ci.run_gate(baseline=baseline, model_path=model_path, allow_model_change=True)
    assert report["overall"] == "pass"


def test_run_gate_identity_no_refusal_when_unpinned(iso, monkeypatch):
    model_path = _model_file(iso, content=b"model-B-bytes")
    baseline = _passing_golden_baseline(model_path, pin_model=False)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 10, "n_correct": 9, "pass_rate": 0.9, "wrong": [], "model": None, "model_sha256": None,
    })
    report = ci.run_gate(baseline=baseline, model_path=model_path)
    assert report["overall"] == "pass"
    assert report["identity_policy"]["match"] is False


def test_run_gate_tiny_regression_fails(iso, monkeypatch):
    model_path = _model_file(iso)
    baseline = {"pin_model": False, "identity": {"model_sha256": _sha(b"fake-gguf-bytes-A")},
               "checks": {"golden": {"enabled": False},
                         "tiny": {"enabled": True, "files": [
                             {"path": "spec1.json", "baseline_passing_tests": ["t1"]}]},
                         "diff": {"enabled": False}}}

    def fake_tiny(path):
        return {"file": path, "status": "fail", "counts": {}, "error": None,
                "by_test": {"t1": "fail"}, "suite": {"tests": [{"name": "t1", "run_id": "r1", "status": "fail"}]}}

    monkeypatch.setattr(ci, "run_tiny_check", fake_tiny)
    report = ci.run_gate(baseline=baseline, model_path=model_path)
    assert report["overall"] == "fail"
    tiny = report["checks"]["tiny"]
    assert tiny["passed"] is False
    assert tiny["worst_offenders"] == [{"file": "spec1.json", "test": "t1", "run_id": "r1",
                                        "reason": "was pass at baseline, now fail"}]


def test_run_gate_tiny_still_passing_ok(iso, monkeypatch):
    model_path = _model_file(iso)
    baseline = {"pin_model": False, "identity": {"model_sha256": _sha(b"fake-gguf-bytes-A")},
               "checks": {"golden": {"enabled": False},
                         "tiny": {"enabled": True, "files": [
                             {"path": "spec1.json", "baseline_passing_tests": ["t1"]}]},
                         "diff": {"enabled": False}}}

    def fake_tiny(path):
        return {"file": path, "status": "pass", "counts": {}, "error": None,
                "by_test": {"t1": "pass"}, "suite": {"tests": [{"name": "t1", "run_id": "r1", "status": "pass"}]}}

    monkeypatch.setattr(ci, "run_tiny_check", fake_tiny)
    report = ci.run_gate(baseline=baseline, model_path=model_path)
    assert report["overall"] == "pass"


def test_run_gate_tiny_file_could_not_run_fails(iso, monkeypatch):
    model_path = _model_file(iso)
    baseline = {"pin_model": False, "identity": {"model_sha256": _sha(b"fake-gguf-bytes-A")},
               "checks": {"golden": {"enabled": False},
                         "tiny": {"enabled": True, "files": [
                             {"path": "spec1.json", "baseline_passing_tests": []}]},
                         "diff": {"enabled": False}}}

    def fake_tiny(path):
        return {"file": path, "status": "error", "error": "could not read spec1.json: nope"}

    monkeypatch.setattr(ci, "run_tiny_check", fake_tiny)
    report = ci.run_gate(baseline=baseline, model_path=model_path)
    assert report["overall"] == "fail"
    assert report["checks"]["tiny"]["passed"] is False


def test_run_gate_diff_within_budget_passes(iso, monkeypatch):
    model_path = _model_file(iso)
    baseline = {"pin_model": False, "identity": {"model_sha256": _sha(b"fake-gguf-bytes-A")},
               "checks": {"golden": {"enabled": False}, "tiny": {"enabled": False, "files": []},
                         "diff": {"enabled": True, "reference": "ref.gguf", "runs": 8,
                                 "max_argmax_flips_total": 5, "max_mean_abs_delta_nats": 0.02}}}
    monkeypatch.setattr(ci, "run_diff_check", lambda reference, candidate, *, runs, cpu: {
        "total_tokens": 200, "total_flipped": 2, "mean_abs_delta_nats_all_mean": 0.01,
        "verdict": "CHANGED", "top_flips": [], "caveat": None, "topk_note": None,
    })
    report = ci.run_gate(baseline=baseline, model_path=model_path)
    assert report["overall"] == "pass"


def test_run_gate_diff_exceeds_flips_budget_fails(iso, monkeypatch):
    model_path = _model_file(iso)
    baseline = {"pin_model": False, "identity": {"model_sha256": _sha(b"fake-gguf-bytes-A")},
               "checks": {"golden": {"enabled": False}, "tiny": {"enabled": False, "files": []},
                         "diff": {"enabled": True, "reference": "ref.gguf", "runs": 8,
                                 "max_argmax_flips_total": 1, "max_mean_abs_delta_nats": 0.02}}}
    monkeypatch.setattr(ci, "run_diff_check", lambda reference, candidate, *, runs, cpu: {
        "total_tokens": 200, "total_flipped": 5, "mean_abs_delta_nats_all_mean": 0.01,
        "verdict": "CHANGED", "top_flips": [{"run_id": "r1", "delta_nats": 1.0}],
        "caveat": "a caveat", "topk_note": None,
    })
    report = ci.run_gate(baseline=baseline, model_path=model_path)
    assert report["overall"] == "fail"
    d = report["checks"]["diff"]
    assert d["passed"] is False
    assert "total_flipped 5 > budget max_argmax_flips_total 1" in d["reason"]
    assert d["caveat"] == "a caveat"


def test_run_gate_diff_exceeds_delta_budget_fails(iso, monkeypatch):
    model_path = _model_file(iso)
    baseline = {"pin_model": False, "identity": {"model_sha256": _sha(b"fake-gguf-bytes-A")},
               "checks": {"golden": {"enabled": False}, "tiny": {"enabled": False, "files": []},
                         "diff": {"enabled": True, "reference": "ref.gguf", "runs": 8,
                                 "max_argmax_flips_total": 10, "max_mean_abs_delta_nats": 0.005}}}
    monkeypatch.setattr(ci, "run_diff_check", lambda reference, candidate, *, runs, cpu: {
        "total_tokens": 200, "total_flipped": 0, "mean_abs_delta_nats_all_mean": 0.05,
        "verdict": "CHANGED", "top_flips": [], "caveat": None, "topk_note": None,
    })
    report = ci.run_gate(baseline=baseline, model_path=model_path)
    assert report["overall"] == "fail"
    assert "mean_abs_delta_nats_all_mean 0.05 > budget max_mean_abs_delta_nats 0.005" in \
        report["checks"]["diff"]["reason"]


def test_run_gate_diff_execution_error_is_failed_not_raised(iso, monkeypatch):
    model_path = _model_file(iso)
    baseline = {"pin_model": False, "identity": {"model_sha256": _sha(b"fake-gguf-bytes-A")},
               "checks": {"golden": {"enabled": False}, "tiny": {"enabled": False, "files": []},
                         "diff": {"enabled": True, "reference": "ref.gguf", "runs": 8,
                                 "max_argmax_flips_total": 10, "max_mean_abs_delta_nats": 0.1}}}

    def boom(reference, candidate, *, runs, cpu):
        raise RuntimeError("no free GPU")

    monkeypatch.setattr(ci, "run_diff_check", boom)
    report = ci.run_gate(baseline=baseline, model_path=model_path)
    assert report["overall"] == "fail"
    assert report["checks"]["diff"]["ran"] is False
    assert "no free GPU" in report["checks"]["diff"]["reason"]


# ================================================================================================ format_ci_report

def test_format_ci_report_renders_overall_and_offenders():
    report = {"overall": "fail", "reason": "budget violated: golden",
             "identity_policy": {"pin_model": False, "match": True, "baseline_sha256": "a" * 20,
                                 "current_sha256": "a" * 20},
             "checks": {"golden": {"passed": False, "reason": "pass_rate 0.5 < budget min_pass_rate 0.9",
                                   "observed": {"n": 10}, "budget": {"min_pass_rate": 0.9},
                                   "worst_offenders": [{"q": "q1"}]}}}
    out = ci.format_ci_report(report)
    assert "FAIL" in out
    assert "budget violated: golden" in out
    assert "[FAIL] golden" in out
    assert "offender: {'q': 'q1'}" in out


# ============================================================================================== cmd_ci_baseline

def _base_args(**overrides):
    base = dict(out=None, model=None, url=ci._DEFAULT_URL, which="all", no_golden=True,
               min_pass_rate=None, tiny=[], reference=None, diff_runs=8,
               max_argmax_flips_total=None, max_mean_abs_delta_nats=None, pin_model=False, cpu=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cmd_ci_baseline_writes_file(iso, capsys):
    model_path = _model_file(iso)
    out_path = str(iso / "baseline.json")
    rc = ci.cmd_ci_baseline(_base_args(out=out_path, model=model_path))
    assert rc == 0
    assert os.path.isfile(out_path)
    with open(out_path, encoding="utf-8") as f:
        written = json.load(f)
    assert written["checks"]["golden"] == {"enabled": False}
    assert "wrote" in capsys.readouterr().out


def test_cmd_ci_baseline_with_golden(iso, monkeypatch):
    model_path = _model_file(iso)
    out_path = str(iso / "baseline.json")
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 4, "n_correct": 4, "pass_rate": 1.0, "wrong": [], "model": "m", "model_sha256": "s",
    })
    rc = ci.cmd_ci_baseline(_base_args(out=out_path, model=model_path, no_golden=False))
    assert rc == 0
    with open(out_path, encoding="utf-8") as f:
        written = json.load(f)
    assert written["checks"]["golden"]["min_pass_rate"] == 1.0


def test_cmd_ci_baseline_bad_model_raises_cloznerror(iso):
    from clozn.cli.main import CloznError
    with pytest.raises(CloznError):
        ci.cmd_ci_baseline(_base_args(out=str(iso / "b.json"), model="totally-not-a-model"))


# =============================================================================================== cmd_ci_check

def _check_args(**overrides):
    base = dict(baseline=None, experiment=None, model=None, url=ci._DEFAULT_URL,
               allow_model_change=False, report=None, github_summary=None, junit_report=None,
               json=False, cpu=False,
               max_execution_errors=0, max_target_regressions=0, max_guard_regressions=0,
               min_target_gains=0, require_run_identity=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cmd_ci_check_exit_0_on_pass(iso, monkeypatch, capsys):
    model_path = _model_file(iso)
    baseline_path = str(iso / "baseline.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(_passing_golden_baseline(model_path), f)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 10, "n_correct": 9, "pass_rate": 0.9, "wrong": [], "model": None, "model_sha256": None,
    })
    rc = ci.cmd_ci_check(_check_args(baseline=baseline_path, model=model_path))
    assert rc == 0
    assert "PASS" in capsys.readouterr().out


def test_cmd_ci_check_exit_1_on_budget_violation(iso, monkeypatch):
    model_path = _model_file(iso)
    baseline_path = str(iso / "baseline.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(_passing_golden_baseline(model_path), f)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 10, "n_correct": 1, "pass_rate": 0.1, "wrong": [], "model": None, "model_sha256": None,
    })
    rc = ci.cmd_ci_check(_check_args(baseline=baseline_path, model=model_path))
    assert rc == 1


def test_cmd_ci_check_exit_2_on_missing_baseline_file(iso, capsys):
    model_path = _model_file(iso)
    rc = ci.cmd_ci_check(_check_args(baseline=str(iso / "nope.json"), model=model_path))
    assert rc == 2
    assert "could not read baseline" in capsys.readouterr().err


def test_cmd_ci_check_exit_2_on_invalid_json_baseline(iso, capsys):
    model_path = _model_file(iso)
    baseline_path = str(iso / "bad.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        f.write("{not json")
    rc = ci.cmd_ci_check(_check_args(baseline=baseline_path, model=model_path))
    assert rc == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_cmd_ci_check_exit_2_on_malformed_baseline_shape(iso, capsys):
    model_path = _model_file(iso)
    baseline_path = str(iso / "bad_shape.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump({"nope": True}, f)
    rc = ci.cmd_ci_check(_check_args(baseline=baseline_path, model=model_path))
    assert rc == 2
    assert "not a `clozn ci baseline` artifact" in capsys.readouterr().err


def test_cmd_ci_check_exit_2_on_unresolvable_model(iso, capsys):
    model_path = _model_file(iso)
    baseline_path = str(iso / "baseline.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(_passing_golden_baseline(model_path), f)
    rc = ci.cmd_ci_check(_check_args(baseline=baseline_path, model="totally-not-a-model"))
    assert rc == 2
    assert "could not resolve model" in capsys.readouterr().err


def test_cmd_ci_check_exit_3_on_identity_refusal(iso, capsys):
    model_path = _model_file(iso, content=b"model-B-bytes")
    baseline_path = str(iso / "baseline.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(_passing_golden_baseline(model_path, pin_model=True), f)   # baseline sha is model-A's
    rc = ci.cmd_ci_check(_check_args(baseline=baseline_path, model=model_path))
    assert rc == 3
    assert "REFUSED" in capsys.readouterr().err


def test_cmd_ci_check_exit_3_writes_refusal_report(iso):
    model_path = _model_file(iso, content=b"model-B-bytes")
    baseline_path = str(iso / "baseline.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(_passing_golden_baseline(model_path, pin_model=True), f)
    report_path = str(iso / "report.json")
    rc = ci.cmd_ci_check(_check_args(baseline=baseline_path, model=model_path, report=report_path))
    assert rc == 3
    with open(report_path, encoding="utf-8") as f:
        written = json.load(f)
    assert written["refused"] is True


def test_cmd_ci_check_allow_model_change_bypasses_refusal(iso, monkeypatch):
    model_path = _model_file(iso, content=b"model-B-bytes")
    baseline_path = str(iso / "baseline.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(_passing_golden_baseline(model_path, pin_model=True), f)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 10, "n_correct": 9, "pass_rate": 0.9, "wrong": [], "model": None, "model_sha256": None,
    })
    rc = ci.cmd_ci_check(_check_args(baseline=baseline_path, model=model_path, allow_model_change=True))
    assert rc == 0


def test_cmd_ci_check_writes_json_report_file(iso, monkeypatch):
    model_path = _model_file(iso)
    baseline_path = str(iso / "baseline.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(_passing_golden_baseline(model_path), f)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 10, "n_correct": 9, "pass_rate": 0.9, "wrong": [], "model": None, "model_sha256": None,
    })
    report_path = str(iso / "report.json")
    rc = ci.cmd_ci_check(_check_args(baseline=baseline_path, model=model_path, report=report_path))
    assert rc == 0
    with open(report_path, encoding="utf-8") as f:
        written = json.load(f)
    assert written["overall"] == "pass"
    assert written["exit_code"] == 0
    assert written["baseline_path"] == baseline_path
    assert written["model"] == os.path.abspath(model_path)


def test_cmd_ci_check_json_stdout(iso, monkeypatch, capsys):
    model_path = _model_file(iso)
    baseline_path = str(iso / "baseline.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(_passing_golden_baseline(model_path), f)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 10, "n_correct": 9, "pass_rate": 0.9, "wrong": [], "model": None, "model_sha256": None,
    })
    rc = ci.cmd_ci_check(_check_args(baseline=baseline_path, model=model_path, json=True))
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["overall"] == "pass"


def test_cmd_ci_check_experiment_exit_0_without_model(iso, capsys):
    path = str(iso / "experiment.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_experiment_result(), f)
    rc = ci.cmd_ci_check(_check_args(experiment=path, min_target_gains=1))
    assert rc == 0
    output = capsys.readouterr().out
    assert "overall: PASS" in output and "experiment: exp_fixture" in output


def test_cmd_ci_check_experiment_exit_1_and_writes_report(iso):
    statuses = {("target", "base"): "pass", ("target", "candidate"): "fail",
                ("guard", "base"): "pass", ("guard", "candidate"): "pass"}
    path, report_path = str(iso / "experiment.json"), str(iso / "report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_experiment_result(statuses), f)
    rc = ci.cmd_ci_check(_check_args(experiment=path, report=report_path))
    assert rc == 1
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["mode"] == "experiment" and report["exit_code"] == 1
    assert report["experiment_path"] == path


def test_cmd_ci_check_experiment_invalid_schema_exit_2(iso, capsys):
    result = _experiment_result()
    result["schema_version"] = "wrong"
    path = str(iso / "experiment.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f)
    rc = ci.cmd_ci_check(_check_args(experiment=path))
    assert rc == 2
    assert "invalid experiment result" in capsys.readouterr().err


def test_cmd_ci_check_experiment_rejects_model_argument(iso, capsys):
    path = str(iso / "experiment.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_experiment_result(), f)
    rc = ci.cmd_ci_check(_check_args(experiment=path, model="model.gguf"))
    assert rc == 2
    assert "only valid with --baseline" in capsys.readouterr().err


def test_cmd_ci_check_baseline_without_model_exit_2(iso, capsys):
    path = str(iso / "baseline.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"checks": {}}, f)
    rc = ci.cmd_ci_check(_check_args(baseline=path))
    assert rc == 2
    assert "MODEL is required" in capsys.readouterr().err


# ================================================================================== clozn.ci-report.v1 (feature 02)
# The report schema itself (clozn/schemas/defs/clozn.ci-report.v1.json) is exercised by fixtures under
# tests/fixtures/schemas/clozn.ci-report.v1/ via tests/test_schema_contracts.py's filesystem walk -- these
# tests instead prove the REAL reports run_gate/gate_experiment_result/cmd_ci_check produce actually
# conform to it (not just a fixture this file imagined), and that the new --github-summary/--junit-report
# flags are wired end to end.

def test_run_gate_report_is_schema_stamped_and_validates(iso, monkeypatch):
    from clozn import schemas
    model_path = _model_file(iso)
    baseline = _passing_golden_baseline(model_path)
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 10, "n_correct": 9, "pass_rate": 0.9, "wrong": [], "model": None, "model_sha256": None,
    })
    report = ci.run_gate(baseline=baseline, model_path=model_path)
    assert report["schema_version"] == ci.CI_REPORT_SCHEMA
    assert report["mode"] == "baseline"
    # run_gate alone never sets exit_code (cmd_ci_check adds it) -- the schema requires it, so a raw
    # run_gate report is expected to fail validation until cmd_ci_check finishes assembling it.
    with pytest.raises(schemas.ValidationError, match="exit_code"):
        schemas.validate(report, ci.CI_REPORT_SCHEMA)
    report["exit_code"] = 0
    schemas.validate(report, ci.CI_REPORT_SCHEMA)      # does not raise


def test_gate_experiment_result_report_is_schema_stamped_and_validates():
    from clozn import schemas
    report = ci.gate_experiment_result(result=_experiment_result(), min_target_gains=1)
    assert report["schema_version"] == ci.CI_REPORT_SCHEMA
    assert report["mode"] == "experiment"
    report["exit_code"] = 0
    schemas.validate(report, ci.CI_REPORT_SCHEMA)      # does not raise


def test_cmd_ci_check_experiment_report_written_by_report_flag_validates(iso):
    from clozn import schemas
    path, report_path = str(iso / "experiment.json"), str(iso / "report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_experiment_result(), f)
    rc = ci.cmd_ci_check(_check_args(experiment=path, min_target_gains=1, report=report_path))
    assert rc == 0
    with open(report_path, encoding="utf-8") as f:
        written = json.load(f)
    schemas.validate(written, ci.CI_REPORT_SCHEMA)      # does not raise


def test_cmd_ci_check_writes_github_summary(iso):
    baseline_path = str(iso / "experiment.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(_experiment_result(), f)
    summary_path = str(iso / "summary.md")
    rc = ci.cmd_ci_check(_check_args(experiment=baseline_path, min_target_gains=1,
                                     github_summary=summary_path))
    assert rc == 0
    with open(summary_path, encoding="utf-8") as f:
        text = f.read()
    assert text.startswith("# clozn ci check -- PASS")
    assert "exp_fixture" in text


def test_cmd_ci_check_writes_junit_report(iso):
    import xml.etree.ElementTree as ET
    baseline_path = str(iso / "experiment.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(_experiment_result(), f)
    junit_path = str(iso / "junit.xml")
    rc = ci.cmd_ci_check(_check_args(experiment=baseline_path, min_target_gains=1,
                                     junit_report=junit_path))
    assert rc == 0
    with open(junit_path, encoding="utf-8") as f:
        root = ET.fromstring(f.read())
    assert root.find("testsuite").get("name") == "clozn ci check (experiment)"


def test_cmd_ci_check_without_the_new_flags_writes_neither_file(iso):
    """--github-summary/--junit-report are opt-in; every existing invocation without them must keep
    working exactly as before (no stray files appear)."""
    baseline_path = str(iso / "experiment.json")
    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(_experiment_result(), f)
    rc = ci.cmd_ci_check(_check_args(experiment=baseline_path, min_target_gains=1))
    assert rc == 0
    assert set(os.listdir(iso)) == {"experiment.json"}
