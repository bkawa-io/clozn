"""tests/test_ci_report_render.py -- clozn/cli/ci_report_render.py (feature 02, GitHub Action for
model-change gating: the Markdown job-summary and JUnit XML renderers).

Both renderers are pure functions of a `clozn ci check` report dict (clozn.ci-report.v1). These tests
build canned report dicts by hand (mirroring tests/test_ci_check.py's own `test_format_ci_report_*`
style) rather than running a live gate, since the whole point of these renderers is that they need
nothing beyond the report -- no model, no gateway, no filesystem, no clozn.schemas import even. A
couple of tests additionally run a real `gate_experiment_result`/`run_gate` report through both
renderers to prove they survive the actual shapes those functions produce, not just a shape this file
imagined.
"""
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

import clozn.cli.commands.ci_check as ci  # noqa: E402
from clozn.cli.ci_report_render import render_job_summary, render_junit_xml  # noqa: E402


# ============================================================================================ canned reports

def _experiment_report(overall="fail"):
    return {
        "schema_version": ci.CI_REPORT_SCHEMA, "mode": "experiment", "overall": overall,
        "reason": "budget violated: target_regressions" if overall == "fail" else None,
        "generated_at": "2026-07-27T00:00:00+00:00", "exit_code": 1 if overall == "fail" else 0,
        "artifact": {"schema_version": "clozn.experiment.result.v0", "experiment_id": "exp_1",
                     "name": "n", "manifest_sha256": "abc", "baseline_variant": "base",
                     "candidate_variants": ["candidate"], "cells": 4},
        "experiment_path": "result.json",
        "checks": {
            "artifact_integrity": {"ran": True, "passed": True, "budget": {"manifest_digest_match": True},
                                   "observed": {"findings": 0}, "reason": None, "worst_offenders": []},
            "execution_errors": {"ran": True, "passed": True,
                                 "budget": {"max_execution_errors": 0, "scope": "whole artifact"},
                                 "observed": {"count": 0}, "reason": None, "worst_offenders": []},
            "target_regressions": {
                "ran": True, "passed": overall != "fail",
                "budget": {"max_target_regressions": 0, "scope": "per candidate"},
                "observed": {"by_variant": {"candidate": 1 if overall == "fail" else 0}},
                "reason": "budget violated by variant(s): candidate" if overall == "fail" else None,
                "worst_offenders": ([{"variant": "candidate", "case": "target-case", "seed": 0,
                                      "baseline_status": "pass", "candidate_status": "fail"}]
                                    if overall == "fail" else []),
            },
            "guard_regressions": {"ran": True, "passed": True,
                                  "budget": {"max_guard_regressions": 0, "scope": "per candidate"},
                                  "observed": {"by_variant": {"candidate": 0}}, "reason": None,
                                  "worst_offenders": []},
            "target_gains": {"ran": True, "passed": True,
                             "budget": {"min_target_gains": 0, "scope": "per candidate"},
                             "observed": {"by_variant": {"candidate": 0}}, "reason": None,
                             "worst_offenders": []},
        },
    }


def _baseline_report(overall="pass"):
    return {
        "schema_version": ci.CI_REPORT_SCHEMA, "mode": "baseline", "overall": overall,
        "reason": "budget violated: golden" if overall == "fail" else None,
        "generated_at": "2026-07-27T00:00:00+00:00", "exit_code": 1 if overall == "fail" else 0,
        "identity": {"model_path": "/models/m.gguf", "model_sha256": "a" * 64},
        "identity_policy": {"pin_model": False, "match": True, "baseline_sha256": "a" * 64,
                            "current_sha256": "a" * 64},
        "live_identity": {"state": "verified", "certified_sha256": "a" * 64, "live_sha256": "a" * 64},
        "baseline_path": "baseline.json", "model": "/models/m.gguf",
        "checks": {
            "golden": {
                "ran": True, "passed": overall != "fail",
                "budget": {"min_pass_rate": 0.9, "which": "all"},
                "observed": {"n": 10, "n_correct": 5 if overall == "fail" else 9,
                            "pass_rate": 0.5 if overall == "fail" else 0.9},
                "reason": "pass_rate 0.5 < budget min_pass_rate 0.9" if overall == "fail" else None,
                "worst_offenders": ([{"q": "q1", "gold": "a", "reply": "b"}] if overall == "fail" else []),
            },
        },
    }


# ============================================================================================= render_job_summary

def test_render_job_summary_experiment_fail_has_table_and_offender():
    out = render_job_summary(_experiment_report("fail"))
    assert "# clozn ci check -- FAIL" in out
    assert "budget violated: target_regressions" in out
    assert "| Case | Role | Seed | Result | Baseline | Candidate | Evidence |" in out
    assert "| target-case | target | 0 | Regression | pass | fail | - |" in out
    assert "[FAIL]" not in out          # this is the Markdown renderer, not format_ci_report's text style
    assert "| target_regressions | FAIL |" in out


def test_render_job_summary_experiment_pass_has_no_offender_rows():
    out = render_job_summary(_experiment_report("pass"))
    assert "# clozn ci check -- PASS" in out
    assert "No target/guard regressions or target gains to report." in out
    assert "| target_regressions | PASS |" in out


def test_render_job_summary_experiment_evidence_column_is_dash_when_run_id_absent():
    """Documented gap (feature 02 plan): experiment-mode worst_offenders carry no run_id. The renderer
    must render '-' honestly rather than inventing a receipt path."""
    out = render_job_summary(_experiment_report("fail"))
    lines = [l for l in out.splitlines() if l.startswith("| target-case")]
    assert len(lines) == 1
    assert lines[0].rstrip().endswith("| - |")


def test_render_job_summary_baseline_shows_model_and_identity():
    out = render_job_summary(_baseline_report("pass"))
    assert "# clozn ci check -- PASS" in out
    assert "Model `/models/m.gguf`" in out
    assert "sha256 " + "a" * 64 in out
    assert "| golden | PASS |" in out


def test_render_job_summary_baseline_identity_mismatch_is_flagged():
    report = _baseline_report("fail")
    report["live_identity"] = {"state": "mismatch", "certified_sha256": "a" * 64, "live_sha256": "b" * 64}
    out = render_job_summary(report)
    assert "## Identity" in out
    assert "NOT the model this report certifies" in out


def test_render_job_summary_experiment_identity_drift_variant_changed():
    report = _experiment_report("pass")
    report["checks"]["artifact_integrity"]["worst_offenders"] = [
        {"kind": "variant_identity_changed", "variant": "candidate",
         "model_sha256_values": ["s1", "s2"]},
    ]
    out = render_job_summary(report)
    assert "## Identity" in out
    assert "variant `candidate` was measured under more than one model sha256: s1, s2" in out


def test_render_job_summary_remediation_command_experiment_includes_budgets():
    report = _experiment_report("fail")
    report["checks"]["target_regressions"]["budget"]["max_target_regressions"] = 2
    out = render_job_summary(report)
    assert "clozn ci check --experiment result.json" in out
    assert "--max-target-regressions 2" in out


def test_render_job_summary_remediation_command_baseline_includes_allow_model_change_when_pinned_mismatch():
    report = _baseline_report("fail")
    report["identity_policy"] = {"pin_model": True, "match": False}
    out = render_job_summary(report)
    assert "clozn ci check --baseline baseline.json /models/m.gguf --allow-model-change" in out


def test_render_job_summary_is_pure_no_mutation_of_input():
    report = _experiment_report("fail")
    before = repr(report)
    render_job_summary(report)
    assert repr(report) == before


def test_render_job_summary_handles_empty_checks_gracefully():
    report = {"schema_version": ci.CI_REPORT_SCHEMA, "mode": "baseline", "overall": "fail",
             "reason": "baseline declares no enabled checks -- nothing to gate on", "checks": {},
             "generated_at": "t", "exit_code": 1}
    out = render_job_summary(report)
    assert "# clozn ci check -- FAIL" in out
    assert "| Check | Result | Reason | Budget | Observed |" in out


# =============================================================================================== render_junit_xml

def test_render_junit_xml_is_well_formed_and_counts_failures():
    xml_text = render_junit_xml(_experiment_report("fail"))
    root = ET.fromstring(xml_text)
    suite = root.find("testsuite")
    assert suite.get("tests") == "5"
    assert suite.get("failures") == "1"
    cases = {c.get("name"): c for c in suite.findall("testcase")}
    assert set(cases) == {"artifact_integrity", "execution_errors", "target_regressions",
                          "guard_regressions", "target_gains"}
    failing = cases["target_regressions"].find("failure")
    assert failing is not None
    assert failing.get("message") == "budget violated by variant(s): candidate"
    assert "max_target_regressions=0" in failing.text
    assert cases["execution_errors"].find("failure") is None


def test_render_junit_xml_all_pass_has_zero_failures_and_no_failure_elements():
    xml_text = render_junit_xml(_experiment_report("pass"))
    root = ET.fromstring(xml_text)
    suite = root.find("testsuite")
    assert suite.get("failures") == "0"
    assert all(case.find("failure") is None for case in suite.findall("testcase"))


def test_render_junit_xml_baseline_mode_uses_its_own_check_names():
    xml_text = render_junit_xml(_baseline_report("fail"))
    root = ET.fromstring(xml_text)
    suite = root.find("testsuite")
    assert suite.get("name") == "clozn ci check (baseline)"
    names = {c.get("name") for c in suite.findall("testcase")}
    assert names == {"golden"}


def test_render_junit_xml_declares_xml_version():
    xml_text = render_junit_xml(_experiment_report("pass"))
    assert xml_text.startswith("<?xml version='1.0' encoding='utf-8'?>")


def test_render_junit_xml_is_pure_no_mutation_of_input():
    report = _baseline_report("fail")
    before = repr(report)
    render_junit_xml(report)
    assert repr(report) == before


# ================================================================= against a REAL gate_experiment_result report

def _real_experiment_result(statuses):
    manifest = {
        "schema_version": "clozn.experiment.v0", "name": "render fixture", "seeds": [0],
        "defaults": {"model": "clozn"}, "baseline_variant": "base",
        "variants": [{"name": "base", "kind": "base"}, {"name": "candidate", "kind": "tuned"}],
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
            run = {"id": run_id, "model": variant, "identity": {"model_sha256": f"sha-{variant}"}}
            cells.append({"suite": suite_name, "case": case_name, "variant": variant,
                          "variant_kind": "base" if variant == "base" else "tuned", "seed": 0,
                          "status": status, "run_id": run_id, "response": "reply", "assertions": [],
                          "error": None, "run": run})
    return {
        "schema_version": ci.EXPERIMENT_RESULT_SCHEMA, "experiment_id": "exp_real", "name": "render fixture",
        "manifest_sha256": ci._experiment_manifest_digest(manifest), "manifest": manifest,
        "seeds": [0], "cells": cells, "summary": {},
    }


def test_renderers_survive_a_real_gate_experiment_result_report():
    statuses = {("target", "base"): "pass", ("target", "candidate"): "fail",
                ("guard", "base"): "pass", ("guard", "candidate"): "pass"}
    report = ci.gate_experiment_result(result=_real_experiment_result(statuses), max_guard_regressions=0)
    md = render_job_summary(report)
    xml_text = render_junit_xml(report)
    assert "target-case" in md
    root = ET.fromstring(xml_text)
    assert root.find("testsuite").get("failures") == "1"


def test_renderers_survive_a_real_run_gate_report(tmp_path, monkeypatch):
    import clozn.runs.identity as identity
    import clozn.cli.formatting as fmt
    monkeypatch.setattr(identity, "_CACHE_PATH", str(tmp_path / "model_hashes.json"))
    monkeypatch.setattr(fmt, "COLOR", False)
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"fake-gguf-bytes")
    baseline = {"pin_model": False, "identity": {}, "checks": {
        "golden": {"enabled": True, "which": "all", "min_pass_rate": 0.9},
        "tiny": {"enabled": False, "files": []}, "diff": {"enabled": False}}}
    monkeypatch.setattr(ci, "run_golden_check", lambda url, which: {
        "n": 10, "n_correct": 5, "pass_rate": 0.5, "wrong": [{"q": "q1", "gold": "a", "reply": "b"}],
        "model": "m", "model_sha256": "s"})
    report = ci.run_gate(baseline=baseline, model_path=str(model_path))
    md = render_job_summary(report)
    xml_text = render_junit_xml(report)
    assert "golden" in md
    root = ET.fromstring(xml_text)
    assert root.find("testsuite").get("name") == "clozn ci check (baseline)"
