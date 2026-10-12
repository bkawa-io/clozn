"""tests/test_experiment_suite_cmd.py -- `clozn experiment stats` (Phase 4.4): argparse wiring + the
command's own validation/dispatch over a real clozn.experiment.result.v0 file on disk. The statistics
math itself is covered in clozn/experiments/test_stats.py; this file only proves the CLI plumbing around
it (parsing, error handling, JSON vs text output) is correct."""
from __future__ import annotations

import json

import pytest

from clozn.cli.main import build_parser, CloznError
from clozn.cli.commands.experiment_suite import cmd_stats
from clozn.experiments import stats, suite


def _write_result(tmp_path):
    manifest = suite.validate_manifest({
        "schema_version": suite.MANIFEST_SCHEMA, "name": "cli-check", "seeds": [0, 1],
        "defaults": {}, "baseline_variant": "base",
        "variants": [{"name": "base", "kind": "base"}, {"name": "cand", "kind": "tuned"}],
        "suites": {
            "target": {"cases": [{"name": "c1", "prompt": "p1"}, {"name": "c2", "prompt": "p2"}]},
            "guard": {"cases": [{"name": "g1", "prompt": "p3"}, {"name": "g2", "prompt": "p4"}]},
        },
    })
    cells = []
    for variant in manifest["variants"]:
        name = variant["name"]
        status = "pass" if name == "cand" else "fail"
        for suite_name, case in (("target", "c1"), ("target", "c2"), ("guard", "g1"), ("guard", "g2")):
            for seed in manifest["seeds"]:
                run_id = f"run-{name}-{suite_name}-{case}-{seed}"
                cells.append({"suite": suite_name, "case": case, "variant": name, "variant_kind": variant["kind"],
                              "seed": seed, "status": status, "run_id": run_id, "response": "x",
                              "assertions": [], "min_confidence": None, "receipts": None, "error": None,
                              "run": {"id": run_id, "model": "clozn",
                                      "meta": {"sampler_mode": "greedy", "temperature": 0.0}}})
    result = suite.validate_result({
        "schema_version": suite.RESULT_SCHEMA, "experiment_id": "exp_cli_check", "name": manifest["name"],
        "created_at": "2026-07-22T00:00:00Z", "manifest_sha256": suite._manifest_digest(manifest),
        "suite_fingerprint": suite.suite_fingerprint(manifest, seeds=manifest["seeds"]),
        "manifest": manifest, "seeds": manifest["seeds"], "cells": cells,
        "summary": suite._summarize(cells, "base", ["base", "cand"]),
    })
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    return str(path)


def test_experiment_stats_is_registered_with_expected_defaults():
    ns = build_parser().parse_args(["experiment", "stats", "result.json"])
    assert ns.fn is cmd_stats
    assert ns.alpha == stats.DEFAULT_ALPHA
    assert ns.resamples == stats.DEFAULT_RESAMPLES
    assert ns.seed == 0 and ns.json is False


def test_cmd_stats_prints_the_human_report(tmp_path, capsys):
    ns = build_parser().parse_args(["experiment", "stats", _write_result(tmp_path), "--resamples", "200"])
    assert cmd_stats(ns) == 0
    text = capsys.readouterr().out
    assert "clozn experiment stats exp_cli_check" in text
    assert "comparisons made: 2" in text                    # 1 non-baseline variant x 2 suites
    assert "replay class: bit_identical_greedy" in text
    assert "significant" not in text.lower()


def test_cmd_stats_json_output_matches_the_module_report(tmp_path, capsys):
    path = _write_result(tmp_path)
    ns = build_parser().parse_args(["experiment", "stats", path, "--json", "--seed", "3", "--resamples", "150"])
    assert cmd_stats(ns) == 0
    printed = json.loads(capsys.readouterr().out)
    result = suite.load_result(path)
    expected = stats.stats_report(result, alpha=stats.DEFAULT_ALPHA, n_resamples=150, seed=3)
    assert printed == expected


def test_cmd_stats_rejects_out_of_range_alpha(tmp_path):
    ns = build_parser().parse_args(["experiment", "stats", _write_result(tmp_path), "--alpha", "1.5"])
    with pytest.raises(CloznError, match="alpha"):
        cmd_stats(ns)


def test_cmd_stats_rejects_too_few_resamples(tmp_path):
    ns = build_parser().parse_args(["experiment", "stats", _write_result(tmp_path), "--resamples", "10"])
    with pytest.raises(CloznError, match="resamples"):
        cmd_stats(ns)


def test_cmd_stats_reports_a_missing_or_malformed_result_file(tmp_path):
    ns = build_parser().parse_args(["experiment", "stats", str(tmp_path / "nope.json")])
    with pytest.raises(CloznError, match="could not read experiment result"):
        cmd_stats(ns)
