from __future__ import annotations

import json

import pytest

from clozn.eval import exports
from clozn.experiments import suite
from clozn.cli import main as cli_main
from clozn.cli.commands import experiment_suite as experiment_cli


def _result(statuses=None):
    statuses = statuses or {}
    manifest = suite.validate_manifest({
        "schema_version": suite.MANIFEST_SCHEMA, "name": "portable-eval", "seeds": [0],
        "defaults": {"temperature": 0, "max_tokens": 32}, "baseline_variant": "base",
        "variants": [{"name": "base", "kind": "base", "model": "org/base"},
                     {"name": "candidate", "kind": "tuned", "model": "org/candidate"}],
        "suites": {
            "target": {"cases": [{"name": "capital", "prompt": "Capital of France?",
                                    "expect": {"contains": "Paris", "not_contains": "Berlin"}}]},
            "guard": {"cases": [{"name": "json", "prompt": "Return JSON",
                                   "expect": {"matches": r"^\{.*\}$"}}]},
        },
    })
    cells = []
    for variant in manifest["variants"]:
        for suite_name, case in (("target", "capital"), ("guard", "json")):
            status = statuses.get((variant["name"], suite_name), "pass")
            rid = f"run-{variant['name']}-{suite_name}"
            cell = {"suite": suite_name, "case": case, "variant": variant["name"],
                    "variant_kind": variant["kind"], "seed": 0, "status": status,
                    "run_id": rid, "response": "Paris" if suite_name == "target" else "{}",
                    "assertions": [{"kind": "contains", "pass": status == "pass"}],
                    "receipts": {"stored": True}, "error": None,
                    "run": {"id": rid, "model": variant["model"], "duration_ms": 12,
                            "usage": {"prompt_tokens": 4, "completion_tokens": 2},
                            "identity": {"model_sha256": variant["name"]}}}
            if status == "error":
                cell.update(run_id=None, run=None, response=None, error="worker failed")
            cells.append(cell)
    result = {"schema_version": suite.RESULT_SCHEMA, "experiment_id": "exp_portable",
              "name": manifest["name"], "created_at": "2026-07-21T12:00:00Z",
              "manifest_sha256": suite._manifest_digest(manifest), "manifest": manifest,
              "suite_fingerprint": suite.suite_fingerprint(manifest, seeds=[0]),
              "seeds": [0], "cells": cells,
              "summary": suite._summarize(cells, "base", ["base", "candidate"])}
    return suite.validate_result(result)


def test_eee_bundle_is_deterministic_linked_and_retains_raw_evidence():
    kwargs = dict(source_organization="Example Lab", evaluator_relationship="third_party",
                  retrieved_timestamp=1234)
    one = exports.export_community_bundle(_result(), **kwargs)
    two = exports.export_community_bundle(_result(), **kwargs)
    assert one == two
    assert len(one["aggregates"]) == 2
    assert len(one["evidence"]) == 4
    aggregate = one["aggregates"][0]["record"]
    instance_name = aggregate["detailed_evaluation_results"]["file_path"]
    instance_text = exports.canonical_jsonl(one["instances"][instance_name])
    assert aggregate["schema_version"] == "0.2.2"
    assert aggregate["model_info"]["additional_details"]["model_sha256"] == "base"
    assert aggregate["detailed_evaluation_results"]["checksum"] == exports._sha256_text(instance_text)
    assert all(row["evaluation_id"] == aggregate["evaluation_id"] for row in one["instances"][instance_name])


def test_eee_excludes_error_cells_from_scores_without_losing_evidence():
    bundle = exports.export_community_bundle(
        _result({("candidate", "target"): "error"}),
        source_organization="Example Lab", evaluator_relationship="other", retrieved_timestamp=1234,
    )
    candidate = next(item["record"] for item in bundle["aggregates"] if item["variant"] == "candidate")
    assert [metric["evaluation_name"] for metric in candidate["evaluation_results"]] == ["guard"]
    assert any(issue["code"] == "no_scored_cells" for issue in bundle["issues"])
    assert any(item["cell"]["status"] == "error" for item in bundle["evidence"])


def test_hf_preview_requires_explicit_benchmark_mapping_and_repo_models():
    bundle = exports.export_community_bundle(
        _result(), source_organization="Example Lab", evaluator_relationship="first_party",
        hf_benchmark={"dataset_id": "org/benchmark", "target_task_id": "target", "guard_task_id": "guard"},
        retrieved_timestamp=1234,
    )
    assert len(bundle["hf_community"]) == 2
    assert bundle["hf_community"][0]["records"][0]["dataset"] == {
        "id": "org/benchmark", "task_id": "target"
    }
    with pytest.raises(exports.EvalExportError, match="OWNER/DATASET"):
        exports.export_community_bundle(
            _result(), source_organization="Lab", evaluator_relationship="other",
            hf_benchmark={"dataset_id": "not-a-repo", "target_task_id": "t", "guard_task_id": "g"},
        )


def test_promptfoo_adapter_translates_supported_assertions_and_reports_provider_gap():
    out = exports.export_adapter(_result(), "promptfoo")
    capital = next(test for test in out["config"]["tests"] if test["description"] == "target/capital")
    assert capital["assert"] == [{"type": "contains", "value": "Paris"},
                                  {"type": "not-contains", "value": "Berlin"}]
    assert any(issue["code"] == "provider_required" for issue in out["issues"])
    assert "providers" not in out["config"]


def test_inspect_and_lighteval_exports_are_data_not_fabricated_runner_logs():
    inspect = exports.export_adapter(_result(), "inspect")
    lighteval = exports.export_adapter(_result(), "lighteval")
    assert inspect["format"] == "inspect-samples-json.v1"
    assert len(inspect["records"]) == 2
    assert lighteval["format"] == "lighteval-dataset-json.v1"
    assert any(issue["code"] == "prompt_function_required" for issue in lighteval["issues"])
    assert lighteval["task_spec"]["dataset_path"] is None


def test_invalid_result_and_unknown_adapter_fail_closed():
    broken = _result()
    broken["manifest_sha256"] = "tampered"
    with pytest.raises(exports.EvalExportError, match="manifest_sha256"):
        exports.export_adapter(broken, "promptfoo")
    with pytest.raises(exports.EvalExportError, match="unknown eval adapter"):
        exports.export_adapter(_result(), "mystery")


def test_experiment_export_parser_does_not_change_the_eval_calibration_leaf():
    export_args = cli_main.build_parser().parse_args([
        "experiment", "export", "result.json", "--format", "inspect", "--out", "bundle"
    ])
    eval_args = cli_main.build_parser().parse_args(["eval", "--set", "arith"])
    assert export_args.fn is experiment_cli.cmd_export
    assert export_args.format == "inspect"
    assert eval_args.fn is cli_main.cmd_eval


def test_cli_materializes_local_promptfoo_and_eee_bundles(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_result()), encoding="utf-8")
    promptfoo = cli_main.build_parser().parse_args([
        "experiment", "export", str(result_path), "--format", "promptfoo",
        "--provider", "openai:chat:local", "--out", str(tmp_path / "promptfoo")
    ])
    assert experiment_cli.cmd_export(promptfoo) == 0
    assert (tmp_path / "promptfoo" / "promptfooconfig.json").is_file()
    assert (tmp_path / "promptfoo" / "export-receipt.json").is_file()

    eee = cli_main.build_parser().parse_args([
        "experiment", "export", str(result_path), "--format", "eee", "--out", str(tmp_path / "eee"),
        "--organization", "Example Lab", "--relationship", "third_party", "--retrieved-timestamp", "1234"
    ])
    assert experiment_cli.cmd_export(eee) == 0
    assert len(list((tmp_path / "eee" / "eee").glob("*.json"))) == 2
    assert len(list((tmp_path / "eee" / "eee").glob("*.jsonl"))) == 2


def test_cli_refuses_existing_output_without_force(tmp_path):
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps(_result()), encoding="utf-8")
    out = tmp_path / "existing"
    out.mkdir()
    args = cli_main.build_parser().parse_args([
        "experiment", "export", str(result_path), "--format", "inspect", "--out", str(out)
    ])
    with pytest.raises(cli_main.CloznError, match="already exists"):
        experiment_cli.cmd_export(args)
