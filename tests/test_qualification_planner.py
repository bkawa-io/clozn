"""Model-free Q1/Q2 qualification-plan and CLI coverage."""
from __future__ import annotations

import json

import pytest

from clozn import schemas
from clozn.cli.commands import qualification
from clozn.cli.main import CloznError, build_parser
from clozn.qualification import planner


def _header():
    return {
        "path": "/models/demo.gguf",
        "file_size_bytes": 2_000_000_000,
        "bytes_read": 4096,
        "name": "Demo",
        "arch": "llama",
        "quant": "Q4_K_M",
        "quant_source": "general.file_type",
        "context_length": 8192,
        "n_layers": 32,
        "embedding_length": 4096,
        "head_count": 32,
        "head_count_kv": 8,
    }


def _deps(present=False):
    return [
        {"module": "torch", "label": "PyTorch", "present": present, "purpose": "calibration"},
        {"module": "transformers", "label": "Transformers", "present": present, "purpose": "reference probes"},
    ]


def test_build_plan_is_model_free_and_distinguishes_boundaries():
    report = planner.build_plan("demo.gguf", _header(), generated_at="2026-08-01T00:00:00Z",
                               lab_dependencies=_deps(False))

    schemas.validate(report, planner.PLAN_SCHEMA)
    assert report["claims"] == {
        "qualification_status": "not_qualified",
        "generation_performed": False,
        "artifacts_installed": False,
        "note": "This plan is a model-free readiness report, not qualification evidence.",
    }
    assert report["resources"]["disk_bytes"] == 2_000_000_000
    assert {step["boundary"] for step in report["steps"]} == {"product", "lab"}
    assert report["steps"][5]["status"] == "blocked"
    assert "PyTorch" in report["steps"][5]["reason"]


def test_present_lab_dependencies_leave_calibration_planned_not_passed():
    report = planner.build_plan("demo.gguf", _header(), lab_dependencies=_deps(True))
    by_id = {step["id"]: step for step in report["steps"]}
    assert by_id["dials"]["status"] == "planned"
    assert by_id["jlens"]["status"] == "planned"
    assert report["claims"]["qualification_status"] == "not_qualified"


def test_plan_from_model_reads_header_only(monkeypatch):
    seen = []
    monkeypatch.setattr(planner.fit_planner, "gguf_header_from_path",
                        lambda path: seen.append(path) or _header())
    # plan_from_model intentionally discovers the dependency list itself; replace it so this test does
    # not make its result depend on what happens to be installed in the test interpreter.
    monkeypatch.setattr(planner, "_dependency_report", lambda: _deps(False))
    monkeypatch.setattr("clozn.cli.commands.models.resolve_model", lambda value: value)
    report = planner.plan_from_model("/tmp/demo.gguf")
    assert seen == ["/tmp/demo.gguf"]
    assert report["model"]["source"] == "local"


def test_cli_registers_plan_and_writes_json(tmp_path, monkeypatch, capsys):
    parser = build_parser()
    args = parser.parse_args(["qualify", "demo.gguf", "--plan", "--out", str(tmp_path / "plan.json")])
    assert args.fn is qualification.cmd_qualify
    monkeypatch.setattr(qualification.planner, "plan_from_model",
                        lambda *a, **kw: planner.build_plan("demo.gguf", _header(),
                                                            lab_dependencies=_deps(False)))
    assert qualification.cmd_qualify(args) == 0
    output = tmp_path / "plan.json"
    assert output.exists()
    saved = json.loads(output.read_text(encoding="utf-8"))
    assert saved["schema_version"] == planner.PLAN_SCHEMA
    assert "NOT QUALIFIED" in capsys.readouterr().out


def test_cli_refuses_execution_without_plan():
    parser = build_parser()
    args = parser.parse_args(["qualify", "demo.gguf"])
    with pytest.raises(CloznError, match="qualification execution is not available"):
        qualification.cmd_qualify(args)
