"""Model-free Q3-Q8 qualification orchestration tests."""
from __future__ import annotations

import json
import sys
import hashlib

import pytest

from clozn import schemas
from clozn.qualification import pipeline
from clozn.qualification import lab


def _identity():
    return {
        "sha256": "a" * 64,
        "architecture": "llama",
        "hidden_size": 4096,
        "layer_count": 32,
        "vocab_size": 128256,
        "tokenizer_sha256": "b" * 64,
        "chat_template_sha256": "c" * 64,
        "quantization": "Q4_K_M",
        "file_size": 123,
    }


def test_build_run_is_honest_without_live_or_lab_evidence():
    report = pipeline.build_run("fixture.gguf", identity=_identity(), generated_at="2026-08-01T00:00:00Z")
    schemas.validate(report, pipeline.RUN_SCHEMA)
    assert report["claims"]["qualification_status"] == "not_qualified"
    by_id = {step["id"]: step for step in report["steps"]}
    assert by_id["core.identity"]["status"] == "passed"
    assert by_id["core.smoke"]["status"] == "not_run"
    assert report["receipt_sha256"] == pipeline._sha256_json({**report, "receipt_sha256": None})


def test_build_run_can_accept_an_injected_live_probe_and_lab_steps():
    report = pipeline.build_run(
        "fixture.gguf", identity=_identity(), live=True,
        live_smoke=lambda _model: {"status": "passed", "run_id": "run_fixture",
                                   "receipt_shape": "new", "elapsed_ms": 4.0},
        jlens={"status": "passed", "evidence": {"artifact": "jlens"}},
        batteries=[{"id": "basic", "status": "passed"}],
    )
    by_id = {step["id"]: step for step in report["steps"]}
    assert report["claims"]["qualification_status"] == "core_passed"
    assert by_id["core.context_receipt"]["status"] == "passed"
    assert by_id["batteries"]["status"] == "passed"


def test_run_external_step_uses_argv_without_a_shell():
    result = pipeline.run_external_step("fixture", [sys.executable, "-c", "print('ok')"])
    assert result["status"] == "passed"
    assert result["evidence"]["returncode"] == 0
    assert "ok" in result["evidence"]["stdout"]


def test_run_external_step_records_failure():
    result = pipeline.run_external_step("fixture", [sys.executable, "-c", "raise SystemExit(3)"])
    assert result["status"] == "failed"
    assert result["evidence"]["returncode"] == 3


def test_cli_exposes_explicit_run_flag():
    from clozn.cli.main import build_parser
    args = build_parser().parse_args(["qualify", "fixture.gguf", "--run", "--json"])
    assert args.run is True
    assert args.plan is False


def test_battery_resumes_only_an_identical_passed_cell(tmp_path):
    first = lab.run_battery([
        {"id": "one", "argv": [sys.executable, "-c", "print('one')"]},
    ])
    second = lab.run_battery([
        {"id": "one", "argv": [sys.executable, "-c", "print('one')"]},
    ], previous=first)
    assert second["status"] == "passed"
    assert second["artifact"]["cells"]["one"]["resumed"] is True


def test_install_and_rollback_artifact_are_identity_guarded(tmp_path):
    identity = _identity()
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"dial payload")
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = {
        "contract_version": 1,
        "artifact_type": "dials",
        "artifact_version": 1,
        "model": {
            "source_id": "fixture",
            "architecture": identity["architecture"],
            "hidden_size": identity["hidden_size"],
            "layer_count": identity["layer_count"],
            "vocab_size": identity["vocab_size"],
            "tokenizer_sha256": identity["tokenizer_sha256"],
            "compatible_gguf_sha256": [identity["sha256"]],
        },
        "files": {"payload.bin": {
            "bytes": payload.stat().st_size,
            "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
        }},
    }
    (artifact / "payload.bin").write_bytes(payload.read_bytes())
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    transaction = lab.install_artifact(identity, str(artifact), artifact_type="dials",
                                       root=str(tmp_path / "registry"))
    assert transaction["status"] == "installed"
    assert (tmp_path / "registry" / "dials" / identity["sha256"] / "manifest.json").exists()
    assert lab.rollback_artifact(transaction)["status"] == "rolled_back"


def test_install_rejects_artifact_type_path_traversal(tmp_path):
    try:
        lab.install_artifact(_identity(), str(tmp_path), artifact_type="../dials", root=str(tmp_path))
    except ValueError as exc:
        assert "safe path component" in str(exc)
    else:
        raise AssertionError("unsafe artifact type was accepted")


def test_acceptance_fixture_requires_all_supplied_stages_to_pass():
    identity = _identity()
    core = pipeline.build_run(
        "fixture.gguf", identity=identity, live=True,
        live_smoke=lambda _model: {"status": "passed", "run_id": "r", "receipt_shape": "new"},
    )
    accepted = lab.acceptance_fixture(
        model="fixture.gguf", core=core,
        calibration={"status": "passed"}, jlens={"status": "passed"},
        battery={"status": "passed"},
    )
    schemas.validate(accepted, pipeline.RUN_SCHEMA)
    assert accepted["claims"]["qualification_status"] == "core_passed"
    rejected = lab.acceptance_fixture(
        model="fixture.gguf", core=core,
        calibration={"status": "failed"}, jlens={"status": "passed"},
        battery={"status": "passed"},
    )
    assert rejected["claims"]["qualification_status"] == "failed"



def test_acceptance_fixture_rejects_cross_model_stage_receipts():
    identity = _identity()
    core = pipeline.build_run(
        "fixture.gguf", identity=identity, live=True,
        live_smoke=lambda _model: {"status": "passed", "run_id": "r", "receipt_shape": "new"},
    )
    result = lab.acceptance_fixture(
        model="fixture.gguf", core=core,
        calibration={"status": "passed", "model_sha256": "d" * 64},
        jlens={"status": "passed", "model_sha256": identity["sha256"]},
        battery={"status": "passed", "evidence": {"models": []}},
    )
    assert result["claims"]["qualification_status"] == "failed"
    assert result["steps"][1]["evidence"]["identity_mismatch"] == ["d" * 64]




