"""Model-free privacy and integrity tests for CI receipt bundles."""
from __future__ import annotations

import json
import zipfile

import pytest

from clozn.receipts.ci_bundle import CIBundleError, build_indexed_bundle


def _run(run_id: str, secret: str) -> dict:
    return {
        "id": run_id,
        "created_at": "2026-07-28T00:00:00Z",
        "messages": [{"role": "user", "content": secret}],
        "response": f"private output for {secret}",
        "assembled_messages": [{"role": "user", "content": secret}],
        "identity": {
            "model_sha256": "a" * 64,
            "model_path": f"C:/private/{secret}.gguf",
            "template_fingerprint": "template-v1",
            "ext": {
                "adapter": {
                    "sha256": "b" * 64,
                    "path": f"C:/private/{secret}.lora",
                },
            },
        },
        "meta": {"temperature": 0.8, "seed": 7, "private": secret},
        "context_receipt": {
            "schema_version": "clozn.context-receipt.v1",
            "delivered": [{
                "segment_id": "seg-1",
                "source_type": "message",
                "content": secret,
                "source_label": secret,
                "content_sha256": "c" * 64,
                "included": True,
            }],
            "rendered": {"sha256": "d" * 64, "content": secret},
            "survived": {"final_prompt": secret},
        },
        "output_contract": {
            "requested_schema": {"description": secret, "type": "object"},
            "raw_output": secret,
            "parser_runtime": {"name": "json", "version": "1", "path": secret},
            "outcome": {"status": "parsed", "value": secret},
        },
    }


def _coordinate(variant: str, run_id: str | None) -> dict:
    value = {
        "suite": "target",
        "case": "case-a",
        "variant": variant,
        "seed": 7,
        "status": "pass",
    }
    if run_id:
        value["run_id"] = run_id
        value["privacy"] = "metadata_only"
    else:
        value["evidence_unavailable"] = "cell has no recorded run ID"
    return value


def _documents(secret="TOP SECRET"):
    indexed = _coordinate("candidate", "run-candidate")
    report = {
        "schema_version": "clozn.ci-report.v1",
        "receipt_index": {"privacy": "metadata_only", "entries": [indexed]},
    }
    evidence = {
        "schema_version": "clozn.experiment.result.v0",
        "cells": [{
            **indexed,
            "run": _run("run-candidate", secret),
        }],
    }
    return report, evidence


def test_bundle_contains_only_indexed_metadata_and_no_private_content(tmp_path):
    report, evidence = _documents()
    # This unindexed cell must never be swept into the archive.
    evidence["cells"].append({
        **_coordinate("unindexed", "run-unindexed"),
        "run": _run("run-unindexed", "UNINDEXED SECRET"),
    })
    path = tmp_path / "clozn-receipts.zip"
    result = build_indexed_bundle(report, evidence, str(path))

    assert result["bundled_runs"] == 1
    assert result["privacy"] == "metadata_only"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        assert names[0] == "index.json"
        assert len(names) == 2
        raw_archive = b"".join(archive.read(name) for name in names)
        assert b"TOP SECRET" not in raw_archive
        assert b"UNINDEXED SECRET" not in raw_archive
        assert b"C:/private/" not in raw_archive
        receipt = json.loads(archive.read(names[1]))
        assert receipt["run_id"] == "run-candidate"
        assert receipt["identity"]["model_sha256"] == "a" * 64
        assert "model_path" not in receipt["identity"]
        assert receipt["output_fingerprint"]["sha256"]
        assert "content" not in receipt["context_receipt"]["delivered"][0]


def test_bundle_is_byte_deterministic_for_identical_inputs(tmp_path):
    report, evidence = _documents()
    first, second = tmp_path / "one.zip", tmp_path / "two.zip"
    one = build_indexed_bundle(report, evidence, str(first))
    two = build_indexed_bundle(report, evidence, str(second))
    assert one["sha256"] == two["sha256"]
    assert first.read_bytes() == second.read_bytes()


def test_missing_or_mismatched_embedded_run_is_explicit_not_fabricated(tmp_path):
    report, evidence = _documents()
    evidence["cells"][0]["run"]["id"] = "wrong-run"
    path = tmp_path / "receipts.zip"
    result = build_indexed_bundle(report, evidence, str(path))
    assert result["bundled_runs"] == 0
    assert result["evidence_unavailable"] == 1
    with zipfile.ZipFile(path) as archive:
        assert archive.namelist() == ["index.json"]
        index = json.loads(archive.read("index.json"))
    assert "does not match" in index["entries"][0]["evidence_unavailable"]


def test_unavailable_index_entry_stays_unavailable(tmp_path):
    report, evidence = _documents()
    report["receipt_index"]["entries"] = [_coordinate("candidate", None)]
    evidence["cells"] = []
    result = build_indexed_bundle(report, evidence, str(tmp_path / "receipts.zip"))
    assert result["bundled_runs"] == 0
    assert result["evidence_unavailable"] == 1


def test_bundle_refuses_non_metadata_index_and_duplicate_coordinates(tmp_path):
    report, evidence = _documents()
    report["receipt_index"]["privacy"] = "full"
    with pytest.raises(CIBundleError):
        build_indexed_bundle(report, evidence, str(tmp_path / "bad.zip"))

    report, evidence = _documents()
    report["receipt_index"]["entries"].append(dict(report["receipt_index"]["entries"][0]))
    with pytest.raises(CIBundleError):
        build_indexed_bundle(report, evidence, str(tmp_path / "bad.zip"))
