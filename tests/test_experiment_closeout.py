from __future__ import annotations

import copy
import io
import json

import pytest

from clozn.cli.commands import ci_check
from clozn.experiments import action_contract, history, promotion, suite
from clozn.server import app as server_app


def _manifest(prompt: str = "target prompt") -> dict:
    return suite.validate_manifest({
        "schema_version": suite.MANIFEST_SCHEMA,
        "name": "closeout",
        "seeds": [0, 1],
        "defaults": {"base_url": "http://127.0.0.1:9999", "model": "clozn", "max_tokens": 32},
        "baseline_variant": "base",
        "variants": [
            {"name": "base", "kind": "base"},
            {"name": "candidate", "kind": "prompt", "system_prompt": "Be exact."},
        ],
        "suites": {
            "target": {"cases": [{"name": "target-case", "prompt": prompt, "expect": {"contains": "ok"}}]},
            "guard": {"cases": [{"name": "guard-case", "messages": [
                {"role": "system", "content": "Stay safe."},
                {"role": "user", "content": "guard prompt"},
            ], "expect": {"not_contains": "bad"}}]},
        },
    })


def _result(*, prompt: str = "target prompt", experiment_id: str = "exp_closeout",
            created_at: str = "2026-07-28T12:00:00Z", secret: str = "") -> dict:
    manifest = _manifest(prompt)
    statuses = {
        ("target", "base", 0): "fail",
        ("target", "base", 1): "fail",
        ("target", "candidate", 0): "pass",
        ("target", "candidate", 1): "fail",
        ("guard", "base", 0): "pass",
        ("guard", "base", 1): "pass",
        ("guard", "candidate", 0): "pass",
        ("guard", "candidate", 1): "pass",
    }
    cells = []
    for role, case_name in (("target", "target-case"), ("guard", "guard-case")):
        case = next(item for item in manifest["suites"][role]["cases"] if item["name"] == case_name)
        messages = copy.deepcopy(
            case.get("messages") or [{"role": "user", "content": case["prompt"]}])
        for variant in manifest["variants"]:
            for seed in manifest["seeds"]:
                name = variant["name"]
                status = statuses[(role, name, seed)]
                run_id = f"run-{role}-{name}-{seed}"
                response = f"ok response {secret}".strip()
                run = {
                    "id": run_id,
                    "model": "clozn",
                    "messages": messages,
                    "response": response,
                    "meta": {"max_tokens": 32, "seed": seed},
                    "identity": {
                        "model_sha256": "a" * 64 if name == "base" else "b" * 64,
                        "engine_build": "engine-1",
                        "ext": {"adapter_sha256": "c" * 64} if name == "candidate" else {},
                    },
                }
                cells.append({
                    "suite": role, "case": case_name, "variant": name,
                    "variant_kind": variant["kind"], "seed": seed, "status": status,
                    "run_id": run_id, "response": response,
                    "assertions": [{"status": status, "check": "fixture"}],
                    "min_confidence": None, "receipts": None, "error": None, "run": run,
                })
    result = {
        "schema_version": suite.RESULT_SCHEMA,
        "experiment_id": experiment_id,
        "name": manifest["name"],
        "created_at": created_at,
        "manifest_sha256": suite._manifest_digest(manifest),
        "suite_fingerprint": suite.suite_fingerprint(manifest, seeds=manifest["seeds"]),
        "manifest": manifest,
        "seeds": manifest["seeds"],
        "cells": cells,
        "summary": suite._summarize(cells, "base", ["base", "candidate"]),
        "vcs": {"commit": "explicit-commit"},
        "artifact_provenance": {"artifact_url": "https://example.invalid/artifact"},
    }
    return suite.validate_result(result)


def _selection(**extra) -> dict:
    return {
        "suite": "target", "case": "target-case", "variant": "candidate", "seed": 0,
        **extra,
    }


def _draft(name: str = "existing") -> dict:
    return {
        "schema_version": "clozn.regression_suite.v1",
        "name": name,
        "state": "draft",
        "cases": [{
            "name": "old-case",
            "prompt": "old",
            "model": "clozn",
            "max_tokens": 8,
            "expect": {"equals": "old"},
            "source": {"run_id": "run-old", "sha256": "d" * 64},
        }],
    }


def test_canonical_fingerprint_ignores_presentation_transport_and_order():
    first = _manifest()
    second = copy.deepcopy(first)
    second["defaults"]["base_url"] = "http://remote.example:1234"
    second["defaults"]["model"] = "C:/different/absolute/model.gguf"
    second["seeds"].reverse()
    second["variants"].reverse()
    second["suites"]["target"]["cases"].reverse()
    second = {key: second[key] for key in reversed(second)}
    assert suite.suite_fingerprint(first) == suite.suite_fingerprint(second)


def test_semantic_edits_and_effective_replicates_change_fingerprint():
    original = suite.suite_fingerprint(_manifest())
    assert suite.suite_fingerprint(_manifest("changed prompt")) != original
    assert suite.suite_fingerprint(_manifest(), seeds=[0, 1, 2]) != original


def test_result_rejects_a_drifted_stored_fingerprint():
    result = _result()
    result["suite_fingerprint"]["sha256"] = "0" * 64
    with pytest.raises(suite.ManifestError, match="suite_fingerprint"):
        suite.validate_result(result)


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz", "bearer_token"),
        ("token sk-abcdefghijklmnopqrstuvwxyz123456", "api_key"),
        ("-----BEGIN PRIVATE KEY-----", "private_key"),
        ("C:\\Users\\alice\\secrets\\model.gguf", "home_path"),
        ("contact person@example.com", "email"),
        ("opaque " + "Z" * 50, "opaque_secret"),
    ],
)
def test_secret_scanner_fixture_coverage(text, kind):
    findings = promotion.scan_case({"prompt": text, "model": "clozn", "expect": {}})
    assert kind in {finding["kind"] for finding in findings}
    assert all(text not in finding["preview"] for finding in findings)


def test_secret_scanner_flags_oversized_documents():
    findings = promotion.scan_case({"prompt": "x" * (promotion.MAX_DOCUMENT_BYTES + 1)})
    assert any(finding["kind"] == "oversized_document" for finding in findings)


def test_preview_applies_only_explicit_redactions_and_scans_the_result(tmp_path):
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    preview = promotion.preview_promotion(
        _result(secret=secret), tmp_path / "suite.json",
        _selection(replacements={secret: "[REDACTED]"}),
    )
    assert secret not in json.dumps(preview["candidate_case"])
    assert preview["redaction_findings"] == []


def test_promotion_preview_is_read_only(tmp_path):
    destination = tmp_path / "missing" / "suite.json"
    preview = promotion.preview_promotion(_result(), destination, _selection())
    assert preview["destination_diff"]["operation"] == "create"
    assert preview["expected_destination_hash"] == promotion.ABSENT_HASH
    assert preview["candidate_case"]["source"]["run_id"] == "run-target-candidate-0"
    assert preview["role"] == "target"
    assert not destination.exists()
    assert not destination.parent.exists()


def test_apply_requires_findings_to_be_reviewed(tmp_path):
    destination = tmp_path / "suite.json"
    result = _result(secret="sk-abcdefghijklmnopqrstuvwxyz123456")
    preview = promotion.preview_promotion(result, destination, _selection())
    assert preview["required_acknowledgements"]
    with pytest.raises(promotion.PromotionServiceError, match="acknowledgement"):
        promotion.apply_promotion(
            result, destination,
            _selection(expected_destination_hash=preview["expected_destination_hash"]),
        )
    assert not destination.exists()


def test_apply_is_atomic_keeps_backup_and_records_transaction(tmp_path):
    destination = tmp_path / "suite.json"
    destination.write_text(json.dumps(_draft()), encoding="utf-8")
    result = _result()
    preview = promotion.preview_promotion(result, destination, _selection(case_name="new-case"))
    transaction = promotion.apply_promotion(
        result, destination,
        _selection(
            case_name="new-case",
            expected_destination_hash=preview["expected_destination_hash"],
            acknowledged_findings=preview["required_acknowledgements"],
        ),
    )
    written = json.loads(destination.read_text(encoding="utf-8"))
    assert [case["name"] for case in written["cases"]] == ["old-case", "new-case"]
    assert transaction["previous_destination_hash"] == preview["expected_destination_hash"]
    assert transaction["destination_sha256"] == promotion.destination_hash(destination)
    assert transaction["backup"] and (tmp_path / (destination.name + ".bak." + transaction["transaction_id"])).exists()
    assert (tmp_path / ".clozn-transactions" / (transaction["transaction_id"] + ".json")).exists()


def test_apply_refuses_destination_drift(tmp_path):
    destination = tmp_path / "suite.json"
    destination.write_text(json.dumps(_draft()), encoding="utf-8")
    preview = promotion.preview_promotion(_result(), destination, _selection())
    changed = _draft("changed-after-preview")
    destination.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(promotion.DestinationDriftError, match="changed"):
        promotion.apply_promotion(
            _result(), destination,
            _selection(
                expected_destination_hash=preview["expected_destination_hash"],
                acknowledged_findings=preview["required_acknowledgements"],
            ),
        )
    assert json.loads(destination.read_text(encoding="utf-8")) == changed


def test_apply_restores_original_when_transaction_record_fails(tmp_path, monkeypatch):
    destination = tmp_path / "suite.json"
    original = json.dumps(_draft()).encode("utf-8")
    destination.write_bytes(original)
    preview = promotion.preview_promotion(_result(), destination, _selection(case_name="new-case"))
    monkeypatch.setattr(
        promotion, "_write_transaction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk full")),
    )
    with pytest.raises(OSError, match="disk full"):
        promotion.apply_promotion(
            _result(), destination,
            _selection(
                case_name="new-case",
                expected_destination_hash=preview["expected_destination_hash"],
                acknowledged_findings=preview["required_acknowledgements"],
            ),
        )
    assert destination.read_bytes() == original


def test_trends_never_mix_incompatible_fingerprints():
    first = _result(experiment_id="exp_one", created_at="2026-07-01T00:00:00Z")
    second = _result(experiment_id="exp_two", created_at="2026-07-02T00:00:00Z")
    incompatible = _result(
        prompt="semantic change", experiment_id="exp_three",
        created_at="2026-07-03T00:00:00Z",
    )
    index = history.build_trend_index([incompatible, second, first])
    assert sorted(len(group["points"]) for group in index["groups"]) == [1, 2]
    compatible = history.select_compatible(index, suite.result_fingerprint(first))
    assert [point["experiment_id"] for point in compatible["points"]] == ["exp_one", "exp_two"]
    point = compatible["points"][0]
    assert point["vcs"] == {"commit": "explicit-commit"}
    assert point["identity"]["adapter_sha256"] == ["c" * 64]
    assert point["replicate_instability"]["coordinate_count"] == 1


def test_ci_preview_contract_and_report_share_fingerprint_cache_identity():
    result = _result()
    preview = action_contract.ci_preview(result, {
        "mode": "verify",
        "result_path": "artifacts/result.json",
        "suite_path": "experiments/suite.json",
        "lockfile_path": "models.lock.json",
        "budgets": {"max_target_regressions": 2, "min_target_gains": 1},
    })
    digest = result["suite_fingerprint"]["sha256"]
    assert digest in preview["cache_key"]
    assert preview["cache_key"] in preview["workflow_yaml"]
    assert "--max-target-regressions 2" in preview["workflow_yaml"]
    assert "clozn model-lock verify models.lock.json" in preview["workflow_yaml"]
    assert "bkawa-io/clozn-action" not in preview["workflow_yaml"]
    report = ci_check.gate_experiment_result(result=result)
    assert report["artifact"]["suite_fingerprint"] == result["suite_fingerprint"]


def test_ci_preview_refuses_paths_outside_checkout():
    with pytest.raises(action_contract.ActionContractError, match="within"):
        action_contract.ci_preview(
            _result(), {"mode": "verify", "result_path": "../result.json", "budgets": {}})


def _dispatch(method: str, path: str, body: dict | None = None):
    raw = json.dumps(body or {}).encode("utf-8")
    handler_type = server_app.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version, handler.command = "HTTP/1.1", method
    getattr(handler, f"do_{method}")()
    head, _, raw_body = handler.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode("latin-1"), json.loads(raw_body)


def test_server_exposes_trends_promotion_and_ci_preview(tmp_path, monkeypatch):
    result = _result()
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    (result_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
    destination_dir = tmp_path / "promotions"
    monkeypatch.setattr(suite, "results_directory", lambda: str(result_dir))
    monkeypatch.setattr(promotion, "promotion_directory", lambda: str(destination_dir))

    head, trends = _dispatch("GET", "/experiment-results/exp_closeout/trends")
    assert "200" in head and len(trends["points"]) == 1

    request = _selection(destination="promoted.json", case_name="promoted-case")
    head, preview = _dispatch(
        "POST", "/experiment-results/exp_closeout/promotion-preview", request)
    assert "200" in head and preview["destination_diff"]["operation"] == "create"
    assert not (destination_dir / "promoted.json").exists()

    head, transaction = _dispatch(
        "POST", "/experiment-results/exp_closeout/promotion-apply", {
            **request,
            "expected_destination_hash": preview["expected_destination_hash"],
            "acknowledged_findings": preview["required_acknowledgements"],
        })
    assert "200" in head and transaction["role"] == "target"
    assert (destination_dir / "promoted.json").exists()

    head, ci = _dispatch("POST", "/experiment-results/exp_closeout/ci-preview", {
        "mode": "verify", "result_path": "artifacts/result.json", "budgets": {},
    })
    assert "200" in head and ci["suite_fingerprint"] == result["suite_fingerprint"]
