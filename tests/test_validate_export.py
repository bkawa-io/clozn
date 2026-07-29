"""Model-free contract tests for the three-arm LoRA merged-export validator."""
from __future__ import annotations

import argparse
import json
import os

import pytest

from clozn import schemas
from clozn.cli.commands import adapter as adapter_cmd
from clozn.cli.commands import validate_export as ve


def _model(path: str, sha: str, *, quant: str = "Q8_0", architecture: str = "qwen2") -> dict:
    return {
        "path": path,
        "sha256": sha,
        "file_size": 100,
        "architecture": architecture,
        "hidden_size": 64,
        "layer_count": 8,
        "vocab_size": 1000,
        "tokenizer_sha256": "a" * 64,
        "chat_template_sha256": "b" * 64,
        "quantization": quant,
        "chat_template_present": True,
        "bos_token_id": 1,
        "eos_token_id": 2,
    }


def _inputs(*, merged_arch: str = "qwen2", merged_quant: str = "Q8_0") -> dict:
    return {
        "base": _model("base.gguf", "c" * 64),
        "adapter": {
            "path": os.path.abspath("tune.gguf"),
            "sha256": "d" * 64,
            "file_size": 50,
            "format": "lora_gguf",
            "architecture": "qwen2",
            "metadata": {
                "general.type": "adapter",
                "general.architecture": "qwen2",
                "adapter.type": "lora",
            },
        },
        "merged": _model(
            "merged.gguf", "e" * 64, quant=merged_quant, architecture=merged_arch),
    }


def _engine_identity() -> dict:
    return {
        "entrypoint": "clozn-server",
        "entrypoint_sha256": "f" * 64,
        "discovery_source": "repo_dev_build",
        "backend": "cpu",
    }


def _suite(**overrides) -> dict:
    document = {
        "schema_version": ve.SUITE_VERSION,
        "suite_id": "fixture",
        "seeds": [0],
        "max_tokens": 16,
        "topk": 2,
        "cases": [{"id": "case-a", "prompt": "What changed?"}],
    }
    document.update(overrides)
    return ve.normalize_suite(document)


def _scores(*, argmax: int = 7, logprob: float = -0.2, topk: bool = True) -> list[dict]:
    return [{
        "id": 7,
        "piece": "answer",
        "logprob": logprob,
        "topk": ([{"id": argmax, "piece": "answer" if argmax == 7 else "other",
                   "logprob": -0.1}] if topk else []),
    }]


class FakeRunner:
    def __init__(self, inputs: dict, *, merged_argmax: int = 7,
                 merged_logprob: float = -0.201, merged_topk: bool = True,
                 error: BaseException | None = None):
        self.inputs = inputs
        self.merged_argmax = merged_argmax
        self.merged_logprob = merged_logprob
        self.merged_topk = merged_topk
        self.error = error
        self.calls = []

    def effective_identity(self):
        base = self.inputs["base"]
        merged = self.inputs["merged"]
        common = {"protocol_version": "1.0", "architecture": "qwen2", "device": "cpu"}
        return {
            "base": {**common, "model_sha256": base["sha256"]},
            "adapted": {
                **common,
                "model_sha256": base["sha256"],
                "lora": {
                    "path": self.inputs["adapter"]["path"],
                    "scale": 1.0,
                    "meta": {
                        "adapter.type": "lora",
                        "general.architecture": "qwen2",
                    },
                },
            },
            "merged": {**common, "model_sha256": merged["sha256"]},
        }

    def run_case(self, case, seed):
        self.calls.append((case["id"], seed))
        if self.error:
            raise self.error
        return {
            "continuation": "answer",
            "continuation_ids": [7],
            "scores": {
                "base": _scores(argmax=9, logprob=-1.2),
                "adapted": _scores(argmax=7, logprob=-0.2),
                "merged": _scores(
                    argmax=self.merged_argmax,
                    logprob=self.merged_logprob,
                    topk=self.merged_topk,
                ),
            },
        }


def _receipt(inputs: dict, suite: dict) -> dict:
    return ve.new_receipt(inputs, suite, "1" * 64, _engine_identity())


def test_suite_normalizes_prompt_and_expands_seeds_deterministically():
    suite = _suite(seeds=[7, 3])
    assert suite["cases"][0]["messages"] == [
        {"role": "user", "content": "What changed?"}
    ]
    assert suite["cases"][0]["seeds"] == [7, 3]
    assert suite["cases"][0]["budgets"] == ve.DEFAULT_BUDGETS


@pytest.mark.parametrize("patch,match", [
    ({"suite_id": ""}, "suite_id"),
    ({"seeds": [0, 0]}, "duplicates"),
    ({"cases": []}, "non-empty"),
    ({"schema_version": "wrong"}, "schema_version"),
])
def test_suite_refuses_malformed_inputs(patch, match):
    with pytest.raises(ValueError, match=match):
        _suite(**patch)


def test_equivalent_receipt_uses_adapted_vs_merged_as_primary_and_base_as_control():
    inputs, suite = _inputs(), _suite()
    runner = FakeRunner(inputs)
    receipt = ve.evaluate_with_runner(_receipt(inputs, suite), suite, runner)

    assert receipt["verdict"]["status"] == "equivalent_within_budget"
    assert runner.calls == [("case-a", 0)]
    case = receipt["cases"][0]
    assert case["teacher_forced"]["base_vs_adapted"]["n_flipped"] == 1
    assert case["teacher_forced"]["adapted_vs_merged"]["n_flipped"] == 0
    assert receipt["claims"]["primary_comparison"] == "base_plus_adapter_vs_merged"
    assert receipt["claims"]["base_role"] == "control"
    schemas.validate(receipt)


def test_bad_merge_is_behavioral_mismatch():
    inputs, suite = _inputs(), _suite()
    receipt = ve.evaluate_with_runner(
        _receipt(inputs, suite), suite,
        FakeRunner(inputs, merged_argmax=12, merged_logprob=-1.2))
    assert receipt["verdict"]["status"] == "behavioral_mismatch"
    assert receipt["cases"][0]["status"] == "behavioral_mismatch"
    failed = [a["name"] for a in receipt["cases"][0]["assertions"] if not a["passed"]]
    assert "max_argmax_flips" in failed


def test_unknown_argmax_is_inconclusive_not_a_false_mismatch_or_pass():
    inputs, suite = _inputs(), _suite()
    receipt = ve.evaluate_with_runner(
        _receipt(inputs, suite), suite,
        FakeRunner(inputs, merged_topk=False))
    assert receipt["verdict"]["status"] == "inconclusive"
    assert receipt["cases"][0]["status"] == "inconclusive"


def test_case_execution_error_is_explicit():
    inputs, suite = _inputs(), _suite()
    receipt = ve.evaluate_with_runner(
        _receipt(inputs, suite), suite,
        FakeRunner(inputs, error=RuntimeError("score failed")))
    assert receipt["verdict"]["status"] == "execution_error"
    assert "score failed" in receipt["cases"][0]["error"]


def test_live_template_identity_failure_wins_over_behavioral_verdict():
    inputs, suite = _inputs(), _suite()
    receipt = ve.evaluate_with_runner(
        _receipt(inputs, suite), suite,
        FakeRunner(inputs, error=ve.ExportIdentityError("rendered prompt mismatch")))
    assert receipt["verdict"]["status"] == "identity_mismatch"
    assert receipt["cases"][0]["status"] == "identity_mismatch"


def test_static_architecture_mismatch_fails_before_runner_cases():
    inputs, suite = _inputs(merged_arch="llama"), _suite()
    runner = FakeRunner(inputs)
    receipt = ve.evaluate_with_runner(_receipt(inputs, suite), suite, runner)
    assert receipt["verdict"]["status"] == "identity_mismatch"
    assert runner.calls == []
    assert "architecture" in [
        check["name"] for check in receipt["preflight"]["checks"]
        if check["status"] == "failed"
    ]


def test_wrong_effective_adapter_identity_fails_before_cases():
    inputs, suite = _inputs(), _suite()
    runner = FakeRunner(inputs)
    health = runner.effective_identity()
    health["adapted"]["lora"]["path"] = "wrong.gguf"
    runner.effective_identity = lambda: health
    receipt = ve.evaluate_with_runner(_receipt(inputs, suite), suite, runner)
    assert receipt["verdict"]["status"] == "identity_mismatch"
    assert runner.calls == []
    assert "adapter_loaded" in [
        check["name"] for check in receipt["preflight"]["checks"]
        if check["status"] == "failed"
    ]


def test_different_quantizations_are_never_called_byte_equivalent():
    inputs, suite = _inputs(merged_quant="Q4_K_M"), _suite()
    receipt = ve.evaluate_with_runner(
        _receipt(inputs, suite), suite, FakeRunner(inputs))
    assert receipt["verdict"]["status"] == "equivalent_within_budget"
    assert receipt["claims"]["byte_equivalent"] is False
    assert "different declared quantization" in receipt["claims"]["quantization_note"]


def test_receipt_records_hashes_not_suite_prompt_or_generated_text():
    inputs, suite = _inputs(), _suite()
    receipt = ve.evaluate_with_runner(
        _receipt(inputs, suite), suite, FakeRunner(inputs))
    encoded = json.dumps(receipt)
    assert "What changed?" not in encoded
    assert '"answer"' not in encoded
    assert receipt["cases"][0]["messages_sha256"]
    assert receipt["cases"][0]["continuation_sha256"]


def test_write_receipt_is_valid_and_leaves_no_temp_file(tmp_path):
    inputs, suite = _inputs(), _suite()
    receipt = ve.evaluate_with_runner(
        _receipt(inputs, suite), suite, FakeRunner(inputs))
    destination = tmp_path / "nested" / "receipt.json"
    assert ve.write_receipt(str(destination), receipt) == str(destination)
    schemas.validate(json.loads(destination.read_text(encoding="utf-8")))
    assert list(destination.parent.glob("*.tmp")) == []


def test_mark_startup_error_still_produces_a_valid_receipt():
    inputs, suite = _inputs(), _suite()
    receipt = _receipt(inputs, suite)
    receipt["preflight"] = {
        "status": "passed",
        "checks": ve.static_preflight(inputs, _engine_identity()),
    }
    ve.mark_execution_error(receipt, RuntimeError("engine boot failed"))
    assert receipt["verdict"]["status"] == "execution_error"
    assert receipt["summary"]["execution_errors"] == 1
    schemas.validate(receipt)


def test_missing_engine_identity_can_be_recorded_as_execution_error():
    inputs, suite = _inputs(), _suite()
    receipt = ve.new_receipt(inputs, suite, "1" * 64, {})
    receipt["preflight"] = {
        "status": "failed",
        "checks": ve.static_preflight(inputs, {}),
    }
    ve.mark_execution_error(receipt, RuntimeError("no engine found"))
    assert "engine" not in receipt["producer"]
    assert receipt["verdict"]["status"] == "execution_error"
    assert "engine_artifact" in [
        check["name"] for check in receipt["preflight"]["checks"]
        if check["status"] == "failed"
    ]
    schemas.validate(receipt)


def test_validate_export_parser_contract():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command")
    ve.add_subparser(sub)
    args = parser.parse_args([
        "validate-export", "base.gguf",
        "--adapter", "tune.gguf",
        "--merged", "merged.gguf",
        "--suite", "export.json",
        "--out", "receipt.json",
        "--cpu", "--json",
    ])
    assert args.base == "base.gguf"
    assert args.adapter == "tune.gguf"
    assert args.merged == "merged.gguf"
    assert args.suite == "export.json"
    assert args.out == "receipt.json"
    assert args.cpu is True and args.json is True
    assert args.fn is ve.cmd_validate_export


def test_adapter_validate_reuses_static_identity_and_base_architecture(monkeypatch):
    monkeypatch.setattr(adapter_cmd, "inspect_adapter", lambda path: {
        "path": path,
        "sha256": "a" * 64,
        "file_size": 1,
        "format": "lora_gguf",
        "architecture": "qwen2",
    })
    report = adapter_cmd.validate_adapter(
        "tune.gguf", {"architecture": "qwen2"})
    assert report["valid"] is True
    assert report["conversion"]["tool_commit"] == adapter_cmd.LLAMA_CPP_COMMIT
    assert "convert_lora_to_gguf.py" in report["conversion"]["command"]


def test_adapter_validate_fails_non_lora_without_importing_optional_ml(monkeypatch):
    monkeypatch.setattr(adapter_cmd, "inspect_adapter", lambda path: {
        "path": path,
        "sha256": "a" * 64,
        "file_size": 1,
        "format": "unsupported_gguf",
        "architecture": "qwen2",
    })
    assert adapter_cmd.validate_adapter("ordinary.gguf")["valid"] is False


def test_inspect_adapter_recognizes_only_declared_lora_gguf(tmp_path, monkeypatch):
    path = tmp_path / "adapter.gguf"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(ve.contracts, "sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(ve, "gguf_header_from_path", lambda _path: {"metadata": {
        "general.type": "adapter",
        "general.architecture": "qwen2",
        "adapter.type": "lora",
        "adapter.lora.alpha": 8.0,
        "unrelated.large.value": "not retained",
    }})
    identity = ve.inspect_adapter(str(path))
    assert identity["format"] == "lora_gguf"
    assert identity["architecture"] == "qwen2"
    assert identity["metadata"]["adapter.lora.alpha"] == 8.0
    assert "unrelated.large.value" not in identity["metadata"]


def test_inspect_adapter_rejects_ordinary_model_metadata(tmp_path, monkeypatch):
    path = tmp_path / "model.gguf"
    path.write_bytes(b"fixture")
    monkeypatch.setattr(ve.contracts, "sha256_file", lambda _path: "a" * 64)
    monkeypatch.setattr(ve, "gguf_header_from_path", lambda _path: {"metadata": {
        "general.type": "model",
        "general.architecture": "qwen2",
    }})
    assert ve.inspect_adapter(str(path))["format"] == "unsupported_gguf"
