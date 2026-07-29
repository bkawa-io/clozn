"""Model-free tests for exact GGUF and lab-to-product artifact contracts."""
from __future__ import annotations

import hashlib
import json

import pytest

from clozn.artifacts import contracts
from clozn.cli.commands.models import KNOWN, PULLABLE


def _header(path):
    return {
        "name": "Fixture 3B Instruct",
        "arch": "fixture",
        "embedding_length": 64,
        "n_layers": 8,
        "quant": "Q4_K_M",
        "file_size_bytes": path.stat().st_size,
        "metadata": {
            "tokenizer.ggml.model": "fixture-bpe",
            "tokenizer.ggml.tokens": ["a", "b", "c"],
            "tokenizer.chat_template": "{{ messages }}",
        },
    }


@pytest.fixture
def model(tmp_path, monkeypatch):
    path = tmp_path / "fixture-Q4_K_M.gguf"
    path.write_bytes(b"exact gguf bytes")
    monkeypatch.setattr(contracts, "gguf_header_from_path", lambda _path: _header(path))
    return path, contracts.gguf_identity(path)


def _manifest(identity, payload, **model_overrides):
    target = {
        "source_id": "owner/fixture-3b-instruct",
        "architecture": identity["architecture"],
        "hidden_size": identity["hidden_size"],
        "layer_count": identity["layer_count"],
        "vocab_size": identity["vocab_size"],
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "compatible_gguf_sha256": [identity["sha256"]],
    }
    target.update(model_overrides)
    return {
        "contract_version": 1,
        "artifact_type": "jlens",
        "artifact_version": 1,
        "model": target,
        "files": {
            payload.name: {
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        },
    }


def _chat_io_manifest(identity, directory, *, template_fingerprint="b" * 64,
                      features=None, evidence_overrides=None):
    features = features or ["tools", "json_object", "json_schema"]
    schema_subsets = {}
    if "tools" in features:
        schema_subsets["tool_parameters"] = contracts.CHAT_IO_JSON_SCHEMA_SUBSET_ID
    if "json_schema" in features:
        schema_subsets["json_schema"] = contracts.CHAT_IO_JSON_SCHEMA_SUBSET_ID
    evidence = {
        "schema_version": contracts.CHAT_IO_EVIDENCE_SCHEMA,
        "suite_id": contracts.CHAT_IO_QUALIFICATION_SUITE_ID,
        "model_sha256": identity["sha256"],
        "template_fingerprint": template_fingerprint,
        "pipeline": dict(contracts.CHAT_IO_PIPELINE),
        "features": features,
        "schema_subsets": schema_subsets,
        "results": {
            name: {"passed": 1, "failed": 0}
            for name in ["pipeline", *features]
        },
    }
    evidence.update(evidence_overrides or {})
    encoded = json.dumps(evidence, sort_keys=True).encode("utf-8")
    payload = directory / "qualification-evidence.json"
    payload.write_bytes(encoded)
    digest = hashlib.sha256(encoded).hexdigest()
    return {
        "contract_version": contracts.CONTRACT_VERSION,
        "artifact_type": contracts.CHAT_IO_ARTIFACT_TYPE,
        "artifact_version": contracts.CHAT_IO_ARTIFACT_VERSION,
        "model": {
            "source_id": "fixture-only",
            "architecture": identity["architecture"],
            "hidden_size": identity["hidden_size"],
            "layer_count": identity["layer_count"],
            "vocab_size": identity["vocab_size"],
            "tokenizer_sha256": identity["tokenizer_sha256"],
            "compatible_gguf_sha256": [identity["sha256"]],
        },
        "profile": {
            "template_fingerprint": template_fingerprint,
            "pipeline": dict(contracts.CHAT_IO_PIPELINE),
            "features": features,
            "schema_subsets": schema_subsets,
            "evidence": {"path": payload.name, "sha256": digest},
        },
        "files": {payload.name: {"bytes": len(encoded), "sha256": digest}},
    }


def test_gguf_identity_pins_file_tokenizer_and_dimensions(model):
    path, identity = model
    assert identity["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity["architecture"] == "fixture"
    assert identity["hidden_size"] == 64
    assert identity["layer_count"] == 8
    assert identity["vocab_size"] == 3
    assert identity["quantization"] == "Q4_K_M"
    assert len(identity["tokenizer_sha256"]) == 64
    assert len(identity["chat_template_sha256"]) == 64
    # _header() (this file's fixture) never sets head_count/head_count_kv -- omitted as None, mirroring
    # every other header-sourced field here when the underlying GGUF metadata key is absent.
    assert identity["head_count"] is None
    assert identity["head_count_kv"] is None


def test_gguf_identity_reads_head_count_and_kv_head_count_from_the_same_header_reader(tmp_path, monkeypatch):
    """clozn.analysis.pair_compatibility's head_count dimension (and the head-site transplant addendum)
    need the query/kv head counts -- read via the SAME gguf_header_from_path() call gguf_identity()
    already makes, reusing clozn.cli.fit_planner's own reader rather than a second GGUF parser."""
    path = tmp_path / "gqa-Q4_K_M.gguf"
    path.write_bytes(b"exact gguf bytes")

    def _gqa_header(_path):
        header = _header(path)
        header["head_count"] = 28
        header["head_count_kv"] = 4
        return header

    monkeypatch.setattr(contracts, "gguf_header_from_path", _gqa_header)
    identity = contracts.gguf_identity(path)
    assert identity["head_count"] == 28
    assert identity["head_count_kv"] == 4


def test_valid_artifact_checks_every_payload(model, tmp_path):
    _path, identity = model
    payload = tmp_path / "J_layer4.f16"
    payload.write_bytes(b"lens matrix")
    manifest = _manifest(identity, payload)
    result = contracts.validate_artifact_manifest(
        manifest, identity, tmp_path, expected_type="jlens"
    )
    assert result == {
        "artifact_type": "jlens",
        "artifact_version": 1,
        "model_sha256": identity["sha256"],
        "files": ["J_layer4.f16"],
    }


@pytest.mark.parametrize("field,bad", [
    ("architecture", "other"),
    ("hidden_size", 128),
    ("layer_count", 9),
    ("vocab_size", 4),
    ("tokenizer_sha256", "0" * 64),
])
def test_model_contract_mismatch_fails_closed(model, tmp_path, field, bad):
    _path, identity = model
    payload = tmp_path / "J_layer4.f16"
    payload.write_bytes(b"lens matrix")
    with pytest.raises(contracts.ArtifactContractError, match=field):
        contracts.validate_artifact_manifest(
            _manifest(identity, payload, **{field: bad}), identity, tmp_path
        )


def test_unqualified_gguf_digest_fails_closed(model, tmp_path):
    _path, identity = model
    payload = tmp_path / "dial.bin"
    payload.write_bytes(b"direction")
    manifest = _manifest(identity, payload,
                         compatible_gguf_sha256=["1" * 64])
    with pytest.raises(contracts.ArtifactContractError, match="not qualified"):
        contracts.validate_artifact_manifest(manifest, identity, tmp_path)


def test_corrupt_payload_fails_closed(model, tmp_path):
    _path, identity = model
    payload = tmp_path / "J_layer4.f16"
    payload.write_bytes(b"lens matrix")
    manifest = _manifest(identity, payload)
    payload.write_bytes(b"corruption!")
    with pytest.raises(contracts.ArtifactContractError, match="checksum mismatch"):
        contracts.validate_artifact_manifest(manifest, identity, tmp_path)


def test_manifest_path_cannot_escape_artifact_directory(model, tmp_path):
    _path, identity = model
    payload = tmp_path / "payload.bin"
    payload.write_bytes(b"x")
    manifest = _manifest(identity, payload)
    spec = manifest["files"].pop(payload.name)
    manifest["files"]["../payload.bin"] = spec
    with pytest.raises(contracts.ArtifactContractError, match="escapes"):
        contracts.validate_artifact_manifest(manifest, identity, tmp_path / "artifact")


def test_wave_one_pull_aliases_are_exact_and_recognized():
    expected = {
        "qwen": "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "qwen3.5": "Qwen3.5-9B-Q4_K_M.gguf",
        "llama": "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "gemma4": "gemma-4-E4B-it-Q4_K_M.gguf",
        "ministral": "Ministral-3-3B-Instruct-2512-Q4_K_M.gguf",
    }
    for alias, filename in expected.items():
        assert PULLABLE[alias][1] == filename
        assert any(alias == friendly and fragment in filename.lower()
                   for fragment, friendly, _flags in KNOWN)


def test_manifest_example_is_json_serializable(model, tmp_path):
    _path, identity = model
    payload = tmp_path / "J_layer4.f16"
    payload.write_bytes(b"lens matrix")
    json.dumps(_manifest(identity, payload))


def test_discovery_selects_only_the_artifact_qualified_for_this_gguf(model, tmp_path):
    _path, identity = model
    directory = tmp_path / "jlens" / "fixture-v1"
    directory.mkdir(parents=True)
    payload = directory / "J_layer4.f16"
    payload.write_bytes(b"lens matrix")
    (directory / "manifest.json").write_text(
        json.dumps(_manifest(identity, payload)), encoding="utf-8"
    )
    assert contracts.find_compatible_artifact("jlens", identity, tmp_path) == str(directory.resolve())


def test_explicit_legacy_manifest_is_refused(model, tmp_path):
    _path, identity = model
    directory = tmp_path / "legacy"
    directory.mkdir()
    (directory / "manifest.json").write_text(
        json.dumps({"model": "filename-only", "d_model": 64}), encoding="utf-8"
    )
    with pytest.raises(contracts.ArtifactContractError, match="contract_version"):
        contracts.find_compatible_artifact(
            "jlens", identity, tmp_path, explicit_dir=directory
        )


def test_chat_io_profile_validates_exact_identity_and_registry_shape(model, tmp_path):
    from clozn.server import structured_io

    _path, identity = model
    manifest = _chat_io_manifest(identity, tmp_path)
    result = contracts.validate_chat_io_profile(
        manifest, identity, "b" * 64, tmp_path
    )
    assert result["model_sha256"] == identity["sha256"]
    assert result["template_fingerprint"] == "b" * 64
    assert result["schema_subsets"] == {
        "tool_parameters": contracts.CHAT_IO_JSON_SCHEMA_SUBSET_ID,
        "json_schema": contracts.CHAT_IO_JSON_SCHEMA_SUBSET_ID
    }
    assert result["pipeline"] == contracts.CHAT_IO_PIPELINE
    assert result["evidence"]["sha256"] == manifest["profile"]["evidence"]["sha256"]
    registry = structured_io.validate_qualification_registry({
        "schema_version": structured_io.QUALIFICATION_SCHEMA,
        "entries": [result["registry_entry"]],
    })
    assert registry["entries"][0]["features"] == [
        "tools", "json_object", "json_schema"
    ]
    assert registry["entries"][0]["pipeline"] == contracts.CHAT_IO_PIPELINE


@pytest.mark.parametrize("field,bad", [
    ("features", ["tools", "future_feature"]),
    ("schema_subsets", {}),
])
def test_chat_io_profile_rejects_contract_drift(model, tmp_path, field, bad):
    _path, identity = model
    manifest = _chat_io_manifest(identity, tmp_path)
    manifest["profile"][field] = bad
    with pytest.raises(contracts.ArtifactContractError, match=field):
        contracts.validate_chat_io_profile(manifest, identity, "b" * 64, tmp_path)


@pytest.mark.parametrize("field", [
    "executor_id", "renderer_id", "grammar_id", "parser_id", "validator_id",
])
def test_chat_io_profile_rejects_each_native_pipeline_drift(model, tmp_path, field):
    _path, identity = model
    manifest = _chat_io_manifest(identity, tmp_path)
    manifest["profile"]["pipeline"][field] += ".drift"
    with pytest.raises(contracts.ArtifactContractError, match="pipeline"):
        contracts.validate_chat_io_profile(manifest, identity, "b" * 64, tmp_path)


def test_chat_io_profile_rejects_v1_and_incomplete_or_failed_results(model, tmp_path):
    _path, identity = model
    manifest = _chat_io_manifest(identity, tmp_path)
    manifest["artifact_version"] = 1
    with pytest.raises(contracts.ArtifactContractError, match="artifact_version"):
        contracts.validate_chat_io_profile(manifest, identity, "b" * 64, tmp_path)

    manifest = _chat_io_manifest(identity, tmp_path)
    evidence_path = tmp_path / manifest["profile"]["evidence"]["path"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    del evidence["results"]["tools"]
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    encoded = evidence_path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    manifest["files"][evidence_path.name] = {"bytes": len(encoded), "sha256": digest}
    manifest["profile"]["evidence"]["sha256"] = digest
    with pytest.raises(contracts.ArtifactContractError, match="cover pipeline"):
        contracts.validate_chat_io_profile(manifest, identity, "b" * 64, tmp_path)

    manifest = _chat_io_manifest(identity, tmp_path)
    evidence_path = tmp_path / manifest["profile"]["evidence"]["path"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["results"]["pipeline"] = {"passed": 1, "failed": 1}
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    encoded = evidence_path.read_bytes()
    digest = hashlib.sha256(encoded).hexdigest()
    manifest["files"][evidence_path.name] = {"bytes": len(encoded), "sha256": digest}
    manifest["profile"]["evidence"]["sha256"] = digest
    with pytest.raises(contracts.ArtifactContractError, match="failed == 0"):
        contracts.validate_chat_io_profile(manifest, identity, "b" * 64, tmp_path)


def test_chat_io_profile_rejects_template_mismatch_and_multi_model_claim(model, tmp_path):
    _path, identity = model
    manifest = _chat_io_manifest(identity, tmp_path)
    with pytest.raises(contracts.ArtifactContractError, match="template fingerprint mismatch"):
        contracts.validate_chat_io_profile(manifest, identity, "c" * 64, tmp_path)

    manifest["model"]["compatible_gguf_sha256"].append("d" * 64)
    with pytest.raises(contracts.ArtifactContractError, match="exactly the loaded GGUF"):
        contracts.validate_chat_io_profile(manifest, identity, "b" * 64, tmp_path)


def test_chat_io_profile_rejects_evidence_checksum_and_identity_drift(model, tmp_path):
    _path, identity = model
    manifest = _chat_io_manifest(identity, tmp_path)
    manifest["profile"]["evidence"]["sha256"] = "e" * 64
    with pytest.raises(contracts.ArtifactContractError, match="checksum"):
        contracts.validate_chat_io_profile(manifest, identity, "b" * 64, tmp_path)

    manifest = _chat_io_manifest(
        identity, tmp_path, evidence_overrides={"model_sha256": "f" * 64}
    )
    with pytest.raises(contracts.ArtifactContractError, match="model_sha256 mismatch"):
        contracts.validate_chat_io_profile(manifest, identity, "b" * 64, tmp_path)

    manifest = _chat_io_manifest(
        identity, tmp_path, evidence_overrides={"suite_id": "clozn.chat_io.future_suite.v1"}
    )
    with pytest.raises(contracts.ArtifactContractError, match="suite_id mismatch"):
        contracts.validate_chat_io_profile(manifest, identity, "b" * 64, tmp_path)


def test_chat_io_tool_only_profile_qualifies_tool_parameter_schema_subset(model, tmp_path):
    _path, identity = model
    manifest = _chat_io_manifest(identity, tmp_path, features=["tools"])
    result = contracts.validate_chat_io_profile(
        manifest, identity, "b" * 64, tmp_path
    )
    assert result["schema_subsets"] == {
        "tool_parameters": contracts.CHAT_IO_JSON_SCHEMA_SUBSET_ID,
    }


def test_chat_io_discovery_is_exact_and_model_free(model, tmp_path):
    _path, identity = model
    wrong = tmp_path / contracts.CHAT_IO_ARTIFACT_TYPE / "wrong-template"
    wrong.mkdir(parents=True)
    wrong_manifest = _chat_io_manifest(identity, wrong, template_fingerprint="c" * 64)
    (wrong / "manifest.json").write_text(json.dumps(wrong_manifest), encoding="utf-8")

    exact = tmp_path / contracts.CHAT_IO_ARTIFACT_TYPE / "exact"
    exact.mkdir()
    exact_manifest = _chat_io_manifest(identity, exact)
    (exact / "manifest.json").write_text(json.dumps(exact_manifest), encoding="utf-8")

    found = contracts.find_compatible_chat_io_profile(identity, "b" * 64, tmp_path)
    assert found["artifact_dir"] == str(exact.resolve())
    assert found["registry_entry"]["model_sha256"] == identity["sha256"]
    assert contracts.find_compatible_chat_io_profile(identity, "d" * 64, tmp_path) is None


def test_chat_io_discovery_refuses_ambiguous_exact_profiles(model, tmp_path):
    _path, identity = model
    for name in ("one", "two"):
        directory = tmp_path / contracts.CHAT_IO_ARTIFACT_TYPE / name
        directory.mkdir(parents=True)
        manifest = _chat_io_manifest(identity, directory)
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(contracts.ArtifactContractError, match="multiple chat_io artifacts"):
        contracts.find_compatible_chat_io_profile(identity, "b" * 64, tmp_path)
