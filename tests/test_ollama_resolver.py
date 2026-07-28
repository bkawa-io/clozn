from __future__ import annotations

import hashlib
import json

import pytest

from clozn.adopt.ollama_resolver import (
    manifest_path,
    parse_parameters,
    resolve_model_blob,
    translate_definition,
)


def _manifest(root, model, layers):
    path = manifest_path(root, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"layers": layers}), encoding="utf-8")
    return path


def test_manifest_selects_exact_model_layer_not_largest_blob(tmp_path):
    root = tmp_path / "models"
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    model_bytes = b"GGUF-model"
    model_sha = hashlib.sha256(model_bytes).hexdigest()
    config_sha = hashlib.sha256(b"x" * 1000).hexdigest()
    (blobs / f"sha256-{model_sha}").write_bytes(model_bytes)
    (blobs / f"sha256-{config_sha}").write_bytes(b"x" * 1000)
    manifest = _manifest(root, "team/model:Q4", [
        {
            "mediaType": "application/vnd.ollama.image.params",
            "digest": f"sha256:{config_sha}",
            "size": 1000,
        },
        {
            "mediaType": "application/vnd.ollama.image.model",
            "digest": f"sha256:{model_sha}",
            "size": len(model_bytes),
        },
    ])

    resolved = resolve_model_blob("team/model:Q4", {}, storage_roots=[str(root)])

    assert resolved["path"] == str((blobs / f"sha256-{model_sha}").resolve())
    assert resolved["manifest_path"] == str(manifest.resolve())
    assert resolved["blob_digest"] == f"sha256:{model_sha}"
    assert resolved["expected_size"] == len(model_bytes)


def test_manifest_refuses_ambiguous_model_layers(tmp_path):
    root = tmp_path / "models"
    blobs = root / "blobs"
    blobs.mkdir(parents=True)
    layers = []
    for payload in (b"one", b"two"):
        digest = hashlib.sha256(payload).hexdigest()
        (blobs / f"sha256-{digest}").write_bytes(payload)
        layers.append({
            "mediaType": "application/vnd.ollama.image.model",
            "digest": f"sha256:{digest}",
            "size": len(payload),
        })
    _manifest(root, "model", layers)

    with pytest.raises(ValueError, match="refusing to guess"):
        resolve_model_blob("model", {}, storage_roots=[str(root)])


def test_explicit_blob_is_a_recorded_fallback(tmp_path):
    blob = (tmp_path / "model.gguf").resolve()
    blob.write_bytes(b"GGUF")
    resolved = resolve_model_blob("cloud:latest", {}, explicit_blob=str(blob))
    assert resolved == {"path": str(blob), "method": "explicit_blob"}


def test_parameter_translation_is_per_component_and_keeps_unknowns_explicit():
    translated = translate_definition({
        "template": "{{ .Prompt }}",
        "system": "Answer briefly.",
        "parameters": "temperature 0.2\nnum_ctx 4096\nstop \"END\"\n",
    })
    by_name = {item["component"]: item for item in translated["components"]}
    assert by_name["template"]["status"] == "unsupported"
    assert by_name["system_prompt"]["status"] == "translated"
    assert by_name["parameter.temperature"]["target"] == "request.temperature"
    assert by_name["parameter.num_ctx"]["target"] == "engine.n_ctx"
    assert by_name["parameter.stop"]["status"] == "unsupported"
    assert parse_parameters("stop \"A\"\nstop \"B\"")["stop"] == ["A", "B"]
