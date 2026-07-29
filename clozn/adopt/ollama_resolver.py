"""Deterministic, read-only resolution of one Ollama model to one GGUF blob.

Ollama's blob directory is content-addressed but may contain configs, projectors,
and stale layers next to model weights.  This module therefore never scans for
the largest file.  A manifest must name exactly one model layer, or resolution
fails explicitly.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path


_SHA256_DIGEST = re.compile(r"^sha256:([0-9a-fA-F]{64})$")
_FROM_LINE = re.compile(r"^\s*FROM\s+(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_MODEL_MEDIA_TYPES = {
    "application/vnd.ollama.image.model",
    "application/vnd.ollama.image.model.v1",
}


def _existing_file(value) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip().strip('"')
    if not os.path.isabs(candidate) or not os.path.isfile(candidate):
        return None
    return os.path.abspath(candidate)


def absolute_from_path(modelfile_text) -> str | None:
    if not modelfile_text:
        return None
    match = _FROM_LINE.search(str(modelfile_text))
    return _existing_file(match.group(1)) if match else None


def _model_reference(name: str) -> tuple[str, list[str], str, str]:
    """Map an Ollama name to its manifest path components.

    `llama3` -> registry.ollama.ai/library/llama3/latest
    `team/model:tag` -> registry.ollama.ai/team/model/tag
    `registry.example/team/model:tag` preserves the explicit registry.
    """
    value = str(name or "").strip()
    if not value:
        raise ValueError("an Ollama model name is required")
    path, sep, tag = value.rpartition(":")
    if not sep or "/" in tag:
        path, tag = value, "latest"
    parts = [part for part in path.split("/") if part]
    if not parts:
        raise ValueError(f"invalid Ollama model name: {name!r}")
    registry = "registry.ollama.ai"
    if len(parts) == 1:
        namespace, model = ["library"], parts[0]
    elif "." in parts[0] or ":" in parts[0]:
        registry, *namespace, model = parts
        if not namespace:
            namespace = ["library"]
    else:
        *namespace, model = parts
    return registry, namespace, model, tag


def manifest_path(storage_root: str | os.PathLike, model_name: str) -> Path:
    registry, namespace, model, tag = _model_reference(model_name)
    return Path(storage_root) / "manifests" / registry / Path(*namespace) / model / tag


def _manifest_resolution(path: Path, storage_root: Path) -> dict:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Ollama manifest {path} is unreadable: {exc}") from None
    layers = manifest.get("layers") if isinstance(manifest, dict) else None
    layers = layers if isinstance(layers, list) else []
    model_layers = [
        layer for layer in layers
        if isinstance(layer, dict)
        and (
            layer.get("mediaType") in _MODEL_MEDIA_TYPES
            or str(layer.get("mediaType") or "").endswith(".image.model")
        )
    ]
    if len(model_layers) != 1:
        raise ValueError(
            f"Ollama manifest {path} names {len(model_layers)} model layers; "
            "refusing to guess among blobs"
        )
    layer = model_layers[0]
    match = _SHA256_DIGEST.fullmatch(str(layer.get("digest") or ""))
    if not match:
        raise ValueError(f"Ollama manifest {path} has an invalid model-layer digest")
    expected_size = layer.get("size")
    if isinstance(expected_size, bool) or (
        expected_size is not None and not isinstance(expected_size, int)
    ):
        raise ValueError(f"Ollama manifest {path} has an invalid model-layer size")
    blob = storage_root / "blobs" / f"sha256-{match.group(1).lower()}"
    if not blob.is_file():
        raise ValueError(f"Ollama manifest {path} points to a missing model blob: {blob}")
    return {
        "path": str(blob.resolve()),
        "method": "manifest_model_layer",
        "manifest_path": str(path.resolve()),
        "blob_digest": f"sha256:{match.group(1).lower()}",
        **({"expected_size": expected_size} if expected_size is not None else {}),
    }


def resolve_model_blob(
    model_name: str,
    shown: dict,
    *,
    storage_roots: list[str] | None = None,
    explicit_blob: str | None = None,
) -> dict | None:
    """Resolve one exact local blob using typed sources, never directory heuristics."""
    shown = shown if isinstance(shown, dict) else {}

    # Some Ollama-compatible servers expose a structured local path.  Accept it
    # only when it is absolute and exists; arbitrary nested string scraping is
    # intentionally excluded.
    for key in ("model_path", "blob_path", "path"):
        candidate = _existing_file(shown.get(key))
        if candidate:
            return {"path": candidate, "method": f"show.{key}"}

    for root_value in storage_roots or []:
        root = Path(root_value).expanduser()
        path = manifest_path(root, model_name)
        if path.is_file():
            return _manifest_resolution(path, root)

    candidate = absolute_from_path(shown.get("modelfile"))
    if candidate:
        return {"path": candidate, "method": "modelfile_absolute_from"}

    if explicit_blob is not None:
        candidate = _existing_file(explicit_blob)
        if not candidate:
            raise ValueError(f"--blob must name an existing absolute file: {explicit_blob}")
        return {"path": candidate, "method": "explicit_blob"}
    return None


def parse_parameters(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    out: dict[str, object] = {}
    for line in str(value or "").splitlines():
        key, sep, raw = line.strip().partition(" ")
        if not sep:
            continue
        raw = raw.strip()
        parsed: object = raw
        try:
            parsed = json.loads(raw)
        except Exception:
            try:
                parsed = float(raw) if "." in raw else int(raw)
            except ValueError:
                pass
        if key == "stop":
            prior = out.get(key)
            out[key] = [*(prior if isinstance(prior, list) else []), parsed]
        else:
            out[key] = parsed
    return out


def translate_definition(shown: dict) -> dict:
    """Classify template/system/parameter fidelity without claiming equivalence."""
    shown = shown if isinstance(shown, dict) else {}
    components = []

    template = shown.get("template")
    if isinstance(template, str) and template:
        components.append({
            "component": "template",
            "status": "unsupported",
            "source_sha256": __import__("hashlib").sha256(template.encode("utf-8")).hexdigest(),
            "note": "captured, but Clozn cannot claim Ollama-template equivalence",
        })
    else:
        components.append({
            "component": "template", "status": "not_present",
            "note": "Ollama did not report a template",
        })

    system = shown.get("system")
    if isinstance(system, str) and system:
        components.append({
            "component": "system_prompt",
            "status": "translated",
            "value": system,
            "target": "request.system_message",
        })
    else:
        components.append({
            "component": "system_prompt", "status": "not_present",
            "note": "Ollama did not report a separate system prompt",
        })

    mappings = {
        "temperature": "request.temperature",
        "top_p": "request.top_p",
        "seed": "request.seed",
        "num_ctx": "engine.n_ctx",
        "num_predict": "request.max_tokens",
    }
    parameters = parse_parameters(shown.get("parameters"))
    for key in sorted(parameters):
        value = parameters[key]
        if key in mappings:
            components.append({
                "component": f"parameter.{key}",
                "status": "translated",
                "value": value,
                "target": mappings[key],
            })
        else:
            components.append({
                "component": f"parameter.{key}",
                "status": "unsupported",
                "value": value,
                "note": "captured for provenance but not automatically applied by adoption",
            })
    warnings = [
        f"{item['component']}: {item['note']}"
        for item in components
        if item.get("status") in {"unsupported", "not_present"} and item.get("note")
    ]
    return {
        "source": "ollama_modelfile",
        "exactly_reproduced": all(
            item["status"] in {"reproduced_exactly", "not_present"} for item in components
        ),
        "components": components,
        "warnings": warnings,
    }
