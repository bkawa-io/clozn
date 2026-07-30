"""Qualified configuration for managed preloaded multi-model serving.

The runtime key cannot be inferred honestly from a friendly model name.  In
particular, current workers may not announce their template fingerprint or
engine build before launch.  RT-BOOT-01 therefore accepts multi-model
definitions only through an explicit qualified manifest.  The existing
``clozn serve MODEL`` path remains the unchanged one-worker compatibility path.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

from clozn.cli.worker_registry import (
    AdapterRuntimeIdentity,
    RuntimeKey,
    WorkerDefinition,
    WorkerRegistryConfigError,
)


SCHEMA_VERSION = "clozn.managed-models.v1"
_ROOT_FIELDS = frozenset({
    "schema_version",
    "default_model_id",
    "preload_model_ids",
    "max_loaded_models",
    "models",
})
_MODEL_FIELDS = frozenset({
    "model_id",
    "model",
    "runtime_key",
    "flags",
    "prefer_gpu",
    "boot_timeout",
    "restart_limit",
    "restart_window",
})
_RUNTIME_KEY_FIELDS = frozenset({
    "key_sha256",
    "gguf_artifact_sha256",
    "context_size",
    "backend",
    "adapter",
    "template_fingerprint",
    "engine_build",
    "white_box_flags",
})
_ADAPTER_FIELDS = frozenset({
    "present", "identity_sha256", "artifact_sha256", "scale",
})


class ManagedModelsConfigError(ValueError):
    """A managed-model manifest is malformed, ambiguous, or unqualified."""


@dataclass(frozen=True)
class ManagedModelsConfig:
    definitions: tuple[WorkerDefinition, ...]
    default_model_id: str
    preload_model_ids: tuple[str, ...]
    max_loaded_models: int
    source_path: str

    def definition(self, model_id: str) -> WorkerDefinition:
        for definition in self.definitions:
            if definition.model_id == model_id:
                return definition
        raise ManagedModelsConfigError(
            f"default model {model_id!r} is not defined"
        )


def _object(value, label: str) -> dict:
    if not isinstance(value, Mapping):
        raise ManagedModelsConfigError(f"{label} must be an object")
    return dict(value)


def _closed(value: dict, allowed: frozenset[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ManagedModelsConfigError(
            f"{label} contains unknown field {unknown[0]!r}"
        )


def _string(value, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ManagedModelsConfigError(f"{label} must be a non-empty string")
    if value != value.strip():
        raise ManagedModelsConfigError(
            f"{label} may not contain surrounding whitespace"
        )
    return value


def _string_list(value, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ManagedModelsConfigError(f"{label} must be an array")
    result = tuple(
        _string(item, f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(result)) != len(result):
        raise ManagedModelsConfigError(f"{label} contains a duplicate")
    return result


def _positive_int(value, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ManagedModelsConfigError(f"{label} must be a positive integer")
    return value


def _adapter(raw) -> AdapterRuntimeIdentity:
    value = _object(raw, "runtime_key.adapter")
    _closed(value, _ADAPTER_FIELDS, "runtime_key.adapter")
    try:
        return AdapterRuntimeIdentity(
            present=value.get("present"),
            identity_sha256=value.get("identity_sha256"),
            artifact_sha256=value.get("artifact_sha256"),
            scale=value.get("scale"),
        )
    except WorkerRegistryConfigError as error:
        raise ManagedModelsConfigError(str(error)) from None


def _runtime_key(raw) -> RuntimeKey:
    value = _object(raw, "runtime_key")
    _closed(value, _RUNTIME_KEY_FIELDS, "runtime_key")
    missing = sorted(_RUNTIME_KEY_FIELDS - set(value))
    if missing:
        raise ManagedModelsConfigError(
            f"runtime_key is missing required field {missing[0]!r}"
        )
    try:
        key = RuntimeKey(
            gguf_artifact_sha256=value["gguf_artifact_sha256"],
            context_size=value["context_size"],
            backend=value["backend"],
            adapter=_adapter(value["adapter"]),
            template_fingerprint=value["template_fingerprint"],
            engine_build=value["engine_build"],
            white_box_flags=value["white_box_flags"],
        )
    except WorkerRegistryConfigError as error:
        raise ManagedModelsConfigError(str(error)) from None
    if value["key_sha256"] != key.key_sha256:
        raise ManagedModelsConfigError(
            "runtime_key.key_sha256 does not match its canonical facets"
        )
    for capability in ("sae", "jlens", "attn_knockout"):
        if capability not in key.white_box_flags:
            raise ManagedModelsConfigError(
                f"runtime_key.white_box_flags is missing {capability!r}"
            )
    return key


def _resolve_file(value, *, root: Path, label: str) -> str:
    path = Path(os.path.expanduser(_string(value, label)))
    if not path.is_absolute():
        path = root / path
    resolved = os.path.abspath(os.fspath(path))
    if not os.path.isfile(resolved):
        raise ManagedModelsConfigError(f"{label} does not exist: {resolved}")
    return resolved


def _definition(raw, *, root: Path, index: int) -> WorkerDefinition:
    value = _object(raw, f"models[{index}]")
    _closed(value, _MODEL_FIELDS, f"models[{index}]")
    required = {"model_id", "model", "runtime_key"}
    missing = sorted(required - set(value))
    if missing:
        raise ManagedModelsConfigError(
            f"models[{index}] is missing required field {missing[0]!r}"
        )
    flags = value.get("flags", {})
    if not isinstance(flags, Mapping):
        raise ManagedModelsConfigError(f"models[{index}].flags must be an object")
    flags = dict(flags)
    if flags.get("adapter"):
        flags["adapter"] = _resolve_file(
            flags["adapter"], root=root, label=f"models[{index}].flags.adapter"
        )
    try:
        definition = WorkerDefinition(
            model_id=_string(value["model_id"], f"models[{index}].model_id"),
            model=_resolve_file(
                value["model"], root=root, label=f"models[{index}].model"
            ),
            runtime_key=_runtime_key(value["runtime_key"]),
            flags=flags,
            prefer_gpu=value.get("prefer_gpu", True),
            boot_timeout=value.get("boot_timeout", 180.0),
            restart_limit=value.get("restart_limit", 3),
            restart_window=value.get("restart_window", 60.0),
        )
    except WorkerRegistryConfigError as error:
        raise ManagedModelsConfigError(str(error)) from None
    if len(definition.runtime_key.template_fingerprint) != 16:
        raise ManagedModelsConfigError(
            f"models[{index}].runtime_key.template_fingerprint must be "
            "the 16-character canonical live apply-template fingerprint"
        )
    if definition.runtime_key.white_box_flags["sae"]:
        raise ManagedModelsConfigError(
            f"models[{index}] cannot enable SAE until its artifact identity "
            "is represented in the routing key"
        )
    if definition.runtime_key.white_box_flags["jlens"]:
        raise ManagedModelsConfigError(
            f"models[{index}] cannot enable J-lens until its artifact identity "
            "is represented in the routing key"
        )
    adapter = definition.runtime_key.adapter
    from clozn.runs.identity import model_sha256
    actual_model_sha = model_sha256(definition.model)
    if actual_model_sha != definition.runtime_key.gguf_artifact_sha256:
        raise ManagedModelsConfigError(
            f"models[{index}] GGUF SHA-256 does not match "
            "runtime_key.gguf_artifact_sha256"
        )
    if adapter.present:
        from clozn.artifacts.contracts import sha256_file
        actual_adapter_sha = sha256_file(definition.flags["adapter"])
        if actual_adapter_sha != adapter.artifact_sha256:
            raise ManagedModelsConfigError(
                f"models[{index}] adapter artifact SHA-256 does not match "
                "runtime_key.adapter.artifact_sha256"
            )
    return definition


def load_managed_models(
    path: str,
    *,
    default_model_id: str | None = None,
    preload_model_ids: Iterable[str] | None = None,
    max_loaded_models: int | None = None,
) -> ManagedModelsConfig:
    """Read and fail-closed validate one qualified managed-model manifest."""
    resolved = os.path.abspath(os.path.expanduser(os.fspath(path)))
    try:
        with open(resolved, encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as error:
        raise ManagedModelsConfigError(
            f"could not read managed-model config {resolved}: {error}"
        ) from None
    value = _object(raw, "managed-model config")
    try:
        from clozn import schemas
        schemas.validate(value, SCHEMA_VERSION)
    except (schemas.ValidationError, schemas.SchemaError) as error:
        raise ManagedModelsConfigError(
            f"managed-model config failed {SCHEMA_VERSION}: {error}"
        ) from None
    _closed(value, _ROOT_FIELDS, "managed-model config")
    if value.get("schema_version") != SCHEMA_VERSION:
        raise ManagedModelsConfigError(
            f"schema_version must be {SCHEMA_VERSION!r}"
        )
    models = value.get("models")
    if not isinstance(models, list) or not models:
        raise ManagedModelsConfigError("models must be a non-empty array")
    root = Path(resolved).parent
    definitions = tuple(
        _definition(item, root=root, index=index)
        for index, item in enumerate(models)
    )

    configured_default = (
        _string(default_model_id, "--default-model")
        if default_model_id is not None
        else _string(value.get("default_model_id"), "default_model_id")
    )
    configured_preloads = (
        tuple(
            _string(item, "--preload")
            for item in preload_model_ids
        )
        if preload_model_ids is not None
        else _string_list(value.get("preload_model_ids"), "preload_model_ids")
    )
    if len(set(configured_preloads)) != len(configured_preloads):
        raise ManagedModelsConfigError("--preload contains a duplicate")
    configured_limit = (
        _positive_int(max_loaded_models, "--max-loaded-models")
        if max_loaded_models is not None
        else _positive_int(
            value.get("max_loaded_models"), "max_loaded_models"
        )
    )
    try:
        # Let RT-02 remain the authority for cross-definition/default/preload
        # invariants without starting any process.
        from clozn.cli.worker_registry import WorkerRegistry
        WorkerRegistry(
            definitions,
            default_model_id=configured_default,
            preload_model_ids=configured_preloads,
            max_loaded_workers=configured_limit,
        )
    except WorkerRegistryConfigError as error:
        raise ManagedModelsConfigError(str(error)) from None
    return ManagedModelsConfig(
        definitions=definitions,
        default_model_id=configured_default,
        preload_model_ids=configured_preloads,
        max_loaded_models=configured_limit,
        source_path=resolved,
    )
