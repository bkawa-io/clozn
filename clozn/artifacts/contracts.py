"""Versioned, fail-closed contracts for forward-only product artifacts.

J-lenses, calibrated dial bundles, and future SAE/readout artifacts are not
portable merely because two model filenames look similar.  They are tied to an
exact tokenizer and residual-space contract, and their application must be
qualified against one or more exact GGUF files.  This module is deliberately
stdlib-only and does not import Torch or the lab runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Mapping

from clozn.cli.fit_planner import gguf_header_from_path


CONTRACT_VERSION = 1
CHAT_IO_ARTIFACT_TYPE = "chat_io"
CHAT_IO_ARTIFACT_VERSION = 2
CHAT_IO_NATIVE_EXECUTOR_ID = "clozn.chat_io.atomic_executor.v1"
CHAT_IO_NATIVE_RENDERER_ID = "clozn.chat_io.llama_common.renderer.v1"
CHAT_IO_NATIVE_GRAMMAR_ID = "clozn.chat_io.ar_grammar.v1"
CHAT_IO_NATIVE_PARSER_ID = "clozn.chat_io.llama_common.parser.v1"
CHAT_IO_VALIDATOR_ID = "clozn.structured_io.native_message_validator.v1"
CHAT_IO_JSON_SCHEMA_SUBSET_ID = "clozn.structured_io.json_schema_subset.v1"
CHAT_IO_EVIDENCE_SCHEMA = "clozn.chat_io.qualification_evidence.v2"
CHAT_IO_QUALIFICATION_SUITE_ID = "clozn.chat_io.qualification_suite.v1"
CHAT_IO_PIPELINE = {
    "executor_id": CHAT_IO_NATIVE_EXECUTOR_ID,
    "renderer_id": CHAT_IO_NATIVE_RENDERER_ID,
    "grammar_id": CHAT_IO_NATIVE_GRAMMAR_ID,
    "parser_id": CHAT_IO_NATIVE_PARSER_ID,
    "validator_id": CHAT_IO_VALIDATOR_ID,
}
_CHAT_IO_FEATURES = frozenset({"tools", "json_object", "json_schema"})
_MODEL_FIELDS = (
    "architecture",
    "hidden_size",
    "layer_count",
    "vocab_size",
    "tokenizer_sha256",
)


class ArtifactContractError(ValueError):
    """An artifact is malformed, corrupt, or incompatible with the model."""


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 8 << 20) -> str:
    """Return the lowercase SHA-256 of a file without loading it into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_digest(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_contract(metadata: Mapping[str, object]) -> dict:
    """The GGUF tokenizer payload, including its chat template when present."""
    return {
        key: metadata[key]
        for key in sorted(metadata)
        if key.startswith("tokenizer.")
    }


def gguf_identity(path: str | os.PathLike[str], *, include_file_hash: bool = True) -> dict:
    """Read an exact, model-agnostic identity from a local GGUF.

    Header fields protect the activation and tokenizer contract.  The whole-file
    digest pins the exact quantized checkpoint used for qualification.  Callers
    performing only inventory may disable the expensive whole-file hash; artifact
    validation always requires it.
    """
    resolved = os.path.abspath(os.fspath(path))
    header = gguf_header_from_path(resolved)
    metadata = header["metadata"]
    tokenizer = _tokenizer_contract(metadata)
    tokens = metadata.get("tokenizer.ggml.tokens")
    vocab_size = len(tokens) if isinstance(tokens, list) else metadata.get("tokenizer.ggml.vocab_size")
    identity = {
        "name": header.get("name"),
        "architecture": header.get("arch"),
        "hidden_size": header.get("embedding_length"),
        "layer_count": header.get("n_layers"),
        "vocab_size": vocab_size,
        # head_count/head_count_kv: read via the SAME gguf_header_from_path() call above -- reusing
        # clozn.cli.fit_planner's existing header reader rather than writing a third GGUF parser. These
        # are the query-head and key/value-head counts respectively; they differ under GQA (grouped-
        # query attention), which is why both are carried rather than just one. d_head (the per-head
        # slice width a `head_write`/`kqv_out-<il>` transplant needs) is `hidden_size // head_count`
        # (query heads -- kqv_out rows are per-Q-head even when KV heads are fewer, per
        # clozn.receipts.hook_vocabulary's GQA note); callers needing d_head compute it from these two
        # fields, this module does not compute it itself.
        "head_count": header.get("head_count"),
        "head_count_kv": header.get("head_count_kv"),
        "tokenizer_sha256": _json_digest(tokenizer),
        "chat_template_sha256": _json_digest(metadata.get("tokenizer.chat_template")),
        "quantization": header.get("quant"),
        "file_size": int(header["file_size_bytes"]),
        "filename": os.path.basename(resolved),
    }
    if include_file_hash:
        # Routed through the persistent (path, size, mtime_ns) cache: spawn_engine calls this on
        # EVERY engine boot, and an uncached whole-file SHA of a 5-8 GB GGUF costs tens of seconds
        # per launch of an UNCHANGED file. Lazy import -- clozn.runs.identity imports this module
        # for sha256_file, so a top-level import here would be a cycle. Falls back to the direct
        # (raising) hash when the cached path returns None, preserving the strict behavior
        # validation callers rely on: a real IO error still raises, never silently omits.
        from clozn.runs.identity import model_sha256
        cached = model_sha256(resolved)
        identity["sha256"] = cached if cached else sha256_file(resolved)
    return identity


def _require_hex_digest(value, label: str) -> str:
    digest = str(value or "").lower()
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ArtifactContractError(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _read_json_object(path: Path, label: str) -> dict:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception as error:
        raise ArtifactContractError(
            f"{label} could not be read: {path}: {error}"
        ) from None
    if not isinstance(value, Mapping):
        raise ArtifactContractError(f"{label} must be a JSON object: {path}")
    return dict(value)


def validate_artifact_manifest(
    manifest: Mapping[str, object],
    model_identity: Mapping[str, object],
    artifact_dir: str | os.PathLike[str],
    *,
    expected_type: str | None = None,
) -> dict:
    """Validate a manifest and every declared payload against one loaded GGUF.

    The manifest shape is::

        {
          "contract_version": 1,
          "artifact_type": "jlens",
          "artifact_version": 1,
          "model": {
            "source_id": "Qwen/Qwen2.5-7B-Instruct",
            "architecture": "qwen2", "hidden_size": 3584,
            "layer_count": 28, "vocab_size": 152064,
            "tokenizer_sha256": "...",
            "compatible_gguf_sha256": ["..."]
          },
          "files": {"J_layer14.f16": {"sha256": "...", "bytes": 25690112}}
        }

    A lab artifact may be fitted on BF16/NF4 and transferred to more than one
    quant, but each product GGUF digest must be listed only after qualification.
    """
    if not isinstance(manifest, Mapping):
        raise ArtifactContractError("artifact manifest must be an object")
    if manifest.get("contract_version") != CONTRACT_VERSION:
        raise ArtifactContractError(
            f"unsupported artifact contract_version {manifest.get('contract_version')!r}; "
            f"expected {CONTRACT_VERSION}"
        )
    artifact_type = str(manifest.get("artifact_type") or "")
    if not artifact_type:
        raise ArtifactContractError("artifact_type is required")
    if expected_type is not None and artifact_type != expected_type:
        raise ArtifactContractError(
            f"artifact_type {artifact_type!r} does not match expected {expected_type!r}"
        )
    if not isinstance(manifest.get("artifact_version"), int):
        raise ArtifactContractError("artifact_version must be an integer")

    target = manifest.get("model")
    if not isinstance(target, Mapping):
        raise ArtifactContractError("model contract is required")
    for field in _MODEL_FIELDS:
        if target.get(field) is None:
            raise ArtifactContractError(f"model.{field} is required")
        if target[field] != model_identity.get(field):
            raise ArtifactContractError(
                f"model.{field} mismatch: artifact has {target[field]!r}, "
                f"GGUF has {model_identity.get(field)!r}"
            )

    actual_model_digest = _require_hex_digest(model_identity.get("sha256"), "GGUF sha256")
    compatible = target.get("compatible_gguf_sha256")
    if not isinstance(compatible, list) or not compatible:
        raise ArtifactContractError("model.compatible_gguf_sha256 must be a non-empty list")
    allowed = [_require_hex_digest(value, "compatible GGUF sha256") for value in compatible]
    if actual_model_digest not in allowed:
        raise ArtifactContractError(
            f"artifact is not qualified for GGUF sha256 {actual_model_digest}"
        )

    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ArtifactContractError("files must declare at least one payload")
    root = Path(artifact_dir).resolve()
    checked = []
    for relative, spec in files.items():
        if not isinstance(relative, str) or not relative or not isinstance(spec, Mapping):
            raise ArtifactContractError("each files entry must map a relative path to an object")
        payload = (root / relative).resolve()
        try:
            payload.relative_to(root)
        except ValueError:
            raise ArtifactContractError(f"artifact payload escapes its directory: {relative}") from None
        if not payload.is_file():
            raise ArtifactContractError(f"artifact payload is missing: {relative}")
        expected_bytes = spec.get("bytes")
        if not isinstance(expected_bytes, int) or expected_bytes < 0:
            raise ArtifactContractError(f"files[{relative!r}].bytes must be a non-negative integer")
        actual_bytes = payload.stat().st_size
        if actual_bytes != expected_bytes:
            raise ArtifactContractError(
                f"artifact payload size mismatch for {relative}: expected {expected_bytes}, got {actual_bytes}"
            )
        expected_digest = _require_hex_digest(spec.get("sha256"), f"files[{relative!r}].sha256")
        actual_digest = sha256_file(payload)
        if actual_digest != expected_digest:
            raise ArtifactContractError(
                f"artifact payload checksum mismatch for {relative}: "
                f"expected {expected_digest}, got {actual_digest}"
            )
        checked.append(relative)

    return {
        "artifact_type": artifact_type,
        "artifact_version": manifest["artifact_version"],
        "model_sha256": actual_model_digest,
        "files": checked,
    }


def validate_chat_io_profile(
    manifest: Mapping[str, object],
    model_identity: Mapping[str, object],
    template_fingerprint: str,
    artifact_dir: str | os.PathLike[str],
) -> dict:
    """Validate one exact-model structured-I/O qualification artifact.

    ``chat_io`` profiles use the ordinary artifact envelope and payload checksum
    rules, then add the exact rendered-template and protocol contracts that are
    behaviorally relevant to structured output.  Unlike transferable activation
    artifacts, one profile may name exactly one GGUF digest: qualification evidence
    for one quantized file is never generalized to another file, model family, or
    filename.

    The v2 profile extension is::

        {
          "profile": {
            "template_fingerprint": "...",
            "pipeline": {
              "executor_id": "clozn.chat_io.atomic_executor.v1",
              "renderer_id": "clozn.chat_io.llama_common.renderer.v1",
              "grammar_id": "clozn.chat_io.ar_grammar.v1",
              "parser_id": "clozn.chat_io.llama_common.parser.v1",
              "validator_id": "clozn.structured_io.native_message_validator.v1"
            },
            "features": ["tools", "json_object", "json_schema"],
            "schema_subsets": {
              "tool_parameters": "clozn.structured_io.json_schema_subset.v1",
              "json_schema": "clozn.structured_io.json_schema_subset.v1"
            },
            "evidence": {"path": "evidence.json", "sha256": "..."}
          }
        }

    The evidence file is checked by the generic ``files`` contract and must echo
    the exact identity and supported contract versions.  Its contents remain
    otherwise extensible so qualification runners can record model-specific cases.
    """
    base = validate_artifact_manifest(
        manifest, model_identity, artifact_dir, expected_type=CHAT_IO_ARTIFACT_TYPE
    )
    if manifest.get("artifact_version") != CHAT_IO_ARTIFACT_VERSION:
        raise ArtifactContractError(
            f"chat_io artifact_version must be {CHAT_IO_ARTIFACT_VERSION}"
        )
    model_contract = manifest["model"]
    actual_model_digest = base["model_sha256"]
    compatible = model_contract["compatible_gguf_sha256"]
    if len(compatible) != 1 or str(compatible[0]).lower() != actual_model_digest:
        raise ArtifactContractError(
            "chat_io model.compatible_gguf_sha256 must contain exactly the loaded GGUF sha256"
        )

    actual_template = str(template_fingerprint or "").lower()
    if (len(actual_template) < 16 or len(actual_template) > 64
            or any(ch not in "0123456789abcdef" for ch in actual_template)):
        raise ArtifactContractError(
            "loaded template fingerprint must be 16 to 64 lowercase hexadecimal characters"
        )

    profile = manifest.get("profile")
    if not isinstance(profile, Mapping):
        raise ArtifactContractError("chat_io profile is required")
    required = {
        "template_fingerprint", "pipeline", "features", "schema_subsets", "evidence",
    }
    extra = sorted(set(profile) - required)
    missing = sorted(required - set(profile))
    if extra or missing:
        raise ArtifactContractError(
            f"chat_io profile fields are invalid: missing={missing!r}, extra={extra!r}"
        )

    claimed_template = str(profile["template_fingerprint"] or "").lower()
    if claimed_template != actual_template:
        raise ArtifactContractError(
            f"chat_io template fingerprint mismatch: profile has {claimed_template!r}, "
            f"loaded template has {actual_template!r}"
        )
    pipeline = profile["pipeline"]
    if not isinstance(pipeline, Mapping) or dict(pipeline) != CHAT_IO_PIPELINE:
        raise ArtifactContractError(
            f"chat_io pipeline must equal {CHAT_IO_PIPELINE!r}"
        )

    features = profile["features"]
    if (not isinstance(features, list) or not features
            or any(not isinstance(feature, str) for feature in features)
            or len(set(features)) != len(features)
            or any(feature not in _CHAT_IO_FEATURES for feature in features)):
        raise ArtifactContractError(
            "chat_io features must be a non-empty unique list containing only "
            "tools, json_object, and json_schema"
        )
    schema_subsets = profile["schema_subsets"]
    expected_subsets = {}
    if "tools" in features:
        expected_subsets["tool_parameters"] = CHAT_IO_JSON_SCHEMA_SUBSET_ID
    if "json_schema" in features:
        expected_subsets["json_schema"] = CHAT_IO_JSON_SCHEMA_SUBSET_ID
    if schema_subsets != expected_subsets:
        raise ArtifactContractError(
            f"chat_io schema_subsets must equal {expected_subsets!r} for the declared features"
        )

    evidence = profile["evidence"]
    if not isinstance(evidence, Mapping) or set(evidence) != {"path", "sha256"}:
        raise ArtifactContractError(
            "chat_io profile.evidence must contain exactly path and sha256"
        )
    evidence_path = evidence["path"]
    if not isinstance(evidence_path, str) or not evidence_path:
        raise ArtifactContractError("chat_io profile.evidence.path must be a relative path")
    evidence_digest = _require_hex_digest(
        evidence["sha256"], "chat_io profile.evidence.sha256"
    )
    files = manifest["files"]
    file_spec = files.get(evidence_path)
    if not isinstance(file_spec, Mapping):
        raise ArtifactContractError(
            "chat_io evidence payload must be declared in files"
        )
    declared_digest = _require_hex_digest(
        file_spec.get("sha256"), f"files[{evidence_path!r}].sha256"
    )
    if evidence_digest != declared_digest:
        raise ArtifactContractError(
            "chat_io evidence checksum does not match its files declaration"
        )

    root = Path(artifact_dir).resolve()
    payload_path = (root / evidence_path).resolve()
    try:
        payload_path.relative_to(root)
    except ValueError:
        raise ArtifactContractError(
            f"artifact payload escapes its directory: {evidence_path}"
        ) from None
    payload = _read_json_object(payload_path, "chat_io qualification evidence")
    evidence_required = {
        "schema_version", "suite_id", "model_sha256", "template_fingerprint",
        "pipeline", "features", "schema_subsets", "results",
    }
    missing_evidence = sorted(evidence_required - set(payload))
    if missing_evidence:
        raise ArtifactContractError(
            f"chat_io qualification evidence is missing fields: {missing_evidence!r}"
        )
    expected_evidence = {
        "schema_version": CHAT_IO_EVIDENCE_SCHEMA,
        "suite_id": CHAT_IO_QUALIFICATION_SUITE_ID,
        "model_sha256": actual_model_digest,
        "template_fingerprint": actual_template,
        "pipeline": CHAT_IO_PIPELINE,
        "features": features,
        "schema_subsets": expected_subsets,
    }
    for field, expected in expected_evidence.items():
        if payload[field] != expected:
            raise ArtifactContractError(
                f"chat_io qualification evidence {field} mismatch: "
                f"expected {expected!r}, got {payload[field]!r}"
            )
    results = payload["results"]
    required_results = {"pipeline", *features}
    if not isinstance(results, Mapping) or not required_results.issubset(results):
        raise ArtifactContractError(
            "chat_io qualification evidence results must cover pipeline and every feature"
        )
    for result_name in sorted(required_results):
        result = results[result_name]
        if not isinstance(result, Mapping) or set(result) != {"passed", "failed"}:
            raise ArtifactContractError(
                f"chat_io qualification result {result_name!r} must contain exactly passed and failed"
            )
        passed = result["passed"]
        failed = result["failed"]
        if (not isinstance(passed, int) or isinstance(passed, bool) or passed < 1
                or not isinstance(failed, int) or isinstance(failed, bool) or failed != 0):
            raise ArtifactContractError(
                f"chat_io qualification result {result_name!r} requires passed >= 1 and failed == 0"
            )

    registry_entry = {
        "model_sha256": actual_model_digest,
        "template_fingerprint": actual_template,
        "features": list(features),
        "schema_subsets": dict(expected_subsets),
        "pipeline": dict(CHAT_IO_PIPELINE),
        "evidence": {
            "schema_version": CHAT_IO_EVIDENCE_SCHEMA,
            "suite_id": CHAT_IO_QUALIFICATION_SUITE_ID,
            "artifact_version": manifest["artifact_version"],
            "payload_sha256": evidence_digest,
        },
    }
    return {
        **base,
        "template_fingerprint": actual_template,
        "pipeline": dict(CHAT_IO_PIPELINE),
        "features": list(features),
        "schema_subsets": dict(schema_subsets),
        "evidence": {
            "path": evidence_path,
            "sha256": evidence_digest,
            "schema_version": CHAT_IO_EVIDENCE_SCHEMA,
            "suite_id": CHAT_IO_QUALIFICATION_SUITE_ID,
        },
        "registry_entry": registry_entry,
    }


def find_compatible_chat_io_profile(
    model_identity: Mapping[str, object],
    template_fingerprint: str,
    root: str | os.PathLike[str],
    *,
    explicit_dir: str | os.PathLike[str] | None = None,
) -> dict | None:
    """Find and validate exactly one ``chat_io`` profile for an active identity.

    Automatic discovery ignores profiles for other exact model/template tuples.
    Once a manifest claims the active tuple, corruption or contract drift is an
    error rather than a reason to continue searching.  The returned normalized
    object includes ``artifact_dir`` and a registry-compatible entry, but merely
    discovering it does not activate structured I/O.
    """
    actual_model_digest = _require_hex_digest(
        model_identity.get("sha256"), "GGUF sha256"
    )
    actual_template = str(template_fingerprint or "").lower()
    if (len(actual_template) < 16 or len(actual_template) > 64
            or any(ch not in "0123456789abcdef" for ch in actual_template)):
        raise ArtifactContractError(
            "loaded template fingerprint must be 16 to 64 lowercase hexadecimal characters"
        )
    if explicit_dir is not None:
        candidates = [Path(explicit_dir)]
        strict = True
    else:
        base = Path(root) / CHAT_IO_ARTIFACT_TYPE
        if not base.is_dir():
            return None
        candidates = sorted({path.parent for path in base.rglob("manifest.json")})
        strict = False

    matches: list[dict] = []
    for directory in candidates:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            if strict:
                raise ArtifactContractError(f"artifact manifest is missing: {manifest_path}")
            continue
        try:
            manifest = _read_json_object(manifest_path, "artifact manifest")
        except ArtifactContractError:
            if strict:
                raise
            continue

        model_contract = manifest.get("model")
        compatible = (model_contract.get("compatible_gguf_sha256")
                      if isinstance(model_contract, Mapping) else None)
        profile = manifest.get("profile")
        claimed_template = (profile.get("template_fingerprint")
                            if isinstance(profile, Mapping) else None)
        claims_identity = (
            isinstance(compatible, list)
            and actual_model_digest in {str(value).lower() for value in compatible}
            and isinstance(claimed_template, str)
            and claimed_template.lower() == actual_template
        )
        if not strict and not claims_identity:
            continue
        normalized = validate_chat_io_profile(
            manifest, model_identity, actual_template, directory
        )
        matches.append({**normalized, "artifact_dir": str(directory.resolve())})

    if len(matches) > 1:
        raise ArtifactContractError(
            "multiple chat_io artifacts claim exact GGUF/template identity "
            f"{actual_model_digest}/{actual_template}: "
            + ", ".join(match["artifact_dir"] for match in matches)
        )
    return matches[0] if matches else None


def find_compatible_artifact(
    artifact_type: str,
    model_identity: Mapping[str, object],
    root: str | os.PathLike[str],
    *,
    explicit_dir: str | os.PathLike[str] | None = None,
) -> str | None:
    """Find exactly one valid artifact directory for a GGUF.

    An explicitly selected directory is strict: a missing, legacy, corrupt, or
    incompatible manifest is an error. Automatic discovery ignores artifacts
    qualified for other models, but refuses ambiguity or corruption in a
    candidate that otherwise claims this GGUF digest.
    """
    if explicit_dir is not None:
        candidates = [Path(explicit_dir)]
        strict = True
    else:
        base = Path(root) / artifact_type
        if not base.is_dir():
            return None
        candidates = sorted({path.parent for path in base.rglob("manifest.json")})
        strict = False

    actual_digest = str(model_identity.get("sha256") or "").lower()
    matches: list[str] = []
    for directory in candidates:
        manifest_path = directory / "manifest.json"
        if not manifest_path.is_file():
            if strict:
                raise ArtifactContractError(f"artifact manifest is missing: {manifest_path}")
            continue
        try:
            with open(manifest_path, encoding="utf-8") as handle:
                manifest = json.load(handle)
        except Exception as error:
            if strict:
                raise ArtifactContractError(
                    f"artifact manifest could not be read: {manifest_path}: {error}"
                ) from None
            continue

        model_contract = manifest.get("model") if isinstance(manifest, Mapping) else None
        claimed = (model_contract.get("compatible_gguf_sha256")
                   if isinstance(model_contract, Mapping) else None)
        claims_this_model = isinstance(claimed, list) and actual_digest in {
            str(value).lower() for value in claimed
        }
        if not strict and not claims_this_model:
            continue
        validate_artifact_manifest(
            manifest, model_identity, directory, expected_type=artifact_type
        )
        matches.append(str(directory.resolve()))

    if len(matches) > 1:
        raise ArtifactContractError(
            f"multiple {artifact_type} artifacts claim GGUF sha256 {actual_digest}: "
            + ", ".join(matches)
        )
    return matches[0] if matches else None
