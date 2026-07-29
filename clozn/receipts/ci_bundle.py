"""Privacy-safe receipt bundles for CI experiment reports.

The GitHub Action's verify mode receives an experiment result produced
elsewhere.  It must never turn that artifact into a second, broader export by
accident.  This module therefore has a deliberately narrow contract:

* only coordinates present in ``clozn.ci-report.v1.receipt_index`` are eligible;
* a receipt is emitted only when the indexed run ID exactly matches the
  experiment cell and its embedded run;
* the default and only v1 privacy tier is ``metadata_only``;
* prompts, messages, responses, rendered prompts, source text, raw tool output,
  local paths, and arbitrary extension payloads are never copied.

The ZIP is deterministic for identical inputs.  Entry names are generated
locally rather than derived from run IDs, JSON is canonical, and timestamps are
fixed.  No model, engine, network, or run-store import is used.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import zipfile

BUNDLE_SCHEMA = "clozn.ci-receipts.v1"
PRIVACY_TIER = "metadata_only"
_FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)
_GENERATION_FIELDS = (
    "sampling",
    "sampler_mode",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "no_repeat_ngram_size",
    "max_tokens",
    "seed",
    "n_ctx",
    "finish_reason_source",
    "capture_tier",
)
_IDENTITY_FIELDS = (
    "model_sha256",
    "model_size_bytes",
    "template_fingerprint",
    "engine_build",
    "clozn_version",
    "captured_at",
)
_SAFE_EXT_IDENTITY_FIELDS = frozenset({
    "sha256",
    "digest",
    "fingerprint",
    "version",
    "build",
    "commit",
    "scale",
    "rank",
    "alpha",
})


class CIBundleError(ValueError):
    """The report/evidence pair cannot produce an honest indexed bundle."""


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _canonical_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _content_fingerprint(value) -> dict | None:
    if not isinstance(value, str):
        return None
    encoded = value.encode("utf-8")
    return {
        "sha256": _sha256_bytes(encoded),
        "bytes": len(encoded),
    }


def _safe_ext_identity(value) -> dict:
    """Keep only scalar reproduction identity, never paths or opaque payloads."""
    out = {}
    for namespace, facet in sorted(_dict(value).items()):
        if not isinstance(namespace, str) or not isinstance(facet, dict):
            continue
        safe = {}
        for key, item in sorted(facet.items()):
            normalized = str(key).casefold()
            if (
                normalized in _SAFE_EXT_IDENTITY_FIELDS
                or normalized.endswith(("_sha256", "_digest", "_fingerprint", "_version"))
            ) and isinstance(item, (str, int, float, bool)):
                safe[str(key)] = item
        if safe:
            out[namespace] = safe
    return out


def _safe_identity(run: dict) -> dict:
    identity = _dict(run.get("identity"))
    out = {
        field: identity[field]
        for field in _IDENTITY_FIELDS
        if isinstance(identity.get(field), (str, int, float, bool))
    }
    ext = _safe_ext_identity(identity.get("ext"))
    if ext:
        out["ext"] = ext
    return out


def _safe_context_receipt(run: dict) -> dict | None:
    receipt = _dict(run.get("context_receipt"))
    if not receipt:
        return None
    out = {
        "schema_version": receipt.get("schema_version"),
        "privacy": PRIVACY_TIER,
    }
    delivered = receipt.get("delivered")
    if isinstance(delivered, list):
        safe_segments = []
        for segment in delivered:
            if not isinstance(segment, dict):
                continue
            safe_segments.append({
                key: segment[key]
                for key in (
                    "segment_id",
                    "source_type",
                    "included",
                    "reason",
                    "sha256",
                    "content_sha256",
                    "char_count",
                    "token_estimate",
                )
                if isinstance(segment.get(key), (str, int, float, bool))
            })
        out["delivered"] = safe_segments
    assembled = receipt.get("assembled")
    if isinstance(assembled, list):
        out["assembled"] = [
            {
                key: segment[key]
                for key in (
                    "segment_id",
                    "source_type",
                    "sha256",
                    "content_sha256",
                    "char_count",
                    "token_estimate",
                )
                if isinstance(segment, dict)
                and isinstance(segment.get(key), (str, int, float, bool))
            }
            for segment in assembled
            if isinstance(segment, dict)
        ]
    rendered = _dict(receipt.get("rendered"))
    if rendered:
        out["rendered"] = {
            key: rendered[key]
            for key in ("sha256", "char_count", "token_estimate")
            if isinstance(rendered.get(key), (str, int, float, bool))
        }
    termination = _dict(receipt.get("termination"))
    if termination:
        out["termination"] = {
            key: termination[key]
            for key in ("reason", "reason_raw", "generated_tokens")
            if isinstance(termination.get(key), (str, int, float, bool))
        }
    return out


def _safe_output_contract(run: dict) -> dict | None:
    contract = _dict(run.get("output_contract"))
    if not contract:
        return None
    requested = contract.get("requested_schema")
    requested_bytes = _canonical_bytes(requested) if isinstance(requested, dict) else None
    parser = _dict(contract.get("parser_runtime"))
    outcome = _dict(contract.get("outcome"))
    out = {
        "requested_schema": (
            {
                "sha256": _sha256_bytes(requested_bytes),
                "bytes": len(requested_bytes),
            }
            if requested_bytes is not None else None
        ),
        "parser_runtime": {
            key: parser[key]
            for key in ("name", "version")
            if isinstance(parser.get(key), (str, int, float, bool))
        } or None,
        "outcome": {
            key: outcome[key]
            for key in ("status", "kind", "error_code")
            if isinstance(outcome.get(key), (str, int, float, bool))
        } or None,
    }
    return out


def metadata_receipt(run: dict, coordinate: dict) -> dict:
    """Return one metadata-only receipt with no verbatim model/user content."""
    meta = _dict(run.get("meta"))
    response = _content_fingerprint(run.get("response"))
    messages = run.get("messages")
    messages_hash = None
    if isinstance(messages, list):
        messages_bytes = _canonical_bytes(messages)
        messages_hash = {
            "sha256": _sha256_bytes(messages_bytes),
            "bytes": len(messages_bytes),
            "count": len(messages),
        }
    receipt = {
        "schema_version": BUNDLE_SCHEMA,
        "privacy": PRIVACY_TIER,
        "run_id": run.get("id"),
        "coordinate": {
            key: coordinate.get(key)
            for key in ("suite", "case", "variant", "seed", "status")
        },
        "created_at": run.get("created_at"),
        "identity": _safe_identity(run),
        "generation": {
            field: meta[field]
            for field in _GENERATION_FIELDS
            if isinstance(meta.get(field), (str, int, float, bool))
        },
        "finish_reason": (
            run.get("finish_reason")
            if isinstance(run.get("finish_reason"), str) else None
        ),
        "input_fingerprint": messages_hash,
        "output_fingerprint": response,
        "context_receipt": _safe_context_receipt(run),
        "output_contract": _safe_output_contract(run),
    }
    return receipt


def _cell_key(value: dict) -> tuple:
    return tuple(value.get(key) for key in ("suite", "case", "variant", "seed"))


def _indexed_entries(report: dict) -> list[dict]:
    index = _dict(report.get("receipt_index"))
    if index.get("privacy") != PRIVACY_TIER:
        raise CIBundleError(
            "CI report receipt_index must declare privacy='metadata_only'")
    entries = index.get("entries")
    if not isinstance(entries, list):
        raise CIBundleError("CI report has no usable receipt_index.entries")
    out = []
    seen = set()
    for position, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CIBundleError(f"receipt_index entry {position} is not an object")
        key = _cell_key(entry)
        if any(item is None for item in key):
            raise CIBundleError(f"receipt_index entry {position} has an incomplete coordinate")
        if key in seen:
            raise CIBundleError(f"receipt_index repeats coordinate {key!r}")
        seen.add(key)
        out.append(dict(entry))
    return out


def _zip_entry(name: str, payload: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, _FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o600 << 16
    info.create_system = 3
    return info, payload


def build_indexed_bundle(report: dict, evidence: dict, out_path: str) -> dict:
    """Atomically write a deterministic metadata-only ZIP from indexed runs.

    Missing or inconsistent embedded evidence remains an explicit manifest
    entry.  It never causes a guessed run path or a lookup in the local journal.
    """
    if not isinstance(report, dict) or report.get("schema_version") != "clozn.ci-report.v1":
        raise CIBundleError("report must be a clozn.ci-report.v1 document")
    if not isinstance(evidence, dict) or evidence.get("schema_version") != "clozn.experiment.result.v0":
        raise CIBundleError("evidence must be a clozn.experiment.result.v0 document")
    if not isinstance(out_path, str) or not out_path:
        raise CIBundleError("a non-empty output path is required")

    entries = _indexed_entries(report)
    cells = {
        _cell_key(cell): cell
        for cell in evidence.get("cells") or []
        if isinstance(cell, dict)
    }
    receipt_payloads = []
    manifest_entries = []
    for position, indexed in enumerate(entries):
        manifest_entry = {
            key: indexed.get(key)
            for key in ("suite", "case", "variant", "seed", "status")
        }
        indexed_run_id = indexed.get("run_id")
        if not isinstance(indexed_run_id, str) or not indexed_run_id:
            manifest_entry["evidence_unavailable"] = (
                indexed.get("evidence_unavailable")
                or "receipt index has no recorded run ID"
            )
            manifest_entries.append(manifest_entry)
            continue
        cell = cells.get(_cell_key(indexed))
        embedded = _dict(cell.get("run")) if isinstance(cell, dict) else {}
        cell_run_id = cell.get("run_id") if isinstance(cell, dict) else None
        if cell_run_id != indexed_run_id or embedded.get("id") != indexed_run_id:
            manifest_entry["run_id"] = indexed_run_id
            manifest_entry["evidence_unavailable"] = (
                "indexed run ID does not match embedded experiment evidence"
            )
            manifest_entries.append(manifest_entry)
            continue

        receipt = metadata_receipt(embedded, indexed)
        payload = _canonical_bytes(receipt)
        receipt_name = f"receipts/{position:05d}-{_sha256_bytes(indexed_run_id.encode())[:16]}.json"
        manifest_entry.update({
            "run_id": indexed_run_id,
            "receipt": receipt_name,
            "sha256": _sha256_bytes(payload),
        })
        manifest_entries.append(manifest_entry)
        receipt_payloads.append((receipt_name, payload))

    report_bytes = _canonical_bytes(report)
    evidence_bytes = _canonical_bytes(evidence)
    manifest = {
        "schema_version": BUNDLE_SCHEMA,
        "privacy": PRIVACY_TIER,
        "source": {
            "ci_report_sha256": _sha256_bytes(report_bytes),
            "experiment_result_sha256": _sha256_bytes(evidence_bytes),
        },
        "entries": manifest_entries,
    }
    manifest_bytes = _canonical_bytes(manifest)

    destination = os.path.abspath(os.path.expanduser(out_path))
    parent = os.path.dirname(destination)
    os.makedirs(parent, exist_ok=True)
    temporary = None
    try:
        fd, temporary = tempfile.mkstemp(prefix=".clozn-receipts-", suffix=".zip", dir=parent)
        os.close(fd)
        with zipfile.ZipFile(temporary, "w") as archive:
            info, payload = _zip_entry("index.json", manifest_bytes)
            archive.writestr(info, payload)
            for name, payload in sorted(receipt_payloads):
                info, payload = _zip_entry(name, payload)
                archive.writestr(info, payload)
        os.replace(temporary, destination)
        temporary = None
    except OSError as exc:
        raise CIBundleError(f"could not write receipt bundle: {type(exc).__name__}") from None
    finally:
        if temporary is not None:
            try:
                os.remove(temporary)
            except OSError:
                pass

    with open(destination, "rb") as handle:
        archive_bytes = handle.read()
    return {
        "ok": True,
        "path": destination,
        "sha256": _sha256_bytes(archive_bytes),
        "size_bytes": len(archive_bytes),
        "privacy": PRIVACY_TIER,
        "indexed_entries": len(manifest_entries),
        "bundled_runs": len(receipt_payloads),
        "evidence_unavailable": sum(
            1 for entry in manifest_entries if entry.get("evidence_unavailable")),
    }


__all__ = [
    "BUNDLE_SCHEMA",
    "CIBundleError",
    "PRIVACY_TIER",
    "build_indexed_bundle",
    "metadata_receipt",
]
