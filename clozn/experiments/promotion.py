"""Safe, model-free promotion of one experiment cell into a regression-suite draft."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import time
import uuid
from typing import Any, Mapping

from clozn.testkit import promotion as regression


ABSENT_HASH = "absent"
MAX_DOCUMENT_BYTES = 64 * 1024


class PromotionServiceError(ValueError):
    pass


class DestinationDriftError(PromotionServiceError):
    pass


_SECRET_PATTERNS = (
    ("private_key", re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----", re.IGNORECASE)),
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}", re.IGNORECASE)),
    ("api_key", re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
        r"AKIA[A-Z0-9]{16})\b")),
    ("home_path", re.compile(
        r"(?:(?:[A-Za-z]:\\Users\\[^\\\s\"']+)|(?:/(?:home|Users)/[^/\s\"']+))")),
    ("email", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("opaque_secret", re.compile(
        r"(?<![A-Za-z0-9])(?:[A-Fa-f0-9]{48,}|[A-Za-z0-9+/=_-]{40,})(?![A-Za-z0-9])")),
)


def _pointer(parts: tuple[str, ...]) -> str:
    return "/" + "/".join(part.replace("~", "~0").replace("/", "~1") for part in parts)


def _masked(value: str) -> str:
    if len(value) <= 8:
        return "[sensitive]"
    return f"{value[:3]}…{value[-3:]}"


def _finding(kind: str, path: str, start: int, end: int, value: str) -> dict:
    material = f"{kind}\0{path}\0{start}\0{end}\0".encode("utf-8") + value.encode("utf-8")
    return {
        "id": "finding_" + hashlib.sha256(material).hexdigest()[:16],
        "kind": kind,
        "path": path,
        "start": start,
        "end": end,
        "preview": _masked(value),
    }


def _walk_strings(value: Any, parts: tuple[str, ...] = ()):
    if isinstance(value, str):
        yield _pointer(parts), value
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _walk_strings(item, (*parts, str(index)))
    elif isinstance(value, dict):
        for key in sorted(value):
            yield from _walk_strings(value[key], (*parts, str(key)))


def scan_case(case: Mapping[str, Any]) -> list[dict]:
    """Return deterministic, non-mutating findings over runnable/evaluated case content."""
    if not isinstance(case, Mapping):
        raise PromotionServiceError("candidate case must be an object")
    # Provenance hashes are intentionally excluded: they are expected opaque digests, not runnable
    # text, and treating every source.sha256 as a secret would make the scanner permanently noisy.
    material = {
        key: case[key] for key in ("prompt", "messages", "model", "expect") if key in case
    }
    findings = []
    for path, text in _walk_strings(material):
        byte_count = len(text.encode("utf-8"))
        if byte_count > MAX_DOCUMENT_BYTES:
            findings.append(_finding("oversized_document", path, 0, len(text), text))
        occupied: set[tuple[int, int]] = set()
        for kind, pattern in _SECRET_PATTERNS:
            for match in pattern.finditer(text):
                span = match.span()
                # Prefer a more specific earlier pattern over the generic opaque-token matcher.
                if kind == "opaque_secret" and any(
                        start <= span[0] and span[1] <= end for start, end in occupied):
                    continue
                findings.append(_finding(kind, path, span[0], span[1], match.group(0)))
                occupied.add(span)
    findings.sort(key=lambda item: (item["path"], item["start"], item["end"], item["kind"]))
    return findings


def destination_hash(path: str | os.PathLike[str]) -> str:
    try:
        with open(path, "rb") as handle:
            digest = hashlib.sha256()
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
            return digest.hexdigest()
    except FileNotFoundError:
        return ABSENT_HASH
    except OSError as exc:
        raise PromotionServiceError(f"could not read promotion destination: {exc}") from exc


def promotion_directory() -> str:
    return os.path.expanduser("~/.clozn/regression-suites")


def resolve_destination(name: str) -> str:
    """Resolve one API-safe artifact name inside the configured promotion directory."""
    if (not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*\.json", name)
            or Path(name).name != name):
        raise PromotionServiceError(
            "destination must be a plain .json artifact name (no directory components)")
    return os.path.join(promotion_directory(), name)


def _coordinate(body: Mapping[str, Any]) -> tuple[str, str, str, int]:
    values = tuple(body.get(field) for field in ("suite", "case", "variant", "seed"))
    role, case_name, variant, seed = values
    if role not in ("target", "guard"):
        raise PromotionServiceError("suite must be 'target' or 'guard'")
    if not isinstance(case_name, str) or not case_name:
        raise PromotionServiceError("case must be a non-empty string")
    if not isinstance(variant, str) or not variant:
        raise PromotionServiceError("variant must be a non-empty string")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise PromotionServiceError("seed must be an integer")
    return role, case_name, variant, seed


def _candidate(result: Mapping[str, Any], request: Mapping[str, Any]) -> tuple[dict, dict, str]:
    role, case_name, variant, seed = _coordinate(request)
    matches = [
        cell for cell in result.get("cells") or []
        if (cell.get("suite"), cell.get("case"), cell.get("variant"), cell.get("seed"))
        == (role, case_name, variant, seed)
    ]
    if len(matches) != 1:
        raise PromotionServiceError(
            "the selected coordinate must identify exactly one complete experiment cell")
    cell = matches[0]
    run = cell.get("run")
    if cell.get("status") == "error" or not isinstance(run, Mapping):
        raise PromotionServiceError("an error or missing-run cell cannot be promoted")
    try:
        draft = regression.create_suite_draft(
            str(request.get("suite_name") or result.get("name") or "experiment-promotion"), [run])
        case = draft["cases"][0]
        case_name_override = request.get("case_name")
        if case_name_override is not None:
            case = regression.edit_case(case, name=case_name_override)
        replacements = request.get("replacements")
        if replacements is not None:
            case = regression.redact_case(case, replacements)
    except regression.PromotionError as exc:
        raise PromotionServiceError(str(exc)) from exc
    return case, dict(cell), role


def _load_destination(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise PromotionServiceError(f"could not read destination suite: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise PromotionServiceError(f"destination suite is invalid JSON: {exc}") from exc
    try:
        suite = regression.validate_suite(raw, require_frozen=False)
    except regression.PromotionError as exc:
        raise PromotionServiceError(f"destination suite is not an editable draft: {exc}") from exc
    if suite.get("state") != "draft":
        raise PromotionServiceError("destination suite is frozen and cannot be edited")
    return suite


def _next_suite(path: str, case: dict, request: Mapping[str, Any]) -> tuple[dict, dict]:
    existing = _load_destination(path)
    if existing is None:
        name = request.get("suite_name") or Path(path).stem
        candidate = {
            "schema_version": regression.REGRESSION_SUITE_SCHEMA,
            "name": name,
            "state": "draft",
            "cases": [case],
        }
        before_count = 0
    else:
        if any(item.get("name") == case["name"] for item in existing["cases"]):
            raise PromotionServiceError(
                f"destination already contains a case named {case['name']!r}")
        candidate = dict(existing)
        candidate["cases"] = [*existing["cases"], case]
        before_count = len(existing["cases"])
    try:
        candidate = regression.validate_suite(candidate, require_frozen=False)
    except regression.PromotionError as exc:
        raise PromotionServiceError(str(exc)) from exc
    return candidate, {
        "operation": "create" if existing is None else "append",
        "before_case_count": before_count,
        "after_case_count": len(candidate["cases"]),
        "added_case": case["name"],
    }


def preview_promotion(
        result: Mapping[str, Any], destination: str | os.PathLike[str],
        request: Mapping[str, Any]) -> dict:
    """Build the exact proposed destination without writing or creating directories."""
    destination = os.path.abspath(os.fspath(destination))
    case, cell, role = _candidate(result, request)
    proposed, diff = _next_suite(destination, case, request)
    findings = scan_case(case)
    return {
        "schema_version": "clozn.promotion-preview.v1",
        "experiment_id": result.get("experiment_id"),
        "source_run_id": cell.get("run_id"),
        "role": role,
        "candidate_case": case,
        "destination": destination,
        "destination_diff": diff,
        "expected_destination_hash": destination_hash(destination),
        "proposed_destination_sha256": hashlib.sha256(
            json.dumps(
                proposed, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest(),
        "redaction_findings": findings,
        "required_acknowledgements": [finding["id"] for finding in findings],
    }


def _atomic_bytes(path: str, payload: bytes) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".clozn-tmp-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _write_transaction(path: str, transaction: dict) -> None:
    payload = json.dumps(
        transaction, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False,
    ).encode("utf-8") + b"\n"
    _atomic_bytes(path, payload)


def apply_promotion(
        result: Mapping[str, Any], destination: str | os.PathLike[str],
        request: Mapping[str, Any]) -> dict:
    """Apply a reviewed preview with expected-hash drift refusal and rollback on failure."""
    destination = os.path.abspath(os.fspath(destination))
    expected = request.get("expected_destination_hash")
    if not isinstance(expected, str) or (
            expected != ABSENT_HASH and not re.fullmatch(r"[0-9a-f]{64}", expected)):
        raise PromotionServiceError(
            "expected_destination_hash must be 'absent' or a lowercase SHA-256 digest")
    if destination_hash(destination) != expected:
        raise DestinationDriftError("promotion destination changed since preview")

    preview = preview_promotion(result, destination, request)
    if preview["expected_destination_hash"] != expected:
        raise DestinationDriftError("promotion destination changed while preparing apply")
    required = set(preview["required_acknowledgements"])
    acknowledged = request.get("acknowledged_findings", [])
    if (not isinstance(acknowledged, list)
            or any(not isinstance(item, str) for item in acknowledged)):
        raise PromotionServiceError("acknowledged_findings must be an array of finding IDs")
    missing = sorted(required - set(acknowledged))
    if missing:
        raise PromotionServiceError(
            f"redaction findings require explicit acknowledgement: {', '.join(missing)}")

    case, cell, role = _candidate(result, request)
    proposed, diff = _next_suite(destination, case, request)
    payload = json.dumps(
        proposed, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False,
    ).encode("utf-8") + b"\n"
    transaction_id = "promotion_" + uuid.uuid4().hex
    directory = os.path.dirname(destination)
    os.makedirs(directory, exist_ok=True)
    lock_path = destination + ".lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise DestinationDriftError("another promotion is already editing this destination") from exc
    backup_path = None
    replaced = False
    completed = False
    try:
        with os.fdopen(lock_fd, "w", encoding="utf-8") as lock:
            lock.write(transaction_id)
            lock.flush()
            os.fsync(lock.fileno())
        if destination_hash(destination) != expected:
            raise DestinationDriftError("promotion destination changed before atomic apply")
        if expected != ABSENT_HASH:
            backup_path = destination + f".bak.{transaction_id}"
            with open(destination, "rb") as source:
                _atomic_bytes(backup_path, source.read())
        _atomic_bytes(destination, payload)
        replaced = True
        written_hash = destination_hash(destination)
        transaction = {
            "schema_version": "clozn.promotion-transaction.v1",
            "transaction_id": transaction_id,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "experiment_id": result.get("experiment_id"),
            "source_run_id": cell.get("run_id"),
            "role": role,
            "destination": destination,
            "previous_destination_hash": expected,
            "destination_sha256": written_hash,
            "backup": backup_path,
            "redaction_finding_ids": preview["required_acknowledgements"],
            "diff": diff,
        }
        transaction_path = os.path.join(
            directory, ".clozn-transactions", transaction_id + ".json")
        _write_transaction(transaction_path, transaction)
        completed = True
        return {**transaction, "transaction_path": transaction_path}
    except BaseException:
        if replaced and not completed:
            if backup_path is not None:
                os.replace(backup_path, destination)
            else:
                try:
                    os.unlink(destination)
                except FileNotFoundError:
                    pass
        elif backup_path is not None and not completed:
            try:
                os.unlink(backup_path)
            except FileNotFoundError:
                pass
        raise
    finally:
        try:
            os.unlink(lock_path)
        except OSError:
            pass


__all__ = [
    "ABSENT_HASH", "DestinationDriftError", "MAX_DOCUMENT_BYTES", "PromotionServiceError",
    "apply_promotion", "destination_hash", "preview_promotion", "promotion_directory",
    "resolve_destination", "scan_case",
]
