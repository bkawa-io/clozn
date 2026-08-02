"""Q4--Q8 lab adapters and transactional artifact lifecycle.

The adapters are intentionally thin.  Heavy fitting/calibration remains an explicit command owned
by the lab environment; Clozn records its output, validates the resulting model-bound manifest, and
only then makes installation possible.  No command is passed through a shell and no product runtime
imports the lab dependencies.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping, Sequence

from clozn import schemas
from clozn._io import atomic_write_json
from clozn.artifacts import contracts
from .pipeline import RUN_SCHEMA, run_external_step, validate_jlens_artifact

LAB_SCHEMA = "clozn.qualification-lab-step.v1"
INSTALL_SCHEMA = "clozn.qualification-install.v1"
DIAL_SCHEMA = "clozn.dial-calibration.v1"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _install_receipt(status: str, artifact_type: str, model_sha256: str, path: Path) -> dict[str, Any]:
    receipt = {
        "schema_version": INSTALL_SCHEMA,
        "generated_at": _now(),
        "status": status,
        "artifact_type": artifact_type,
        "model_sha256": model_sha256,
        "path": str(path),
    }
    schemas.validate(receipt, INSTALL_SCHEMA)
    return receipt


def _lab_receipt(step_id: str, status: str, *, model_sha256: str | None = None,
                 evidence: Any = None, reason: str | None = None,
                 artifact: Mapping[str, Any] | None = None) -> dict[str, Any]:
    document: dict[str, Any] = {
        "schema_version": LAB_SCHEMA,
        "generated_at": _now(),
        "step_id": step_id,
        "boundary": "lab",
        "status": status,
        "model_sha256": model_sha256,
        "evidence": evidence if evidence is not None else {},
        "reason": reason,
        "artifact": dict(artifact) if artifact is not None else None,
        "receipt_sha256": None,
    }
    document["receipt_sha256"] = _digest({**document, "receipt_sha256": None})
    schemas.validate(document, LAB_SCHEMA)
    return document


def run_dial_calibration(model_identity: Mapping[str, Any], argv: Sequence[str], *,
                         output_dir: str, timeout: float = 3600.0) -> dict[str, Any]:
    """Q4: run an explicit calibration command and validate its ``dials`` manifest."""
    sha = str(model_identity.get("sha256") or "").lower() or None
    command = run_external_step("dials", argv, timeout=timeout, output_dir=output_dir)
    if command["status"] != "passed":
        return _lab_receipt("dials", "failed", model_sha256=sha,
                            evidence=command.get("evidence"), reason=command.get("reason"))
    return _validate_generated_artifact("dials", model_identity, output_dir,
                                        command.get("evidence") or {})


def run_jlens_fit(model_identity: Mapping[str, Any], argv: Sequence[str], *,
                  output_dir: str, timeout: float = 7200.0) -> dict[str, Any]:
    """Q5: run an explicit fit command and validate the model-bound J-lens manifest."""
    sha = str(model_identity.get("sha256") or "").lower() or None
    command = run_external_step("jlens", argv, timeout=timeout, output_dir=output_dir)
    if command["status"] != "passed":
        return _lab_receipt("jlens", "failed", model_sha256=sha,
                            evidence=command.get("evidence"), reason=command.get("reason"))
    return _validate_generated_artifact("jlens", model_identity, output_dir,
                                        command.get("evidence") or {})


def record_dial_calibration(model_identity: Mapping[str, Any], report: Mapping[str, Any], *,
                            command_evidence: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Normalize ``engine_autocalibrate.py`` output into a Q4 lab receipt.

    A calibration with no usable dial is real negative evidence, not a successful empty artifact.  A
    mixed report remains visible as a partial success through its per-dial lists; installation still
    requires a separate reviewed manifest containing the exact model identity.
    """
    if not isinstance(report, Mapping) or not report:
        return _lab_receipt("dials", "failed", model_sha256=str(model_identity.get("sha256") or "").lower() or None,
                            reason="calibration report is empty or not an object")
    usable = []
    unusable = []
    for name, value in report.items():
        if not isinstance(name, str) or not isinstance(value, Mapping):
            return _lab_receipt("dials", "failed", model_sha256=str(model_identity.get("sha256") or "").lower() or None,
                                reason="calibration report contains an invalid dial row")
        works = value.get("works", value.get("range_valid"))
        if works is True and isinstance(value.get("usable_range"), list):
            usable.append(name)
        else:
            unusable.append(name)
    evidence = {
        "usable_dials": usable,
        "unusable_dials": unusable,
        "dial_count": len(report),
        "report": dict(report),
    }
    if command_evidence:
        evidence["command"] = dict(command_evidence)
    status = "passed" if usable else "failed"
    reason = None if usable else "no dial cleared the coherence/effect/null calibration gate"
    return _lab_receipt("dials", status,
                        model_sha256=str(model_identity.get("sha256") or "").lower() or None,
                        evidence=evidence, reason=reason)


def install_dial_calibration(model_identity: Mapping[str, Any], report: Mapping[str, Any], *,
                             root: str) -> dict[str, Any]:
    """Q4/Q7: install a successful calibration under the exact GGUF hash.

    The runtime accepts only ``~/.clozn/models/<sha>/dial_calibration.json`` for a real model.  The
    historical root-level ``~/.clozn/dial_calibration.json`` is deliberately never read or promoted
    by this function.  Reports with no usable dial are refused rather than becoming an empty-looking
    success artifact.
    """
    sha = str(model_identity.get("sha256") or "").lower()
    if len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha):
        raise ValueError("model identity must contain a lowercase SHA-256 digest")
    receipt = record_dial_calibration(model_identity, report)
    if receipt["status"] != "passed":
        raise ValueError(receipt.get("reason") or "dial calibration did not produce a usable dial")
    dials = {
        name: {**dict(value), "works": bool(value.get("works", value.get("range_valid")))}
        for name, value in report.items()
    }
    document = {
        "schema_version": DIAL_SCHEMA,
        "generated_at": _now(),
        "model_sha256": sha,
        "method": "engine_autocalibrate.py coherence/effect/shuffled-null sweep",
        "dials": dials,
    }
    schemas.validate(document, DIAL_SCHEMA)
    target = Path(root).resolve() / "models" / sha / "dial_calibration.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            with target.open(encoding="utf-8") as handle:
                existing = json.load(handle)
            schemas.validate(existing, DIAL_SCHEMA)
        except Exception as exc:
            raise ValueError(f"refusing to overwrite an invalid scoped dial calibration: {exc}") from None
        if existing.get("model_sha256") != sha:
            raise ValueError("refusing to overwrite a calibration for a different model identity")
        if _digest(existing.get("dials")) == _digest(dials):
            return _install_receipt("already_present", "dials", sha, target)
        raise ValueError("refusing to overwrite a different calibration under the same model identity")
    atomic_write_json(str(target), document, indent=2, ensure_ascii=False)
    return _install_receipt("installed", "dials", sha, target)


def rollback_dial_calibration(transaction: Mapping[str, Any]) -> dict[str, Any]:
    """Remove one model-scoped dial calibration installed by ``install_dial_calibration``.

    Dial calibration is stored under ``models/<sha>/dial_calibration.json`` rather than the
    generic ``<artifact_type>/<sha>`` registry layout. Keep its rollback guard explicit so a
    receipt cannot be redirected to an arbitrary file or a different model.
    """
    if not isinstance(transaction, Mapping) or transaction.get("status") != "installed":
        raise ValueError("only an installed dial calibration can be rolled back")
    sha = str(transaction.get("model_sha256") or "").lower()
    path = Path(str(transaction.get("path") or "")).resolve()
    if (len(sha) != 64 or any(char not in "0123456789abcdef" for char in sha)
            or path.name != "dial_calibration.json"
            or path.parent.name != sha
            or path.parent.parent.name != "models"):
        raise ValueError("rollback target is not a recognized model-scoped dial calibration")
    if not path.is_file():
        raise ValueError("rollback target does not exist")
    try:
        with path.open(encoding="utf-8") as handle:
            document = json.load(handle)
    except Exception as exc:
        raise ValueError(f"rollback target is not valid dial calibration: {exc}") from None
    if not isinstance(document, Mapping) or str(document.get("model_sha256") or "").lower() != sha:
        raise ValueError("rollback target does not match its model identity")
    path.unlink()
    return _install_receipt("rolled_back", "dials", sha, path)


def _validate_generated_artifact(artifact_type: str, model_identity: Mapping[str, Any],
                                 output_dir: str, command_evidence: Mapping[str, Any]) -> dict[str, Any]:
    try:
        manifest_path = Path(output_dir) / "manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        checked = contracts.validate_artifact_manifest(
            manifest, model_identity, output_dir, expected_type=artifact_type,
        )
    except Exception as exc:
        return _lab_receipt(
            artifact_type, "failed", model_sha256=str(model_identity.get("sha256") or "").lower() or None,
            evidence={"command": dict(command_evidence)},
            reason=f"generated artifact is invalid: {type(exc).__name__}: {exc}",
        )
    return _lab_receipt(
        artifact_type, "passed", model_sha256=checked["model_sha256"],
        evidence={"command": dict(command_evidence), "validated": checked},
        artifact={"artifact_type": artifact_type, "directory": os.path.abspath(output_dir),
                  "files": checked["files"]},
    )


def record_jlens_validation(model_identity: Mapping[str, Any], artifact_dir: str) -> dict[str, Any]:
    """Normalize an existing Q5 J-lens validation into the lab-step schema."""
    result = validate_jlens_artifact(model_identity, artifact_dir)
    sha = str(model_identity.get("sha256") or "").lower() or None
    if result.get("status") != "passed":
        return _lab_receipt("jlens", "failed", model_sha256=sha, reason=result.get("reason"))
    evidence = result.get("evidence") or {}
    return _lab_receipt(
        "jlens", "passed", model_sha256=sha, evidence=evidence,
        artifact={"artifact_type": "jlens", "directory": os.path.abspath(artifact_dir),
                  "files": evidence.get("files", [])},
    )


def run_battery(cells: Sequence[Mapping[str, Any]], *, previous: Mapping[str, Any] | None = None,
                timeout: float = 3600.0, output: str | None = None) -> dict[str, Any]:
    """Q6: run model-specific/cross-model cells with resume and partial failure visibility.

    Each cell is ``{"id": str, "argv": [str, ...], "output_dir": optional}``.  A previous receipt
    may be supplied; a passed cell is reused only when its exact command digest matches.
    """
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise ValueError("battery cells must be a sequence")
    old = previous.get("cells", {}) if isinstance(previous, Mapping) else {}
    if not old and isinstance(previous, Mapping):
        artifact = previous.get("artifact")
        old = artifact.get("cells", {}) if isinstance(artifact, Mapping) else {}
    old = old if isinstance(old, Mapping) else {}
    results: dict[str, Any] = {}
    for cell in cells:
        if not isinstance(cell, Mapping) or not isinstance(cell.get("id"), str):
            raise ValueError("each battery cell requires an id")
        cell_id = cell["id"]
        argv = cell.get("argv")
        if not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)) or not argv:
            raise ValueError(f"battery cell {cell_id!r} requires a non-empty argv")
        command_digest = _digest({"argv": list(argv), "output_dir": cell.get("output_dir"),
                                  "model_sha256": cell.get("model_sha256")})
        prior = old.get(cell_id)
        if isinstance(prior, Mapping) and prior.get("status") == "passed" and prior.get("command_sha256") == command_digest:
            results[cell_id] = dict(prior)
            results[cell_id]["resumed"] = True
            continue
        command = run_external_step(cell_id, argv, cwd=cell.get("cwd"),
                                    timeout=float(cell.get("timeout", timeout)),
                                    output_dir=cell.get("output_dir"))
        result = dict(command)
        result["command_sha256"] = command_digest
        result["resumed"] = False
        if cell.get("model") is not None:
            result["model"] = cell["model"]
        if cell.get("model_sha256") is not None:
            result["model_sha256"] = str(cell["model_sha256"]).lower()
        results[cell_id] = result
    passed = sum(item.get("status") == "passed" for item in results.values())
    failed = sum(item.get("status") != "passed" for item in results.values())
    model_rows = [
        {"id": cell_id, "model": value.get("model"), "model_sha256": value.get("model_sha256")}
        for cell_id, value in results.items()
        if value.get("model") is not None or value.get("model_sha256") is not None
    ]
    report = {
        "schema_version": LAB_SCHEMA,
        "generated_at": _now(),
        "step_id": "batteries",
        "boundary": "lab",
        "status": "passed" if results and failed == 0 else ("failed" if failed else "not_run"),
        "model_sha256": None,
        "evidence": {"cells_total": len(results), "cells_passed": passed, "cells_failed": failed,
                     "models": model_rows},
        "reason": None if failed == 0 and results else "battery has no passing cells or contains failures",
        "artifact": {"cells": results},
        "receipt_sha256": None,
    }
    report["receipt_sha256"] = _digest({**report, "receipt_sha256": None})
    schemas.validate(report, LAB_SCHEMA)
    if output:
        atomic_write_json(output, report, indent=2, ensure_ascii=False)
    return report


def install_artifact(model_identity: Mapping[str, Any], artifact_dir: str, *,
                     artifact_type: str, root: str) -> dict[str, Any]:
    """Q7: validate then atomically install one artifact, refusing identity overwrite."""
    if (not isinstance(artifact_type, str) or not artifact_type
            or Path(artifact_type).name != artifact_type or artifact_type in {".", ".."}):
        raise ValueError("artifact_type must be one safe path component")
    artifact_path = Path(artifact_dir).resolve()
    root_path = Path(root).resolve()
    manifest_path = artifact_path / "manifest.json"
    try:
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        checked = contracts.validate_artifact_manifest(
            manifest, model_identity, artifact_path, expected_type=artifact_type,
        )
    except Exception as exc:
        raise ValueError(f"artifact cannot be installed: {type(exc).__name__}: {exc}") from None
    model_sha = checked["model_sha256"]
    target = root_path / artifact_type / model_sha
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        try:
            with (target / "manifest.json").open(encoding="utf-8") as handle:
                existing = json.load(handle)
            existing_checked = contracts.validate_artifact_manifest(
                existing, model_identity, target, expected_type=artifact_type,
            )
        except Exception as exc:
            raise ValueError(f"refusing to overwrite an incompatible installed artifact: {exc}") from None
        if existing_checked == checked:
            return _install_receipt("already_present", artifact_type, model_sha, target)
        raise ValueError("refusing to overwrite a different artifact under the same model identity")
    staging = Path(tempfile.mkdtemp(prefix=".qualification-install-", dir=str(target.parent)))
    try:
        staged = staging / "payload"
        shutil.copytree(artifact_path, staged)
        os.replace(str(staged), str(target))
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return _install_receipt("installed", artifact_type, model_sha, target)


def rollback_artifact(transaction: Mapping[str, Any]) -> dict[str, Any]:
    """Q7 rollback, guarded by the exact transaction path and model digest."""
    if not isinstance(transaction, Mapping) or transaction.get("status") != "installed":
        raise ValueError("only an installed transaction can be rolled back")
    path = Path(str(transaction.get("path") or "")).resolve()
    if not path.is_dir() or path.parent.name != str(transaction.get("artifact_type") or ""):
        raise ValueError("rollback target is not a recognized installed artifact directory")
    if path.name != str(transaction.get("model_sha256") or "").lower():
        raise ValueError("rollback target does not match its model identity")
    shutil.rmtree(path)
    return _install_receipt("rolled_back", str(transaction["artifact_type"]),
                            str(transaction["model_sha256"]).lower(), path)


def acceptance_fixture(*, model: str, core: Mapping[str, Any], calibration: Mapping[str, Any],
                       jlens: Mapping[str, Any], battery: Mapping[str, Any],
                       installs: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Q8 acceptance reducer: require every stage to be explicit and passed.

    This is intentionally an inspection/assembly helper; it does not invent a fit or start a worker.
    A real acceptance fixture can feed it the receipts produced by Q3-Q7.
    """
    stages = {"core": core, "calibration": calibration, "jlens": jlens, "battery": battery}
    statuses = {name: str(value.get("claims", {}).get("qualification_status")
                           or value.get("status") or "unknown")
                for name, value in stages.items()}
    core_model = core.get("model") if isinstance(core, Mapping) else None
    identity = core_model.get("identity") if isinstance(core_model, Mapping) else None
    anchor_sha = str(identity.get("sha256") or "").lower() if isinstance(identity, Mapping) else ""
    identity_mismatches: dict[str, list[str]] = {}
    for name, value in stages.items():
        candidates: list[str] = []
        if isinstance(value, Mapping) and value.get("model_sha256"):
            candidates.append(str(value["model_sha256"]).lower())
        if name == "battery" and isinstance(value, Mapping):
            evidence = value.get("evidence")
            for row in evidence.get("models", []) if isinstance(evidence, Mapping) else []:
                if isinstance(row, Mapping) and row.get("model_sha256"):
                    candidates.append(str(row["model_sha256"]).lower())
        if anchor_sha:
            mismatches = sorted({candidate for candidate in candidates if candidate != anchor_sha})
            if mismatches:
                identity_mismatches[name] = mismatches
                statuses[name] = "identity_mismatch"
    status = "passed" if statuses["core"] == "core_passed" and not identity_mismatches and all(
        statuses[name] == "passed" for name in ("calibration", "jlens", "battery")
    ) else "failed"
    document = {
        "schema_version": RUN_SCHEMA, "generated_at": _now(),
        "model": {"input": model, "identity": identity if isinstance(identity, Mapping) else None},
        "steps": [{"id": name, "boundary": "product" if name == "core" else "lab",
                    "status": "passed" if statuses[name] in {"passed", "core_passed"} else "failed",
                    "evidence": {"receipt": value,
                                 "identity_mismatch": identity_mismatches.get(name)}
                    } for name, value in stages.items()],
        "claims": {"qualification_status": "core_passed" if status == "passed" else "failed",
                   "generation_performed": bool(core.get("claims", {}).get("generation_performed")),
                   "artifacts_installed": bool(installs),
                   "note": ("Acceptance assembly reflects supplied Q3-Q7 receipts; it does not fit artifacts."
                             if not identity_mismatches else
                             "Acceptance refused because stage receipts do not share the core model identity." )},
        "receipt_sha256": None,
    }
    document["receipt_sha256"] = _digest({**document, "receipt_sha256": None})
    schemas.validate(document, RUN_SCHEMA)
    return document


__all__ = ["LAB_SCHEMA", "INSTALL_SCHEMA", "DIAL_SCHEMA", "run_dial_calibration", "record_dial_calibration",
           "install_dial_calibration", "rollback_dial_calibration", "run_jlens_fit", "record_jlens_validation",
           "run_battery", "install_artifact", "rollback_artifact", "acceptance_fixture"]
