"""receipt_privacy -- how much of a context receipt's OWN duplicated content is retained.

Distinct from clozn.runs.capture_mode (which governs per-token TRACE depth) and distinct from the
run-level redact/delete lifecycle in clozn.runs.mutations (which governs run['messages'] etc.). This
module governs only what clozn.runs.context_receipt.build_context_receipt() writes into a run's
`context_receipt` field, which today only ever duplicates content already present on the run
('survived.assembled_messages', 'survived.final_prompt') -- everything this tier hides is still
reachable via the run's own fields at 'full' redaction, exactly like today; a stricter tier trims what
the RECEIPT repeats, not what the run itself stores.

Four tiers, in the shape feature 06's spec asks for:

  full            everything build_context_receipt can capture -- today's default behavior.
  metadata_only   segment types/labels/order/hashes/reasons kept; full text (final_prompt,
                  assembled_messages) dropped from the receipt.
  hashes_only     segment ids/hashes/reason kept; everything else metadata-shaped drops too
                  (no source_label, no byte counts).
  off             the receipt is not built at all beyond the required schema_version/run_id/privacy
                  marker -- an explicit "disabled", never a silently empty-looking full receipt.

The setting lives in the shared studio_settings.json, but mutations use this
module's byte-exact preview/CAS/transaction path. Reads retain settings.py's
never-raise default behavior; writes refuse malformed state and stale previews.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from datetime import datetime, timezone
from uuid import uuid4

import clozn.settings as settings

TIERS = ("full", "metadata_only", "hashes_only", "off")
DEFAULT = "full"
_KEY = "receipt_privacy"
TRANSACTION_SCHEMA = "clozn.receipt-privacy-transaction.v1"
_TRANSACTION_ID = re.compile(r"^rp_[0-9a-f]{32}$")


class PrivacyMutationError(ValueError):
    """A privacy-setting mutation could not be completed safely."""


class PrivacyDriftError(PrivacyMutationError):
    """The settings bytes no longer match the caller's preview."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _target_path() -> Path:
    return Path(os.path.abspath(os.path.expanduser(settings.SETTINGS_PATH)))


def _transaction_dir() -> Path:
    return _target_path().parent / "transactions" / "receipt_privacy"


def _transaction_path(transaction_id: str) -> Path:
    value = str(transaction_id or "")
    if not _TRANSACTION_ID.fullmatch(value):
        raise PrivacyMutationError("transaction_id is invalid")
    return _transaction_dir() / f"{value}.json"


def _atomic_write_bytes(path: Path, value: bytes) -> None:
    """Atomically replace ``path`` with exact bytes, including backup restores."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        dir=str(path.parent), prefix=".tmp-receipt-privacy-", suffix=".bin",
    )
    try:
        try:
            os.chmod(temporary, 0o600)
        except OSError:
            pass
        with os.fdopen(fd, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def _read_state() -> tuple[Path, bool, bytes, dict]:
    target = _target_path()
    if target.is_symlink():
        raise PrivacyMutationError(f"refusing to mutate symlinked settings file: {target}")
    if not target.exists():
        return target, False, b"", {}
    if not target.is_file():
        raise PrivacyMutationError(f"settings path is not a regular file: {target}")
    try:
        raw = target.read_bytes()
        decoded = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PrivacyMutationError(
            f"settings file is unreadable or invalid JSON and was left unchanged: {exc}"
        ) from None
    if not isinstance(decoded, dict):
        raise PrivacyMutationError("settings file must contain a JSON object and was left unchanged")
    return target, True, raw, decoded


def _normalized_tier(value) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in TIERS:
        raise PrivacyMutationError(f"unknown tier (want one of {list(TIERS)})")
    return normalized


def preview_tier(name: str) -> dict:
    """Return a byte-exact mutation preview that can be echoed into ``apply_tier``."""
    normalized = _normalized_tier(name)
    target, existed, before, document = _read_state()
    updated = dict(document)
    updated[_KEY] = normalized
    if document.get(_KEY) == normalized:
        proposed = before
    else:
        proposed = json.dumps(updated).encode("utf-8")
    current = {"exists": existed, "tier": tier()}
    if existed:
        current["sha256"] = _sha256(before)
        current["bytes"] = len(before)
    return {
        "target": str(target),
        "tier": normalized,
        "current": current,
        "proposed": {
            "sha256": _sha256(proposed),
            "bytes": len(proposed),
            "tier": normalized,
        },
        "changed": before != proposed or not existed,
        "expected": {
            "exists": existed,
            **({"sha256": _sha256(before)} if existed else {}),
        },
    }


def apply_tier(name: str, *, expected_exists: bool, expected_sha256: str | None = None) -> dict:
    """CAS-apply one previewed privacy tier and persist an undo transaction.

    An existing target requires its exact prior SHA-256. A nonexistent target
    requires ``expected_exists=False`` and no fabricated hash.
    """
    normalized = _normalized_tier(name)
    if not isinstance(expected_exists, bool):
        raise PrivacyMutationError("expected existence must be a boolean")
    if expected_exists:
        if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise PrivacyMutationError("expected sha256 is required for an existing settings file")
    elif expected_sha256 is not None:
        raise PrivacyMutationError("expected sha256 must be omitted when the settings file did not exist")

    preview = preview_tier(normalized)
    current = preview["current"]
    if current["exists"] != expected_exists:
        raise PrivacyDriftError("settings file existence changed after preview; refusing stale apply")
    if expected_exists and current.get("sha256") != expected_sha256:
        raise PrivacyDriftError("settings file hash changed after preview; refusing stale apply")
    if not preview["changed"]:
        return {
            "status": "unchanged",
            "tier": normalized,
            "target": preview["target"],
            "current_sha256": current["sha256"],
            "proposed_sha256": preview["proposed"]["sha256"],
        }

    target, existed, before, document = _read_state()
    # Close the ordinary preview/apply drift window before creating backups.
    if existed != expected_exists or (existed and _sha256(before) != expected_sha256):
        raise PrivacyDriftError("settings file changed while applying; refusing stale apply")
    updated = dict(document)
    updated[_KEY] = normalized
    proposed = json.dumps(updated).encode("utf-8")
    after_sha256 = _sha256(proposed)
    transaction_id = f"rp_{uuid4().hex}"
    transaction_path = _transaction_path(transaction_id)
    backup_path = transaction_path.with_suffix(".prior")

    transaction = {
        "schema_version": TRANSACTION_SCHEMA,
        "transaction_id": transaction_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "target": str(target),
        "target_existed": existed,
        **({"before_sha256": _sha256(before), "backup_path": str(backup_path)} if existed else {}),
        "after_sha256": after_sha256,
        "tier": normalized,
    }
    from clozn import schemas
    from clozn._io import atomic_write_json
    schemas.validate(transaction, TRANSACTION_SCHEMA)

    backup_written = False
    target_written = False
    try:
        if existed:
            _atomic_write_bytes(backup_path, before)
            backup_written = True
            if _sha256(backup_path.read_bytes()) != transaction["before_sha256"]:
                raise PrivacyMutationError("prior-settings backup verification failed")
        _atomic_write_bytes(target, proposed)
        target_written = True
        if _sha256(target.read_bytes()) != after_sha256:
            raise PrivacyMutationError("settings verification failed after atomic apply")
        transaction_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            str(transaction_path), transaction,
            ensure_ascii=False, indent=2, sort_keys=True,
        )
    except BaseException:
        if target_written:
            try:
                if existed:
                    _atomic_write_bytes(target, before)
                elif target.exists():
                    target.unlink()
            except OSError:
                pass
        if backup_written:
            try:
                backup_path.unlink()
            except OSError:
                pass
        raise
    return {
        "status": "applied",
        "tier": normalized,
        "target": str(target),
        **({"before_sha256": transaction["before_sha256"]} if existed else {}),
        "after_sha256": after_sha256,
        "transaction_id": transaction_id,
        "transaction_path": str(transaction_path),
    }


def undo_tier(transaction_id: str) -> dict:
    """Restore exact prior settings bytes, or remove a file created by apply."""
    transaction_path = _transaction_path(transaction_id)
    if transaction_path.is_symlink() or not transaction_path.is_file():
        raise PrivacyMutationError(f"no receipt-privacy transaction found for {transaction_id}")
    try:
        transaction = json.loads(transaction_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrivacyMutationError(f"transaction receipt is unreadable: {exc}") from None
    from clozn import schemas
    schemas.validate(transaction, TRANSACTION_SCHEMA)
    if transaction.get("transaction_id") != transaction_id:
        raise PrivacyMutationError("transaction receipt identity does not match its filename")

    target = _target_path()
    if os.path.abspath(transaction["target"]) != str(target):
        raise PrivacyMutationError("transaction targets a different settings file")
    existed = transaction["target_existed"]
    if target.is_symlink() or (target.exists() and not target.is_file()):
        raise PrivacyDriftError("settings target is no longer the recorded regular file")
    if not target.exists():
        if not existed:
            return {"status": "already_undone", "transaction_id": transaction_id, "target": str(target)}
        raise PrivacyDriftError("settings file was removed after apply; refusing unsafe undo")

    current = target.read_bytes()
    current_sha256 = _sha256(current)
    if existed and current_sha256 == transaction.get("before_sha256"):
        return {"status": "already_undone", "transaction_id": transaction_id, "target": str(target)}
    if current_sha256 != transaction["after_sha256"]:
        raise PrivacyDriftError("settings file changed after apply; refusing to overwrite external edits")

    if existed:
        backup_path = Path(transaction["backup_path"])
        if (
            backup_path.parent != transaction_path.parent
            or backup_path != transaction_path.with_suffix(".prior")
            or backup_path.is_symlink()
            or not backup_path.is_file()
        ):
            raise PrivacyMutationError("recorded prior-settings backup is unavailable")
        prior = backup_path.read_bytes()
        if _sha256(prior) != transaction["before_sha256"]:
            raise PrivacyDriftError("prior-settings backup changed; refusing unsafe restore")
        _atomic_write_bytes(target, prior)
        if _sha256(target.read_bytes()) != transaction["before_sha256"]:
            raise PrivacyMutationError("settings verification failed after undo restore")
        status = "restored"
    else:
        target.unlink()
        status = "removed"
    return {
        "status": status,
        "transaction_id": transaction_id,
        "target": str(target),
        **({"restored_sha256": transaction["before_sha256"]} if existed else {}),
    }


def tier() -> str:
    """The active receipt-privacy tier; absent / unknown / garbage -> "full" (today's existing behavior,
    unchanged for anyone who never touches this setting)."""
    v = str(settings.get_setting(_KEY, DEFAULT) or "").strip().lower()
    return v if v in TIERS else DEFAULT


def set_tier(name: str) -> bool:
    """Compatibility wrapper: atomically preview+CAS apply, never raising.

    New mutation callers should use ``preview_tier`` and ``apply_tier`` so the
    expected prior state is explicit. Existing internal callers retain their
    historical boolean contract without bypassing transaction safety.
    """
    try:
        preview = preview_tier(name)
        expected = preview["expected"]
        apply_tier(
            name,
            expected_exists=expected["exists"],
            expected_sha256=expected.get("sha256"),
        )
        return True
    except Exception:
        return False


def includes_full_text(name: str | None = None) -> bool:
    """Does this tier retain full rendered/assembled TEXT on the receipt (final_prompt,
    assembled_messages)?  Only "full" does."""
    return (name if name is not None else tier()) == "full"


def includes_segment_metadata(name: str | None = None) -> bool:
    """Does this tier retain per-segment source_label/byte counts, or only id+hash+reason?"""
    return (name if name is not None else tier()) in ("full", "metadata_only")


def builds_receipt(name: str | None = None) -> bool:
    """Does this tier build a receipt at all? Only "off" declines."""
    return (name if name is not None else tier()) != "off"
