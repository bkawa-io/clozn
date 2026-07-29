from __future__ import annotations

import json

import pytest

import clozn.settings as settings
from clozn.runs import receipt_privacy


def _use_settings(tmp_path, monkeypatch):
    target = tmp_path / "studio_settings.json"
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(target))
    return target


def _apply_preview(preview):
    return receipt_privacy.apply_tier(
        preview["tier"],
        expected_exists=preview["expected"]["exists"],
        expected_sha256=preview["expected"].get("sha256"),
    )


def test_new_settings_preview_apply_and_repeated_undo(tmp_path, monkeypatch):
    target = _use_settings(tmp_path, monkeypatch)
    preview = receipt_privacy.preview_tier("hashes_only")
    assert preview["target"] == str(target)
    assert preview["current"] == {"exists": False, "tier": "full"}
    assert preview["expected"] == {"exists": False}
    assert len(preview["proposed"]["sha256"]) == 64

    applied = _apply_preview(preview)
    assert applied["status"] == "applied"
    assert json.loads(target.read_text(encoding="utf-8")) == {"receipt_privacy": "hashes_only"}
    transaction = json.loads(
        receipt_privacy._transaction_path(applied["transaction_id"]).read_text(encoding="utf-8")
    )
    assert transaction["target_existed"] is False
    assert "before_sha256" not in transaction
    assert "backup_path" not in transaction
    assert None not in transaction.values()

    undone = receipt_privacy.undo_tier(applied["transaction_id"])
    assert undone["status"] == "removed"
    assert not target.exists()
    assert receipt_privacy.undo_tier(applied["transaction_id"])["status"] == "already_undone"

    # A complete second cycle gets its own immutable transaction and remains reversible.
    second = _apply_preview(receipt_privacy.preview_tier("off"))
    assert second["transaction_id"] != applied["transaction_id"]
    assert receipt_privacy.undo_tier(second["transaction_id"])["status"] == "removed"


def test_existing_settings_undo_restores_exact_prior_bytes(tmp_path, monkeypatch):
    target = _use_settings(tmp_path, monkeypatch)
    prior = b'{\r\n  "other": "preserved",\r\n  "receipt_privacy": "full"\r\n}\r\n'
    target.write_bytes(prior)
    preview = receipt_privacy.preview_tier("metadata_only")
    assert preview["current"]["exists"] is True
    assert preview["current"]["bytes"] == len(prior)
    assert preview["expected"]["sha256"] == preview["current"]["sha256"]

    applied = _apply_preview(preview)
    transaction = json.loads(
        receipt_privacy._transaction_path(applied["transaction_id"]).read_text(encoding="utf-8")
    )
    assert transaction["before_sha256"] == preview["current"]["sha256"]
    assert transaction["backup_path"]
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "other": "preserved",
        "receipt_privacy": "metadata_only",
    }

    undone = receipt_privacy.undo_tier(applied["transaction_id"])
    assert undone["status"] == "restored"
    assert target.read_bytes() == prior
    assert receipt_privacy.undo_tier(applied["transaction_id"])["status"] == "already_undone"


def test_apply_refuses_existence_and_hash_drift(tmp_path, monkeypatch):
    target = _use_settings(tmp_path, monkeypatch)
    absent = receipt_privacy.preview_tier("off")
    target.write_text('{"external": true}', encoding="utf-8")
    with pytest.raises(receipt_privacy.PrivacyDriftError, match="existence changed"):
        _apply_preview(absent)
    assert json.loads(target.read_text(encoding="utf-8")) == {"external": True}

    existing = receipt_privacy.preview_tier("off")
    target.write_text('{"external": "edited"}', encoding="utf-8")
    with pytest.raises(receipt_privacy.PrivacyDriftError, match="hash changed"):
        _apply_preview(existing)
    assert json.loads(target.read_text(encoding="utf-8")) == {"external": "edited"}


def test_undo_refuses_post_apply_drift(tmp_path, monkeypatch):
    target = _use_settings(tmp_path, monkeypatch)
    target.write_text('{"other": 1}', encoding="utf-8")
    applied = _apply_preview(receipt_privacy.preview_tier("off"))
    target.write_text('{"other": 2, "receipt_privacy": "off"}', encoding="utf-8")

    with pytest.raises(receipt_privacy.PrivacyDriftError, match="external edits"):
        receipt_privacy.undo_tier(applied["transaction_id"])
    assert json.loads(target.read_text(encoding="utf-8"))["other"] == 2


def test_invalid_existing_settings_are_never_silently_overwritten(tmp_path, monkeypatch):
    target = _use_settings(tmp_path, monkeypatch)
    target.write_bytes(b"{not-json")
    with pytest.raises(receipt_privacy.PrivacyMutationError, match="left unchanged"):
        receipt_privacy.preview_tier("metadata_only")
    assert target.read_bytes() == b"{not-json"
