from __future__ import annotations

import json

import clozn.settings as settings


def test_unknown_settings_are_ignored_and_removed_on_current_write(tmp_path, monkeypatch):
    target = tmp_path / "studio_settings.json"
    target.write_text(json.dumps({"sampling": True, "unknown": "value"}), encoding="utf-8")
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(target))

    assert settings.get_setting("unknown", "default") == "default"
    assert settings.set_setting("sample_top_k", 20) is True
    assert json.loads(target.read_text(encoding="utf-8")) == {
        "sampling": True,
        "sample_top_k": 20,
    }


def test_unknown_setting_cannot_be_persisted(tmp_path, monkeypatch):
    target = tmp_path / "studio_settings.json"
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(target))

    assert settings.set_setting("unknown", True) is False
    assert not target.exists()
