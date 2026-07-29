from __future__ import annotations

from pathlib import Path

import pytest

from clozn.cli.commands._connector import (
    GenericOpenAIConnector,
    OllamaSDKConnector,
    OpenWebUIConnector,
)


@pytest.mark.parametrize(
    ("factory", "expected"),
    [
        (GenericOpenAIConnector, "OPENAI_BASE_URL"),
        (OpenWebUIConnector, "OPENAI_API_BASE_URLS"),
        (OllamaSDKConnector, "OLLAMA_HOST"),
    ],
)
def test_environment_connectors_preview_apply_and_undo(tmp_path, factory, expected):
    target = tmp_path / "client.env"
    target.write_text("PRESERVE_ME=yes\n", encoding="utf-8")
    before = target.read_bytes()
    state = tmp_path / "state.json"
    connector = factory(config_path=target)
    kwargs = {
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "clozn-local",
        "api_key": "local-key",
        "state_path": state,
    }

    preview = connector.plan(**kwargs)
    assert preview.status == "dry_run"
    assert target.read_bytes() == before
    assert not state.exists()

    transaction = connector.apply(preview, **kwargs)
    assert transaction.report["status"] == "updated"
    assert expected in target.read_text(encoding="utf-8")
    assert "PRESERVE_ME=yes" in target.read_text(encoding="utf-8")
    assert state.is_file()

    undone = connector.undo(state_path=state)
    assert undone.status == "restored"
    assert target.read_bytes() == before
    assert not state.exists()


def test_environment_connector_undo_refuses_external_drift(tmp_path):
    target = tmp_path / "client.env"
    state = tmp_path / "state.json"
    connector = GenericOpenAIConnector(config_path=target)
    kwargs = dict(
        base_url="http://127.0.0.1:8080/v1",
        model="clozn-local",
        api_key="key",
        state_path=state,
    )
    connector.apply(connector.plan(**kwargs), **kwargs)
    target.write_text("EXTERNAL=edit\n", encoding="utf-8")

    with pytest.raises(ValueError, match="external edits"):
        connector.undo(state_path=state)

    assert target.read_text(encoding="utf-8") == "EXTERNAL=edit\n"
    assert state.is_file()


def test_open_webui_values_are_sourceable_and_do_not_expose_unrelated_text(tmp_path):
    target = Path(tmp_path) / "open-webui.env"
    connector = OpenWebUIConnector(config_path=target)
    kwargs = dict(
        base_url="http://127.0.0.1:8080/v1",
        model="clozn-local",
        api_key="local-key",
        state_path=tmp_path / "state.json",
    )
    connector.apply(connector.plan(**kwargs), **kwargs)
    text = target.read_text(encoding="utf-8")
    assert 'OPENAI_API_BASE_URLS="http://127.0.0.1:8080/v1"' in text
    assert 'OPENAI_API_KEYS="local-key"' in text
    assert 'DEFAULT_MODELS="clozn-local"' in text


def test_environment_connector_apply_refuses_drift_since_plan(tmp_path):
    target = tmp_path / "client.env"
    state = tmp_path / "state.json"
    target.write_text("PRESERVE=yes\n", encoding="utf-8")
    connector = GenericOpenAIConnector(config_path=target)
    kwargs = {
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "clozn-local",
        "api_key": "key",
        "state_path": state,
    }
    preview = connector.plan(**kwargs)
    target.write_text("EXTERNAL=edit\n", encoding="utf-8")

    with pytest.raises(ValueError, match="changed since preview"):
        connector.apply(preview, **kwargs)

    assert target.read_text(encoding="utf-8") == "EXTERNAL=edit\n"
    assert not state.exists()
