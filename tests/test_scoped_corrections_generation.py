"""Generation-gateway integration tests for confirmed F5 corrections."""
from __future__ import annotations

import time

import clozn.runs.store as store
from clozn.runs import corrections
from clozn.server import app as server_app
from clozn.server import generation_gateway


class _Sub:
    def identity_meta(self):
        return {"model_sha256": "a" * 64}


def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", str(tmp_path / "runs"))
    store._schema_verified.clear()


def test_scoped_correction_injection_is_confirmed_and_content_is_not_used_for_selection(
    tmp_path, monkeypatch
):
    _isolated(tmp_path, monkeypatch)
    doc = corrections.draft_correction(
        scope_kind="session", scope_value="generation-session", correction_type="style",
        content="Use short paragraphs.",
    )
    messages = [{"role": "user", "content": "Explain this."}]
    handler = type("Handler", (), {"headers": {"X-Clozn-Session-Id": "generation-session"}})()
    monkeypatch.setattr(server_app, "SUB", _Sub())

    unchanged, evidence = generation_gateway.apply_scoped_corrections(handler, messages)
    assert unchanged == messages
    assert evidence is None
    assert not hasattr(handler, "_correction_resolution")

    corrections.confirm_correction(doc["id"])
    applied, evidence = generation_gateway.apply_scoped_corrections(handler, messages)
    assert messages == [{"role": "user", "content": "Explain this."}]
    assert applied[0]["role"] == "system"
    assert "Use short paragraphs." in applied[0]["content"]
    assert applied[1:] == messages
    assert evidence["applied_corrections"][0]["correction_id"] == doc["id"]
    assert "content" not in evidence["applied_corrections"][0]
    assert handler._correction_resolution["applied"][0]["correction_id"] == doc["id"]


def test_log_run_attaches_the_same_resolution_to_receipt_and_event_ledger(tmp_path, monkeypatch):
    _isolated(tmp_path, monkeypatch)
    monkeypatch.setattr(server_app, "SUB", None)
    doc = corrections.draft_correction(
        scope_kind="global_local", correction_type="output_format", content="Answer as JSON.",
    )
    corrections.confirm_correction(doc["id"])
    handler = object.__new__(server_app.make_handler())
    handler.headers = {"User-Agent": "pytest"}
    original = [{"role": "user", "content": "Return a value."}]
    corrected = [{"role": "system", "content": "Answer as JSON."}, *original]
    corrected, evidence = generation_gateway.apply_scoped_corrections(handler, original)
    rid = handler._log_run(
        "openai_api", original, "{\"value\": 1}", "model", time.time(),
        mem_out={"assembled_messages": corrected, "final_prompt": "Answer as JSON.\n\nReturn a value."},
    )
    run = store.get_run(rid)
    assert run["applied_corrections"] == evidence["applied_corrections"]
    assert run["context_receipt"]["applied_corrections"] == evidence["applied_corrections"]
    assert run["messages"] == original
    exported = corrections.export_correction(doc["id"])
    assert any(event.get("event_type") == "applied" and event.get("run_id") == rid
               for event in exported["events"])


def test_flatten_messages_for_native_preserves_order_and_text():
    assert generation_gateway.flatten_messages_for_native([
        {"role": "system", "content": "Answer as JSON."},
        {"role": "user", "content": "What is 2+2?"},
    ]) == "Answer as JSON.\n\nWhat is 2+2?"
