"""Read-only route coverage for persisted stable text-span projections."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import clozn.runs.store as runlog
from clozn.runs.context_receipt import build_context_receipt
from clozn.runs.text_span_addresses import (
    project_context_addresses,
    project_influence_addresses,
)
from clozn.server.routes import span_addresses as route


class Handler:
    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _run(run_id: str = "run_spans") -> dict:
    messages = [{"role": "user", "content": "PRIVATE CONTEXT", "source_id": "doc-1"}]
    return {
        "id": run_id,
        "messages": messages,
        "response": "PRIVATE ANSWER",
        "final_prompt": "<user>PRIVATE CONTEXT</user>",
        "context_receipt": build_context_receipt(
            messages=messages,
            assembled_messages=messages,
            final_prompt="<user>PRIVATE CONTEXT</user>",
            run_id=run_id,
            privacy="full",
        ),
    }


def _influence() -> dict:
    source = "PRIVATE CONTEXT"
    answer = "PRIVATE ANSWER"
    return {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": {
            "name": "teacher_forced_matched_context_replacement",
            "mode": "forced_score_intervention",
            "claim_limit": "the recorded continuation under this exact intervention",
            "caveat": "does not prove source correctness or reveal hidden reasoning",
        },
        "identity": {"run_id": "run_spans"},
        "prompt_sources": [{
            "id": "p.m000",
            "source_kind": "assembled_message",
            "segment_id": "seg_1111111111111111",
            "client_source_id": "doc-1",
            "selected": True,
            "start": 0,
            "end": len(source),
            "text": source,
        }],
        "prompt_spans": [{
            "id": "p.m000.c000",
            "parent_id": "p.m000",
            "segment_id": "seg_1111111111111111",
            "client_source_id": "doc-1",
            "start": 0,
            "end": 7,
            "text": "PRIVATE",
        }],
        "answer": {"scored_text": answer},
        "answer_spans": [{
            "id": "a.t0000",
            "token_index": 0,
            "start": 0,
            "end": len(answer),
            "text": answer,
        }],
        "links": [],
    }


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def test_route_ids_match_the_pure_context_and_influence_projections(monkeypatch):
    run = _run()
    influence = _influence()
    run["influence_map"] = influence
    before = copy.deepcopy(run)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler()
    assert route.try_get(h, f"/runs/{run['id']}/span-addresses") is True
    assert h.status == 200
    expected = [
        *project_context_addresses(run, privacy="metadata_only"),
        *project_influence_addresses(run["id"], influence, privacy="metadata_only"),
    ]
    assert [item["address_id"] for item in h.body["addresses"]] == [
        item["address_id"] for item in expected
    ]
    assert run == before
    assert h.body["privacy"] == "metadata_only"
    assert "PRIVATE CONTEXT" not in repr(h.body)
    assert "PRIVATE ANSWER" not in repr(h.body)


def test_route_redacted_run_leaks_no_text_and_keeps_unresolved_refs(monkeypatch):
    run = _run("run_redacted")
    run["messages"] = []
    run["response"] = ""
    run["final_prompt"] = None
    run["redaction"] = {"status": "redacted"}
    run["flags"] = ["redacted"]
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler()
    route.try_get(h, f"/runs/{run['id']}/span-addresses")

    assert h.status == 200
    assert "PRIVATE CONTEXT" not in repr(h.body)
    assert all(item["resolution"]["state"] == "redacted" for item in h.body["addresses"])
    assert all("canonical" not in item["resolution"] for item in h.body["addresses"])
    assert h.body["source_artifacts"][0]["native_status"] == "redacted"
    assert h.body["source_artifacts"][0]["privacy"] == "redacted"


def test_route_projects_legacy_messages_without_rewriting_them(monkeypatch):
    run = {
        "id": "run_legacy",
        "messages": [{"role": "user", "content": "OLD PRIVATE PROMPT"}],
        "context_receipt": {
            "schema": "clozn.context_receipt.v1",
            "delivered": {
                "messages": [{"role": "user", "content": "OLD PRIVATE PROMPT"}],
            },
        },
    }
    before = copy.deepcopy(run)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler()
    route.try_get(h, f"/runs/{run['id']}/span-addresses")

    assert h.status == 200
    assert run == before
    assert h.body["addresses"][0]["native_ref"]["collection"] == "run.messages"
    assert "OLD PRIVATE PROMPT" not in repr(h.body)


@pytest.mark.parametrize(
    ("influence_value", "native_status", "reason_fragment"),
    [
        (None, "not_recorded", "no persisted influence"),
        ({"unexpected": True}, "failed", "unsupported or missing schema"),
        (
            {"schema": "clozn.context_answer_influence.v1", "status": "ok"},
            "failed",
            "does not satisfy its native schema",
        ),
        (
            {"unavailable": "influence map blob missing", "sha256": "a" * 64},
            "unavailable",
            "blob missing",
        ),
    ],
)
def test_missing_invalid_and_blob_unavailable_influence_are_distinct(
    monkeypatch, influence_value, native_status, reason_fragment,
):
    run = _run()
    if influence_value is not None:
        run["influence_map"] = influence_value
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)

    h = Handler()
    route.try_get(h, f"/runs/{run['id']}/span-addresses")

    assert h.status == 200
    source = next(
        item for item in h.body["source_artifacts"]
        if "influence" in item["schema"]
    )
    assert source["native_status"] == native_status
    assert source["available"] is False
    assert reason_fragment in source["reason"]
    assert h.body["addresses"]  # valid context evidence remains usable


def test_actual_blob_backed_influence_is_loaded_without_worker_or_measurement(isolated):
    messages = [{"role": "user", "content": "PRIVATE CONTEXT", "source_id": "doc-1"}]
    rid = runlog.record(
        source="test",
        messages=messages,
        assembled_messages=messages,
        final_prompt="<user>PRIVATE CONTEXT</user>",
        response="PRIVATE ANSWER",
        trace={},
    )
    assert rid
    run = runlog.get_run(rid)
    influence = _influence()
    influence["identity"]["run_id"] = rid
    run["influence_map"] = influence
    assert runlog.replace_run(run) is True

    with runlog._connect() as db:
        payload = json.loads(db.execute(
            "SELECT payload_json FROM runs WHERE id = ?", (rid,),
        ).fetchone()["payload_json"])
    assert "influence_map_ref" in payload
    assert "influence_map" not in payload

    h = Handler()
    route.try_get(h, f"/runs/{rid}/span-addresses")

    assert h.status == 200
    source = next(item for item in h.body["source_artifacts"] if "influence" in item["schema"])
    assert source["native_status"] == "ok"
    assert any(
        item["native_ref"]["collection"] == "influence.answer_spans"
        for item in h.body["addresses"]
    )
    assert "PRIVATE CONTEXT" not in repr(h.body)
    assert "PRIVATE ANSWER" not in repr(h.body)


def test_actual_corrupt_blob_returns_context_addresses_and_unavailable_source(isolated):
    rid = runlog.record(
        source="test",
        messages=[{"role": "user", "content": "context"}],
        assembled_messages=[{"role": "user", "content": "context"}],
        final_prompt="context",
        response="answer",
        trace={},
    )
    run = runlog.get_run(rid)
    run["influence_map"] = _influence()
    assert runlog.replace_run(run) is True
    with runlog._connect() as db:
        payload = json.loads(db.execute(
            "SELECT payload_json FROM runs WHERE id = ?", (rid,),
        ).fetchone()["payload_json"])
    digest = payload["influence_map_ref"]["sha256"]
    Path(runlog._blob_path(digest)).write_bytes(b"corrupt")

    h = Handler()
    route.try_get(h, f"/runs/{rid}/span-addresses")

    assert h.status == 200
    source = next(item for item in h.body["source_artifacts"] if "influence" in item["schema"])
    assert source["native_status"] == "unavailable"
    assert "corrupt" in source["reason"]
    assert source["artifact_sha256"] == digest
    assert any(
        item["native_ref"]["collection"] == "context_receipt.delivered"
        for item in h.body["addresses"]
    )


def test_route_not_found_and_autoload_registration(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: None)
    h = Handler()
    assert route.try_get(h, "/runs/missing/span-addresses") is True
    assert h.status == 404

    from clozn.server import app as server
    assert route in server._GET_ROUTES
    assert server._GET_ROUTES.index(route) < server._GET_ROUTES.index(server._runs_fallback_routes)


def test_route_contract_failure_does_not_echo_private_exception_text(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr(
        "clozn.runs.text_span_addresses.build_persisted_text_span_addresses",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("PRIVATE MALFORMED SOURCE")
        ),
    )

    h = Handler()
    assert route.try_get(h, f"/runs/{run['id']}/span-addresses") is True

    assert h.status == 500
    assert h.body == {
        "error": "run span addresses could not be composed",
        "code": "span_address_contract_invalid",
    }
    assert "PRIVATE MALFORMED SOURCE" not in repr(h.body)
