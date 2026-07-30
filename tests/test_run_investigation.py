from __future__ import annotations

import clozn.runs.store as runlog
from clozn.runs.context_receipt import build_context_receipt
from clozn.runs.investigation import build
from clozn.server.routes import investigation as route


class Handler:
    def __init__(self, sub=None):
        self._inj_sub = sub
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _run(run_id="run_current", *, parent_run_id=None):
    messages = [{"role": "system", "content": "Be concise."},
                {"role": "user", "content": "Explain this."}]
    receipt = build_context_receipt(
        messages=messages,
        assembled_messages=messages,
        final_prompt="RENDERED PRIVATE PROMPT",
        run_id=run_id,
        privacy="full",
    )
    return {
        "id": run_id,
        "model": "fixture-model",
        "messages": messages,
        "response": "A response.",
        "context_receipt": receipt,
        "identity": {"model_sha256": "a" * 64},
        "meta": {},
        "timing": {"started_at": 2.0, "ended_at": 3.0},
        "created_ts": 2.0,
        "recorded_ts": 3.0,
        "parent_run_id": parent_run_id,
    }


def _registry():
    return {
        "schema": "clozn.action-registry.v1",
        "actions": [{
            "id": "less-verbose",
            "label": "Make it shorter",
            "description": "Retry with a concise response policy.",
            "backends": [{"type": "prompt_policy", "available": True}],
            "scope_eligibility": [{"scope": "once", "available": True}],
        }],
    }


def test_build_is_model_free_privacy_safe_and_declares_actions():
    run = _run()
    out = build(
        run,
        related_runs=[run],
        corrective_registry=_registry(),
        scoring_available=True,
    )
    assert out["schema_version"] == "clozn.run-investigation.v1"
    assert out["sections"]["received_context"]["state"] == "delivered_not_measured"
    assert out["sections"]["received_context"]["delivered"][0]["content_hash"]
    assert "survived" not in out["sections"]["received_context"]
    assert "RENDERED PRIVATE PROMPT" not in repr(out)
    span_section = out["sections"]["text_span_addresses"]
    assert span_section["state"] == "supported"
    assert span_section["privacy"] == "metadata_only"
    assert span_section["href"] == f"/runs/{run['id']}/span-addresses"
    assert span_section["influence_native_status"] == "not_recorded"
    assert next(item for item in out["actions"]
                if item["id"] == "open_text_span_addresses")["method"] == "GET"
    influence = out["sections"]["prompt_source_influence"]
    assert influence["state"] == "delivered_not_measured"
    action = next(item for item in out["actions"]
                  if item["id"] == "measure_prompt_source_influence")
    assert action["availability"] == "ready"
    assert any(item["id"] == "corrective:less-verbose" for item in out["actions"])


def test_measured_effect_below_floor_and_unknown_native_state_fail_closed():
    run = _run()
    base = {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": {"name": "forced_score_matched_replacement"},
        "prompt_sources": [{"id": "p.m000", "text": "PRIVATE SOURCE"}],
        "prompt_spans": [{"id": "p.m000.c000", "text": "PRIVATE SOURCE"}],
        "answer_spans": [{"id": "a.t0000", "text": "PRIVATE ANSWER"}],
        "thresholds": {"cell_abs_delta_nats": 0.05},
        "summary": {},
    }
    run["influence_map"] = {**base, "links": [{"evidence_state": "observed"}]}
    below = build(run, related_runs=[run], corrective_registry=_registry())
    assert below["sections"]["prompt_source_influence"]["state"] == "below_measurement_floor"
    assert "PRIVATE SOURCE" not in repr(below)
    assert "PRIVATE ANSWER" not in repr(below)
    assert below["sections"]["prompt_source_influence"]["prompt_sources"][0]["text_sha256"]

    run["influence_map"] = {**base, "links": [{"evidence_state": "causally_supported"}]}
    measured = build(run, related_runs=[run], corrective_registry=_registry())
    assert measured["sections"]["prompt_source_influence"]["state"] == "measured_effect"

    run["influence_map"] = {**base, "links": [{"evidence_state": "mystery"}]}
    unknown = build(run, related_runs=[run], corrective_registry=_registry())
    section = unknown["sections"]["prompt_source_influence"]
    assert section["state"] == "inconclusive"
    assert "mystery" in section["reason"]


def test_parent_comparison_is_composed_without_generation():
    parent = _run("run_parent")
    current = _run(parent_run_id="run_parent")
    out = build(
        current,
        related_runs=[parent, current],
        corrective_registry=_registry(),
    )
    comparison = out["sections"]["comparisons"]
    assert comparison["state"] == "supported"
    assert comparison["reference_run_id"] == "run_parent"
    assert comparison["comparison"]["schema_version"] == "clozn.run-diff.v1"


def test_blob_unavailable_and_invalid_influence_remain_typed_without_breaking_spans():
    run = _run()
    run["influence_map"] = {
        "unavailable": "influence map blob corrupt (digest mismatch)",
        "sha256": "b" * 64,
    }
    unavailable = build(run, related_runs=[run], corrective_registry=_registry())
    influence = unavailable["sections"]["prompt_source_influence"]
    span_section = unavailable["sections"]["text_span_addresses"]
    assert influence["state"] == "unavailable"
    assert "corrupt" in influence["reason"]
    assert span_section["state"] == "supported"
    assert span_section["influence_native_status"] == "unavailable"
    assert "corrupt" in span_section["reason"]

    run["influence_map"] = {
        "schema": "clozn.unknown-influence.v9",
        "status": "ok",
        "available": True,
    }
    invalid = build(run, related_runs=[run], corrective_registry=_registry())
    assert invalid["sections"]["prompt_source_influence"]["state"] == "failed"
    assert invalid["sections"]["text_span_addresses"]["influence_native_status"] == "failed"


def test_route_is_read_only_and_never_calls_scorer(monkeypatch):
    current = _run()

    class Sub:
        steer = None

        def score_tokens(self, *_args, **_kwargs):
            raise AssertionError("GET investigation must not execute scoring")

    monkeypatch.setattr(runlog, "get_run", lambda run_id: current if run_id == current["id"] else None)
    monkeypatch.setattr(runlog, "iter_runs", lambda limit=200: iter([current]))
    h = Handler(Sub())
    assert route.try_get(h, f"/runs/{current['id']}/investigation") is True
    assert h.status == 200
    assert h.body["run_id"] == current["id"]
    assert next(item for item in h.body["actions"]
                if item["id"] == "measure_prompt_source_influence")["availability"] == "ready"


def test_route_not_found_and_autoload_registration(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _run_id: None)
    h = Handler()
    assert route.try_get(h, "/runs/missing/investigation") is True
    assert h.status == 404

    from clozn.server import app as server
    assert route in server._GET_ROUTES
    assert server._GET_ROUTES.index(route) < server._GET_ROUTES.index(server._runs_fallback_routes)
