"""Milestone E backend: GET /runs/<id>/tokens/<index>/workbench, model-free.

Mirrors tests/test_run_investigation.py's own conventions: plain-dict runs for the pure builder
(clozn.runs.token_workbench.build), monkeypatched clozn.runs.store for the route
(clozn.server.routes.token_workbench), no real sqlite I/O needed for either half. Nothing here boots an
engine or a worker -- `worker_ready`/`scoring_available` are always supplied as plain booleans, exactly
as the route itself computes them from local attribute lookups, never a network call.

Covers:
  * the five sections (run/token/context/comparison/readouts) and the four capabilities compose already-
    recorded evidence -- never a fresh measurement, even when everything is "available".
  * each capability keeps its OWN vocabulary -- exact_fork's snapshot_state, causal_trace/source_
    measurement's status, mechanistic_diff's always-present reason -- never flattened into one shared
    shape (a stray cross-artifact field is schema-rejected; see tests/fixtures/schemas/
    clozn.token-workbench.v1/invalid__unknown_capability_field.json for the same rule at the schema
    layer).
  * "never a bare false": every `available: false` capability across every scenario below carries a
    non-empty `reason`.
  * the route's 404/400/200 contract, that it never touches score_tokens/engine, that an unresolvable
    ?reference_run_id degrades (not 404s) the comparison/mechanistic_diff sections, and autoload
    registration.
"""
from __future__ import annotations

from copy import deepcopy

import pytest

import clozn.runs.store as runlog
from clozn.runs.investigation import build as build_investigation
from clozn.runs.token_workbench import SCHEMA_VERSION, build
from clozn.server.routes import token_workbench as route
from clozn import schemas


# =================================================================================== fixtures (plain dicts)
def _organic_run(run_id="run_current", **overrides):
    """A fully organic engine run: eligible for BOTH exact_fork and causal_trace, with a recorded
    alternative at position 1 so the token section has something to show."""
    values = {
        "id": run_id,
        "model": "fixture-model",
        "substrate": "engine",
        "source": "engine_chat",
        "messages": [{"role": "user", "content": "count"}],
        "response": "one two three",
        "final_prompt": "<prompt>",
        "trace": {
            "tokens": ["one", " two", " three"],
            "token_ids": [11, 22, 33],
            "confidence": [0.9, 0.8, 0.7],
            "logprobs": [-0.1, -0.2, -0.3],
            "alternatives": [[], [{"piece": " four", "token_id": 44, "prob": 0.02}], []],
        },
        "behavior": {"active_dials": {}},
        "meta": {
            "n_ctx": 4096, "device": "cpu", "prompt_tokens": 2, "stream": False,
            "decode": {"mode": "greedy", "temperature": 0.0, "seed": 0},
        },
        "identity": {
            "model_sha256": "a" * 64, "template_fingerprint": "b" * 16,
            "engine_build": "test-build", "white_box_flags": {},
        },
    }
    values.update(overrides)
    return values


def _investigation_for(run, related_runs=None):
    return build_investigation(
        run, related_runs=related_runs or [run], corrective_registry=None, scoring_available=False)


# ============================================================================================= schema shape
def test_build_validates_and_has_the_five_sections_and_four_capabilities():
    run = _organic_run()
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=True)
    schemas.validate(doc, SCHEMA_VERSION)
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["run_id"] == "run_current"
    assert doc["index"] == 1
    assert set(doc["sections"]) == {"run", "token", "context", "comparison", "readouts"}
    assert set(doc["capabilities"]) == {
        "exact_fork", "source_measurement", "causal_trace", "mechanistic_diff"}


def test_build_never_mutates_the_run_or_investigation_doc():
    run = _organic_run()
    investigation_doc = _investigation_for(run)
    before_run, before_inv = deepcopy(run), deepcopy(investigation_doc)
    build(run, 1, investigation_doc=investigation_doc, worker_ready=True)
    assert run == before_run
    assert investigation_doc == before_inv


def test_token_out_of_range_raises_value_error():
    run = _organic_run()
    with pytest.raises(ValueError, match="out of range"):
        build(run, 99, investigation_doc=_investigation_for(run), worker_ready=True)
    with pytest.raises(ValueError, match="out of range"):
        build(run, -1, investigation_doc=_investigation_for(run), worker_ready=True)


def test_run_with_no_trace_raises_value_error():
    run = {k: v for k, v in _organic_run().items() if k != "trace"}
    with pytest.raises(ValueError, match="no trace"):
        build(run, 0, investigation_doc=_investigation_for(run), worker_ready=True)


def test_missing_run_id_raises_value_error():
    with pytest.raises(ValueError, match="non-empty id"):
        build({"trace": {"tokens": ["a"]}}, 0, investigation_doc={}, worker_ready=True)


# ============================================================================================= token/readouts
def test_token_section_carries_position_piece_and_recorded_alternative():
    run = _organic_run()
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=True)
    token = doc["sections"]["token"]
    assert token["index"] == 1
    assert token["piece"] == " two"
    assert token["token_id"] == 22
    assert token["prefix_kept"] == "one"
    assert token["alternatives"] == [{"piece": " four", "token_id": 44, "prob": 0.02}]


def test_readouts_supported_when_measurements_recorded():
    run = _organic_run()
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=True)
    readouts = doc["sections"]["readouts"]
    assert readouts["state"] == "supported"
    assert readouts["measurements"]["confidence"] == 0.8
    assert readouts["measurements"]["logprob"] == -0.2


def test_readouts_unavailable_when_nothing_recorded_at_this_position():
    run = _organic_run(trace={"tokens": ["one", " two", " three"], "token_ids": [11, 22, 33]})
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=True)
    readouts = doc["sections"]["readouts"]
    assert readouts["state"] == "unavailable"
    assert readouts["reason"]
    assert readouts["measurements"] == {}


def test_workspace_readouts_filtered_to_this_position_only():
    run = _organic_run()
    run["trace"]["workspace_readouts"] = [
        {"position": 0, "provider": "jlens", "value": "not this one"},
        {"position": 1, "provider": "jlens", "value": "this one"},
    ]
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=True)
    workspace = doc["sections"]["readouts"]["workspace_readouts"]
    assert len(workspace) == 1
    assert workspace[0]["value"] == "this one"


# ============================================================================================= context
def test_context_section_is_investigations_received_context_verbatim():
    run = _organic_run()
    investigation_doc = _investigation_for(run)
    doc = build(run, 1, investigation_doc=investigation_doc, worker_ready=True)
    assert doc["sections"]["context"] == investigation_doc["sections"]["received_context"]


# ============================================================================================= comparison
def test_comparison_explicit_reference_run_id_overrides_auto_selection():
    run = _organic_run()
    parent = _organic_run(run_id="run_parent")
    explicit = _organic_run(run_id="run_explicit_ref")
    run["parent_run_id"] = "run_parent"
    doc = build(
        run, 1, investigation_doc=_investigation_for(run, related_runs=[run, parent, explicit]),
        related_runs=[run, parent, explicit],
        reference_run_id="run_explicit_ref", reference_run=explicit,
    )
    comparison = doc["sections"]["comparison"]
    assert comparison["state"] == "supported"
    assert comparison["reference_run_id"] == "run_explicit_ref"
    assert comparison["selection"] == {"mode": "explicit", "reference_run_id": "run_explicit_ref"}
    assert comparison["href"] == "/runs/compare?a=run_explicit_ref&b=run_current"


def test_comparison_auto_selects_parent_when_no_explicit_reference():
    run = _organic_run()
    parent = _organic_run(run_id="run_parent")
    run["parent_run_id"] = "run_parent"
    doc = build(
        run, 1, investigation_doc=_investigation_for(run, related_runs=[run, parent]),
        related_runs=[run, parent],
    )
    comparison = doc["sections"]["comparison"]
    assert comparison["state"] == "supported"
    assert comparison["reference_run_id"] == "run_parent"
    assert comparison["selection"]["mode"] == "parent"


def test_comparison_unavailable_when_no_reference_found_automatically():
    run = _organic_run()
    doc = build(run, 1, investigation_doc=_investigation_for(run), related_runs=[run])
    comparison = doc["sections"]["comparison"]
    assert comparison["state"] == "unavailable"
    assert "reference_run_id=" in comparison["reason"]


def test_comparison_unresolvable_explicit_reference_is_labeled_not_404d():
    run = _organic_run()
    doc = build(
        run, 1, investigation_doc=_investigation_for(run), related_runs=[run],
        reference_run_id="run_missing", reference_run=None,
    )
    comparison = doc["sections"]["comparison"]
    assert comparison["state"] == "unavailable"
    assert "run_missing" in comparison["reason"]


# ============================================================================================= exact_fork
def test_exact_fork_available_for_an_organic_run_with_a_ready_worker():
    run = _organic_run()
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=True)
    exact_fork = doc["capabilities"]["exact_fork"]
    assert exact_fork == {
        "available": True,
        "snapshot_state": "not_attempted",
        # Milestone F: points at the token-workbench action endpoint, not the pre-Milestone-F
        # /runs/<id>/fork route (still live, but no longer what this preview recommends).
        "action": {"method": "POST", "href": "/runs/run_current/tokens/1/fork"},
    }


def test_exact_fork_unavailable_when_no_worker_is_reachable():
    run = _organic_run()
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=False)
    exact_fork = doc["capabilities"]["exact_fork"]
    assert exact_fork["available"] is False
    assert exact_fork["snapshot_state"] == "worker_unreachable"
    assert exact_fork["reason"]


def test_exact_fork_unavailable_for_a_historical_run_missing_prompt_boundary():
    run = _organic_run()
    del run["meta"]["prompt_tokens"]
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=True)
    exact_fork = doc["capabilities"]["exact_fork"]
    assert exact_fork["available"] is False
    assert exact_fork["snapshot_state"] == "missing_prompt_boundary"
    assert exact_fork["reason"]


def test_exact_fork_unavailable_for_a_child_run_with_a_prior_intervention():
    run = _organic_run(parent_run_id="run_ancestor")
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=True)
    exact_fork = doc["capabilities"]["exact_fork"]
    assert exact_fork["available"] is False
    assert exact_fork["snapshot_state"] == "run_shape_ineligible"
    assert exact_fork["reason"]


def test_exact_fork_unavailable_missing_final_prompt():
    run = _organic_run()
    run["final_prompt"] = None
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=True)
    exact_fork = doc["capabilities"]["exact_fork"]
    assert exact_fork["available"] is False
    assert exact_fork["snapshot_state"] == "missing_final_prompt"


# ============================================================================================= causal_trace
def test_causal_trace_ready_for_an_organic_run_with_a_ready_worker():
    run = _organic_run()
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=True)
    causal = doc["capabilities"]["causal_trace"]
    assert causal == {
        "available": True,
        "status": "ready",
        # Milestone F: the token-workbench action endpoint carries `index` in the URL itself, so no
        # request_body hint is needed (unlike the pre-Milestone-F /runs/<id>/causal-trace route, which
        # took `position` in the body).
        "action": {"method": "POST", "href": "/runs/run_current/tokens/1/causal-trace"},
    }


def test_causal_trace_unavailable_without_a_worker_even_with_a_full_recorded_run():
    run = _organic_run()
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=False)
    causal = doc["capabilities"]["causal_trace"]
    assert causal["available"] is False
    assert causal["status"] == "unavailable"
    assert causal["reason"]


def test_causal_trace_unavailable_missing_response_text():
    run = _organic_run()
    run["response"] = ""
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=True)
    causal = doc["capabilities"]["causal_trace"]
    assert causal["available"] is False
    assert "response" in causal["reason"]


# ============================================================================================= source_measurement
def test_source_measurement_reuses_investigations_own_state_verbatim():
    run = _organic_run()
    investigation_doc = _investigation_for(run)
    doc = build(run, 1, investigation_doc=investigation_doc, worker_ready=True)
    source = doc["capabilities"]["source_measurement"]
    assert source["status"] == investigation_doc["sections"]["prompt_source_influence"]["state"]
    assert source["available"] is False  # nothing was ever measured on this fixture run
    assert source["reason"]


def test_source_measurement_available_when_investigation_reports_a_measured_effect():
    run = _organic_run()
    run["influence_map"] = {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": {"name": "teacher_forced_matched_context_replacement"},
        "prompt_sources": [], "prompt_spans": [], "answer_spans": [],
        "thresholds": {}, "summary": {},
        "links": [{"evidence_state": "causally_supported"}],
    }
    investigation_doc = _investigation_for(run)
    doc = build(run, 1, investigation_doc=investigation_doc, worker_ready=True)
    source = doc["capabilities"]["source_measurement"]
    assert source["status"] == "measured_effect"
    assert source["available"] is True
    assert "reason" not in source


# ============================================================================================= mechanistic_diff
def test_mechanistic_diff_unavailable_with_no_reference_selected():
    run = _organic_run()
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=True)
    mech = doc["capabilities"]["mechanistic_diff"]
    assert mech["available"] is False
    assert mech["reason"]
    assert "action" not in mech


def test_mechanistic_diff_unavailable_when_reference_is_the_same_model():
    run = _organic_run()
    same_model_reference = _organic_run(run_id="run_same_model")
    doc = build(
        run, 1, investigation_doc=_investigation_for(run), worker_ready=True,
        reference_run_id="run_same_model", reference_run=same_model_reference,
    )
    mech = doc["capabilities"]["mechanistic_diff"]
    assert mech["available"] is False
    assert "same model" in mech["reason"] or "model identity" in mech["reason"]


def test_mechanistic_diff_available_for_a_genuinely_different_model_reference():
    run = _organic_run()
    other_model = _organic_run(run_id="run_other_model", model="other-model")
    other_model["identity"]["model_sha256"] = "c" * 64
    doc = build(
        run, 1, investigation_doc=_investigation_for(run), worker_ready=True,
        reference_run_id="run_other_model", reference_run=other_model,
    )
    mech = doc["capabilities"]["mechanistic_diff"]
    assert mech["available"] is True
    assert mech["reason"]
    # Points at the token-workbench mechanistic-diff action, which runs this SAME pair-compatibility
    # gate authoritatively and queues managed-router execution when the gateway has a model registry.
    assert mech["action"] == {
        "method": "POST", "href": "/runs/run_current/tokens/1/mechanistic-diff",
        "request_body": {"reference_run_id": "run_other_model"},
    }


def test_mechanistic_diff_unresolvable_reference_is_labeled_not_404d():
    run = _organic_run()
    doc = build(
        run, 1, investigation_doc=_investigation_for(run), worker_ready=True,
        reference_run_id="run_missing", reference_run=None,
    )
    mech = doc["capabilities"]["mechanistic_diff"]
    assert mech["available"] is False
    assert "run_missing" in mech["reason"]


# ============================================================================================= the anti-flattening rule
def test_capabilities_keep_their_own_distinct_vocabulary_not_one_shared_shape():
    run = _organic_run()
    doc = build(run, 1, investigation_doc=_investigation_for(run), worker_ready=True)
    caps = doc["capabilities"]
    assert "snapshot_state" in caps["exact_fork"] and "status" not in caps["exact_fork"]
    assert "status" in caps["source_measurement"] and "snapshot_state" not in caps["source_measurement"]
    assert "status" in caps["causal_trace"] and "snapshot_state" not in caps["causal_trace"]
    assert "status" not in caps["mechanistic_diff"] and "snapshot_state" not in caps["mechanistic_diff"]
    assert "reason" in caps["mechanistic_diff"]  # mechanistic_diff's reason is ALWAYS present


@pytest.mark.parametrize(
    ("worker_ready", "reference_run_id", "reference_run", "delete_meta_key"),
    [
        (False, None, None, None),
        (True, None, None, "prompt_tokens"),
        (True, "run_missing", None, None),
    ],
)
def test_never_a_bare_false_every_unavailable_capability_has_a_reason(
    worker_ready, reference_run_id, reference_run, delete_meta_key,
):
    run = _organic_run()
    if delete_meta_key:
        del run["meta"][delete_meta_key]
    doc = build(
        run, 1, investigation_doc=_investigation_for(run), worker_ready=worker_ready,
        reference_run_id=reference_run_id, reference_run=reference_run,
    )
    for name, capability in doc["capabilities"].items():
        if capability.get("available") is False:
            assert isinstance(capability.get("reason"), str) and capability["reason"], (
                f"{name} is unavailable but carries no typed reason: {capability}")
    schemas.validate(doc, SCHEMA_VERSION)


# ============================================================================================= route
class Handler:
    def __init__(self, sub=None, path=""):
        self._inj_sub = sub
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def test_route_happy_path_returns_a_valid_document(monkeypatch):
    run = _organic_run()

    class Sub:
        steer = None
        engine = object()

        def score_tokens(self, *_args, **_kwargs):
            raise AssertionError("GET workbench must not execute scoring")

    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    monkeypatch.setattr(runlog, "iter_runs", lambda limit=200: iter([run]))
    h = Handler(Sub(), path=f"/runs/{run['id']}/tokens/1/workbench")

    assert route.try_get(h, f"/runs/{run['id']}/tokens/1/workbench") is True
    assert h.status == 200
    schemas.validate(h.body, SCHEMA_VERSION)
    assert h.body["run_id"] == run["id"]
    assert h.body["index"] == 1
    assert h.body["capabilities"]["exact_fork"]["available"] is True


def test_route_never_calls_engine_or_scorer(monkeypatch):
    """The GET must not trigger expensive computation -- checking readiness is a local attribute
    lookup, never a call."""
    run = _organic_run()

    class BoomEngine:
        def __getattr__(self, name):
            raise AssertionError(f"GET workbench must never touch engine.{name}")

    class Sub:
        steer = None
        engine = BoomEngine()

        def score_tokens(self, *_args, **_kwargs):
            raise AssertionError("GET workbench must not execute scoring")

    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    monkeypatch.setattr(runlog, "iter_runs", lambda limit=200: iter([run]))
    h = Handler(Sub())

    assert route.try_get(h, f"/runs/{run['id']}/tokens/1/workbench") is True
    assert h.status == 200
    assert h.body["capabilities"]["exact_fork"]["available"] is True  # worker_ready was still true


def test_route_unknown_run_404(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: None)
    h = Handler()
    assert route.try_get(h, "/runs/run_nope/tokens/0/workbench") is True
    assert h.status == 404


def test_route_non_integer_index_400(monkeypatch):
    run = _organic_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    h = Handler()
    assert route.try_get(h, f"/runs/{run['id']}/tokens/abc/workbench") is True
    assert h.status == 400
    assert "integer" in h.body["error"]


def test_route_out_of_range_index_400(monkeypatch):
    run = _organic_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    monkeypatch.setattr(runlog, "iter_runs", lambda limit=200: iter([run]))
    h = Handler(path=f"/runs/{run['id']}/tokens/99/workbench")
    assert route.try_get(h, f"/runs/{run['id']}/tokens/99/workbench") is True
    assert h.status == 400
    assert "out of range" in h.body["error"]


def test_route_reference_run_id_query_param_is_used(monkeypatch):
    run = _organic_run()
    other = _organic_run(run_id="run_other_model", model="other-model")
    other["identity"]["model_sha256"] = "c" * 64
    runs_by_id = {run["id"]: run, other["id"]: other}
    monkeypatch.setattr(runlog, "get_run", lambda rid: runs_by_id.get(rid))
    monkeypatch.setattr(runlog, "iter_runs", lambda limit=200: iter([run, other]))
    h = Handler(path=f"/runs/{run['id']}/tokens/1/workbench?reference_run_id={other['id']}")

    assert route.try_get(h, f"/runs/{run['id']}/tokens/1/workbench") is True
    assert h.status == 200
    assert h.body["reference_run_id"] == other["id"]
    assert h.body["sections"]["comparison"]["state"] == "supported"
    assert h.body["capabilities"]["mechanistic_diff"]["available"] is True


def test_route_unresolvable_reference_run_id_degrades_not_404s(monkeypatch):
    run = _organic_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    monkeypatch.setattr(runlog, "iter_runs", lambda limit=200: iter([run]))
    h = Handler(path=f"/runs/{run['id']}/tokens/1/workbench?reference_run_id=run_missing")

    assert route.try_get(h, f"/runs/{run['id']}/tokens/1/workbench") is True
    assert h.status == 200  # the PRIMARY run is still valid -- never a 404 for the whole request
    assert h.body["sections"]["comparison"]["state"] == "unavailable"
    assert h.body["capabilities"]["mechanistic_diff"]["available"] is False


def test_route_ignores_unrelated_paths(monkeypatch):
    h = Handler()
    for path in (
        "/runs/run_x/tokens/0",           # missing /workbench
        "/runs/run_x/workbench",          # missing /tokens/<index>
        "/runs/run_x/investigation",
        "/timetravel/mode",
    ):
        assert route.try_get(h, path) is False


def test_route_investigation_contract_failure_maps_to_500_not_a_leak(monkeypatch):
    run = _organic_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    monkeypatch.setattr(runlog, "iter_runs", lambda limit=200: iter([run]))
    monkeypatch.setattr(
        "clozn.runs.investigation.build",
        lambda *_a, **_k: (_ for _ in ()).throw(ValueError("PRIVATE INTERNAL DETAIL")))
    h = Handler()

    assert route.try_get(h, f"/runs/{run['id']}/tokens/1/workbench") is True
    assert h.status == 500
    assert h.body["code"] == "investigation_contract_invalid"
    assert h.body["error"]


def test_route_workbench_contract_failure_maps_to_500(monkeypatch):
    run = _organic_run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    monkeypatch.setattr(runlog, "iter_runs", lambda limit=200: iter([run]))
    monkeypatch.setattr(
        "clozn.runs.token_workbench.build",
        lambda *_a, **_k: {"schema_version": SCHEMA_VERSION})  # missing every other required field
    h = Handler()

    assert route.try_get(h, f"/runs/{run['id']}/tokens/1/workbench") is True
    assert h.status == 500
    assert h.body["code"] == "token_workbench_contract_invalid"


def test_route_autoload_registration():
    from clozn.server import app as server

    assert route in server._GET_ROUTES
    assert server._GET_ROUTES.index(route) < server._GET_ROUTES.index(server._runs_fallback_routes)
