"""Focused HTTP/job tests for the persisted Context Dependence capability."""
from __future__ import annotations

from types import SimpleNamespace
import threading
import time

import pytest

import clozn.runs.store as runlog
from clozn.server import app as server
from clozn.server.influence_jobs import JOBS
from clozn.server.routes import context_dependence as route


class Handler:
    def __init__(self, sub=None):
        self._inj_sub = sub
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


class ScoreSub:
    def __init__(self, *, entered=None, release=None):
        self.calls = []
        self.entered = entered
        self.release = release

    def score_tokens(self, messages, ids, **_kwargs):
        self.calls.append([message.get("content") for message in messages])
        if self.entered is not None and not self.entered.is_set():
            self.entered.set()
            assert self.release is not None
            self.release.wait(timeout=2)
        removed = {content for content in ("A", "B") if content not in self.calls[-1]}
        penalty = 0.4 * len(removed)
        return [
            {"id": 7, "piece": "O", "logprob": -1.0 - penalty / 2},
            {"id": 8, "piece": "K", "logprob": -1.0 - penalty / 2},
        ]


class RegenerationSub:
    def __init__(self):
        self.chat_calls = 0
        self.seen_messages = None

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self.chat_calls += 1
        self.seen_messages = [message.get("content") for message in messages]
        assert sample is False
        if trace_out is not None:
            trace_out.extend([
                {"id": 21, "piece": "N", "conf": 0.9, "alts": []},
                {"id": 22, "piece": "O", "conf": 0.8, "alts": []},
            ])
        if mem_out is not None:
            mem_out.update(assembled_messages=list(messages), final_prompt="regenerated")
        return "NO"


def _wait(run_id, job_id, states={"completed", "failed", "cancelled"}):
    deadline = time.monotonic() + 3
    last = None
    while time.monotonic() < deadline:
        last = JOBS.get(run_id, job_id)
        assert last is not None
        if last["state"] in states:
            return last
        time.sleep(0.01)
    pytest.fail(f"job did not finish: {last}")


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    old_router, old_sub, old_engine = server.MODEL_ROUTER, server.SUB, server.ENGINE
    JOBS.clear_for_tests()
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    server.MODEL_ROUTER = None
    server.SUB = None
    server.ENGINE = None
    yield
    JOBS.clear_for_tests()
    server.MODEL_ROUTER, server.SUB, server.ENGINE = old_router, old_sub, old_engine


def _run():
    run_id = runlog.record(
        source="test", client="test", model="run-model", substrate="fake",
        messages=[
            {"role": "user", "content": "A", "source_id": "a"},
            {"role": "user", "content": "B", "source_id": "b"},
        ],
        response="OK", trace={"token_ids": [7, 8]},
        identity={"model_sha256": "a" * 64, "template_fingerprint": "0123456789abcdef"},
    )
    assert run_id
    return runlog.get_run(run_id)


def _post(sub, path, body=None):
    h = Handler(sub)
    assert route.try_post(h, path, body or {}) is True
    return h


def _get(path):
    h = Handler()
    assert route.try_get(h, path) is True
    return h


def test_autoload_and_absent_get_are_read_only(isolated, monkeypatch):
    run = _run()
    assert route.CLOZN_ROUTE_AUTOLOAD is True
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: pytest.fail("GET must never select or touch an engine"),
    )
    h = _get(f"/runs/{run['id']}/context-dependence")
    assert h.status == 404
    assert h.body["available"] is False
    assert "context_dependence_study" not in runlog.get_run(run["id"])


def test_start_poll_persist_and_get_preserve_legacy_influence_map(isolated):
    run = _run()
    original = {"schema": "clozn.context_answer_influence.v1", "available": True}
    run["influence_map"] = original
    assert runlog.replace_run(run)

    sub = ScoreSub()
    h = _post(sub, f"/runs/{run['id']}/context-dependence/jobs", {"compute_level": "Quick"})
    assert h.status == 202 and h.body["kind"] == "context_dependence"
    final = _wait(run["id"], h.body["job_id"])
    assert final["state"] == "completed"
    assert final["result"]["schema_version"] == "clozn.context-dependence-study.v2"
    persisted = runlog.get_run(run["id"])
    assert persisted["influence_map"] == original
    assert persisted["context_dependence_study"] == final["result"]
    got = _get(f"/runs/{run['id']}/context-dependence")
    assert got.status == 200 and got.body == final["result"]


def test_cancel_prevents_attachment_between_experiments(isolated):
    run = _run()
    entered, release = threading.Event(), threading.Event()
    h = _post(
        ScoreSub(entered=entered, release=release),
        f"/runs/{run['id']}/context-dependence/jobs", {"compute_level": "Quick"},
    )
    assert h.status == 202 and entered.wait(timeout=1)
    cancelled = _post(None, f"/runs/{run['id']}/context-dependence/jobs/{h.body['job_id']}/cancel")
    assert cancelled.status == 200 and cancelled.body["cancel_accepted"] is True
    release.set()
    final = _wait(run["id"], h.body["job_id"])
    assert final["state"] == "cancelled"
    assert "context_dependence_study" not in runlog.get_run(run["id"])


def test_cache_hit_and_identity_misses_for_source_set_and_method_but_rejects_target(isolated):
    run = _run()
    sub = ScoreSub()
    first = _post(sub, f"/runs/{run['id']}/context-dependence/jobs", {"compute_level": "Quick"})
    assert _wait(run["id"], first.body["job_id"])["state"] == "completed"
    hits = _post(sub, f"/runs/{run['id']}/context-dependence/jobs", {"compute_level": "Quick"})
    assert hits.status == 202 and hits.body["cached"] is True
    source_ids = [item["segment_id"] for item in runlog.get_run(run["id"])["context_receipt"]["delivered"]]
    target = _post(sub, f"/runs/{run['id']}/context-dependence/jobs", {
        "compute_level": "Quick", "target": {"recorded_token_range": [0, 1]},
    })
    assert target.status == 400
    assert "target is not accepted" in target.body["error"]
    requests = [
        {"compute_level": "Quick", "root_source_ids": [source_ids[0]]},
        {"compute_level": "Quick", "method": "different-direct-method"},
    ]
    for request in requests:
        miss = _post(sub, f"/runs/{run['id']}/context-dependence/jobs", request)
        assert miss.status == 202 and miss.body["cached"] is False
        assert _wait(run["id"], miss.body["job_id"])["state"] == "completed"


def test_v2_cache_binding_has_no_target_coordinate_component(isolated):
    from clozn.runs.context_dependence_execution import cache_binding, normalize_request

    run = _run()
    binding = cache_binding(run, normalize_request({"compute_level": "Quick"}))

    assert binding["schema_version"] == "clozn.context-dependence-study.v2"
    assert "target" not in binding


def test_explicit_neutralization_controls_are_receipt_ordered_deduped_and_cache_bound(isolated):
    from clozn.runs.context_dependence_execution import (
        cache_identity,
        normalize_request,
        run_context_dependence_execution,
    )
    from tests.test_context_dependence_measurement import FakeScoreSub, _run

    run = _run()
    source_a, _source_b, source_c = [item["segment_id"] for item in run["context_receipt"]["assembled"]]
    request = normalize_request({
        "compute_level": "Quick",
        # Same source set twice in different caller orders: one separately
        # named control must remain after canonical-set normalization.
        "neutralization_source_sets": [[source_c, source_a], [source_a, source_c]],
    })
    without_controls = normalize_request({"compute_level": "Quick"})

    assert cache_identity(run, request) != cache_identity(run, without_controls)
    result = run_context_dependence_execution(run, FakeScoreSub(), request)

    assert len(result["experiments"]) == 1  # canonical delete root only in Quick
    assert result["experiments"][0]["intervention_operator"] == "delete_source"
    assert len(result["robustness_controls"]) == 1
    control = result["robustness_controls"][0]
    assert control["neutralized_source_ids"] == sorted([source_a, source_c])
    assert control["intervention_operator"] == "neutralize_source"
    assert result["execution"]["requested_neutralization_source_sets"] == [sorted([source_a, source_c])]
    assert result["execution"]["cache_binding"]["neutralization_source_sets"] == [sorted([source_a, source_c])]
    assert result["budget"]["passes_consumed"] == 3  # baseline + delete root + separate control


def test_requested_neutralization_controls_fail_before_scoring_if_the_policy_cannot_fit_all():
    from clozn.runs.context_dependence_execution import ContextDependenceExecutionError, run_context_dependence_execution
    from tests.test_context_dependence_measurement import FakeScoreSub, _run

    run = _run()
    source_a, source_b, source_c = [item["segment_id"] for item in run["context_receipt"]["assembled"]]
    all_sets = [
        [source_a], [source_b], [source_c], [source_a, source_b], [source_a, source_c],
        [source_b, source_c], [source_a, source_b, source_c],
    ]
    sub = FakeScoreSub()

    with pytest.raises(ContextDependenceExecutionError, match="every requested neutralization control"):
        run_context_dependence_execution(
            run, sub, {"compute_level": "Quick", "neutralization_source_sets": all_sets},
        )
    assert sub.calls == []


def test_invalid_producer_result_is_not_attached(isolated, monkeypatch):
    run = _run()
    monkeypatch.setattr(
        "clozn.runs.context_dependence_execution.run_context_dependence_execution",
        lambda *_args, **_kwargs: {"schema_version": "clozn.context-dependence-study.v2"},
    )
    h = _post(ScoreSub(), f"/runs/{run['id']}/context-dependence/jobs", {})
    final = _wait(run["id"], h.body["job_id"])
    assert final["state"] == "failed"
    assert final["error"]["code"] == "context_dependence_schema_invalid"
    assert "context_dependence_study" not in runlog.get_run(run["id"])


def test_managed_selection_uses_the_recorded_model_and_worker_refusal_is_typed(isolated, monkeypatch):
    run = _run()
    selected = []
    chosen = ScoreSub()

    def select(_handler, model, *, route):
        selected.append((model, route))
        return SimpleNamespace(sub=chosen)

    monkeypatch.setattr("clozn.server.model_routing.select_control_model_for_run", select)
    h = _post(None, f"/runs/{run['id']}/context-dependence/jobs", {})
    assert h.status == 202 and selected == [("run-model", "/runs/<id>/context-dependence/jobs")]
    assert _wait(run["id"], h.body["job_id"])["state"] == "completed"

    second = _run()
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: SimpleNamespace(sub=object()),
    )
    unavailable = _post(None, f"/runs/{second['id']}/context-dependence/jobs", {})
    assert unavailable.status == 503
    assert unavailable.body["code"] == "context_dependence_worker_unavailable"


def test_standard_execution_caps_masks_to_small_receipts_finite_subset_space():
    from tests.test_context_dependence_measurement import FakeScoreSub, _run
    from clozn.runs.context_dependence_execution import run_context_dependence_execution

    result = run_context_dependence_execution(
        _run(), FakeScoreSub(), {"compute_level": "Standard"},
    )

    assert len(result["screen"]["masks"]) <= 7  # 2**3 - 1 non-empty source subsets
    assert result["budget"]["passes_consumed"] <= result["budget"]["passes_requested"]
    assert result["screen"]["budget"]["passes_consumed"] == result["budget"]["passes_consumed"]
    assert result["preserving_subsets"]["budget"]["passes_consumed"] == result["budget"]["passes_consumed"]


def _measured_run_with_experiment():
    from clozn.runs.context_dependence_execution import run_context_dependence_execution

    run = _run()
    artifact = run_context_dependence_execution(run, ScoreSub(), {"compute_level": "Quick"})
    run["context_dependence_study"] = artifact
    assert runlog.replace_run(run)
    return runlog.get_run(run["id"]), artifact["experiments"][0]["experiment_id"]


def test_regenerate_experiment_executes_one_child_and_returns_canonical_diff(isolated):
    run, experiment_id = _measured_run_with_experiment()
    sub = RegenerationSub()

    result = _post(
        sub,
        f"/runs/{run['id']}/context-dependence/experiments/{experiment_id}/regenerate",
    )

    assert result.status == 200
    assert sub.chat_calls == 1
    assert sub.seen_messages == []
    regeneration = result.body["regeneration"]
    assert regeneration["state"] == "completed"
    assert regeneration["generation_calls"] == 1
    # Canonical model_diff is always returned; this light synthetic fixture
    # does not promise persisted trace detail for a token-level view.
    assert "first_divergence_view" in regeneration["comparison"]
    child = runlog.get_run(regeneration["child_run_id"])
    assert child["parent_run_id"] == run["id"]
    assert child["changes_applied"]["context_dependence_regeneration"]["experiment_id"] == experiment_id


def test_regenerate_experiment_typed_preflight_failures_never_select_a_worker(isolated, monkeypatch):
    selected = []
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: selected.append(True) or pytest.fail("invalid input must not select a worker"),
    )
    absent = _post(None, "/runs/run_missing/context-dependence/experiments/cdx_missing/regenerate")
    assert absent.status == 404
    assert absent.body["code"] == "context_dependence_regeneration_run_not_found"

    run = _run()
    no_study = _post(None, f"/runs/{run['id']}/context-dependence/experiments/cdx_missing/regenerate")
    assert no_study.status == 404
    assert no_study.body["code"] == "context_dependence_regeneration_study_unavailable"

    run, _experiment_id = _measured_run_with_experiment()
    missing = _post(None, f"/runs/{run['id']}/context-dependence/experiments/cdx_missing/regenerate")
    assert missing.status == 404
    assert missing.body["code"] == "context_dependence_regeneration_experiment_not_found"
    assert selected == []


def test_regenerate_experiment_stale_binding_never_selects_or_generates(isolated, monkeypatch):
    run, experiment_id = _measured_run_with_experiment()
    stale = runlog.get_run(run["id"])
    stale["context_dependence_study"]["experiments"][0]["exact_removed_ranges"][0]["message_index"] = 99
    assert runlog.replace_run(stale)
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: pytest.fail("stale binding must not select a worker"),
    )

    result = _post(None, f"/runs/{run['id']}/context-dependence/experiments/{experiment_id}/regenerate")

    assert result.status == 409
    assert result.body["code"] == "context_dependence_regeneration_stale"


def test_regenerate_experiment_selects_immutable_run_model_and_rejects_no_chat_worker(isolated, monkeypatch):
    run, experiment_id = _measured_run_with_experiment()
    selected = []

    def select(_handler, model, *, route):
        selected.append((model, route))
        return SimpleNamespace(sub=object())

    monkeypatch.setattr("clozn.server.model_routing.select_control_model_for_run", select)
    result = _post(None, f"/runs/{run['id']}/context-dependence/experiments/{experiment_id}/regenerate")

    assert selected == [("run-model", "/runs/<id>/context-dependence/experiments/<experiment_id>/regenerate")]
    assert result.status == 503
    assert result.body["code"] == "context_dependence_regeneration_worker_unavailable"
