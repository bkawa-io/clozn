"""Backend hardening contracts for Minimal Context persistence and routing."""
from __future__ import annotations

from contextlib import closing
import json
import threading
import time

import pytest

import clozn.runs.store as runlog
from clozn.server import app as server
from clozn.server.influence_jobs import JOBS
from clozn.server.routes import minimal_context as route
from clozn.runs.minimal_context_execution import cache_binding, normalize_request


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
        self.calls.append([item.get("content") for item in messages])
        if self.entered is not None and len(self.calls) == 2:
            self.entered.set()
            self.release.wait(timeout=2)
        removed = {item for item in ("A", "B") if item not in self.calls[-1]}
        penalty = 0.2 * len(removed)
        return [
            {"id": 7, "piece": "O", "logprob": -1.0 - penalty},
            {"id": 8, "piece": "K", "logprob": -1.0 - penalty},
        ]


class IdentitySub:
    def __init__(self):
        self.elapsed = 1
        self.model_sha256 = "a" * 64
        self.template = "0123456789abcdef"

    def identity_meta(self):
        return {"model_sha256": self.model_sha256, "template_fingerprint": self.template}

    def run_meta(self):
        return {"duration_ms": self.elapsed, "finish_reason": "stop", "prompt_tokens": self.elapsed}


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
        source="test",
        client="test",
        model="run-model",
        substrate="fake",
        messages=[
            {"role": "user", "content": "A", "source_id": "a"},
            {"role": "user", "content": "B", "source_id": "b"},
        ],
        response="OK",
        trace={"token_ids": [7, 8]},
        identity={"model_sha256": "a" * 64, "template_fingerprint": "0123456789abcdef"},
    )
    return runlog.get_run(run_id)


def _wait(run_id, job_id):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        snapshot = JOBS.get(run_id, job_id)
        if snapshot and snapshot["state"] in {"completed", "failed", "cancelled"}:
            return snapshot
        time.sleep(0.01)
    pytest.fail("Minimal Context job did not finish")


def test_likelihood_cache_identity_ignores_last_request_metadata_but_binds_model_and_continuation():
    run = {
        "id": "run-cache",
        "model": "fixture",
        "substrate": "fake",
        "identity": {},
        "trace": {"token_ids": [1, 2]},
        "messages": [],
        "response": "ab",
    }
    request = normalize_request({
        "preservation": {"kind": "teacher_forced_likelihood", "tolerance_nats": 0.3},
        "search_probe_budget": 4,
        "certification_probe_budget": 8,
    })
    universe = {"basis_context_units_digest": "digest", "universe_id": "mcu_fixture"}
    sub = IdentitySub()
    first = cache_binding(run, request, universe, sub)
    sub.elapsed = 900
    assert cache_binding(run, request, universe, sub) == first
    sub.model_sha256 = "b" * 64
    assert cache_binding(run, request, universe, sub) != first


def test_minimal_context_support_is_blob_packed_and_round_trips(isolated):
    rid = runlog.record(source="test", messages=[{"role": "user", "content": "q"}], response="a")
    run = runlog.get_run(rid)
    support = {"cds_fixture": {"schema_version": "clozn.context-dependence-study.v2", "experiments": [{"n": 1}]}}
    run["minimal_context_support"] = support
    assert runlog.replace_run(run)

    with closing(runlog._connect()) as db:
        payload = json.loads(db.execute("SELECT payload_json FROM runs WHERE id = ?", (rid,)).fetchone()["payload_json"])
    assert "minimal_context_support" not in payload
    assert payload["minimal_context_support_ref"]["sha256"]
    assert runlog.get_run(rid)["minimal_context_support"] == support


def test_minimal_context_support_blob_write_failure_is_explicit(isolated, monkeypatch):
    rid = runlog.record(source="test", messages=[{"role": "user", "content": "q"}], response="a")
    run = runlog.get_run(rid)

    def broken(*_args, **_kwargs):
        raise OSError("disk full")

    run["minimal_context_support"] = {"cds_fixture": {"large": True}}
    monkeypatch.setattr(runlog, "atomic_write_json", broken)
    assert runlog.replace_run(run)
    loaded = runlog.get_run(rid)
    assert "minimal-context-support-missing" in loaded["flags"]
    assert "minimal_context_support_write_failed" in loaded["meta"]
    assert loaded["minimal_context_support"]["unavailable"].startswith("minimal context support evidence write failed")


def test_minimal_context_route_persists_result_and_support(isolated):
    run = _run()
    sub = ScoreSub()
    handler = Handler(sub)
    assert route.try_post(handler, f"/runs/{run['id']}/minimal-context/jobs", {
        "search_probe_budget": 4,
        "certification_probe_budget": 4,
        "universe": {"max_units": 2},
        "preservation": {"kind": "teacher_forced_likelihood", "tolerance_nats": 0.3},
    })
    assert handler.status == 202
    final = _wait(run["id"], handler.body["job_id"])
    assert final["state"] == "completed", final
    persisted = runlog.get_run(run["id"])
    assert final["result"]["result_id"] in persisted["minimal_context_results"]
    assert persisted["minimal_context_support"]


def test_larger_budget_reuses_compatible_support_without_rescoring_baseline(isolated):
    run = _run()
    sub = ScoreSub()
    first = Handler(sub)
    request = {
        "search_probe_budget": 4,
        "certification_probe_budget": 4,
        "universe": {"max_units": 2},
    }
    route.try_post(first, f"/runs/{run['id']}/minimal-context/jobs", request)
    assert _wait(run["id"], first.body["job_id"])["state"] == "completed"
    baseline_calls = sum(1 for call in sub.calls if call == ["A", "B"])

    second = Handler(sub)
    route.try_post(second, f"/runs/{run['id']}/minimal-context/jobs", {
        **request,
        "certification_probe_budget": 8,
    })
    assert _wait(run["id"], second.body["job_id"])["state"] == "completed"
    assert sum(1 for call in sub.calls if call == ["A", "B"]) == baseline_calls


def test_minimal_context_route_cancellation_does_not_persist_result(isolated):
    run = _run()
    entered, release = threading.Event(), threading.Event()
    sub = ScoreSub(entered=entered, release=release)
    handler = Handler(sub)
    route.try_post(handler, f"/runs/{run['id']}/minimal-context/jobs", {
        "search_probe_budget": 4,
        "certification_probe_budget": 4,
        "universe": {"max_units": 2},
    })
    assert handler.status == 202
    assert entered.wait(timeout=1)
    cancel = Handler()
    route.try_post(cancel, f"/runs/{run['id']}/minimal-context/jobs/{handler.body['job_id']}/cancel", {})
    assert cancel.status == 200
    release.set()
    final = _wait(run["id"], handler.body["job_id"])
    assert final["state"] == "cancelled"
    saved = runlog.get_run(run["id"])
    assert not saved.get("minimal_context_results")
