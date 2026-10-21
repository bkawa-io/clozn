from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from clozn import schemas
from clozn.server.routes import context_investigation_v1 as route
from clozn.recipes.context_effects import plan_context_effects
from clozn.recipes.lean_context import run_lean_context
from tests.test_context_effects_kernel import ScoreSubstrate
from tests.test_context_investigation_kernel import (
    MemoryStore,
    _single_message_run,
    _run,
    _view,
)


class Handler:
    def __init__(self, path):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status = status
        self.body = body


def test_reader_route_is_read_only_and_does_not_select_model(monkeypatch):
    run, source_ids = _run()
    plan = plan_context_effects(run)
    store = MemoryStore()
    monkeypatch.setattr(route, "ObservationStore", lambda: store)
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("reader selected a model")),
    )

    handler = Handler(f"/runs/{run['id']}/context-investigation/reader")
    assert route.try_get(handler, handler.path) is True
    assert handler.status == 200
    assert handler.body["status"] == "not_measured"
    assert handler.body["measurement"]["experiment_id"] == plan.experiment_id


def test_reader_keeps_recorded_prose_when_nothing_is_removable(monkeypatch):
    run = _single_message_run("")
    original = deepcopy(run)
    monkeypatch.setattr(route, "ObservationStore", lambda: MemoryStore())
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)

    handler = Handler(f"/runs/{run['id']}/context-investigation/reader?floor_nats=0.1")
    assert route.try_get(handler, handler.path) is True
    assert handler.status == 200
    assert handler.body["status"] == "unavailable"
    assert handler.body["reason_code"] == "context_investigation_no_exactly_removable_source"
    assert handler.body["context"]["blocks"][0]["text"] == ""
    assert handler.body["answer"]["text"] == run["response"]
    assert handler.body["capabilities"]["context_effects"] == "unavailable"
    assert run == original
    schemas.validate(handler.body, "clozn.context-investigation-reader.v1")


def test_effects_job_rejects_an_empty_default_measurement_universe(monkeypatch):
    run = _single_message_run("")
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("empty job selected a model")),
    )

    handler = Handler(f"/runs/{run['id']}/context-investigation/effects/jobs")
    assert route.try_post(handler, handler.path, {}) is True
    assert handler.status == 409
    assert handler.body["code"] == "context_investigation_plan_unavailable"


def test_query_route_is_read_only_and_projects_persisted_vectors(monkeypatch):
    run, source_ids = _run()
    store = MemoryStore(_view(run, source_ids))
    monkeypatch.setattr(route, "ObservationStore", lambda: store)
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("query selected a model")),
    )

    handler = Handler(f"/runs/{run['id']}/context-investigation/query")
    assert route.try_post(
        handler, handler.path, {"answer_start": 5, "answer_end": 11},
    ) is True
    assert handler.status == 200
    assert handler.body["status"] == "completed"
    assert handler.body["selection"]["text"] == "answer"


def test_effects_job_request_accepts_only_canonical_source_ids(monkeypatch):
    run, _source_ids = _run()
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)

    handler = Handler(f"/runs/{run['id']}/context-investigation/effects/jobs")
    assert route.try_post(handler, handler.path, {"execution_policy": "fast"}) is True
    assert handler.status == 400
    assert handler.body["code"] == "unsupported_effect_request"

    assert route.try_post(handler, handler.path, {"source_ids": ["", "src"]}) is True
    assert handler.status == 400
    assert handler.body["code"] == "invalid_source_ids"


def test_effects_job_start_uses_score_capability_and_shared_registry(monkeypatch):
    run, source_ids = _run()
    calls = []

    class FakeJobs:
        def start(self, run_id, worker, *, kind):
            calls.append((run_id, worker, kind))
            return {"job_id": "infjob_context", "state": "queued", "kind": kind}

    class ScoreSubstrate:
        def score_tokens(self, *_args, **_kwargs):
            raise AssertionError("route start must not execute scoring synchronously")

    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr("clozn.server.influence_jobs.JOBS", FakeJobs())
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: SimpleNamespace(sub=ScoreSubstrate()),
    )

    handler = Handler(f"/runs/{run['id']}/context-investigation/effects/jobs")
    assert route.try_post(handler, handler.path, {"source_ids": source_ids[:1]}) is True
    assert handler.status == 202
    assert handler.body["kind"] == "context_investigation_effects"
    assert calls[0][0] == run["id"]
    assert calls[0][2] == "context_investigation_effects"


def test_effects_worker_attaches_the_compact_direct_measurement_result(monkeypatch):
    run, source_ids = _run()
    plan = plan_context_effects(run, source_ids=source_ids[:2])
    attached = []

    class Control:
        def checkpoint(self, **_event):
            return None

        def cancel_requested(self):
            return False

        def attach_result(self, payload):
            attached.append(payload)

    rows = [
        SimpleNamespace(
            observation=SimpleNamespace(completed=True), diagnostics={}, state="completed",
        ),
        SimpleNamespace(
            observation=None, diagnostics={"observation_status": "unavailable"}, state="completed",
        ),
    ]
    result = SimpleNamespace(experiment_id=plan.experiment_id, arms=rows, state="completed")
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr(route, "measure_context_effects", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(route, "ObservationStore", lambda: object())

    outcome = route._job_worker(run, object(), source_ids[:2], plan.experiment_id)(Control())

    assert outcome == {"state": "completed"}
    assert attached == [{
        "schema_version": "clozn.context-investigation-effects-job-result.v1",
        "experiment_id": plan.experiment_id,
        "source_count": 2,
        "completed_source_count": 1,
        "unavailable_source_count": 1,
        "failed_source_count": 0,
        "not_measured_source_count": 0,
        "state": "completed",
    }]


def test_effects_job_paths_reject_a_job_of_the_wrong_kind(monkeypatch):
    run, _source_ids = _run()

    class FakeJobs:
        def get(self, _run_id, _job_id):
            return {"kind": "minimal_context", "state": "running"}

        def cancel(self, *_args):
            raise AssertionError("wrong-kind job must not be cancelled")

    monkeypatch.setattr("clozn.server.influence_jobs.JOBS", FakeJobs())
    path = f"/runs/{run['id']}/context-investigation/effects/jobs/job_other"
    handler = Handler(path)
    assert route.try_get(handler, path) is True
    assert handler.status == 404

    cancel_path = f"{path}/cancel"
    handler = Handler(cancel_path)
    assert route.try_post(handler, cancel_path, {}) is True
    assert handler.status == 404


def test_removed_context_effects_actions_are_not_claimed():
    for tail in (
        "/effects-search/jobs",
        "/remove-test/jobs",
        "/counterfactuals/jobs",
        "/counterfactuals/materialize",
    ):
        handler = Handler(f"/runs/run_a/context-investigation{tail}")
        assert route.try_post(handler, handler.path, {}) is False


def test_lean_context_job_uses_a_fixed_profile_and_shared_job_registry(monkeypatch):
    run, _source_ids = _run()
    calls = []

    class FakeJobs:
        def start(self, run_id, worker, *, kind):
            calls.append((run_id, worker, kind))
            return {"job_id": "infjob_lean", "state": "queued", "kind": kind}

    class ScoreOnly:
        def score_tokens(self, *_args, **_kwargs):
            raise AssertionError("route start must not score synchronously")

    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr("clozn.server.influence_jobs.JOBS", FakeJobs())
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: SimpleNamespace(sub=ScoreOnly(), engine=object()),
    )
    path = f"/runs/{run['id']}/context-investigation/lean-context/jobs"

    rejected = Handler(path)
    assert route.try_post(rejected, path, {"mean_loss_tolerance_nats": 1.0}) is True
    assert rejected.status == 400
    assert rejected.body["code"] == "unsupported_lean_context_request"

    handler = Handler(path)
    assert route.try_post(handler, path, {}) is True
    assert handler.status == 202
    assert handler.body["kind"] == "context_investigation_lean_context"
    assert calls[0][0] == run["id"]
    assert calls[0][2] == "context_investigation_lean_context"


def test_lean_context_worker_attaches_the_validated_full_result(monkeypatch):
    run, _source_ids = _run()

    class Engine:
        def apply_template_info(self, messages):
            return {"prompt_tokens": sum(len(str(item.get("content", ""))) for item in messages)}

    expected = run_lean_context(run, substrate=ScoreSubstrate(), engine=Engine())
    attached = []

    class Control:
        def checkpoint(self, **_event):
            return None

        def cancel_requested(self):
            return False

        def attach_result(self, payload):
            attached.append(payload)

    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr(route, "run_lean_context", lambda *_args, **_kwargs: expected)
    monkeypatch.setattr(route, "ObservationStore", lambda: object())
    universe_id = run["context_units"]
    from clozn.runs.context_search_universe import plan_context_search_universe
    planned = plan_context_search_universe(run, universe_id, max_units=16)

    outcome = route._lean_job_worker(
        run, object(), object(), planned["universe_id"],
    )(Control())

    assert outcome == {"state": "completed"}
    assert attached == [expected]
    schemas.validate(attached[0], "clozn.lean-context-result.v1")


def test_lean_context_job_paths_reject_a_job_of_the_wrong_kind(monkeypatch):
    run, _source_ids = _run()

    class FakeJobs:
        def get(self, _run_id, _job_id):
            return {"kind": "context_investigation_effects", "state": "running"}

        def cancel(self, *_args):
            raise AssertionError("wrong-kind job must not be cancelled")

    monkeypatch.setattr("clozn.server.influence_jobs.JOBS", FakeJobs())
    path = f"/runs/{run['id']}/context-investigation/lean-context/jobs/job_other"
    handler = Handler(path)
    assert route.try_get(handler, path) is True
    assert handler.status == 404

    cancel_path = f"{path}/cancel"
    handler = Handler(cancel_path)
    assert route.try_post(handler, cancel_path, {}) is True
    assert handler.status == 404
