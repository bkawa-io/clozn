from __future__ import annotations

from types import SimpleNamespace

from clozn.experiments.persistence import ObservationStore
from clozn.server.routes import context_investigation_v1 as route

from tests.test_context_investigation_kernel import MemoryStore, _run, _view
from clozn.recipes.context_effects import plan_context_effects


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
    plan = plan_context_effects(run, source_ids=source_ids[:2])
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


def test_query_route_is_read_only_and_projects_persisted_vectors(monkeypatch):
    run, source_ids = _run()
    plan = plan_context_effects(run, source_ids=source_ids[:2])
    view = _view(run, source_ids[:2])
    store = MemoryStore(view)
    monkeypatch.setattr(route, "ObservationStore", lambda: store)
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("query selected a model")),
    )

    handler = Handler(f"/runs/{run['id']}/context-investigation/query")
    assert route.try_post(
        handler,
        handler.path,
        {"answer_start": 5, "answer_end": 11},
    ) is True
    assert handler.status == 200
    assert handler.body["status"] == "completed"
    assert handler.body["selection"]["text"] == "answer"


def test_effect_job_request_accepts_only_source_ids(monkeypatch):
    run, source_ids = _run()
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    handler = Handler(f"/runs/{run['id']}/context-investigation/effects/jobs")
    assert route.try_post(
        handler,
        handler.path,
        {"answer_start": 0, "answer_end": 1},
    ) is True
    assert handler.status == 400
    assert handler.body["code"] == "unsupported_effect_request"


def test_effect_job_start_uses_score_capability_and_generic_job_registry(monkeypatch):
    run, source_ids = _run()
    calls = []

    class FakeJobs:
        def start(self, run_id, worker, *, kind):
            calls.append((run_id, worker, kind))
            return {"job_id": "infjob_context", "state": "queued", "kind": kind}

    class ScoreSubstrate:
        def score_tokens(self, *args, **kwargs):
            raise AssertionError("job worker must not run synchronously in route test")

    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr("clozn.server.influence_jobs.JOBS", FakeJobs())
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: SimpleNamespace(sub=ScoreSubstrate(), engine=None),
    )
    handler = Handler(f"/runs/{run['id']}/context-investigation/effects/jobs")
    assert route.try_post(handler, handler.path, {"source_ids": source_ids[:1]}) is True
    assert handler.status == 202
    assert handler.body["kind"] == "context_investigation_effects"
    assert calls and calls[0][0] == run["id"]


def test_generic_experiment_and_observation_routes_return_canonical_reads(monkeypatch):
    from clozn.server.routes import experiments_read

    run, source_ids = _run()
    view = _view(run, source_ids[:1])

    class Store(ObservationStore):
        def get_experiment(self, _):
            return view

        def get_observation(self, observation_id):
            if observation_id == view.control.observation_id:
                return view.control
            raise KeyError(observation_id)

    monkeypatch.setattr(experiments_read, "ObservationStore", lambda: Store())
    experiment_handler = Handler(f"/experiments/{view.experiment_id}")
    assert experiments_read.try_get(experiment_handler, experiment_handler.path) is True
    assert experiment_handler.status == 200
    assert experiment_handler.body["experiment_id"] == view.experiment_id

    observation_handler = Handler(f"/observations/{view.control.observation_id}")
    assert experiments_read.try_get(observation_handler, observation_handler.path) is True
    assert observation_handler.status == 200
    assert observation_handler.body["observation_id"] == view.control.observation_id
