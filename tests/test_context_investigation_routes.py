from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from clozn import schemas
from clozn.experiments.persistence import ObservationStore
from clozn.server.routes import context_investigation_v1 as route

from tests.test_context_investigation_kernel import (
    MemoryStore,
    _protected_only_run,
    _run,
    _view,
)
from clozn.recipes.context_effects import plan_context_effects
from clozn.recipes.context_effects_search import (
    ContextEffectsSearchBudget,
    measure_context_effects_coarse_to_fine,
)
from tests.test_context_effects_search import (
    _EffectAdapter as SearchEffectAdapter,
    _key as search_key,
    _run as search_run,
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


def test_reader_route_keeps_canonical_prose_when_default_q2_has_no_legal_arm(monkeypatch):
    run = _protected_only_run()
    original = deepcopy(run)
    monkeypatch.setattr(route, "ObservationStore", lambda: MemoryStore())
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("reader selected a model")),
    )

    handler = Handler(f"/runs/{run['id']}/context-investigation/reader?floor_nats=0.1")
    assert route.try_get(handler, handler.path) is True
    assert handler.status == 200
    assert handler.body["status"] == "unavailable"
    assert handler.body["reason"] == "no canonical Context Receipt source is exactly removable"
    assert handler.body["reason_code"] == "context_investigation_no_exactly_removable_source"
    assert handler.body["measurement"]["experiment_id"] is None
    assert handler.body["measurement"]["source_universe"] == []
    assert handler.body["context"]["blocks"][0]["text"] == "only current request"
    assert handler.body["answer"]["text"] == run["response"]
    assert handler.body["loci"] == []
    assert handler.body["capabilities"]["context_effects"] == "unavailable"
    assert all(
        source["measurement"]["status"] == "unavailable"
        for source in handler.body["sources"]
    )
    assert run == original
    schemas.validate(handler.body, "clozn.context-investigation-reader.v1")


def test_effects_job_does_not_accept_an_empty_default_measurement_universe(monkeypatch):
    run = _protected_only_run()
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("empty job selected a model")),
    )

    handler = Handler(f"/runs/{run['id']}/context-investigation/effects/jobs")
    assert route.try_post(handler, handler.path, {}) is True
    assert handler.status == 409
    assert handler.body == {
        "error": "no automatically discovered sources are exactly removable",
        "code": "context_investigation_plan_unavailable",
    }


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


def _effects_search_body(**overrides):
    body = {
        "budget": {
            "max_coarse_arms": 2,
            "max_fine_arms": 2,
            "max_joint_arms": 0,
        },
        "execution_policy": "trusted_scalar_only",
    }
    body.update(overrides)
    return body


def test_effects_search_is_additive_and_rejects_unknown_request_fields(monkeypatch):
    run, _source_ids = _run()
    monkeypatch.setattr(
        "clozn.runs.store.get_run",
        lambda run_id: run if run_id == run["id"] else None,
    )
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid search selected a model")
        ),
    )

    handler = Handler(f"/runs/{run['id']}/context-investigation/effects-search/jobs")
    assert route.try_post(
        handler, handler.path, _effects_search_body(surprise=True),
    ) is True
    assert handler.status == 400
    assert handler.body["code"] == "invalid_effects_search_request"

    old_handler = Handler(f"/runs/{run['id']}/context-investigation/effects/jobs")
    assert route.try_post(
        old_handler, old_handler.path, {"budget": {}},
    ) is True
    assert old_handler.status == 400
    assert old_handler.body["code"] == "unsupported_effect_request"


def test_effects_search_rejects_sources_outside_canonical_universe_before_model_selection(monkeypatch):
    run, _source_ids = _run()
    monkeypatch.setattr(
        "clozn.runs.store.get_run",
        lambda run_id: run if run_id == run["id"] else None,
    )
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid search selected a model")
        ),
    )
    handler = Handler(f"/runs/{run['id']}/context-investigation/effects-search/jobs")
    assert route.try_post(
        handler,
        handler.path,
        _effects_search_body(source_ids=["src_not_in_this_receipt"]),
    ) is True
    assert handler.status == 400
    assert handler.body["code"] == "invalid_effects_search_request"


def test_effects_search_native_policy_fails_closed_when_worker_has_no_native_strategy(monkeypatch):
    run, _source_ids = _run()

    class ScalarOnlySubstrate:
        score_tokens_many_execution_strategy = "scalar_serial"

        def score_tokens(self, *_args, **_kwargs):
            raise AssertionError("route capability check must not execute scoring")

    monkeypatch.setattr(
        "clozn.runs.store.get_run",
        lambda run_id: run if run_id == run["id"] else None,
    )
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            sub=ScalarOnlySubstrate(), engine=None,
        ),
    )
    handler = Handler(f"/runs/{run['id']}/context-investigation/effects-search/jobs")
    assert route.try_post(
        handler,
        handler.path,
        _effects_search_body(
            execution_policy="native_screen_then_scalar_refine",
        ),
    ) is True
    assert handler.status == 503
    assert handler.body["code"] == "context_investigation_native_screening_unavailable"


def test_effects_search_starts_distinct_job_kind_and_normalizes_spec(monkeypatch):
    run, source_ids = _run()
    calls = []

    class FakeJobs:
        def start(self, run_id, worker, *, kind):
            calls.append((run_id, worker, kind))
            return {"job_id": "infjob_search", "state": "queued", "kind": kind}

    class ScalarSubstrate:
        score_tokens_many_execution_strategy = "scalar_serial"

        def score_tokens(self, *_args, **_kwargs):
            raise AssertionError("job worker must not run synchronously")

    monkeypatch.setattr(
        "clozn.runs.store.get_run",
        lambda run_id: run if run_id == run["id"] else None,
    )
    monkeypatch.setattr("clozn.server.influence_jobs.JOBS", FakeJobs())
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: SimpleNamespace(
            sub=ScalarSubstrate(), engine=None,
        ),
    )
    handler = Handler(f"/runs/{run['id']}/context-investigation/effects-search/jobs")
    assert route.try_post(
        handler,
        handler.path,
        _effects_search_body(
            source_ids=source_ids[:2], coarse_group_size=2,
            drilldown_source_ids=[source_ids[0]],
        ),
    ) is True
    assert handler.status == 202
    assert handler.body["kind"] == "context_investigation_effects_search"
    assert calls[0][0] == run["id"]
    assert calls[0][2] == "context_investigation_effects_search"


def test_reader_projection_associates_only_trusted_singletons(tmp_path, monkeypatch):
    from clozn.runs import store as run_store

    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, source_ids = search_run(3)
    store = ObservationStore()
    result = measure_context_effects_coarse_to_fine(
        run,
        source_ids=source_ids,
        coarse_groups=[source_ids],
        budget=ContextEffectsSearchBudget(
            max_coarse_arms=1, max_fine_arms=1, max_joint_arms=0,
        ),
        refine_threshold_nats=0.5,
        drilldown_source_ids=[source_ids[0]],
        execution_adapter=SearchEffectAdapter({
            search_key(*source_ids): 2.0,
            search_key(source_ids[0]): 1.0,
        }),
        observation_store=store,
    )
    assert result.arm_for("coarse", source_ids).evidence_status == "trusted"

    canonical_plan = plan_context_effects(run)
    projection = route._publish_search_reader_projection(
        store, run, canonical_plan,
    )
    assert projection["trusted_singleton_source_ids"] == [source_ids[0]]
    assert projection["status"] == "partial"
    view = store.get_experiment(canonical_plan.experiment_id)
    measured = [
        row.intervention.source_ids[0]
        for row in view.arms if row.observation is not None
    ]
    assert measured == [source_ids[0]]


def test_effects_search_worker_attaches_validated_result_and_reader_projection(monkeypatch):
    run, source_ids = search_run(3)
    canonical_plan = plan_context_effects(run)
    request = route._normalize_effects_search_request(_effects_search_body(
        source_ids=list(source_ids),
        coarse_groups=[list(source_ids)],
        drilldown_source_ids=[source_ids[0]],
    ))
    spec = route._resolve_effects_search_spec(request, canonical_plan)
    measured = measure_context_effects_coarse_to_fine(
        run,
        source_ids=source_ids,
        coarse_groups=[source_ids],
        budget=ContextEffectsSearchBudget(
            max_coarse_arms=2, max_fine_arms=2, max_joint_arms=0,
        ),
        refine_threshold_nats=0.25,
        drilldown_source_ids=[source_ids[0]],
        execution_adapter=SearchEffectAdapter({
            search_key(*source_ids): 2.0,
            search_key(source_ids[0]): 1.0,
        }),
    )

    class Control:
        def __init__(self):
            self.events = []
            self.result = None

        def cancel_requested(self):
            return False

        def checkpoint(self, **event):
            self.events.append(event)

        def commit(self, persist):
            assert persist() is True

        def attach_result(self, result):
            self.result = result

    control = Control()
    monkeypatch.setattr(
        "clozn.runs.store.get_run",
        lambda run_id: run if run_id == run["id"] else None,
    )
    monkeypatch.setattr(route, "ObservationStore", lambda: object())
    monkeypatch.setattr(
        route, "measure_context_effects_coarse_to_fine",
        lambda *_args, **_kwargs: measured,
    )
    monkeypatch.setattr(
        route, "_publish_search_reader_projection",
        lambda *_args, **_kwargs: {
            "experiment_id": canonical_plan.experiment_id,
            "status": "partial",
            "control_observation_id": "obs_control",
            "trusted_singleton_source_ids": [source_ids[0]],
        },
    )
    worker = route._effects_search_worker(
        run, object(), request, spec, canonical_plan.experiment_id,
    )

    assert worker(control) == {"state": "completed"}
    schemas.validate(
        control.result,
        "clozn.context-investigation-effects-search-job-result.v1",
    )
    assert control.result["search_spec"] == spec
    assert control.result["reader_projection"]["status"] == "partial"
    assert all(
        "numerical_regime" in arm and "execution_identity" in arm
        for arm in control.result["arms"]
    )
    trusted = next(
        arm for arm in control.result["arms"]
        if arm["evidence_status"] == "trusted"
    )
    assert trusted["numerical_regime"] == "scalar_reference"
    assert trusted["execution_identity"] == {
        "kind": "clozn.score-tokens-scalar-reference.v1",
    }
    assert [event["phase"] for event in control.events] == [
        "planning", "planning", "publishing_reader_projection",
    ]


def test_effects_search_worker_never_promotes_unrefined_selected_native_row(monkeypatch):
    run, source_ids = search_run(2)
    canonical_plan = plan_context_effects(run)
    request = route._normalize_effects_search_request(_effects_search_body(
        source_ids=list(source_ids),
        execution_policy="native_screen_then_scalar_refine",
    ))
    spec = route._resolve_effects_search_spec(request, canonical_plan)
    unrefined = SimpleNamespace(
        arms=(SimpleNamespace(phase="fine", evidence_status="screening_only"),),
    )

    class Control:
        def cancel_requested(self):
            return False

        def checkpoint(self, **_event):
            pass

    monkeypatch.setattr(
        "clozn.runs.store.get_run",
        lambda run_id: run if run_id == run["id"] else None,
    )
    monkeypatch.setattr(route, "ObservationStore", lambda: object())
    monkeypatch.setattr(
        route, "measure_context_effects_coarse_to_fine",
        lambda *_args, **_kwargs: unrefined,
    )
    monkeypatch.setattr(
        route, "_publish_search_reader_projection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("unrefined evidence reached reader publication")
        ),
    )
    worker = route._effects_search_worker(
        run, object(), request, spec, canonical_plan.experiment_id,
    )

    outcome = worker(Control())
    assert outcome["state"] == "failed"
    assert outcome["error"]["code"] == (
        "context_investigation_effects_search_scalar_refinement_required"
    )


def test_effects_search_cancellation_retains_actual_work_without_reader_publication(monkeypatch):
    from clozn.server.influence_jobs import JobCancelled

    run, source_ids = search_run(2)
    canonical_plan = plan_context_effects(run)
    request = route._normalize_effects_search_request(_effects_search_body(
        source_ids=list(source_ids), coarse_groups=[list(source_ids)],
    ))
    spec = route._resolve_effects_search_spec(request, canonical_plan)
    measured = measure_context_effects_coarse_to_fine(
        run,
        source_ids=source_ids,
        coarse_groups=[source_ids],
        budget=ContextEffectsSearchBudget(
            max_coarse_arms=2, max_fine_arms=2, max_joint_arms=0,
        ),
        execution_adapter=SearchEffectAdapter({search_key(*source_ids): 1.0}),
    )

    class Control:
        def __init__(self):
            self.cancelled = False
            self.result = None

        def cancel_requested(self):
            return self.cancelled

        def checkpoint(self, **_event):
            pass

        def attach_cancellation_result(self, result):
            self.result = result

    control = Control()

    def cancel_after_measurement(*_args, **_kwargs):
        control.cancelled = True
        return measured

    monkeypatch.setattr(
        "clozn.runs.store.get_run",
        lambda run_id: run if run_id == run["id"] else None,
    )
    monkeypatch.setattr(route, "ObservationStore", lambda: object())
    monkeypatch.setattr(
        route, "measure_context_effects_coarse_to_fine",
        cancel_after_measurement,
    )
    monkeypatch.setattr(
        route, "_publish_search_reader_projection",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("cancelled search published reader evidence")
        ),
    )
    worker = route._effects_search_worker(
        run, object(), request, spec, canonical_plan.experiment_id,
    )

    with pytest.raises(JobCancelled):
        worker(control)
    schemas.validate(
        control.result,
        "clozn.context-investigation-effects-search-job-result.v1",
    )
    assert control.result["reader_projection"]["status"] == (
        "not_published_cancelled"
    )
    actual = control.result["budget_usage"]["actual_work"]
    assert actual == measured.budget_usage["actual_work"]
    assert actual["model_measurements_executed_total"] > 0


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


def test_action_job_paths_reject_a_same_run_job_of_the_wrong_kind(monkeypatch):
    class FakeJobs:
        def get(self, run_id, job_id):
            assert run_id == "run_kind"
            assert job_id == "infjob_other"
            return {
                "run_id": run_id,
                "job_id": job_id,
                "kind": "context_investigation_counterfactual",
            }

        def cancel(self, *_args, **_kwargs):
            raise AssertionError("wrong-kind job must not be cancelled")

    monkeypatch.setattr("clozn.server.influence_jobs.JOBS", FakeJobs())

    get_handler = Handler(
        "/runs/run_kind/context-investigation/remove-test/jobs/infjob_other"
    )
    assert route.try_get(get_handler, get_handler.path) is True
    assert get_handler.status == 404
    assert get_handler.body["code"] == "job_not_found"

    cancel_handler = Handler(
        "/runs/run_kind/context-investigation/remove-test/jobs/infjob_other/cancel"
    )
    assert route.try_post(cancel_handler, cancel_handler.path, {}) is True
    assert cancel_handler.status == 404
    assert cancel_handler.body["code"] == "job_not_found"


@pytest.mark.parametrize(
    ("worker_factory_name", "planner_name"),
    [
        ("_remove_test_worker", "plan_removability"),
        ("_counterfactual_worker", "plan_context_counterfactual"),
    ],
)
def test_source_action_workers_fail_closed_when_the_run_identity_changes(
    monkeypatch, worker_factory_name, planner_name,
):
    run, source_ids = _run()

    class Control:
        def checkpoint(self, **_event):
            raise AssertionError("stale action reached model work")

    monkeypatch.setattr(
        "clozn.runs.store.get_run",
        lambda run_id: run if run_id == run["id"] else None,
    )
    monkeypatch.setattr(
        route,
        planner_name,
        lambda *_args, **_kwargs: SimpleNamespace(experiment_id="experiment_changed"),
    )
    worker = getattr(route, worker_factory_name)(
        run, object(), source_ids[0], "experiment_original",
    )

    outcome = worker(Control())

    assert outcome["state"] == "failed"
    assert outcome["error"]["code"] == "context_investigation_run_changed"


@pytest.mark.parametrize(
    ("path", "kind"),
    [
        (
            "/runs/run_failure/context-investigation/remove-test/jobs/infjob_failure",
            "context_investigation_remove_test",
        ),
        (
            "/runs/run_failure/context-investigation/counterfactuals/jobs/infjob_failure",
            "context_investigation_counterfactual",
        ),
    ],
)
def test_source_action_job_routes_preserve_worker_failures(monkeypatch, path, kind):
    failure = {
        "schema_version": "clozn.influence-map-job.v1",
        "run_id": "run_failure",
        "job_id": "infjob_failure",
        "kind": kind,
        "state": "failed",
        "error": {
            "code": f"{kind}_job_failed",
            "message": "RuntimeError: worker disconnected",
        },
    }

    class FakeJobs:
        def get(self, run_id, job_id):
            assert (run_id, job_id) == ("run_failure", "infjob_failure")
            return failure

    monkeypatch.setattr("clozn.server.influence_jobs.JOBS", FakeJobs())

    handler = Handler(path)
    assert route.try_get(handler, handler.path) is True
    assert handler.status == 200
    assert handler.body is failure
    assert handler.body["state"] == "failed"
    assert handler.body["error"]["code"] == f"{kind}_job_failed"


def test_counterfactual_materialization_reuses_existing_child_idempotently(monkeypatch):
    parent = {"id": "run_parent"}
    request = {
        "experiment_id": "exp_fixture",
        "arm_id": "arm_fixture",
        "observation_id": "obs_fixture",
    }
    child = {
        "id": "run_child",
        "parent_run_id": parent["id"],
        "trace": {},
        "changes_applied": {
            "experiment": {
                **request,
                "operation": "delete_source_generate",
                "base_state": {
                    "realized_fidelity": "direct_context_generation",
                },
            },
        },
    }
    monkeypatch.setattr(
        "clozn.runs.store.get_run",
        lambda run_id: parent if run_id == parent["id"] else None,
    )
    monkeypatch.setattr("clozn.runs.store.iter_runs", lambda: [child])
    monkeypatch.setattr(
        "clozn.analysis.model_diff.diff_runs",
        lambda left, right: {"run_a": left["id"], "run_b": right["id"]},
    )
    monkeypatch.setattr(
        route,
        "materialize_generated_observation",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("existing materialization created another child")
        ),
    )

    handler = Handler(
        f"/runs/{parent['id']}/context-investigation/counterfactuals/materialize"
    )
    assert route.try_post(handler, handler.path, request) is True
    assert handler.status == 201
    assert handler.body["child_run_id"] == child["id"]
    assert handler.body["parent_run_id"] == parent["id"]
    assert handler.body["compare_path"] == "#/compare/run_parent/run_child"
    schemas.validate(handler.body, "clozn.experiment-materialization.v1")
