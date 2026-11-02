from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

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


def test_reader_configuration_projects_the_authoritative_message_partition(monkeypatch):
    run, _source_ids = _run()
    roots = [row["segment_id"] for row in run["context_receipt"]["delivered"]]
    plan = plan_context_effects(run, source_ids=roots)
    monkeypatch.setattr(route, "ObservationStore", lambda: MemoryStore())
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)

    handler = Handler(
        f"/runs/{run['id']}/context-investigation/reader"
        "?floor_nats=0.1&granularity=message&max_arms=3"
    )
    assert route.try_get(handler, handler.path) is True

    assert handler.status == 200
    assert handler.body["status"] == "not_measured"
    assert handler.body["measurement"]["source_universe"] == roots
    assert handler.body["measurement"]["experiment_id"] == plan.experiment_id


def test_reader_accepts_an_explicit_custom_source_selection(monkeypatch):
    run, source_ids = _run()
    selected = source_ids[:2]
    plan = plan_context_effects(run, source_ids=selected)
    monkeypatch.setattr(route, "ObservationStore", lambda: MemoryStore())
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)

    handler = Handler(
        f"/runs/{run['id']}/context-investigation/reader"
        f"?floor_nats=0.1&source_id={selected[0]}&source_id={selected[1]}"
    )
    assert route.try_get(handler, handler.path) is True

    assert handler.status == 200
    assert handler.body["measurement"]["source_universe"] == selected
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


def test_query_configuration_reuses_only_its_persisted_partition(monkeypatch):
    run, _source_ids = _run()
    roots = [row["segment_id"] for row in run["context_receipt"]["delivered"]]
    store = MemoryStore(_view(run, roots))
    monkeypatch.setattr(route, "ObservationStore", lambda: store)
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("query selected a model")),
    )

    handler = Handler(f"/runs/{run['id']}/context-investigation/query")
    assert route.try_post(handler, handler.path, {
        "answer_start": 5,
        "answer_end": 11,
        "granularity": "message",
        "max_arms": 3,
    }) is True

    assert handler.status == 200
    assert handler.body["status"] == "completed"
    assert {effect["source_id"] for effect in handler.body["effects"]} == set(roots)


def test_query_accepts_an_explicit_custom_source_selection(monkeypatch):
    run, source_ids = _run()
    selected = source_ids[:1]
    store = MemoryStore(_view(run, selected))
    monkeypatch.setattr(route, "ObservationStore", lambda: store)
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)

    handler = Handler(f"/runs/{run['id']}/context-investigation/query")
    assert route.try_post(handler, handler.path, {
        "answer_start": 5,
        "answer_end": 11,
        "source_ids": selected,
    }) is True

    assert handler.status == 200
    assert handler.body["status"] == "completed"
    assert [effect["source_id"] for effect in handler.body["effects"]] == selected


def test_effects_job_request_accepts_only_canonical_source_ids(monkeypatch):
    run, source_ids = _run()
    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)

    handler = Handler(f"/runs/{run['id']}/context-investigation/effects/jobs")
    assert route.try_post(handler, handler.path, {"execution_policy": "fast"}) is True
    assert handler.status == 400
    assert handler.body["code"] == "unsupported_effect_request"

    assert route.try_post(handler, handler.path, {"source_ids": ["", "src"]}) is True
    assert handler.status == 400
    assert handler.body["code"] == "invalid_source_ids"

    assert route._normalize_effect_request({
        "granularity": "adaptive", "max_arms": 12,
        }) == {
            "source_ids": None, "granularity": "adaptive", "max_arms": 12,
            "scoring_mode": "exact",
        }
    assert route._normalize_effect_request({
        "granularity": "adaptive", "max_arms": 12, "scoring_mode": "fast",
    })["scoring_mode"] == "fast"
    with pytest.raises(route.ContextInvestigationRouteError, match="scoring_mode"):
        route._normalize_effect_request({"scoring_mode": "turbo"})
    with pytest.raises(route.ContextInvestigationRouteError, match="cannot be combined"):
        route._normalize_effect_request({
            "source_ids": source_ids[:1],
            "granularity": "message",
            "max_arms": 3,
        })


def test_effects_job_derives_configured_message_partition(monkeypatch):
    run, _source_ids = _run()
    calls = []

    class FakeJobs:
        def start(self, run_id, worker, *, kind):
            calls.append((run_id, worker, kind))
            return {"job_id": "infjob_configured", "state": "queued", "kind": kind}

    class ScoreOnly:
        def score_tokens(self, *_args, **_kwargs):
            raise AssertionError("route start must not score synchronously")

    captured = {}

    def job_worker(_run, _sub, source_ids, plan_id, *, scoring_mode="exact"):
        captured["source_ids"] = source_ids
        captured["plan_id"] = plan_id
        captured["scoring_mode"] = scoring_mode
        return lambda _control: {"state": "completed"}

    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr("clozn.server.influence_jobs.JOBS", FakeJobs())
    monkeypatch.setattr(route, "_job_worker", job_worker)
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: SimpleNamespace(sub=ScoreOnly()),
    )

    handler = Handler(f"/runs/{run['id']}/context-investigation/effects/jobs")
    assert route.try_post(handler, handler.path, {
        "granularity": "message", "max_arms": 3,
    }) is True

    assert handler.status == 202
    assert captured["source_ids"] == [
        row["segment_id"] for row in run["context_receipt"]["delivered"]
    ]
    assert captured["plan_id"] == plan_context_effects(
        run, source_ids=captured["source_ids"],
    ).experiment_id


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
    checkpoints = []
    measurement_options = {}

    class Control:
        def checkpoint(self, **event):
            checkpoints.append(event)

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
    result = SimpleNamespace(
        experiment_id=plan.experiment_id,
        arms=rows,
        state="completed",
        control_disposition="executed",
        arm_dispositions=("reused", "executed"),
    )

    def measure(*_args, **kwargs):
        measurement_options.update(kwargs)
        kwargs["on_progress"]({
            "phase": "scoring_recorded_continuation",
            "completed_units": 1,
            "total_units": 2,
        })
        return result

    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr(route, "measure_context_effects", measure)
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
        "executed_score_count": 2,
        "reused_score_count": 1,
        "elapsed_seconds": attached[0]["elapsed_seconds"],
        "scoring_mode": "exact",
        "evidence_quality": "exact",
        "execution_strategy": "scalar_serial",
        "numerical_regime": "scalar_reference",
        "state": "completed",
    }]
    assert attached[0]["elapsed_seconds"] >= 0
    assert measurement_options["arm_batch_size"] == 1
    assert measurement_options["batch_control"] is False
    assert checkpoints[1] == {
        "phase": "scoring_recorded_continuation",
        "completed": 1,
        "total": 2,
    }


def test_fast_sequence_batch_size_uses_recorded_capacity_conservatively():
    run, _source_ids = _run()
    run["meta"]["n_ctx"] = 8192
    run["meta"]["prompt_tokens"] = 115
    run["trace"]["token_ids"] = list(range(64))

    assert route._fast_sequence_batch_size(run) == 16

    run["meta"]["n_ctx"] = 512
    assert route._fast_sequence_batch_size(run) == 2

    run["meta"]["n_ctx"] = 8192
    run["meta"]["prompt_tokens"] = 2000
    run["trace"]["token_ids"] = list(range(8))
    assert route._fast_sequence_batch_size(run) == 2


def test_fast_sequence_batch_size_falls_back_without_recorded_token_facts():
    run, _source_ids = _run()
    run["meta"].pop("prompt_tokens", None)

    assert route._fast_sequence_batch_size(run) == 4


def test_fast_execution_plan_selects_scalar_for_long_answer_and_native_for_prompt_heavy_work():
    run, _source_ids = _run()
    run["meta"]["n_ctx"] = 8192
    run["meta"]["prompt_tokens"] = 115
    run["trace"]["token_ids"] = list(range(64))

    scalar = route._fast_execution_plan(run, sequence_count=6)
    assert scalar["selected_strategy"] == "scalar_prefix_cache"
    assert scalar["reason"] == "continuation_decode_dominates"

    run["meta"]["prompt_tokens"] = 798
    run["trace"]["token_ids"] = list(range(14))
    native = route._fast_execution_plan(run, sequence_count=4)
    assert native["selected_strategy"] == "native_prefix_trie"
    assert native["reason"] == "prompt_dominant_short_continuation"
    assert native["native_sequence_batch_size"] >= 4


def test_fast_compact_result_never_upgrades_mixed_native_evidence_to_exact():
    approximate = SimpleNamespace(completed=True, trusted=False)
    exact = SimpleNamespace(completed=True, trusted=True)
    result = SimpleNamespace(
        experiment_id="exp_mixed",
        control=approximate,
        arms=[
            SimpleNamespace(
                observation=approximate, state="completed",
                diagnostics={
                    "execution_strategy": "native_prefix_trie",
                    "numerical_regime": "native_prefix_trie_target_logprob_v1",
                },
            ),
            SimpleNamespace(
                observation=exact, state="completed",
                diagnostics={
                    "execution_strategy": "scalar_serial",
                    "numerical_regime": "scalar_reference",
                },
            ),
        ],
        state="completed",
        diagnostics={},
        control_disposition="executed",
        arm_dispositions=("executed", "executed"),
    )

    compact = route._compact_job_result(
        result, elapsed_seconds=1.0, scoring_mode="fast",
    )
    assert compact["evidence_quality"] == "approximate"
    assert compact["execution_strategy"] == "mixed"
    assert compact["numerical_regime"] == "mixed"


def test_effects_job_paths_reject_a_job_of_the_wrong_kind(monkeypatch):
    run, _source_ids = _run()

    class FakeJobs:
        def get(self, _run_id, _job_id):
            return {"kind": "different_analysis", "state": "running"}

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


def test_lean_context_job_uses_fixed_tolerance_with_configurable_budget(monkeypatch):
    run, _source_ids = _run()
    calls = []

    class FakeJobs:
        def start(self, run_id, worker, *, kind):
            calls.append((run_id, worker, kind))
            return {"job_id": "infjob_lean", "state": "queued", "kind": kind}

    class ScoreOnly:
        def score_tokens(self, *_args, **_kwargs):
            raise AssertionError("route start must not score synchronously")

        def chat(self, *_args, **_kwargs):
            raise AssertionError("route start must not generate synchronously")

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
    assert route.try_post(handler, path, {"max_new_executions": 32}) is True
    assert handler.status == 202
    assert handler.body["kind"] == "context_investigation_lean_context"
    assert calls[0][0] == run["id"]
    assert calls[0][2] == "context_investigation_lean_context"


@pytest.mark.parametrize("value", [0, 65, -1, True, 1.5, "16"])
def test_lean_context_job_rejects_invalid_search_budgets(value):
    with pytest.raises(route.ContextInvestigationRouteError) as caught:
        route._normalize_lean_request({"max_new_executions": value})
    assert caught.value.code == "invalid_lean_context_budget"


def test_lean_context_worker_attaches_the_validated_full_result(monkeypatch):
    run, _source_ids = _run()

    class Engine:
        def apply_template_info(self, messages):
            return {"prompt_tokens": sum(len(str(item.get("content", ""))) for item in messages)}

    expected = run_lean_context(run, substrate=ScoreSubstrate(), engine=Engine())
    attached = []
    received = {}

    class Control:
        def checkpoint(self, **_event):
            return None

        def cancel_requested(self):
            return False

        def attach_result(self, payload):
            attached.append(payload)

    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    def lean_context(*_args, **kwargs):
        received.update(kwargs)
        return expected

    monkeypatch.setattr(route, "run_lean_context", lean_context)
    monkeypatch.setattr(route, "ObservationStore", lambda: object())
    universe_id = run["context_units"]
    from clozn.runs.context_search_universe import plan_context_search_universe
    planned = plan_context_search_universe(run, universe_id)

    outcome = route._lean_job_worker(
        run, object(), object(), planned["universe_id"], {"max_new_executions": 32},
    )(Control())

    assert outcome == {"state": "completed"}
    assert attached == [expected]
    assert received["max_new_executions"] == 32
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


def test_interactions_job_is_opt_in_configurable_and_uses_shared_registry(monkeypatch):
    run, _source_ids = _run()
    calls = []

    class FakeJobs:
        def start(self, run_id, worker, *, kind):
            calls.append((run_id, worker, kind))
            return {"job_id": "infjob_interactions", "state": "queued", "kind": kind}

    class ScoreOnly:
        def score_tokens(self, *_args, **_kwargs):
            raise AssertionError("route start must not score synchronously")

    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr("clozn.server.influence_jobs.JOBS", FakeJobs())
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: SimpleNamespace(sub=ScoreOnly(), engine=object()),
    )
    path = f"/runs/{run['id']}/context-investigation/interactions/jobs"

    rejected = Handler(path)
    assert route.try_post(rejected, path, {"semantic_matching": True}) is True
    assert rejected.status == 400
    assert rejected.body["code"] == "unsupported_interactions_request"

    handler = Handler(path)
    assert route.try_post(handler, path, {
        "max_new_executions": 24,
        "max_pairs": 40,
        "seed": 7,
    }) is True
    assert handler.status == 202
    assert handler.body["kind"] == "context_investigation_interactions"
    assert calls[0][0] == run["id"]
    assert calls[0][2] == "context_investigation_interactions"


@pytest.mark.parametrize("body", [
    {"max_new_executions": -1},
    {"max_new_executions": 65},
    {"max_pairs": 0},
    {"max_pairs": 257},
    {"seed": -1},
    {"seed": True},
])
def test_interactions_job_rejects_invalid_budgets(body):
    with pytest.raises(route.ContextInvestigationRouteError):
        route._normalize_interactions_request(body)


def test_interactions_worker_streams_partial_estimates_and_attaches_final_result(monkeypatch):
    run, _source_ids = _run()
    from clozn.runs.context_search_universe import plan_context_search_universe
    universe = plan_context_search_universe(run, run["context_units"])
    attached = []
    checkpoints = []

    class Control:
        def checkpoint(self, **event):
            checkpoints.append(event)

        def cancel_requested(self):
            return False

        def attach_result(self, payload):
            attached.append(payload)

    def sample(_run, **kwargs):
        kwargs["progress"]({"used_new_executions": 2})
        kwargs["sample_progress"]({
            "sample": {
                "source_id": universe["source_ids"][0],
                "without_source_ids": [],
                "delta_nats": 1.25,
            },
            "completed_pairs": 1,
            "used_new_executions": 2,
            "reused_observation_count": 1,
        })
        return {
            "run_id": run["id"],
            "status": "completed",
            "stop_reason": "execution_budget",
            "seed": kwargs["seed"],
            "universe": universe,
            "budget": {
                "max_new_executions": kwargs["max_new_executions"],
                "used_new_executions": 4,
                "reused_observation_count": 1,
                "completed_pairs": 1,
            },
            "source_estimates": [{
                "source_id": source_id,
                "sample_count": 1 if index == 0 else 0,
                "unique_pair_count": 1 if index == 0 else 0,
                "mean_marginal_nats": 1.25 if index == 0 else None,
                "sample_stddev_nats": None,
                "leave_one_out_nats": None,
                "mean_minus_leave_one_out_nats": None,
            } for index, source_id in enumerate(universe["source_ids"])],
            "elapsed_seconds": 0.4,
            "best": None,
        }

    monkeypatch.setattr("clozn.runs.store.get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr(route, "run_random_coalitions", sample)
    monkeypatch.setattr(route, "ObservationStore", lambda: object())

    outcome = route._interactions_job_worker(
        run,
        object(),
        object(),
        universe,
        {"max_new_executions": 4, "max_pairs": 8, "seed": 3},
    )(Control())

    assert outcome == {"state": "completed"}
    assert len(attached) == 2
    assert attached[0]["status"] == "running"
    assert attached[0]["source_estimates"][0]["mean_marginal_nats"] == 1.25
    assert attached[1]["status"] == "completed"
    assert attached[1]["elapsed_seconds"] == 0.4
    schemas.validate(attached[0], "clozn.context-interactions-result.v1")
    schemas.validate(attached[1], "clozn.context-interactions-result.v1")
    assert checkpoints[-1] == {
        "phase": "sampling_context_interactions",
        "completed": 2,
        "total": 4,
    }


def test_interactions_job_paths_reject_a_job_of_the_wrong_kind(monkeypatch):
    run, _source_ids = _run()

    class FakeJobs:
        def get(self, _run_id, _job_id):
            return {"kind": "context_investigation_effects", "state": "running"}

        def cancel(self, *_args):
            raise AssertionError("wrong-kind job must not be cancelled")

    monkeypatch.setattr("clozn.server.influence_jobs.JOBS", FakeJobs())
    path = f"/runs/{run['id']}/context-investigation/interactions/jobs/job_other"
    handler = Handler(path)
    assert route.try_get(handler, path) is True
    assert handler.status == 404

    cancel_path = f"{path}/cancel"
    handler = Handler(cancel_path)
    assert route.try_post(handler, cancel_path, {}) is True
    assert handler.status == 404
