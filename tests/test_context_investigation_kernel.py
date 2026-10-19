from __future__ import annotations

from copy import deepcopy

import pytest

from clozn import schemas
from clozn.experiments.context_investigation import (
    DEFAULT_MEASUREMENT_FLOOR_NATS,
    build_context_investigation_reader,
    project_locus_details,
    project_source_loci,
    query_answer_effects,
)
from clozn.experiments.observations import TokenScoreObservation
from clozn.experiments.persistence import ExperimentView, ObservationNotFound, ObservationStore
from clozn.experiments.selections import AnswerSelection
from clozn.recipes.context_effects import measure_context_effects, plan_context_effects
from clozn.recipes.context_effects import plan_context_effects_reader
from clozn.runs.context_receipt import build_context_receipt
from clozn.runs.context_units import build_context_unit_manifest

from tests.test_context_effects_kernel import ScoreSubstrate, _run


class MemoryStore(ObservationStore):
    def __init__(self, view=None):
        super().__init__()
        self.view = view

    def get_experiment(self, experiment_id):
        if self.view is None:
            raise ObservationNotFound(experiment_id)
        return self.view


def _protected_only_run():
    run, _source_ids = _run()
    messages = [{"role": "user", "content": "only current request"}]
    run["id"] = "run_context_investigation_protected_only"
    run["messages"] = deepcopy(messages)
    run["assembled_messages"] = deepcopy(messages)
    run["final_prompt"] = "only current request"
    run["context_receipt"] = build_context_receipt(
        messages=messages,
        assembled_messages=messages,
        final_prompt=run["final_prompt"],
        run_id=run["id"],
        privacy="full",
    )
    run["context_units"] = build_context_unit_manifest(run)
    return run


def _view(run, source_ids):
    result = measure_context_effects(run, source_ids=source_ids, substrate=ScoreSubstrate())
    return ExperimentView(
        experiment_id=result.experiment_id,
        base=result.base,
        evaluator=result.evaluator,
        control=result.control,
        arm_rows=result.arm_rows,
        state=result.state,
    )


def test_q2_plan_preserves_identity_and_separates_display_from_measurement():
    run, source_ids = _run()
    first = plan_context_effects(run, source_ids=source_ids[:2])
    second = plan_context_effects(deepcopy(run), source_ids=source_ids[:2])

    assert first.experiment_id == second.experiment_id
    assert [arm.arm_id for arm in first.experiment.arms] == [
        arm.arm_id for arm in second.experiment.arms
    ]
    assert first.measurement_source_ids == tuple(source_ids[:2])
    assert first.display_source_ids == tuple(source_ids)


def test_reader_before_measurement_has_context_and_answer_without_model_work():
    run, source_ids = _run()
    plan = plan_context_effects(run, source_ids=source_ids[:2])
    document = build_context_investigation_reader(
        run, plan, observation_store=MemoryStore(),
    )

    assert document["status"] == "not_measured"
    assert [block["text"] for block in document["context"]["blocks"]] == [
        "stable context", "removable source", "current question",
    ]
    assert document["answer"]["text"] == "same answer"
    assert document["loci"] == []
    schemas.validate(document, "clozn.context-investigation-reader.v1")


def test_reader_plan_preserves_prose_without_minting_an_empty_q2_experiment():
    run = _protected_only_run()
    with pytest.raises(ValueError, match="no automatically discovered sources are exactly removable"):
        plan_context_effects(run)

    plan = plan_context_effects_reader(run)
    assert plan.measurement_source_ids == ()
    assert plan.display_source_ids
    assert not hasattr(plan, "experiment")

    class NoExperimentStore(ObservationStore):
        def get_experiment(self, _experiment_id):
            raise AssertionError("an unavailable measurement must not look up an Experiment")

    document = build_context_investigation_reader(
        run, plan, observation_store=NoExperimentStore(),
    )
    assert document["status"] == "unavailable"
    assert document["reason"] == "no canonical Context Receipt source is exactly removable"
    assert document["reason_code"] == "context_investigation_no_exactly_removable_source"
    assert document["measurement"] == {
        "status": "unavailable",
        "experiment_id": None,
        "evaluator": "score_recorded_continuation",
        "intervention": "delete_source",
        "source_universe": [],
        "source_universe_basis": "canonical_context_units",
        "completed_source_count": 0,
        "unavailable_source_count": 0,
        "failed_source_count": 0,
        "not_measured_source_count": 0,
    }
    assert [block["text"] for block in document["context"]["blocks"]] == [
        "only current request",
    ]
    assert document["answer"]["text"] == run["response"]
    assert document["sources"]
    assert {source["measurement"]["status"] for source in document["sources"]} == {
        "unavailable",
    }
    assert document["loci"] == []
    assert document["capabilities"] == {
        "context_effects": "unavailable",
        "measurement": "unavailable",
        "answer_selection_projection": "available",
    }
    schemas.validate(document, "clozn.context-investigation-reader.v1")


def test_reader_projects_direct_loci_and_keeps_protected_display_source_unmeasured():
    run, source_ids = _run()
    plan = plan_context_effects(run, source_ids=source_ids[:2])
    view = _view(run, source_ids[:2])
    document = build_context_investigation_reader(
        run, plan, observation_store=MemoryStore(view), floor_nats=0.1,
    )

    assert document["status"] == "completed"
    assert document["measurement"]["completed_source_count"] == 2
    protected = next(item for item in document["sources"] if item["source_id"] == source_ids[2])
    assert protected["measurement"]["status"] == "not_measured"
    assert document["loci"]
    details = project_locus_details(document, document["loci"][0]["locus_id"])
    assert details["experiment"]["experiment_id"] == view.experiment_id
    assert details["evaluator"] == "score_recorded_continuation"
    schemas.validate(document, "clozn.context-investigation-reader.v1")


def test_locus_grouping_respects_floor_sign_and_gaps():
    run, source_ids = _run()
    view = _view(run, source_ids[:1])
    template = view.control

    def score(logprobs):
        return TokenScoreObservation(
            status="completed",
            recorded_token_ids=[1, 2, 3, 4],
            token_pieces=["a", "b", "c", "d"],
            token_spans=[[0, 1], [1, 2], [2, 3], [3, 4]],
            token_logprobs=logprobs,
            total_continuation_logprob=sum(logprobs),
            run_id=template.run_id,
            base_execution_fingerprint=template.base_execution_fingerprint,
            evaluator=template.evaluator,
            condition=template.condition,
            contract=template.contract,
            score_basis=template.score_basis,
        )

    baseline = score([0.0, 0.0, 0.0, 0.0])
    support = score([0.0, -0.4, -0.7, 0.0])
    loci = project_source_loci(
        baseline, support, source_id="src_a", floor_nats=0.1,
        experiment_id="exp_a", arm_id="arm_a",
    )
    assert [(item["answer_token_range"], item["direction"], item["delta_nats"]) for item in loci] == [
        ([1, 3], "support", 1.1),
    ]
    assert loci[0]["classification"] == "measured_support"

    gap = score([0.0, -0.4, -0.01, -0.5])
    gap_loci = project_source_loci(
        baseline, gap, source_id="src_a", floor_nats=0.1,
        experiment_id="exp_a", arm_id="arm_a",
    )
    assert [item["answer_token_range"] for item in gap_loci] == [[1, 2], [3, 4]]


def test_locus_projection_does_not_turn_eos_effects_into_empty_text_ranges():
    run, source_ids = _run()
    template = _view(run, source_ids[:1]).control

    def score(logprobs):
        return TokenScoreObservation(
            status="completed",
            recorded_token_ids=[1, 2],
            token_pieces=["answer", ""],
            token_spans=[[0, 6], [6, 6]],
            token_logprobs=logprobs,
            total_continuation_logprob=sum(logprobs),
            run_id=template.run_id,
            base_execution_fingerprint=template.base_execution_fingerprint,
            evaluator=template.evaluator,
            condition=template.condition,
            contract=template.contract,
            score_basis=template.score_basis,
        )

    baseline = score([0.0, 0.0])
    intervention = score([-0.4, -0.8])

    loci = project_source_loci(
        baseline, intervention, source_id="src_eos", floor_nats=0.1,
        experiment_id="exp_eos", arm_id="arm_eos",
    )

    assert [item["answer_token_range"] for item in loci] == [[0, 1]]
    assert [item["answer_unicode_range"] for item in loci] == [[0, 6]]
    assert loci[0]["answer_text"] == "answer"


def test_reader_stale_baseline_does_not_retokenize_response():
    run, source_ids = _run()
    plan = plan_context_effects(run, source_ids=source_ids[:1])
    view = _view(run, source_ids[:1])
    baseline = view.control
    stale = TokenScoreObservation(
        status="completed",
        recorded_token_ids=baseline.recorded_token_ids,
        token_pieces=["stale", " answer"],
        token_spans=[[0, 5], [5, 12]],
        token_logprobs=baseline.token_logprobs,
        total_continuation_logprob=baseline.total_continuation_logprob,
        run_id=baseline.run_id,
        base_execution_fingerprint=baseline.base_execution_fingerprint,
        evaluator=baseline.evaluator,
        condition=baseline.condition,
        contract=baseline.contract,
        score_basis=baseline.score_basis,
    )
    stale_view = ExperimentView(
        experiment_id=view.experiment_id, base=view.base, evaluator=view.evaluator,
        control=stale, arm_rows=view.arm_rows, state=view.state,
    )
    document = build_context_investigation_reader(
        run, plan, observation_store=MemoryStore(stale_view),
    )
    assert document["status"] == "stale"
    assert "reconstruct" in document["reason"]


def test_answer_query_reuses_vectors_and_ranks_signed_effects():
    run, source_ids = _run()
    view = _view(run, source_ids[:2])
    result = query_answer_effects(view, AnswerSelection("answer"))
    assert result["selection"]["text"] == "answer"
    assert result["effects"][0]["delta_nats"] is not None
    assert result["effects"][0]["direction"] in {"support", "suppression", "below_floor"}
    assert result["effects"][0]["classification"] in {
        "measured_support", "measured_suppression", "below_floor",
    }
    assert result["measurement_floor_nats"] == DEFAULT_MEASUREMENT_FLOOR_NATS


def test_exact_span_display_address_is_not_shortened():
    first = "## Removal\n" + ("remove " * 12)
    second = "## Other\n" + ("other " * 12)
    content = first + "\n\n" + second
    messages = [
        {"role": "user", "content": content},
        {"role": "user", "content": "current question"},
    ]
    clean = deepcopy(messages)
    run = {
        "id": "run_context_investigation_span",
        "messages": clean,
        "assembled_messages": deepcopy(clean),
        "context_receipt": build_context_receipt(
            messages=messages, assembled_messages=clean,
            run_id="run_context_investigation_span", privacy="full",
        ),
        "response": "yes",
        "trace": {"steps": [{"token_id": 1, "piece": "yes"}]},
    }
    run["context_units"] = build_context_unit_manifest(run)
    span_id = run["context_receipt"]["delivered"][0]["sources"][0]["source_id"]
    plan = plan_context_effects(run, source_ids=[span_id])
    document = build_context_investigation_reader(run, plan, observation_store=MemoryStore())
    source = next(item for item in document["sources"] if item["source_id"] == span_id)
    assert source["granularity"] == "exact_span"
    expected = run["context_receipt"]["delivered"][0]["sources"][0]["unicode_range"]
    assert source["unicode_range"] == expected
    assert source["display_unicode_range"] == expected
