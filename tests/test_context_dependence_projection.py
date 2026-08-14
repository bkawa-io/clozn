"""Exact read-only answer-selection projections for Context Dependence v2."""
from __future__ import annotations

import copy

import pytest

import clozn.runs.store as runlog
from clozn import schemas
from clozn.receipts.context_dependence import ContextDependenceStudy, measure_removal_effect
from clozn.runs.context_dependence_projection import (
    ContextDependenceProjectionError,
    build_context_dependence_query,
)
from clozn.runs.context_receipt import build_context_receipt
from clozn.server.routes import context_dependence as route


class Handler:
    def __init__(self, path):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


_RESPONSE = "Hi 🌍!"
_IDS = ("cds_" + "a" * 24, "cdx_" + "b" * 24, "cdx_" + "c" * 24)


def _study(*, vector_a=None, schema_version="clozn.context-dependence-study.v2"):
    # Python code-point coordinates: the globe occupies exactly one position.
    tokens = [
        {"index": 0, "token_id": 101, "piece": "H", "unicode_range": [0, 1], "logprob": -0.2},
        {"index": 1, "token_id": 102, "piece": "i", "unicode_range": [1, 2], "logprob": -0.3},
        {"index": 2, "token_id": 103, "piece": " ", "unicode_range": [2, 3], "logprob": -0.4},
        {"index": 3, "token_id": 104, "piece": "🌍", "unicode_range": [3, 4], "logprob": -0.5},
        {"index": 4, "token_id": 105, "piece": "!", "unicode_range": [4, 5], "logprob": -0.6},
    ]
    vector_a = vector_a if vector_a is not None else [0.1, 0.2, 0.3, 0.4, 0.5]
    vector_b = [-0.1, -0.2, -0.3, -0.4, -0.5]

    def experiment(experiment_id, source_id, vector):
        total = sum(vector)
        return {
            "experiment_id": experiment_id,
            "intervention_operator": "delete_source",
            "removed_source_ids": [source_id],
            "exact_removed_ranges": [{
                "source_id": source_id, "message_index": 0 if source_id == "source_a" else 1,
                "unicode_range": [0, 10], "byte_range": [0, 10],
            }],
            "context_hash": "d" * 64,
            "intervened_logp": -2.0 - total,
            "baseline_logp": -2.0,
            "delta_nats": total,
            "per_token_delta_nats": vector,
            "token_indices": [0, 1, 2, 3, 4],
            "provenance": "measured",
        }

    return {
        "schema_version": schema_version,
        "study_id": _IDS[0],
        "run_id": "run_projection",
        "context_model_identity": {},
        "source_identity": {
            "kind": "context_receipt_segment_id", "view": "delivered",
            "sources": [
                {"source_id": "source_a", "segment_id": "segment_a", "message_index": 0,
                 "unicode_range": [0, 10], "byte_range": [0, 10], "role": "user",
                 "source_label": "Policy §4", "provenance_kind": "retrieved_document"},
                {"source_id": "source_b", "segment_id": "segment_b", "message_index": 1,
                 "unicode_range": [0, 10], "byte_range": [0, 10], "role": "system",
                 "source_label": "System instruction", "provenance_kind": "system_instruction"},
            ],
        },
        "continuation": {
            "kind": "recorded_token_ids", "fidelity": "exact_recorded_token_ids",
            "token_ids_exact": True, "retokenized": False, "recorded_text": _RESPONSE,
            "scored_text": _RESPONSE, "unicode_offset_basis": "recorded_response_unicode",
            "token_ids": [101, 102, 103, 104, 105],
        },
        "baseline": {
            "teacher_forced_logp": -2.0, "context_hash": "e" * 64, "scored_once": True,
            "provenance": "measured", "tokens": tokens,
        },
        "experiments": [
            experiment(_IDS[1], "source_a", vector_a),
            experiment(_IDS[2], "source_b", vector_b),
        ],
        "budget": {"passes_requested": 3, "passes_consumed": 3},
    }


def _run(**overrides):
    run = {"id": "run_projection", "response": _RESPONSE, "context_dependence_study": _study()}
    run.update(overrides)
    return run


def test_exact_unicode_selection_projects_every_direct_full_vector_without_mutating_run():
    run = _run()
    before = copy.deepcopy(run)

    document = build_context_dependence_query(run, output_start=3, output_end=4)

    schemas.validate(document, "clozn.context-dependence-query.v1")
    assert document["selection"] == {
        "unicode_range": [3, 4],
        "text": "🌍",
        "registration": "exact",
        "fidelity": "exact_recorded_token_ids",
        "recorded_token_range": [3, 4],
        "conditioned_prefix": {
            "unicode_range": [0, 3], "recorded_token_range": [0, 3], "text": "Hi ",
        },
    }
    assert [(effect["experiment_id"], effect["delta_nats"], effect["full_continuation_delta_nats"])
            for effect in document["measured_removal_effects"]] == [
        (_IDS[1], 0.4, 1.5), (_IDS[2], -0.4, -1.5),
    ]
    assert document["measured_removal_effects"][0]["sources"] == [{
        "source_id": "source_a", "segment_id": "segment_a", "message_index": 0,
        "unicode_range": [0, 10], "byte_range": [0, 10], "role": "user",
        "source_label": "Policy §4", "provenance_kind": "retrieved_document",
    }]
    assert run == before


def test_projection_consumes_live_span_aware_v2_measurement_output():
    content = "Policy four. Question?"
    journal_messages = [{
        "role": "user", "content": content,
        "_clozn_sources": [{
            "source_id": "policy-four", "label": "Policy §4",
            "unicode_range": [0, 12], "provenance_kind": "retrieved_document",
        }],
    }]
    model_messages = [{"role": "user", "content": content}]
    receipt = build_context_receipt(
        messages=journal_messages, assembled_messages=model_messages,
        run_id="run_live_projection", privacy="full",
    )
    run = {
        "id": "run_live_projection", "messages": journal_messages,
        "assembled_messages": model_messages, "context_receipt": receipt,
        "response": "is eligible", "trace": {"token_ids": [7, 8]},
    }
    span_id = receipt["assembled"][0]["sources"][0]["source_id"]

    class ScoreSub:
        def score_tokens(self, messages, continuation_ids, **_kwargs):
            penalty = 1.0 if messages[0]["content"] == " Question?" else 0.0
            assert "_clozn_sources" not in messages[0]
            return [
                {"id": 7, "piece": "is", "logprob": -0.2 - penalty},
                {"id": 8, "piece": " eligible", "logprob": -0.3 - penalty},
            ]

    study = measure_removal_effect(run, ScoreSub(), removed_source_ids=[span_id])
    stored = {**run, "context_dependence_study": study}
    document = build_context_dependence_query(stored, output_start=2, output_end=11)

    assert document["selection"]["recorded_token_range"] == [1, 2]
    assert document["measured_removal_effects"][0]["delta_nats"] == 1.0
    assert document["measured_removal_effects"][0]["sources"][0]["source_label"] == "Policy §4"


def test_projection_keeps_matched_length_controls_out_of_delete_effects():
    from tests.test_context_dependence_measurement import FakeScoreSub, _run as measured_run

    run = measured_run()
    source_a = run["context_receipt"]["assembled"][0]["segment_id"]
    study = ContextDependenceStudy(run, FakeScoreSub())
    delete = study.measure_removal_effect([source_a])
    control = study.measure_neutralization_control([source_a])
    stored = {**run, "context_dependence_study": study.document()}

    document = build_context_dependence_query(stored, output_start=0, output_end=3)

    assert [item["experiment_id"] for item in document["measured_removal_effects"]] == [delete["experiment_id"]]
    assert len(document["matched_length_neutralization_controls"]) == 1
    projected = document["matched_length_neutralization_controls"][0]
    assert projected["control_id"] == control["control_id"]
    assert projected["intervention_operator"] == "neutralize_source"
    assert projected["provenance"] == "measured_matched_length_neutralization_control"
    assert projected["neutralized_source_ids"] == [source_a]
    assert projected["delta_nats"] == sum(control["per_token_delta_nats"][:1])
    assert projected["full_continuation_delta_nats"] == control["delta_nats"]
    assert projected["neutralization"]["recipe"] == "clozn.matched_length_neutral_filler.v1"
    schemas.validate(document, "clozn.context-dependence-query.v1")


@pytest.mark.parametrize("start,end,code", [
    (4, 4, "invalid_output_range"),
    (5, 6, "invalid_output_range"),
])
def test_invalid_or_non_token_boundary_selections_fail_closed(start, end, code):
    with pytest.raises(ContextDependenceProjectionError) as exc:
        build_context_dependence_query(_run(), output_start=start, output_end=end)
    assert exc.value.status == 400
    assert exc.value.code == code


def test_selection_that_splits_a_persisted_token_is_rejected():
    study = _study()
    study["baseline"]["tokens"] = [
        {"index": 0, "token_id": 101, "piece": "Hi", "unicode_range": [0, 2], "logprob": -0.5},
        {"index": 1, "token_id": 103, "piece": " ", "unicode_range": [2, 3], "logprob": -0.4},
        {"index": 2, "token_id": 104, "piece": "🌍", "unicode_range": [3, 4], "logprob": -0.5},
        {"index": 3, "token_id": 105, "piece": "!", "unicode_range": [4, 5], "logprob": -0.6},
    ]
    study["continuation"]["token_ids"] = [101, 103, 104, 105]
    for experiment in study["experiments"]:
        sign = 1 if experiment["delta_nats"] > 0 else -1
        experiment["per_token_delta_nats"] = [0.3 * sign, 0.3 * sign, 0.4 * sign, 0.5 * sign]
        experiment["token_indices"] = [0, 1, 2, 3]

    with pytest.raises(ContextDependenceProjectionError) as exc:
        build_context_dependence_query(_run(context_dependence_study=study), output_start=1, output_end=2)
    assert exc.value.status == 400
    assert exc.value.code == "invalid_output_range"


def test_recomputed_recorded_response_fidelity_is_explicit_in_selection():
    study = _study()
    study["continuation"].update({
        "kind": "recorded_response_text_retokenized",
        "fidelity": "recomputed_from_recorded_response_text",
        "token_ids_exact": False,
        "retokenized": True,
    })
    del study["continuation"]["token_ids"]
    for token in study["baseline"]["tokens"]:
        del token["token_id"]

    document = build_context_dependence_query(
        _run(context_dependence_study=study), output_start=3, output_end=4,
    )
    assert document["selection"]["registration"] == "recomputed"
    assert document["selection"]["fidelity"] == "recomputed_from_recorded_response_text"


def test_stale_or_partial_vector_evidence_is_rejected_not_reinterpreted():
    stale = _run(response="A changed answer")
    with pytest.raises(ContextDependenceProjectionError) as exc:
        build_context_dependence_query(stale, output_start=0, output_end=1)
    assert exc.value.status == 409

    partial = _run(context_dependence_study=_study(vector_a=[0.4]))
    with pytest.raises(ContextDependenceProjectionError) as exc:
        build_context_dependence_query(partial, output_start=3, output_end=4)
    assert exc.value.status == 409
    assert exc.value.code == "context_dependence_projection_stale"


def test_empty_whole_message_source_range_remains_projectable():
    study = _study()
    source = study["source_identity"]["sources"][0]
    source["unicode_range"] = [0, 0]
    source["byte_range"] = [0, 0]
    removed = study["experiments"][0]["exact_removed_ranges"][0]
    removed["unicode_range"] = [0, 0]
    removed["byte_range"] = [0, 0]

    document = build_context_dependence_query(
        _run(context_dependence_study=study), output_start=3, output_end=4,
    )

    assert document["measured_removal_effects"][0]["delta_nats"] == 0.4


def test_route_query_is_read_only_and_legacy_study_remains_ordinary_get_readable(monkeypatch):
    run = _run()
    before = copy.deepcopy(run)
    monkeypatch.setattr(runlog, "get_run", lambda run_id: run if run_id == run["id"] else None)
    monkeypatch.setattr(
        "clozn.server.model_routing.select_control_model_for_run",
        lambda *_args, **_kwargs: pytest.fail("selection projection must not choose a model"),
    )

    h = Handler("/runs/run_projection/context-dependence/query?output_start=3&output_end=4")
    assert route.try_get(h, "/runs/run_projection/context-dependence/query") is True
    assert h.status == 200
    assert h.body["measured_removal_effects"][0]["delta_nats"] == 0.4
    assert run == before

    run["context_dependence_study"] = _study(schema_version="clozn.context-dependence-study.v1")
    legacy_get = Handler("/runs/run_projection/context-dependence")
    assert route.try_get(legacy_get, "/runs/run_projection/context-dependence") is True
    assert legacy_get.status == 200
    legacy_query = Handler("/runs/run_projection/context-dependence/query?output_start=0&output_end=1")
    assert route.try_get(legacy_query, "/runs/run_projection/context-dependence/query") is True
    assert legacy_query.status == 409
    assert legacy_query.body["code"] == "context_dependence_projection_legacy_artifact"


def test_route_query_uses_clear_http_errors_for_missing_parameters_and_bad_artifact(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda run_id: run if run_id == run["id"] else None)

    missing = Handler("/runs/run_projection/context-dependence/query?output_start=0")
    route.try_get(missing, "/runs/run_projection/context-dependence/query")
    assert missing.status == 400 and missing.body["code"] == "invalid_output_range"

    run["context_dependence_study"]["experiments"][0]["per_token_delta_nats"] = [0.4]
    malformed = Handler("/runs/run_projection/context-dependence/query?output_start=3&output_end=4")
    route.try_get(malformed, "/runs/run_projection/context-dependence/query")
    assert malformed.status == 409
    assert malformed.body["code"] == "context_dependence_projection_stale"
