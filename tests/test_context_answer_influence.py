"""Model-free tests for the context <-> answer influence evidence core."""
from __future__ import annotations

from copy import deepcopy

from clozn.receipts.context_answer_influence import (
    ERROR_CODES,
    EVIDENCE_STATE_CAUSALLY_SUPPORTED,
    EVIDENCE_STATE_OBSERVED,
    MODE,
    SCHEMA,
    _PERSISTENT_CAVEAT,
    context_answer_influence,
    segment_context,
)


TOKENS = [
    {"id": 101, "piece": "Blue", "logprob": -0.1},
    {"id": 102, "piece": " grass", "logprob": -0.2},
]


class FakeScoreSub:
    def __init__(self, score_fn=None, fail_after=None):
        self.calls = []
        self.score_fn = score_fn or (lambda _messages, _block: [-0.1, -0.2])
        self.fail_after = fail_after

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        self.calls.append({
            "messages": deepcopy(messages),
            "continuation_ids": deepcopy(continuation_ids),
            "continuation": continuation,
            "block": block,
            "steer_strengths": deepcopy(steer_strengths),
        })
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise RuntimeError("controlled arm failure")
        logprobs = self.score_fn(messages, block)
        return [
            {**token, "logprob": logprob}
            for token, logprob in zip(TOKENS, logprobs)
        ]


class NoScoreSub:
    pass


class StepClock:
    def __init__(self):
        self.value = 10.0

    def __call__(self):
        value = self.value
        self.value += 0.001
        return value


def test_segment_context_carries_canonical_receipt_source_identity():
    segmented = segment_context(
        [{"role": "user", "content": "Document sentence."}],
        receipt_sources={
            0: {
                "segment_id": "seg_0123456789abcdef",
                "client_source_id": "doc-7",
                "source_label": "Handbook",
            }
        },
    )
    source = segmented["sources"][0]
    span = segmented["spans"][0]
    assert source["segment_id"] == "seg_0123456789abcdef"
    assert source["external_source_id"] == "doc-7"
    assert source["name"] == "Handbook"
    assert span["segment_id"] == source["segment_id"]
    assert span["client_source_id"] == "doc-7"
    assert span["byte_start"] == 0
    assert span["byte_end"] == len("Document sentence.".encode("utf-8"))


def _run():
    return {
        "id": "run-map-1",
        "model": "qwen-test",
        "substrate": "FakeScoreSub",
        "messages": [{"role": "user", "content": "RAW_ONLY"}],
        "assembled_messages": [
            {"role": "system", "content": "Policy."},
            {"role": "user", "name": "retrieval", "source_id": "doc-7",
             "content": "The sky is blue. Grass is green."},
        ],
        "response": "Blue grass",
        "trace": {"token_ids": [101, 102]},
        "behavior": {"active_dials": {"careful": 0.5}},
        "identity": {"model_sha256": "abc", "template_fingerprint": "tpl"},
        "final_prompt": "<rendered exact prompt>",
    }


def _influence_scores(messages, _block):
    text = "\n".join(message.get("content", "") for message in messages)
    if "Policy." not in text:
        return [-0.2, -0.25]
    if "The sky is blue." not in text:
        return [-1.1, -0.1]
    if "Grass is green." not in text:
        return [-0.1, -1.2]
    return [-0.1, -0.2]


def test_source_aware_segmentation_is_bounded_deterministic_and_exact():
    messages = [{"role": "system", "content": "Policy."}]
    messages.extend(
        {"role": "user", "content": f"Message {index}."}
        for index in range(1, 10)
    )
    first = segment_context(messages, max_spans=8)
    second = segment_context(messages, max_spans=8)

    assert first == second
    assert len(first["spans"]) == 8
    assert first["selection"]["selected_source_ids"] == [
        "p.m000", "p.m003", "p.m004", "p.m005", "p.m006", "p.m007", "p.m008", "p.m009",
    ]
    assert first["selection"]["omitted_source_ids"] == ["p.m001", "p.m002"]
    for span in first["spans"]:
        message = messages[span["message_index"]]
        assert span["role"] == message["role"]
        assert message["content"][span["start"]:span["end"]] == span["text"]
        assert span["id"].startswith(span["parent_id"] + ".c")

    one = segment_context(messages, max_spans=1)
    assert [span["parent_id"] for span in one["spans"]] == ["p.m000"]



def test_below_floor_is_explicitly_no_clear_source():
    run = _run()
    run["assembled_messages"] = [{"role": "user", "content": "A weak source."}]
    sub = FakeScoreSub(lambda messages, block: (
        [-0.1, -0.2] if "A weak source." in str(messages) else [-0.11, -0.21]
    ))
    out = context_answer_influence(run, sub, clock=StepClock())

    assert out["matrix"] == [[0.01, 0.01]]
    assert all(link["clears_floor"] is False for link in out["links"])
    # Below-floor is `observed`, never a silently-dropped link and never "irrelevant" -- the spec's own
    # rule ("Never convert absence of measured effect into proof the source was irrelevant") applied.
    assert all(link["evidence_state"] == EVIDENCE_STATE_OBSERVED for link in out["links"])
    assert out["summary"]["has_any_clear_source"] is False
    assert out["summary"]["no_clear_source"] is True
    assert out["summary"]["answer_span_ids_without_clear_source"] == ["a.t0000", "a.t0001"]
    assert out["thresholds"]["calibration"] == "fixed_default_not_model_calibrated"


def test_run_is_not_mutated_and_evidence_is_stable():
    run = _run()
    before = deepcopy(run)
    first = context_answer_influence(run, FakeScoreSub(_influence_scores), clock=StepClock())
    second = context_answer_influence(run, FakeScoreSub(_influence_scores), clock=StepClock())

    assert run == before
    assert first == second
    assert len(first["artifact_sha256"]) == 64
    assert first["answer"] == {
        "recorded_text": "Blue grass",
        "scored_text": "Blue grass",
        "scored_text_matches_recorded": True,
        "offset_basis": "scored_text",
    }
    assert first["answer_spans"][1]["start"] == 4


class _OffsetClock:
    """A clock that never agrees with itself between two calls, timing-wise -- proves artifact_sha256
    is truly independent of wall-clock jitter, unlike StepClock() instances re-created identically."""
    def __init__(self, start: float, step: float):
        self.value = start
        self.step = step

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


def test_artifact_hash_is_stable_across_different_wall_clock_timings():
    """redundancy_check nests its own score_ms (machine-dependent) -- it must be stripped from the
    digest input exactly like the top-level timing block, or artifact_sha256 would silently vary run to
    run on the same evidence purely from timing jitter."""
    run = _run()
    first = context_answer_influence(run, FakeScoreSub(_influence_scores), clock=_OffsetClock(1.0, 0.01))
    second = context_answer_influence(run, FakeScoreSub(_influence_scores), clock=_OffsetClock(500.0, 0.37))

    assert first["timing"] != second["timing"]
    assert first["redundancy_check"]["score_ms"] != second["redundancy_check"]["score_ms"]
    assert first["artifact_sha256"] == second["artifact_sha256"]


def test_response_text_fallback_reuses_the_exact_recorded_continuation():
    run = _run()
    run["trace"] = {}
    sub = FakeScoreSub(_influence_scores)
    out = context_answer_influence(run, sub, clock=StepClock())

    assert out["status"] == "ok"
    assert out["continuation"] == {
        "text_exact": True,
        "token_ids_exact": False,
        "retokenized": True,
        "kind": "recorded_response_text",
    }
    assert all(call["continuation_ids"] is None for call in sub.calls)
    assert all(call["continuation"] == "Blue grass" for call in sub.calls)


def test_missing_score_surface_is_an_honest_unavailable_shape():
    out = context_answer_influence(_run(), NoScoreSub(), clock=StepClock())
    assert out["schema"] == SCHEMA
    assert out["status"] == "unavailable"
    assert out["available"] is False
    assert out["error"]["code"] == "scoring_unavailable"
    assert out["error"]["code"] in ERROR_CODES
    assert out["method"]["generation_used"] is False
    assert "matrix" not in out


def test_failed_intervention_does_not_masquerade_as_a_complete_map():
    out = context_answer_influence(_run(), FakeScoreSub(_influence_scores, fail_after=1),
                                   clock=StepClock())
    assert out["status"] == "error"
    assert out["available"] is False
    assert out["error"]["code"] == "intervention_score_failed"
    assert out["error"]["code"] in ERROR_CODES
    assert out["failed_context_span_id"] == "p.m000.c000"
    assert out["completed_context_span_ids"] == []
    assert "matrix" not in out and "links" not in out


def test_method_carries_mode_and_persistent_caveat_in_every_shape():
    """`method` (and therefore `mode`/`caveat`) comes from `_base_result`, so it is present whether the
    call succeeds, degrades to `unavailable`, or fails outright -- a consumer reading ANY shape this
    module returns is one key away from the sentence bounding every number in it (spec's required
    persistent caveat, verbatim)."""
    ok = context_answer_influence(_run(), FakeScoreSub(_influence_scores), clock=StepClock())
    unavailable = context_answer_influence(_run(), NoScoreSub(), clock=StepClock())
    for out in (ok, unavailable):
        assert out["method"]["mode"] == MODE == "forced_score_intervention"
        assert out["method"]["caveat"] == _PERSISTENT_CAVEAT
        assert "does not prove the document is correct" in out["method"]["caveat"]


def test_coarse_to_fine_refinement_splits_only_the_strongest_clearing_span():
    run = _run()
    run["assembled_messages"] = [
        {"role": "user", "content": "Cats are fuzzy. Dogs are loyal. Birds can fly."},
    ]

    def scores(messages, _block):
        text = "\n".join(m.get("content", "") for m in messages)
        if "Cats are fuzzy." not in text:
            return [-2.0, -0.2]
        if "Dogs are loyal." not in text:
            return [-0.11, -0.21]
        if "Birds can fly." not in text:
            return [-0.1, -0.2]
        return [-0.1, -0.2]

    out = context_answer_influence(run, FakeScoreSub(scores), max_context_spans=1,
                                   check_redundant_pair=False, clock=StepClock())
    assert out["status"] == "ok"
    coarse = next(s for s in out["prompt_spans"] if s["level"] == "coarse")
    fine = [s for s in out["prompt_spans"] if s["level"] == "fine"]
    assert coarse["child_unit_count"] == 3
    assert len(fine) == 3
    assert all(s["parent_id"] == coarse["id"] for s in fine)
    assert len(out["prompt_spans"]) == 4
    assert out["matrix_shape"] == [4, 2]

    cats = next(s for s in fine if "Cats" in s["text"])
    dogs = next(s for s in fine if "Dogs" in s["text"])
    birds = next(s for s in fine if "Birds" in s["text"])
    assert out["matrix"][out["prompt_spans"].index(cats)] == [1.9, 0.0]
    assert out["matrix"][out["prompt_spans"].index(dogs)] == [0.01, 0.01]
    assert out["matrix"][out["prompt_spans"].index(birds)] == [0.0, 0.0]

    refinement = out["selection"]["refinement"]
    assert refinement["refined_context_span_ids"] == [coarse["id"]]
    assert refinement["fine_span_count"] == 3

    # The refined coarse parent stays in the full matrix/links (never discarded) but the top-ranked
    # display list for the answer prefers the strictly more specific fine child that actually carried
    # the effect, rather than double-counting the same text at two granularities.
    top = out["summary"]["answer_to_context"][0]
    assert top["top_context_span_ids"][0] == cats["id"]
    assert coarse["id"] not in top["top_context_span_ids"]
    assert coarse["id"] not in out["summary"]["answer_span_ids_without_clear_source"]

    # One baseline, one coarse arm, three fine arms -- refinement is bounded and reused the baseline.
    assert len(out["prompt_spans"]) == 4
    assert out["timing"]["score_calls"] == 5


def test_refinement_is_skipped_when_the_strongest_span_is_already_atomic():
    run = _run()
    sub = FakeScoreSub(_influence_scores)
    out = context_answer_influence(run, sub, check_redundant_pair=False, clock=StepClock())
    # Every coarse span in the default fixture is already a single sentence -- there is nothing finer
    # to split, so no "level": "fine" spans should appear and the call budget stays coarse-only.
    assert all(span["level"] == "coarse" for span in out["prompt_spans"])
    assert out["selection"]["refinement"]["refined_context_span_ids"] == []
    assert out["selection"]["refinement"]["fine_span_count"] == 0
    assert len(sub.calls) == 1 + len(out["prompt_spans"])


def test_redundant_pair_check_measures_joint_replacement_of_the_two_strongest_spans():
    run = _run()
    sub = FakeScoreSub(_influence_scores)
    out = context_answer_influence(run, sub, clock=StepClock())

    redundancy = out["redundancy_check"]
    assert redundancy["performed"] is True
    sky_id = next(span["id"] for span in out["prompt_spans"] if "sky" in span["text"])
    grass_id = next(span["id"] for span in out["prompt_spans"] if "grass" in span["text"].lower())
    assert sorted(redundancy["context_span_ids"]) == sorted([sky_id, grass_id])
    assert len(redundancy["per_answer_token"]) == 2
    first, second = redundancy["per_answer_token"]
    assert (first["individual_sum_nats"], first["joint_delta_nats"], first["interaction_nats"]) == (
        1.0, 1.0, 0.0,
    )
    assert (second["individual_sum_nats"], second["joint_delta_nats"], second["interaction_nats"]) == (
        0.9, -0.1, -1.0,
    )
    assert "percentage of total explanation" in redundancy["claim_limit"]
    assert len(sub.calls) == 1 + len(out["prompt_spans"]) + 1
    assert out["timing"]["redundancy_check_ms"] >= 0.0


def test_redundant_pair_check_is_honest_when_fewer_than_two_spans_clear():
    run = _run()
    run["assembled_messages"] = [{"role": "user", "content": "A weak source."}]
    sub = FakeScoreSub(lambda messages, block: (
        [-0.1, -0.2] if "A weak source." in str(messages) else [-0.11, -0.21]
    ))
    out = context_answer_influence(run, sub, clock=StepClock())
    assert out["redundancy_check"] == {
        "performed": False,
        "reason": "fewer than two context spans clear the measurement floor",
    }
    assert len(sub.calls) == 1 + len(out["prompt_spans"])


def test_redundant_pair_check_can_be_disabled():
    run = _run()
    sub = FakeScoreSub(_influence_scores)
    out = context_answer_influence(run, sub, check_redundant_pair=False, clock=StepClock())
    assert out["redundancy_check"] == {
        "performed": False,
        "reason": "redundant-pair check disabled for this call",
    }
    assert len(sub.calls) == 1 + len(out["prompt_spans"])


def test_legacy_prompt_block_is_a_real_replaceable_source():
    run = _run()
    run.pop("assembled_messages")
    run["messages"] = [{"role": "user", "content": "Question."}]
    run["memory"] = {"prompt_block": "Remember the sky is blue."}

    def scores(_messages, block):
        return [-0.1, -0.2] if block and "sky is blue" in block else [-1.0, -0.2]

    out = context_answer_influence(run, FakeScoreSub(scores), clock=StepClock())
    assert out["status"] == "ok"
    assert out["identity"]["prompt_view"] == "messages_plus_prompt_block"
    block_source = next(source for source in out["prompt_sources"] if source["id"] == "p.b000")
    assert block_source["source_kind"] == "prompt_block"
    block_span = next(span for span in out["prompt_spans"] if span["parent_id"] == "p.b000")
    row = out["matrix"][out["prompt_spans"].index(block_span)]
    assert row == [0.9, 0.0]
