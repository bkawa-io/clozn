"""Turn Receipt v1: pure composition, explicit missing evidence, privacy, and Markdown tests."""
from __future__ import annotations

import copy

from clozn import schemas
from clozn.runs.turn_receipt import build_turn_receipt, to_markdown


def _context(*, privacy="metadata_only", omitted=False, prompt=2481, window=8192, generated=184):
    delivered = [
        {"segment_id": "seg_aaaaaaaaaaaaaaaa", "source_label": "system", "included": True},
        {"segment_id": "seg_bbbbbbbbbbbbbbbb", "source_label": "Policy Handbook",
         "client_source_id": "policy-2026", "included": not omitted},
    ]
    assembled = [delivered[0]] if omitted else list(delivered)
    return {
        "schema_version": "clozn.context-receipt.v1",
        "run_id": "run_x",
        "privacy": privacy,
        "limits": {"prompt_tokens": prompt, "context_window_tokens": window,
                   "generated_tokens": generated},
        "delivered": delivered if privacy != "off" else None,
        "assembled": assembled if privacy != "off" else None,
        "omissions": ([{"segment_id": delivered[1]["segment_id"], "reason": "context_budget"}]
                       if omitted and privacy != "off" else []),
        "transformations": ([{"reason": "template_transformed", "segment_ids": [s["segment_id"] for s in delivered]}]
                             if privacy != "off" else []),
        "termination": {"reason": "eos", "generated_tokens": generated},
    }


def _run(**over):
    run = {
        "id": "run_x",
        "model": "llama-3.1-8b",
        "substrate": "gguf",
        "response": "A recorded answer.",
        "finish_reason": "stop",
        "trace": {"tokens": ["A", " recorded", " answer."]},
        "meta": {"quant": "Q4_K_M", "prefill_duration_ms": 412,
                 "generation_duration_ms": 3812, "prompt_tokens_per_second": 6021,
                 "generation_tokens_per_second": 48.3},
        "timing": {"duration_ms": 4300},
        "identity": {"model_sha256": "a" * 64, "template_fingerprint": "b" * 16,
                     "engine_build": "engine-test", "clozn_version": "1.0"},
        "context_receipt": _context(),
    }
    run.update(over)
    return run


def _utilization(*, sources, prompt_sources=None, measured_sources=None, clear=0, below=0, not_measured=0):
    return {
        "schema_version": "clozn.context-utilization.v1",
        "run_id": "run_x",
        "privacy": "metadata_only",
        "measurement": {"state": "available"},
        "sources": sources,
        "summary": {
            "prompt_sources": len(sources) if prompt_sources is None else prompt_sources,
            "measured_sources": (sum(1 for item in sources if item.get("measurement_state") == "measured")
                                  if measured_sources is None else measured_sources),
            "sources_with_clear_measured_effect": clear,
            "sources_below_measured_floor": below,
            "sources_not_measured": not_measured,
        },
    }


def test_healthy_completed_run_is_compact_and_schema_valid():
    receipt = build_turn_receipt(_run())
    schemas.validate(receipt)
    assert receipt["outcome"] == {"state": "completed", "finish_reason": "eos", "generated_tokens": 184}
    assert receipt["model"] == {"name": "llama-3.1-8b", "quant": "Q4_K_M", "substrate": "gguf"}
    assert receipt["context"]["sources"] == {"delivered": 2, "assembled": 2, "omitted": 0}
    assert receipt["context"]["window_occupancy"] == 0.3029
    assert receipt["signals"] == []
    assert receipt["technical"]["model_sha256"] == "a" * 64


def test_truncated_and_errored_outcomes_reuse_normalized_termination():
    truncated = _run(
        finish_reason="length",
        context_receipt={**_context(), "termination": {"reason": "max_tokens", "generated_tokens": 184}},
    )
    errored = _run(
        error="worker failed",
        context_receipt={**_context(), "termination": {"reason": "worker_error", "generated_tokens": 0}},
    )
    assert build_turn_receipt(truncated)["outcome"]["state"] == "truncated"
    assert build_turn_receipt(errored)["outcome"]["state"] == "errored"


def test_context_occupancy_is_literal_window_occupancy_and_pressure_is_factual():
    receipt = build_turn_receipt(_run(context_receipt=_context(prompt=910, window=1000)))
    assert receipt["context"]["window_occupancy"] == 0.91
    pressure = next(item for item in receipt["signals"] if item["code"] == "context_window_pressure")
    assert pressure["summary"] == "Prompt occupies 91% of the context window."
    assert "used" not in pressure["summary"]


def test_omitted_context_is_explicit_and_does_not_claim_low_effect():
    receipt = build_turn_receipt(_run(context_receipt=_context(omitted=True)))
    assert receipt["context"]["sources"]["omitted"] == 1
    assert any(item["code"] == "context_omitted" for item in receipt["signals"])
    assert "what_mattered" in receipt and receipt["what_mattered"]["measurement_state"] == "not_measured"


def test_missing_influence_measurement_is_normal_not_measured_state():
    mattered = build_turn_receipt(_run())["what_mattered"]
    assert mattered == {"measurement_state": "not_measured"}


def test_partial_coverage_clear_effect_below_floor_and_three_source_cap():
    sources = [
        {"source_span_id": "span_" + "a" * 24, "measurement_state": "measured",
         "effect_state": "clear_measured_effect", "supporting_clear_links": 1,
         "suppressing_clear_links": 0, "native": {"source_id": "policy"}},
        {"source_span_id": "span_" + "b" * 24, "measurement_state": "measured",
         "effect_state": "clear_measured_effect", "supporting_clear_links": 0,
         "suppressing_clear_links": 1, "native": {"source_id": "system"}},
        {"source_span_id": "span_" + "c" * 24, "measurement_state": "measured",
         "effect_state": "clear_measured_effect", "supporting_clear_links": 1,
         "suppressing_clear_links": 1, "native": {"source_id": "mixed"}},
        {"source_span_id": "span_" + "d" * 24, "measurement_state": "measured",
         "effect_state": "below_measured_floor", "supporting_clear_links": 0,
         "suppressing_clear_links": 0, "native": {"source_id": "below"}},
        {"source_span_id": "span_" + "e" * 24, "measurement_state": "not_measured",
         "reason": "omitted_by_measurement_selection", "native": {"source_id": "omitted"}},
    ]
    run = _run(
        context_utilization=_utilization(sources=sources, clear=3, below=1, not_measured=1),
        influence_map={"prompt_sources": [
            {"id": "policy", "source_label": "Policy Handbook"},
            {"id": "system", "source_label": "System message"},
        ]},
    )
    mattered = build_turn_receipt(run)["what_mattered"]
    assert mattered["coverage"] == {"prompt_sources": 5, "measured_sources": 4, "not_measured_sources": 1}
    assert mattered["effect_summary"] == {
        "sources_with_clear_measured_effect": 3, "sources_below_measured_floor": 1,
    }
    assert [item["effect"] for item in mattered["notable_sources"]] == ["supporting", "suppressing", "mixed"]
    assert len(mattered["notable_sources"]) == 3
    assert any(item["code"] == "influence_coverage_partial" and item["level"] == "info"
               for item in build_turn_receipt(run)["signals"])


def test_context_tension_available_and_no_tension_are_distinct():
    tense = _run(context_tension={
        "schema_version": "clozn.context-tension.v1", "measurement": {"state": "available"},
        "summary": {"answer_spans_with_tension": 2, "tension_pairs": 3},
    })
    clean = _run(context_tension={
        "schema_version": "clozn.context-tension.v1", "measurement": {"state": "available"},
        "summary": {"answer_spans_with_tension": 0, "tension_pairs": 0},
    })
    tension = build_turn_receipt(tense)
    assert tension["context_tension"] == {
        "measurement_state": "available", "answer_spans_with_tension": 2, "tension_pairs": 3,
    }
    assert any(item["code"] == "context_tension_detected" for item in tension["signals"])
    assert build_turn_receipt(clean)["signals"] == []


def test_partial_timing_omits_unknown_values_and_never_zero_fills():
    performance = build_turn_receipt(_run(meta={"generation_tokens_per_second": 48.3}))['performance']
    assert performance["generation_tokens_per_second"] == 48.3
    assert "prefill_ms" not in performance and "decode_ms" not in performance


def test_parent_first_divergence_is_projected_without_recomputing_a_diff():
    run = _run(
        parent_run_id="run_parent",
        first_divergence_view={
            "schema_version": "clozn.first-divergence-view.v1", "state": "available",
            "divergence": {"index": 37, "kind": "token_mismatch",
                           "a": {"piece": "eligible"}, "b": {"piece": "ineligible"}},
        },
    )
    receipt = build_turn_receipt(run, parent_run={"id": "run_parent", "response": "not copied"})
    assert receipt["comparison"] == {
        "state": "available", "parent_run_id": "run_parent",
        "first_divergence": {"index": 37, "kind": "token_mismatch",
                              "a_piece": "eligible", "b_piece": "ineligible"},
    }
    assert any(item["code"] == "first_divergence_available" for item in receipt["signals"])


def test_ordinary_run_has_no_comparison_and_rewind_is_compact():
    receipt = build_turn_receipt(_run())
    assert receipt["comparison"] is None
    assert set(receipt["rewind"]) == {
        "reconstructed_replay", "exact_rewind", "historically_verified_boundaries",
    }
    assert "live" not in repr(receipt["rewind"])


def test_privacy_hashes_only_keeps_ids_but_does_not_restore_labels():
    run = _run(context_receipt={**_context(privacy="hashes_only"), "delivered": [
        {"segment_id": "seg_aaaaaaaaaaaaaaaa", "content_hash": "a" * 16, "included": True},
    ], "assembled": [{"segment_id": "seg_aaaaaaaaaaaaaaaa", "included": True}]})
    context = build_turn_receipt(run)["context"]
    assert context["provenance_state"] == "available"
    assert context["provenance"] == [{"segment_id": "seg_aaaaaaaaaaaaaaaa", "included": True}]
    assert "Policy Handbook" not in repr(context)


def test_privacy_off_acknowledges_unavailable_provenance():
    context = build_turn_receipt(_run(context_receipt={"schema_version": "clozn.context-receipt.v1",
                                                        "run_id": "run_x", "privacy": "off"}))['context']
    assert context == {"privacy": "off", "provenance_state": "unavailable"}


def test_turn_receipt_has_no_raw_prompt_or_response_and_does_not_mutate_run():
    run = _run(messages=[{"role": "user", "content": "PRIVATE FULL PROMPT TEXT"}],
               response="PRIVATE FULL RESPONSE TEXT")
    before = copy.deepcopy(run)
    receipt = build_turn_receipt(run)
    assert "PRIVATE FULL PROMPT TEXT" not in repr(receipt)
    assert "PRIVATE FULL RESPONSE TEXT" not in repr(receipt)
    assert run == before


def test_builder_is_deterministic():
    run = _run()
    assert build_turn_receipt(copy.deepcopy(run)) == build_turn_receipt(copy.deepcopy(run))


def test_markdown_is_compact_and_uses_product_language():
    receipt = build_turn_receipt(_run())
    markdown = to_markdown(receipt)
    assert markdown.startswith("# Clozn Receipt")
    assert "What mattered" in markdown
    assert "Generation" in markdown
    assert "Context-window occupancy" in markdown
    assert "context used" not in markdown
    assert "PRIVATE" not in markdown
