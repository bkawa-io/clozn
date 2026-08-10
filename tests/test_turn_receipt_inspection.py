"""Turn Receipt -> Selection Reference integration stays a read-only decorator."""
from __future__ import annotations

from copy import deepcopy

from clozn import schemas
from clozn.runs.receipt_inspection import attach_inspection_targets, build_inspect_target
from clozn.runs.turn_receipt import build_turn_receipt, to_markdown
from tests.test_selection_inspection import _influence, _influence_ids, _run as selection_run
from tests.test_turn_receipt import _run as receipt_run


def test_visible_relationship_gets_canonical_inspect_target_without_mutating_inputs():
    run = selection_run(influence_map=_influence(selection_run()))
    source_id, answer_id = _influence_ids(run)
    receipt = {
        "schema_version": "clozn.turn-receipt.v1",
        "run_id": run["id"],
        "what_mattered": {"notable_sources": [{
            "source_span_id": source_id,
            "answer_span_id": answer_id,
            "effect": "supports",
            "evidence_state": "observed",
            "abs_delta_nats": 0.2,
        }]},
        "signals": [],
        "comparison": None,
    }
    before_run, before_receipt = deepcopy(run), deepcopy(receipt)
    result = attach_inspection_targets(run, receipt)

    inspect = result["what_mattered"]["notable_sources"][0]["inspect"]
    assert inspect["selection"] == {
        "kind": "context_span", "source_span_id": source_id, "answer_span_id": answer_id,
    }
    assert inspect["selection_ref"].startswith("sel1.")
    assert inspect["deep_link"]["run_id"] == run["id"]
    assert run == before_run
    assert receipt == before_receipt


def test_source_only_legacy_item_gets_source_selection_when_span_resolves():
    run = selection_run()
    # The shared selection reference contract can resolve ordinary legacy message spans without any
    # influence artifact.  This is the compatibility path for old source-level Receipt items.
    from clozn.runs.text_span_addresses import build_persisted_text_span_addresses
    source_id = next(item["address_id"] for item in build_persisted_text_span_addresses(run)["addresses"]
                     if item.get("native_ref", {}).get("client_source_id") == "doc-1")
    receipt = {"what_mattered": {"notable_sources": [{"source_span_id": source_id}]}, "signals": []}
    result = attach_inspection_targets(run, receipt)
    assert result["what_mattered"]["notable_sources"][0]["inspect"]["selection"] == {
        "kind": "context_span", "source_span_id": source_id,
    }


def test_context_tension_targets_common_answer_span_not_a_source():
    run = selection_run(influence_map=_influence(selection_run()))
    _source_id, answer_id = _influence_ids(run)
    run["context_tension"] = {
        "schema_version": "clozn.context-tension.v1",
        "measurement": {"state": "available"},
        "tensions": [{"tension_id": "tension_aaaaaaaaaaaaaaaaaaaaaaaa",
                       "answer_span_id": answer_id}],
    }
    receipt = {
        "context_tension": {"measurement_state": "available", "tension_pairs": 1},
        "signals": [{"code": "context_tension_detected", "level": "attention",
                     "summary": "Competing measured context effects were detected.",
                     "evidence": "context_tension"}],
    }
    result = attach_inspection_targets(run, receipt)
    inspect = result["signals"][0]["inspect"]
    assert inspect["selection"]["kind"] == "answer_span"
    assert inspect["selection"]["start"] == 0
    assert inspect["selection"]["end"] == len(run["response"])
    assert "source_span_id" not in inspect["selection"]


def test_comparison_and_specific_breakpoint_targets_use_response_token_coordinates():
    run = selection_run()
    receipt = {
        "comparison": {"state": "available", "parent_run_id": "parent",
                       "first_divergence": {"index": 1, "kind": "token_mismatch"}},
        "signals": [{"code": "suggested_breakpoint", "level": "attention",
                     "summary": "A useful test location.", "evidence": "breakpoint",
                     "position": 0}],
    }
    result = attach_inspection_targets(run, receipt)
    assert result["comparison"]["inspect"]["selection"] == {"kind": "response_token", "position": 1}
    assert result["signals"][0]["inspect"]["selection"] == {"kind": "response_token", "position": 0}
    assert result["comparison"]["inspect"]["selection_ref"] != result["signals"][0]["inspect"]["selection_ref"]


def test_invalid_or_unavailable_target_does_not_remove_finding_or_fail_receipt():
    run = selection_run()
    receipt = {
        "what_mattered": {"notable_sources": [{"source_span_id": "span_" + "f" * 24}]},
        "signals": [{"code": "first_divergence_available", "level": "info",
                     "summary": "This branch first diverged at token 0.",
                     "evidence": "first_divergence_view"}],
        "comparison": {"state": "available", "parent_run_id": "parent",
                       "first_divergence": {"index": 0, "kind": "token_mismatch"}},
    }
    result = attach_inspection_targets(run, receipt)
    assert result["what_mattered"]["notable_sources"]
    assert result["signals"]
    assert "inspect" not in result["what_mattered"]["notable_sources"][0]
    assert "inspect" in result["signals"][0]


def test_real_receipt_schema_and_markdown_remain_compact():
    run = receipt_run()
    receipt = build_turn_receipt(run)
    schemas.validate(receipt, "clozn.turn-receipt.v1")
    assert "sel1." not in to_markdown(receipt)
    assert run["response"] not in repr(receipt)


def test_same_target_is_deterministic_and_target_helper_is_optional():
    run = selection_run()
    selection = {"kind": "response_token", "position": 0}
    first = build_inspect_target(run, selection)
    second = build_inspect_target(run, selection)
    assert first == second
    assert first["selection"] == selection
    assert build_inspect_target(run, {"kind": "response_token", "position": 999}) is None
