"""Zero-config canonical Context Unit coverage."""
from __future__ import annotations

from copy import deepcopy
import pytest

from clozn import schemas
from clozn.replay.span_bridge import (
    ContextReceiptSourceResolutionError,
    resolve_context_receipt_source_set,
)
from clozn.runs.context_receipt import build_context_receipt
from clozn.runs.context_units import (
    build_context_unit_manifest,
    derive_auto_source_spans,
    max_context_arms,
    protected_message_indices,
)


def _run(messages: list[dict], *, run_id: str = "run_units") -> dict:
    source_messages = deepcopy(messages)
    receipt = build_context_receipt(messages=source_messages, run_id=run_id, privacy="full")
    run = {"id": run_id, "messages": deepcopy(messages), "context_receipt": receipt}
    run["context_units"] = build_context_unit_manifest(run)
    return run


def test_structured_message_is_partitioned_without_a_duplicate_root():
    content = "# Rules\nalpha\n\n# Documents\nDocument 1\none\n\n# Examples\ntwo"
    run = _run([{"role": "user", "content": content}, {"role": "user", "content": "q"}])
    manifest = run["context_units"]

    assert len(manifest["default_source_ids"]) >= 2
    assert all(source_id.startswith("src_") for source_id in manifest["default_source_ids"])
    ranges = [unit["unicode_range"] for unit in manifest["units"]]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(content)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert "seg_" not in manifest["default_source_ids"]


def test_short_unstructured_message_keeps_the_existing_root():
    run = _run([{"role": "user", "content": "A short context note."}, {"role": "user", "content": "q"}])
    manifest = run["context_units"]

    assert len(manifest["default_source_ids"]) == 1
    assert manifest["default_source_ids"][0].startswith("seg_")
    assert manifest["units"][0]["derivation"] == "message_root"
    assert not run["context_receipt"]["delivered"][0].get("sources")


def test_previous_user_and_assistant_are_context_but_current_turn_is_protected():
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
        {"role": "user", "content": "current question"},
    ]
    run = _run(messages)
    manifest = run["context_units"]

    assert manifest["protected_message_indices"] == [0, 3]
    assert {unit["message_index"] for unit in manifest["units"]} == {1, 2}


def test_instructions_and_current_question_are_protected():
    run = _run([
        {"role": "system", "content": "Follow the answer format."},
        {"role": "user", "content": "First context block."},
        {"role": "user", "content": "Second context block."},
        {"role": "user", "content": "What is the answer?"},
    ])
    manifest = run["context_units"]
    assert manifest["protected_message_indices"] == [0, 3]
    assert {unit["message_index"] for unit in manifest["units"]} == {1, 2}
    assert all(unit["message_index"] != 3 for unit in manifest["units"])


def test_trailing_prefill_is_protected_and_no_user_falls_back_to_final_message():
    with_user = _run([
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "current"},
        {"role": "assistant", "content": "prefill"},
    ], run_id="run_prefill")
    assert with_user["context_units"]["protected_message_indices"] == [0, 1, 2]
    assert with_user["context_units"]["units"] == []

    no_user = _run([
        {"role": "system", "content": "rules"},
        {"role": "assistant", "content": "last"},
    ], run_id="run_no_user")
    assert no_user["context_units"]["protected_message_indices"] == [0, 1]
    assert no_user["context_units"]["units"] == []


def test_client_source_annotations_are_ignored_and_replaced_by_derived_units():
    content = "# Alpha\n" + "a" * 80 + "\n\n# Beta\n" + "b" * 80
    messages = [{
        "role": "user",
        "content": content,
        "_clozn_sources": [{
            "source_id": "client-choice",
            "unicode_range": [0, len(content)],
            "provenance_kind": "client_supplied",
        }],
    }, {"role": "user", "content": "q"}]
    run = _run(messages, run_id="run_ignored_client_sources")

    assert len(run["context_units"]["default_source_ids"]) == 2
    assert all(unit["derivation"] == "adaptive_markdown" for unit in run["context_units"]["units"])
    assert all(
        source.get("client_source_id") != "client-choice"
        for source in run["context_receipt"]["delivered"][0].get("sources", [])
    )


def test_arm_cap_is_configurable_and_merges_adjacent_markdown_chunks(monkeypatch):
    content = "\n\n".join(f"# Section {index}\n" + (str(index) * 60) for index in range(12))
    messages = [{"role": "user", "content": content}, {"role": "user", "content": "q"}]

    assert len(derive_auto_source_spans(messages, max_arms=3)) == 3
    monkeypatch.setenv("CLOZN_MAX_CONTEXT_ARMS", "5")
    assert max_context_arms() == 5
    assert len(derive_auto_source_spans(messages)) <= 5
    monkeypatch.setenv("CLOZN_MAX_CONTEXT_ARMS", "not-an-integer")
    assert max_context_arms() == 16


def test_unicode_offsets_and_strict_resolver_bind_to_recorded_bytes():
    content = "# A\ncafé 🙂 東京\n\n# B\nfin"
    run = _run([{"role": "user", "content": content}, {"role": "user", "content": "q"}],
               run_id="run_unicode")
    manifest = run["context_units"]
    for unit in manifest["units"]:
        start, end = unit["unicode_range"]
        byte_start, byte_end = unit["byte_range"]
        selected = content[start:end]
        assert selected
        assert content[:start].encode("utf-8").__len__() == byte_start
        assert content[:end].encode("utf-8").__len__() == byte_end
        assert unit["content_sha256"]
    resolved = resolve_context_receipt_source_set(run, manifest["default_source_ids"])
    assert len(resolved["exact_removed_ranges"]) == len(manifest["units"])


def test_manifest_and_canonical_ids_are_deterministic():
    messages = [{"role": "user", "content": "# A\nα\n\n# B\n🙂"}, {"role": "user", "content": "q"}]
    first = _run(messages, run_id="run_determinism")
    second = _run(messages, run_id="run_determinism")
    assert first["context_receipt"] == second["context_receipt"]
    assert first["context_units"] == second["context_units"]


def test_drifted_message_refuses_an_existing_auto_source():
    run = _run([{"role": "user", "content": "# A\nalpha\n\n# B\nbeta"}, {"role": "user", "content": "q"}],
               run_id="run_drift")
    source_id = run["context_units"]["default_source_ids"][0]
    run["messages"][0]["content"] += " tampered"
    with pytest.raises(ContextReceiptSourceResolutionError):
        resolve_context_receipt_source_set(run, [source_id])


def test_manifest_schema_is_valid():
    run = _run([{"role": "user", "content": "# A\na\n\n# B\nb"}, {"role": "user", "content": "q"}],
               run_id="run_schema")
    schemas.validate(run["context_units"])
