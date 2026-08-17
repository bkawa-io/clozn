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
    protected_message_indices,
)


def _run(messages: list[dict], *, run_id: str = "run_units", explicit: bool = False) -> dict:
    source_messages = deepcopy(messages)
    if explicit:
        source_messages[0]["_clozn_sources"] = [
            {"source_id": "left", "unicode_range": [0, 5], "provenance_kind": "message"},
            {"source_id": "right", "unicode_range": [5, len(source_messages[0]["content"])],
             "provenance_kind": "message"},
        ]
    receipt = build_context_receipt(messages=source_messages, run_id=run_id, privacy="full")
    run = {"id": run_id, "messages": deepcopy(messages), "context_receipt": receipt}
    run["context_units"] = build_context_unit_manifest(run)
    return run


def test_structured_message_is_partitioned_without_a_duplicate_root():
    content = "# Rules\nalpha\n\n# Documents\nDocument 1\none\n\n# Examples\ntwo"
    run = _run([{"role": "system", "content": content}, {"role": "user", "content": "q"}])
    manifest = run["context_units"]

    assert len(manifest["default_source_ids"]) >= 3
    assert all(source_id.startswith("src_") for source_id in manifest["default_source_ids"])
    ranges = [unit["unicode_range"] for unit in manifest["units"]]
    assert ranges[0][0] == 0
    assert ranges[-1][1] == len(content)
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert "seg_" not in manifest["default_source_ids"]


def test_short_unstructured_message_keeps_the_existing_root():
    run = _run([{"role": "system", "content": "You are helpful."}, {"role": "user", "content": "q"}])
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

    assert manifest["protected_message_indices"] == [3]
    assert {unit["message_index"] for unit in manifest["units"]} == {0, 1, 2}
    assert all(unit["message_index"] != 3 for unit in manifest["units"])


def test_trailing_prefill_is_protected_and_no_user_falls_back_to_final_message():
    with_user = _run([
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "current"},
        {"role": "assistant", "content": "prefill"},
    ], run_id="run_prefill")
    assert with_user["context_units"]["protected_message_indices"] == [1, 2]
    assert all(unit["message_index"] == 0 for unit in with_user["context_units"]["units"])

    no_user = _run([
        {"role": "system", "content": "rules"},
        {"role": "assistant", "content": "last"},
    ], run_id="run_no_user")
    assert no_user["context_units"]["protected_message_indices"] == [1]
    assert [unit["message_index"] for unit in no_user["context_units"]["units"]] == [0]


def test_explicit_sources_win_and_nested_parent_child_is_not_selected_together():
    content = "abcdefghij"
    messages = [{"role": "system", "content": content}, {"role": "user", "content": "q"}]
    messages[0]["_clozn_sources"] = [
        {"source_id": "parent", "unicode_range": [0, 10], "provenance_kind": "message"},
        {"source_id": "left", "unicode_range": [0, 5], "parent_source_id": "parent",
         "provenance_kind": "message"},
        {"source_id": "right", "unicode_range": [5, 10], "parent_source_id": "parent",
         "provenance_kind": "message"},
    ]
    receipt = build_context_receipt(messages=messages, run_id="run_explicit", privacy="full")
    run = {"id": "run_explicit", "messages": deepcopy(messages), "context_receipt": receipt}
    manifest = build_context_unit_manifest(run)

    ids = manifest["default_source_ids"]
    assert len(ids) == 2
    assert all(item.startswith("src_") for item in ids)
    assert not any(
        source.get("source_id") in ids and source.get("parent_source_id") in ids
        for source in receipt["delivered"][0]["sources"]
    )
    assert all(unit["derivation"] == "caller_explicit" for unit in manifest["units"])


def test_unicode_offsets_and_strict_resolver_bind_to_recorded_bytes():
    content = "# A\ncafé 🙂 東京\n\n# B\nfin"
    run = _run([{"role": "system", "content": content}, {"role": "user", "content": "q"}],
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
    messages = [{"role": "system", "content": "# A\nα\n\n# B\n🙂"}, {"role": "user", "content": "q"}]
    first = _run(messages, run_id="run_determinism")
    second = _run(messages, run_id="run_determinism")
    assert first["context_receipt"] == second["context_receipt"]
    assert first["context_units"] == second["context_units"]


def test_drifted_message_refuses_an_existing_auto_source():
    run = _run([{"role": "system", "content": "# A\nalpha\n\n# B\nbeta"}, {"role": "user", "content": "q"}],
               run_id="run_drift")
    source_id = run["context_units"]["default_source_ids"][0]
    run["messages"][0]["content"] += " tampered"
    with pytest.raises(ContextReceiptSourceResolutionError):
        resolve_context_receipt_source_set(run, [source_id])


def test_manifest_schema_is_valid():
    run = _run([{"role": "system", "content": "# A\na\n\n# B\nb"}, {"role": "user", "content": "q"}],
               run_id="run_schema")
    schemas.validate(run["context_units"])
