"""Deterministic bounded Context Search Universe planning."""
from __future__ import annotations

from copy import deepcopy

from clozn import schemas
from clozn.replay.span_bridge import resolve_context_receipt_source_set
from clozn.runs.context_receipt import build_context_receipt
from clozn.runs.context_search_universe import plan_context_search_universe
from clozn.runs.context_units import build_context_unit_manifest


def _run(content: str, *, run_id: str = "run_universe", explicit: bool = False) -> dict:
    messages = [{"role": "system", "content": content}, {"role": "user", "content": "current"}]
    if explicit:
        messages[0]["_clozn_sources"] = [
            {"source_id": "left", "unicode_range": [0, len(content) // 2], "provenance_kind": "message"},
            {"source_id": "right", "unicode_range": [len(content) // 2, len(content)], "provenance_kind": "message"},
        ]
    receipt = build_context_receipt(messages=messages, run_id=run_id, privacy="full")
    run = {"id": run_id, "messages": deepcopy(messages), "context_receipt": receipt}
    run["context_units"] = build_context_unit_manifest(run)
    return run


def test_under_cap_is_unchanged_and_protected_turn_is_absent():
    run = _run("\n\n".join(f"# H{i}\nblock {i}" for i in range(20)))
    original = list(run["context_units"]["default_source_ids"])
    artifact = plan_context_search_universe(run, run["context_units"], max_units=50)
    assert artifact["status"] == "planned"
    assert artifact["source_ids"] == original
    assert artifact["coverage"]["protected_message_indices"] == [1]
    schemas.validate(artifact)


def test_over_cap_merges_deterministically_without_losing_exact_coverage():
    run = _run("\n\n".join(f"# H{i}\nblock {i}" for i in range(80)), run_id="run_many_universe")
    first = plan_context_search_universe(run, run["context_units"], max_units=50)
    second = plan_context_search_universe(run, run["context_units"], max_units=50)
    assert first == second
    assert first["source_count"] == 50
    assert first["coverage"]["removable_content_covered"] is True
    resolved = resolve_context_receipt_source_set(run, first["source_ids"])
    assert resolved["canonical_source_ids"] == sorted(first["source_ids"])
    schemas.validate(first)


def test_explicit_nested_structure_is_preserved_and_bound_failure_is_typed():
    content = "abcdefghij"
    messages = [{"role": "system", "content": content}, {"role": "user", "content": "current"}]
    messages[0]["_clozn_sources"] = [
        {"source_id": "parent", "unicode_range": [0, 10], "provenance_kind": "message"},
        {"source_id": "left", "unicode_range": [0, 5], "parent_source_id": "parent", "provenance_kind": "message"},
        {"source_id": "right", "unicode_range": [5, 10], "parent_source_id": "parent", "provenance_kind": "message"},
    ]
    receipt = build_context_receipt(messages=messages, run_id="run_explicit_universe", privacy="full")
    run = {"id": "run_explicit_universe", "messages": deepcopy(messages), "context_receipt": receipt}
    run["context_units"] = build_context_unit_manifest(run)
    artifact = plan_context_search_universe(run, run["context_units"], max_units=1)
    assert artifact["status"] == "bound_exceeded"
    assert artifact["condition"]["code"] == "explicit_units_exceed_max_units"
    assert artifact["source_count"] == 2
    schemas.validate(artifact)


def test_manifest_change_changes_universe_identity():
    run = _run("# A\nalpha\n\n# B\nbeta", run_id="run_identity_universe")
    first = plan_context_search_universe(run, run["context_units"])
    altered = deepcopy(run["context_units"])
    altered["units"][0]["source_label"] = "changed-display-label"
    second = plan_context_search_universe(run, altered)
    assert first["universe_id"] != second["universe_id"]
