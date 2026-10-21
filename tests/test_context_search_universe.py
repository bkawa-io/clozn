"""Deterministic flat Context Search Universe planning."""
from __future__ import annotations

from copy import deepcopy

from clozn import schemas
from clozn.runs.context_receipt import build_context_receipt
from clozn.runs.context_search_universe import plan_context_search_universe
from clozn.runs.context_units import build_context_unit_manifest


def _run(content: str, *, run_id: str = "run_universe") -> dict:
    messages = [{"role": "user", "content": content}, {"role": "user", "content": "current"}]
    receipt = build_context_receipt(messages=messages, run_id=run_id, privacy="full")
    run = {"id": run_id, "messages": deepcopy(messages), "context_receipt": receipt}
    run["context_units"] = build_context_unit_manifest(run)
    return run


def test_derived_universe_includes_every_nonempty_message():
    content = "\n\n".join(f"# H{i}\n" + (str(i) * 50) for i in range(12))
    run = _run(content)
    original = list(run["context_units"]["default_source_ids"])

    artifact = plan_context_search_universe(run, run["context_units"], max_units=16)

    assert artifact["status"] == "planned"
    assert artifact["source_ids"] == original
    assert artifact["coverage"]["removable_message_indices"] == [0, 1]
    assert artifact["policy"] == {
        "kind": "flat_markdown_partition",
        "version": "flat_markdown_partition.v1",
        "max_units": 16,
    }
    schemas.validate(artifact)


def test_lower_runtime_bound_returns_typed_failure_without_repartitioning():
    content = "\n\n".join(f"# H{i}\n" + (str(i) * 80) for i in range(8))
    run = _run(content, run_id="run_bound")
    source_ids = list(run["context_units"]["default_source_ids"])
    assert len(source_ids) > 1

    artifact = plan_context_search_universe(run, run["context_units"], max_units=1)

    assert artifact["status"] == "bound_exceeded"
    assert artifact["condition"]["code"] == "derived_units_exceed_max_units"
    assert artifact["source_ids"] == source_ids
    schemas.validate(artifact)


def test_manifest_change_changes_universe_identity():
    content = "# A\n" + ("a" * 80) + "\n\n# B\n" + ("b" * 80)
    run = _run(content, run_id="run_identity_universe")
    first = plan_context_search_universe(run, run["context_units"])
    altered = deepcopy(run["context_units"])
    altered["units"][0]["role"] = "assistant"
    second = plan_context_search_universe(run, altered)
    assert first["universe_id"] != second["universe_id"]
