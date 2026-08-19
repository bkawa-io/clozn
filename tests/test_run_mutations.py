"""Focused transactional run-journal privacy and retention tests."""
from __future__ import annotations

from contextlib import closing
import json
import os
import time

import pytest

from clozn.runs import mutations
from clozn.runs import store


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _record(*, prompt="private prompt", response="private response", trace=None, started=None):
    return store.record(
        source="openai_api", client="private-client-label",
        client_key="client_opaque", session_key="session_opaque", project_key="project_opaque",
        model="model", substrate="engine",
        messages=[{"role": "user", "content": prompt}],
        assembled_messages=[{"role": "system", "content": "private memory"},
                            {"role": "user", "content": prompt}],
        final_prompt="rendered private prompt", response=response,
        reasoning={"blocks": [{"text": "private reasoning"}]},
        trace=trace or {"tokens": ["private", " token"], "confidence": [0.9, 0.8]},
        meta={"max_tokens": 32, "private_extension": "private meta"},
        identity={"model_path": "C:/Users/private-name/models/model.gguf",
                  "model_sha256": "a" * 64},
        output_contract={"raw_model_output": "private tool output",
                         "request": {"tools": [{"description": "private tool"}]}},
        started=started, ended=started,
    )


def _raw(run_id):
    with closing(store._connect()) as db:
        row = db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
    return row, json.loads(row["payload_json"])


def test_redact_replaces_content_with_tombstone_and_cleans_trace(isolated):
    run_id = _record()
    source = store.get_run(run_id)
    source["influence_map"] = {
        "schema": "clozn.context_answer_influence.v1",
        "prompt_spans": [{"text": "private influence context"}],
        "answer_spans": [{"text": "private influence answer"}],
    }
    assert store.replace_run(source)
    _, raw = _raw(run_id)
    digest = raw["trace_ref"]["sha256"]
    path = store._blob_path(digest)
    # The influence map is potentially-large derived evidence and goes through the SAME blob
    # machinery as trace (store._pack), keyed by influence_map_ref -- it must never be lost
    # (regular GC path) AND must be scrubbed immediately on redaction (privacy path), exactly like trace.
    influence_digest = raw["influence_map_ref"]["sha256"]
    influence_path = store._blob_path(influence_digest)
    assert os.path.isfile(influence_path)
    with open(influence_path, encoding="utf-8") as handle:
        assert "private influence" in handle.read()

    result = mutations.redact_run(run_id)

    assert result["trace_cleanup"] == {"status": "deleted", "sha256": digest}
    assert not os.path.exists(path)
    assert result["influence_map_cleanup"] == {"status": "deleted", "sha256": influence_digest}
    assert not os.path.exists(influence_path)
    redacted = store.get_run(run_id)
    assert redacted["redaction"]["schema"] == mutations.REDACTION_SCHEMA
    assert redacted["redaction"]["influence_evidence_removed"] is True
    assert redacted["flags"] == ["redacted"]
    assert redacted["messages"] == [] and redacted["response"] == ""
    assert redacted["reasoning"] == {} and redacted["output_contract"] == {}
    assert redacted["trace"] == {}
    assert "model_path" not in redacted["identity"]
    assert "influence_map" not in redacted
    encoded = json.dumps(redacted)
    for secret in ("private prompt", "private response", "private reasoning", "private tool",
                   "private memory", "private preference", "client_opaque", "session_opaque",
                   "project_opaque", "private meta", "private-name"):
        assert secret not in encoded
    assert "private influence" not in encoded
    row, _ = _raw(run_id)
    assert row["client_key"] is None and row["session_key"] is None
    assert row["prompt_summary"] == "" and row["response_summary"] == ""


def test_redaction_is_idempotent(isolated):
    run_id = _record()
    first = mutations.redact_run(run_id)
    second = mutations.redact_run(run_id)
    assert first["redaction"] == second["redaction"]
    assert second["already_redacted"] is True
    assert second["trace_cleanup"]["status"] == "not_applicable"


def test_shared_trace_survives_until_last_reference_is_removed(isolated):
    trace = {"tokens": ["same"], "confidence": [0.5]}
    first = _record(prompt="one", trace=trace, started=1.0)
    second = _record(prompt="two", trace=trace, started=2.0)
    _, raw = _raw(first)
    path = store._blob_path(raw["trace_ref"]["sha256"])

    assert mutations.redact_run(first)["trace_cleanup"]["status"] == "retained_shared"
    assert os.path.isfile(path)
    assert mutations.delete_run(second)["trace_cleanup"]["status"] == "deleted"
    assert not os.path.exists(path)


def test_delete_truly_removes_row_and_unique_trace(isolated):
    run_id = _record()
    _, raw = _raw(run_id)
    path = store._blob_path(raw["trace_ref"]["sha256"])
    assert mutations.delete_run(run_id)["ok"] is True
    assert store.get_run(run_id) is None
    assert run_id not in {row["id"] for row in store.list_runs()}
    assert not os.path.exists(path)


def test_invalid_and_missing_exact_ids_are_clean(isolated):
    with pytest.raises(mutations.MutationError, match="exact valid run ID"):
        mutations.redact_run("../escape")
    with pytest.raises(mutations.MutationError, match="exact valid run ID"):
        mutations.delete_run("")
    with pytest.raises(mutations.MutationError, match="not found"):
        mutations.redact_run("run_missing")
    with pytest.raises(mutations.MutationError, match="not found"):
        mutations.delete_run("run_missing")


def test_redaction_database_failure_rolls_back_and_keeps_blob(isolated):
    run_id = _record()
    before = store.get_run(run_id)
    _, raw = _raw(run_id)
    path = store._blob_path(raw["trace_ref"]["sha256"])
    with closing(store._connect()) as db, db:
        db.execute("CREATE TRIGGER reject_redaction BEFORE UPDATE ON runs "
                   "BEGIN SELECT RAISE(ABORT, 'blocked'); END")
    with pytest.raises(mutations.MutationError, match="blocked"):
        mutations.redact_run(run_id)
    assert store.get_run(run_id) == before
    assert os.path.isfile(path)


def test_delete_database_failure_rolls_back_and_keeps_blob(isolated):
    run_id = _record()
    _, raw = _raw(run_id)
    path = store._blob_path(raw["trace_ref"]["sha256"])
    with closing(store._connect()) as db, db:
        db.execute("CREATE TRIGGER reject_delete BEFORE DELETE ON runs "
                   "BEGIN SELECT RAISE(ABORT, 'blocked-delete'); END")
    with pytest.raises(mutations.MutationError, match="blocked-delete"):
        mutations.delete_run(run_id)
    assert store.get_run(run_id) is not None
    assert os.path.isfile(path)


def test_prune_dry_run_plans_exact_oldest_rows_without_mutation(isolated):
    ids = [_record(prompt=str(index), started=float(index),
                   trace={"tokens": [str(index)], "confidence": [0.5]})
           for index in range(1, 5)]
    result = mutations.prune_to(2)
    assert result["dry_run"] is True
    assert result["delete_count"] == 2 and result["deleted_count"] == 0
    assert [entry["run_id"] for entry in result["runs"]] == ids[:2]
    assert {run["id"] for run in store.iter_runs()} == set(ids)
    assert len(result["orphaned_trace_digests"]) == 2


def test_prune_live_removes_rows_but_defers_blob_cleanup(isolated):
    ids = [_record(prompt=str(index), started=float(index),
                   trace={"tokens": [str(index)], "confidence": [0.5]})
           for index in range(1, 4)]
    result = mutations.prune_to(1, dry_run=False)
    assert result["deleted_count"] == 2
    assert [entry["run_id"] for entry in result["runs"]] == ids[:2]
    assert [run["id"] for run in store.iter_runs()] == [ids[2]]
    assert result["blob_cleanup"] == "deferred_to_gc"
    for digest in result["orphaned_trace_digests"]:
        assert os.path.isfile(store._blob_path(digest))


def test_prune_live_failure_rolls_back_all_candidate_rows(isolated):
    ids = [_record(prompt=str(index), started=float(index)) for index in range(1, 4)]
    with closing(store._connect()) as db, db:
        # Run IDs are generated from store._SAFE_RID; interpolation here is test-only and
        # avoids SQLite's prohibition on bound parameters inside CREATE TRIGGER.
        db.execute("CREATE TRIGGER reject_partial_retention BEFORE DELETE ON runs "
                   f"WHEN OLD.id = '{ids[1]}' "
                   "BEGIN SELECT RAISE(ABORT, 'blocked-retention'); END")
    with pytest.raises(mutations.MutationError, match="blocked-retention"):
        mutations.prune_to(0, dry_run=False)
    assert {run["id"] for run in store.iter_runs()} == set(ids)


def test_retention_kept_shared_digest_is_not_reported_orphan(isolated):
    shared = {"tokens": ["same"], "confidence": [0.5]}
    old = _record(prompt="old", trace=shared, started=1.0)
    kept = _record(prompt="new", trace=shared, started=2.0)
    result = mutations.prune_to(1)
    assert [entry["run_id"] for entry in result["runs"]] == [old]
    assert result["orphaned_trace_digests"] == []
    assert kept in {run["id"] for run in store.iter_runs()}


@pytest.mark.parametrize("keep", [-1, True, 1.5, "1"])
def test_retention_rejects_invalid_keep(isolated, keep):
    with pytest.raises(mutations.MutationError):
        mutations.prune_to(keep)


# ============================================================================== literal-scoped redaction

def test_redact_literal_scrubs_only_matching_text_and_leaves_trace_alone(isolated):
    run_id = _record(prompt="my api key is sk-super-secret-999", response="ok sk-super-secret-999 noted")
    _, raw = _raw(run_id)
    digest = raw["trace_ref"]["sha256"]
    path = store._blob_path(digest)

    result = mutations.redact_run(run_id, literals=["sk-super-secret-999"])

    assert result["ok"] is True and result["already_redacted"] is False
    assert result["redaction"]["schema"] == mutations.LITERAL_REDACTION_SCHEMA
    assert result["redaction"]["literal_count"] == 1
    assert result["redaction"]["replacement_count"] >= 2
    assert result["trace_cleanup"] == {"status": "not_applicable", "sha256": None}
    assert os.path.isfile(path)  # trace blob is untouched by literal-mode redaction

    redacted = store.get_run(run_id)
    assert redacted["messages"][0]["content"] == "my api key is [REDACTED]"
    assert redacted["response"] == "ok [REDACTED] noted"
    # Everything else about the run survives -- this is not a tombstone.
    assert redacted["client"] == "private-client-label"
    encoded = json.dumps(redacted)
    assert "sk-super-secret-999" not in encoded


def test_redact_literal_scrubs_sql_summary_columns_too(isolated):
    run_id = _record(prompt="topsecretvalue")
    mutations.redact_run(run_id, literals=["topsecretvalue"])
    row, _ = _raw(run_id)
    assert "topsecretvalue" not in (row["prompt_summary"] or "")


def test_redact_literal_longest_match_wins_over_a_contained_shorter_literal(isolated):
    run_id = _record(prompt="value is topsecretvalue right there")
    mutations.redact_run(run_id, literals=["secret", "topsecretvalue"])
    redacted = store.get_run(run_id)
    assert redacted["messages"][0]["content"] == "value is [REDACTED] right there"


def test_redact_literal_on_already_fully_redacted_run_is_a_noop(isolated):
    run_id = _record(prompt="private prompt")
    mutations.redact_run(run_id)
    second = mutations.redact_run(run_id, literals=["private"])
    assert second["already_redacted"] is True
    assert second["trace_cleanup"] == {"status": "not_applicable", "sha256": None}


@pytest.mark.parametrize("literals", [["", "ok"], [123], "not-a-list"])
def test_redact_literal_rejects_invalid_literals(isolated, literals):
    run_id = _record()
    with pytest.raises(mutations.MutationError):
        mutations.redact_run(run_id, literals=literals)


# ================================================================================= delete: children/cascade

def test_delete_refuses_run_with_children_and_lists_them(isolated):
    parent = _record(prompt="parent", started=1.0)
    child = store.record(source="replay", client="sdk", model="model", substrate="engine",
                         messages=[{"role": "user", "content": "child"}], response="ok",
                         parent_run_id=parent, trace={"tokens": ["c"], "confidence": [0.5]},
                         started=2.0, ended=2.0)

    with pytest.raises(mutations.RunHasChildrenError) as excinfo:
        mutations.delete_run(parent)
    assert excinfo.value.run_id == parent
    assert excinfo.value.children == [child]
    assert child in str(excinfo.value)
    # Refused atomically: neither row was touched.
    assert store.get_run(parent) is not None
    assert store.get_run(child) is not None


def test_delete_cascade_removes_full_descendant_subtree(isolated):
    grandparent = _record(prompt="gp", started=1.0)
    parent = store.record(source="replay", client="sdk", model="model", substrate="engine",
                          messages=[{"role": "user", "content": "p"}], response="ok",
                          parent_run_id=grandparent, trace={"tokens": ["p"], "confidence": [0.5]},
                          started=2.0, ended=2.0)
    child = store.record(source="replay", client="sdk", model="model", substrate="engine",
                         messages=[{"role": "user", "content": "c"}], response="ok",
                         parent_run_id=parent, trace={"tokens": ["c"], "confidence": [0.5]},
                         started=3.0, ended=3.0)

    result = mutations.delete_run(grandparent, cascade=True)

    assert result["ok"] is True and result["cascade"] is True
    assert set(result["deleted_run_ids"]) == {grandparent, parent, child}
    assert result["cascade_deleted_count"] == 2
    assert isinstance(result["trace_cleanup"], list) and len(result["trace_cleanup"]) == 3
    assert store.get_run(grandparent) is None
    assert store.get_run(parent) is None
    assert store.get_run(child) is None  # never left dangling on a deleted parent


def test_delete_without_children_keeps_the_original_single_object_shape(isolated):
    run_id = _record()
    result = mutations.delete_run(run_id, cascade=True)  # cascade requested but there is nothing to cascade
    assert result["cascade"] is True
    assert "cascade_deleted_count" not in result
    assert isinstance(result["trace_cleanup"], dict) and "status" in result["trace_cleanup"]


# ======================================================================= retention: lineage + age cutoff

def test_prune_reports_orphaned_parent_refs_without_blocking(isolated):
    parent = _record(prompt="parent", started=1.0)
    child = store.record(source="replay", client="sdk", model="model", substrate="engine",
                         messages=[{"role": "user", "content": "child"}], response="ok",
                         parent_run_id=parent, trace={"tokens": ["c"], "confidence": [0.5]},
                         started=2.0, ended=2.0)

    result = mutations.prune_to(1)  # keeps only the newest (the child); parent would be deleted

    assert result["run_ids"] == [parent]
    assert result["orphaned_parent_refs"] == [child]
    assert store.get_run(parent) is not None  # dry run: nothing actually removed


def _set_recorded_ts(run_id: str, ts: float) -> None:
    with closing(store._connect()) as db, db:
        db.execute("UPDATE runs SET recorded_ts=? WHERE id=?", (ts, run_id))


def test_prune_older_than_dry_run_and_live(isolated):
    now = time.time()
    old = _record(prompt="old", started=now - 40 * 86400)
    new = _record(prompt="new", started=now)
    _set_recorded_ts(old, now - 40 * 86400)
    _set_recorded_ts(new, now)

    preview = mutations.prune_older_than(30, dry_run=True)
    assert preview["dry_run"] is True and preview["days"] == 30
    assert preview["run_ids"] == [old]
    assert store.get_run(old) is not None  # dry run changed nothing

    applied = mutations.prune_older_than(30, dry_run=False)
    assert applied["dry_run"] is False
    assert applied["run_ids"] == [old]
    assert applied["deleted_count"] == 1
    assert store.get_run(old) is None
    assert store.get_run(new) is not None


@pytest.mark.parametrize("days", [0, -1, True, 1.5, "30"])
def test_prune_older_than_rejects_invalid_days(isolated, days):
    with pytest.raises(mutations.MutationError):
        mutations.prune_older_than(days)
