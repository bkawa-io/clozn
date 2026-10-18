"""Contract tests for the run store's single current SQLite schema."""
from __future__ import annotations

import sqlite3

import pytest

from clozn.runs import schema


def _tables(db: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        if not str(row[0]).startswith("sqlite_")
    }


def test_bootstrap_creates_the_complete_current_schema(tmp_path):
    db = sqlite3.connect(tmp_path / "runs.sqlite3")
    try:
        schema.bootstrap(db)
        schema.validate(db)
        assert _tables(db) == {
            "schema_meta", "runs", "sessions", "experiments",
            "experiment_arms", "observations", "minimal_context_results",
        }
    finally:
        db.close()


def test_bootstrap_is_idempotent_and_preserves_current_rows(tmp_path):
    db = sqlite3.connect(tmp_path / "runs.sqlite3")
    try:
        schema.bootstrap(db)
        db.execute(
            """INSERT INTO runs(
                   id, created_ts, recorded_ts, created_at, source, client, client_key, session_key,
                   model, substrate, parent_run_id, finish_reason, error, prompt_summary,
                   response_summary, duration_ms, payload_json
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            ("run_current", 1.0, 2.0, "now", "cli", "test", None, None, "model", "engine",
             None, None, None, "prompt", "response", 10, '{"id":"run_current"}'),
        )
        db.commit()

        schema.bootstrap(db)

        assert db.execute("SELECT payload_json FROM runs WHERE id = 'run_current'").fetchone()[0] == (
            '{"id":"run_current"}'
        )
        schema.validate(db)
    finally:
        db.close()


def test_bootstrap_rejects_an_existing_noncurrent_shape_without_modifying_it(tmp_path):
    db = sqlite3.connect(tmp_path / "runs.sqlite3")
    db.execute("CREATE TABLE schema_meta (wrong_key TEXT)")
    db.commit()
    try:
        with pytest.raises(schema.SchemaError, match="does not match this build"):
            schema.bootstrap(db)
        assert _tables(db) == {"schema_meta"}
    finally:
        db.close()


def test_validate_rejects_a_missing_required_index(tmp_path):
    db = sqlite3.connect(tmp_path / "runs.sqlite3")
    try:
        schema.bootstrap(db)
        db.execute("DROP INDEX runs_recorded_idx")
        with pytest.raises(schema.SchemaError, match="does not match this build"):
            schema.validate(db)
    finally:
        db.close()


def test_bootstrap_works_in_memory():
    db = sqlite3.connect(":memory:")
    try:
        schema.bootstrap(db)
        schema.bootstrap(db)
        schema.validate(db)
    finally:
        db.close()
