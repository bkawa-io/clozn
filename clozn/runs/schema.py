"""Current SQLite schema for the local run store.

The product has one supported on-disk shape. A fresh store is created in one
transaction, and every initialized store is verified before use.
"""
from __future__ import annotations

import sqlite3


class SchemaError(RuntimeError):
    """The run database does not have the shape required by this build."""


_STATEMENTS = (
    "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """
    CREATE TABLE IF NOT EXISTS runs (
        id TEXT PRIMARY KEY,
        created_ts REAL NOT NULL,
        recorded_ts REAL NOT NULL,
        created_at TEXT NOT NULL,
        source TEXT NOT NULL,
        client TEXT NOT NULL,
        client_key TEXT,
        session_key TEXT,
        model TEXT NOT NULL,
        substrate TEXT NOT NULL,
        parent_run_id TEXT,
        finish_reason TEXT,
        error TEXT,
        prompt_summary TEXT NOT NULL,
        response_summary TEXT NOT NULL,
        duration_ms INTEGER NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS runs_created_idx ON runs(created_ts DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS runs_source_idx ON runs(source, created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS runs_parent_idx ON runs(parent_run_id, created_ts ASC)",
    "CREATE INDEX IF NOT EXISTS runs_model_idx ON runs(model, created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS runs_recorded_idx ON runs(recorded_ts DESC, id DESC)",
    "CREATE INDEX IF NOT EXISTS runs_client_latest_idx ON runs(client_key, recorded_ts DESC, id DESC)",
    """CREATE INDEX IF NOT EXISTS runs_session_latest_idx
       ON runs(session_key, recorded_ts DESC, id DESC) WHERE session_key IS NOT NULL""",
    """
    CREATE TABLE IF NOT EXISTS pinned_checkpoints (
        run_id TEXT PRIMARY KEY,
        pin_id TEXT NOT NULL,
        pinned_ts REAL NOT NULL,
        pinned_at TEXT NOT NULL,
        note TEXT,
        checkpoint_id TEXT NOT NULL,
        source_worker_generation_id TEXT NOT NULL,
        model_sha256 TEXT NOT NULL,
        architecture TEXT NOT NULL,
        n_embd INTEGER NOT NULL,
        n_layer INTEGER NOT NULL,
        vocab_size INTEGER NOT NULL,
        n_ctx INTEGER NOT NULL,
        protocol_version TEXT NOT NULL,
        engine_version TEXT NOT NULL,
        build_id TEXT NOT NULL,
        llama_cpp_commit TEXT NOT NULL,
        n_tokens INTEGER NOT NULL,
        n_past INTEGER NOT NULL,
        prompt_tokens INTEGER NOT NULL,
        causal INTEGER NOT NULL,
        has_sampler INTEGER NOT NULL,
        has_steer INTEGER NOT NULL,
        blob_sha256 TEXT NOT NULL,
        kv_bytes INTEGER NOT NULL,
        envelope_bytes INTEGER NOT NULL,
        payload_sha256 TEXT NOT NULL,
        manifest_json TEXT NOT NULL
    )
    """,
    """CREATE INDEX IF NOT EXISTS pinned_checkpoints_pinned_idx
       ON pinned_checkpoints(pinned_ts DESC, run_id DESC)""",
    """CREATE INDEX IF NOT EXISTS pinned_checkpoints_blob_idx
       ON pinned_checkpoints(blob_sha256)""",
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        created_ts REAL NOT NULL,
        created_at TEXT NOT NULL,
        client_key TEXT,
        title TEXT,
        privacy_json TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS sessions_created_idx ON sessions(created_ts DESC, id DESC)",
    """CREATE INDEX IF NOT EXISTS sessions_client_idx
       ON sessions(client_key, created_ts DESC) WHERE client_key IS NOT NULL""",
    """
    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        run_id TEXT NOT NULL,
        base_execution_fingerprint TEXT NOT NULL,
        evaluator_kind TEXT NOT NULL,
        state TEXT NOT NULL,
        created_ts REAL NOT NULL,
        updated_ts REAL NOT NULL,
        plan_json TEXT NOT NULL,
        diagnostics_json TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS experiment_arms (
        experiment_id TEXT NOT NULL,
        arm_id TEXT NOT NULL,
        ordinal INTEGER NOT NULL,
        is_control INTEGER NOT NULL,
        intervention_json TEXT,
        condition_json TEXT NOT NULL,
        state TEXT NOT NULL,
        observation_id TEXT,
        error_json TEXT,
        diagnostics_json TEXT NOT NULL,
        PRIMARY KEY (experiment_id, arm_id),
        UNIQUE (experiment_id, ordinal),
        FOREIGN KEY (experiment_id) REFERENCES experiments(id) ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS observations (
        id TEXT PRIMARY KEY,
        observation_key_sha256 TEXT NOT NULL UNIQUE,
        run_id TEXT NOT NULL,
        base_execution_fingerprint TEXT NOT NULL,
        evaluator_kind TEXT NOT NULL,
        condition_kind TEXT NOT NULL,
        status TEXT NOT NULL,
        created_ts REAL NOT NULL,
        artifact_ref_json TEXT NOT NULL,
        summary_json TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS experiments_run_idx ON experiments(run_id, created_ts DESC)",
    """CREATE INDEX IF NOT EXISTS experiment_arms_order_idx
       ON experiment_arms(experiment_id, ordinal ASC)""",
    """CREATE INDEX IF NOT EXISTS experiment_arms_observation_idx
       ON experiment_arms(observation_id)""",
    "CREATE INDEX IF NOT EXISTS observations_run_idx ON observations(run_id, created_ts DESC)",
    "CREATE INDEX IF NOT EXISTS observations_key_idx ON observations(observation_key_sha256)",
    """
    CREATE TABLE IF NOT EXISTS minimal_context_results (
        result_id TEXT PRIMARY KEY,
        search_id TEXT NOT NULL,
        run_id TEXT NOT NULL,
        base_execution_fingerprint TEXT NOT NULL,
        created_ts REAL NOT NULL,
        payload_json TEXT NOT NULL
    )
    """,
    """CREATE INDEX IF NOT EXISTS minimal_context_results_run_idx
       ON minimal_context_results(run_id, created_ts DESC, result_id DESC)""",
    """CREATE INDEX IF NOT EXISTS minimal_context_results_search_idx
       ON minimal_context_results(search_id, created_ts DESC, result_id DESC)""",
)


def _user_tables(db: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )
    }


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _shape(db: sqlite3.Connection) -> tuple:
    """Return the structural facts that make a database safe for current code."""
    tables = _user_tables(db)
    columns = tuple(
        (
            table,
            tuple(
                (str(row[1]), str(row[2]).upper(), int(row[3]), row[4], int(row[5]))
                for row in db.execute(f"PRAGMA table_info({_identifier(table)})")
            ),
        )
        for table in sorted(tables)
    )
    indexes = []
    for table in sorted(tables):
        for row in db.execute(f"PRAGMA index_list({_identifier(table)})"):
            name = str(row[1])
            if name.startswith("sqlite_autoindex_"):
                continue
            key_columns = tuple(
                (item[2], int(item[3]), str(item[4]), int(item[5]))
                for item in db.execute(f"PRAGMA index_xinfo({_identifier(name)})")
                if int(item[5]) == 1
            )
            indexes.append((table, name, int(row[2]), str(row[3]), int(row[4]), key_columns))
    foreign_keys = tuple(
        (table, tuple(tuple(row) for row in db.execute(f"PRAGMA foreign_key_list({_identifier(table)})")))
        for table in sorted(tables)
    )
    return tuple(sorted(tables)), columns, tuple(sorted(indexes)), foreign_keys


def _expected_shape() -> tuple:
    db = sqlite3.connect(":memory:")
    try:
        for statement in _STATEMENTS:
            db.execute(statement)
        return _shape(db)
    finally:
        db.close()


_CURRENT_SHAPE = _expected_shape()


def validate(db: sqlite3.Connection) -> None:
    """Raise :class:`SchemaError` unless the database exactly matches this build."""
    if _shape(db) != _CURRENT_SHAPE:
        raise SchemaError(
            "run database schema does not match this build; choose a fresh run directory"
        )


def bootstrap(db: sqlite3.Connection) -> None:
    """Create or verify the complete current schema in one transaction."""
    prior_isolation = db.isolation_level
    db.isolation_level = None
    try:
        db.execute("BEGIN IMMEDIATE")
        try:
            if not _user_tables(db):
                for statement in _STATEMENTS:
                    db.execute(statement)
            validate(db)
        except BaseException:
            db.execute("ROLLBACK")
            raise
        else:
            db.execute("COMMIT")
    finally:
        db.isolation_level = prior_isolation


__all__ = ["SchemaError", "bootstrap", "validate"]
