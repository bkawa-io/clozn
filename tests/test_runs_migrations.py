"""Contract tests for clozn/runs/migrations.py (BACKLOG §2: real transactional migrations, replacing the
old `_ensure()` schema-stamping).

Covers, per the backlog item's explicit contract:
  1. fresh-DB schema identity: migration 0 -> current produces the SAME structural schema as an old
     `_ensure()` database upgraded in place.
  2. legacy-DB upgrade: a DB actually left by the OLD `_ensure()` -- constructed here byte-for-byte the way
     that code built it -- upgrades in place, losslessly (every run row survives unchanged) and ends up
     structurally identical to a fresh migration.
  3. mid-migration failure: a step that fails partway rolls back cleanly and leaves the DB at exactly the
     prior version, still fully usable -- proven generically against a throwaway fabricated migration list
     (not by intentionally breaking the real, shipped migration).
  4. current_version/pending/status bookkeeping, and idempotency (migrating an up-to-date DB is a no-op).
"""
from __future__ import annotations

import os
import re
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clozn.runs.migrations as migrations  # noqa: E402


# ---------------------------------------------------------------------------------------------- reference SQL
# Verbatim copy of the OLD clozn/runs/store.py `_ensure()` (pre-BACKLOG-§2), the ad-hoc
# `executescript` + upsert-a-stamp approach this module replaces. Hardcoded here (not imported -- that
# code no longer exists) so this test has an independent, frozen ground truth for "what a fresh DB used to
# look like" that can never silently drift just because migrations.py changes.
_LEGACY_ENSURE_SQL = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    created_ts REAL NOT NULL,
    created_at TEXT NOT NULL,
    source TEXT NOT NULL,
    client TEXT NOT NULL,
    model TEXT NOT NULL,
    substrate TEXT NOT NULL,
    parent_run_id TEXT,
    finish_reason TEXT,
    error TEXT,
    prompt_summary TEXT NOT NULL,
    response_summary TEXT NOT NULL,
    duration_ms INTEGER NOT NULL,
    payload_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS runs_created_idx ON runs(created_ts DESC, id DESC);
CREATE INDEX IF NOT EXISTS runs_source_idx ON runs(source, created_ts DESC);
CREATE INDEX IF NOT EXISTS runs_parent_idx ON runs(parent_run_id, created_ts ASC);
CREATE INDEX IF NOT EXISTS runs_model_idx ON runs(model, created_ts DESC);
"""


def _build_legacy_db(path: str) -> None:
    """Reproduce exactly what the OLD `_ensure()` left on disk: the schema above, plus its
    schema_version=1 stamp row -- and nothing resembling the new migration ledger (no `migration:N` rows)."""
    db = sqlite3.connect(path)
    try:
        db.executescript(_LEGACY_ENSURE_SQL)
        db.execute(
            "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            ("1",),
        )
        db.commit()
    finally:
        db.close()


def _schema_dump(db: sqlite3.Connection) -> list[str]:
    """A structural, whitespace-insensitive dump of every table/index definition -- excludes DATA (rows),
    only the shape. SQLite already drops "IF NOT EXISTS" from the stored text, but internal whitespace
    from a multi-line CREATE TABLE is preserved verbatim, so we still normalize it (collapse runs of
    whitespace to one space) before comparing; this is what makes the comparison robust to the two call
    sites indenting their SQL differently in source, which is incidental formatting, not schema."""
    rows = db.execute(
        "SELECT type, name, sql FROM sqlite_master WHERE type IN ('table', 'index') AND sql IS NOT NULL"
    ).fetchall()
    def _norm(sql: str) -> str:
        collapsed = re.sub(r"\s+", " ", sql).strip()
        return collapsed.replace("( ", "(").replace(" )", ")")   # "CREATE TABLE x (\n    a,\n)" vs "(a)"

    normalized = [f"{t}:{name}:{_norm(sql)}" for (t, name, sql) in rows]
    return sorted(normalized)


def _row_ids(db: sqlite3.Connection) -> list[str]:
    return sorted(r[0] for r in db.execute("SELECT id FROM runs"))


# ======================================================================================= 1. fresh-DB identity

def test_fresh_migration_matches_upgraded_legacy_schema(tmp_path):
    """A fresh current DB and an in-place-upgraded legacy DB have identical structural schemas."""
    legacy_path = str(tmp_path / "legacy.sqlite3")
    _build_legacy_db(legacy_path)
    legacy_db = sqlite3.connect(legacy_path)
    legacy_applied = migrations.migrate(legacy_db)

    fresh_path = str(tmp_path / "fresh.sqlite3")
    fresh_db = sqlite3.connect(fresh_path)
    applied = migrations.migrate(fresh_db)

    try:
        assert legacy_applied == [migration.version for migration in migrations.MIGRATIONS]
        assert applied == [migration.version for migration in migrations.MIGRATIONS]
        assert _schema_dump(fresh_db) == _schema_dump(legacy_db)
    finally:
        legacy_db.close()
        fresh_db.close()


def test_fresh_migration_reaches_target_version(tmp_path):
    db = sqlite3.connect(str(tmp_path / "fresh.sqlite3"))
    try:
        assert migrations.current_version(db) == 0
        assert migrations.pending(db) == list(migrations.MIGRATIONS)
        migrations.migrate(db)
        assert migrations.current_version(db) == migrations.TARGET_VERSION
        assert migrations.pending(db) == []
    finally:
        db.close()


def test_scope_reset_drops_durable_correction_tables_from_existing_journals(tmp_path):
    """Migration 6 removes F5/F6 storage from an install that previously reached version 5."""
    db = sqlite3.connect(str(tmp_path / "retired-corrections.sqlite3"))
    try:
        v5 = tuple(migration for migration in migrations.MIGRATIONS if migration.version <= 5)
        assert migrations.migrate(db, v5) == [migration.version for migration in v5]

        assert migrations.migrate(db) == [6, 7]
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        assert "corrections" not in tables
        assert "correction_events" not in tables
        assert "experiments" in tables
    finally:
        db.close()


def test_claimed_superseder_suppresses_missing_superseded_migration(tmp_path):
    """A cleanup ledger row prevents a missing predecessor from recreating retired tables."""
    db = sqlite3.connect(str(tmp_path / "superseded-ledger-gap.sqlite3"))
    try:
        v4 = tuple(migration for migration in migrations.MIGRATIONS if migration.version <= 4)
        assert migrations.migrate(db, v4) == [migration.version for migration in v4]

        migration_6 = next(migration for migration in migrations.MIGRATIONS if migration.version == 6)
        db.execute(
            "INSERT INTO schema_meta(key, value) VALUES(?, ?)",
            ("migration:6", migration_6.description),
        )
        db.commit()

        assert 5 not in [migration.version for migration in migrations.pending(db)]
        assert migrations.migrate(db) == [7]
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        assert "corrections" not in tables
        assert "correction_events" not in tables
    finally:
        db.close()


def test_stale_pending_snapshot_cannot_reapply_a_superseded_migration(tmp_path, monkeypatch):
    """A concurrent cleanup between planning and locking cannot let migration 5 recreate its tables."""
    path = str(tmp_path / "stale-snapshot.sqlite3")
    first = sqlite3.connect(path)
    second = sqlite3.connect(path)
    original_pending = migrations.pending
    try:
        v5 = tuple(migration for migration in migrations.MIGRATIONS if migration.version <= 5)
        assert migrations.migrate(first, v5) == [migration.version for migration in v5]
        first.execute("DROP TABLE correction_events")
        first.execute("DROP TABLE corrections")
        first.commit()

        stale_pending = original_pending(first)
        assert [migration.version for migration in stale_pending] == [5, 6, 7]
        interleaved = False

        def _pending_after_concurrent_cleanup(db, migration_set=migrations.MIGRATIONS):
            nonlocal interleaved
            if db is first and not interleaved:
                interleaved = True
                assert migrations.migrate(second) == [5, 6, 7]
                return stale_pending
            return original_pending(db, migration_set)

        monkeypatch.setattr(migrations, "pending", _pending_after_concurrent_cleanup)
        assert migrations.migrate(first) == []
        assert interleaved is True
        tables = {row[0] for row in first.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        assert "corrections" not in tables
        assert "correction_events" not in tables
    finally:
        second.close()
        first.close()


def test_fresh_current_migration_has_no_durable_correction_tables(tmp_path):
    db = sqlite3.connect(str(tmp_path / "fresh-current.sqlite3"))
    try:
        assert migrations.migrate(db) == [migration.version for migration in migrations.MIGRATIONS]
        tables = {row[0] for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )}
        assert "corrections" not in tables
        assert "correction_events" not in tables
    finally:
        db.close()


def test_migrate_is_idempotent_on_an_up_to_date_db(tmp_path):
    db = sqlite3.connect(str(tmp_path / "fresh.sqlite3"))
    try:
        first = migrations.migrate(db)
        second = migrations.migrate(db)
        assert first == [migration.version for migration in migrations.MIGRATIONS]
        assert second == []                          # nothing pending -> no-op, not a re-apply
    finally:
        db.close()


# ======================================================================================= 2. legacy DB upgrade

def test_legacy_db_upgrades_in_place_losslessly(tmp_path):
    """A DB actually shaped by the old `_ensure()` -- carrying real run rows -- must upgrade with zero data
    loss: every row survives byte-for-byte, and the ledger correctly reports the new version afterward."""
    path = str(tmp_path / "legacy.sqlite3")
    _build_legacy_db(path)
    db = sqlite3.connect(path)
    try:
        # A legacy DB predates the migration ledger entirely -- current_version() must not choke on that,
        # and must not mistake "schema_version=1 legacy stamp" for "migration:1 applied".
        assert migrations.current_version(db) == 0
        assert migrations.pending(db) == list(migrations.MIGRATIONS)

        db.execute(
            "INSERT INTO runs(id, created_ts, created_at, source, client, model, substrate, parent_run_id,"
            " finish_reason, error, prompt_summary, response_summary, duration_ms, payload_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run_legacy_1", 1000.0, "2026-01-01T00:00:00", "cli", "unknown", "qwen", "engine", None,
             None, None, "hi", "hello", 5, '{"id": "run_legacy_1"}'),
        )
        db.execute(
            "INSERT INTO runs(id, created_ts, created_at, source, client, model, substrate, parent_run_id,"
            " finish_reason, error, prompt_summary, response_summary, duration_ms, payload_json)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("run_legacy_2", 2000.0, "2026-01-01T00:01:00", "cli", "unknown", "qwen", "engine", None,
             None, None, "bye", "goodbye", 7, '{"id": "run_legacy_2"}'),
        )
        db.commit()
        before = _row_ids(db)
        assert before == ["run_legacy_1", "run_legacy_2"]

        applied = migrations.migrate(db)

        assert applied == [migration.version for migration in migrations.MIGRATIONS]
        assert migrations.current_version(db) == migrations.TARGET_VERSION
        assert migrations.pending(db) == []
        assert _row_ids(db) == before                  # lossless: same rows, same ids, nothing dropped

        # both original rows' full column contents are untouched
        row = db.execute("SELECT * FROM runs WHERE id = 'run_legacy_1'").fetchone()
        assert row is not None

        # post-migration the legacy DB's structural schema matches a from-scratch migration exactly
        fresh_db = sqlite3.connect(str(tmp_path / "fresh_compare.sqlite3"))
        migrations.migrate(fresh_db)
        assert _schema_dump(db) == _schema_dump(fresh_db)
        fresh_db.close()
    finally:
        db.close()


def test_legacy_db_keeps_the_coarse_schema_version_stamp_in_sync(tmp_path):
    """Nothing else in the repo reads schema_meta['schema_version'] today (grepped), but it predates this
    module and costs nothing to keep correct -- migrating must not leave it stale at the old value."""
    path = str(tmp_path / "legacy.sqlite3")
    _build_legacy_db(path)
    db = sqlite3.connect(path)
    try:
        migrations.migrate(db)
        value = db.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'").fetchone()[0]
        assert value == str(migrations.TARGET_VERSION)
    finally:
        db.close()


# ============================================================================ 3. mid-migration failure rollback

def test_mid_migration_failure_rolls_back_and_leaves_prior_version_usable(tmp_path):
    """Engine-level contract, proven against a throwaway fabricated migration list (not the real shipped
    one, which has nothing worth intentionally breaking): step 1 lands and commits; step 2 partially runs
    DDL then raises; step 2's DDL must be fully rolled back, the ledger must still read version 1 (not a
    half-applied 2), and the DB must remain fully usable afterward (not corrupted, not locked)."""
    db = sqlite3.connect(str(tmp_path / "test.sqlite3"))

    def _step1(conn):
        conn.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY)")

    def _step2_broken(conn):
        conn.execute("CREATE TABLE t2_partial (id INTEGER PRIMARY KEY)")   # this DDL must NOT survive
        raise RuntimeError("boom mid-step")

    fake = (
        migrations.Migration(1, "create t1", _step1),
        migrations.Migration(2, "create t2 (broken)", _step2_broken),
    )

    try:
        with pytest.raises(RuntimeError, match="boom mid-step"):
            migrations.migrate(db, fake)

        assert migrations.current_version(db) == 1                # NOT 2 -- step 2 never landed
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "t1" in tables                                      # step 1's DDL survives (committed)
        assert "t2_partial" not in tables                          # step 2's DDL was rolled back

        # the DB is still fully usable, not left locked/half-open by the manual transaction handling
        db.execute("INSERT INTO t1 (id) VALUES (1)")
        db.commit()
        assert db.execute("SELECT COUNT(*) FROM t1").fetchone()[0] == 1

        # recovery: fix step 2 and retry -- must resume from version 1, not re-run step 1
        ran_step1_again = []

        def _step1_spy(conn):
            ran_step1_again.append(True)

        def _step2_fixed(conn):
            conn.execute("CREATE TABLE t2 (id INTEGER PRIMARY KEY)")

        fixed = (
            migrations.Migration(1, "create t1", _step1_spy),
            migrations.Migration(2, "create t2", _step2_fixed),
        )
        applied = migrations.migrate(db, fixed)
        assert applied == [2]
        assert not ran_step1_again                                 # version 1 was already applied -- skipped
        assert migrations.current_version(db) == 2
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "t2" in tables
    finally:
        db.close()


def test_migration_failure_does_not_apply_later_pending_steps(tmp_path):
    """Three fabricated steps, the middle one broken: the third must never run even though it doesn't
    depend on the second -- migrations apply strictly in order, never skip-ahead past a failure."""
    db = sqlite3.connect(str(tmp_path / "test.sqlite3"))
    ran = []

    def _ok(n):
        def _apply(conn):
            ran.append(n)
        return _apply

    def _broken(conn):
        ran.append("broken-ran")
        raise ValueError("nope")

    fake = (
        migrations.Migration(1, "ok 1", _ok(1)),
        migrations.Migration(2, "broken", _broken),
        migrations.Migration(3, "ok 3 -- must not run", _ok(3)),
    )
    try:
        with pytest.raises(ValueError):
            migrations.migrate(db, fake)
        assert ran == [1, "broken-ran"]                            # step 3 never attempted
        assert migrations.current_version(db) == 1
    finally:
        db.close()


# ========================================================================================= 4. status() reporting

def test_status_reports_current_target_and_pending(tmp_path):
    db = sqlite3.connect(str(tmp_path / "test.sqlite3"))
    try:
        report = migrations.status(db)
        assert report["current_version"] == 0
        assert report["target_version"] == migrations.TARGET_VERSION
        assert report["up_to_date"] is False
        assert [p["version"] for p in report["pending"]] == [m.version for m in migrations.MIGRATIONS]

        migrations.migrate(db)
        report2 = migrations.status(db)
        assert report2["current_version"] == migrations.TARGET_VERSION
        assert report2["up_to_date"] is True
        assert report2["pending"] == []
    finally:
        db.close()


def test_status_is_read_only(tmp_path):
    """Calling status() must never itself apply anything -- only migrate() writes."""
    db = sqlite3.connect(str(tmp_path / "test.sqlite3"))
    try:
        migrations.status(db)
        migrations.status(db)
        assert migrations.current_version(db) == 0            # still unmigrated after two status() calls
    finally:
        db.close()
