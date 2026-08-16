"""Real, transactional schema migrations for clozn/runs/store.py's SQLite database (BACKLOG §2).

Replaces the old `_ensure()` "CREATE TABLE IF NOT EXISTS + upsert a stamp" approach. That approach worked
by accident (every DDL statement was individually idempotent) but had three real gaps:
  - no audit trail -- a bare integer said "we're at version 1" but nothing recorded WHICH steps actually
    ran, or when;
  - no failure semantics -- `executescript` runs several statements back to back with no defined recovery
    if one of them ever fails partway (SQLite CAN roll back DDL, but only if the driver is told to use a
    real transaction, which the old code never did);
  - no dry-run -- `clozn` mutated the on-disk DB the instant anything opened it, with no way to preview.

Design
------
Each migration is a small, ordered (version, description, apply(db)) step. `migrate()` applies every
PENDING step in order, each inside its own explicit transaction: the step's DDL/DML and the ledger row
that marks it applied land in the SAME COMMIT, or neither lands at all (ROLLBACK propagates the original
exception to the caller). A failure at step N therefore leaves the DB at EXACTLY version N-1 -- fully
usable, never half-migrated -- and a subsequent `migrate()` call retries from N.

The ledger deliberately reuses the pre-existing `schema_meta(key, value)` table (rather than adding a new
`schema_migrations` table), so migration bookkeeping remains extra ROWS rather than an extra schema
object. Migration 1 preserves the old baseline exactly; later append-only migrations evolve it normally.
Fresh and upgraded legacy databases are asserted structurally identical in tests/test_runs_migrations.py.
Per-migration rows are keyed
`migration:<version>` (JSON value: description + applied_at); the coarse `schema_version` key is kept in
sync too since it predates this module and nothing else in the repo reads the per-migration rows.

Python's sqlite3 module does NOT auto-open a transaction before DDL in its default ("") isolation mode --
only before INSERT/UPDATE/DELETE -- so a naive `db.executescript(...)` between two explicit
BEGIN/COMMIT calls silently runs outside any transaction and can't be rolled back. `migrate()` works
around this by switching the connection to `isolation_level = None` (manual/autocommit mode) for its own
duration and issuing `BEGIN IMMEDIATE` / `COMMIT` / `ROLLBACK` itself -- SQLite the engine fully supports
transactional DDL once the driver gets out of its way.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

_MIGRATION_KEY_RE = re.compile(r"^migration:(\d+)$")


@dataclass(frozen=True)
class Migration:
    version: int
    description: str
    apply: Callable[[sqlite3.Connection], None]
    verify: Callable[[sqlite3.Connection], bool] | None = None
    supersedes: frozenset[int] = frozenset()


def _migration_0001_initial_schema(db: sqlite3.Connection) -> None:
    """The baseline schema: byte-for-byte what the old `_ensure()` created in one `executescript` call --
    but issued as individual `execute()` calls here, NOT `executescript()`. `executescript()` implicitly
    COMMITs any already-open transaction before it runs (Python sqlite3 docs: "If there is a pending
    transaction, an implicit COMMIT statement is executed first") -- inside `migrate()`'s explicit `BEGIN
    IMMEDIATE ... COMMIT` wrapper that silently ends OUR transaction partway through, so a later step's
    failure could no longer roll this one back. Individual `execute()` calls have no such side effect.
    Kept as ONE migration (not split further) because there is nothing partial about it to test -- the
    mid-migration-failure contract is proven generically in tests/test_runs_migrations.py against a
    throwaway fabricated migration list, not by intentionally breaking this real one."""
    db.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute(
        """
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
        )
        """
    )
    db.execute("CREATE INDEX IF NOT EXISTS runs_created_idx ON runs(created_ts DESC, id DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS runs_source_idx ON runs(source, created_ts DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS runs_parent_idx ON runs(parent_run_id, created_ts ASC)")
    db.execute("CREATE INDEX IF NOT EXISTS runs_model_idx ON runs(model, created_ts DESC)")


def _verify_0001(db: sqlite3.Connection) -> bool:
    """Schema-level check that migration 1 actually landed: the runs table must exist."""
    return db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='runs'").fetchone() is not None


def _migration_0002_run_association(db: sqlite3.Connection) -> None:
    """Add insertion-order cursors and opaque client/session lookup columns.

    ``created_ts`` is generation start time, so it is not a safe polling cursor: a slow request may be
    journaled after a request that started later.  ``recorded_ts`` captures the actual journal insertion
    order.  Existing records use their start time as the only honest backfill available.
    """
    columns = {row[1] for row in db.execute("PRAGMA table_info(runs)")}
    if "recorded_ts" not in columns:
        db.execute("ALTER TABLE runs ADD COLUMN recorded_ts REAL")
    if "client_key" not in columns:
        db.execute("ALTER TABLE runs ADD COLUMN client_key TEXT")
    if "session_key" not in columns:
        db.execute("ALTER TABLE runs ADD COLUMN session_key TEXT")
    db.execute("UPDATE runs SET recorded_ts = created_ts WHERE recorded_ts IS NULL")
    db.execute("CREATE INDEX IF NOT EXISTS runs_recorded_idx ON runs(recorded_ts DESC, id DESC)")
    db.execute(
        "CREATE INDEX IF NOT EXISTS runs_client_latest_idx "
        "ON runs(client_key, recorded_ts DESC, id DESC)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS runs_session_latest_idx "
        "ON runs(session_key, recorded_ts DESC, id DESC) WHERE session_key IS NOT NULL"
    )


def _verify_0002(db: sqlite3.Connection) -> bool:
    columns = {row[1] for row in db.execute("PRAGMA table_info(runs)")}
    indexes = {row[1] for row in db.execute("PRAGMA index_list(runs)")}
    return (
        {"recorded_ts", "client_key", "session_key"}.issubset(columns)
        and {"runs_recorded_idx", "runs_client_latest_idx", "runs_session_latest_idx"}.issubset(indexes)
    )


def _migration_0003_pinned_checkpoints(db: sqlite3.Connection) -> None:
    """FORK-PIN-01: durable checkpoint pin metadata. One row per pinned run (``run_id`` is the
    primary key -- ``clozn snapshot pin/unpin`` address a pin by the run it was captured from, not a
    separate opaque id). The KV bytes themselves never touch this table or this database file: they
    live content-addressed under ``RUNS_DIR/blobs/checkpoints/sha256`` (clozn.replay.checkpoint_pin_store,
    mirroring clozn.analysis.tensor_store's binary-blob-plus-JSON-sidecar convention) -- this row is
    purely metadata + the sha256 reference, so a pin's presence/absence is answerable from SQLite
    alone without ever touching the (potentially large) blob file. No FOREIGN KEY to runs(id): a pin
    outliving the row it was captured from is a valid, inspectable state (the run's own deletion
    policy is store.py/mutations.py's concern, not this table's), and coupling deletion failure modes
    across two independently-evolving tables was judged a bigger risk than a dangling run_id string.
    """
    db.execute(
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
        """
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS pinned_checkpoints_pinned_idx "
        "ON pinned_checkpoints(pinned_ts DESC, run_id DESC)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS pinned_checkpoints_blob_idx "
        "ON pinned_checkpoints(blob_sha256)"
    )


def _verify_0003(db: sqlite3.Connection) -> bool:
    columns = {row[1] for row in db.execute("PRAGMA table_info(pinned_checkpoints)")}
    indexes = {row[1] for row in db.execute("PRAGMA index_list(pinned_checkpoints)")}
    return (
        {"run_id", "blob_sha256", "manifest_json"}.issubset(columns)
        and {"pinned_checkpoints_pinned_idx", "pinned_checkpoints_blob_idx"}.issubset(indexes)
    )


def _migration_0004_sessions(db: sqlite3.Connection) -> None:
    """F1: first-class session records -- promotes `X-Clozn-Session-Id` from a lookup column
    (`runs.session_key`, migration 2) into a persisted conversation ENTITY: the substrate for session
    traces (F2) and the conversation investigation view (F3).

    One row per session id (`id` = the SAME opaque `session_<24hex>` key already written into
    `runs.session_key` -- clozn.runs.association.session_key()'s normalization is reused as-is, not
    reinvented, so a session created through this table and a run recorded elsewhere with the identical
    caller-supplied token always resolve to the SAME id with zero extra plumbing). Deliberately narrow:
    only session-level IDENTITY/metadata lives here (id, creation stamp, the creating client's opaque
    identity facet, an optional title, and privacy settings) -- `first/last activity` and `run_count` are
    NOT stored columns. They are computed on every read as MIN/MAX(recorded_ts)/COUNT(*) over
    `runs WHERE session_key = ?` (clozn/runs/sessions.py), which migration 2's own
    `runs_session_latest_idx` already indexes. Two reasons this is a stored-vs-derived split, not an
    oversight:
      1. RUN IMMUTABILITY: a materialized `last_activity_ts` column would need updating on every single
         `record()` call for a session's runs, and clozn/runs/store.py's `record()` is a hot path wrapped
         in a blanket try/except that treats ANY failure as "the whole run silently vanished" -- adding a
         second table write to that path is exactly the kind of coupling BACKLOG's "additive only" rule
         on store.py warns against.
      2. STALENESS UNDER MUTATION: clozn/runs/mutations.py's `redact_run` clears a run's `session_key`
         and `delete_run`/cascade removes rows outright, entirely independently of this table. A
         materialized aggregate would drift out of sync with no synchronization hook between the two
         modules; a derived-on-read aggregate is definitionally always correct, because it re-reads
         `runs` fresh every time -- there is nothing to keep in sync.

    NO existing-run backfill runs here, by design (the "your call; justify it" backfill decision landed
    on LAZY, not migration-time): a legacy session_key already fully round-trips through
    clozn.runs.sessions.get_session()'s derived-aggregate path with zero rows in THIS table -- the only
    thing a migration-time backfill could add is the identity/title/privacy row, and for old data there is
    no honest client_key to backfill (a legacy session_key may span runs from several DIFFERENT client_key
    values recorded before this concept existed -- picking one would fabricate provenance) and no title
    ever existed to backfill (the field is new). A migration that scanned the full `runs` table to
    synthesize approximate rows would also hold `_ensure_schema_locked`'s cross-process lock
    (store.py's module docstring) for however long that scan takes, on every install's very next store
    open -- unacceptable for a step whose payoff is "the exact same read result, sooner." See
    clozn.runs.sessions.materialize_session() for the actual lazy, idempotent, race-safe backfill path,
    triggered opportunistically (e.g. GET /sessions/<id>) rather than unconditionally here.
    """
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            created_ts REAL NOT NULL,
            created_at TEXT NOT NULL,
            client_key TEXT,
            title TEXT,
            privacy_json TEXT NOT NULL,
            materialized_from TEXT NOT NULL
        )
        """
    )
    db.execute("CREATE INDEX IF NOT EXISTS sessions_created_idx ON sessions(created_ts DESC, id DESC)")
    db.execute(
        "CREATE INDEX IF NOT EXISTS sessions_client_idx "
        "ON sessions(client_key, created_ts DESC) WHERE client_key IS NOT NULL"
    )


def _verify_0004(db: sqlite3.Connection) -> bool:
    columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
    indexes = {row[1] for row in db.execute("PRAGMA index_list(sessions)")}
    return (
        {"id", "created_ts", "client_key", "title", "privacy_json", "materialized_from"}.issubset(columns)
        and {"sessions_created_idx", "sessions_client_idx"}.issubset(indexes)
    )


def _migration_0005_corrections(db: sqlite3.Connection) -> None:
    """F5: the scoped correction store ("Teach Once") -- `corrections` (one row per correction, exactly
    the fields clozn.correction.v1 requires plus the mutable lifecycle timestamps) and `correction_events`
    (an APPEND-ONLY ledger: drafted/confirmed/disabled/enabled/deleted lifecycle rows plus applied/
    conflict_lost rows tying a correction to a specific run -- see clozn/runs/corrections.py's module
    docstring for why this is one ledger table rather than a separate "history" table per concern).

    No FOREIGN KEY from correction_events to corrections, deliberately -- the same reasoning migration 3
    (`pinned_checkpoints`) already gives for skipping a FK to `runs`: `delete_correction()` scrubs the
    `corrections` row's `content` but the module never deletes the row outright, and correction_events
    rows must keep referencing a correction_id that is still resolvable (just content-scrubbed) forever,
    so a run's `applied_corrections` receipt entry stays checkable even after the correction it names has
    been deleted. Coupling the two tables' failure modes together was judged the bigger risk, exactly as
    migration 3 judged it for checkpoints.

    `correction_events.run_id` has no FOREIGN KEY to `runs.id` either: an applied/conflict_lost event must
    survive `clozn.runs.mutations.delete_run` on the run it names (disable/delete-run are independent
    lifecycles; the run's own deletion policy must never quietly rewrite a correction's audit trail), so
    the reference is an inert string, not a referential-integrity constraint.
    """
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS corrections (
            id TEXT PRIMARY KEY,
            scope_kind TEXT NOT NULL,
            scope_value TEXT,
            type TEXT NOT NULL,
            content TEXT,
            content_hash TEXT NOT NULL,
            enabled INTEGER NOT NULL,
            created_ts REAL NOT NULL,
            created_at TEXT NOT NULL,
            confirmed_ts REAL,
            disabled_ts REAL,
            deleted_ts REAL,
            deleted_reason TEXT
        )
        """
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS corrections_resolve_idx "
        "ON corrections(type, scope_kind, scope_value) WHERE confirmed_ts IS NOT NULL AND enabled = 1"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS corrections_scope_idx ON corrections(scope_kind, scope_value)"
    )
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS correction_events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            correction_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            event_ts REAL NOT NULL,
            run_id TEXT,
            detail_json TEXT
        )
        """
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS correction_events_correction_idx "
        "ON correction_events(correction_id, seq ASC)"
    )
    db.execute(
        "CREATE INDEX IF NOT EXISTS correction_events_run_idx "
        "ON correction_events(run_id) WHERE run_id IS NOT NULL"
    )


def _verify_0005(db: sqlite3.Connection) -> bool:
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if not {"corrections", "correction_events"}.issubset(tables):
        return False
    columns = {row[1] for row in db.execute("PRAGMA table_info(corrections)")}
    event_columns = {row[1] for row in db.execute("PRAGMA table_info(correction_events)")}
    return (
        {"id", "scope_kind", "scope_value", "type", "content", "content_hash", "enabled",
         "confirmed_ts", "disabled_ts", "deleted_ts"}.issubset(columns)
        and {"seq", "correction_id", "event_type", "event_ts", "run_id"}.issubset(event_columns)
    )


def _migration_0006_remove_retired_corrections(db: sqlite3.Connection) -> None:
    """Remove the unused durable-correction store and its event ledger.

    The F5/F6 domain, CLI, and HTTP surface have all been removed.  Keeping tables that no supported
    code can read or write leaves stale behavioral-policy data in every new run journal, so this cleanup
    drops both tables for existing installs as well as keeping them out of fresh databases.
    """
    db.execute("DROP TABLE IF EXISTS correction_events")
    db.execute("DROP TABLE IF EXISTS corrections")


def _migration_0007_durable_experiments(db: sqlite3.Connection) -> None:
    """Durable new-kernel experiment plans, arm associations, and observations.

    Large direct evidence remains in the existing content-addressed JSON blob
    tree; these rows contain only identity, lifecycle, and searchable summary
    metadata.  Observation rows intentionally have no foreign key to runs:
    evidence remains inspectable if a separate run-retention policy removes its
    parent later.
    """
    db.execute(
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
        """
    )
    db.execute(
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
        """
    )
    db.execute(
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
        """
    )
    db.execute("CREATE INDEX IF NOT EXISTS experiments_run_idx ON experiments(run_id, created_ts DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS experiment_arms_order_idx ON experiment_arms(experiment_id, ordinal ASC)")
    db.execute("CREATE INDEX IF NOT EXISTS experiment_arms_observation_idx ON experiment_arms(observation_id)")
    db.execute("CREATE INDEX IF NOT EXISTS observations_run_idx ON observations(run_id, created_ts DESC)")
    db.execute("CREATE INDEX IF NOT EXISTS observations_key_idx ON observations(observation_key_sha256)")


def _verify_0007(db: sqlite3.Connection) -> bool:
    required = {
        "experiments": {"id", "run_id", "base_execution_fingerprint", "evaluator_kind", "state", "plan_json"},
        "experiment_arms": {"experiment_id", "arm_id", "ordinal", "state", "observation_id"},
        "observations": {"id", "observation_key_sha256", "run_id", "status", "artifact_ref_json"},
    }
    for table, columns in required.items():
        actual = {row[1] for row in db.execute(f"PRAGMA table_info({table})")}
        if not columns.issubset(actual):
            return False
    indexes = {row[1] for row in db.execute("PRAGMA index_list(experiments)")}
    arm_indexes = {row[1] for row in db.execute("PRAGMA index_list(experiment_arms)")}
    observation_indexes = {row[1] for row in db.execute("PRAGMA index_list(observations)")}
    return (
        "experiments_run_idx" in indexes
        and "experiment_arms_order_idx" in arm_indexes
        and "experiment_arms_observation_idx" in arm_indexes
        and "observations_run_idx" in observation_indexes
        and "observations_key_idx" in observation_indexes
    )


# The shipped, ordered migration set. Append-only: once released, a migration's `apply` must never be
# edited (a DB that already applied it would silently diverge from one that applies the edited version) --
# ship a NEW migration with a higher version instead.
MIGRATIONS: tuple[Migration, ...] = (
    Migration(1, "initial schema: schema_meta + runs + indexes", _migration_0001_initial_schema,
              verify=_verify_0001),
    Migration(2, "run association: insertion cursor + opaque client/session keys",
              _migration_0002_run_association, verify=_verify_0002),
    Migration(3, "FORK-PIN-01: durable checkpoint pin metadata (pinned_checkpoints table)",
              _migration_0003_pinned_checkpoints, verify=_verify_0003),
    Migration(4, "F1: first-class session records (sessions table)",
              _migration_0004_sessions, verify=_verify_0004),
    Migration(5, "F5: scoped correction store (corrections + correction_events tables)",
              _migration_0005_corrections, verify=_verify_0005),
    Migration(6, "scope reset: remove retired durable-correction storage",
              _migration_0006_remove_retired_corrections, supersedes=frozenset({5})),
    Migration(7, "durable experimental plans, arms, and observations",
              _migration_0007_durable_experiments, verify=_verify_0007),
)

TARGET_VERSION = max(m.version for m in MIGRATIONS)


def _ensure_ledger_table(db: sqlite3.Connection) -> None:
    """Bootstrap the ledger table itself, outside any migration transaction. Safe to call unconditionally
    on both a brand-new DB file (creates it) and an existing legacy one (already has this exact table from
    the old `_ensure()` -- a no-op). This is NOT "migration 0": it never needs rolling back, because
    creating an empty key/value table has no partial state to roll back TO."""
    db.execute("CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")


def current_version(db: sqlite3.Connection) -> int:
    """The highest migration version whose ledger row is present. 0 for a brand-new DB (including one that
    doesn't even have the schema_meta table yet -- e.g. an in-memory DB nobody has touched)."""
    _ensure_ledger_table(db)
    rows = db.execute("SELECT key FROM schema_meta WHERE key LIKE 'migration:%'").fetchall()
    versions = []
    for row in rows:
        m = _MIGRATION_KEY_RE.match(row[0])
        if m:
            versions.append(int(m.group(1)))
    return max(versions, default=0)


def _claimed_versions(db: sqlite3.Connection) -> set[int]:
    """Return the migration versions recorded in the ledger.

    Callers use this both while planning and after acquiring a migration transaction's write lock.  The
    latter re-read is important: another process may have completed a cleanup migration after this process
    took its initial pending snapshot.
    """
    rows = db.execute("SELECT key FROM schema_meta WHERE key LIKE 'migration:%'").fetchall()
    claimed: set[int] = set()
    for row in rows:
        match = _MIGRATION_KEY_RE.match(row[0])
        if match:
            claimed.add(int(match.group(1)))
    return claimed


def _superseded_versions(migrations: Sequence[Migration], claimed: set[int]) -> set[int]:
    """Versions made permanently inapplicable by claimed cleanup migrations.

    Supersession is transitive.  If a claimed migration supersedes a predecessor which itself supersedes
    another migration, neither predecessor may be repaired or re-applied.  This keeps an old verifier from
    recreating schema intentionally removed by a later migration.
    """
    superseded: set[int] = set()
    frontier = set(claimed)
    while frontier:
        newly_superseded = {
            version
            for migration in migrations
            if migration.version in frontier
            for version in migration.supersedes
        } - superseded
        if not newly_superseded:
            break
        superseded.update(newly_superseded)
        frontier = newly_superseded
    return superseded


def pending(db: sqlite3.Connection, migrations: Sequence[Migration] = MIGRATIONS) -> list[Migration]:
    """Migrations not yet applied to `db`, in ascending version order.

    A migration is pending if its ledger row is absent or its verifier reports an inconsistent schema,
    except when a claimed later migration supersedes it.  A superseded step must never repair its old
    schema, even if its own ledger row is missing or its verifier now fails by design.
    """
    _ensure_ledger_table(db)
    claimed = _claimed_versions(db)
    superseded = _superseded_versions(migrations, claimed)
    result = []
    for m in sorted(migrations, key=lambda x: x.version):
        if m.version in superseded:
            continue
        if m.version not in claimed:
            result.append(m)
        elif m.verify is not None and not m.verify(db):
            result.append(m)
    return result


def migrate(db: sqlite3.Connection, migrations: Sequence[Migration] = MIGRATIONS) -> list[int]:
    """Apply every pending migration to `db`, each in its own transaction. Returns the versions actually
    applied (empty list if already current). Raises on the first failing step WITHOUT applying any step
    after it -- the caller decides whether that's fatal (the CLI) or should degrade quietly (store._ensure,
    which already tolerated an unusable DB before this module existed)."""
    versions = [m.version for m in migrations]
    dupes = sorted({v for v in versions if versions.count(v) > 1})
    if dupes:
        raise ValueError(f"duplicate migration version(s): {dupes}")
    _ensure_ledger_table(db)
    applied: list[int] = []
    prior_isolation = db.isolation_level
    db.isolation_level = None      # manual transaction control -- see module docstring for why this
                                    # matters: default mode never auto-BEGINs around DDL, so without this
                                    # a mid-step failure would leave whatever DDL already ran committed.
    try:
        for m in pending(db, migrations):
            db.execute("BEGIN IMMEDIATE")
            try:
                # `pending()` was a snapshot taken before this transaction acquired its write lock.  A
                # different process can have applied a later cleanup migration in between; re-read the
                # ledger under the lock so a stale predecessor can never recreate retired schema.
                claimed = _claimed_versions(db)
                if m.version in _superseded_versions(migrations, claimed):
                    db.execute("ROLLBACK")
                    continue
                already = m.version in claimed
                if already and (m.verify is None or m.verify(db)):
                    db.execute("ROLLBACK")
                    continue
                m.apply(db)
                stamp = json.dumps({"description": m.description,
                                     "applied_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
                db.execute(
                    "INSERT OR REPLACE INTO schema_meta(key, value) VALUES(?, ?)",
                    (f"migration:{m.version}", stamp),
                )
                db.execute(
                    "INSERT INTO schema_meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(m.version),),
                )
            except BaseException:
                db.execute("ROLLBACK")
                raise
            else:
                db.execute("COMMIT")
            applied.append(m.version)
    finally:
        db.isolation_level = prior_isolation
    return applied


def status(db: sqlite3.Connection, migrations: Sequence[Migration] = MIGRATIONS) -> dict:
    """A doctor-style snapshot for `clozn migrate` / `clozn migrate --dry-run`: current version, target
    version, and the ordered list of steps that would run. Read-only -- never mutates `db`."""
    current = current_version(db)
    target = max((m.version for m in migrations), default=0)
    todo = pending(db, migrations)
    return {
        "current_version": current,
        "target_version": target,
        "up_to_date": len(todo) == 0,
        "pending": [{"version": m.version, "description": m.description} for m in todo],
    }
