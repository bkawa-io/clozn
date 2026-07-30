"""Hostile / edge-case pressure tests for six recently-shipped subsystems:

  1. clozn/runs/migrations.py -- transactional schema migrations
  2. clozn/runs/gc.py         -- content-addressed blob garbage collection
  3. clozn/server/request_context.py + clozn/server/sse.py + clozn/server/request_gate.py -- cancellation
  4. clozn/server/routes/engine.py's POST /cancel proxy + EngineClient.cancel() -- the gateway-exempt
     cancel path that lets a client interrupt a running generation without waiting behind POST_GATE
  5. clozn/server/routes/journal.py's GET /journal/calibration -- the TRUTH-tier reliability_bins and
     risk_coverage arrays recomputed live, on every request, from clozn/eval/calibration.py against
     whatever `rows` were last persisted to disk by `clozn eval --save`
  6. clozn/server/routes/engine.py's POST /cancel `req_id` resolution -- correlates the GATEWAY's own
     RequestContext.request_id, via the active substrate's live `_request`, to the WORKER's own req
     (`_request.engine_req`, stamped off the first parsed SSE frame -- see EngineSubstrate.chat_stream in
     clozn/server/substrates.py) before proxying the cancel, so a caller that only ever learned the
     gateway's own id can still interrupt the right in-flight generation

Goal: find real bugs, not confirm happy paths. Each section documents, in its own docstring, exactly what
adversarial scenario it drives and why. Where a test demonstrates a genuine finding (not just "this already
works"), the docstring says so explicitly -- see the module-level report this file's author returned
alongside it for a summary.

Model-free throughout: no GPU, no live engine, no real HTTP socket (section 4 drives the real do_POST
dispatch in-process via a handler stub, same technique as test_runtime_architecture.py's
raw_gateway_request -- no socket is ever opened).
"""
from __future__ import annotations

import io
import json
import os
import sqlite3
import sys
import threading
import time
from contextlib import closing

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "engine", "client"))

import clozn.runs.gc as gc                                   # noqa: E402
import clozn.runs.migrations as migrations                    # noqa: E402
import clozn.runs.store as store                               # noqa: E402
from clozn.eval import store as eval_store                      # noqa: E402
from clozn.server import app as cs                             # noqa: E402
from clozn.server import sse                                    # noqa: E402
from clozn.server.request_context import RequestContext, new_request_id  # noqa: E402
from clozn.server.request_gate import RequestGate               # noqa: E402
from clozn.server.routes import engine as engine_routes         # noqa: E402
from clozn.server.routes import journal as journal_routes       # noqa: E402
from clozn_engine import EngineClient                            # noqa: E402


# ======================================================================================================
# 1. MIGRATIONS
# ======================================================================================================

def test_rollback_survives_a_baseexception_not_just_a_normal_exception(tmp_path):
    """"Interrupted migration": migrate()'s except clause is `except BaseException`, which must also catch
    SystemExit/KeyboardInterrupt-style crashes, not merely ordinary Exception subclasses. Proves the
    rollback contract holds even for the crash-like exception classes a real process kill/Ctrl-C would
    raise inside a signal handler."""
    db = sqlite3.connect(str(tmp_path / "crash.sqlite3"))

    def _step1(conn):
        conn.execute("CREATE TABLE t1 (id INTEGER PRIMARY KEY)")

    def _step2_crashes(conn):
        conn.execute("CREATE TABLE t2_partial (id INTEGER PRIMARY KEY)")
        raise SystemExit(1)   # BaseException, NOT Exception

    fake = (
        migrations.Migration(1, "ok", _step1),
        migrations.Migration(2, "crashes like a killed process", _step2_crashes),
    )
    try:
        with pytest.raises(SystemExit):
            migrations.migrate(db, fake)
        assert migrations.current_version(db) == 1
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "t2_partial" not in tables
        # the connection must still be fully usable afterward -- not wedged mid-transaction
        db.execute("INSERT INTO t1 (id) VALUES (1)")
        db.commit()
    finally:
        db.close()


def test_corrupt_schema_meta_wrong_columns_fails_loudly_not_silently(tmp_path):
    """"Corrupt schema_meta table": a schema_meta that exists but doesn't even have a `key` column (as
    corrupt as on-disk state gets without the file itself being unreadable). `_ensure_ledger_table`'s
    CREATE TABLE IF NOT EXISTS is a no-op against the existing (wrong-shaped) table, so the very next SELECT
    must fail loudly with a clear sqlite3.OperationalError -- not silently return an empty/wrong version."""
    path = str(tmp_path / "corrupt.sqlite3")
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE schema_meta (not_key TEXT, not_value TEXT)")
    db.commit()
    try:
        with pytest.raises(sqlite3.OperationalError):
            migrations.current_version(db)
    finally:
        db.close()


def test_ledger_lying_about_an_applied_migration_is_caught_by_verify(tmp_path):
    """FIX VERIFIED: when the ledger claims migration 1 but the runs table is missing, verify detects
    the inconsistency -- status reports pending, and migrate() re-applies the migration."""
    path = str(tmp_path / "lying.sqlite3")
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("INSERT INTO schema_meta(key, value) VALUES ('migration:1', '{\"description\": \"fake\"}')")
    db.commit()
    try:
        report = migrations.status(db)
        assert report["up_to_date"] is False
        assert [step["version"] for step in report["pending"]] == [1, 2, 3, 4]

        applied = migrations.migrate(db)
        assert applied == [1, 2, 3, 4]

        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "runs" in tables
    finally:
        db.close()


def test_schema_meta_ignores_non_matching_or_garbled_migration_rows(tmp_path):
    """A `migration:` row with a non-integer suffix, and a row whose JSON value is garbage, must not crash
    current_version()/status() -- only the KEY's shape is examined, never the value's contents."""
    path = str(tmp_path / "garbled.sqlite3")
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("INSERT INTO schema_meta(key, value) VALUES ('migration:abc', 'not even json{{{')")
    db.execute("INSERT INTO schema_meta(key, value) VALUES ('migration:', 'also garbage')")
    db.commit()
    try:
        assert migrations.current_version(db) == 0     # neither row matches \d+ -- both ignored, no crash
        report = migrations.status(db)
        assert report["current_version"] == 0
    finally:
        db.close()


def test_duplicate_version_numbers_in_a_migration_list_are_rejected_upfront(tmp_path):
    """FIX VERIFIED: migrate() validates version uniqueness before running anything. Duplicate versions
    raise ValueError immediately — no callbacks run, no partial state."""
    db = sqlite3.connect(str(tmp_path / "dup.sqlite3"))
    ran = []
    fake = (
        migrations.Migration(1, "first with version 1", lambda c: ran.append("first")),
        migrations.Migration(1, "second, same version", lambda c: ran.append("second")),
    )
    try:
        with pytest.raises(ValueError, match="duplicate migration version"):
            migrations.migrate(db, fake)
        assert ran == []
    finally:
        db.close()


def test_migrations_list_out_of_declaration_order_still_applies_by_version_ascending(tmp_path):
    """pending() sorts by version, so a migration LIST authored out of order must still apply low-to-high,
    never in declaration order."""
    db = sqlite3.connect(str(tmp_path / "unordered.sqlite3"))
    ran = []
    fake = (
        migrations.Migration(3, "third", lambda c: ran.append(3)),
        migrations.Migration(1, "first", lambda c: ran.append(1)),
        migrations.Migration(2, "second", lambda c: ran.append(2)),
    )
    try:
        applied = migrations.migrate(db, fake)
        assert applied == [1, 2, 3]
        assert ran == [1, 2, 3]
    finally:
        db.close()


def test_migrate_on_an_in_memory_db_nobody_has_touched(tmp_path):
    """Brand-new DB, this time genuinely in-memory (":memory:") -- the module docstring explicitly calls
    this out ("even an in-memory DB nobody has touched")."""
    db = sqlite3.connect(":memory:")
    try:
        assert migrations.current_version(db) == 0
        applied = migrations.migrate(db)
        assert applied == [1, 2, 3, 4]
        assert migrations.current_version(db) == migrations.TARGET_VERSION
        assert migrations.migrate(db) == []       # already-latest: idempotent no-op
    finally:
        db.close()


def test_concurrent_migrate_calls_from_two_connections_can_raise_integrityerror(tmp_path, monkeypatch):
    """FINDING (real concurrency bug): migrate()'s `for m in pending(db, migrations):` evaluates `pending()`
    ONCE, up front, then only afterward takes the BEGIN IMMEDIATE write lock for the first step. Two
    independent connections to the SAME db file (e.g. a CLI invocation and the server starting up at the
    same instant, both calling store._ensure()) can each read `current_version() == 0` before either has
    taken the lock -- a snapshot-then-lock TOCTOU. This test reproduces that deterministically by widening
    the (normally microsecond) window with a monkeypatched delay inside connection A's `pending()` call --
    it does not fabricate a new code path, it just makes a real, existing race land reliably instead of
    requiring a many-run flake hunt.

    Result: connection B (fast) computes its own snapshot, wins the lock, and fully applies+commits
    migration 1 while A is still "asleep" inside its own (already-returned-a-stale-snapshot) pending() call.
    When A wakes up and finally takes the lock, it tries to INSERT a `migration:1` ledger row that's already
    there and gets an uncaught sqlite3.IntegrityError propagated all the way out of migrate() -- not
    retried, not recognized as "someone else already did this," not swallowed."""
    path = str(tmp_path / "race.sqlite3")
    db_a = sqlite3.connect(path, timeout=5.0, check_same_thread=False)
    db_b = sqlite3.connect(path, timeout=5.0, check_same_thread=False)

    real_pending = migrations.pending

    def _slow_for_a(db, migs=migrations.MIGRATIONS):
        result = real_pending(db, migs)
        if db is db_a:
            time.sleep(0.4)     # widen A's race window deterministically; B is unaffected
        return result

    monkeypatch.setattr(migrations, "pending", _slow_for_a)

    results: dict = {}
    errors: dict = {}

    def _run(tag, db):
        try:
            results[tag] = migrations.migrate(db)
        except Exception as exc:
            errors[tag] = exc

    t_a = threading.Thread(target=_run, args=("a", db_a))
    t_b = threading.Thread(target=_run, args=("b", db_b))
    t_a.start()
    time.sleep(0.1)   # let A enter migrate() -> pending(db_a) -> the 0.4s sleep, mid-call
    t_b.start()
    t_a.join(timeout=10)
    t_b.join(timeout=10)
    db_a.close()
    db_b.close()

    assert not errors, (
        f"expected concurrent migrate() calls to both succeed (the loser skips already-applied "
        f"migrations inside the write lock); errors={errors}"
    )
    check_db = sqlite3.connect(path)
    try:
        assert migrations.current_version(check_db) == migrations.TARGET_VERSION
    finally:
        check_db.close()
    all_applied = (results.get("a", []) or []) + (results.get("b", []) or [])
    assert 1 in all_applied


# ======================================================================================================
# 2. BLOB GC
# ======================================================================================================

@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


def _write_orphan_blob(digest: str, content: bytes = b'{"orphan": true}') -> str:
    path = store._blob_path(digest)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(content)
    return path


_FAKE_DIGEST_A = "a" * 64


def test_symlink_escaping_the_blob_root_is_refused_not_deleted(isolated, tmp_path):
    """Symlink attack: a "blob" that's actually a symlink pointing OUTSIDE the blob root (e.g. dropped by a
    broken install/migration script, or deliberately hostile). `_is_contained` resolves symlinks via
    os.path.realpath before checking containment -- must refuse to delete, and the real target file must
    never be touched. Skips gracefully if this environment can't create symlinks (common on Windows without
    Developer Mode / SeCreateSymbolicLinkPrivilege) rather than falsely reporting the protection as broken."""
    secret = tmp_path / "outside" / "secret.txt"
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text("do-not-delete-me", encoding="utf-8")

    digest = "c" * 64
    link_path = store._blob_path(digest)
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    try:
        os.symlink(str(secret), link_path)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlink creation unavailable in this environment: {exc}")

    result = gc.collect(dry_run=False)

    failed_digests = {e["digest"] for e in result["failed"]}
    assert digest in failed_digests
    entry = next(e for e in result["failed"] if e["digest"] == digest)
    assert "refused" in entry["error"]
    assert secret.exists()
    assert secret.read_text(encoding="utf-8") == "do-not-delete-me"
    assert os.path.lexists(link_path)         # the symlink itself untouched too
    assert {e["digest"] for e in result["deleted"]} == set()


def test_real_concurrent_write_lands_between_plan_and_delete_and_is_protected(isolated):
    """Concurrent write, driven through the REAL store.record() path end to end (not a monkeypatched
    canned return value like the existing suite's TOCTOU test): a run's blob becomes orphaned (its row is
    deleted, mimicking a prune/replace), then -- via a hook fired from inside the SECOND
    `_referenced_digests()` call collect() makes internally -- a NEW run lands referencing the exact same
    content-addressed blob (same trace content -> same sha256) in the narrow window between planning and
    deleting. The re-check must see it and protect the blob."""
    trace = {"tokens": ["race"], "confidence": [0.42]}
    r0 = store.record(source="cli", messages=[{"role": "user", "content": "orig"}],
                      response="landed-before-gc", trace=trace)
    with closing(store._connect()) as db:
        row = db.execute("SELECT payload_json FROM runs WHERE id = ?", (r0,)).fetchone()
    digest = json.loads(row["payload_json"])["trace_ref"]["sha256"]
    blob_path = store._blob_path(digest)
    assert os.path.isfile(blob_path)

    # orphan it: drop the referencing row (as store._prune()/replace_run() would leave behind)
    with closing(store._connect()) as db, db:
        db.execute("DELETE FROM runs WHERE id = ?", (r0,))

    real_referenced = gc._referenced_digests
    calls = {"n": 0}
    new_rid_holder = {}

    def _hooked(db):
        calls["n"] += 1
        result = real_referenced(db)
        if calls["n"] == 1:
            # a second thread lands a real run referencing the SAME content in the race window
            new_rid_holder["rid"] = store.record(
                source="cli", messages=[{"role": "user", "content": "race"}],
                response="landed-during-gc", trace=trace)
        return result

    import unittest.mock as mock
    with mock.patch.object(gc, "_referenced_digests", _hooked):
        result = gc.collect(dry_run=False)

    assert calls["n"] == 2                                   # both plan-phase and delete-phase reads ran
    assert new_rid_holder["rid"] is not None
    assert digest not in {e["digest"] for e in result["deleted"]}
    assert os.path.isfile(blob_path)                          # protected, not destroyed out from under itself


def test_corrupt_payload_json_row_does_not_crash_gc_and_orphans_its_blob(isolated):
    """"Corrupt DB row" analog to the migrations section's corrupt-ledger test: a run row whose
    payload_json is unparseable contributes nothing to the referenced set (per _referenced_digests' own
    `except Exception: continue`). Must not crash plan()/collect(), and the now-unreferenced blob becomes a
    legitimate delete candidate."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="hey",
                       trace={"tokens": ["a"], "confidence": [0.9]})
    with closing(store._connect()) as db:
        row = db.execute("SELECT payload_json FROM runs WHERE id = ?", (rid,)).fetchone()
    digest = json.loads(row["payload_json"])["trace_ref"]["sha256"]
    blob_path = store._blob_path(digest)
    assert os.path.isfile(blob_path)

    with closing(store._connect()) as db, db:
        db.execute("UPDATE runs SET payload_json = ? WHERE id = ?", ("{not valid json!!", rid))

    result = gc.collect(dry_run=False)          # must not raise

    assert digest in {e["digest"] for e in result["deleted"]}
    assert not os.path.isfile(blob_path)


def test_referenced_digests_ignores_a_hostile_non_hex_sha256_value(isolated):
    """Path traversal in a *referenced* digest value: a payload_json whose trace_ref.sha256 is
    attacker-shaped garbage (`"../../../../etc/passwd"`) must never be added to the referenced set --
    _referenced_digests' hex-shape check filters it out before it ever reaches path-building code."""
    store._ensure()
    rid = "run_hostile00000000000_abcdef"
    hostile_payload = json.dumps({"id": rid, "trace_ref": {"sha256": "../../../../etc/passwd"}})
    with closing(store._connect()) as db, db:
        db.execute(
            "INSERT INTO runs(id, created_ts, created_at, source, client, model, substrate, parent_run_id,"
            " finish_reason, error, prompt_summary, response_summary, duration_ms, payload_json)"
            " VALUES (?, 0, '', 'cli', 'unknown', '', '', NULL, NULL, NULL, '', '', 0, ?)",
            (rid, hostile_payload),
        )
    with closing(store._connect()) as db:
        referenced = gc._referenced_digests(db)
    assert referenced == set()


def test_uppercase_hex_digest_filename_is_malformed_not_a_deletion_candidate(isolated):
    """hashlib.sha256().hexdigest() only ever produces lowercase hex. A hand-dropped file using uppercase
    must be treated as malformed (never matched against the referenced set, never deleted) -- not silently
    lowercased and matched anyway."""
    root = store._blob_root()
    upper_dir = os.path.join(root, "AB")
    os.makedirs(upper_dir, exist_ok=True)
    path = os.path.join(upper_dir, "AB" + "C" * 62 + ".json")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("{}")

    result = gc.collect(dry_run=False)

    assert any(e["path"] == path for e in result["malformed"])
    assert all(e["path"] != path for e in result["delete"] + result["deleted"])
    assert os.path.isfile(path)


def test_blob_removed_by_another_process_between_plan_and_delete_is_reported_not_raised(isolated, monkeypatch):
    """A genuinely concurrent filesystem actor (a second GC run, an admin's `rm`, antivirus quarantine...)
    unlinks the orphan file in the narrow window between collect()'s plan and its own os.remove() call --
    the resulting FileNotFoundError must land in `failed`, never propagate as an uncaught exception."""
    orphan_path = _write_orphan_blob(_FAKE_DIGEST_A)
    real_referenced = gc._referenced_digests
    calls = {"n": 0}

    def _wrapped(db):
        calls["n"] += 1
        if calls["n"] == 2:
            try:
                os.remove(orphan_path)         # simulate the concurrent external unlink
            except OSError:
                pass
        return real_referenced(db)

    monkeypatch.setattr(gc, "_referenced_digests", _wrapped)
    result = gc.collect(dry_run=False)          # must not raise

    assert result["deleted"] == []
    failed_digests = {e["digest"] for e in result["failed"]}
    assert _FAKE_DIGEST_A in failed_digests
    entry = next(e for e in result["failed"] if e["digest"] == _FAKE_DIGEST_A)
    assert "Error" in entry["error"]


def test_plan_with_a_blob_root_that_was_never_created_is_empty_not_an_error(isolated, tmp_path):
    """Empty/nonexistent blob directory: plan(blob_root=...) pointed at a path with no ancestor directory
    ever created -- glob.glob on it must return [] cleanly, never raise."""
    nonexistent = str(tmp_path / "totally" / "does" / "not" / "exist")
    result = gc.plan(blob_root=nonexistent)
    assert result == {
        "blob_root": os.path.abspath(nonexistent),
        "referenced_count": 0,
        "total_blobs": 0,
        "keep": [],
        "delete": [],
        "malformed": [],
    }


def test_is_contained_false_across_different_drive_letters(tmp_path):
    """Windows-specific edge the code comments call out directly ("different drives on Windows -> not
    contained") but the existing suite never exercises with an actually different-looking drive letter --
    os.path.commonpath raises ValueError across drives, and _is_contained must turn that into False, not
    let the exception escape."""
    root = str(tmp_path / "blobs" / "sha256")
    os.makedirs(root, exist_ok=True)
    assert gc._is_contained(root, "Z:\\nonexistent\\evil.json") is False


def test_traversal_disguised_digest_filename_never_reaches_a_delete(isolated):
    """Path traversal in blob "names": even a filename that LOOKS like a real digest but is embedded in a
    hostile relative path shape is rejected before ever being treated as a delete candidate -- belt (regex
    shape check in _digest_from_path) and suspenders (_is_contained) both refuse independently."""
    root = store._blob_root()
    evil = os.path.join(root, "..", "..", "..", "etc", "passwd")
    assert gc._digest_from_path(root, evil) is None
    assert gc._is_contained(root, evil) is False


# ======================================================================================================
# 3. CANCELLATION
# ======================================================================================================

# ---------------------------------------------------------------------------------------- RequestContext

def test_double_cancel_is_idempotent():
    ctx = RequestContext()
    ctx.cancel()
    ctx.cancel()
    assert ctx.is_cancelled() is True


def test_cancel_before_anything_else_touches_the_context_is_fine():
    """Cancel before generation starts: nothing else populated yet -- cancelling must not disturb any other
    field's default."""
    ctx = RequestContext()
    ctx.cancel()
    assert ctx.is_cancelled() is True
    assert ctx.trace == []
    assert ctx.finish_reason is None
    assert ctx.diverged is None


def test_cancel_after_the_context_is_fully_populated_is_harmless():
    """Cancel after generation completes: a fully-populated, "finished" context can still be cancelled
    (nothing marks a context as immutable/closed) -- must not raise, and must not clobber the fields already
    written by the completed call."""
    ctx = RequestContext()
    ctx.finish_reason = "stop"
    ctx.trace = [{"piece": "hi"}]
    ctx.cancel()
    assert ctx.is_cancelled() is True
    assert ctx.finish_reason == "stop"        # cancelling after the fact doesn't retroactively erase this
    assert ctx.trace == [{"piece": "hi"}]


def test_two_independent_contexts_never_share_cancellation_state():
    """Directly exercises the module docstring's isolation claim: a caller holding a reference to an OLD
    context must never see a later, unrelated context's cancellation (or vice versa)."""
    old = RequestContext()
    new = RequestContext()
    old.cancel()
    assert old.is_cancelled() is True
    assert new.is_cancelled() is False


def test_rapid_fire_context_creation_yields_unique_request_ids():
    ids = {new_request_id() for _ in range(5000)}
    assert len(ids) == 5000
    assert all(rid.startswith("req_") for rid in ids)


def test_concurrent_cancel_and_is_cancelled_from_many_threads_never_raises():
    """Rapid-fire cancel/read hammering from many threads on ONE shared context."""
    ctx = RequestContext()
    errors = []

    def _hammer():
        try:
            for _ in range(500):
                ctx.cancel()
                ctx.is_cancelled()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert errors == []
    assert ctx.is_cancelled() is True


# ------------------------------------------------------------------------------------------------- sse.py

class FakeStreamSub:
    """Chat-stream double using a REAL RequestContext (not a stub) so its actual threading.Event-backed
    cancel()/is_cancelled() is exercised, not a hand-rolled stand-in."""

    def __init__(self, pieces, delay=0.0):
        self.pieces = list(pieces)
        self.delay = delay
        self._request = None
        self.closed = False
        self.yielded = []

    def chat_stream(self, messages, max_new, mem_out=None):
        self._request = RequestContext()
        try:
            for piece in self.pieces:
                if self._request.is_cancelled():
                    return
                if self.delay:
                    time.sleep(self.delay)
                if self._request.is_cancelled():
                    return
                self.yielded.append(piece)
                yield piece
        finally:
            self.closed = True

    def last_finish_reason(self):
        return "stop"

    def last_stream_trace(self):
        return []


class _FailingWfile:
    def __init__(self, fail_after):
        self.fail_after = fail_after
        self.calls = 0
        self._buf = []

    def write(self, b):
        self.calls += 1
        if self.calls > self.fail_after:
            raise BrokenPipeError("simulated client disconnect")
        self._buf.append(b)
        return len(b)

    def flush(self):
        pass

    def getvalue(self):
        return b"".join(self._buf)


class RecordingHandler:
    def __init__(self, wfile=None):
        self.code = None
        self.headers = {"Host": "localhost"}
        self.wfile = wfile if wfile is not None else _FailingWfile(fail_after=10 ** 9)
        self.log_calls = []

    def send_response(self, code):
        self.code = code

    def send_header(self, key, value):
        pass

    def end_headers(self):
        pass

    def _log_run(self, source, messages, response, model, started, error=None, trace=None,
                mem_out=None, finish_reason=None, finish_reason_fallback=None, extra_meta=None,
                sections=None):
        self.log_calls.append(dict(source=source, messages=messages, response=response, model=model,
                                   started=started, error=error, trace=trace, mem_out=mem_out,
                                   finish_reason=finish_reason,
                                   finish_reason_fallback=finish_reason_fallback, extra_meta=extra_meta,
                                   sections=sections))
        return "run_test"


def test_external_cancel_mid_stream_stops_generation_without_any_write_failure(monkeypatch):
    """Cancel during generation via a path OTHER than a client write failure: proves the substrate's own
    is_cancelled() check is honored generically. A background thread calls sub._request.cancel() directly
    (as a future explicit /cancel endpoint might) while every write to the "client" keeps succeeding --
    the stream must still stop early."""
    sub = FakeStreamSub(["a", "b", "c", "d", "e", "f", "g", "h"], delay=0.03)
    monkeypatch.setattr(cs, "SUB", sub)
    handler = RecordingHandler()

    def _cancel_soon():
        while sub._request is None:
            time.sleep(0.005)
        time.sleep(0.08)
        sub._request.cancel()

    t = threading.Thread(target=_cancel_soon)
    t.start()
    sse.sse_chat(handler, [{"role": "user", "content": "hi"}], 8, "m")
    t.join(timeout=10)

    assert sub.closed is True
    assert len(sub.yielded) < len(sub.pieces)       # stopped early -- did not run to completion
    assert len(handler.log_calls) == 1


def test_double_cancel_after_a_client_disconnect_does_not_raise_or_relog(monkeypatch):
    """Double-cancel: sse.py itself calls req.cancel() once on a detected disconnect. A SECOND, external
    cancel() call afterward (e.g. a lagging /cancel request racing the disconnect detection) must be a
    harmless no-op -- no exception, and no retroactive second log entry."""
    sub = FakeStreamSub(["Hel", "lo", "!"])
    monkeypatch.setattr(cs, "SUB", sub)
    handler = RecordingHandler(wfile=_FailingWfile(fail_after=2))

    sse.sse_chat(handler, [{"role": "user", "content": "hi"}], 8, "m")
    assert len(handler.log_calls) == 1
    assert sub._request.is_cancelled() is True

    sub._request.cancel()
    sub._request.cancel()
    assert sub._request.is_cancelled() is True
    assert len(handler.log_calls) == 1              # no retroactive re-log


def test_cancel_after_normal_completion_does_not_retroactively_change_the_log(monkeypatch):
    """Cancel after generation completes cleanly: calling cancel() on a context whose stream already
    finished and was logged must not alter anything already recorded."""
    sub = FakeStreamSub(["Hel", "lo", "!"])
    monkeypatch.setattr(cs, "SUB", sub)
    handler = RecordingHandler()

    sse.sse_chat(handler, [{"role": "user", "content": "hi"}], 8, "m")
    assert len(handler.log_calls) == 1
    logged = dict(handler.log_calls[0])

    sub._request.cancel()                            # hostile: cancel a call that's already fully done

    assert len(handler.log_calls) == 1
    assert handler.log_calls[0] == logged             # byte-for-byte unchanged


def test_rapid_fire_sequential_requests_do_not_cross_contaminate_cancellation(monkeypatch):
    """Rapid-fire cancel/new-request cycles: run many sse_chat() calls back-to-back, keeping a reference to
    every call's RequestContext. Cancelling ALL of them afterward -- in a batch, including doubled cancels
    -- must never have affected any call's own outcome (each context is a fresh object; publishing a new
    one never touches an old one, per the request_context.py docstring's isolation claim)."""
    contexts = []
    for i in range(20):
        sub = FakeStreamSub([f"tok{i}-a", f"tok{i}-b"])
        monkeypatch.setattr(cs, "SUB", sub)
        handler = RecordingHandler()
        sse.sse_chat(handler, [{"role": "user", "content": "hi"}], 8, "m")
        contexts.append(sub._request)
        assert len(handler.log_calls) == 1
        assert handler.log_calls[0]["response"] == f"tok{i}-atok{i}-b"
        assert handler.log_calls[0]["finish_reason"] == "stop"

    for ctx in contexts:
        ctx.cancel()
        ctx.cancel()

    assert all(c.is_cancelled() for c in contexts)
    assert len({c.request_id for c in contexts}) == 20   # every call got its own id, never reused


# --------------------------------------------------------------------------------------- RequestGate (POST_GATE)

def test_cancel_check_true_returns_cancelled_and_frees_the_slot_for_the_next_waiter():
    """POST_GATE waits are cancellable: a queued request whose cancel_check flips True must leave the
    queue promptly (bounded by poll_interval, not the full wait_timeout) and must give its slot back so a
    later request isn't starved by a cancelled one."""
    gate = RequestGate(capacity=2, wait_timeout=5.0)
    assert gate.acquire() is None            # request #1 takes the only "turn"

    calls = {"n": 0}

    def _cancel_on_second_poll():
        calls["n"] += 1
        return calls["n"] >= 2

    started = time.monotonic()
    result = gate.acquire(cancel_check=_cancel_on_second_poll, poll_interval=0.01)
    elapsed = time.monotonic() - started

    assert result == "cancelled"
    assert elapsed < 1.0                      # left promptly, not after the 5s wait_timeout

    gate.release()                            # request #1 finishes
    assert gate.acquire() is None             # the freed slot is usable -- nothing leaked by the cancel


def test_cancel_check_that_raises_does_not_leak_the_semaphore_permit():
    """FIX VERIFIED: a raising cancel_check propagates the exception but the semaphore permit is released
    in the finally block, so capacity is not permanently reduced."""
    gate = RequestGate(capacity=2, wait_timeout=1.0)
    assert gate.acquire() is None             # request A: holds the only "turn", deliberately never released yet

    def _boom():
        raise RuntimeError("adversarial cancel_check")

    with pytest.raises(RuntimeError, match="adversarial cancel_check"):
        gate.acquire(cancel_check=_boom, poll_interval=0.01)   # request B: crashes while queued behind A

    gate.release()                             # A finishes and releases perfectly cleanly

    # With the fix, full capacity (2) is available — B's slot was released in the finally block.
    # The gate serializes via a turn lock, so we acquire/release sequentially to prove both slots work.
    first = gate.acquire()
    assert first is None
    gate.release()
    second = gate.acquire()
    assert second is None, (
        "expected full capacity restored after the crashed cancel_check — the semaphore permit should "
        "have been released in the finally block"
    )
    gate.release()


def test_release_called_more_times_than_acquired_raises_not_silently_corrupts():
    """Hostile misuse: releasing more times than acquired. Documents current behavior -- it raises loudly
    (releasing an unlocked threading.Lock) rather than silently drifting the gate's internal accounting,
    which would be far worse (a quiet, undetectable capacity miscount)."""
    gate = RequestGate(capacity=2, wait_timeout=1.0)
    assert gate.acquire() is None
    gate.release()
    with pytest.raises(RuntimeError):
        gate.release()


def test_rapid_fire_concurrent_acquire_and_cancel_never_over_admits():
    """Real multi-threaded stress: many threads hammering acquire()/release(), a third of them with an
    always-true (non-raising) cancel_check, verifying the gate's own "one turn at a time" invariant holds
    under real concurrency, not just in a single-threaded trace."""
    gate = RequestGate(capacity=4, wait_timeout=3.0)
    max_active_seen = {"v": 0}
    errors = []
    lock = threading.Lock()

    def _worker(i):
        try:
            cancel = (lambda: True) if i % 3 == 0 else None
            result = gate.acquire(cancel_check=cancel, poll_interval=0.005)
            if result is None:
                with lock:
                    snap = gate.snapshot()
                    max_active_seen["v"] = max(max_active_seen["v"], snap["active"])
                time.sleep(0.01)
                gate.release()
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_worker, args=(i,)) for i in range(40)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)

    assert errors == []
    assert max_active_seen["v"] <= 1     # RequestGate serializes actual execution to exactly one at a time


# ======================================================================================================
# 4. GATEWAY /cancel PROXY (clozn/server/routes/engine.py + EngineClient.cancel)
# ======================================================================================================
#
# Two thin layers stacked on top of each other:
#   - EngineClient.cancel(req_id)  -- the SDK wrapper, POST /cancel {"req": req_id} to the C++ engine
#   - routes/engine.py's try_post("/cancel", ...) -- the gateway's proxy: ENGINE down -> 503, ENGINE.cancel
#     raising -> 502, otherwise the engine's JSON rides back verbatim. Registered in app._GATE_EXEMPT_POSTS
#     so it never sits behind POST_GATE's one-turn-at-a-time serialization -- the entire point of a cancel
#     path is to interrupt something that may currently be holding that very turn.

class _JSONCaptureHandler:
    """Minimal handler double: try_post only ever touches `_json` on the handler, never wfile/send_header/
    etc directly -- so this is all sections 1-3's request-shaped work needs. Whatever's captured here is
    exactly what a real client would receive as the HTTP status + JSON body."""

    def __init__(self):
        self.code = None
        self.body = None

    def _json(self, code, obj, extra_headers=None):
        self.code = code
        self.body = obj


def _raw_post(path, body_bytes=b"{}", headers=None):
    """Drive the REAL gateway handler's do_POST without opening a socket -- same technique as
    test_runtime_architecture.py's raw_gateway_request, trimmed to just POST since that's all the
    gate-exemption tests below need. Used only where the point of the test is the actual HTTP entry point
    (do_POST's _GATE_EXEMPT_POSTS check + _dispatch_post), not routes/engine.py's try_post in isolation."""
    handler_type = cs.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(body_bytes)
    handler.wfile = io.BytesIO()
    handler.headers = {"Content-Length": str(len(body_bytes)), "User-Agent": "pytest", **(headers or {})}
    handler.requestline = f"POST {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = "POST"
    handler.close_connection = False
    handler.do_POST()
    head, _, payload = handler.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode("latin-1"), payload


def test_cancel_proxy_503s_when_engine_is_none(monkeypatch):
    """No model worker connected (offline import, or the worker crashed mid-boot): the proxy must fail
    closed with a clean 503, never crash trying to call `.cancel()` on None."""
    monkeypatch.setattr(cs, "ENGINE", None)
    handler = _JSONCaptureHandler()

    claimed = engine_routes.try_post(handler, "/cancel", {"req": "req_1"})

    assert claimed is True
    assert handler.code == 503
    assert handler.body == {"error": "no engine connected"}


def test_cancel_proxy_502s_when_engine_cancel_raises(monkeypatch):
    """The engine process can be mid-crash, socket-reset, or otherwise unreachable when a cancel comes in
    -- ENGINE.cancel() raising must surface as a clean 502 with the exception message folded in, never an
    uncaught exception that takes the whole gateway request down."""
    class _BoomEngine:
        def cancel(self, req_id):
            raise RuntimeError("worker socket reset")

    monkeypatch.setattr(cs, "ENGINE", _BoomEngine())
    handler = _JSONCaptureHandler()

    claimed = engine_routes.try_post(handler, "/cancel", {"req": "req_2"})

    assert claimed is True
    assert handler.code == 502
    assert "engine cancel failed" in handler.body["error"]
    assert "worker socket reset" in handler.body["error"]


@pytest.mark.parametrize("body", [{}, {"req": ""}], ids=["missing_req_key", "explicit_empty_string"])
def test_cancel_with_missing_or_empty_req_field_calls_engine_with_empty_string(monkeypatch, body):
    """`body.get("req", "")` treats "no req key at all" and "req explicitly sent as ''" identically -- both
    must reach ENGINE.cancel("") rather than raising a KeyError or forwarding None."""
    calls = []

    class _FakeEngine:
        def cancel(self, req_id):
            calls.append(req_id)
            return {"cancelled": False, "req": req_id}

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine())
    handler = _JSONCaptureHandler()

    claimed = engine_routes.try_post(handler, "/cancel", body)

    assert claimed is True
    assert calls == [""]
    assert handler.code == 200
    assert handler.body == {"cancelled": False, "req": ""}


def test_cancel_with_explicit_null_req_normalizes_to_empty_string(monkeypatch):
    """FIX VERIFIED: `body.get("req") or ""` normalizes both absent-key AND explicit-null to "". A body of
    `{"req": null}` (a plausible client bug -- e.g. a frontend that hasn't populated its last request id
    yet) now hits ENGINE.cancel(""), identical to a missing key -- not ENGINE.cancel(None)."""
    calls = []

    class _FakeEngine:
        def cancel(self, req_id):
            calls.append(req_id)
            return {"cancelled": False, "req": req_id}

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine())
    handler = _JSONCaptureHandler()

    claimed = engine_routes.try_post(handler, "/cancel", {"req": None})

    assert claimed is True
    assert calls == [""]            # normalized to "" -- None no longer leaks through
    assert handler.code == 200


def test_cancel_with_a_valid_looking_req_id_returns_the_engines_result_verbatim(monkeypatch):
    """The proxy is a pure pass-through: whatever ENGINE.cancel() returns rides back as the response body
    unmodified -- including the honest {"cancelled": false, ...} for an unknown or already-finished id
    (cancel is documented idempotent: it never 404s on an unrecognized id, it just reports it didn't
    cancel anything)."""
    class _FakeEngine:
        def cancel(self, req_id):
            assert req_id == "req_a1b2c3"
            return {"cancelled": False, "req": req_id}

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine())
    handler = _JSONCaptureHandler()

    claimed = engine_routes.try_post(handler, "/cancel", {"req": "req_a1b2c3"})

    assert claimed is True
    assert handler.code == 200
    assert handler.body == {"cancelled": False, "req": "req_a1b2c3"}


def test_cancel_is_gate_exempt_the_post_gate_is_never_touched(monkeypatch):
    """Drives the REAL do_POST dispatch (not routes/engine.py's try_post in isolation) to prove /cancel is
    gate-exempt at the actual HTTP entry point -- mirrors test_runtime_architecture.py's
    PostGateScopeTests pattern used for app._GATE_EXEMPT_POSTS' other two paths. POST_GATE.acquire must
    never even be CALLED for /cancel, let alone block on it."""
    class _FakeEngine:
        def cancel(self, req_id):
            return {"cancelled": False, "req": req_id}

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine())
    calls = []
    original_acquire = cs.POST_GATE.acquire
    monkeypatch.setattr(cs.POST_GATE, "acquire",
                        lambda *a, **kw: calls.append(1) or original_acquire(*a, **kw))

    head, payload = _raw_post("/cancel", b'{"req": "req_x"}')

    assert " 200 " in head
    assert json.loads(payload) == {"cancelled": False, "req": "req_x"}
    assert calls == []          # the gate was never even asked


def test_concurrent_cancel_fires_while_a_slow_generation_holds_the_post_gate(monkeypatch):
    """Race probe: this is the actual reason /cancel needs to be gate-exempt. While a slow generation holds
    POST_GATE's one "turn" (a real in-flight, e.g., /engine/chat-shaped call), a concurrent /cancel for
    that SAME request must still reach ENGINE.cancel() promptly -- if /cancel were NOT exempt, it would
    queue behind the very call it's trying to interrupt and could never actually cancel anything still
    running. Uses the module's REAL cs.POST_GATE, held busy by a second thread, and the REAL do_POST
    dispatch (not a mocked acquire) -- a genuine concurrency exercise, not just a code-path check."""
    calls = []

    class _FakeEngine:
        def cancel(self, req_id):
            calls.append(req_id)
            return {"cancelled": True, "req": req_id}

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine())
    release_holder = threading.Event()

    def _hold_the_gate():
        assert cs.POST_GATE.acquire() is None      # simulates a slow generation taking the one turn
        release_holder.wait(timeout=5)
        cs.POST_GATE.release()

    holder = threading.Thread(target=_hold_the_gate)
    holder.start()
    try:
        deadline = time.monotonic() + 2.0           # wait for the holder to actually take the turn
        while cs.POST_GATE.snapshot()["active"] == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert cs.POST_GATE.snapshot()["active"] == 1, "holder thread never actually acquired the gate"

        started = time.monotonic()
        head, payload = _raw_post("/cancel", b'{"req": "req_race"}')
        elapsed = time.monotonic() - started
    finally:
        release_holder.set()
        holder.join(timeout=5)

    assert " 200 " in head
    assert elapsed < 1.0                             # never queued behind the busy gate
    assert calls == ["req_race"]
    assert json.loads(payload) == {"cancelled": True, "req": "req_race"}


def test_engine_client_cancel_posts_req_id_under_req_key_verbatim(monkeypatch):
    """EngineClient.cancel() is a thin wire-format contract: `req_id` must land under the "req" key at
    POST /cancel, unmodified -- no coercion, no extra fields, nothing silently dropped or renamed. Mirrors
    test_engine_apply_template.py's style for the sibling apply_template() wire test."""
    seen = {}
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post",
                        lambda path, body: seen.update(path=path, body=body) or
                        {"cancelled": False, "req": body["req"]})

    out = ec.cancel("req_deadbeef")

    assert seen["path"] == "/cancel"
    assert seen["body"] == {"req": "req_deadbeef"}
    assert out == {"cancelled": False, "req": "req_deadbeef"}


# ======================================================================================================
# 5. TRUTH-TIER CALIBRATION ENDPOINT (GET /journal/calibration -- reliability_bins + risk_coverage)
# ======================================================================================================
#
# clozn/server/routes/journal.py's try_get() recomputes calibration.ece(pairs)["bins"] and
# calibration.risk_coverage(pairs) live, on every request, from whatever `rows` were persisted by the
# last `clozn eval --save`. That saved payload is untrusted as far as this route is concerned -- it was
# written by a prior process, possibly a prior version of the eval harness, and nothing re-validates its
# shape on the read path. These tests hammer the row-to-pairs extraction (`r["score"], r["correct"]`,
# gated on `r.get(...) is not None`) in journal.py and calibration.py's own `_clean()` against missing,
# empty, and hostile row shapes -- using the same `_JSONCaptureHandler` + monkeypatch(eval_store.load)
# pattern as section 4.

def test_calibration_no_saved_report_returns_available_false_with_no_bins(monkeypatch):
    """No `clozn eval --save` has ever run (or the file was deleted/corrupted): eval_store.load() returns
    None, and the route must answer available:false with neither reliability_bins nor risk_coverage
    present at all -- not an empty list standing in for "we computed this and there was nothing", which
    would be a different, stronger claim than "nothing was ever saved"."""
    monkeypatch.setattr(eval_store, "load", lambda *a, **kw: None)
    h = _JSONCaptureHandler()

    claimed = journal_routes.try_get(h, "/journal/calibration")

    assert claimed is True
    assert h.code == 200
    assert h.body["available"] is False
    assert "reliability_bins" not in h.body
    assert "risk_coverage" not in h.body


def test_calibration_valid_rows_produce_well_formed_reliability_bins_and_risk_coverage(monkeypatch):
    """Happy-path shape check pinned to exact numbers, not just "is a non-empty list": six rows spread
    across six distinct 10%-wide bins, each with n=1, so every populated bin's mean_score/accuracy/gap is
    exactly that one row's own values -- and the risk-coverage sweep is hand-verified point by point. This
    locks down both the ReliabilityBin and CoveragePoint dataclass field names as they ride the wire."""
    rows = [
        {"score": 0.95, "correct": True},
        {"score": 0.85, "correct": True},
        {"score": 0.75, "correct": False},
        {"score": 0.55, "correct": True},
        {"score": 0.35, "correct": False},
        {"score": 0.15, "correct": False},
    ]
    monkeypatch.setattr(eval_store, "load", lambda *a, **kw: {"saved_ts": 100.0, "rows": rows})
    h = _JSONCaptureHandler()

    claimed = journal_routes.try_get(h, "/journal/calibration")

    assert claimed is True
    assert h.body["available"] is True
    bins = h.body["reliability_bins"]
    assert len(bins) == 10                                          # fixed n_bins=10, empty bins included
    for b in bins:
        assert set(b.keys()) == {"lo", "hi", "n", "mean_score", "accuracy", "gap", "ci_lo", "ci_hi"}
    populated = {round(b["lo"], 2): b for b in bins if b["n"] > 0}
    assert set(populated) == {0.1, 0.3, 0.5, 0.7, 0.8, 0.9}
    assert populated[0.9]["mean_score"] == pytest.approx(0.95) and populated[0.9]["accuracy"] == 1.0
    assert populated[0.7]["accuracy"] == 0.0 and populated[0.7]["gap"] == pytest.approx(0.75)

    rc = h.body["risk_coverage"]
    assert len(rc) == 6
    for pt in rc:
        assert set(pt.keys()) == {"threshold", "coverage", "error", "n_answered"}
    assert [pt["n_answered"] for pt in rc] == [1, 2, 3, 4, 5, 6]
    assert rc[0]["error"] == 0.0                                    # top-scored item (.95) was correct
    assert rc[2]["error"] == pytest.approx(1 / 3)                   # 3 answered, 1 wrong so far (.75)
    assert rc[-1]["coverage"] == 1.0 and rc[-1]["error"] == 0.5


def test_calibration_empty_rows_list_is_available_true_with_no_bins_key(monkeypatch):
    """A saved report whose `rows` is present but an empty list (e.g. `clozn eval --save` ran against a
    zero-item probe set): `rows` is falsy, so the route's `if rows:` guard skips the whole recompute block
    entirely -- available stays true (a report WAS saved), but reliability_bins/risk_coverage must not
    appear at all, not as empty lists silently standing in for "nothing to show"."""
    monkeypatch.setattr(eval_store, "load", lambda *a, **kw: {"saved_ts": 100.0, "rows": []})
    h = _JSONCaptureHandler()

    claimed = journal_routes.try_get(h, "/journal/calibration")

    assert claimed is True
    assert h.body["available"] is True
    assert "reliability_bins" not in h.body
    assert "risk_coverage" not in h.body


def test_calibration_rows_with_none_or_missing_score_or_correct_are_dropped_not_crashed(monkeypatch):
    """Two hostile-but-plausible row shapes in one list: an explicit `"score": None` (an ungradeable item
    upstream) and a row missing the "correct" key entirely (a truncated write, or an older schema version).
    Both must be dropped by the `r.get(...) is not None` gate in journal.py BEFORE the `r["score"]`/
    `r["correct"]` subscript ever runs -- a naive `r["correct"]` on the second row would KeyError. Because
    `rows` itself is non-empty, the recompute block still runs; with zero valid pairs surviving the filter
    it must still emit 10 empty bins (not crash, not silently drop the key) and an empty risk_coverage."""
    rows = [{"score": None, "correct": True}, {"score": 0.5}]
    monkeypatch.setattr(eval_store, "load", lambda *a, **kw: {"saved_ts": 100.0, "rows": rows})
    h = _JSONCaptureHandler()

    claimed = journal_routes.try_get(h, "/journal/calibration")

    assert claimed is True
    assert h.body["available"] is True
    bins = h.body["reliability_bins"]
    assert len(bins) == 10
    assert all(b["n"] == 0 for b in bins)
    assert all(b["mean_score"] is None and b["accuracy"] is None and b["gap"] is None for b in bins)
    assert h.body["risk_coverage"] == []


def test_calibration_non_bool_truthy_correct_values_silently_count_as_correct(monkeypatch):
    """FINDING: calibration.py's `_clean()` docstring says `correct` is "coercible to bool" -- but Python's
    bool() coerces ANY non-empty string, including the literal string "false", to True. A row written as
    {"score": 0.9, "correct": "false"} (plausible from an eval harness that stringifies a Python bool
    somewhere upstream, or a lossy CSV/JSON round-trip) is silently counted as a CORRECT answer here, not
    dropped and not flagged. This pins down that current, surprising behavior so a future change to
    `_clean()`'s coercion rule is a deliberate decision, not an accidental regression discovered in prod."""
    rows = [
        {"score": 0.9, "correct": "false"},
        {"score": 0.9, "correct": "no"},
        {"score": 0.9, "correct": True},
    ]
    monkeypatch.setattr(eval_store, "load", lambda *a, **kw: {"saved_ts": 100.0, "rows": rows})
    h = _JSONCaptureHandler()

    claimed = journal_routes.try_get(h, "/journal/calibration")

    assert claimed is True
    top_bin = h.body["reliability_bins"][-1]
    assert top_bin["n"] == 3
    assert top_bin["accuracy"] == 1.0          # all three -- including the two string rows -- count correct
    assert top_bin["mean_score"] == pytest.approx(0.9)


@pytest.mark.parametrize("all_correct", [True, False], ids=["all_correct", "all_wrong"])
def test_calibration_all_correct_or_all_wrong_rows_do_not_crash_ece_or_risk_coverage(monkeypatch, all_correct):
    """Single-outcome-class edge case: calibration.py's own fit_temperature() explicitly refuses an
    all-right/all-wrong probe set (needs both outcomes to identify a transform). GET /journal/calibration
    never calls fit_temperature() itself -- only ece()/risk_coverage() -- so this proves THOSE two stay
    well-behaved (no division-by-zero, no crash) even at that single-class extreme, which any future change
    wiring temperature fitting into this same route would need to preserve."""
    rows = [{"score": s, "correct": all_correct} for s in (0.1, 0.3, 0.5, 0.7, 0.9)]
    monkeypatch.setattr(eval_store, "load", lambda *a, **kw: {"saved_ts": 100.0, "rows": rows})
    h = _JSONCaptureHandler()

    claimed = journal_routes.try_get(h, "/journal/calibration")

    assert claimed is True
    populated = [b for b in h.body["reliability_bins"] if b["n"] > 0]
    assert len(populated) == 5
    expected_acc = 1.0 if all_correct else 0.0
    assert all(b["accuracy"] == expected_acc for b in populated)

    rc = h.body["risk_coverage"]
    assert len(rc) == 5
    expected_error = 0.0 if all_correct else 1.0
    assert all(pt["error"] == expected_error for pt in rc)


def test_calibration_rows_key_absent_entirely_behaves_like_empty_rows(monkeypatch):
    """A saved report from before `rows` existed in the schema, or a hand-edited/half-written file missing
    the key entirely: `out.get("rows") or []` must treat a missing key identically to an empty list, never
    KeyError and never diverge in behavior from the empty-list case. available stays true (a report WAS
    saved) and saved_ago_s is still computed from the given saved_ts alone."""
    monkeypatch.setattr(eval_store, "load", lambda *a, **kw: {"saved_ts": 123.0})
    h = _JSONCaptureHandler()

    claimed = journal_routes.try_get(h, "/journal/calibration")

    assert claimed is True
    assert h.body["available"] is True
    assert "reliability_bins" not in h.body
    assert "risk_coverage" not in h.body
    assert h.body["saved_ago_s"] >= 0


# ======================================================================================================
# 6. REQ-ID CANCEL CORRELATION (clozn/server/routes/engine.py's POST /cancel `req_id` resolution)
# ======================================================================================================
#
# The NEW layer on top of section 4's plain `req` proxy: a caller that only ever learned the GATEWAY's
# own RequestContext.request_id (e.g. from run_meta, never having seen the worker's own wire-level id)
# can still cancel the right in-flight generation. try_post's exact logic (routes/engine.py):
#
#   engine_req = str(body.get("req") or "")
#   req_id = str(body.get("req_id") or "")
#   if req_id and not engine_req:                       # only resolve when `req` is absent/empty
#       sub = ctx.active_sub(h)
#       current = getattr(sub, "_request", None) if sub is not None else None
#       if current is not None and current.request_id == req_id:
#           current.cancel()                             # local stop signal fires UNCONDITIONALLY on match
#           engine_req = current.engine_req or ""
#       if not engine_req:                               # unmatched req_id, or matched but no worker req yet
#           h._json(200, {"cancelled": False, "req": req_id})   # honest no-op, ENGINE.cancel never called
#           return True
#   result = ctx.ENGINE.cancel(engine_req)               # otherwise: normal proxy, engine's result verbatim
#
# `active_sub(h)` resolves to `cs.SUB` for a handler with no `_inj_sub` (clozn/server/app.py), so these
# tests monkeypatch cs.SUB directly -- same technique section 3's FakeStreamSub tests use.

def test_req_id_matching_active_request_resolves_to_engine_req(monkeypatch):
    """The core correlation: `req_id` matches the active substrate's live RequestContext.request_id, which
    already carries the worker's own req (`_request.engine_req`, stamped off the first SSE frame by
    chat_stream) -- THAT is what reaches ENGINE.cancel(), never the gateway id itself, and the engine's
    own result rides back verbatim (this is section 4's proxy contract, reached via a different route in)."""
    calls = []

    class _FakeEngine:
        def cancel(self, req_id):
            calls.append(req_id)
            return {"cancelled": True, "req": req_id}

    class _FakeSub:
        pass

    sub = _FakeSub()
    sub._request = RequestContext()
    sub._request.request_id = "req_abc"
    sub._request.engine_req = "engine_xyz"

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine())
    monkeypatch.setattr(cs, "SUB", sub)
    handler = _JSONCaptureHandler()

    claimed = engine_routes.try_post(handler, "/cancel", {"req_id": "req_abc"})

    assert claimed is True
    assert calls == ["engine_xyz"]
    assert handler.code == 200
    assert handler.body == {"cancelled": True, "req": "engine_xyz"}
    assert sub._request.is_cancelled() is True     # the local stop signal fired too, not just the proxy


def test_req_id_match_fires_local_cancel_even_when_engine_req_is_none(monkeypatch):
    """The generation just started -- chat_stream hasn't parsed its first SSE frame yet, so
    `_request.engine_req` is still None (no worker id to correlate to). The match must still fire the
    LOCAL stop signal (current.cancel()) unconditionally -- chat_stream's read loop polls is_cancelled()
    independent of any worker-side cancel -- but there is nothing to hand the worker's own /cancel, so the
    response honestly reports nothing was forwarded rather than proxying an empty id (which the worker
    itself 400s on)."""
    calls = []

    class _FakeEngine:
        def cancel(self, req_id):
            calls.append(req_id)                 # must never be reached in this scenario
            return {"cancelled": True, "req": req_id}

    class _FakeSub:
        pass

    sub = _FakeSub()
    sub._request = RequestContext()
    sub._request.request_id = "req_abc"
    # engine_req stays at its dataclass default (None) -- no SSE frame parsed yet

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine())
    monkeypatch.setattr(cs, "SUB", sub)
    handler = _JSONCaptureHandler()

    claimed = engine_routes.try_post(handler, "/cancel", {"req_id": "req_abc"})

    assert claimed is True
    assert calls == []                            # nothing to forward -- ENGINE.cancel is never called
    assert handler.code == 200
    assert handler.body == {"cancelled": False, "req": "req_abc"}
    assert sub._request.is_cancelled() is True     # but the local stop signal DID fire


def test_req_id_not_matching_the_active_request_is_a_noop(monkeypatch):
    """A stale/unknown req_id: the gateway id doesn't match the CURRENT live request (already finished, or
    from a since-replaced RequestContext -- a new call always publishes its own fresh object per
    request_context.py's isolation guarantee). Must be a clean no-op: no ENGINE.cancel call, and -- just as
    important -- the CURRENT (unrelated) request's own cancellation state must be left completely alone."""
    calls = []

    class _FakeEngine:
        def cancel(self, req_id):
            calls.append(req_id)
            return {"cancelled": True, "req": req_id}

    class _FakeSub:
        pass

    sub = _FakeSub()
    sub._request = RequestContext()
    sub._request.request_id = "req_abc"
    sub._request.engine_req = "engine_xyz"

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine())
    monkeypatch.setattr(cs, "SUB", sub)
    handler = _JSONCaptureHandler()

    claimed = engine_routes.try_post(handler, "/cancel", {"req_id": "req_different"})

    assert claimed is True
    assert calls == []
    assert handler.code == 200
    assert handler.body == {"cancelled": False, "req": "req_different"}
    assert sub._request.is_cancelled() is False    # the unrelated, still-live request is untouched


def test_req_id_with_no_active_substrate_degrades_cleanly(monkeypatch):
    """No active substrate at all (e.g. between a substrate teardown and the next one coming up): must
    degrade to a clean, honest no-op -- never crash trying to read `._request` off None."""
    class _FakeEngine:
        def cancel(self, req_id):
            raise AssertionError("ENGINE.cancel must not be called when there is no active substrate")

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine())
    monkeypatch.setattr(cs, "SUB", None)
    handler = _JSONCaptureHandler()

    claimed = engine_routes.try_post(handler, "/cancel", {"req_id": "req_abc"})

    assert claimed is True
    assert handler.code == 200
    assert handler.body == {"cancelled": False, "req": "req_abc"}


def test_req_id_with_substrate_missing_the_request_attribute_degrades_cleanly(monkeypatch):
    """A substrate exists but has never handled a single generation yet (fresh construction, before its
    first chat()/chat_stream() call ever ran self._new_request()) -- `._request` doesn't exist at all, not
    even as None. `getattr(sub, "_request", None)` must absorb this cleanly rather than raising
    AttributeError."""
    class _FakeEngine:
        def cancel(self, req_id):
            raise AssertionError("ENGINE.cancel must not be called with no _request on the substrate")

    class _FakeSub:
        pass       # deliberately no ._request attribute at all

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine())
    monkeypatch.setattr(cs, "SUB", _FakeSub())
    handler = _JSONCaptureHandler()

    claimed = engine_routes.try_post(handler, "/cancel", {"req_id": "req_abc"})

    assert claimed is True
    assert handler.code == 200
    assert handler.body == {"cancelled": False, "req": "req_abc"}


def test_req_and_req_id_both_provided_req_wins(monkeypatch):
    """The pre-existing `req` key takes precedence: `req_id` is only ever consulted when `req` is
    absent/empty (`if req_id and not engine_req:`). A body sending BOTH must forward the raw `req` verbatim
    and never even attempt req_id resolution -- proven here by making the substrate's live request_id a
    value that WOULD match req_id if it were (wrongly) consulted, and asserting neither the substrate's
    engine_req nor its cancel() got touched."""
    calls = []

    class _FakeEngine:
        def cancel(self, req_id):
            calls.append(req_id)
            return {"cancelled": True, "req": req_id}

    class _FakeSub:
        pass

    sub = _FakeSub()
    sub._request = RequestContext()
    sub._request.request_id = "req_abc"                  # would match req_id below, if consulted
    sub._request.engine_req = "engine_should_not_be_used"

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine())
    monkeypatch.setattr(cs, "SUB", sub)
    handler = _JSONCaptureHandler()

    claimed = engine_routes.try_post(handler, "/cancel", {"req": "raw_worker_id", "req_id": "req_abc"})

    assert claimed is True
    assert calls == ["raw_worker_id"]                     # the pre-existing `req` key wins, unmodified
    assert handler.code == 200
    assert handler.body == {"cancelled": True, "req": "raw_worker_id"}
    assert sub._request.is_cancelled() is False           # req_id resolution never ran -- no local cancel


def test_req_id_empty_string_is_treated_as_absent(monkeypatch):
    """`str(body.get("req_id") or "")` normalizes an explicit empty string the same as a missing key -- the
    resolution branch (`if req_id and not engine_req:`) must not even be entered, so it must never touch
    the active substrate at all. Proven hostilely: cs.SUB is a substrate whose `_request` property raises
    if ever read -- if resolution were mistakenly attempted, this test would fail on that exception rather
    than merely on a wrong response body."""
    calls = []

    class _FakeEngine:
        def cancel(self, req_id):
            calls.append(req_id)
            return {"cancelled": False, "req": req_id}

    class _MustNotBeTouchedSub:
        @property
        def _request(self):
            raise AssertionError("resolution must never even look at the active substrate for an empty req_id")

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine())
    monkeypatch.setattr(cs, "SUB", _MustNotBeTouchedSub())
    handler = _JSONCaptureHandler()

    claimed = engine_routes.try_post(handler, "/cancel", {"req_id": ""})

    assert claimed is True
    assert calls == [""]                     # falls through to the pre-existing req path, empty like a missing req
    assert handler.code == 200
    assert handler.body == {"cancelled": False, "req": ""}


def test_concurrent_engine_req_arriving_mid_cancel_never_crashes_or_double_forwards(monkeypatch):
    """Real-thread race, the scenario request_context.py's docstring calls out by name: a background thread
    plays chat_stream()'s role, stamping `_request.engine_req` off "the first SSE frame" at an
    unpredictable moment, concurrently with many /cancel calls resolving req_id against that SAME live,
    already-published RequestContext. Not a torn read -- both sides are ordinary attribute ops on one
    object, GIL-atomic -- but a genuine race in wall-clock terms the handler must survive regardless of
    which side of the write each call lands on: never raise, never forward a stale/partial id, and every
    matching call still fires the local cancel signal."""
    calls = []
    call_lock = threading.Lock()

    class _FakeEngine:
        def cancel(self, req_id):
            with call_lock:
                calls.append(req_id)
            return {"cancelled": True, "req": req_id}

    class _FakeSub:
        pass

    sub = _FakeSub()
    sub._request = RequestContext()
    sub._request.request_id = "req_abc"      # engine_req starts None, like a call that just began

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine())
    monkeypatch.setattr(cs, "SUB", sub)

    def _stamp_engine_req_soon():
        time.sleep(0.01)
        sub._request.engine_req = "engine_landed_late"

    stamper = threading.Thread(target=_stamp_engine_req_soon)
    stamper.start()

    results = []
    for _ in range(30):
        handler = _JSONCaptureHandler()
        engine_routes.try_post(handler, "/cancel", {"req_id": "req_abc"})   # must never raise
        results.append((handler.code, handler.body))
        time.sleep(0.001)
    stamper.join(timeout=5)

    assert all(code == 200 for code, _ in results)
    for _, body in results:
        # every response is internally consistent -- either the honest "nothing to forward yet" no-op, or
        # a forward of the (by-then-stamped) real worker id -- never a mix, never a crash
        if body.get("cancelled") is False:
            assert body == {"cancelled": False, "req": "req_abc"}
        else:
            assert body == {"cancelled": True, "req": "engine_landed_late"}
    assert set(calls) <= {"engine_landed_late"}     # never forwards a bogus/partial id
    assert sub._request.is_cancelled() is True       # local stop signal fired on every one of these matches
