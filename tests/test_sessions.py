"""Contract tests for F1: first-class session records (clozn/runs/sessions.py, migration 4, the
`clozn.session.v1` schema, and clozn/server/routes/sessions.py).

NOT tests/test_sections.py -- that file covers prompt-section ablation (clozn.runs.sections), an
unrelated, pre-existing module this suite does not touch.
"""
from __future__ import annotations

import io
import json
import os
import sys
import threading
from contextlib import closing

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clozn.runs.migrations as migrations         # noqa: E402
import clozn.runs.store as store                    # noqa: E402
from clozn.runs import sessions                     # noqa: E402
from clozn.runs.association import client_key, session_key  # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Redirect the run store to a temp dir and clear the per-process schema-verified cache -- same
    isolation contract as tests/test_runs_store_concurrency.py's `isolated` fixture."""
    runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(store, "RUNS_DIR", runs_dir)
    store._schema_verified.clear()
    yield runs_dir
    store._schema_verified.clear()


def _run(*, session=None, client=None, source="cli", prompt="hi", response="ok", started=1.0):
    return store.record(
        source=source, client="tester", client_key=client_key(client) if client else None,
        session_key=session_key(session) if session else None,
        messages=[{"role": "user", "content": prompt}], response=response, started=started,
    )


def _run_at(recorded_ts: float, *, session=None, client=None):
    """A run whose `recorded_ts` (the actual insertion-order/cursor column -- NOT `started`, which only
    seeds `created_ts`; `store.record()` always stamps `recorded_ts` with the wall-clock time of the
    call) is pinned to an exact, deterministic value, by writing it back through `store.replace_run`.
    Ordering tests use this instead of relying on the wall-clock gap between successive `_run()` calls."""
    rid = _run(session=session, client=client)
    rec = store.get_run(rid)
    rec["recorded_ts"] = recorded_ts
    assert store.replace_run(rec)
    return rid


# ============================================================================================ migration

def test_migration_4_creates_the_sessions_table(isolated):
    db_path = os.path.join(store.RUNS_DIR, "runs.sqlite3")
    store._ensure()
    import sqlite3
    db = sqlite3.connect(db_path)
    try:
        assert migrations.current_version(db) == migrations.TARGET_VERSION
        assert migrations.TARGET_VERSION >= 4
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "sessions" in tables
        columns = {r[1] for r in db.execute("PRAGMA table_info(sessions)")}
        assert {"id", "created_ts", "created_at", "client_key", "title", "privacy_json",
                "materialized_from"}.issubset(columns)
    finally:
        db.close()


def test_migration_4_does_not_touch_existing_run_rows(isolated):
    """RUN IMMUTABILITY: applying migration 4 to a DB that already has run rows leaves them byte-for-byte
    unchanged (it only creates a brand-new table -- no ALTER TABLE runs, no data migration)."""
    rid = _run(session="thread-1")
    before = store.get_run(rid)
    store._ensure()
    after = store.get_run(rid)
    assert before == after


# ==================================================================================== identity / resolution

def test_resolve_session_id_mints_a_fresh_opaque_id_when_none():
    a = sessions.resolve_session_id(None)
    b = sessions.resolve_session_id(None)
    assert a != b
    assert a.startswith("session_") and b.startswith("session_")
    import re
    assert re.fullmatch(r"session_[0-9a-f]{24}", a)


def test_resolve_session_id_digests_a_raw_token_deterministically():
    a = sessions.resolve_session_id("my-conversation-42")
    b = sessions.resolve_session_id("my-conversation-42")
    assert a == b
    assert a == session_key("my-conversation-42")


def test_resolve_session_id_different_raw_tokens_yield_different_ids():
    a = sessions.resolve_session_id("thread-a")
    b = sessions.resolve_session_id("thread-b")
    assert a != b


def test_resolve_session_id_accepts_an_already_opaque_key_as_is():
    minted = sessions.resolve_session_id(None)
    reused = sessions.resolve_session_id(minted)
    assert reused == minted


def test_resolve_session_id_rejects_empty_string():
    with pytest.raises(sessions.SessionValueError):
        sessions.resolve_session_id("   ")


# ================================================================================================ create

def test_create_session_minimal(isolated):
    doc = sessions.create_session()
    assert doc["schema_version"] == "clozn.session.v1"
    assert doc["materialized_from"] == "explicit"
    assert doc["privacy"] == {"visibility": "visible"}
    assert "client_key" not in doc
    assert "title" not in doc
    assert "first_activity_ts" not in doc          # no runs yet -- omitted, never zero-padded


def test_create_session_with_client_and_title(isolated):
    doc = sessions.create_session("thread-1", client_id="cli-wrapper", title="  Debug run  ",
                                  visibility="hidden")
    assert doc["client_key"] == client_key("cli-wrapper")
    assert doc["title"] == "Debug run"
    assert doc["privacy"] == {"visibility": "hidden"}
    assert doc["id"] == session_key("thread-1")


def test_create_session_persists_and_is_fetchable(isolated):
    created = sessions.create_session("thread-1")
    fetched = sessions.get_session("thread-1")
    assert fetched["id"] == created["id"]


def test_create_session_is_idempotent_first_writer_wins(isolated):
    first = sessions.create_session("thread-1", title="First title")
    second = sessions.create_session("thread-1", title="Second title")
    assert first["id"] == second["id"]
    assert second["title"] == "First title"        # the racer's title never overwrote the winner's

    with closing(store._connect()) as db:
        count = db.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", (first["id"],)).fetchone()[0]
    assert count == 1


def test_create_session_rejects_bad_visibility(isolated):
    with pytest.raises(sessions.SessionValueError):
        sessions.create_session(visibility="public")


def test_create_session_rejects_non_string_title(isolated):
    with pytest.raises(sessions.SessionValueError):
        sessions.create_session(title=12345)


def test_create_session_empty_title_is_omitted_not_stored_empty(isolated):
    doc = sessions.create_session(title="   ")
    assert "title" not in doc


# =========================================================================== cross-client reuse / collision

def test_cross_client_reuse_same_raw_token_merges_into_one_session(isolated):
    """Two DIFFERENT clients supplying the IDENTICAL raw session token land on the SAME session id --
    documented, intentional behavior (module docstring), not an accidental collision."""
    created_by_cli = sessions.create_session("shared-thread", client_id="cli-wrapper")
    created_by_browser = sessions.create_session("shared-thread", client_id="studio-browser")
    assert created_by_cli["id"] == created_by_browser["id"]
    # first writer's client_key facet survives; the session ENTITY has one creating-client facet even
    # though member runs below may carry a different client_key each.
    assert created_by_cli["client_key"] == client_key("cli-wrapper")

    rid_a = _run(session="shared-thread", client="cli-wrapper")
    rid_b = _run(session="shared-thread", client="studio-browser")
    page = sessions.list_session_runs("shared-thread")
    ids = [r["id"] for r in page["runs"]]
    assert rid_a in ids and rid_b in ids
    # the two runs genuinely carry different client_key values -- reuse merges the SESSION, not identity
    run_a, run_b = store.get_run(rid_a), store.get_run(rid_b)
    assert run_a["client_key"] != run_b["client_key"]


# ================================================================================================== get

def test_get_session_nonexistent_returns_none(isolated):
    assert sessions.get_session("never-used-token") is None


def test_get_session_explicit_with_no_runs_omits_activity_fields(isolated):
    sessions.create_session("empty-thread")
    doc = sessions.get_session("empty-thread")
    assert "first_activity_ts" not in doc
    assert "last_activity_ts" not in doc
    assert "run_count" not in doc


def test_get_session_includes_derived_activity_once_runs_exist(isolated):
    sessions.create_session("thread-1")
    _run(session="thread-1", started=10.0)
    _run(session="thread-1", started=20.0)
    doc = sessions.get_session("thread-1")
    assert doc["run_count"] == 2
    assert doc["first_activity_ts"] <= doc["last_activity_ts"]


def test_get_session_legacy_session_key_with_no_row_is_a_pure_read(isolated):
    """A run recorded with a session_key that predates any explicit `sessions` row still resolves via
    the derived-aggregate path -- and a plain get_session() call must NOT write a row."""
    _run(session="legacy-thread", started=5.0)
    doc = sessions.get_session("legacy-thread")
    assert doc is not None
    assert doc["materialized_from"] == "lazy_backfill"
    assert doc["run_count"] == 1

    with closing(store._connect()) as db:
        count = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert count == 0, "a plain (materialize=False) get_session() must never write"


def test_get_session_materialize_true_persists_the_lazy_row(isolated):
    _run(session="legacy-thread", started=5.0)
    doc = sessions.get_session("legacy-thread", materialize=True)
    assert doc["materialized_from"] == "lazy_backfill"
    with closing(store._connect()) as db:
        row = db.execute("SELECT id FROM sessions WHERE id = ?", (doc["id"],)).fetchone()
    assert row is not None


def test_get_session_by_raw_and_opaque_token_agree(isolated):
    created = sessions.create_session("thread-1")
    assert sessions.get_session("thread-1")["id"] == created["id"]
    assert sessions.get_session(created["id"])["id"] == created["id"]


# ============================================================================================ materialize

def test_materialize_session_returns_none_when_truly_nonexistent(isolated):
    assert sessions.materialize_session("never-used") is None


def test_materialize_session_single_client_key_is_backfilled(isolated):
    _run(session="legacy-thread", client="only-client", started=1.0)
    _run(session="legacy-thread", client="only-client", started=2.0)
    doc = sessions.materialize_session("legacy-thread")
    assert doc["client_key"] == client_key("only-client")


def test_materialize_session_ambiguous_client_key_is_omitted(isolated):
    """Member runs from TWO DIFFERENT client_key values -- never guess which one 'owns' the session."""
    _run(session="legacy-thread", client="client-a", started=1.0)
    _run(session="legacy-thread", client="client-b", started=2.0)
    doc = sessions.materialize_session("legacy-thread")
    assert "client_key" not in doc


def test_materialize_session_created_ts_is_the_earliest_run(isolated):
    _run_at(100.0, session="legacy-thread")
    _run_at(50.0, session="legacy-thread")
    _run_at(200.0, session="legacy-thread")
    doc = sessions.materialize_session("legacy-thread")
    assert doc["created_ts"] == 50.0


def test_materialize_session_idempotent_second_call_same_row(isolated):
    _run(session="legacy-thread", started=1.0)
    first = sessions.materialize_session("legacy-thread")
    second = sessions.materialize_session("legacy-thread")
    assert first["id"] == second["id"]
    with closing(store._connect()) as db:
        count = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert count == 1


# =============================================================================================== update

def test_update_session_rename_and_visibility(isolated):
    sessions.create_session("thread-1", title="Old title")
    updated = sessions.update_session("thread-1", title="New title", visibility="hidden")
    assert updated["title"] == "New title"
    assert updated["privacy"] == {"visibility": "hidden"}


def test_update_session_clear_title_with_empty_string(isolated):
    sessions.create_session("thread-1", title="Has a title")
    updated = sessions.update_session("thread-1", title="   ")
    assert "title" not in updated


def test_update_session_returns_none_for_nonexistent_row(isolated):
    assert sessions.update_session("never-created") is None


def test_update_session_no_op_call_still_returns_current_document(isolated):
    sessions.create_session("thread-1", title="Stays")
    result = sessions.update_session("thread-1")
    assert result["title"] == "Stays"


def test_update_session_rejects_bad_visibility(isolated):
    sessions.create_session("thread-1")
    with pytest.raises(sessions.SessionValueError):
        sessions.update_session("thread-1", visibility="nope")


def test_update_session_does_not_materialize_a_legacy_session(isolated):
    """update_session must never fabricate a row -- only create_session/materialize_session do that."""
    _run(session="legacy-thread", started=1.0)
    assert sessions.update_session("legacy-thread", title="x") is None
    with closing(store._connect()) as db:
        count = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert count == 0


# =============================================================================================== delete

def test_delete_session_removes_the_row(isolated):
    sessions.create_session("thread-1")
    result = sessions.delete_session("thread-1")
    assert result == {"ok": True, "action": "delete_session",
                       "session_id": session_key("thread-1"), "deleted_row": True}
    assert sessions.get_session("thread-1") is None


def test_delete_session_never_touches_member_runs(isolated):
    sessions.create_session("thread-1")
    rid = _run(session="thread-1")
    sessions.delete_session("thread-1")
    run = store.get_run(rid)
    assert run is not None
    assert run["session_key"] == session_key("thread-1")   # untouched -- run keeps its own lifecycle
    # the runs are still queryable as a (now-implicit, lazy) session even after the entity row is gone
    page = sessions.list_session_runs("thread-1")
    assert rid in [r["id"] for r in page["runs"]]


def test_delete_session_is_idempotent_when_no_row_exists(isolated):
    result = sessions.delete_session("never-created")
    assert result["deleted_row"] is False
    assert result["ok"] is True


# ================================================================================================== list

def test_list_sessions_merges_explicit_and_lazy_sorted_by_last_activity(isolated):
    sessions.create_session("thread-old", title="Old")
    _run(session="thread-old", started=10.0)
    _run(session="thread-legacy", started=50.0)          # never explicitly created
    sessions.create_session("thread-new", title="New")
    _run(session="thread-new", started=99.0)

    listed = sessions.list_sessions()
    ids = [d["id"] for d in listed]
    assert session_key("thread-new") in ids
    assert session_key("thread-legacy") in ids
    assert session_key("thread-old") in ids
    # newest activity first
    assert ids.index(session_key("thread-new")) < ids.index(session_key("thread-legacy"))
    assert ids.index(session_key("thread-legacy")) < ids.index(session_key("thread-old"))


def test_list_sessions_excludes_hidden_by_default(isolated):
    sessions.create_session("visible-thread", visibility="visible")
    sessions.create_session("hidden-thread", visibility="hidden")
    ids = [d["id"] for d in sessions.list_sessions()]
    assert session_key("visible-thread") in ids
    assert session_key("hidden-thread") not in ids
    ids_with_hidden = [d["id"] for d in sessions.list_sessions(include_hidden=True)]
    assert session_key("hidden-thread") in ids_with_hidden


def test_list_sessions_respects_limit(isolated):
    for i in range(5):
        sessions.create_session(f"thread-{i}")
    assert len(sessions.list_sessions(limit=2)) == 2


# ======================================================================================= session-scoped runs

def test_list_session_runs_unknown_or_empty_id(isolated):
    page = sessions.list_session_runs(None)
    assert page == {"session_id": None, "runs": [], "next_cursor": None, "count": 0}


def test_list_session_runs_ordered_oldest_first(isolated):
    rid_1 = _run(session="thread-1", started=10.0, prompt="first")
    rid_2 = _run(session="thread-1", started=20.0, prompt="second")
    rid_3 = _run(session="thread-1", started=30.0, prompt="third")
    page = sessions.list_session_runs("thread-1")
    assert [r["id"] for r in page["runs"]] == [rid_1, rid_2, rid_3]


def test_list_session_runs_scoped_to_exactly_one_session(isolated):
    rid_a = _run(session="thread-a", started=1.0)
    rid_b = _run(session="thread-b", started=2.0)
    page_a = sessions.list_session_runs("thread-a")
    assert [r["id"] for r in page_a["runs"]] == [rid_a]
    assert rid_b not in [r["id"] for r in page_a["runs"]]


def test_list_session_runs_paginates_with_cursor(isolated):
    ids = [_run(session="thread-1", started=float(i)) for i in range(5)]
    page_1 = sessions.list_session_runs("thread-1", limit=2)
    assert [r["id"] for r in page_1["runs"]] == ids[:2]
    assert page_1["next_cursor"] is not None

    page_2 = sessions.list_session_runs("thread-1", cursor=page_1["next_cursor"], limit=2)
    assert [r["id"] for r in page_2["runs"]] == ids[2:4]
    assert page_2["next_cursor"] is not None

    page_3 = sessions.list_session_runs("thread-1", cursor=page_2["next_cursor"], limit=2)
    assert [r["id"] for r in page_3["runs"]] == ids[4:5]
    assert page_3["next_cursor"] is None                  # exhausted -- fewer than `limit` returned


def test_list_session_runs_excludes_derived_by_default(isolated):
    parent = _run(session="thread-1", started=1.0)
    child = store.record(source="replay", client="tester", session_key=session_key("thread-1"),
                         messages=[{"role": "user", "content": "hi"}], response="ok",
                         started=2.0, parent_run_id=parent)
    default_page = sessions.list_session_runs("thread-1")
    assert child not in [r["id"] for r in default_page["runs"]]
    full_page = sessions.list_session_runs("thread-1", include_derived=True)
    assert child in [r["id"] for r in full_page["runs"]]


def test_list_session_runs_belongs_to_at_most_one_session(isolated):
    """Structural: runs.session_key is a single scalar column -- a run recorded under one session id
    never appears under a different one."""
    rid = _run(session="thread-a", started=1.0)
    assert rid in [r["id"] for r in sessions.list_session_runs("thread-a")["runs"]]
    assert rid not in [r["id"] for r in sessions.list_session_runs("thread-b")["runs"]]


# ============================================================================================= sessionless

def test_sessionless_runs_remain_first_class(isolated):
    rid = _run(session=None)
    run = store.get_run(rid)
    assert run["session_key"] is None
    # not swept into any session's run list
    assert rid not in [r["id"] for r in sessions.list_session_runs("thread-1")["runs"]]


def test_get_session_of_none_is_none(isolated):
    assert sessions.get_session(None) is None


# ========================================================================================== concurrency

def _race_create_session(session_id: str, n_threads: int) -> list:
    barrier = threading.Barrier(n_threads)
    results: list = [None] * n_threads

    def worker(i):
        barrier.wait()
        try:
            results[i] = sessions.create_session(session_id, client_id="racer")["id"]
        except BaseException as exc:      # noqa: BLE001 -- see everything, exactly like store's own race test
            results[i] = f"ERROR:{type(exc).__name__}:{exc}"

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results


def test_concurrent_create_session_same_id_yields_one_row(isolated):
    """20 threads race create_session() on the SAME id, released simultaneously via a Barrier. Every call
    must succeed, every call must report the same id, and exactly one row must land in the table."""
    results = _race_create_session("concurrent-thread", 20)
    failures = [r for r in results if isinstance(r, str) and r.startswith("ERROR")]
    assert failures == [], f"{len(failures)}/{len(results)} create_session() calls failed: {failures[:5]}"
    assert len(set(results)) == 1, f"threads disagreed on the resulting session id: {set(results)}"

    with closing(store._connect()) as db:
        count = db.execute("SELECT COUNT(*) FROM sessions WHERE id = ?", (results[0],)).fetchone()[0]
    assert count == 1, f"expected exactly 1 row, found {count}"


def test_concurrent_create_session_flake_hunt(tmp_path):
    """25 trials x 20 threads = 500 create_session() calls, each trial racing ONE session id against a
    FRESH store, in a single assertion -- a lone green run proves nothing at these odds (house standard:
    tests/test_runs_store_concurrency.py's test_ensure_concurrency_flake_hunt runs the same 25x20 bar for
    the underlying schema-init lock this module's writes also go through)."""
    n_trials, n_threads = 25, 20
    total = 0
    failures: list = []
    bad_trials: list = []
    for trial in range(n_trials):
        store.RUNS_DIR = str(tmp_path / f"trial_{trial}" / "runs")
        store._schema_verified.clear()
        session_id = f"race-session-{trial}"
        results = _race_create_session(session_id, n_threads)
        total += len(results)
        failures.extend(r for r in results if isinstance(r, str) and r.startswith("ERROR"))
        if len(set(results)) != 1:
            bad_trials.append((trial, sorted(set(map(str, results)))))
            continue
        with closing(store._connect()) as db:
            count = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        if count != 1:
            bad_trials.append((trial, f"{count} rows"))

    assert failures == [], f"{len(failures)}/{total} calls failed across {n_trials} trials: {failures[:5]}"
    assert bad_trials == [], f"trials that did not converge on exactly one row: {bad_trials[:5]}"


# =========================================================================================== schema-first

def test_created_document_validates_against_its_own_schema(isolated):
    from clozn import schemas
    doc = sessions.create_session("thread-1", client_id="cli", title="t")
    schemas.validate(doc, "clozn.session.v1")


def test_lazy_document_validates_against_its_own_schema(isolated):
    """get_session()'s runtime read view is a documented SUPERSET of the schema (it merges in derived
    first_activity_ts/last_activity_ts/run_count, which the schema deliberately excludes -- see
    clozn.session.v1's description and clozn/runs/sessions.py's module docstring). Validate the stored
    CORE document -- the part the schema actually governs -- by stripping the derived keys back off."""
    from clozn import schemas
    _run(session="legacy-thread", started=1.0)
    doc = sessions.get_session("legacy-thread")
    core = {k: v for k, v in doc.items()
            if k not in ("first_activity_ts", "last_activity_ts", "run_count")}
    schemas.validate(core, "clozn.session.v1")


# ================================================================================================== route

class Handler:
    def __init__(self, path="/", headers=None):
        self.path = path
        self.headers = headers or {}
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


@pytest.fixture
def routed(isolated):
    from clozn.server.routes import sessions as route
    return route


def test_route_get_sessions_list(routed):
    sessions.create_session("thread-1")
    h = Handler("/sessions")
    assert routed.try_get(h, "/sessions") is True
    assert h.status == 200
    assert any(s["id"] == session_key("thread-1") for s in h.body["sessions"])


def test_route_get_session_detail_404(routed):
    h = Handler("/sessions/nope")
    assert routed.try_get(h, "/sessions/nope") is True
    assert h.status == 404


def test_route_get_session_detail_materializes_legacy(routed):
    _run(session="legacy-thread", started=1.0)
    h = Handler("/sessions/legacy-thread")
    assert routed.try_get(h, "/sessions/legacy-thread") is True
    assert h.status == 200
    assert h.body["materialized_from"] == "lazy_backfill"
    with closing(store._connect()) as db:
        count = db.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    assert count == 1


def test_route_get_session_runs(routed):
    rid = _run(session="thread-1", started=1.0)
    h = Handler("/sessions/thread-1/runs")
    assert routed.try_get(h, "/sessions/thread-1/runs") is True
    assert h.status == 200
    assert [r["id"] for r in h.body["runs"]] == [rid]


def test_route_post_sessions_create(routed):
    h = Handler("/sessions")
    ok = routed.try_post(h, "/sessions", {"session_id": "thread-1", "title": "hello"})
    assert ok is True
    assert h.status == 200
    assert h.body["ok"] is True
    assert h.body["session"]["title"] == "hello"


def test_route_post_sessions_create_bad_visibility_is_400(routed):
    h = Handler("/sessions")
    routed.try_post(h, "/sessions", {"visibility": "public"})
    assert h.status == 400


def test_route_post_sessions_update(routed):
    sessions.create_session("thread-1")
    h = Handler("/sessions/thread-1")
    ok = routed.try_post(h, "/sessions/thread-1", {"title": "renamed"})
    assert ok is True
    assert h.status == 200
    assert h.body["session"]["title"] == "renamed"


def test_route_post_sessions_update_missing_is_404(routed):
    h = Handler("/sessions/never-created")
    routed.try_post(h, "/sessions/never-created", {"title": "x"})
    assert h.status == 404


def test_route_post_sessions_delete(routed):
    sessions.create_session("thread-1")
    h = Handler("/sessions/thread-1/delete")
    ok = routed.try_post(h, "/sessions/thread-1/delete", {})
    assert ok is True
    assert h.status == 200
    assert h.body["deleted_row"] is True
    assert sessions.get_session("thread-1") is None


def test_route_unrelated_path_is_not_claimed(routed):
    h = Handler("/unrelated")
    assert routed.try_get(h, "/unrelated") is False
    assert routed.try_post(h, "/unrelated", {}) is False


def test_route_module_is_registered_and_loaded_cleanly():
    from clozn.server import app as cs
    from clozn.server.routes import _autoload
    from clozn.server.routes import sessions as route
    assert route in cs._GET_ROUTES
    assert route in cs._POST_ROUTES
    assert not any(name.endswith(".sessions") for name, _exc in _autoload.LOAD_FAILURES)


def _dispatch(method, path, body_obj=None):
    from clozn.server import app as cs
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"{method} {path} HTTP/1.1", "HTTP/1.1", method
    getattr(h, f"do_{method}")()
    _, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


def test_end_to_end_http_round_trip(isolated):
    created = _dispatch("POST", "/sessions", {"session_id": "e2e-thread", "title": "End to end"})
    assert created["ok"] is True
    sid = created["session"]["id"]

    fetched = _dispatch("GET", f"/sessions/{sid}")
    assert fetched["title"] == "End to end"

    listed = _dispatch("GET", "/sessions")
    assert any(s["id"] == sid for s in listed["sessions"])
