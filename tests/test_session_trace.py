"""Contract tests for F2: the session/agent trace API (clozn/runs/session_trace.py, the
`clozn.session-trace.v1` schema, and clozn/server/routes/session_trace.py).

Test matrix (per the F2 spec): a linear 3-turn session; a session with a fork branch (parent/child); a
failed run mid-session; settings drift mid-session; a redacted run mid-session; pagination across the
trace; the worker-never-touched proof; determinism twice-run. tmp_path only throughout.
"""
from __future__ import annotations

import io
import json
import os
import sys
from contextlib import closing

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clozn.runs.store as store              # noqa: E402
from clozn.runs import mutations, session_trace, sessions   # noqa: E402
from clozn.runs.association import client_key, session_key  # noqa: E402
from clozn import schemas                     # noqa: E402

GENERATED_AT = "2026-07-30T00:00:00Z"


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(store, "RUNS_DIR", runs_dir)
    store._schema_verified.clear()
    yield runs_dir
    store._schema_verified.clear()


def _run(*, session=None, client="tester", started=1.0, duration=0.05, prompt="hi", response="ok",
        parent=None, source="cli", error=None, finish_reason=None, meta=None):
    return store.record(
        source=source, client=client, client_key=client_key(client) if client else None,
        session_key=session_key(session) if session else None,
        messages=[{"role": "user", "content": prompt}], response=response,
        started=started, ended=started + duration, parent_run_id=parent, error=error,
        finish_reason=finish_reason, meta=meta,
    )


def _set_context_receipt(rid, **fields):
    """Overwrite a run's context_receipt for a deterministic, controllable fixture -- store.record()'s
    real build_context_receipt() output is fine for defaults but does not let a test dictate delivered
    segments/limits directly."""
    run = store.get_run(rid)
    run["context_receipt"] = dict(fields)
    assert store.replace_run(run)


def _receipt(delivered, *, prompt_tokens=None, generated_tokens=None, context_window_tokens=4096):
    limits = {"context_window_tokens": context_window_tokens}
    if prompt_tokens is not None:
        limits["prompt_tokens"] = prompt_tokens
    if generated_tokens is not None:
        limits["generated_tokens"] = generated_tokens
    return {"delivered": delivered, "limits": limits}


def _segment(sid, content_hash="a" * 16):
    return {"segment_id": sid, "included": True, "content_hash": content_hash}


# =========================================================================================== basic shape

def test_unknown_session_returns_none(isolated):
    assert session_trace.build_trace("never-existed", generated_at=GENERATED_AT) is None


def test_sessionless_run_does_not_create_a_traceable_session(isolated):
    _run(session=None)
    assert session_trace.build_trace("some-random-id", generated_at=GENERATED_AT) is None


def test_empty_explicit_session_has_zero_turns(isolated):
    sessions.create_session("thread-1")
    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    assert doc["turns"] == []
    assert doc["branches"] == []
    assert doc["first_went_wrong_candidates"] == []
    assert doc["totals_through_this_page"] == {
        "turn_count": 0, "duration_ms_total": 0, "prompt_tokens_total": 0, "generated_tokens_total": 0}
    schemas.validate(doc, "clozn.session-trace.v1")


def test_document_validates_against_its_own_schema(isolated):
    sessions.create_session("thread-1")
    _run(session="thread-1")
    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    schemas.validate(doc, "clozn.session-trace.v1")


def test_diagnostic_rule_registry_matches_the_real_rule_engine(isolated):
    from clozn.runs.diagnosis_rules import RULE_REGISTRY
    sessions.create_session("thread-1")
    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    assert doc["diagnostic_rule_registry"] == [
        {"rule_id": rid, "rule_name": name} for rid, name, _fn in RULE_REGISTRY]


# ================================================================================== a linear 3-turn session

def test_linear_three_turn_session_ordering_and_content(isolated):
    sessions.create_session("thread-1")
    r1 = _run(session="thread-1", started=1.0, prompt="first")
    r2 = _run(session="thread-1", started=2.0, prompt="second")
    r3 = _run(session="thread-1", started=3.0, prompt="third")
    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    assert [t["run_id"] for t in doc["turns"]] == [r1, r2, r3]
    assert "turn_comparison" not in doc["turns"][0]     # first turn: nothing to compare against
    assert doc["turns"][1]["turn_comparison"]["compared_to_run_id"] == r1
    assert doc["turns"][2]["turn_comparison"]["compared_to_run_id"] == r2
    for turn in doc["turns"]:
        assert turn["diagnostic_highlights"]["status_counts"]["finding"] >= 0
    assert doc["totals_through_this_page"]["turn_count"] == 3


def test_cumulative_timing_and_tokens_accumulate_per_turn(isolated):
    """duration_ms values are read back from whatever store.record() actually computed (float(ended -
    started) * 1000, truncated -- not exactly round-trippable for an arbitrary float second offset) rather
    than hardcoded, so this test is not coupled to that unrelated floating-point rounding behavior; what it
    proves is that `cumulative` is the exact running SUM of the per-turn values the trace itself reports."""
    sessions.create_session("thread-1")
    r1 = _run(session="thread-1", started=1.0, duration=0.10)
    _set_context_receipt(r1, **_receipt([], prompt_tokens=10, generated_tokens=5))
    r2 = _run(session="thread-1", started=2.0, duration=0.20)
    _set_context_receipt(r2, **_receipt([], prompt_tokens=20, generated_tokens=8))
    r3 = _run(session="thread-1", started=3.0, duration=0.05)
    _set_context_receipt(r3, **_receipt([], prompt_tokens=15, generated_tokens=3))

    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    t1, t2, t3 = doc["turns"]
    assert t1["context_usage"] == {"prompt_tokens": 10, "context_window_tokens": 4096, "generated_tokens": 5}
    d1, d2, d3 = (t["timing"]["duration_ms"] for t in (t1, t2, t3))

    assert t1["cumulative"] == {"turn_count": 1, "duration_ms_total": d1,
                                "prompt_tokens_total": 10, "generated_tokens_total": 5}
    assert t2["cumulative"] == {"turn_count": 2, "duration_ms_total": d1 + d2,
                                "prompt_tokens_total": 30, "generated_tokens_total": 13}
    assert t3["cumulative"] == {"turn_count": 3, "duration_ms_total": d1 + d2 + d3,
                                "prompt_tokens_total": 45, "generated_tokens_total": 16}
    assert doc["totals_through_this_page"] == t3["cumulative"]


def test_context_usage_omits_absent_keys_never_zero_pads(isolated):
    sessions.create_session("thread-1")
    r1 = _run(session="thread-1")
    _set_context_receipt(r1, delivered=[], limits={})   # nothing recorded at all
    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    assert "context_usage" not in doc["turns"][0]


# ========================================================================== context carried forward vs new

def test_context_changes_distinguish_carried_forward_from_new(isolated):
    sessions.create_session("thread-1")
    r1 = _run(session="thread-1", started=1.0)
    _set_context_receipt(r1, **_receipt([_segment("s1")]))
    r2 = _run(session="thread-1", started=2.0)
    _set_context_receipt(r2, **_receipt([_segment("s1"), _segment("s2")]))   # s1 carried, s2 new

    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    changes = doc["turns"][1]["turn_comparison"]["context_changes"]
    new_ids = {d["dimension"].rsplit(".", 1)[-1] for d in changes["new_segments"]}
    assert new_ids == {"s2"}
    assert changes["dropped_segments"] == []
    assert changes["carried_forward_segment_count"] == 1        # s1 -- present both turns, not "new"


def test_context_changes_dropped_segment(isolated):
    sessions.create_session("thread-1")
    r1 = _run(session="thread-1", started=1.0)
    _set_context_receipt(r1, **_receipt([_segment("s1"), _segment("s2")]))
    r2 = _run(session="thread-1", started=2.0)
    _set_context_receipt(r2, **_receipt([_segment("s1")]))    # s2 dropped

    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    changes = doc["turns"][1]["turn_comparison"]["context_changes"]
    dropped_ids = {d["dimension"].rsplit(".", 1)[-1] for d in changes["dropped_segments"]}
    assert dropped_ids == {"s2"}
    assert changes["new_segments"] == []
    assert changes["carried_forward_segment_count"] == 1


# =============================================================================== settings drift mid-session

def test_settings_drift_mid_session(isolated):
    sessions.create_session("thread-1")
    r1 = _run(session="thread-1", started=1.0, meta={"temperature": 0.0})
    r2 = _run(session="thread-1", started=2.0, meta={"temperature": 0.8})
    r3 = _run(session="thread-1", started=3.0, meta={"temperature": 0.8})   # unchanged from r2

    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    t2, t3 = doc["turns"][1], doc["turns"][2]
    dims = {d["dimension"] for d in t2["turn_comparison"]["settings_changes"]}
    assert "generation.temperature" in dims
    # R12 fires as a diagnostic highlight for the SAME drift -- composed, not re-derived
    fired = {f["rule_id"] for f in t2["diagnostic_highlights"]["findings"]}
    assert "R12" in fired
    # r3 vs r2: no drift
    assert t3["turn_comparison"]["settings_changes"] == []
    assert "first_settings_drift" in {c["kind"] for c in doc["first_went_wrong_candidates"]}
    drift_candidate = next(c for c in doc["first_went_wrong_candidates"] if c["kind"] == "first_settings_drift")
    assert drift_candidate["run_id"] == r2
    assert drift_candidate["compared_to_run_id"] == r1


def test_run_diffs_own_generated_at_is_never_embedded(isolated):
    """run_diff.compare_runs() stamps a live wall-clock generated_at on its own return value -- this
    module must never copy that field into the trace (that would break determinism). Prove it structurally
    by scanning the WHOLE document for run_diff's own top-level keys."""
    sessions.create_session("thread-1")
    _run(session="thread-1", started=1.0, meta={"temperature": 0.0})
    _run(session="thread-1", started=2.0, meta={"temperature": 0.8})
    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    text = json.dumps(doc)
    assert '"comparison_selection"' not in text
    assert '"ranking"' not in text
    assert '"privacy_limited"' not in text
    assert '"summary_axes"' not in text


# ======================================================================================= a failed run

def test_failed_run_included_never_omitted(isolated):
    sessions.create_session("thread-1")
    r1 = _run(session="thread-1", started=1.0)
    r2 = _run(session="thread-1", started=2.0, error="boom: worker crashed")
    r3 = _run(session="thread-1", started=3.0)

    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    assert [t["run_id"] for t in doc["turns"]] == [r1, r2, r3]
    assert doc["turns"][1]["error"] == "boom: worker crashed"

    kinds = {c["kind"]: c for c in doc["first_went_wrong_candidates"]}
    assert "first_failed_run" in kinds
    assert kinds["first_failed_run"]["run_id"] == r2


def test_cancelled_run_via_termination_reason_counts_as_failed(isolated):
    sessions.create_session("thread-1")
    r1 = _run(session="thread-1", started=1.0)
    r2 = _run(session="thread-1", started=2.0)
    _set_context_receipt(r2, delivered=[], limits={}, termination={"reason": "client_cancelled"})
    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    kinds = {c["kind"]: c for c in doc["first_went_wrong_candidates"]}
    assert kinds["first_failed_run"]["run_id"] == r2


def test_ordinary_finish_reason_length_is_not_a_failure(isolated):
    """finish_reason == 'length' is an ordinary, successful stop -- not 'failed or cancelled'."""
    sessions.create_session("thread-1")
    _run(session="thread-1", started=1.0, finish_reason="length")
    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    assert "first_failed_run" not in {c["kind"] for c in doc["first_went_wrong_candidates"]}


# ============================================================================================== branches

def test_fork_branch_appears_separately_never_flattened(isolated):
    sessions.create_session("thread-1")
    r1 = _run(session="thread-1", started=1.0)
    r2 = _run(session="thread-1", started=2.0)
    fork = _run(session="thread-1", started=1.5, source="branch", parent=r1)
    retry = _run(session="thread-1", started=1.6, source="replay", parent=r1)

    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    turn_ids = [t["run_id"] for t in doc["turns"]]
    assert turn_ids == [r1, r2]                 # branches never appear in the linear turns list
    assert fork not in turn_ids
    assert retry not in turn_ids

    assert len(doc["branches"]) == 1
    branch = doc["branches"][0]
    assert branch["parent_run_id"] == r1
    child_ids = {c["id"] for c in branch["children"]}
    assert child_ids == {fork, retry}


def test_no_branches_key_present_when_no_forks_exist(isolated):
    sessions.create_session("thread-1")
    _run(session="thread-1")
    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    assert doc["branches"] == []


# ============================================================================================== redaction

def test_literal_redaction_mid_session_derives_what_it_can(isolated):
    sessions.create_session("thread-1")
    r1 = _run(session="thread-1", started=1.0, prompt="the secret is orange")
    _set_context_receipt(r1, **_receipt([_segment("s1")], prompt_tokens=10, generated_tokens=5))
    r2 = _run(session="thread-1", started=2.0)
    _set_context_receipt(r2, **_receipt([_segment("s1")], prompt_tokens=12, generated_tokens=6))

    result = mutations.redact_run(r1, literals=["orange"])
    assert result["ok"] and result["already_redacted"] is False

    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    # the redacted run is STILL a turn -- literal redaction never severs session_key
    assert [t["run_id"] for t in doc["turns"]] == [r1, r2]
    assert doc["turns"][0]["redacted"] is True
    # numeric/structural evidence survives literal redaction -- still derivable
    assert doc["turns"][0]["context_usage"]["prompt_tokens"] == 10
    assert doc["turns"][1]["turn_comparison"]["available"] is True
    schemas.validate(doc, "clozn.session-trace.v1")


def test_full_tombstone_redaction_severs_session_membership_honestly(isolated):
    """FULL (non-literal) redaction clears session_key as part of mutations.py's own documented contract
    -- the trace must not claim the run is still part of this session's evidence trail once that has
    happened everywhere else in the codebase."""
    sessions.create_session("thread-1")
    r1 = _run(session="thread-1", started=1.0)
    r2 = _run(session="thread-1", started=2.0, error="boom")
    r3 = _run(session="thread-1", started=3.0)

    mutations.redact_run(r2)      # full tombstone -- no literals

    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    assert [t["run_id"] for t in doc["turns"]] == [r1, r3]
    assert r2 not in [t["run_id"] for t in doc["turns"]]
    # the now-invisible failed run cannot be surfaced as a candidate either -- it is genuinely gone
    # from this session's evidence trail, matching store.find_runs/list_session_runs everywhere else.
    assert "first_failed_run" not in {c["kind"] for c in doc["first_went_wrong_candidates"]}
    # comparisons quietly skip the now-absent turn rather than crashing
    assert doc["turns"][1]["turn_comparison"]["compared_to_run_id"] == r1


# ============================================================================================ pagination

def test_pagination_across_the_trace(isolated):
    sessions.create_session("thread-1")
    ids = [_run(session="thread-1", started=float(i)) for i in range(5)]

    page1 = session_trace.build_trace("thread-1", generated_at=GENERATED_AT, limit=2)
    assert [t["run_id"] for t in page1["turns"]] == ids[:2]
    assert page1["page"]["next_cursor"] is not None
    assert page1["page"]["cursor"] is None

    page2 = session_trace.build_trace(
        "thread-1", cursor=page1["page"]["next_cursor"], generated_at=GENERATED_AT, limit=2)
    assert [t["run_id"] for t in page2["turns"]] == ids[2:4]
    assert page2["page"]["cursor"] == page1["page"]["next_cursor"]
    # the SAME comparison baseline continues seamlessly across the page boundary
    assert page2["turns"][0]["turn_comparison"]["compared_to_run_id"] == ids[1]
    # running totals continue from where page1 left off, not reset to zero
    assert page2["turns"][0]["cumulative"]["turn_count"] == 3

    page3 = session_trace.build_trace(
        "thread-1", cursor=page2["page"]["next_cursor"], generated_at=GENERATED_AT, limit=2)
    assert [t["run_id"] for t in page3["turns"]] == ids[4:5]
    assert page3["page"]["next_cursor"] is None
    assert page3["turns"][0]["cumulative"]["turn_count"] == 5


def test_pagination_running_totals_match_a_single_unpaginated_fetch(isolated):
    sessions.create_session("thread-1")
    for i in range(5):
        rid = _run(session="thread-1", started=float(i), duration=0.01 * (i + 1))
        _set_context_receipt(rid, **_receipt([], prompt_tokens=i + 1, generated_tokens=i))

    full = session_trace.build_trace("thread-1", generated_at=GENERATED_AT, limit=100)
    last_full_cumulative = full["turns"][-1]["cumulative"]

    cursor = None
    last_paged_cumulative = None
    while True:
        page = session_trace.build_trace("thread-1", cursor=cursor, generated_at=GENERATED_AT, limit=2)
        if page["turns"]:
            last_paged_cumulative = page["turns"][-1]["cumulative"]
        cursor = page["page"]["next_cursor"]
        if cursor is None:
            break
    assert last_paged_cumulative == last_full_cumulative


def test_first_went_wrong_candidates_scan_beyond_the_current_page(isolated):
    """The candidate scan is session-wide, not scoped to whatever page was requested."""
    sessions.create_session("thread-1")
    _run(session="thread-1", started=1.0)
    failing = _run(session="thread-1", started=2.0, error="boom")
    _run(session="thread-1", started=3.0)
    _run(session="thread-1", started=4.0)

    page = session_trace.build_trace("thread-1", generated_at=GENERATED_AT, limit=1)
    assert len(page["turns"]) == 1
    kinds = {c["kind"]: c for c in page["first_went_wrong_candidates"]}
    assert kinds["first_failed_run"]["run_id"] == failing


# ================================================================================== the worker-never-touched proof

def test_build_trace_never_touches_a_substrate_or_engine(isolated, monkeypatch):
    """Mirrors tests/test_token_workbench.py's test_route_never_calls_engine_or_scorer: a substrate/engine
    double whose __getattr__ raises on ANY access, wired into clozn.server.app.active_sub. build_trace()
    must complete successfully without ever touching it -- it has no reason to: this module is engine-free
    by construction (see clozn/runs/session_trace.py's own module docstring)."""
    class BoomEngine:
        def __getattr__(self, name):
            raise AssertionError(f"session_trace must never touch engine.{name}")

    class BoomSub:
        steer = None
        engine = BoomEngine()

        def __getattr__(self, name):
            raise AssertionError(f"session_trace must never touch substrate.{name}")

    import clozn.server.app as app_module
    monkeypatch.setattr(app_module, "active_sub", lambda h: BoomSub())

    sessions.create_session("thread-1")
    _run(session="thread-1", started=1.0)
    _run(session="thread-1", started=2.0, meta={"temperature": 0.5})
    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    assert doc is not None
    assert len(doc["turns"]) == 2

    # the poisoned accessor genuinely raises if anything ever calls it -- prove the double itself is live
    with pytest.raises(AssertionError):
        app_module.active_sub(None).engine.chat


def test_route_never_touches_engine(isolated, monkeypatch):
    class BoomEngine:
        def __getattr__(self, name):
            raise AssertionError(f"GET /sessions/<id>/trace must never touch engine.{name}")

    class BoomSub:
        engine = BoomEngine()

        def __getattr__(self, name):
            raise AssertionError(f"GET /sessions/<id>/trace must never touch substrate.{name}")

    import clozn.server.app as app_module
    monkeypatch.setattr(app_module, "active_sub", lambda h: BoomSub())

    from clozn.server.routes import session_trace as route
    sessions.create_session("thread-1")
    _run(session="thread-1")

    h = Handler(f"/sessions/thread-1/trace")
    assert route.try_get(h, "/sessions/thread-1/trace") is True
    assert h.status == 200


# ==================================================================================== determinism (twice-run)

def test_determinism_twice_run_byte_identical(isolated):
    sessions.create_session("thread-1", title="Determinism check")
    r1 = _run(session="thread-1", started=1.0, meta={"temperature": 0.0})
    _set_context_receipt(r1, **_receipt([_segment("s1")], prompt_tokens=10, generated_tokens=5))
    r2 = _run(session="thread-1", started=2.0, meta={"temperature": 0.8}, error="boom")
    _set_context_receipt(r2, **_receipt([_segment("s1"), _segment("s2")], prompt_tokens=20, generated_tokens=8))
    _run(session="thread-1", started=1.5, source="branch", parent=r1)

    first = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    second = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_determinism_holds_across_pagination_too(isolated):
    sessions.create_session("thread-1")
    for i in range(6):
        _run(session="thread-1", started=float(i))

    first = session_trace.build_trace("thread-1", generated_at=GENERATED_AT, limit=3)
    second = session_trace.build_trace("thread-1", generated_at=GENERATED_AT, limit=3)
    assert first == second

    cursor = first["page"]["next_cursor"]
    first_p2 = session_trace.build_trace("thread-1", cursor=cursor, generated_at=GENERATED_AT, limit=3)
    second_p2 = session_trace.build_trace("thread-1", cursor=cursor, generated_at=GENERATED_AT, limit=3)
    assert first_p2 == second_p2


# ==================================================================================== no causal vocabulary

_BANNED_CAUSAL_WORDS = ("because", "caused", "causes", "causing", "due to", "responsible for",
                        "leads to", "results in", "the reason")


def test_no_causal_vocabulary_in_generated_documents(isolated):
    sessions.create_session("thread-1")
    _run(session="thread-1", started=1.0)
    r2 = _run(session="thread-1", started=2.0, meta={"temperature": 0.9}, error="boom: worker crashed")
    _run(session="thread-1", started=3.0, finish_reason="length")
    mutations.redact_run(r2, literals=["boom"])

    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    text = json.dumps(doc).lower()
    for word in _BANNED_CAUSAL_WORDS:
        assert word not in text, f"causal vocabulary {word!r} leaked into a session trace document"


def test_module_source_never_contains_causal_vocabulary():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "clozn", "runs", "session_trace.py")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    quoted_mentions = ('"because"', '"caused"', '"causes"', '"causing"', '"due to"', '"responsible for"',
                       '"leads to"', '"results in"', '"the reason"')
    remainder = text
    for mention in quoted_mentions:
        assert mention in remainder, f"expected the module docstring to name {mention} as banned vocabulary"
        remainder = remainder.replace(mention, "", 1)
    lowered = remainder.lower()
    for word in _BANNED_CAUSAL_WORDS:
        assert word not in lowered, f"causal vocabulary {word!r} used outside its one quoted mention"


def test_first_went_wrong_candidates_are_labeled_candidates_not_causes(isolated):
    sessions.create_session("thread-1")
    _run(session="thread-1", started=1.0)
    _run(session="thread-1", started=2.0, error="boom")
    doc = session_trace.build_trace("thread-1", generated_at=GENERATED_AT)
    assert doc["first_went_wrong_candidates"]
    for candidate in doc["first_went_wrong_candidates"]:
        assert candidate["kind"] in ("first_finding", "first_settings_drift", "first_failed_run")
        assert "cause" not in candidate["summary"].lower()


# =========================================================================================== the route

class Handler:
    def __init__(self, path="/", headers=None):
        self.path = path
        self.headers = headers or {}
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def test_route_happy_path(isolated):
    from clozn.server.routes import session_trace as route
    sessions.create_session("thread-1")
    _run(session="thread-1")
    h = Handler("/sessions/thread-1/trace")
    assert route.try_get(h, "/sessions/thread-1/trace") is True
    assert h.status == 200
    schemas.validate(h.body, "clozn.session-trace.v1")


def test_route_404_for_unknown_session(isolated):
    from clozn.server.routes import session_trace as route
    h = Handler("/sessions/never-existed/trace")
    assert route.try_get(h, "/sessions/never-existed/trace") is True
    assert h.status == 404


def test_route_400_on_bad_limit(isolated):
    from clozn.server.routes import session_trace as route
    sessions.create_session("thread-1")
    h = Handler("/sessions/thread-1/trace?limit=notanumber")
    assert route.try_get(h, "/sessions/thread-1/trace") is True
    assert h.status == 400


def test_route_400_on_bad_cursor(isolated):
    from clozn.server.routes import session_trace as route
    sessions.create_session("thread-1")
    h = Handler("/sessions/thread-1/trace?cursor=not-a-real-cursor!!")
    assert route.try_get(h, "/sessions/thread-1/trace") is True
    assert h.status == 400


def test_route_unrelated_path_not_claimed():
    from clozn.server.routes import session_trace as route
    h = Handler("/sessions/thread-1")
    assert route.try_get(h, "/sessions/thread-1") is False
    h2 = Handler("/sessions/thread-1/runs")
    assert route.try_get(h2, "/sessions/thread-1/runs") is False


def test_route_pagination_query_params(isolated):
    from clozn.server.routes import session_trace as route
    sessions.create_session("thread-1")
    for i in range(3):
        _run(session="thread-1", started=float(i))
    h = Handler("/sessions/thread-1/trace?limit=2")
    assert route.try_get(h, "/sessions/thread-1/trace") is True
    assert len(h.body["turns"]) == 2
    cursor = h.body["page"]["next_cursor"]
    h2 = Handler(f"/sessions/thread-1/trace?cursor={cursor}&limit=2")
    assert route.try_get(h2, "/sessions/thread-1/trace") is True
    assert len(h2.body["turns"]) == 1


def test_module_registered_before_the_sessions_generic_fallback():
    """Autoload dispatch order is alphabetical by module name -- 'session_trace.py' sorts before
    'sessions.py' ('_' < 's' in ASCII), so this route always gets first refusal on
    /sessions/<id>/trace before F1's own generic /sessions/<id> handler in sessions.py ever sees it. This
    locks that in as a verified contract rather than an incidental accident of filenames."""
    from clozn.server import app as cs
    from clozn.server.routes import session_trace as trace_route
    from clozn.server.routes import sessions as sessions_route
    assert trace_route in cs._GET_ROUTES
    assert sessions_route in cs._GET_ROUTES
    assert cs._GET_ROUTES.index(trace_route) < cs._GET_ROUTES.index(sessions_route)


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


def test_end_to_end_http_reaches_the_trace_route_not_the_sessions_fallback(isolated):
    sessions.create_session("thread-1")
    _run(session="thread-1")
    out = _dispatch("GET", "/sessions/thread-1/trace")
    assert out["schema_version"] == "clozn.session-trace.v1"
    assert len(out["turns"]) == 1
