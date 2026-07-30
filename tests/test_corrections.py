"""Contract tests for F5: the scoped correction store ("Teach Once") -- clozn/runs/corrections.py,
migration 5, and the clozn.correction.v1 / clozn.correction-resolution.v1 / clozn.correction-export.v1
schemas.

Covers the four acceptance criteria from the owner's brief directly:
  * no correction applies without appearing in the delivered-context receipt (test_receipt_integration_*)
  * scope is always explicit (test_scope_validation_*, test_resolve_never_matches_without_explicit_scope)
  * conflicts are surfaced, never silently resolved (test_resolve_*conflict*)
  * disable != delete history (test_disable_is_reversible_and_preserves_history,
    test_delete_is_permanent_and_scrubs_content)

And the structural (not merely conventional) claims from clozn/runs/corrections.py's own module
docstring: unconfirmed drafts are never selectable (test_unconfirmed_draft_is_never_selected), and
resolution never reads `content` to decide anything (test_resolution_is_content_blind).
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clozn.runs.migrations as migrations              # noqa: E402
import clozn.runs.store as store                          # noqa: E402
from clozn.runs import corrections                        # noqa: E402
from clozn import schemas                                 # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(store, "RUNS_DIR", runs_dir)
    store._schema_verified.clear()
    yield runs_dir
    store._schema_verified.clear()


SESSION_A = "conversation-abc"
SESSION_B = "conversation-xyz"
CLIENT_A = "vs-code-extension"
PROJECT_A = "clozn-repo"
MODEL_SHA = "a" * 64


def _draft(scope_kind="session", scope_value=SESSION_A, correction_type="style",
          content="Answer in bullet points."):
    return corrections.draft_correction(
        scope_kind=scope_kind, scope_value=scope_value, correction_type=correction_type, content=content)


def _draft_and_confirm(**kwargs):
    doc = _draft(**kwargs)
    return corrections.confirm_correction(doc["id"])["correction"]


# ============================================================================================ migration

def test_migration_5_creates_correction_tables(isolated):
    import sqlite3
    store._ensure()
    db = sqlite3.connect(os.path.join(store.RUNS_DIR, "runs.sqlite3"))
    try:
        assert migrations.current_version(db) == migrations.TARGET_VERSION
        assert migrations.TARGET_VERSION >= 5
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"corrections", "correction_events"}.issubset(tables)
    finally:
        db.close()


# ============================================================================================== schema

def test_drafted_document_validates_against_schema(isolated):
    doc = _draft()
    schemas.validate(doc, "clozn.correction.v1")
    assert doc["enabled"] is False
    assert "confirmed_ts" not in doc          # omitted, never null-padded


def test_scope_kind_cannot_express_relevance_matching():
    """The whole point of a closed enum: there is no scope kind a caller could set to mean 'whatever the
    topic gate thinks is relevant'. Extending it to add one would be an obvious, reviewable schema edit,
    not a hidden behavior change."""
    assert set(corrections.SCOPE_KINDS) == {"session", "client", "model", "project", "global_local"}
    with pytest.raises(corrections.CorrectionValueError):
        corrections.validate_scope("topic_relevance", "anything")


# ==================================================================================== lifecycle: draft/confirm

def test_unconfirmed_draft_is_never_selected(isolated):
    """STRUCTURAL claim from the module docstring: resolve_corrections()'s own SQL predicate requires
    confirmed_ts IS NOT NULL. Prove it by drafting (never confirming) and resolving against the exact
    matching scope."""
    _draft(scope_kind="session", scope_value=SESSION_A)
    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    assert resolution["applied"] == []
    assert resolution["conflicts"] == []


def test_confirm_is_required_and_explicit(isolated):
    doc = _draft()
    assert doc["enabled"] is False
    result = corrections.confirm_correction(doc["id"])
    confirmed = result["correction"]
    assert confirmed["enabled"] is True
    assert "confirmed_ts" in confirmed
    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    assert [a["correction_id"] for a in resolution["applied"]] == [doc["id"]]


def test_confirm_is_idempotent_no_duplicate_event(isolated):
    doc = _draft()
    corrections.confirm_correction(doc["id"])
    corrections.confirm_correction(doc["id"])
    export = corrections.export_correction(doc["id"])
    confirmed_events = [e for e in export["events"] if e["event_type"] == "confirmed"]
    assert len(confirmed_events) == 1


def test_confirm_surfaces_potential_conflicts_without_blocking(isolated):
    a = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style")
    b = _draft(scope_kind="session", scope_value=SESSION_A, correction_type="style", content="Be terse.")
    result = corrections.confirm_correction(b["id"])
    ids = {c["correction_id"] for c in result["potential_conflicts"]}
    assert a["id"] in ids


def test_draft_rejects_empty_content(isolated):
    with pytest.raises(corrections.CorrectionValueError):
        corrections.draft_correction(scope_kind="session", scope_value=SESSION_A,
                                     correction_type="style", content="   ")


def test_draft_rejects_unknown_type(isolated):
    with pytest.raises(corrections.CorrectionValueError):
        corrections.draft_correction(scope_kind="session", scope_value=SESSION_A,
                                     correction_type="mind_reading", content="x")


# ========================================================================================= scope validation

def test_scope_requires_explicit_value_except_global_local(isolated):
    with pytest.raises(corrections.CorrectionValueError):
        corrections.draft_correction(scope_kind="session", scope_value=None,
                                     correction_type="style", content="x")
    # global_local requires the ABSENCE of a value
    with pytest.raises(corrections.CorrectionValueError):
        corrections.draft_correction(scope_kind="global_local", scope_value="anything",
                                     correction_type="style", content="x")
    doc = corrections.draft_correction(scope_kind="global_local", scope_value=None,
                                       correction_type="style", content="x")
    assert doc["scope"] == {"kind": "global_local"}


def test_model_scope_requires_a_sha256_digest(isolated):
    with pytest.raises(corrections.CorrectionValueError):
        corrections.draft_correction(scope_kind="model", scope_value="qwen2.5-0.5b",
                                     correction_type="style", content="x")
    doc = corrections.draft_correction(scope_kind="model", scope_value=MODEL_SHA,
                                       correction_type="style", content="x")
    assert doc["scope"]["value"] == MODEL_SHA


def test_session_scope_reuses_association_key_normalization(isolated):
    """A correction scoped to a raw session token resolves against the SAME opaque key
    clozn.runs.store.record()/find_runs already use -- no second identity scheme."""
    from clozn.runs.association import session_key
    doc = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A)
    assert doc["scope"]["value"] == session_key(SESSION_A)


def test_resolve_never_matches_without_explicit_scope(isolated):
    """Scope is always explicit: a client-scoped correction never applies just because a run happens to
    share a session with one, and a correction for session B never applies while resolving session A."""
    _draft_and_confirm(scope_kind="session", scope_value=SESSION_B, correction_type="style")
    resolution = corrections.resolve_corrections(session_id=SESSION_A, include_global_local=False)
    assert resolution["applied"] == []


# =================================================================================== disable / enable / undo

def test_disable_is_reversible_and_preserves_history(isolated):
    doc = _draft_and_confirm()
    disabled = corrections.disable_correction(doc["id"])
    assert disabled["enabled"] is False
    assert disabled["confirmed_ts"] == doc["confirmed_ts"]          # untouched
    assert disabled.get("content") == doc.get("content")            # content intact, unlike delete
    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    assert resolution["applied"] == []

    enabled = corrections.enable_correction(doc["id"])
    assert enabled["enabled"] is True
    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    assert [a["correction_id"] for a in resolution["applied"]] == [doc["id"]]


def test_disable_requires_prior_confirmation(isolated):
    draft = _draft()
    with pytest.raises(corrections.CorrectionStateError):
        corrections.disable_correction(draft["id"])


def test_undo_toggles_disable_then_re_disable(isolated):
    doc = _draft_and_confirm()
    corrections.disable_correction(doc["id"])
    undone = corrections.undo_last_change(doc["id"])
    assert undone["enabled"] is True
    # a second undo call reverts the undo itself (it recorded a real "enabled" event) -> disabled again
    undone_again = corrections.undo_last_change(doc["id"])
    assert undone_again["enabled"] is False


def test_undo_confirm_reverts_to_drafted(isolated):
    doc = _draft_and_confirm()
    reverted = corrections.undo_last_change(doc["id"])
    assert reverted["enabled"] is False
    assert "confirmed_ts" not in reverted
    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    assert resolution["applied"] == []


def test_undo_with_nothing_undoable_raises(isolated):
    draft = _draft()
    with pytest.raises(corrections.CorrectionStateError):
        corrections.undo_last_change(draft["id"])


# ============================================================================================== delete

def test_delete_is_permanent_and_scrubs_content(isolated):
    doc = _draft_and_confirm()
    deleted = corrections.delete_correction(doc["id"], reason="no longer needed")
    assert "content" not in deleted                       # scrubbed, never null-padded
    assert deleted["content_hash"] == doc["content_hash"]  # hash survives forever
    assert deleted["enabled"] is False
    assert deleted["deleted_reason"] == "no longer needed"

    with pytest.raises(corrections.CorrectionStateError):
        corrections.undo_last_change(doc["id"])
    with pytest.raises(corrections.CorrectionStateError):
        corrections.confirm_correction(doc["id"])

    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    assert resolution["applied"] == []


def test_delete_is_idempotent(isolated):
    doc = _draft_and_confirm()
    first = corrections.delete_correction(doc["id"])
    second = corrections.delete_correction(doc["id"])
    assert first["deleted_ts"] == second["deleted_ts"]


def test_delete_history_survives_in_export_even_after_deletion(isolated):
    """The exact 'disable != delete history' claim: even a fully deleted correction's event ledger --
    including any applied/conflict_lost rows recorded before deletion -- stays fully readable."""
    doc = _draft_and_confirm()
    rid, resolution = corrections.apply_and_record(
        run_kwargs={"source": "cli", "messages": [{"role": "user", "content": "hi"}], "response": "ok"},
        session_id=SESSION_A)
    assert rid is not None
    corrections.delete_correction(doc["id"])
    export = corrections.export_correction(doc["id"])
    event_types = [e["event_type"] for e in export["events"]]
    assert "applied" in event_types
    assert "deleted" in event_types
    applied_event = next(e for e in export["events"] if e["event_type"] == "applied")
    assert applied_event["run_id"] == rid


# ============================================================================================ resolution

def test_multiple_types_all_apply_without_conflict(isolated):
    style = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style")
    fmt = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A,
                             correction_type="output_format", content="Reply in Markdown.")
    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    ids = {a["correction_id"] for a in resolution["applied"]}
    assert ids == {style["id"], fmt["id"]}
    assert resolution["conflicts"] == []


def test_conflict_across_scope_kinds_resolved_by_precedence_and_surfaced(isolated):
    session_corr = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style")
    client_corr = _draft_and_confirm(scope_kind="client", scope_value=CLIENT_A, correction_type="style",
                                     content="Be terse.")
    resolution = corrections.resolve_corrections(session_id=SESSION_A, client_id=CLIENT_A,
                                                 include_global_local=False)
    assert [a["correction_id"] for a in resolution["applied"]] == [session_corr["id"]]
    assert len(resolution["conflicts"]) == 1
    conflict = resolution["conflicts"][0]
    assert conflict["rule"] == "precedence"
    assert conflict["winner_id"] == session_corr["id"]
    assert conflict["losing_ids"] == [client_corr["id"]]
    # the loser is STILL named among the candidates -- never silently dropped
    candidate_ids = {c["correction_id"] for c in conflict["candidates"]}
    assert candidate_ids == {session_corr["id"], client_corr["id"]}


def test_same_scope_conflict_breaks_tie_by_recency(isolated):
    older = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style",
                               content="Be verbose.")
    import time
    time.sleep(0.01)
    newer = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style",
                               content="Be terse.")
    resolution = corrections.resolve_corrections(session_id=SESSION_A, include_global_local=False)
    conflict = resolution["conflicts"][0]
    assert conflict["winner_id"] == newer["id"]
    assert conflict["rule"] == "most_recently_confirmed"
    assert older["id"] in conflict["losing_ids"]


def test_global_local_always_applies_and_conflicts_with_specific_scope(isolated):
    global_corr = _draft_and_confirm(scope_kind="global_local", scope_value=None, correction_type="style")
    session_corr = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style",
                                      content="Different style.")
    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    assert len(resolution["conflicts"]) == 1
    assert resolution["conflicts"][0]["winner_id"] == session_corr["id"]  # session outranks global_local

    resolution_excluded = corrections.resolve_corrections(session_id=SESSION_A, include_global_local=False)
    assert [a["correction_id"] for a in resolution_excluded["applied"]] == [session_corr["id"]]
    assert resolution_excluded["conflicts"] == []


def test_resolution_is_content_blind(isolated):
    """No applied/candidate entry ever carries `content`, and resolution outcome is unaffected by what
    the competing corrections' text actually says -- only scope/type/confirmed_ts/id matter."""
    a = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style",
                           content="Answer only in haiku.")
    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    assert resolution["applied"] == [
        {k: v for k, v in entry.items() if k != "content"} for entry in resolution["applied"]
    ]
    for entry in resolution["applied"] + [c for conflict in resolution["conflicts"]
                                          for c in conflict["candidates"]]:
        assert "content" not in entry
    assert resolution["applied"][0]["correction_id"] == a["id"]


def test_resolve_corrections_takes_no_message_content_parameter():
    """The structural claim itself: inspect the function's signature and confirm it has no parameter
    that could carry a prompt/message/content payload for a future relevance matcher to read."""
    import inspect
    params = set(inspect.signature(corrections.resolve_corrections).parameters)
    for forbidden in ("messages", "content", "prompt", "text", "query"):
        assert forbidden not in params


# ============================================================================================ list / export

def test_list_filters_by_scope_and_disabled_state(isolated):
    a = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style")
    b = _draft_and_confirm(scope_kind="client", scope_value=CLIENT_A, correction_type="style",
                           content="x")
    corrections.disable_correction(b["id"])
    session_only = corrections.list_corrections(scope_kind="session")
    assert [d["id"] for d in session_only] == [a["id"]]
    enabled_only = corrections.list_corrections(include_disabled=False)
    assert [d["id"] for d in enabled_only] == [a["id"]]


def test_export_validates_and_lists_events_oldest_first(isolated):
    doc = _draft_and_confirm()
    corrections.disable_correction(doc["id"])
    export = corrections.export_correction(doc["id"])
    schemas.validate(export, "clozn.correction-export.v1")
    seqs = [e["seq"] for e in export["events"]]
    assert seqs == sorted(seqs)
    assert [e["event_type"] for e in export["events"]] == ["drafted", "confirmed", "disabled"]


def test_export_unknown_id_returns_none(isolated):
    assert corrections.export_correction("corr_" + "0" * 24) is None
    assert corrections.get_correction("corr_" + "0" * 24) is None


# ==================================================================================== receipt integration

def test_receipt_integration_applied_corrections_appear_on_run(isolated):
    """THE acceptance criterion: a correction that applies is on the run's context_receipt AND the run's
    own top-level applied_corrections field -- attached once, at creation."""
    doc = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style")
    rid, resolution = corrections.apply_and_record(
        run_kwargs={"source": "cli", "messages": [{"role": "user", "content": "hi"}], "response": "ok"},
        session_id=SESSION_A)
    assert rid is not None
    run = store.get_run(rid)
    assert run["applied_corrections"] == resolution["applied"]
    assert run["context_receipt"]["applied_corrections"] == resolution["applied"]
    assert [a["correction_id"] for a in run["applied_corrections"]] == [doc["id"]]


def test_receipt_integration_absent_when_never_resolved(isolated):
    """A run created the ordinary way (no corrections resolved at all) carries no applied_corrections
    key -- absent, not an empty list, per the schema's own distinction."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="ok")
    run = store.get_run(rid)
    assert "applied_corrections" not in run
    assert "applied_corrections" not in run["context_receipt"]


def test_receipt_integration_empty_list_when_resolved_but_nothing_matched(isolated):
    rid, resolution = corrections.apply_and_record(
        run_kwargs={"source": "cli", "messages": [{"role": "user", "content": "hi"}], "response": "ok"},
        session_id="an-unrelated-session")
    run = store.get_run(rid)
    assert run["applied_corrections"] == []
    assert run["context_receipt"]["applied_corrections"] == []


def test_receipt_integration_conflicts_are_recorded_on_the_run(isolated):
    session_corr = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style")
    _draft_and_confirm(scope_kind="client", scope_value=CLIENT_A, correction_type="style", content="x")
    rid, resolution = corrections.apply_and_record(
        run_kwargs={"source": "cli", "messages": [{"role": "user", "content": "hi"}], "response": "ok"},
        session_id=SESSION_A, client_id=CLIENT_A, include_global_local=False)
    run = store.get_run(rid)
    assert run["correction_conflicts"] == resolution["conflicts"]
    assert run["context_receipt"]["correction_conflicts"][0]["winner_id"] == session_corr["id"]


def test_apply_and_record_writes_conflict_lost_events(isolated):
    session_corr = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style")
    client_corr = _draft_and_confirm(scope_kind="client", scope_value=CLIENT_A, correction_type="style",
                                     content="x")
    rid, _ = corrections.apply_and_record(
        run_kwargs={"source": "cli", "messages": [{"role": "user", "content": "hi"}], "response": "ok"},
        session_id=SESSION_A, client_id=CLIENT_A, include_global_local=False)
    winner_export = corrections.export_correction(session_corr["id"])
    loser_export = corrections.export_correction(client_corr["id"])
    assert any(e["event_type"] == "applied" and e["run_id"] == rid for e in winner_export["events"])
    assert any(e["event_type"] == "conflict_lost" and e["run_id"] == rid for e in loser_export["events"])


def test_run_record_immutability_not_retroactively_edited(isolated):
    """Applied-correction ids attach at creation only -- confirming a NEW correction after a run exists
    must never retroactively appear on that already-created run."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}], response="ok",
                       session_key="session_" + "0" * 24)
    _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style")
    run = store.get_run(rid)
    assert "applied_corrections" not in run


def test_context_receipt_schema_validates_with_applied_corrections(isolated):
    from clozn.runs.context_receipt import build_context_receipt
    doc = _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style")
    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    receipt = build_context_receipt(
        messages=[{"role": "user", "content": "hi"}], run_id="run_test123",
        applied_corrections=resolution["applied"], correction_conflicts=resolution["conflicts"])
    schemas.validate(receipt)
    assert "schema_validation_error" not in receipt
    assert receipt["applied_corrections"][0]["correction_id"] == doc["id"]


# ============================================================================ delete/redact mutation coverage

def test_full_redaction_removes_applied_corrections_from_run(isolated):
    from clozn.runs import mutations
    _draft_and_confirm(scope_kind="session", scope_value=SESSION_A, correction_type="style")
    rid, _ = corrections.apply_and_record(
        run_kwargs={"source": "cli", "messages": [{"role": "user", "content": "hi"}], "response": "ok"},
        session_id=SESSION_A)
    mutations.redact_run(rid)
    run = store.get_run(rid)
    assert "applied_corrections" not in run
    assert run["context_receipt"] == {}
