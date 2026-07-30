"""Contract tests for F6: the verify-before-save teaching loop -- clozn/runs/teaching_loop.py, composed
over F5 (clozn/runs/corrections.py) and D4 (clozn/replay/controlled.py's match_run()).

Covers the owner's brief acceptance criteria directly:
  * no promotion without a comparison (test_promotion_requires_both_run_ids_as_mandatory_kwargs,
    test_unknown_run_id_raises_before_any_write, test_verification_event_and_confirm_event_land_together)
  * promotion records the verification pair, not a bare boolean
    (test_promoted_correction_records_real_run_ids, test_confirmed_event_carries_the_pair)
  * failed corrections stay drafts (test_verify_leaves_draft_when_child_reproduces_target_failure,
    test_criterion_unavailable_counts_as_failed_not_promoted)
  * a failed verification is preserved as evidence, never discarded
    (test_failed_verification_is_recorded_as_evidence)
  * undo restores prior config transactionally, composed from F5's undo_last_change
    (test_undo_restores_prior_config_transactionally)
  * later runs show firing -- F5's existing receipt integration, reused unmodified
    (test_receipt_integration_still_fires_after_teaching_loop_promotion)
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clozn.runs.store as store                          # noqa: E402
from clozn.runs import corrections, teaching_loop          # noqa: E402
from clozn import schemas                                  # noqa: E402


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    runs_dir = str(tmp_path / "runs")
    monkeypatch.setattr(store, "RUNS_DIR", runs_dir)
    store._schema_verified.clear()
    yield runs_dir
    store._schema_verified.clear()


SESSION_A = "conversation-abc"


def _draft(content="Cite your sources.", correction_type="source_requirement"):
    return corrections.draft_correction(
        scope_kind="session", scope_value=SESSION_A, correction_type=correction_type, content=content)


def _record(response: str, *, messages=None) -> str:
    rid = store.record(
        source="cli", messages=list(messages or [{"role": "user", "content": "hi"}]), response=response)
    assert rid is not None
    return rid


# =================================================================================== mechanism: no promotion
# ================================================================================ without a comparison

def test_promotion_requires_both_run_ids_as_mandatory_kwargs():
    """Structural claim: target_run_id/child_run_id have no default -- inspect the signature directly,
    the same style test_corrections.py uses for resolve_corrections()'s content-blindness."""
    import inspect
    params = inspect.signature(teaching_loop.verify_and_promote).parameters
    assert params["target_run_id"].default is inspect.Parameter.empty
    assert params["child_run_id"].default is inspect.Parameter.empty
    assert params["target_run_id"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["child_run_id"].kind is inspect.Parameter.KEYWORD_ONLY


def test_unknown_run_id_raises_before_any_write(isolated):
    doc = _draft()
    child = _record("good")
    with pytest.raises(teaching_loop.TeachingLoopRunNotFoundError):
        teaching_loop.verify_and_promote(
            doc["id"], target_run_id="run_does_not_exist", child_run_id=child)
    # nothing was written: still exactly the original 'drafted' event, still unconfirmed.
    export = corrections.export_correction(doc["id"])
    assert [e["event_type"] for e in export["events"]] == ["drafted"]
    assert "confirmed_ts" not in export["correction"]


def test_target_and_child_must_differ(isolated):
    doc = _draft()
    rid = _record("same")
    with pytest.raises(teaching_loop.TeachingLoopValueError):
        teaching_loop.verify_and_promote(doc["id"], target_run_id=rid, child_run_id=rid)


def test_unknown_match_criterion_raises(isolated):
    doc = _draft()
    target, child = _record("bad"), _record("good")
    with pytest.raises(teaching_loop.TeachingLoopValueError):
        teaching_loop.verify_and_promote(
            doc["id"], target_run_id=target, child_run_id=child, match_criterion="vibes")


def test_cannot_verify_already_confirmed_correction(isolated):
    doc = _draft()
    corrections.confirm_correction(doc["id"])
    target, child = _record("bad"), _record("good")
    with pytest.raises(corrections.CorrectionStateError):
        teaching_loop.verify_and_promote(doc["id"], target_run_id=target, child_run_id=child)


def test_cannot_verify_deleted_correction(isolated):
    doc = _draft()
    corrections.delete_correction(doc["id"])
    target, child = _record("bad"), _record("good")
    with pytest.raises(corrections.CorrectionStateError):
        teaching_loop.verify_and_promote(doc["id"], target_run_id=target, child_run_id=child)


def test_unknown_correction_id_raises(isolated):
    target, child = _record("bad"), _record("good")
    with pytest.raises(corrections.CorrectionNotFoundError):
        teaching_loop.verify_and_promote(
            "corr_" + "0" * 24, target_run_id=target, child_run_id=child)


# ========================================================================================= passing verification

def test_verify_promotes_when_child_diverges_from_target_failure(isolated):
    doc = _draft()
    target = _record("I don't know, no sources.")
    child = _record("Per [1], the answer is X.")
    result = teaching_loop.verify_and_promote(doc["id"], target_run_id=target, child_run_id=child)
    schemas.validate(result, teaching_loop.SCHEMA_NAME)
    assert result["verification"] == "passed"
    assert result["promoted"] is True
    assert result["target_run_id"] == target
    assert result["child_run_id"] == child
    assert result["correction"]["enabled"] is True
    assert "confirmed_ts" in result["correction"]

    # promotion behaves EXACTLY like a hand confirm for resolution/receipt purposes.
    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    assert [a["correction_id"] for a in resolution["applied"]] == [doc["id"]]


def test_promoted_correction_records_real_run_ids_not_a_boolean(isolated):
    """'Promotion records the verification pair.' Not verified:true -- the actual ids."""
    doc = _draft()
    target, child = _record("bad"), _record("good")
    result = teaching_loop.verify_and_promote(doc["id"], target_run_id=target, child_run_id=child)
    assert result["target_run_id"] == target
    assert result["child_run_id"] == child
    export = corrections.export_correction(doc["id"])
    passed = next(e for e in export["events"] if e["event_type"] == "verification_passed")
    assert passed["run_id"] == child
    assert passed["detail"]["target_run_id"] == target
    assert passed["detail"]["child_run_id"] == child


def test_confirmed_event_carries_the_pair_distinguishing_teaching_loop_from_hand_confirm(isolated):
    hand = _draft(content="hand confirmed")
    corrections.confirm_correction(hand["id"])
    hand_export = corrections.export_correction(hand["id"])
    hand_confirmed = next(e for e in hand_export["events"] if e["event_type"] == "confirmed")
    assert "detail" not in hand_confirmed

    taught = _draft(content="teaching loop confirmed")
    target, child = _record("bad"), _record("good")
    teaching_loop.verify_and_promote(taught["id"], target_run_id=target, child_run_id=child)
    taught_export = corrections.export_correction(taught["id"])
    taught_confirmed = next(e for e in taught_export["events"] if e["event_type"] == "confirmed")
    assert taught_confirmed["detail"]["promoted_by"] == "teaching_loop"
    assert taught_confirmed["detail"]["target_run_id"] == target
    assert taught_confirmed["detail"]["child_run_id"] == child


def test_verification_event_and_confirm_event_land_together(isolated):
    """One call, one transaction: the verification_passed event that justified promotion and the
    confirmed event it produced are adjacent in sequence -- never confirmed without the event just
    before it, never the event without the confirm."""
    doc = _draft()
    target, child = _record("bad"), _record("good")
    teaching_loop.verify_and_promote(doc["id"], target_run_id=target, child_run_id=child)
    export = corrections.export_correction(doc["id"])
    types = [e["event_type"] for e in export["events"]]
    assert types == ["drafted", "verification_passed", "confirmed"]


# ========================================================================================= failing verification

def test_verify_leaves_draft_when_child_reproduces_target_failure(isolated):
    doc = _draft()
    target = _record("I don't know, no sources.")
    child = _record("I don't know, no sources.")  # identical -- nothing changed
    result = teaching_loop.verify_and_promote(doc["id"], target_run_id=target, child_run_id=child)
    assert result["verification"] == "failed"
    assert result["promoted"] is False
    assert "confirmed_ts" not in result["correction"]
    assert result["correction"]["enabled"] is False

    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    assert resolution["applied"] == []


def test_criterion_unavailable_counts_as_failed_not_promoted(isolated):
    """Neither run recorded a tool_parse outcome (ordinary store.record() calls never set
    output_contract) -> the criterion is genuinely unavailable on both sides -> failed, never promoted.
    Unavailable evidence must never be treated as an implicit pass."""
    doc = _draft()
    target, child = _record("bad"), _record("good")
    result = teaching_loop.verify_and_promote(
        doc["id"], target_run_id=target, child_run_id=child, match_criterion="tool_parse")
    assert result["comparison"]["available"] is False
    assert result["verification"] == "failed"
    assert result["promoted"] is False
    assert "confirmed_ts" not in result["correction"]


def test_failed_verification_is_recorded_as_evidence(isolated):
    """'A failed attempt is evidence too' -- the ledger keeps it forever, never silently discarded."""
    doc = _draft()
    target = _record("bad output")
    child = _record("bad output")
    teaching_loop.verify_and_promote(doc["id"], target_run_id=target, child_run_id=child)
    export = corrections.export_correction(doc["id"])
    failed = [e for e in export["events"] if e["event_type"] == "verification_failed"]
    assert len(failed) == 1
    assert failed[0]["run_id"] == child
    assert failed[0]["detail"]["target_run_id"] == target
    assert failed[0]["detail"]["comparison"]["matched"] is True
    # and the draft is still fully intact, not deleted or scrubbed.
    assert "content" in export["correction"]


def test_can_retry_after_a_failed_verification(isolated):
    """A failed attempt does not lock the draft -- a later, better retry can still be verified."""
    doc = _draft()
    target = _record("bad output")
    first_child = _record("bad output")
    first = teaching_loop.verify_and_promote(doc["id"], target_run_id=target, child_run_id=first_child)
    assert first["promoted"] is False

    second_child = _record("actually fixed now")
    second = teaching_loop.verify_and_promote(doc["id"], target_run_id=target, child_run_id=second_child)
    assert second["promoted"] is True
    export = corrections.export_correction(doc["id"])
    types = [e["event_type"] for e in export["events"]]
    assert types == ["drafted", "verification_failed", "verification_passed", "confirmed"]


# ================================================================================================== undo

def test_undo_restores_prior_config_transactionally(isolated):
    doc = _draft()
    target, child = _record("bad"), _record("good")
    teaching_loop.verify_and_promote(doc["id"], target_run_id=target, child_run_id=child)
    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    assert [a["correction_id"] for a in resolution["applied"]] == [doc["id"]]

    reverted = corrections.undo_last_change(doc["id"])
    assert reverted["enabled"] is False
    assert "confirmed_ts" not in reverted

    resolution = corrections.resolve_corrections(session_id=SESSION_A)
    assert resolution["applied"] == []

    # the verification_passed event that justified the promotion is untouched history, not erased by undo.
    export = corrections.export_correction(doc["id"])
    types = [e["event_type"] for e in export["events"]]
    assert types == ["drafted", "verification_passed", "confirmed", "drafted"]
    assert export["events"][1]["detail"]["target_run_id"] == target


# =========================================================================================== receipt reuse

def test_receipt_integration_still_fires_after_teaching_loop_promotion(isolated):
    """'Later runs show firing' -- F5's existing apply_and_record()/resolve_corrections() path, reused
    unmodified; F6 adds no second path for this."""
    doc = _draft()
    target, child = _record("bad"), _record("good")
    teaching_loop.verify_and_promote(doc["id"], target_run_id=target, child_run_id=child)

    rid, resolution = corrections.apply_and_record(
        run_kwargs={"source": "cli", "messages": [{"role": "user", "content": "hi"}], "response": "ok"},
        session_id=SESSION_A)
    run = store.get_run(rid)
    assert [a["correction_id"] for a in run["applied_corrections"]] == [doc["id"]]
    assert run["context_receipt"]["applied_corrections"] == run["applied_corrections"]


# ================================================================================================= schema

def test_result_document_validates_against_schema(isolated):
    doc = _draft()
    target, child = _record("bad"), _record("good")
    result = teaching_loop.verify_and_promote(doc["id"], target_run_id=target, child_run_id=child)
    schemas.validate(result)  # schema_version-inferred, exactly like an untrusted-document reader would


def test_export_with_verification_events_validates_against_schema(isolated):
    doc = _draft()
    target, child = _record("bad"), _record("bad")
    teaching_loop.verify_and_promote(doc["id"], target_run_id=target, child_run_id=child)
    export = corrections.export_correction(doc["id"])
    schemas.validate(export, "clozn.correction-export.v1")
