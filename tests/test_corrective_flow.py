from __future__ import annotations

import json
import threading

import pytest

from clozn import schemas
from clozn.behavior import corrective_flow as flow


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "_PATH", str(tmp_path / "flow.json"))


def _run(**changes):
    run = {
        "id": "run_parent",
        "messages": [{"role": "user", "content": "Explain this."}],
        "response": "Stored sampled reply.",
        "identity": {"model_sha256": "a" * 64, "template_fingerprint": "template"},
        "session_key": "session_0123456789abcdef01234567",
        "meta": {},
    }
    run.update(changes)
    return run


def _success(preview):
    return {
        "stored_original_reply": "Stored sampled reply.",
        "baseline_reply": "A matched baseline reply.",
        "corrected_reply": "A concise reply.",
        "delta": {"words_a": 4, "words_b": 3, "word_delta": -1},
        "changed": True,
        "coherence": {"degenerate": False, "reasons": []},
        "intervention_observed": True,
        "comparison_note": "matched greedy baseline; stored original is context only",
        "child_outcomes": {
            "baseline": {"status": "success", "run_id": "run_baseline"},
            "corrected": {"status": "success", "run_id": "run_corrected"},
        },
        "requested_backend": preview["execution"]["requested_backend"],
        "executed_backend": "prompt_policy",
        "backend": "prompt_policy",
        "backend_fallback": preview["execution"]["expected_fallback"],
        "execution_identity": {"parent_run_id": "run_parent"},
        "outcome": {"status": "succeeded"},
    }


def _confirmed(run):
    preview = flow.create_preview(run, "less-verbose", now=100.0)
    result = flow.confirm_preview(
        preview["preview_id"], "confirm-key-0001", run, _success, now=101.0
    )
    return preview, result


def test_registry_for_run_discovers_exactly_six_actions_with_only_once_scope(isolated):
    doc = flow.registry_for_run(_run())
    assert len(doc["actions"]) == 6
    for action in doc["actions"]:
        # Durable session/profile scoping was retired -- a kept correction only ever selects
        # itself as its own parent run's revision, never a standing policy. See
        # docs/CAPABILITIES.md.
        assert action["scopes"] == ["once"]
        assert [item["scope"] for item in action["scope_eligibility"]] == ["once"]
        assert next(b for b in action["backends"] if b["type"] == "prompt_policy")[
            "qualification_id"
        ] == "clozn.prompt-policy.generic.v1"


def test_preview_is_bounded_and_reports_explicit_control_vector_fallback(isolated):
    with pytest.raises(flow.CorrectiveFlowError, match="action_id must be one of"):
        flow.create_preview(_run(), "write-any-system-prompt")
    preview = flow.create_preview(_run(), "use-context", "control_vector")
    assert preview["execution"]["requested_backend"] == "control_vector"
    assert preview["execution"]["expected_executed_backend"] == "prompt_policy"
    assert preview["execution"]["expected_fallback"] is True
    assert preview["execution"]["unavailability_reason"]


def test_confirm_is_one_shot_idempotent_persisted_and_never_touches_a_standing_policy(isolated):
    run = _run()
    preview = flow.create_preview(run, "less-verbose", now=100.0)
    calls = 0

    def execute(saved):
        nonlocal calls
        calls += 1
        return _success(saved)

    first = flow.confirm_preview(
        preview["preview_id"], "confirm-key-0001", run, execute, now=101.0
    )
    second = flow.confirm_preview(
        preview["preview_id"], "confirm-key-0001", run, execute, now=102.0
    )
    assert first == second
    assert calls == 1
    assert first["comparison"]["stored_original_reply"] == "Stored sampled reply."
    assert first["comparison"]["baseline_reply"] == "A matched baseline reply."
    assert first["comparison"]["corrected_reply"] == "A concise reply."
    assert first["execution"]["requested_backend"] == "prompt_policy"
    assert first["execution"]["executed_backend"] == "prompt_policy"
    schemas.validate(first, flow.RESULT_SCHEMA)


def test_confirm_refuses_run_evidence_drift_before_generation(isolated):
    run = _run()
    preview = flow.create_preview(run, "less-verbose")
    drifted = {**run, "response": "Externally changed"}
    called = False

    def execute(_preview):
        nonlocal called
        called = True
        return {}

    with pytest.raises(flow.CorrectiveFlowError, match="run evidence changed"):
        flow.confirm_preview(preview["preview_id"], "confirm-key-0001", drifted, execute)
    assert called is False


def test_partial_execution_error_preserves_independent_child_outcomes(isolated):
    run = _run()
    preview = flow.create_preview(run, "more-concrete")

    def corrected_failed(saved):
        return {
            "stored_original_reply": run["response"],
            "baseline_reply": "baseline succeeded",
            "child_outcomes": {
                "baseline": {"status": "success", "run_id": "run_baseline"},
                "corrected": {
                    "status": "error",
                    "error": {"code": "generation_failed", "message": "no child"},
                },
            },
            "requested_backend": saved["execution"]["requested_backend"],
            "outcome": {"status": "execution_error"},
        }

    result = flow.confirm_preview(
        preview["preview_id"], "confirm-key-0001", run, corrected_failed
    )
    assert result["outcome"]["status"] == "execution_error"
    assert result["children"]["baseline"] == {
        "status": "success", "run_id": "run_baseline"
    }
    assert result["children"]["corrected"]["error"]["code"] == "generation_failed"
    schemas.validate(result, flow.RESULT_SCHEMA)


def test_cancellation_during_generation_persists_cancelled_and_cannot_be_kept(isolated):
    run = _run()
    preview = flow.create_preview(run, "less-verbose")
    entered, release = threading.Event(), threading.Event()
    output = {}

    def execute(saved):
        entered.set()
        assert release.wait(timeout=3)
        return _success(saved)

    def confirm():
        output["result"] = flow.confirm_preview(
            preview["preview_id"], "confirm-key-0001", run, execute
        )

    worker = threading.Thread(target=confirm)
    worker.start()
    assert entered.wait(timeout=3)
    cancelled = flow.cancel_preview(preview["preview_id"])
    assert cancelled["status"] == "cancel_requested"
    release.set()
    worker.join(timeout=3)
    assert output["result"]["outcome"]["status"] == "cancelled"
    with pytest.raises(flow.CorrectiveFlowError, match="only a successful"):
        flow.keep_result(
            output["result"]["result_id"], "once",
            preview["scope_eligibility"][0]["prior_hash"], "keep-key-000001",
            get_run=lambda _rid: run, replace_run=lambda _run: True,
        )


def test_keep_once_selects_corrected_child_without_policy_and_undo_restores_prior(isolated):
    run = _run()
    runs = {
        run["id"]: dict(run),
        "run_baseline": {"id": "run_baseline"},
        "run_corrected": {"id": "run_corrected"},
    }

    def get_run(run_id):
        value = runs.get(run_id)
        return json.loads(json.dumps(value)) if value else None

    def replace_run(value):
        runs[value["id"]] = json.loads(json.dumps(value))
        return True

    preview, result = _confirmed(run)
    prior = next(s for s in preview["scope_eligibility"] if s["scope"] == "once")
    kept = flow.keep_result(
        result["result_id"], "once", prior["prior_hash"], "keep-key-000001",
        get_run=get_run, replace_run=replace_run, now=102.0,
    )
    assert runs["run_parent"]["selected_revision"]["child_run_id"] == "run_corrected"
    # A repeated request with the same key is a read, not a second mutation.
    assert flow.keep_result(
        result["result_id"], "once", prior["prior_hash"], "keep-key-000001",
        get_run=get_run, replace_run=replace_run, now=103.0,
    ) == kept
    txid = kept["transaction"]["id"]
    flow.undo_keep(txid, get_run=get_run, replace_run=replace_run, now=104.0)
    assert "selected_revision" not in runs["run_parent"]
    with pytest.raises(flow.CorrectiveFlowError, match="already undone"):
        flow.undo_keep(txid, get_run=get_run, replace_run=replace_run, now=105.0)


def test_keep_once_refuses_selected_revision_drift(isolated):
    run = _run()
    runs = {
        run["id"]: {**run, "selected_revision": {"child_run_id": "newer"}},
        "run_corrected": {"id": "run_corrected"},
    }
    preview, result = _confirmed(run)
    with pytest.raises(flow.CorrectiveFlowError, match="selected revision changed"):
        flow.keep_result(
            result["result_id"], "once",
            preview["scope_eligibility"][0]["prior_hash"], "keep-key-000001",
            get_run=lambda rid: runs.get(rid), replace_run=lambda _run: True,
        )


def test_keep_result_refuses_durable_session_or_profile_scope(isolated):
    """Durable session/profile scoping (a kept correction silently shaping future, unrelated
    requests) was retired -- `keep_result` only ever accepts `once` now. `meta.active_profile`
    is historical-only data here (Profiles were retired too); keep_result must not care either
    way."""
    run = _run(meta={"active_profile": "work"})
    preview, result = _confirmed(run)
    prior = next(s for s in preview["scope_eligibility"] if s["scope"] == "once")
    for scope in ("session", "profile"):
        with pytest.raises(flow.CorrectiveFlowError, match="scope must be once"):
            flow.keep_result(
                result["result_id"], scope, prior["prior_hash"], "keep-key-000001",
                get_run=lambda _rid: run, replace_run=lambda _run: True,
            )


def test_undo_keep_refuses_a_pre_retirement_session_or_profile_transaction(isolated):
    """A transaction persisted before durable corrections were retired must not silently no-op
    or crash; it must refuse explicitly, since nothing can reverse it anymore."""
    with flow._LOCK:
        doc = flow._load(strict=True)
        doc["transactions"]["repair_legacy"] = {
            "id": "repair_legacy", "scope": "session", "target": "session_x",
            "created_ts": 1.0, "undone_ts": None, "result_id": "fix_result_legacy",
        }
        flow._save(doc)
    with pytest.raises(flow.CorrectiveFlowError, match="no longer be undone"):
        flow.undo_keep(
            "repair_legacy", get_run=lambda _rid: None, replace_run=lambda _run: True,
        )


def _map(clear_ids, *, threshold=0.1):
    return {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": {"name": "forced_score", "version": "1"},
        "thresholds": {"cell_abs_delta_nats": threshold},
        "answer_spans": [{"id": "a1"}, {"id": "a2"}],
        "links": [
            {"answer_span_id": answer_id, "clears_floor": True}
            for answer_id in clear_ids
        ],
    }


def test_source_use_requires_compatible_precomputed_maps_and_uses_honest_label(isolated):
    run = _run()
    _, result = _confirmed(run)
    runs = {
        "run_baseline": {"id": "run_baseline", "influence_map": _map(["a1"])},
        "run_corrected": {
            "id": "run_corrected", "influence_map": _map(["a1", "a2"])
        },
    }
    compared = flow.compare_source_use(
        result["result_id"], get_run=lambda rid: runs.get(rid)
    )
    assert compared["delta_observed_source_dependence_ratio"] == 0.5
    assert "does not establish" in compared["caveat"]
    runs["run_corrected"]["influence_map"] = _map(["a1"], threshold=0.2)
    with pytest.raises(flow.CorrectiveFlowError, match="different method/version/threshold"):
        flow.compare_source_use(result["result_id"], get_run=lambda rid: runs.get(rid))
