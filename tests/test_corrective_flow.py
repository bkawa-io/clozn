from __future__ import annotations

import json
import threading

import pytest

from clozn import schemas
from clozn.behavior import corrective_flow as flow
from clozn.behavior import corrective_retries as policy
from clozn.profiles import store as profiles


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "_PATH", str(tmp_path / "flow.json"))
    monkeypatch.setattr(policy, "_PATH", str(tmp_path / "policy.json"))
    profile_store = profiles.ProfileStore(str(tmp_path / "profiles"))
    monkeypatch.setattr(policy, "_profile_store", lambda: profile_store)
    return profile_store


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


def _confirmed(run, *, active_profile=None):
    preview = flow.create_preview(
        run, "less-verbose", active_profile=active_profile, now=100.0
    )
    result = flow.confirm_preview(
        preview["preview_id"], "confirm-key-0001", run, _success, now=101.0
    )
    return preview, result


def test_registry_for_run_discovers_exactly_six_actions_and_unique_scopes(isolated):
    doc = flow.registry_for_run(_run(), active_profile=None)
    assert len(doc["actions"]) == 6
    for action in doc["actions"]:
        assert action["scopes"] == ["once", "session", "profile"]
        assert len(set(action["scopes"])) == 3
        assert [item["scope"] for item in action["scope_eligibility"]] == [
            "once", "session", "profile"
        ]
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


def test_confirm_is_one_shot_idempotent_persisted_and_does_not_mutate_policy(isolated):
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
    assert policy.session_presets(run["session_key"]) == []
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
    assert policy.session_presets(run["session_key"]) == []
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


def test_profile_apply_creates_exact_backup_and_undo_restores_exact_bytes(isolated):
    isolated.save(profiles.new_profile("work"))
    profile_path = isolated._path("work")
    before = open(profile_path, "rb").read()
    run = _run(meta={"active_profile": "work"})
    preview, result = _confirmed(run, active_profile="work")
    eligibility = next(s for s in preview["scope_eligibility"] if s["scope"] == "profile")
    runs = {"run_corrected": {"id": "run_corrected"}}
    kept = flow.keep_result(
        result["result_id"], "profile", eligibility["prior_hash"], "keep-key-000001",
        get_run=lambda rid: runs.get(rid), replace_run=lambda _run: True, now=102.0,
    )
    backup = kept["transaction"]["policy"]["backup_path"]
    assert open(backup, "rb").read() == before
    assert isolated.load("work")["response_policies"] == ["less-verbose"]
    flow.undo_keep(
        kept["transaction"]["id"],
        get_run=lambda _rid: None, replace_run=lambda _run: False, now=103.0,
    )
    assert open(profile_path, "rb").read() == before


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
