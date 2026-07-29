"""Unit tests for clozn.triage.artifact.build_triage_artifact: assembly + schema validation + the
explicit not_run placeholders for every step family this build does not implement."""
from __future__ import annotations

import pytest

from clozn import schemas
from clozn.triage.artifact import ALL_FAMILIES, build_triage_artifact, unimplemented_steps


def _run(run_id, **identity):
    return {"id": run_id, "identity": identity, "context_receipt": {}}


def test_minimal_artifact_validates_against_its_own_schema():
    doc = build_triage_artifact(
        baseline_run=_run("run_b", model_sha256="a" * 64),
        candidate_run=_run("run_c", model_sha256="a" * 64),
        families=["identity", "context"],
    )
    schemas.validate(doc, "clozn.triage.v1")
    assert doc["schema_version"] == "clozn.triage.v1"
    assert doc["baseline_run_id"] == "run_b"
    assert doc["candidate_run_id"] == "run_c"


def test_default_families_include_every_declared_step_family():
    doc = build_triage_artifact(baseline_run=_run("run_b"), candidate_run=_run("run_c"))
    kinds = {s["kind"] for s in doc["steps"]}
    assert any(k.startswith("identity_diff:") for k in kinds)
    assert any(k.startswith("context_diff:") for k in kinds)
    assert "template_swap" in kinds        # replay family, always not_run in this build
    assert any(kind.startswith("quant_export_diff:") for kind in kinds)
    assert any(kind.startswith("tool_contract_diff:") for kind in kinds)
    assert "internal_localization" in kinds
    unimplemented_kinds = {"template_swap", "context_swap", "sampling_swap",
                           "internal_localization"}
    for kind in unimplemented_kinds:
        step = next(s for s in doc["steps"] if s["kind"] == kind)
        assert step["status"] == "not_run"
        assert step["reason"]


def test_narrow_families_selection_yields_no_not_run_placeholders():
    doc = build_triage_artifact(baseline_run=_run("run_b"), candidate_run=_run("run_c"),
                                families=["identity"])
    kinds = {s["kind"] for s in doc["steps"]}
    assert all(k.startswith("identity_diff:") for k in kinds)


def test_unknown_family_raises_rather_than_silently_skipping():
    with pytest.raises(ValueError):
        build_triage_artifact(baseline_run=_run("run_b"), candidate_run=_run("run_c"),
                              families=["not_a_real_family"])


def test_missing_run_id_on_both_sides_raises_rather_than_writing_an_empty_id():
    with pytest.raises(ValueError):
        build_triage_artifact(baseline_run={"identity": {}}, candidate_run={"identity": {}})


def test_explicit_run_id_overrides_when_the_run_dict_lacks_one():
    doc = build_triage_artifact(baseline_run={"identity": {}}, candidate_run={"identity": {}},
                                baseline_run_id="run_x", candidate_run_id="run_y",
                                families=["identity"])
    assert doc["baseline_run_id"] == "run_x"
    assert doc["candidate_run_id"] == "run_y"


def test_source_experiment_id_and_case_id_are_omitted_when_not_given():
    doc = build_triage_artifact(baseline_run=_run("run_b"), candidate_run=_run("run_c"),
                                families=["identity"])
    assert "source_experiment_id" not in doc
    assert "case_id" not in doc


def test_source_experiment_id_and_case_id_are_carried_through_when_given():
    doc = build_triage_artifact(baseline_run=_run("run_b"), candidate_run=_run("run_c"),
                                source_experiment_id="exp_1", case_id="greeting",
                                families=["identity"])
    assert doc["source_experiment_id"] == "exp_1"
    assert doc["case_id"] == "greeting"
    schemas.validate(doc, "clozn.triage.v1")


def test_deep_flag_and_budgets_are_visible_on_not_run_placeholders():
    steps = unimplemented_steps("replay", deep_requested=True, max_runs=4, max_seconds=30.0)
    for step in steps:
        assert step["inputs"]["deep_requested"] is True
        assert step["inputs"]["max_runs"] == 4
        assert step["inputs"]["max_seconds"] == 30.0
        assert "--deep was requested" in step["reason"]


def test_unimplemented_steps_rejects_an_unknown_family():
    with pytest.raises(ValueError):
        unimplemented_steps("not_a_real_family")


def test_all_families_covers_both_implemented_and_stub_families():
    assert set(ALL_FAMILIES) == {
        "identity", "context", "sampling", "replay", "quant_export", "tool_contract",
        "internal_localization",
    }


def test_summary_is_recomputable_and_matches_rules_classify_directly():
    from clozn.triage.rules import classify
    doc = build_triage_artifact(
        baseline_run=_run("run_b", model_sha256="a" * 64),
        candidate_run=_run("run_c", model_sha256="b" * 64),
        families=["identity"],
    )
    assert doc["summary"] == classify(doc["steps"])


def _controlled_document(*, status="completed", test_status="causally_supported"):
    return {
        "schema_version": "clozn.run-change-test.v1",
        "generated_at": "2026-07-28T00:00:00+00:00",
        "run_a": "run_b", "run_b": "run_c", "status": status, "dry_run": False,
        "match_criterion": {"kind": "exact_output", "note": "exact hashes only"},
        "budget": {
            "max_runs": 2, "max_seconds": 30, "runs_used": 2, "duration_ms": 5,
            "remaining_runs": 0, "remaining_seconds": 29.995,
            **({"stop_reason": "budget_exhausted"} if status == "budget_exhausted" else {}),
        },
        "tests": [{
            "kind": "context", "status": test_status, "ran": True, "runs_used": 2,
            "duration_ms": 5,
            "budget": {
                "max_runs": 2, "max_seconds": 30, "runs_used": 2, "duration_ms": 5,
                "remaining_runs": 0, "remaining_seconds": 29.995,
                **({"stop_reason": "budget_exhausted"} if status == "budget_exhausted" else {}),
            },
            "evidence": [
                {"run_id": "run_control", "arm": "control"},
                {"run_id": "run_treatment", "arm": "treatment"},
            ],
            "arms": [
                {"name": "control", "status": "completed", "duration_ms": 2,
                 "match_target": "candidate", "run_id": "run_control"},
                {"name": "treatment", "status": "completed", "duration_ms": 3,
                 "match_target": "baseline", "run_id": "run_treatment"},
            ],
            "reason": "controlled recovery",
        }],
        "summary": {
            "classification": "context" if test_status == "causally_supported" else "undetermined",
            "causally_supported": ["context"] if test_status == "causally_supported" else [],
            "entangled": False,
        },
    }


def test_controlled_test_artifact_maps_child_runs_and_cost_into_triage_steps():
    controlled = _controlled_document()
    doc = build_triage_artifact(
        baseline_run=_run("run_b"), candidate_run=_run("run_c"),
        families=["replay"], max_runs=2, max_seconds=30,
        controlled_test_artifact=controlled,
    )
    step = doc["steps"][0]
    assert step["kind"] == "context_swap"
    assert step["status"] == "causally_supported"
    assert step["runs_used"] == 2
    assert step["artifact_refs"] == ["run_control", "run_treatment"]
    assert doc["summary"]["classification"] == "context_swap"
    assert doc["controlled_tests"] == controlled


def test_budget_exhaustion_marks_later_unimplemented_tiers():
    controlled = _controlled_document(status="budget_exhausted", test_status="inconclusive")
    doc = build_triage_artifact(
        baseline_run=_run("run_b"), candidate_run=_run("run_c"),
        families=["replay", "internal_localization"], deep=True,
        max_runs=2, max_seconds=30, controlled_test_artifact=controlled,
    )
    later = [s for s in doc["steps"] if s["kind"] == "internal_localization"]
    assert later
    assert all(s["stop_reason"] == "budget_exhausted" for s in later)
    assert all(s["reason"].startswith("budget_exhausted") for s in later)
