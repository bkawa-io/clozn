"""Tests for clozn.replay.execution_fork_results -- immutable terminal execution-fork receipt storage,
focused on the read-only `list_for_parent` helper Rewind Fidelity (E10) consumes.
"""
from __future__ import annotations

import os

from clozn.replay import execution_fork_results as efr


def _receipt(execution_id, parent_run_id, *, phase="completed"):
    return {
        "schema_version": "clozn.execution-fork.v1",
        "plan_id": "fork_plan_" + ("0" * 20),
        "execution_id": execution_id,
        "phase": phase,
        "classification": "exact_execution_fork",
        "parent_run_id": parent_run_id,
        "parent_fingerprint_sha256": "a" * 64,
        "request": {"position": 1, "change": {"type": "none"},
                   "execution_change": {"type": "none"}, "change_sha256": "b" * 64},
        "identity": {},
        "exactness": {"regime": "generated_token_live_kv", "source": "live_kv",
                     "proof_status": "confirmed", "truncate_to": 11, "boundary_shape_true": True},
        "unavoidable_differences": [],
        "unchanged_control": {"required": True, "status": "matched",
                              "result": {"status": "matched", "exact_match": True}},
        "child_lineage": {"parent_run_id": parent_run_id, "source": "fork",
                          "change_sha256": "b" * 64, "receipt_status": "created"},
        "execution": {"status": "succeeded", "started_ts": 1.0, "ended_ts": 2.0},
        "reasons": [{"code": "execution_succeeded", "message": "ok"}],
    }


def test_list_for_parent_on_missing_db_returns_empty_and_creates_nothing(tmp_path, monkeypatch):
    results_dir = tmp_path / "execution-forks"
    monkeypatch.setattr(efr, "RESULTS_DIR", str(results_dir))
    assert efr.list_for_parent("run_x") == []
    assert not results_dir.exists()
    assert not os.path.isfile(os.path.join(str(results_dir), "results.sqlite3"))


def test_list_for_parent_filters_by_parent(tmp_path, monkeypatch):
    monkeypatch.setattr(efr, "RESULTS_DIR", str(tmp_path / "execution-forks"))
    efr.save(_receipt("fork_exec_" + "a" * 20, "run_a"))
    efr.save(_receipt("fork_exec_" + "b" * 20, "run_a"))
    efr.save(_receipt("fork_exec_" + "c" * 20, "run_b"))

    result_a = efr.list_for_parent("run_a")
    result_b = efr.list_for_parent("run_b")
    assert {r["execution_id"] for r in result_a} == {"fork_exec_" + "a" * 20, "fork_exec_" + "b" * 20}
    assert [r["execution_id"] for r in result_b] == ["fork_exec_" + "c" * 20]
    assert efr.list_for_parent("run_nonexistent") == []


def test_list_for_parent_deterministic_ordering(tmp_path, monkeypatch):
    monkeypatch.setattr(efr, "RESULTS_DIR", str(tmp_path / "execution-forks"))
    efr.save(_receipt("fork_exec_" + "c" * 20, "run_a"))
    efr.save(_receipt("fork_exec_" + "a" * 20, "run_a"))
    efr.save(_receipt("fork_exec_" + "b" * 20, "run_a"))

    first = [r["execution_id"] for r in efr.list_for_parent("run_a")]
    second = [r["execution_id"] for r in efr.list_for_parent("run_a")]
    assert first == second
    assert first == sorted(first)


def test_list_for_parent_never_mutates_stored_payload(tmp_path, monkeypatch):
    monkeypatch.setattr(efr, "RESULTS_DIR", str(tmp_path / "execution-forks"))
    saved = efr.save(_receipt("fork_exec_" + "a" * 20, "run_a"))

    results = efr.list_for_parent("run_a")
    results[0]["phase"] = "tampered"
    results[0]["execution_id"] = "tampered"

    reread = efr.get("fork_exec_" + "a" * 20)
    assert reread == saved
    assert reread["phase"] == "completed"


def test_list_for_parent_skips_malformed_rows_without_raising(tmp_path, monkeypatch):
    import sqlite3

    monkeypatch.setattr(efr, "RESULTS_DIR", str(tmp_path / "execution-forks"))
    efr.save(_receipt("fork_exec_" + "a" * 20, "run_a"))

    db = sqlite3.connect(efr._path())
    db.execute(
        "INSERT INTO execution_fork_results (execution_id, parent_run_id, phase, payload_json) "
        "VALUES (?, ?, ?, ?)",
        ("fork_exec_" + "b" * 20, "run_a", "completed", "not valid json{"),
    )
    db.execute(
        "INSERT INTO execution_fork_results (execution_id, parent_run_id, phase, payload_json) "
        "VALUES (?, ?, ?, ?)",
        ("fork_exec_" + "c" * 20, "run_a", "completed", "42"),  # valid JSON, not an object
    )
    db.commit()
    db.close()

    results = efr.list_for_parent("run_a")
    assert [r["execution_id"] for r in results] == ["fork_exec_" + "a" * 20]


def test_list_for_parent_rejects_non_string_and_empty_parent_id(tmp_path, monkeypatch):
    monkeypatch.setattr(efr, "RESULTS_DIR", str(tmp_path / "execution-forks"))
    assert efr.list_for_parent("") == []
    assert efr.list_for_parent(None) == []
