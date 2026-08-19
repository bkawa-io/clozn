"""Model-free tests for runlog.lineage(), the branch/replay tree read model."""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import clozn.runs.store as runlog  # noqa: E402


@pytest.fixture
def store(tmp_path):
    original = runlog.RUNS_DIR
    runlog.RUNS_DIR = str(tmp_path / "runs")
    try:
        yield runlog
    finally:
        runlog.RUNS_DIR = original


def _record(store, prompt, response, started, parent=None, changes=None):
    return store.record(
        source="replay" if parent else "studio_chat",
        client="studio",
        model="clozn-qwen",
        messages=[{"role": "user", "content": prompt}],
        response=response,
        parent_run_id=parent,
        changes_applied=changes,
        started=started,
        ended=started + 0.25,
    )


def test_lineage_returns_ancestors_siblings_children_and_tree(store):
    root = _record(store, "root prompt", "root reply", 1000.0)
    child_a = _record(store, "child a", "a", 1001.0, root, {"plain": True})
    child_b = _record(store, "child b", "b", 1002.0, root, {"corrective_retry": {"preset": "shorter", "arm": "candidate"}})
    grand = _record(store, "grand child", "g", 1003.0, child_a,
                    {"branch_turn": 1, "edited_user": True, "kv_snapshot": True})

    out = store.lineage(grand)

    assert out["run_id"] == grand
    assert out["root_id"] == root
    assert out["original"]["id"] == root
    assert out["current"]["id"] == grand
    assert [n["id"] for n in out["ancestors"]] == [root, child_a]
    assert [n["id"] for n in out["siblings"]] == []
    assert out["children"] == []
    assert out["tree"]["id"] == root
    assert [n["id"] for n in out["tree"]["children"]] == [child_a, child_b]
    assert out["tree"]["children"][0]["children"][0]["id"] == grand
    assert out["tree"]["children"][0]["children"][0]["is_current"] is True
    assert out["tree"]["children"][0]["change_label"] == "re-roll"
    assert out["tree"]["children"][1]["change_label"] == "retry shorter (candidate)"
    assert out["current"]["change_label"] == "branched from turn 1 (edited question) + KV snapshot"


def test_lineage_reports_siblings_for_child_runs(store):
    root = _record(store, "root", "root", 2000.0)
    child_a = _record(store, "child a", "a", 2001.0, root, {"plain": True})
    child_b = _record(store, "child b", "b", 2002.0, root, {"corrective_retry": {"preset": "shorter", "arm": "candidate"}})

    out = store.lineage(child_a)

    assert [n["id"] for n in out["siblings"]] == [child_b]
    assert out["siblings"][0]["change_label"] == "retry shorter (candidate)"


def test_lineage_missing_run_returns_none(store):
    assert store.lineage("run_missing") is None
