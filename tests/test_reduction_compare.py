from clozn.analysis.reduction_compare import compare_reduction


def _run(run_id, *, tokens, response="answer", finish="stop", parent=None, removed=()):
    run = {
        "id": run_id,
        "response": response,
        "finish_reason": finish,
        "identity": {
            "model_sha256": "a" * 64,
            "template_fingerprint": "b" * 64,
            "engine_build": "engine-v1",
        },
        "context_receipt": {"rendered": {"tokens": tokens}},
        "context_units": {"default_source_ids": ["src_a", "src_b", "src_c"]},
    }
    if parent:
        run.update({
            "parent_run_id": parent,
            "context_units": {"default_source_ids": ["src_a"]},
            "changes_applied": {"experiment": {
                "intervention": {"kind": "delete_source", "source_ids": list(removed)},
                "derived_provenance": {"minimal_context": {
                    "retained_source_ids": ["src_a"],
                    "removed_source_ids": list(removed),
                }},
            }},
        })
    return run


def test_comparison_is_narrow_and_reports_v1_reduction_facts():
    parent = _run("run_parent", tokens=20)
    child = _run("run_child", tokens=11, parent="run_parent", removed=("src_b", "src_c"))

    result = compare_reduction(parent, child)

    assert result["ok"] is True
    assert result["context"] == {
        "parent_source_ids": ["src_a", "src_b", "src_c"],
        "retained_source_ids": ["src_a"],
        "removed_source_ids": ["src_b", "src_c"],
    }
    assert result["rendered_prompt_tokens"] == {
        "parent": 20, "child": 11, "saved": 9, "improved": True,
    }
    assert result["preservation"] == {
        "answer_exact_match": True, "finish_reason_match": True, "exact": True,
    }
    assert result["generation_identity"]["match"] is True


def test_comparison_rejects_an_arbitrary_run_pair():
    result = compare_reduction(_run("run_parent", tokens=20), _run("run_other", tokens=10))

    assert result["ok"] is False
    assert result["error"]["code"] == "not_reduction_child"


def test_comparison_does_not_hide_answer_or_stop_drift():
    parent = _run("run_parent", tokens=20)
    child = _run(
        "run_child", tokens=11, response="different", finish="length",
        parent="run_parent", removed=("src_b",),
    )

    result = compare_reduction(parent, child)

    assert result["preservation"] == {
        "answer_exact_match": False, "finish_reason_match": False, "exact": False,
    }
