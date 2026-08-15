from clozn.runs.parent_anchor_geometry import aggregate_batches, aggregate_case, build_probe_row, exact_lcp


def test_exact_lcp_and_first_changed_index_are_token_exact():
    assert exact_lcp((1, 2, 3), (1, 2, 4, 5)) == 2
    row = build_probe_row(
        ordinal=2,
        stage="coarse",
        batch_id=1,
        parent_source_ids=("a", "b"),
        child_source_ids=("a",),
        parent_token_ids=(10, 11, 12, 13),
        child_token_ids=(10, 11, 99),
        preserved=False,
        accepted_as_best=False,
    )
    assert row["exact_lcp_tokens"] == 2
    assert row["first_changed_token_index"] == 2
    assert row["required_child_suffix_rows"] == 1


def test_batch_models_count_parent_once_and_persistent_parent_zero():
    rows = [
        build_probe_row(
            ordinal=2, stage="coarse", batch_id=1,
            parent_source_ids=("a", "b"), child_source_ids=("a",),
            parent_token_ids=(1, 2, 3, 4), child_token_ids=(1, 2, 8),
            preserved=True, accepted_as_best=True,
        ),
        build_probe_row(
            ordinal=3, stage="coarse", batch_id=1,
            parent_source_ids=("a", "b"), child_source_ids=("b",),
            parent_token_ids=(1, 2, 3, 4), child_token_ids=(1, 2, 9, 10),
            preserved=False, accepted_as_best=False,
        ),
    ]
    batch = aggregate_batches(rows)[0]["row_models"]
    assert batch["naive_logical_child_rows"] == 7
    assert batch["total_lcp_rows"] == 4
    assert batch["total_suffix_rows"] == 3
    assert batch["request_local_parent_ideal_rows"] == 7
    assert batch["persistent_parent_ideal_rows"] == 3
    assert batch["request_local_reduction_percent"] == 0.0
    assert batch["persistent_reduction_percent"] == round(100 * 4 / 7, 6)


def test_aggregate_case_reports_empty_refine_stage_and_deterministic_distribution():
    row = build_probe_row(
        ordinal=2, stage="inclusion", batch_id=2,
        parent_source_ids=("a",), child_source_ids=(),
        parent_token_ids=(1, 2, 3), child_token_ids=(1, 2),
        preserved=False, accepted_as_best=False,
    )
    aggregate = aggregate_case([row])
    assert aggregate["search_batch_count"] == 1
    assert aggregate["by_stage"]["refine"]["probe_count"] == 0
    assert aggregate["lcp_distribution_tokens"] == {"min": 2, "median": 2, "p90": 2, "max": 2}
    assert aggregate["current_native_physical_rows"] is None
