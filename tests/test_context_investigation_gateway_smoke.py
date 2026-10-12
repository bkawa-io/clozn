from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "smoke"
    / "context_investigation_gateway_smoke.py"
)
SPEC = importlib.util.spec_from_file_location(
    "context_investigation_gateway_smoke", SCRIPT,
)
assert SPEC is not None and SPEC.loader is not None
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def _reader(*, observation_id=None):
    measured = observation_id is not None
    return {
        "status": "partial" if measured else "not_measured",
        "measurement": {
            "status": "partial" if measured else "not_measured",
            "experiment_id": "exp_reader",
            "baseline_observation_id": observation_id,
            "completed_source_count": int(measured),
            "unavailable_source_count": 0,
            "failed_source_count": 0,
            "not_measured_source_count": 0 if measured else 1,
        },
        "sources": [{
            "source_id": "src_a",
            "measurement": {
                "status": "measured" if measured else "not_measured",
                "experiment_id": "exp_reader" if measured else None,
                "arm_id": "arm_a" if measured else None,
                "observation_id": observation_id,
            },
        }],
    }


def _cancelled_job(*, executed=1, skipped=2):
    return {
        "state": "cancelled",
        "cancel_requested": True,
        "result": {
            "schema_version": (
                "clozn.context-investigation-effects-search-job-result.v1"
            ),
            "reader_projection": {
                "status": "not_published_cancelled",
                "control_observation_id": None,
                "trusted_singleton_source_ids": [],
            },
            "budget_usage": {
                "actual_work": {
                    "model_measurements_executed_total": executed,
                    "model_measurements_skipped_total": skipped,
                    "saved_work_breakdown": {
                        "scheduler_or_cancellation_skip": skipped,
                    },
                },
            },
        },
    }


def test_cancellation_fixture_has_non_overlapping_exact_source_ranges():
    text, sources = smoke._cancellation_source_fixture("nonce", 8)

    assert len(sources) == 8
    for index, source in enumerate(sources, start=1):
        start, end = source["unicode_range"]
        assert text[start:end].startswith(f"Record {index:02d}:")
        if index > 1:
            assert sources[index - 2]["unicode_range"][1] < start


def test_cancelled_search_gate_requires_partial_work_and_unchanged_reader():
    reader = _reader()

    summary = smoke._validate_cancelled_search(
        _cancelled_job(),
        reader_before=reader,
        reader_after=reader,
        planned_measurements=4,
    )

    assert summary == {
        "model_measurements_executed": 1,
        "model_measurements_skipped": 2,
        "planned_measurements": 4,
        "reader_projection_status": "not_published_cancelled",
    }


@pytest.mark.parametrize(
    ("job", "after", "message"),
    [
        (_cancelled_job(executed=0), _reader(), "before any real model measurement"),
        (_cancelled_job(skipped=0), _reader(), "skipped-work accounting"),
        (_cancelled_job(executed=4), _reader(), "after all 4 planned measurements"),
        (_cancelled_job(), _reader(observation_id="obs_published"), "reader changed"),
    ],
)
def test_cancelled_search_gate_rejects_false_positive_qualification(job, after, message):
    with pytest.raises(smoke.SmokeFailure, match=message):
        smoke._validate_cancelled_search(
            job,
            reader_before=_reader(),
            reader_after=after,
            planned_measurements=4,
        )
