"""Model-free coverage for clozn.runs.perf_trace.build_trace() / attach_diagnoses()."""
from __future__ import annotations

import pytest

from clozn.runs import perf_trace


def test_minimal_run_yields_only_the_required_fields():
    """A run with no timing/meta/identity evidence must still produce a VALID document -- absence of a
    phase/metric is honest, not an error."""
    doc = perf_trace.build_trace({"id": "run_sparse"})
    assert doc == {
        "schema_version": "clozn.performance-trace.v1",
        "run_id": "run_sparse",
        "clock": "monotonic",
    }


def test_run_missing_an_id_raises_rather_than_producing_an_unattributable_trace():
    with pytest.raises(perf_trace.PerfTraceError, match="no non-empty string id"):
        perf_trace.build_trace({"meta": {"generation_duration_ms": 100}})
    with pytest.raises(perf_trace.PerfTraceError):
        perf_trace.build_trace({"id": "   "})


def test_decode_phase_is_derived_from_the_worker_measured_generation_timing():
    run = {
        "id": "run_decode",
        "meta": {"generation_duration_ms": 1500.0, "generation_tokens_per_second": 18.2,
                 "generation_tokens": 512},
        "context_receipt": {"limits": {"prompt_tokens": 7411, "context_window_tokens": 8192,
                                       "generated_tokens": 512}},
        "timing": {"started_at": 100.0, "ended_at": 124.1, "duration_ms": 24100},
    }
    doc = perf_trace.build_trace(run)
    assert doc["phases"] == [
        {"name": "decode", "owner": "clozn_worker", "duration_ns": 1_500_000_000}
    ]
    assert doc["metrics"] == {
        "prompt_tokens": 7411,
        "generated_tokens": 512,
        "decode_tokens_per_second": 18.2,
        "wall_clock_total_ms": 24100.0,
    }
    # decode has no known offset within the request -- nothing upstream of it is timed yet.
    assert "start_ns" not in doc["phases"][0]


def test_no_generation_duration_means_no_phases_key_at_all():
    """Unmeasured != a zero-duration phase. Confirms load/prefill/queue stay entirely absent, matching
    the survey finding that no production code path writes load_duration_ms/prefill_duration_ms."""
    doc = perf_trace.build_trace({"id": "run_no_decode", "meta": {"device": "cuda"},
                                  "timing": {"duration_ms": 500}})
    assert "phases" not in doc
    assert doc["metrics"] == {"wall_clock_total_ms": 500.0}


def test_machine_and_engine_identity_are_read_from_the_run_identity_block():
    run = {
        "id": "run_identity",
        "identity": {
            "model_sha256": "a" * 64, "engine_build": "clozn-server-dev", "clozn_version": "0.9",
            "ext": {"machine": {"os": "Windows", "cpu_count": 16}, "adapter": {"name": "not-ours"}},
        },
    }
    doc = perf_trace.build_trace(run)
    assert doc["machine_identity"] == {"os": "Windows", "cpu_count": 16}
    assert doc["engine_identity"] == {
        "model_sha256": "a" * 64, "engine_build": "clozn-server-dev", "clozn_version": "0.9",
    }
    # another feature's namespace under ext must never leak into this artifact.
    assert "adapter" not in doc.get("machine_identity", {})


def test_absent_ext_or_machine_namespace_omits_machine_identity_entirely():
    doc = perf_trace.build_trace({"id": "run_no_machine", "identity": {"clozn_version": "0.9"}})
    assert "machine_identity" not in doc
    assert doc["engine_identity"] == {"clozn_version": "0.9"}


def test_errored_run_still_produces_a_valid_partial_trace():
    """Worker crash / partial run: build_trace() must not raise just because the run carries an error --
    it degrades to whatever evidence exists, same as diagnosis.py already does."""
    run = {"id": "run_crashed", "error": "worker_exited", "finish_reason": None,
          "timing": {"started_at": 5.0, "ended_at": 5.2, "duration_ms": 200}}
    doc = perf_trace.build_trace(run)
    assert doc["run_id"] == "run_crashed"
    assert doc["metrics"] == {"wall_clock_total_ms": 200.0}
    assert "phases" not in doc


def test_light_capture_tier_run_still_produces_a_valid_trace():
    """capture_mode='light' drops the per-token trace but not meta/timing -- build_trace() never reads
    run['trace'] at all, so a light-tier run degrades the same way a standard-tier one does."""
    run = {"id": "run_light", "meta": {"capture_tier": "light", "generation_duration_ms": 900.0},
          "timing": {"duration_ms": 1200}}
    doc = perf_trace.build_trace(run)
    assert doc["phases"] == [{"name": "decode", "owner": "clozn_worker", "duration_ns": 900_000_000}]


def test_attach_diagnoses_is_a_separate_optional_layer():
    trace = perf_trace.build_trace({"id": "run_x"})
    assert "diagnoses" not in trace

    with_diagnoses = perf_trace.attach_diagnoses(trace, [
        {"rule": "cold_model_load", "rule_version": "1", "status": "unavailable",
         "reason": "no load_duration_ms recorded"},
    ])
    assert with_diagnoses["diagnoses"][0]["rule"] == "cold_model_load"
    # the original document is untouched -- attach_diagnoses returns a copy.
    assert "diagnoses" not in trace


def test_attach_diagnoses_ignores_non_mapping_entries_and_empty_list():
    trace = perf_trace.build_trace({"id": "run_y"})
    assert "diagnoses" not in perf_trace.attach_diagnoses(trace, [])
    assert "diagnoses" not in perf_trace.attach_diagnoses(trace, ["not a dict", 5, None])
