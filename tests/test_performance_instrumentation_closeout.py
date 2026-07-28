"""Model-free closeout coverage for protocol-native performance instrumentation."""
from __future__ import annotations

from clozn.runs.perf_diagnosis import build_performance_report, diagnose_performance
from clozn.runs.perf_trace import build_trace
from clozn.runs.trace import generation_timing_from_frames


def _worker_timing(*phases, metrics=None):
    return {
        "schema_version": "clozn.worker-timing.v1",
        "unit": "nanoseconds",
        "clock": "steady_clock",
        "clock_owner": "clozn_worker",
        "phases": list(phases),
        "metrics": dict(metrics or {}),
    }


def _gateway_timing(*phases):
    return {
        "schema_version": "clozn.gateway-timing.v1",
        "unit": "nanoseconds",
        "clock": "monotonic",
        "clock_owner": "clozn_gateway",
        "phases": list(phases),
    }


def _phase(name, duration_ns, *, owner="clozn_worker", aggregation="exclusive", **extra):
    return {
        "name": name,
        "owner": owner,
        "duration_ns": duration_ns,
        "measurement": "measured",
        "aggregation": aggregation,
        **extra,
    }


def _by_rule(entries):
    return {entry["rule"]: entry for entry in entries}


def test_new_worker_timing_keeps_prefill_and_decode_separate_and_decode_excludes_prefill():
    frames = [{
        "type": "gen_finished",
        "wall_ms": 110,
        "new_tokens": 10,
        "steps_total": 10,
        "tok_per_s": 90.9,
        "timing": _worker_timing(
            _phase("prefill", 10_000_000, start_ns=0),
            _phase("decode", 100_000_000, start_ns=10_000_000),
            metrics={"prompt_tokens_per_second": 2000.0, "decode_tokens_per_second": 100.0},
        ),
    }]
    parsed = generation_timing_from_frames(frames)
    assert parsed["prefill_duration_ms"] == 10
    assert parsed["generation_duration_ms"] == 100
    assert parsed["generation_tokens_per_second"] == 100
    assert parsed["worker_timing"]["phases"][0]["start_ns"] == 0


def test_old_worker_terminal_frame_remains_compatible():
    assert generation_timing_from_frames([{
        "type": "gen_finished", "wall_ms": 25.0, "new_tokens": 2,
        "steps_total": 2, "tok_per_s": 80.0,
    }]) == {
        "generation_duration_ms": 25.0,
        "generation_tokens": 2,
        "generation_steps": 2,
        "generation_tokens_per_second": 80.0,
    }


def test_zero_duration_is_measured_while_absent_phase_stays_absent():
    run = {
        "id": "run_zero",
        "meta": {"worker_timing": _worker_timing(_phase("prefill", 0))},
        "timing": {"duration_ms": 2},
    }
    trace = build_trace(run)
    assert trace["phases"][0]["duration_ns"] == 0
    assert trace["phases"][0]["measurement"] == "measured"
    assert not any(phase["name"] == "decode" for phase in trace["phases"])


def test_clock_domains_are_not_aligned_and_overlaps_are_not_double_counted():
    run = {
        "id": "run_domains",
        "meta": {
            "gateway_timing": _gateway_timing(
                _phase("gateway_queue", 10_000_000, owner="clozn_gateway", start_ns=0),
                _phase("worker_dispatch", 90_000_000, owner="clozn_gateway",
                       aggregation="overlapping"),
                _phase("gateway_serialize", 2_000_000, owner="clozn_gateway",
                       aggregation="context_only"),
            ),
            "worker_timing": _worker_timing(
                _phase("prefill", 20_000_000, start_ns=0),
                _phase("decode", 50_000_000, start_ns=20_000_000),
            ),
        },
        "timing": {"duration_ms": 100},
    }
    trace = build_trace(run)
    assert {phase["clock_domain"] for phase in trace["phases"]} == {
        "clozn_gateway:monotonic", "clozn_worker:steady_clock",
    }
    assert trace["aggregation"] == {
        "known_duration_ns": 80_000_000,
        "phase_count": 5,
        "exclusive_phase_count": 3,
        "wall_clock_total_ns": 100_000_000,
        "unaccounted_duration_ns": 20_000_000,
        "consistency": "consistent",
        "measurement_coverage": 0.8,
    }


def test_cold_first_run_evaluates_load_and_warm_run_does_not_invent_one():
    cold = {
        "id": "cold",
        "meta": {"worker_timing": _worker_timing(
            _phase("model_load", 500_000_000, owner="model_load",
                   aggregation="context_only", scope="process_startup"),
        )},
    }
    warm = {"id": "warm", "meta": {"worker_timing": _worker_timing(_phase("decode", 1))}}
    assert _by_rule(diagnose_performance(cold))["cold_model_load"]["status"] == "fired"
    assert _by_rule(diagnose_performance(warm))["cold_model_load"]["status"] == "unavailable"


def test_long_prompt_uses_measured_prefill_instead_of_allocating_unknown_time():
    run = {
        "id": "long",
        "context_receipt": {"limits": {"prompt_tokens": 7500, "context_window_tokens": 8192}},
        "meta": {
            "generation_duration_ms": 100,
            "worker_timing": _worker_timing(_phase("prefill", 700_000_000)),
        },
        "timing": {"duration_ms": 1000},
    }
    finding = _by_rule(diagnose_performance(run))["large_context"]
    assert finding["status"] == "fired"
    assert any(item["path"] == "phases.prefill.duration_ms" for item in finding["evidence"])


def test_partial_stream_flush_can_evaluate_backpressure_without_claiming_causality():
    run = {
        "id": "cancelled",
        "error": "client disconnected",
        "meta": {"gateway_timing": _gateway_timing(
            _phase("stream_flush", 300_000_000, owner="client", aggregation="overlapping"),
        )},
        "timing": {"duration_ms": 1000},
    }
    finding = _by_rule(diagnose_performance(run))["client_backpressure"]
    assert finding["status"] == "fired"
    assert finding["evidence_state"] == "correlated"


def test_decode_regression_baseline_does_not_compare_cpu_to_gpu():
    run = {
        "id": "candidate", "model": "m",
        "meta": {"generation_tokens_per_second": 5.0, "device": "cpu", "gpu_layers": 0},
    }
    peers = [
        {"id": "cpu1", "model": "m",
         "meta": {"generation_tokens_per_second": 10.0, "device": "cpu", "gpu_layers": 0}},
        {"id": "cpu2", "model": "m",
         "meta": {"generation_tokens_per_second": 11.0, "device": "cpu", "gpu_layers": 0}},
        {"id": "gpu1", "model": "m",
         "meta": {"generation_tokens_per_second": 100.0, "device": "cuda", "gpu_layers": 99}},
    ]
    finding = _by_rule(diagnose_performance(run, peers))["slow_decode"]
    assert finding["status"] == "fired"
    assert "10.5 tok/s" in finding["likely_cause"]


def test_report_exposes_regression_attribution_and_consistent_aggregation():
    run = {
        "id": "report",
        "model": "m",
        "meta": {
            "generation_tokens_per_second": 5.0,
            "gateway_timing": _gateway_timing(
                _phase("gateway_queue", 200_000_000, owner="clozn_gateway"),
            ),
        },
        "timing": {"duration_ms": 500},
    }
    report = build_performance_report(run)
    assert report["aggregation"]["unaccounted_duration_ns"] == 300_000_000
    assert report["regression_attribution"]["status"] == "attributed"
    assert "queue_contention" in report["regression_attribution"]["rules"]


def test_ollama_timing_fields_are_only_emitted_from_measured_worker_phases():
    from clozn.server.ndjson import _timing_fields

    old_worker = _timing_fields(1.0, 1.1, [], 5)
    assert "prompt_eval_duration" not in old_worker
    assert "eval_duration" not in old_worker
    assert "load_duration" not in old_worker

    measured = _timing_fields(1.0, 1.1, [], 5, {
        "prefill_duration_ms": 2.5,
        "generation_duration_ms": 75.0,
        "worker_timing": _worker_timing(
            _phase("model_load", 300_000_000, owner="model_load", aggregation="context_only"),
        ),
    })
    assert measured["prompt_eval_duration"] == 2_500_000
    assert measured["eval_duration"] == 75_000_000
    assert measured["load_duration"] == 300_000_000
