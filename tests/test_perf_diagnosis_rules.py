"""Model-free coverage for clozn.runs.perf_diagnosis -- the versioned rule engine.

Mirrors tests/test_run_diagnosis.py's fixture-dict style (no GPU, no model, no live engine): every case
here is a synthetic run/related_runs dict shaped like what clozn.runs.store.record() produces.
"""
from __future__ import annotations

from clozn.runs import perf_diagnosis, perf_trace
from clozn.runs.perf_diagnosis import RULE_VERSION, diagnose_performance


def _by_rule(entries):
    return {item["rule"]: item for item in entries}


_ALL_RULE_NAMES = {
    "large_context", "slow_decode", "cold_model_load", "client_backpressure",
    "queue_contention", "adapter_reload", "memory_pressure",
}


def test_every_rule_is_always_present_even_on_a_totally_empty_run():
    """The binding discipline: silence for an uninstrumented rule would read as 'nothing wrong here'.
    An empty run must still produce all seven entries, each carrying rule_version and a reason."""
    entries = diagnose_performance({})
    assert len(entries) == 7
    by_rule = _by_rule(entries)
    assert set(by_rule) == _ALL_RULE_NAMES
    for name, item in by_rule.items():
        assert item["rule_version"] == RULE_VERSION
        assert item["status"] == "unavailable", name
        assert item.get("reason"), f"{name} unavailable with no stated reason"


def test_structurally_uninstrumented_rules_are_always_unavailable_regardless_of_run_shape():
    """cold_model_load / client_backpressure / queue_contention / adapter_reload have no evidence path
    in this codebase at all yet -- they must report unavailable even on a rich, fully-populated run."""
    rich_run = {
        "id": "run_rich", "model": "qwen",
        "timing": {"duration_ms": 5000},
        "meta": {"generation_duration_ms": 1000.0, "generation_tokens_per_second": 40.0,
                "cpu_spill_bytes": 0},
        "context_receipt": {"limits": {"prompt_tokens": 100, "context_window_tokens": 8192}},
        "identity": {"model_sha256": "a" * 64},
    }
    by_rule = _by_rule(diagnose_performance(rich_run))
    for name in ("cold_model_load", "client_backpressure", "queue_contention", "adapter_reload"):
        assert by_rule[name]["status"] == "unavailable"
        assert by_rule[name].get("evidence_state") is None
        assert by_rule[name].get("likely_cause") is None


# ------------------------------------------------------------------------------------- large_context

def test_large_context_unavailable_without_context_receipt_limits():
    entry = _by_rule(diagnose_performance({"id": "r", "timing": {"duration_ms": 1000},
                                           "meta": {"generation_duration_ms": 100}}))["large_context"]
    assert entry["status"] == "unavailable"
    assert "prompt_tokens" in entry["reason"]


def test_large_context_unavailable_without_timing_or_decode_duration():
    run = {"id": "r", "context_receipt": {"limits": {"prompt_tokens": 7000, "context_window_tokens": 8192}}}
    entry = _by_rule(diagnose_performance(run))["large_context"]
    assert entry["status"] == "unavailable"
    assert "duration_ms" in entry["reason"]


def test_large_context_not_fired_on_a_small_prompt_with_fast_decode():
    run = {
        "id": "r", "timing": {"duration_ms": 1000},
        "meta": {"generation_duration_ms": 950},
        "context_receipt": {"limits": {"prompt_tokens": 100, "context_window_tokens": 8192}},
    }
    entry = _by_rule(diagnose_performance(run))["large_context"]
    assert entry["status"] == "not_fired"
    assert entry.get("likely_cause") is None
    assert "reason" in entry


def test_large_context_fires_when_prompt_is_large_and_decode_is_a_small_share_of_total():
    run = {
        "id": "r", "timing": {"duration_ms": 20000},
        "meta": {"generation_duration_ms": 2000},   # decode is 10% of the 20s total
        "context_receipt": {"limits": {"prompt_tokens": 7500, "context_window_tokens": 8192}},
    }
    entry = _by_rule(diagnose_performance(run))["large_context"]
    assert entry["status"] == "fired"
    assert entry["evidence_state"] == "correlated"
    assert entry["likely_cause"]
    assert entry["possible_fix"]
    assert entry["evidence"]
    # never overclaims a specific phase it hasn't measured
    assert "prefill" not in entry["likely_cause"].lower() or "does not yet time prefill" in entry["likely_cause"]


# --------------------------------------------------------------------------------------- slow_decode

def test_slow_decode_unavailable_without_a_recorded_rate():
    entry = _by_rule(diagnose_performance({"id": "r", "meta": {}}))["slow_decode"]
    assert entry["status"] == "unavailable"
    assert "generation_tokens_per_second" in entry["reason"]


def test_slow_decode_unavailable_with_fewer_than_two_comparable_peers():
    run = {"id": "r", "model": "qwen", "meta": {"generation_tokens_per_second": 5.0}}
    related = [{"id": "peer1", "model": "qwen", "source": "openai_api",
               "meta": {"generation_tokens_per_second": 40.0}}]
    entry = _by_rule(diagnose_performance(run, related))["slow_decode"]
    assert entry["status"] == "unavailable"
    assert "baseline" in entry["reason"]


def test_slow_decode_ignores_derived_and_mismatched_and_unrelated_peers():
    run = {"id": "r", "model": "qwen", "meta": {"generation_tokens_per_second": 38.0}}
    related = [
        {"id": "peer1", "model": "qwen", "source": "openai_api",
         "meta": {"generation_tokens_per_second": 40.0}},
        {"id": "peer2", "model": "qwen", "source": "replay",           # derived -- excluded
         "meta": {"generation_tokens_per_second": 1.0}},
        {"id": "peer3", "model": "llama", "source": "openai_api",      # different model -- excluded
         "meta": {"generation_tokens_per_second": 1.0}},
        {"id": "r", "model": "qwen", "source": "openai_api",           # self -- excluded
         "meta": {"generation_tokens_per_second": 1.0}},
        {"id": "peer4", "model": "qwen", "source": "openai_api",
         "meta": {"generation_tokens_per_second": 42.0}},
    ]
    entry = _by_rule(diagnose_performance(run, related))["slow_decode"]
    assert entry["status"] == "not_fired"   # 38 tok/s vs a ~41 tok/s median -- not slow
    assert "n=2" in entry["evidence"][-1]["path"]


def test_slow_decode_prefers_model_sha256_over_the_model_label_when_both_have_one():
    """Same displayed model label, different exact model bytes: once both runs carry a real sha256, a
    label-only match would wrongly treat a same-name-different-quant run as a peer. Set up so the WRONG
    behavior (falling back to the label) and the correct behavior (strict sha256 match) disagree: only
    one peer shares the exact hash, which is below _MIN_PEER_RUNS, so the honest answer is unavailable."""
    run = {"id": "r", "model": "qwen-q4", "identity": {"model_sha256": "a" * 64},
          "meta": {"generation_tokens_per_second": 5.0}}
    related = [
        # same label, different sha256 -- must NOT count as a peer once both runs carry a real hash.
        {"id": "p1", "model": "qwen-q4", "identity": {"model_sha256": "b" * 64}, "source": "openai_api",
         "meta": {"generation_tokens_per_second": 40.0}},
        {"id": "p2", "model": "qwen-q4", "identity": {"model_sha256": "a" * 64}, "source": "openai_api",
         "meta": {"generation_tokens_per_second": 42.0}},
    ]
    entry = _by_rule(diagnose_performance(run, related))["slow_decode"]
    assert entry["status"] == "unavailable"   # p1 correctly excluded -> only 1 real peer, below _MIN_PEER_RUNS
    assert "1 found" in entry["reason"]


def test_slow_decode_fires_when_well_below_the_peer_median():
    run = {"id": "r", "model": "qwen", "meta": {"generation_tokens_per_second": 5.0}}
    related = [
        {"id": "p1", "model": "qwen", "source": "openai_api", "meta": {"generation_tokens_per_second": 40.0}},
        {"id": "p2", "model": "qwen", "source": "openai_api", "meta": {"generation_tokens_per_second": 42.0}},
    ]
    entry = _by_rule(diagnose_performance(run, related))["slow_decode"]
    assert entry["status"] == "fired"
    assert entry["evidence_state"] == "correlated"
    assert "5.0" in entry["likely_cause"]
    assert entry["possible_fix"]


# ---------------------------------------------------------------------------------- memory_pressure

def test_memory_pressure_unavailable_without_cpu_spill_fields():
    entry = _by_rule(diagnose_performance({"id": "r", "meta": {}}))["memory_pressure"]
    assert entry["status"] == "unavailable"


def test_memory_pressure_not_fired_when_worker_explicitly_recorded_no_spill():
    entry = _by_rule(diagnose_performance({"id": "r", "meta": {"cpu_spill_bytes": 0}}))["memory_pressure"]
    assert entry["status"] == "not_fired"
    assert entry["evidence"]


def test_memory_pressure_fires_on_a_recorded_spill():
    entry = _by_rule(diagnose_performance({"id": "r", "meta": {"cpu_spill_bytes": 512_000_000}}))["memory_pressure"]
    assert entry["status"] == "fired"
    assert entry["evidence_state"] == "observed"
    assert entry["possible_fix"]


# ------------------------------------------------------------------------------- rule_version + composition

def test_rule_version_is_pinned_and_shared_across_the_module():
    assert RULE_VERSION == "1"
    for entry in diagnose_performance({}):
        assert entry["rule_version"] == "1"


def test_build_performance_report_composes_trace_and_diagnoses_and_validates():
    run = {
        "id": "run_full", "model": "qwen",
        "timing": {"duration_ms": 5000},
        "meta": {"generation_duration_ms": 1000.0, "generation_tokens_per_second": 40.0},
        "context_receipt": {"limits": {"prompt_tokens": 100, "context_window_tokens": 8192}},
    }
    report = perf_diagnosis.build_performance_report(run)
    assert report["schema_version"] == "clozn.performance-trace.v1"
    assert report["run_id"] == "run_full"
    assert len(report["diagnoses"]) == 7
    # already validated inside attach_diagnoses(), but assert again explicitly as a regression guard.
    from clozn import schemas
    schemas.validate(report)


def test_build_performance_report_never_raises_on_a_crashed_run():
    report = perf_diagnosis.build_performance_report(
        {"id": "run_crashed", "error": "worker_exited"}, related_runs=[{"malformed": True}])
    assert report["run_id"] == "run_crashed"
    assert len(report["diagnoses"]) == 7
