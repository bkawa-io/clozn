"""Model-free coverage for clozn.runs.perf_spans.SpanRecorder -- the monotonic span helper future phase
instrumentation (queue wait, model load, prefill) will build on. Nothing in the request path calls this
yet; these tests cover the helper in isolation.
"""
from __future__ import annotations

import statistics
import time

import pytest

from clozn.runs.perf_spans import SpanRecorder, known_owner


def test_a_single_span_records_the_expected_shape():
    rec = SpanRecorder()
    with rec.span("decode", owner="clozn_worker"):
        pass
    assert len(rec.spans) == 1
    span = rec.spans[0]
    assert span["name"] == "decode"
    assert span["owner"] == "clozn_worker"
    assert span["start_ns"] == 0          # the first span always anchors the origin at 0
    assert span["duration_ns"] >= 0


def test_sequential_spans_get_increasing_relative_start_offsets():
    rec = SpanRecorder()
    with rec.span("queue", owner="clozn_gateway"):
        time.sleep(0.001)
    with rec.span("decode", owner="clozn_worker"):
        time.sleep(0.001)
    assert len(rec.spans) == 2
    queue, decode = rec.spans
    assert queue["start_ns"] == 0
    assert decode["start_ns"] >= queue["start_ns"] + queue["duration_ns"]  # decode starts after queue ends


def test_nested_spans_are_both_recorded():
    rec = SpanRecorder()
    with rec.span("model_load", owner="model_load"):
        with rec.span("gpu_init", owner="model_load"):
            pass
    names = [s["name"] for s in rec.spans]
    assert names == ["gpu_init", "model_load"]  # inner span's `with` exits (and is recorded) first


def test_an_exception_inside_the_block_still_gets_recorded_and_propagates():
    rec = SpanRecorder()
    with pytest.raises(RuntimeError, match="boom"):
        with rec.span("decode", owner="clozn_worker"):
            raise RuntimeError("boom")
    assert len(rec.spans) == 1  # the span's own bookkeeping ran in `finally`


def test_a_falsy_name_or_owner_falls_back_to_unknown_rather_than_raising():
    rec = SpanRecorder()
    with rec.span("", owner=""):
        pass
    assert rec.spans[0]["name"] == "unknown"
    assert rec.spans[0]["owner"] == "unknown"


def test_as_phases_returns_a_defensive_copy():
    rec = SpanRecorder()
    with rec.span("decode", owner="clozn_worker"):
        pass
    phases = rec.as_phases()
    phases[0]["name"] = "mutated"
    assert rec.spans[0]["name"] == "decode"  # the recorder's own state is untouched


def test_known_owner_matches_the_schemas_ownership_enum():
    for owner in ("clozn_gateway", "clozn_worker", "model_load", "client", "external_tool", "unknown"):
        assert known_owner(owner)
    assert not known_owner("something_else")


def test_span_overhead_is_small_and_bounded():
    """Documents the actual per-span cost (informational) and guards against a regression that would make
    this helper unsafe to use on the hot request path. The bound is deliberately generous (5 ms/span) to
    stay stable on slow/loaded CI hardware -- typical CPython overhead for this helper is low
    single-digit microseconds; a real regression (e.g. accidental I/O in the recorder) would blow well
    past 5 ms, so this remains a meaningful guard without being a source of CI flakiness."""
    rec = SpanRecorder()
    n = 2000
    for i in range(n):
        with rec.span(f"span_{i}", owner="unknown"):
            pass
    durations = [s["duration_ns"] for s in rec.spans]
    median_ns = statistics.median(durations)
    assert median_ns < 5_000_000, f"median span overhead {median_ns} ns exceeds the 5 ms guard"
