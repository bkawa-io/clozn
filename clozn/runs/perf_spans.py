"""clozn/runs/perf_spans.py -- a lightweight monotonic span recorder for future phase instrumentation.

Nothing calls this yet. It exists as the scaffolding real instrumentation (queue wait, model load,
prefill -- see the 09-performance-diagnosis plan's Slice 6, deferred pending coordination with feature
01's engine-discovery refactor) will build on, so that when it lands it emits the exact shape
clozn.runs.perf_trace.build_trace() and the clozn.performance-trace.v1 schema already expect
(`{name, owner, start_ns, duration_ns}`) rather than another ad hoc timing dict.

WHY MONOTONIC, AND WHY RELATIVE start_ns
-------------------------------------------
clozn.runs.store's existing `timing.duration_ms` is a delta of two time.time() calls -- wall clock, which
an NTP step or a clock adjustment can move backwards mid-request. The roadmap spec is explicit that a
trace's phase durations must never be able to go negative from clock skew; time.monotonic_ns() cannot
move backwards by definition, which is the whole reason to use it here instead of matching store.py's
existing pattern. `start_ns` is recorded relative to this recorder's FIRST span, not as a raw
monotonic_ns() reading (an arbitrary, process-relative number with no meaning on its own) -- matching the
schema's own documented contract that phase offsets are relative to the trace, not absolute clock values.

A SPAN NEVER SWALLOWS THE CALLER'S EXCEPTION
-----------------------------------------------
`span()` measures whatever runs inside the `with` block and always lets its exception propagate; only the
recorder's OWN bookkeeping (formatting/appending the recorded span) is guarded, mirroring
clozn.runs.identity_ext's "a broken facet must cost its own thing, never the run" discipline applied to
timing instead of identity.
"""
from __future__ import annotations

import time
from contextlib import contextmanager

_VALID_OWNERS = frozenset({
    "clozn_gateway", "clozn_worker", "model_load", "client", "external_tool", "unknown",
})


class SpanRecorder:
    """Records `{name, owner, start_ns, duration_ns}` spans against one shared monotonic origin.

    One recorder is meant to live for the lifetime of a single request/trace -- create a new instance per
    request, never share one across concurrent requests (it is not lock-protected; the request-serializing
    RequestGate this project already uses means today's request path never needs it to be).
    """

    def __init__(self) -> None:
        self._origin_ns: int | None = None
        self.spans: list[dict] = []

    @contextmanager
    def span(self, name: str, owner: str = "unknown"):
        """Time the wrapped block and record it as a phase span. `owner` should be one of
        clozn_gateway/clozn_worker/model_load/client/external_tool/unknown (the schema's ownership
        boundary enum) -- an unrecognized value is still recorded, never rejected, since a caller mid-typo
        should not lose its request over a timing helper's opinion."""
        start = time.monotonic_ns()
        if self._origin_ns is None:
            self._origin_ns = start
        try:
            yield
        finally:
            end = time.monotonic_ns()
            try:
                self.spans.append({
                    "name": str(name) if name else "unknown",
                    "owner": str(owner) if owner else "unknown",
                    "start_ns": max(0, start - self._origin_ns),
                    "duration_ns": max(0, end - start),
                })
            except Exception:
                pass

    def as_phases(self) -> list[dict]:
        """A defensive copy of the recorded spans, in `clozn.performance-trace.v1`'s `phases[]` shape."""
        return [dict(item) for item in self.spans]


def known_owner(owner: str) -> bool:
    """Whether `owner` is one of the schema's closed ownership-boundary values -- for callers that want to
    validate before recording rather than after; span() itself never rejects an unknown owner (see its
    docstring)."""
    return owner in _VALID_OWNERS
