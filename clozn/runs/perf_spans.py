"""Monotonic, clock-domain-aware performance spans.

The gateway and worker have unrelated monotonic clocks.  A timing document therefore names the process
that owns its clock and phases only carry offsets inside that document.  Consumers may sum measured,
exclusive durations, but must never subtract offsets from different documents.

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
TIMING_SCHEMA = "clozn.gateway-timing.v1"


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
    def span(self, name: str, owner: str = "unknown", *, aggregation: str = "exclusive"):
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
                    "measurement": "measured",
                    "aggregation": aggregation if aggregation in {
                        "exclusive", "overlapping", "context_only"
                    } else "exclusive",
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


def timing_document(phases, *, owner: str = "clozn_gateway", clock: str = "monotonic") -> dict:
    """Return the versioned wrapper persisted in run metadata.

    Empty documents are valid internally but callers normally omit them from the run.  The defensive
    copies prevent a later request from mutating timing evidence already attached to a run.
    """
    return {
        "schema_version": TIMING_SCHEMA if owner == "clozn_gateway" else "clozn.worker-timing.v1",
        "unit": "nanoseconds",
        "clock": str(clock),
        "clock_owner": str(owner),
        "phases": [dict(phase) for phase in phases if isinstance(phase, dict)],
    }


def merge_timing_documents(*documents, owner: str = "clozn_gateway") -> dict:
    """Merge documents from the *same* clock owner without aligning unrelated offsets.

    Offsets are dropped when more than one independently-originated document contributes phases. Their
    durations remain directly usable; pretending their local origins were shared would not be.
    """
    docs = [doc for doc in documents if isinstance(doc, dict)]
    phase_groups = [
        [dict(phase) for phase in doc.get("phases", []) if isinstance(phase, dict)]
        for doc in docs
    ]
    nonempty = [group for group in phase_groups if group]
    phases = []
    for group in nonempty:
        for phase in group:
            if len(nonempty) > 1:
                phase.pop("start_ns", None)
            phases.append(phase)
    return timing_document(phases, owner=owner)
