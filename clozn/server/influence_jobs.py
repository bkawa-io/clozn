"""Bounded local lifecycle for background jobs -- originally influence-map only, now the ONE shared job
registry every bounded async action in the gateway uses (Milestone F: the token-workbench action
endpoints reuse this verbatim rather than forking a parallel registry/cancellation story -- see
clozn/runs/token_workbench_actions.py and clozn/server/routes/token_workbench_actions.py).

Jobs are process-local convenience state, not durable evidence.  A computed result is attached to its
run (or wherever its own producer persists it) only at that artifact's OWN existing persistence
boundary -- this registry never decides how/where a result is stored, only how its computation's
lifecycle (queued/running/persisting/completed/failed/cancelled) is tracked and cancelled.
Cancellation and persistence share one per-job lock: once a cancel response acknowledges cancellation,
that job cannot subsequently enter its commit.

`kind` (added for Milestone F, additive/backward-compatible: defaults to "influence_map" so every
existing caller/snapshot shape is unchanged) distinguishes which action a job belongs to
("influence_map" | "causal_trace" | "fork" | ...) purely for observability -- it changes no lifecycle
behavior. One registry, one `JOBS` singleton, one capacity budget shared across every kind: a causal-
trace job and an influence-map job compete for the SAME bounded thread pool and job-table slots, which
is intentional (a single, honest capacity story rather than per-kind budgets that could be gamed by
spreading load across kinds).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
import time
import uuid

JOB_SCHEMA = "clozn.influence-map-job.v1"
TERMINAL_STATES = frozenset({"completed", "failed", "cancelled"})


class JobCapacityError(RuntimeError):
    pass


class JobCancelled(RuntimeError):
    pass


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_error(exc: Exception, *, kind: str = "influence_map") -> dict:
    return {
        "code": f"{kind}_job_failed",
        "message": f"{type(exc).__name__}: {exc}",
    }


def _progress(phase: str, completed: int, total: int) -> dict:
    completed = max(0, int(completed))
    total = max(0, int(total))
    if total:
        completed = min(completed, total)
        percent = round((completed / total) * 100.0, 1)
    else:
        percent = 100.0 if phase == "done" else 0.0
    return {
        "phase": phase,
        "completed_units": completed,
        "total_units": total,
        "percent": percent,
    }


@dataclass
class _Job:
    job_id: str
    run_id: str
    kind: str = "influence_map"
    state: str = "queued"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    progress: dict = field(default_factory=lambda: _progress("queued", 0, 0))
    cancel_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    cancel_requested: bool = False
    persisted: bool = False
    cached: bool = False
    error: dict | None = None
    result: dict | None = None


class JobControl:
    """The only worker-facing mutation seam."""

    def __init__(self, job: _Job):
        self._job = job

    def cancel_requested(self) -> bool:
        return self._job.cancel_event.is_set()

    def attach_result(self, payload: dict) -> None:
        """Stash the eventual artifact ON the job snapshot itself (Milestone F addition, additive/
        opt-in: a worker that never calls this -- every influence-map worker today -- leaves
        `result` out of its snapshot exactly as before; `omit, never null-pad`). Lets a caller poll
        ONE job resource to both watch progress and read the terminal artifact, instead of a second
        bespoke fetch endpoint per action. Only takes effect while the job is not yet cancelled, so a
        cancelled job never surfaces a result it raced to attach."""
        with self._job.lock:
            if self._job.cancel_event.is_set() and not self._job.persisted:
                raise JobCancelled("job cancelled before its result was attached")
            self._job.result = dict(payload) if isinstance(payload, dict) else payload

    def checkpoint(self, *, phase: str, completed: int, total: int,
                   best_retained_source_count: int | None = None,
                   certificate_candidate_kind: str | None = None) -> None:
        with self._job.lock:
            if self._job.cancel_event.is_set():
                raise JobCancelled("influence-map job cancelled")
            if self._job.state not in TERMINAL_STATES:
                self._job.state = "running"
                progress = _progress(phase, completed, total)
                if isinstance(best_retained_source_count, int) and best_retained_source_count >= 0:
                    progress["best_retained_source_count"] = best_retained_source_count
                if isinstance(certificate_candidate_kind, str) and certificate_candidate_kind:
                    progress["certificate_candidate_kind"] = certificate_candidate_kind
                self._job.progress = progress
                self._job.updated_at = _now_iso()

    def commit(self, persist) -> None:
        """Run the persistence callback only while cancellation is excluded."""
        with self._job.lock:
            if self._job.cancel_event.is_set():
                raise JobCancelled("influence-map job cancelled before persistence")
            self._job.state = "persisting"
            self._job.progress = _progress("persisting", 0, 1)
            self._job.updated_at = _now_iso()
            if persist() is not True:
                raise RuntimeError("influence-map could not be attached to the run")
            # cancel() treats persisted jobs as no longer cancellable while this
            # worker performs its final in-memory state transition.
            self._job.persisted = True
            self._job.progress = _progress("persisting", 1, 1)


class InfluenceJobRegistry:
    def __init__(
        self,
        *,
        max_jobs: int = 64,
        max_workers: int = 2,
        terminal_ttl_seconds: float = 3600.0,
        clock=time.monotonic,
    ):
        self.max_jobs = max(1, int(max_jobs))
        self.terminal_ttl_seconds = max(1.0, float(terminal_ttl_seconds))
        self._clock = clock
        self._jobs: dict[str, _Job] = {}
        self._terminal_at: dict[str, float] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="clozn-influence",
        )

    def _prune_locked(self) -> None:
        cutoff = self._clock() - self.terminal_ttl_seconds
        for job_id, terminal_at in list(self._terminal_at.items()):
            if terminal_at <= cutoff:
                self._jobs.pop(job_id, None)
                self._terminal_at.pop(job_id, None)
        if len(self._jobs) < self.max_jobs:
            return
        for job_id, _ in sorted(self._terminal_at.items(), key=lambda item: item[1]):
            self._jobs.pop(job_id, None)
            self._terminal_at.pop(job_id, None)
            if len(self._jobs) < self.max_jobs:
                return

    def start(self, run_id: str, worker=None, *, cached: bool = False, kind: str = "influence_map",
              cached_result: dict | None = None) -> dict:
        with self._lock:
            self._prune_locked()
            if len(self._jobs) >= self.max_jobs:
                raise JobCapacityError("job capacity is full")
            job = _Job(
                job_id=f"infjob_{uuid.uuid4().hex}", run_id=run_id, kind=str(kind or "influence_map"),
                cached=bool(cached))
            if isinstance(cached_result, dict):
                job.result = dict(cached_result)
            self._jobs[job.job_id] = job
            if cached:
                with job.lock:
                    job.state = "completed"
                    job.progress = _progress("done", 1, 1)
                    job.updated_at = _now_iso()
                self._terminal_at[job.job_id] = self._clock()
            else:
                self._executor.submit(self._execute, job, worker)
            return self._snapshot(job)

    def _execute(self, job: _Job, worker) -> None:
        cancelled_before_start = False
        with job.lock:
            if job.cancel_event.is_set() or job.state == "cancelled":
                job.state = "cancelled"
                job.updated_at = _now_iso()
                cancelled_before_start = True
            else:
                job.state = "running"
                job.progress = _progress("starting", 0, 1)
                job.updated_at = _now_iso()
        if cancelled_before_start:
            self._mark_terminal(job)
            return
        control = JobControl(job)
        try:
            outcome = worker(control)
            if isinstance(outcome, dict) and outcome.get("state") == "failed":
                with job.lock:
                    if job.cancel_event.is_set() and not job.persisted:
                        raise JobCancelled("influence-map job cancelled")
                    job.state = "failed"
                    job.error = dict(outcome.get("error") or {
                        "code": "influence_map_job_failed",
                        "message": "influence-map measurement did not produce available evidence",
                    })
                    job.updated_at = _now_iso()
            else:
                with job.lock:
                    if job.cancel_event.is_set() and not job.persisted:
                        raise JobCancelled("influence-map job cancelled")
                    job.state = "completed"
                    job.progress = _progress("done", 1, 1)
                    job.updated_at = _now_iso()
        except JobCancelled:
            with job.lock:
                job.state = "cancelled"
                job.cancel_requested = True
                job.progress = {
                    **job.progress,
                    "phase": "cancelled",
                }
                job.updated_at = _now_iso()
        except Exception as exc:
            with job.lock:
                job.state = "failed"
                job.error = _safe_error(exc, kind=job.kind)
                job.updated_at = _now_iso()
        self._mark_terminal(job)

    def _mark_terminal(self, job: _Job) -> None:
        with self._lock:
            self._terminal_at[job.job_id] = self._clock()

    def _find(self, run_id: str, job_id: str) -> _Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job if job is not None and job.run_id == run_id else None

    def get(self, run_id: str, job_id: str) -> dict | None:
        job = self._find(run_id, job_id)
        return self._snapshot(job) if job is not None else None

    def cancel(self, run_id: str, job_id: str) -> dict | None:
        job = self._find(run_id, job_id)
        if job is None:
            return None
        terminal_now = False
        with job.lock:
            accepted = False
            if job.state not in TERMINAL_STATES and not job.persisted:
                if not job.cancel_requested:
                    accepted = True
                job.cancel_requested = True
                job.cancel_event.set()
                if job.state == "queued":
                    job.state = "cancelled"
                    job.progress = {**job.progress, "phase": "cancelled"}
                    terminal_now = True
                else:
                    job.state = "cancelling"
                job.updated_at = _now_iso()
            snapshot = self._snapshot_locked(job)
            snapshot["cancel_accepted"] = accepted
        if terminal_now:
            self._mark_terminal(job)
        return snapshot

    def _snapshot(self, job: _Job) -> dict:
        with job.lock:
            return self._snapshot_locked(job)

    @staticmethod
    def _snapshot_locked(job: _Job) -> dict:
        out = {
            "schema_version": JOB_SCHEMA,
            "job_id": job.job_id,
            "run_id": job.run_id,
            "kind": job.kind,
            "state": job.state,
            "progress": dict(job.progress),
            "cancel_requested": job.cancel_requested,
            "cancellable": job.state not in TERMINAL_STATES and not job.persisted,
            "cached": job.cached,
            "created_at": job.created_at,
            "updated_at": job.updated_at,
        }
        if job.error:
            out["error"] = dict(job.error)
        if job.result is not None:
            out["result"] = dict(job.result) if isinstance(job.result, dict) else job.result
        return out

    def clear_for_tests(self) -> None:
        """Cancel/drop local records. Only tests should call this."""
        with self._lock:
            jobs = list(self._jobs.values())
            for job in jobs:
                job.cancel_event.set()
            self._jobs.clear()
            self._terminal_at.clear()


JOBS = InfluenceJobRegistry()


__all__ = [
    "JOBS",
    "JOB_SCHEMA",
    "JobCancelled",
    "JobCapacityError",
    "JobControl",
    "InfluenceJobRegistry",
    "TERMINAL_STATES",
]
