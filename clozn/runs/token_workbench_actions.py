"""Milestone F: the token-workbench ACTION logic -- where the expensive work clozn.runs.token_workbench
(Milestone E, read-only) only PREVIEWS is actually requested. clozn/server/routes/token_workbench_
actions.py is the thin HTTP surface over this module.

THE CONTRACT
------------
Every action (fork / causal-trace / source-measure / mechanistic-diff) resolves to exactly ONE of three
shapes -- never a bare error, never a silent no-op, never a 200 hiding "nothing happened":
  1. `{"outcome": "cached", "artifact": ...}`   -- already computed for this exact identity.
  2. `{"outcome": "job", "job": ...}`           -- long-running work started (poll/cancel by job_id).
  3. `{"outcome": "unavailable", "reason": {"code", "message"}}` -- typed, never a bare false.
The route layer builds this envelope; this module supplies the artifact/job-worker/refusal each action
actually needs, composing existing producers rather than reimplementing them:
  * fork              -> clozn.replay.fork.compat_fork (already returns exact_execution_fork /
                         reconstructed_replay / unavailable -- that vocabulary rides through untouched
                         as a job's `result`, never renamed).
  * causal-trace       -> clozn.analysis.tracer.trace, the SAME computation POST /runs/<id>/causal-trace
                         already runs synchronously (composed read-only; that route is unchanged).
  * source-measure     -> clozn.server.routes.influence_map's own job worker + cache_matches (composed
                         read-only) -- source-measure IS the influence-map machinery, not a second one.
  * mechanistic-diff    -> clozn.analysis.pair_compatibility.assess(), composed read-only and pure
                         (no engine, GGUF-header-shaped identity dicts only).

ONE JOB SYSTEM
--------------
Every long-running action goes through clozn.server.influence_jobs.JOBS -- the SAME registry
influence-map jobs already use, additively generalized (a `kind` field, an optional `result` payload
on the snapshot) rather than forked into a second lifecycle/cancellation story. See that module's own
docstring for what changed and why it stays backward compatible.

CACHING
--------
"cache by run identity + token index + method version + artifact identity": `action_cache_key()`
digests the run's own immutable content (never its mutable/derived fields), the token index, the
method's own version tag (bump it if that method's default behavior changes materially), the action's
request parameters, and the CURRENT worker's runtime identity (a causal trace against a different
engine build is not the same evidence). Cached causal-trace results are persisted on the run itself
(`run["token_workbench_actions"]`, schema `clozn.token-workbench-action.v1`) so a repeat request -- even
from a different process -- is a cache hit, never a recompute. fork's cache is its own already-durable
child runs (an existing child with the same parent + position + forced piece IS the cached artifact;
no separate cache entry duplicates it). source-measure reuses influence-map's own persisted
`run["influence_map"]` + `cache_matches` verbatim. mechanistic-diff's gate is pure and cheap (GGUF-header
digests, no engine) -- there is nothing expensive to cache until cross-model execution is wired (see
that function's own docstring for why this milestone stops at the typed-refusal gate).

WHY THIS DOES NOT WIRE A SUCCESSFUL mechanistic-diff EXECUTION
-------------------------------------------------------------------
Actually running a cross-model diff (clozn.analysis.mechanistic_diff.compare()) needs two GGUFs loaded
SEQUENTIALLY -- the worker_registry/worker_handle/model_routing cold-load-and-evict machinery another
agent owns and is actively changing this same wave. Wiring a job against infrastructure that is moving
underneath this task would risk a job that silently cannot actually run, which is worse than an honest
typed `unavailable`. mechanistic_diff_gate() below still does the REAL, useful, pure work (the pair-
compatibility refusal), so a genuinely incompatible pair is refused with pair_compatibility's own typed
reason -- verbatim, never re-derived -- exactly as asked; a genuinely compatible pair gets an honest
"not yet wired" reason instead of a job that would need infrastructure this module does not touch.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from typing import Any, Callable, Mapping, Sequence

SCHEMA_VERSION = "clozn.token-workbench-action.v1"
CAUSAL_TRACE_METHOD_VERSION = "causal_trace.v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ==================================================================================== cache-by-identity
def _run_fingerprint(run: Mapping[str, Any]) -> str:
    """A stable digest over the run's IMMUTABLE content relevant to action caching. Deliberately NOT
    clozn.replay.execution_fork's own parent_execution_fingerprint -- that module is under concurrent
    development this wave (see this file's module docstring); a small, independent, equally honest
    proxy that needs no import from it."""
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
    identity = run.get("identity") if isinstance(run.get("identity"), Mapping) else {}
    payload = {
        "final_prompt": run.get("final_prompt"),
        "response": run.get("response"),
        "token_ids": trace.get("token_ids"),
        "model_sha256": identity.get("model_sha256"),
    }
    return _sha(payload)


def action_cache_key(
    run: Mapping[str, Any], index: int, action: str, method_version: str, params: Mapping[str, Any],
    *, runtime_identity: Mapping[str, Any] | None = None,
) -> str:
    """sha256 over (run identity, token index, method version, request params, current artifact/worker
    identity) -- a repeat request with all five unchanged is the SAME evidence, never recomputed."""
    payload = {
        "run_fingerprint": _run_fingerprint(run),
        "index": int(index),
        "action": str(action),
        "method_version": str(method_version),
        "params": dict(params),
        "runtime_identity": dict(runtime_identity) if isinstance(runtime_identity, Mapping) else None,
    }
    return _sha(payload)


def build_action_entry(
    *, action: str, cache_key: str, method_version: str, run_id: str, index: int, outcome: str,
    result: Mapping[str, Any] | None = None,
) -> dict:
    """One `clozn.token-workbench-action.v1` cache entry. `result` is the underlying producer's OWN
    artifact, embedded verbatim -- never reshaped into a new vocabulary."""
    entry: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "action": str(action),
        "cache_key": str(cache_key),
        "method_version": str(method_version),
        "run_id": str(run_id),
        "index": int(index),
        "computed_at": _now_iso(),
        "outcome": str(outcome),
    }
    if result is not None:
        entry["result"] = dict(result)
    return entry


def find_cached_action(run: Mapping[str, Any], cache_key: str) -> dict | None:
    for entry in run.get("token_workbench_actions") or []:
        if isinstance(entry, Mapping) and entry.get("cache_key") == cache_key:
            return deepcopy(dict(entry))
    return None


def store_action_result(run_id: str, entry: Mapping[str, Any]) -> bool:
    """Persist `entry` onto the run's `token_workbench_actions` list, replacing any prior entry with
    the SAME cache_key (a repeat compute under an identical key supersedes, never duplicates)."""
    import clozn.runs.store as runlog

    latest = runlog.get_run(run_id)
    if latest is None:
        return False
    updated = dict(latest)
    existing = [
        item for item in (updated.get("token_workbench_actions") or [])
        if not (isinstance(item, Mapping) and item.get("cache_key") == entry.get("cache_key"))
    ]
    updated["token_workbench_actions"] = existing + [dict(entry)]
    return runlog.replace_run(updated)


# =================================================================================== fork's own cache
def _forced_piece_from_child(child: Mapping[str, Any]) -> tuple[Any, str] | None:
    """(position, forced piece text) a child run's OWN changes_applied already recorded, reading
    EITHER the legacy splice shape (`changes.fork`) or the exact-execution-fork shape
    (`changes.execution_fork`) -- fork's two regimes record the forced piece under different keys, but
    both keep it, so this is not a guess."""
    changes = child.get("changes_applied")
    if not isinstance(changes, Mapping):
        return None
    legacy = changes.get("fork")
    if isinstance(legacy, Mapping) and "position" in legacy:
        return legacy.get("position"), str(legacy.get("token", ""))
    exact = changes.get("execution_fork")
    if isinstance(exact, Mapping):
        intervention = exact.get("intervention")
        piece = intervention.get("token_piece") if isinstance(intervention, Mapping) else None
        return exact.get("position"), str(piece or "")
    return None


def find_cached_fork_child(
    related_runs: Sequence[Mapping[str, Any]], parent_run_id: str, position: int, forced_piece: str,
) -> dict | None:
    """An existing child run already produced by forcing the SAME piece at the SAME position on this
    parent. Fork's own cache IS its already-durable children -- runlog.record makes every fork
    immutable and persistent already, so no separate cache entry would do anything but duplicate it."""
    for candidate in related_runs:
        if not isinstance(candidate, Mapping) or candidate.get("parent_run_id") != parent_run_id:
            continue
        if _forced_piece_from_child(candidate) == (position, forced_piece):
            return deepcopy(dict(candidate))
    return None


# ========================================================================================= job workers
def fork_worker(
    run: Mapping[str, Any], sub, index: int, *, token=None, token_id=None,
    runtime_identity: Mapping[str, Any] | None, worker_identity: Mapping[str, Any] | None,
) -> Callable:
    """The JobControl-shaped `worker(control)` callable for a fork job: composes
    clozn.replay.fork.compat_fork() read-only. Whatever outcome compat_fork reaches
    (exact_execution_fork / reconstructed_replay / unavailable) becomes this job's `result` VERBATIM --
    this function never re-labels it. compat_fork has no cooperative cancellation hook (each of its
    worker round trips is one blocking call), so a cancel request here is honored at the STATE level:
    clozn.server.influence_jobs's own generic post-call check still reports "cancelled" rather than
    "completed" and skips persistence, but the call already in flight keeps running to completion in
    its own thread -- the same caveat causal_trace_worker documents below."""

    def worker(control):
        import clozn.replay.fork as fork_mod

        control.checkpoint(phase="forking", completed=0, total=1)
        try:
            child = fork_mod.compat_fork(
                run, sub, index, token=token, token_id=token_id,
                runtime_identity=runtime_identity, worker_identity=worker_identity)
        except ValueError as exc:
            # The route validates position/token BEFORE ever starting a job; a ValueError reaching
            # here is a genuine internal inconsistency, not a client input error -- an honest job
            # failure, never silently swallowed.
            return {"state": "failed", "error": {
                "code": "fork_job_invalid_request", "message": str(exc)}}
        control.checkpoint(phase="validating", completed=1, total=1)
        if child is None:
            return {"state": "failed", "error": {
                "code": "fork_job_failed",
                "message": (
                    "fork failed (generation error, or the run's prompt could not be reconstructed)"),
            }}
        control.attach_result(child)
        return {"state": "completed"}

    return worker


def causal_trace_worker(
    run: Mapping[str, Any], index: int, *, seed: int, screen_mode: str, contrast,
    engine_url: str | None, cache_key: str,
) -> Callable:
    """The JobControl-shaped `worker(control)` callable for a causal-trace job: composes
    clozn.analysis.tracer.trace() read-only -- the SAME computation POST /runs/<id>/causal-trace already
    runs synchronously (that route is unchanged; this is an independent second caller of the same
    function). tracer.trace() accepts no cancel_requested/progress callback, so it cannot be
    cooperatively interrupted mid-trace; cancellation is honored at the job-STATE level only (see
    fork_worker's docstring for the identical caveat) -- a cancelled job never reports "completed" and
    never persists its result, even though the trace itself keeps running to completion in the
    background."""
    run_id = str(run.get("id") or "")
    prompt = run.get("final_prompt")
    response = run.get("response")

    def worker(control):
        from clozn.analysis import tracer

        control.checkpoint(phase="tracing", completed=0, total=1)
        kwargs: dict[str, Any] = {"seed": seed, "screen_mode": screen_mode, "contrast": contrast}
        if engine_url:
            kwargs["engine_url"] = engine_url
        try:
            result = tracer.trace(prompt, response, index, **kwargs)
        except Exception as exc:
            return {"state": "failed", "error": {
                "code": "causal_trace_job_failed", "message": f"{type(exc).__name__}: {exc}"}}
        control.checkpoint(phase="validating", completed=1, total=1)
        if not isinstance(result, dict):
            return {"state": "failed", "error": {
                "code": "causal_trace_job_failed",
                "message": "causal trace did not produce a result object"}}
        # tracer.trace()'s OWN convention (causal_trace.py's route docstring): {"ok": False, "blocked":
        # ...} is a COMPLETED analysis that honestly could not proceed, not a failed request -- so this
        # job always reaches "completed" once trace() returns any dict, mirroring that route exactly.
        entry = build_action_entry(
            action="causal_trace", cache_key=cache_key, method_version=CAUSAL_TRACE_METHOD_VERSION,
            run_id=run_id, index=index, outcome="ok" if result.get("ok") else "blocked", result=result)

        def persist():
            return store_action_result(run_id, entry)

        control.commit(persist)
        control.attach_result(entry)
        return {"state": "completed"}

    return worker


def source_measure_job_worker(run: Mapping[str, Any], sub, max_spans: int) -> Callable:
    """Thin pass-through to clozn.server.routes.influence_map's OWN job worker (composed read-only):
    source-measure IS the section-influence/influence-map machinery, not a second implementation of
    it. The returned worker persists to `run["influence_map"]` exactly as the existing
    POST /runs/<id>/influence-map/jobs route already does; a repeat request hits that SAME cache via
    clozn.receipts.context_answer_influence.cache_matches, checked by the route before this is ever
    called."""
    from clozn.server.routes.influence_map import _job_worker

    return _job_worker(run, sub, max_spans)


# ============================================================================== mechanistic-diff's gate
def mechanistic_diff_gate(run: Mapping[str, Any], reference_run: Mapping[str, Any]) -> dict:
    """Typed pair-compatibility gate for a cross-model mechanistic diff, composing
    clozn.analysis.pair_compatibility.assess() read-only and PURE (GGUF-header-shaped identity dicts,
    no engine, no GPU). Returns {"permitted": bool, "report": <clozn.pair-compatibility.v1, verbatim>,
    "reason": <pair_compatibility's OWN operation reason text, never re-derived>}.

    mechanistic_diff.py's own preflight (composed read-only there, mirrored here) requires per-token
    comparison (tokenizer-exact) AND a matching hidden_size, so both gates must permit. Most recorded
    runs today carry only the lightweight reproduction-identity block (model_sha256/template_fingerprint/
    engine_build), not the full GGUF-header identity (architecture/hidden_size/tokenizer_sha256/...) --
    assess() does not treat that as an error, it reports each missing dimension "unknown" (roadmap rule
    2: omit, never null-pad) and correctly REFUSES both operations with an honest reason, which is
    exactly the typed-unavailable outcome this gate is supposed to produce for that common case."""
    from clozn.analysis import pair_compatibility

    identity_a = dict(run.get("identity") or {})
    identity_b = dict(reference_run.get("identity") or {})
    report = pair_compatibility.assess(
        identity_a, identity_b,
        label_a=str(run.get("model") or "") or None,
        label_b=str(reference_run.get("model") or "") or None,
    )
    per_token = pair_compatibility.may_per_token_compare(report)
    transplant = pair_compatibility.may_residual_transplant(report)
    operations = (report.get("verdict") or {}).get("operations") or {}
    reasons = [
        operations.get("per_token_comparison", {}).get("reason"),
        operations.get("residual_transplant", {}).get("reason"),
    ]
    reason = " ".join(text for text in reasons if isinstance(text, str) and text)
    return {
        "permitted": bool(per_token and transplant),
        "report": report,
        "reason": reason or "pair compatibility could not be assessed",
    }
