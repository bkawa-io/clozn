"""Milestone F: the token-workbench ACTION logic -- where the expensive work clozn.runs.token_workbench
(Milestone E, read-only) only PREVIEWS is actually requested. clozn/server/routes/token_workbench_
actions.py is the thin HTTP surface over this module.

THE CONTRACT
------------
Every action (force-token / causal-trace / source-measure / mechanistic-diff) resolves to exactly ONE of three
shapes -- never a bare error, never a silent no-op, never a 200 hiding "nothing happened":
  1. `{"outcome": "cached", "artifact": ...}`   -- already computed for this exact identity.
  2. `{"outcome": "job", "job": ...}`           -- long-running work started (poll/cancel by job_id).
  3. `{"outcome": "unavailable", "reason": {"code", "message"}}` -- typed, never a bare false.
The route layer builds this envelope; this module supplies the artifact/job-worker/refusal each action
actually needs, composing existing producers rather than reimplementing them:
  * force-token       -> clozn.recipes.time_travel.run_time_travel, returning a
                         GeneratedObservation-backed result without creating a child Run.
  * causal-trace       -> clozn.analysis.tracer.trace, the SAME computation POST /runs/<id>/causal-trace
                         already runs synchronously (composed read-only; that route is unchanged).
  * source-measure     -> clozn.server.routes.influence_map's own job worker + cache_matches (composed
                         read-only) -- source-measure IS the influence-map machinery, not a second one.
  * mechanistic-diff    -> clozn.analysis.mechanistic_diff.compare(), run through the managed router's
                         sequential model loaders after the same pair-compatibility gate.

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
from a different process -- is a cache hit, never a recompute. force-token has no Workbench child cache;
the ObservationStore is its only reusable evidence authority. source-measure reuses influence-map's own persisted
`run["influence_map"]` + `cache_matches` verbatim. mechanistic-diff's cache includes both run
fingerprints, the capture grid, top-k request, and tensor-retention policy.

EXECUTION AND FAILURE BOUNDARY
------------------------------
The managed-router path now executes a compatible comparison through one model loader at a time. The
pair gate remains pure and authoritative, while the job worker owns exact run evidence, model selection,
progress, persistence, and typed engine failures. A gateway without a managed registry still returns the
legacy `cross_model_execution_not_wired` refusal; it does not fabricate a successful artifact.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from contextlib import contextmanager
from typing import Any, Callable, Mapping

SCHEMA_VERSION = "clozn.token-workbench-action.v1"
CAUSAL_TRACE_METHOD_VERSION = "causal_trace.v1"
MECHANISTIC_DIFF_METHOD_VERSION = "mechanistic_diff.v1"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# ==================================================================================== cache-by-identity
def _run_fingerprint(run: Mapping[str, Any]) -> str:
    """A stable digest over the run's IMMUTABLE content relevant to action caching. Deliberately NOT
    clozn.experiments.execution_facts.parent_execution_fingerprint: this is a cache key over the
    fields that matter to an action's result, not an execution-identity fact, and conflating the two
    would make a cache decision look like a fidelity claim."""
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
    identity = run.get("identity") if isinstance(run.get("identity"), Mapping) else {}
    payload = {
        "final_prompt": run.get("final_prompt"),
        "response": run.get("response"),
        "token_ids": trace.get("token_ids"),
        "model_sha256": identity.get("model_sha256"),
    }
    return _sha(payload)


def _mechanistic_run_fingerprint(run: Mapping[str, Any]) -> str:
    """Include the full recorded runtime identity for mechanistic evidence caching.

    The general workbench fingerprint intentionally stays small for legacy action caches. A
    cross-model capture also depends on template, context, backend, adapter, and engine identity, so
    this action gets a stricter fingerprint without changing cache identities for the other actions.
    """
    identity = run.get("identity") if isinstance(run.get("identity"), Mapping) else {}
    return _sha({
        "run_fingerprint": _run_fingerprint(run),
        "model": run.get("model"),
        "identity": dict(identity),
    })


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


# ========================================================================================= job workers
def force_token_worker(
    run: Mapping[str, Any], sub, index: int, *, token_piece: str | None = None, token_id: int | None = None,
    runtime_identity: Mapping[str, Any] | None, worker_identity: Mapping[str, Any] | None,
    max_new: int = 32,
) -> Callable:
    """Run one Workbench ForceToken arm through the canonical time-travel recipe.

    This worker attaches a TimeTravelResult to the shared job only. It never calls a legacy fork
    executor and never persists a child Run; promotion is an explicit `/time-travel/materialize`
    request handled by the generic materializer.
    """

    def worker(control):
        from clozn.experiments.persistence import ObservationStore
        from clozn.recipes.time_travel import checkpoint_reference_from_pin, run_time_travel
        from clozn.server.influence_jobs import JobCancelled

        control.checkpoint(phase="resolving", completed=0, total=1)
        if control.cancel_requested():
            raise JobCancelled("force-token job cancelled before generation")

        # A durable pin is a read-only exact-state input. Do not capture a fresh checkpoint here:
        # that would be an untracked execution and would recreate the old fork orchestration.
        checkpoint = None
        try:
            from clozn.replay.checkpoint_pin_store import resolve_pin
            checkpoint = checkpoint_reference_from_pin(
                resolve_pin(run.get("id")), run_id=str(run.get("id") or ""),
            )
        except Exception:
            checkpoint = None

        control.checkpoint(phase="generating", completed=0, total=1)
        import clozn.runs.store as runlog
        result = run_time_travel(
            run, position=index, token_id=token_id, token_piece=token_piece, max_new=max_new,
            policy="exact_preferred", checkpoint=checkpoint,
            runtime_identity=runtime_identity, worker_identity=worker_identity,
            substrate=sub, run_loader=runlog.get_run, observation_store=ObservationStore(),
            cancel=control.cancel_requested,
        )
        control.checkpoint(phase="recording", completed=1, total=1)
        control.attach_result(result.to_dict())
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
    the force-token worker's job-level cancellation boundary -- a cancelled job never reports "completed" and
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


def mechanistic_diff_cache_key(
    run: Mapping[str, Any], reference_run: Mapping[str, Any], index: int,
    *, layers: Sequence[int], topk: int, store_tensors: bool,
) -> str:
    """Identity key for a cross-model mechanistic action.

    The anchor run's immutable content is not enough: the selected reference run, capture grid, top-k
    request, and tensor-retention policy all change the evidence.  Keep this key beside the other
    workbench action keys so a repeat request can never reuse a document from a different pair.
    """
    return _sha({
        "run_fingerprint": _mechanistic_run_fingerprint(run),
        "reference_run_fingerprint": _mechanistic_run_fingerprint(reference_run),
        "index": int(index),
        "action": "mechanistic_diff",
        "method_version": MECHANISTIC_DIFF_METHOD_VERSION,
        "layers": sorted({int(layer) for layer in layers}),
        "topk": int(topk),
        "store_tensors": bool(store_tensors),
    })


def mechanistic_diff_worker(
    run: Mapping[str, Any], reference_run: Mapping[str, Any], index: int, *,
    pair_compatibility: Mapping[str, Any], router, layers: Sequence[int], topk: int,
    store_tensors: bool, cache_key: str,
) -> Callable:
    """Build the cancellable job worker for one managed cross-model comparison.

    ``clozn.analysis.mechanistic_diff.compare`` owns the artifact and its validation.  This adapter only
    supplies exact recorded prompt/continuation evidence and sequential managed-router loaders.  The
    loaders enter and leave one model at a time, allowing a resident limit of one to evict the first arm
    before loading the second without ever exposing a private worker port.
    """
    run_id = str(run.get("id") or "")
    prompt = run.get("final_prompt")
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
    continuation_ids = trace.get("token_ids") if isinstance(trace.get("token_ids"), list) else []

    def worker(control):
        from clozn.analysis import mechanistic_diff

        control.checkpoint(phase="preparing", completed=0, total=2)
        if not isinstance(prompt, str) or not prompt:
            return {"state": "failed", "error": {
                "code": "mechanistic_diff_missing_prompt",
                "message": "the anchor run has no exact final_prompt to compare",
            }}
        if not continuation_ids or any(
                isinstance(token_id, bool) or not isinstance(token_id, int)
                for token_id in continuation_ids):
            return {"state": "failed", "error": {
                "code": "mechanistic_diff_missing_token_ids",
                "message": "the anchor run has no exact recorded continuation token IDs",
            }}

        @contextmanager
        def selected_engine(model_id: str):
            selection = router.select_control_model(
                model_id, route="/runs/<id>/tokens/<index>/mechanistic-diff")
            engine = getattr(selection, "engine", None)
            if engine is None:
                raise RuntimeError(f"managed model {model_id!r} has no ready engine")
            yield engine

        try:
            result = mechanistic_diff.compare(
                pair_compat=dict(pair_compatibility),
                reference_loader=lambda: selected_engine(str(run.get("model") or "")),
                candidate_loader=lambda: selected_engine(str(reference_run.get("model") or "")),
                prompt=prompt,
                continuation_ids=list(continuation_ids),
                continuation_indices=[int(index)],
                layers=list(layers),
                topk=int(topk),
                store_tensors=bool(store_tensors),
            )
        except Exception as exc:  # noqa: BLE001 -- job boundary reports typed failure, never raises
            return {"state": "failed", "error": {
                "code": "mechanistic_diff_job_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }}
        control.checkpoint(phase="captured", completed=2, total=2)
        if not isinstance(result, dict):
            return {"state": "failed", "error": {
                "code": "mechanistic_diff_job_failed",
                "message": "mechanistic diff did not produce a result object",
            }}

        if not result.get("ok"):
            error = result.get("error")
            return {"state": "failed", "error": {
                "code": "mechanistic_diff_unavailable",
                "message": str(error or "mechanistic diff was unavailable"),
            }}
        artifact = result["document"]
        entry = build_action_entry(
            action="mechanistic_diff", cache_key=cache_key,
            method_version=MECHANISTIC_DIFF_METHOD_VERSION, run_id=run_id, index=index,
            outcome="ok", result=artifact,
        )

        def persist():
            return store_action_result(run_id, entry)

        control.commit(persist)
        control.attach_result(entry)
        return {"state": "completed"}

    return worker


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

    def enriched_identity(record: Mapping[str, Any]) -> dict:
        identity = dict(record.get("identity") or {})
        # Normal run receipts intentionally keep GGUF header facts optional. When the immutable model
        # path is present, enrich the pure gate from that file's header without hashing the multi-GB
        # payload; a missing/legacy path simply preserves the existing explicit-unknown refusal.
        model_path = identity.get("model_path")
        if isinstance(model_path, str) and model_path:
            try:
                from clozn.artifacts.contracts import gguf_identity
                header = gguf_identity(model_path, include_file_hash=False)
                for field in (
                    "architecture", "layer_count", "hidden_size", "vocab_size", "head_count",
                    "head_count_kv", "tokenizer_sha256", "chat_template_sha256", "filename",
                ):
                    if field not in identity and field in header:
                        identity[field] = header[field]
            except Exception:
                pass
        return identity

    identity_a = enriched_identity(run)
    identity_b = enriched_identity(reference_run)
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
