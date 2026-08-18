"""Read-only synthesis for one token of one recorded run: ``clozn.token-workbench.v1``.

Milestone E's backend. Companion to :mod:`clozn.runs.investigation` (composed here, not reimplemented)
at TOKEN granularity: given a run and a response-token index, project what already exists about that
token -- its identity, its recorded per-token readouts, the run's received context, an optional
structural comparison against a reference run -- plus a ``capabilities`` block describing, for each
related expensive action, whether it is honestly available right now and why/why not.

THE CENTRAL RULE
-----------------
Building this document must never execute a model measurement, call a live worker, or mutate a run.
Every section is a journal read or a deterministic projection over already-recorded evidence.
``capabilities`` entries describe whether a SEPARATE POST action (fork execution, an influence job, a
causal trace, a cross-model diff) is available -- they never start one.  Even the "is a live engine
reachable" checks below are local attribute lookups (mirroring investigation.py's own
``scoring_available`` convention), never a network round trip to the worker.

WHY THIS DOES NOT INVENT ONE UNIVERSAL STATUS ENUM
-----------------------------------------------------
clozn deliberately keeps separate closed evidence vocabularies per artifact type (docs/SEAMS.md rule 4
and clozn.runs.investigation's own docstring). ``capabilities`` therefore does NOT flatten
exact_fork/source_measurement/causal_trace/mechanistic_diff into one shared shape:
  * ``exact_fork``          -- {available, snapshot_state, reason?, action?}. ``snapshot_state`` is a
    *preview* label (workbench-local, not the checkpoint-reference schema's own {available, missing,
    expired} vocabulary, since this route never captures a checkpoint to observe that state) built by
    replaying, read-only, the SAME pure structural preconditions
    ``clozn.replay.checkpoint_capture.capture_parent_checkpoint`` checks before it ever touches a
    worker (composed via that module's private ``_trace_history``/``_sampler``/``_steering`` helpers,
    so this preview cannot drift from the canonical force-token action's exact preconditions).
  * ``source_measurement``  -- {available, status, reason?, action?}. ``status`` is
    clozn.run-investigation.v1's OWN ``prompt_source_influence`` section state, carried verbatim.
  * ``causal_trace``        -- {available, status, reason?, action?}. There is no persisted native
    status for this one (a causal trace is always computed fresh); ``status`` is a plain readiness word.
  * ``mechanistic_diff``    -- {available, reason, action?}. A binary structural check (does a
    DIFFERENT-model reference run exist to diff against) with no native status vocabulary to carry, so
    ``reason`` is always present and explains the determination either way.
A capability that is unavailable always carries a `reason` -- never a bare `false`.

WHY clozn/runs/, NOT clozn/analysis/
--------------------------------------
This module runs no analysis of its own -- no forward pass, no comparison math beyond what
clozn.analysis.run_diff.compare_runs (composed read-only, exactly as investigation.py already does)
already provides. It is a run projection, the same charter as clozn.runs.investigation and
clozn.runs.text_span_addresses, both of which already live here.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "clozn.token-workbench.v1"


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def _workbench_action(method: str, href: str, *, request_body: dict | None = None) -> dict:
    """This module's own small `{method, href, request_body?}` action shape (clozn.token-workbench.v1's
    `Action` $def) -- deliberately narrower than investigation's own action vocabulary. A capability
    entry's `available`/`status`/`reason` already carries the readiness facts; `action` is only ever a
    navigation pointer, never a second copy of investigation's id/label/kind/availability/reason.

    Every capability's `action` points at a REAL, live Token Workbench endpoint
    (clozn/server/routes/token_workbench_actions.py); this projection never advertises a retired
    compatibility route."""
    out: dict[str, Any] = {"method": method, "href": href}
    if request_body is not None:
        out["request_body"] = dict(request_body)
    return out


# ------------------------------------------------------------------------------------------- sections
def _run_section(run: Mapping[str, Any]) -> dict:
    identity = run.get("identity") if isinstance(run.get("identity"), Mapping) else {}
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
    pieces = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
    run_id = str(run.get("id") or "")
    return _drop_none({
        "id": run_id,
        "model": str(run.get("model") or "") or None,
        "substrate": str(run.get("substrate") or "") or None,
        "source": str(run.get("source") or "") or None,
        "parent_run_id": run.get("parent_run_id"),
        "token_count": len(pieces),
        "model_sha256": identity.get("model_sha256"),
        "href": f"/runs/{run_id}",
    })


def _token_section(run: Mapping[str, Any], index: int) -> dict:
    """The token's identity at `index`. Raises ValueError (the route's 400) when the run has no trace
    or `index` is out of range -- mirrors the canonical recorded-token validation contract."""
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
    pieces = trace.get("tokens")
    if not isinstance(pieces, list) or not pieces:
        raise ValueError("run has no trace to inspect")
    if index < 0 or index >= len(pieces):
        raise ValueError(f"token index {index} out of range (the reply has {len(pieces)} trace tokens)")
    token_ids = trace.get("token_ids") if isinstance(trace.get("token_ids"), list) else []
    alt_lists = trace.get("alternatives") if isinstance(trace.get("alternatives"), list) else []
    alts = alt_lists[index] if index < len(alt_lists) and isinstance(alt_lists[index], list) else []
    alternatives = []
    for a in alts:
        if not isinstance(a, Mapping):
            continue
        piece = str(a.get("piece", a.get("text", "")))
        alternatives.append(_drop_none({
            "piece": piece,
            "token_id": a.get("token_id", a.get("id")),
            "prob": a.get("prob"),
        }))
    out = _drop_none({
        "index": index,
        "piece": str(pieces[index]),
        "prefix_kept": "".join(str(p) for p in pieces[:index]),
        "token_id": token_ids[index] if index < len(token_ids) else None,
    })
    out["alternatives"] = alternatives
    return out


def _readouts_section(run: Mapping[str, Any], index: int) -> dict:
    """Already-recorded per-token measurements at `index` -- never a fresh measurement."""
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
    confidence = trace.get("confidence") if isinstance(trace.get("confidence"), list) else []
    logprobs = trace.get("logprobs") if isinstance(trace.get("logprobs"), list) else []
    topk_entropy = trace.get("topk_entropy") if isinstance(trace.get("topk_entropy"), list) else []
    steps = trace.get("steps") if isinstance(trace.get("steps"), list) else []
    step = steps[index] if index < len(steps) and isinstance(steps[index], Mapping) else {}
    measurements = _drop_none({
        "confidence": confidence[index] if index < len(confidence) else step.get("confidence", step.get("prob")),
        "logprob": logprobs[index] if index < len(logprobs) else step.get("logprob"),
        "topk_entropy": topk_entropy[index] if index < len(topk_entropy) else step.get("topk_entropy"),
    })
    workspace = [
        deepcopy(dict(item)) for item in (trace.get("workspace_readouts") or [])
        if isinstance(item, Mapping) and item.get("position") == index
    ]
    state = "supported" if (measurements or workspace) else "unavailable"
    out = {"state": state, "measurements": measurements, "workspace_readouts": workspace}
    if state == "unavailable":
        out["reason"] = "no per-token measurements were recorded at this position"
    return out


def _context_section(investigation_doc: Mapping[str, Any]) -> dict:
    section = (investigation_doc.get("sections") or {}).get("received_context")
    if isinstance(section, Mapping):
        return deepcopy(dict(section))
    return {"state": "unavailable", "reason": "no context receipt was recorded for this run"}


def _comparison_section(
    run: Mapping[str, Any],
    related_runs: Sequence[Mapping[str, Any]],
    *,
    reference_run_id: str | None,
    reference_run: Mapping[str, Any] | None,
) -> dict:
    run_id = str(run.get("id") or "")
    if reference_run_id and reference_run is None:
        return {
            "state": "unavailable",
            "reason": f"reference run {reference_run_id!r} was not found",
        }
    if reference_run is not None:
        reference = dict(reference_run)
        selection: dict[str, Any] = {
            "mode": "explicit", "reference_run_id": str(reference.get("id") or reference_run_id),
        }
    else:
        from clozn.analysis import run_diff

        records = [dict(item) for item in related_runs if isinstance(item, Mapping)]
        by_id = {str(item.get("id")): item for item in records if item.get("id")}
        parent_id = run.get("parent_run_id")
        reference = None
        selection = {}
        if isinstance(parent_id, str) and parent_id in by_id:
            reference = dict(by_id[parent_id])
            selection = {"mode": "parent", "reference_run_id": parent_id}
        else:
            selected = run_diff.select_reference_run(dict(run), records, mode="previous_compatible")
            if selected.get("ok") is True and isinstance(selected.get("run"), Mapping):
                reference = dict(selected["run"])
                selection = deepcopy(selected.get("selection") or {"mode": "previous_compatible"})
        if reference is None:
            return {
                "state": "unavailable",
                "reason": (
                    "no unambiguous earlier compatible run was found automatically; pass "
                    "?reference_run_id= to choose one explicitly"
                ),
            }

    from clozn.analysis import run_diff

    reference_id = str(reference.get("id"))
    try:
        comparison = run_diff.compare_runs(reference, dict(run))
    except Exception as exc:  # structural synthesis must fail closed, never lose the rest of the doc
        return {
            "state": "failed",
            "reference_run_id": reference_id,
            "reason": f"structural run comparison failed: {type(exc).__name__}: {exc}",
        }
    if comparison.get("ok") is False:
        return {
            "state": "failed",
            "reference_run_id": reference_id,
            "reason": str(comparison.get("error") or "structural run comparison failed"),
        }
    return {
        "state": "supported",
        "reference_run_id": reference_id,
        "selection": selection,
        "comparison": comparison,
        "href": f"/runs/compare?a={reference_id}&b={run_id}",
    }


# ---------------------------------------------------------------------------------------- capabilities
def _exact_fork_capability(run: Mapping[str, Any], index: int, *, worker_ready: bool) -> dict:
    """A cheap PREVIEW of whether the observation-first force-token action is requestable.

    Composes clozn.replay.checkpoint_capture's own pure precondition helpers (never its network-calling
    capture_parent_checkpoint) so this preview cannot silently drift from what that route actually
    accepts. It is a preview, not authoritative: the POST route re-validates independently and is the
    only thing allowed to actually capture a checkpoint."""
    action = _workbench_action("POST", f"/runs/{run.get('id')}/tokens/{index}/force-token")
    if not worker_ready:
        return {
            "available": False, "snapshot_state": "worker_unreachable",
            "reason": "no live engine worker is currently reachable", "action": action,
        }

    from clozn.replay.checkpoint_capture import _sampler, _trace_history

    history = _trace_history(run)
    if history is None:
        return {
            "available": False, "snapshot_state": "missing_token_history",
            "reason": "the run needs complete, aligned response token pieces and token ids",
            "action": action,
        }
    pieces, output_ids = history
    if index < 0 or index >= len(pieces):
        return {
            "available": False, "snapshot_state": "position_out_of_range",
            "reason": f"token index {index} is outside the recorded response's {len(pieces)} tokens",
            "action": action,
        }
    final_prompt = run.get("final_prompt")
    if not isinstance(final_prompt, str) or not final_prompt:
        return {
            "available": False, "snapshot_state": "missing_final_prompt",
            "reason": "the run has no exact recorded prompt text to restore a checkpoint from",
            "action": action,
        }
    meta = run.get("meta") if isinstance(run.get("meta"), Mapping) else {}
    prompt_tokens = meta.get("prompt_tokens")
    if not (isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool) and prompt_tokens > 0):
        return {
            "available": False, "snapshot_state": "missing_prompt_boundary",
            "reason": "the run has no positive recorded prompt-token boundary", "action": action,
        }
    if (
        run.get("substrate") != "engine"
        or not isinstance(meta.get("stream"), bool)
        or run.get("parent_run_id") is not None
        or bool(run.get("changes_applied"))
        or bool(run.get("reasoning"))
        or bool(run.get("output_contract"))
        or "clozn_guard_receipt" in meta
    ):
        return {
            "available": False, "snapshot_state": "run_shape_ineligible",
            "reason": (
                "exact-fork capture needs an organic engine run with a recorded stream shape and no "
                "structured grammar, hidden reasoning, generation guard, or prior intervention"
            ),
            "action": action,
        }
    if _sampler(run, len(output_ids)) is None:
        return {
            "available": False, "snapshot_state": "sampler_provenance_missing",
            "reason": "the run's sampler mode, parameters, and fixed seed are not exactly recoverable",
            "action": action,
        }
    return {"available": True, "snapshot_state": "not_attempted", "action": action}


def _source_measurement_capability(investigation_doc: Mapping[str, Any], run_id: str, index: int) -> dict:
    section = (investigation_doc.get("sections") or {}).get("prompt_source_influence")
    if not isinstance(section, Mapping):
        return {
            "available": False, "status": "unavailable",
            "reason": "prompt/source influence evidence is unavailable",
        }
    state = str(section.get("state") or "unavailable")
    out: dict[str, Any] = {
        "available": state in {"measured_effect", "below_measurement_floor"},
        "status": state,
    }
    reason = section.get("reason")
    if not out["available"]:
        # "never a bare false": fall back to a generic-but-honest reason if the composed investigation
        # section for some reason omitted one -- this capability must never report unavailable silently.
        out["reason"] = (
            reason if isinstance(reason, str) and reason
            else f"prompt/source influence evidence is {state}, not a completed measurement"
        )
    # Milestone F: source-measure IS this same investigation evidence, requested through the token-
    # workbench's OWN action surface (which delegates read-only to the identical influence-map job
    # machinery investigation.py's own action already pointed at) -- point here, not at the legacy
    # investigation action, now that it exists.
    out["action"] = _workbench_action("POST", f"/runs/{run_id}/tokens/{index}/source-measure")
    return out


def _causal_trace_capability(run: Mapping[str, Any], index: int, *, worker_ready: bool) -> dict:
    action = _workbench_action("POST", f"/runs/{run.get('id')}/tokens/{index}/causal-trace")
    final_prompt = run.get("final_prompt")
    if not (isinstance(final_prompt, str) and final_prompt):
        return {
            "available": False, "status": "unavailable",
            "reason": "run has no recorded final_prompt (the exact rendered prompt) to trace",
            "action": action,
        }
    response = run.get("response")
    if not (isinstance(response, str) and response):
        return {
            "available": False, "status": "unavailable",
            "reason": "run has no recorded response text to trace", "action": action,
        }
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
    pieces = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
    if index < 0 or index >= len(pieces):
        return {
            "available": False, "status": "unavailable",
            "reason": f"token index {index} is outside the recorded response's {len(pieces)} tokens",
            "action": action,
        }
    if not worker_ready:
        return {
            "available": False, "status": "unavailable",
            "reason": "no live engine worker is currently reachable", "action": action,
        }
    return {"available": True, "status": "ready", "action": action}


def _mechanistic_diff_capability(
    run: Mapping[str, Any], index: int, *, reference_run_id: str | None,
    reference_run: Mapping[str, Any] | None,
) -> dict:
    if not reference_run_id:
        return {
            "available": False,
            "reason": (
                "no reference run selected; pass ?reference_run_id= naming a run from a DIFFERENT "
                "model to compare against"
            ),
        }
    if reference_run is None:
        return {"available": False, "reason": f"reference run {reference_run_id!r} was not found"}
    ref_identity = reference_run.get("identity") if isinstance(reference_run.get("identity"), Mapping) else {}
    run_identity = run.get("identity") if isinstance(run.get("identity"), Mapping) else {}
    ref_sha = ref_identity.get("model_sha256")
    run_sha = run_identity.get("model_sha256")
    if not (isinstance(run_sha, str) and run_sha):
        return {"available": False, "reason": "this run has no recorded model identity"}
    if not (isinstance(ref_sha, str) and ref_sha):
        return {"available": False, "reason": "reference run has no recorded model identity"}
    if ref_sha == run_sha:
        return {
            "available": False,
            "reason": (
                "reference run shares this run's model identity; mechanistic diff compares two "
                "different models"
            ),
        }
    return {
        "available": True,
        "reason": (
            "reference run has a different recorded model identity, eligible for a cross-model "
            "mechanistic diff"
        ),
        # POST here runs the same pair-compatibility gate authoritatively and, when the gateway has a
        # managed model registry, queues the comparison through the sequential model-loader path. A
        # legacy single-worker gateway returns a typed unavailable response; the CLI remains a useful
        # fallback for environments without managed model routing.
        "action": _workbench_action(
            "POST", f"/runs/{run.get('id')}/tokens/{index}/mechanistic-diff",
            request_body={"reference_run_id": reference_run_id}),
    }


# ------------------------------------------------------------------------------------------- the build
def build(
    run: Mapping[str, Any],
    index: int,
    *,
    investigation_doc: Mapping[str, Any],
    related_runs: Sequence[Mapping[str, Any]] = (),
    reference_run_id: str | None = None,
    reference_run: Mapping[str, Any] | None = None,
    worker_ready: bool = False,
) -> dict:
    """Compose existing evidence for one (run, token index) pair without executing or persisting
    anything. `investigation_doc` is an already-built clozn.run-investigation.v1 document (composed by
    the caller via clozn.runs.investigation.build) -- this function never rebuilds it. Raises
    ValueError when the run has no trace or `index` is out of range (the route's 400)."""
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run.get("id"):
        raise ValueError("token workbench requires a stored run with a non-empty id")
    if not isinstance(investigation_doc, Mapping):
        raise ValueError("token workbench requires an already-built investigation document")
    index = int(index)
    run_id = str(run["id"])

    token_section = _token_section(run, index)  # raises ValueError first -- the cheapest possible check

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "index": index,
        "sections": {
            "run": _run_section(run),
            "token": token_section,
            "context": _context_section(investigation_doc),
            "comparison": _comparison_section(
                run, related_runs, reference_run_id=reference_run_id, reference_run=reference_run),
            "readouts": _readouts_section(run, index),
        },
        "capabilities": {
            "exact_fork": _exact_fork_capability(run, index, worker_ready=worker_ready),
            "source_measurement": _source_measurement_capability(investigation_doc, run_id, index),
            "causal_trace": _causal_trace_capability(run, index, worker_ready=worker_ready),
            "mechanistic_diff": _mechanistic_diff_capability(
                run, index, reference_run_id=reference_run_id, reference_run=reference_run),
        },
    }
    if reference_run_id:
        document["reference_run_id"] = reference_run_id
    return document
