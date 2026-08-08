"""Preview/confirm/keep orchestration for bounded corrective actions.

The generation step and the keep step are deliberately separate.  Confirming a preview creates one
matched greedy comparison and persists its structured outcome; it never changes any standing policy.
Keeping a successful corrected child selects it as ITS OWN parent run's revision -- a per-run,
request-local choice, not a session/profile preference that would shape a later, unrelated request.
Durable, auto-applied corrections (session/profile scope, persistent activation) were retired; see
docs/CAPABILITIES.md. The keep transaction still has its own idempotency key and undo, but "undo"
here only ever reverts which revision THIS run points at.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from copy import deepcopy

from clozn._io import atomic_write_json
from clozn.behavior import registry


SCHEMA = "clozn.corrective-flow.v1"
RESULT_SCHEMA = "clozn.corrective-retry-result.v1"
_PATH = os.path.join(os.path.expanduser("~"), ".clozn", "corrective_flow.json")
_LOCK = threading.RLock()
_IDEMPOTENCY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,127}$")


class CorrectiveFlowError(ValueError):
    """A conflict-safe, user-actionable flow refusal."""


def _path() -> str:
    return _PATH


def _empty() -> dict:
    return {
        "schema": SCHEMA,
        "previews": {},
        "results": {},
        "idempotency": {},
        "transactions": {},
    }


def _load(*, strict: bool = False) -> dict:
    try:
        with open(_path(), encoding="utf-8") as handle:
            raw = json.load(handle)
        if not isinstance(raw, dict) or raw.get("schema") != SCHEMA:
            raise ValueError("invalid corrective action flow store")
        return {
            "schema": SCHEMA,
            "previews": dict(raw.get("previews") or {}),
            "results": dict(raw.get("results") or {}),
            "idempotency": dict(raw.get("idempotency") or {}),
            "transactions": dict(raw.get("transactions") or {}),
        }
    except FileNotFoundError:
        return _empty()
    except Exception as exc:
        if strict:
            raise CorrectiveFlowError(f"corrective action flow store is unreadable: {exc}") from None
        return _empty()


def _save(doc: dict) -> None:
    # These are bounded product receipts, not an unbounded event stream.
    for key, limit in (("previews", 200), ("results", 200), ("idempotency", 400),
                       ("transactions", 200)):
        values = list((doc.get(key) or {}).items())
        if len(values) > limit:
            doc[key] = dict(values[-limit:])
    atomic_write_json(_path(), doc, ensure_ascii=False, indent=2, sort_keys=True)


def _digest(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def run_fingerprint(run: Mapping) -> str:
    """Hash only evidence that governs a replay, excluding unrelated derived attachments."""
    return _digest({
        "id": run.get("id"),
        "messages": run.get("messages"),
        "response": run.get("response"),
        "identity": run.get("identity"),
        "context_receipt": run.get("context_receipt"),
        "session_key": run.get("session_key"),
        "active_profile": (run.get("meta") or {}).get("active_profile"),
        "behavior": run.get("behavior"),
    })


def _revision_hash(run: Mapping) -> str:
    return _digest(run.get("selected_revision"))


def _validate_key(value: str) -> str:
    value = str(value or "")
    if not _IDEMPOTENCY_RE.fullmatch(value):
        raise CorrectiveFlowError(
            "idempotency_key must be 8-128 letters, digits, dots, colons, dashes, or underscores"
        )
    return value


def scope_eligibility(run: Mapping, action_id: str, *, active_profile: str | None) -> list[dict]:
    """Return the (now singular) once-scoped keep eligibility for this run/action.

    ``active_profile`` is accepted for call-signature stability with existing callers
    (clozn.server.routes.corrective_actions) even though it no longer selects a scope -- durable
    session/profile scoping was retired.
    """
    del active_profile
    return [{
        "scope": "once",
        "available": True,
        "target": str(run.get("id") or ""),
        "prior_hash": _revision_hash(run),
        "note": "selects the corrected child as this run's revision; no future request is affected",
    }]


def registry_for_run(run: Mapping, *, steer=None, active_profile: str | None = None) -> dict:
    doc = registry.build_registry(steer=steer)
    for action in doc["actions"]:
        action["scope_eligibility"] = scope_eligibility(
            run, action["id"], active_profile=active_profile
        )
    doc["run_id"] = run.get("id")
    doc["run_fingerprint"] = run_fingerprint(run)
    return doc


def _backend_preview(action: dict, requested_backend: str) -> dict:
    if requested_backend not in {"prompt_policy", "control_vector"}:
        raise CorrectiveFlowError(
            "requested_backend must be prompt_policy or control_vector"
        )
    entries = [
        item for item in action.get("backends", [])
        if item.get("type") == requested_backend
        or item.get("requested_type") == requested_backend
    ]
    entry = entries[0] if entries else {}
    if requested_backend == "prompt_policy":
        return {
            "requested_backend": requested_backend,
            "expected_executed_backend": "prompt_policy",
            "expected_fallback": False,
            "qualification": entry.get("qualification", "generic"),
            "qualification_id": entry.get(
                "qualification_id", "clozn.prompt-policy.generic.v1"
            ),
        }
    if entry.get("type") == "control_vector" and entry.get("available") is True:
        return {
            "requested_backend": requested_backend,
            "expected_executed_backend": "control_vector",
            "expected_fallback": False,
            "qualification": "model_build_exact",
            "qualification_id": entry.get("qualification_id"),
        }
    return {
        "requested_backend": requested_backend,
        "expected_executed_backend": "prompt_policy",
        "expected_fallback": True,
        "qualification": "generic",
        "qualification_id": "clozn.prompt-policy.generic.v1",
        "unavailability_reason": (
            entry.get("unavailability_reason")
            or entry.get("reason")
            or "control-vector backend is not qualified for this exact model"
        ),
    }


def create_preview(
    run: Mapping,
    action_id: str,
    requested_backend: str = "prompt_policy",
    *,
    steer=None,
    active_profile: str | None = None,
    now: float | None = None,
) -> dict:
    if not isinstance(run, Mapping) or not run.get("id"):
        raise CorrectiveFlowError("run must be a stored run with an id")
    actions = registry_for_run(run, steer=steer, active_profile=active_profile)
    action = next((item for item in actions["actions"] if item["id"] == action_id), None)
    if action is None:
        raise CorrectiveFlowError(
            "action_id must be one of: " + ", ".join(registry.action_ids())
        )
    clock = float(time.time() if now is None else now)
    preview = {
        "preview_id": "fix_preview_" + uuid.uuid4().hex[:20],
        "status": "ready",
        "created_ts": clock,
        "expires_ts": clock + 3600.0,
        "parent_run_id": str(run["id"]),
        "parent_run_fingerprint": run_fingerprint(run),
        "action": {
            "id": action["id"],
            "label": action["label"],
            "description": action["description"],
        },
        "execution": _backend_preview(action, requested_backend),
        "scope_eligibility": deepcopy(action["scope_eligibility"]),
        "comparison_contract": {
            "baseline": "matched greedy replay under the current runtime policy",
            "corrected": "matched greedy replay with the bounded action",
            "stored_original": "context only; it may have been sampled",
        },
    }
    with _LOCK:
        doc = _load(strict=True)
        doc["previews"][preview["preview_id"]] = preview
        _save(doc)
    return deepcopy(preview)


def cancel_preview(preview_id: str) -> dict:
    with _LOCK:
        doc = _load(strict=True)
        preview = doc["previews"].get(str(preview_id))
        if preview is None:
            raise CorrectiveFlowError("unknown corrective action preview")
        status = preview.get("status")
        if status == "ready":
            preview["status"] = "cancelled"
        elif status == "confirming":
            preview["status"] = "cancel_requested"
        elif status not in {"cancelled", "cancel_requested"}:
            raise CorrectiveFlowError(f"preview cannot be cancelled from status {status!r}")
        _save(doc)
        return deepcopy(preview)


def _result_from_comparison(preview: dict, comparison: Mapping, *, now: float) -> dict:
    requested = preview["execution"]["requested_backend"]
    executed = comparison.get("executed_backend") or comparison.get("backend")
    fallback = bool(comparison.get("backend_fallback"))
    children = deepcopy(comparison.get("child_outcomes") or {})
    for arm in ("baseline", "corrected"):
        children.setdefault(arm, {"status": "not_run"})
        # Replies live once in comparison; child outcome remains linkage/error metadata.
        children[arm].pop("reply", None)
    outcome = deepcopy(comparison.get("outcome") or {"status": "execution_error"})
    result = {
        "schema_version": RESULT_SCHEMA,
        "result_id": "fix_result_" + uuid.uuid4().hex[:20],
        "created_ts": now,
        "parent_run_id": preview["parent_run_id"],
        "preview_id": preview["preview_id"],
        "action": deepcopy(preview["action"]),
        "user_intent": {"action_id": preview["action"]["id"]},
        "execution": {
            "requested_backend": requested,
            "executed_backend": executed,
            "fallback": fallback,
            "qualification": (
                "generic" if executed == "prompt_policy"
                else preview["execution"].get("qualification")
            ),
            "qualification_id": (
                "clozn.prompt-policy.generic.v1" if executed == "prompt_policy"
                else preview["execution"].get("qualification_id")
            ),
            "identity": deepcopy(comparison.get("execution_identity")),
        },
        "children": children,
        "comparison": {
            "stored_original_reply": str(comparison.get("stored_original_reply") or ""),
            "baseline_reply": str(comparison.get("baseline_reply") or ""),
            "corrected_reply": str(comparison.get("corrected_reply") or ""),
            "note": str(comparison.get("comparison_note") or ""),
            "changed": bool(comparison.get("changed")),
        },
        "metrics": deepcopy(comparison.get("delta") or {}),
        "coherence": deepcopy(comparison.get("coherence") or {}),
        "intervention_observed": bool(comparison.get("intervention_observed")),
        "scope_eligibility": deepcopy(preview["scope_eligibility"]),
        "outcome": outcome,
    }
    refusal = None
    if outcome.get("status") == "succeeded":
        if result["coherence"].get("degenerate") is True:
            refusal = "the corrected child was degenerate"
        elif result["intervention_observed"] is not True:
            refusal = "the requested corrective intervention was not observed"
    if refusal:
        result["scope_eligibility"] = [
            {
                **item,
                "available": False,
                "unavailability_reason": refusal,
            }
            for item in result["scope_eligibility"]
        ]
    return result


def confirm_preview(
    preview_id: str,
    idempotency_key: str,
    current_run: Mapping,
    execute: Callable[[dict], Mapping | None],
    *,
    now: float | None = None,
) -> dict:
    key = _validate_key(idempotency_key)
    clock = float(time.time() if now is None else now)
    with _LOCK:
        doc = _load(strict=True)
        existing = doc["idempotency"].get(key)
        if existing:
            if existing.get("operation") != "confirm" or existing.get("preview_id") != preview_id:
                raise CorrectiveFlowError("idempotency key was already used for another operation")
            if existing.get("status") == "completed":
                return deepcopy(doc["results"][existing["result_id"]])
            raise CorrectiveFlowError("confirmation with this idempotency key is already in progress")
        preview = doc["previews"].get(str(preview_id))
        if preview is None:
            raise CorrectiveFlowError("unknown corrective action preview")
        if preview.get("status") != "ready":
            raise CorrectiveFlowError(
                f"preview cannot be confirmed from status {preview.get('status')!r}"
            )
        if float(preview.get("expires_ts") or 0) <= clock:
            preview["status"] = "expired"
            _save(doc)
            raise CorrectiveFlowError("corrective action preview expired; create a new preview")
        if run_fingerprint(current_run) != preview["parent_run_fingerprint"]:
            raise CorrectiveFlowError("run evidence changed after preview; refusing stale confirmation")
        preview["status"] = "confirming"
        doc["idempotency"][key] = {
            "operation": "confirm",
            "preview_id": preview_id,
            "status": "in_progress",
        }
        _save(doc)

    try:
        comparison = execute(deepcopy(preview))
        if not isinstance(comparison, Mapping):
            comparison = {
                "stored_original_reply": str(current_run.get("response") or ""),
                "child_outcomes": {
                    "baseline": {"status": "error", "error": {
                        "code": "generation_failed",
                        "message": "corrective comparison returned no structured outcome",
                    }},
                    "corrected": {"status": "not_run"},
                },
                "outcome": {"status": "execution_error"},
            }
    except Exception as exc:
        comparison = {
            "stored_original_reply": str(current_run.get("response") or ""),
            "child_outcomes": {
                "baseline": {"status": "error", "error": {
                    "code": "generation_error", "message": str(exc),
                }},
                "corrected": {"status": "not_run"},
            },
            "outcome": {"status": "execution_error"},
        }

    with _LOCK:
        doc = _load(strict=True)
        saved_preview = doc["previews"].get(preview_id) or preview
        cancelled = saved_preview.get("status") == "cancel_requested"
        result = _result_from_comparison(saved_preview, comparison, now=clock)
        if cancelled:
            result["outcome"] = {
                "status": "cancelled",
                "note": "generation completed after cancellation; no revision or policy was kept",
            }
        from clozn import schemas
        schemas.validate(result, RESULT_SCHEMA)
        saved_preview["status"] = "cancelled" if cancelled else "consumed"
        saved_preview["result_id"] = result["result_id"]
        doc["previews"][preview_id] = saved_preview
        doc["results"][result["result_id"]] = result
        doc["idempotency"][key] = {
            "operation": "confirm",
            "preview_id": preview_id,
            "status": "completed",
            "result_id": result["result_id"],
        }
        _save(doc)
        return deepcopy(result)


def get_result(result_id: str) -> dict | None:
    with _LOCK:
        result = _load().get("results", {}).get(str(result_id))
        return deepcopy(result) if isinstance(result, dict) else None


def get_preview(preview_id: str) -> dict | None:
    with _LOCK:
        preview = _load().get("previews", {}).get(str(preview_id))
        return deepcopy(preview) if isinstance(preview, dict) else None


def keep_result(
    result_id: str,
    scope: str,
    expected_prior_hash: str,
    idempotency_key: str,
    *,
    get_run: Callable[[str], dict | None],
    replace_run: Callable[[dict], bool],
    now: float | None = None,
) -> dict:
    key = _validate_key(idempotency_key)
    if scope != "once":
        raise CorrectiveFlowError(
            "scope must be once; durable session/profile corrections were retired -- "
            "keeping a result only ever selects it as this run's own revision"
        )
    clock = float(time.time() if now is None else now)
    with _LOCK:
        doc = _load(strict=True)
        existing = doc["idempotency"].get(key)
        if existing:
            if (existing.get("operation") != "keep"
                    or existing.get("result_id") != result_id
                    or existing.get("scope") != scope):
                raise CorrectiveFlowError("idempotency key was already used for another operation")
            if existing.get("status") == "completed":
                return deepcopy(doc["results"][result_id])
            raise CorrectiveFlowError("keep operation is already in progress")
        result = doc["results"].get(str(result_id))
        if result is None:
            raise CorrectiveFlowError("unknown corrective action result")
        if (result.get("outcome") or {}).get("status") != "succeeded":
            raise CorrectiveFlowError("only a successful corrected result can be kept")
        if result.get("transaction"):
            raise CorrectiveFlowError("this corrective result was already kept")
        corrected_id = ((result.get("children") or {}).get("corrected") or {}).get("run_id")
        if not corrected_id or get_run(str(corrected_id)) is None:
            raise CorrectiveFlowError("corrected child run is missing; refusing keep")
        eligibility = next(
            (item for item in result.get("scope_eligibility", [])
             if item.get("scope") == scope),
            None,
        )
        if not eligibility or eligibility.get("available") is not True:
            raise CorrectiveFlowError(
                str((eligibility or {}).get("unavailability_reason")
                    or f"{scope} scope is unavailable")
            )
        if str(expected_prior_hash or "") != str(eligibility.get("prior_hash") or ""):
            raise CorrectiveFlowError("prior hash does not match the previewed scope state")
        doc["idempotency"][key] = {
            "operation": "keep", "result_id": result_id, "scope": scope,
            "status": "in_progress",
        }
        _save(doc)

        parent = get_run(result["parent_run_id"])
        if parent is None:
            doc["idempotency"].pop(key, None)
            _save(doc)
            raise CorrectiveFlowError("parent run is missing; refusing revision selection")
        if _revision_hash(parent) != expected_prior_hash:
            doc["idempotency"].pop(key, None)
            _save(doc)
            raise CorrectiveFlowError(
                "selected revision changed after preview; refusing stale apply"
            )
        before = deepcopy(parent.get("selected_revision"))
        after = {
            "result_id": result_id,
            "child_run_id": str(corrected_id),
            "action_id": result["action"]["id"],
            "selected_ts": clock,
        }
        parent["selected_revision"] = after
        if not replace_run(parent):
            doc["idempotency"].pop(key, None)
            _save(doc)
            raise CorrectiveFlowError("failed to select corrected child revision")
        transaction = {
            "id": "revision_" + uuid.uuid4().hex[:20],
            "scope": "once",
            "target": result["parent_run_id"],
            "before": before,
            "after": after,
            "created_ts": clock,
            "undone_ts": None,
        }
        result["user_intent"]["kept_scope"] = scope
        result["transaction"] = transaction
        from clozn import schemas
        schemas.validate(result, RESULT_SCHEMA)
        doc["transactions"][transaction["id"]] = {
            **deepcopy(transaction), "result_id": result_id
        }
        doc["results"][result_id] = result
        doc["idempotency"][key] = {
            "operation": "keep", "result_id": result_id, "scope": scope,
            "status": "completed",
        }
        _save(doc)
        return deepcopy(result)


def undo_keep(
    transaction_id: str,
    *,
    get_run: Callable[[str], dict | None],
    replace_run: Callable[[dict], bool],
    now: float | None = None,
) -> dict:
    clock = float(time.time() if now is None else now)
    with _LOCK:
        doc = _load(strict=True)
        tx = doc["transactions"].get(str(transaction_id))
        if tx is None:
            raise CorrectiveFlowError("unknown corrective action undo id")
        if tx.get("undone_ts") is not None:
            raise CorrectiveFlowError("corrective action was already undone")
        if tx.get("scope") != "once":
            # A transaction from before durable session/profile keeps were retired. The module that
            # could reverse it (clozn.behavior.corrective_retries) is gone; there is nothing left it
            # could still be applying to a new generation (see generation_gateway.py), so refuse
            # rather than guess at a mutation this build no longer knows how to perform.
            raise CorrectiveFlowError(
                "this transaction predates the retirement of durable session/profile corrections "
                "and can no longer be undone through this API; it has no effect on new generations"
            )
        parent = get_run(str(tx.get("target") or ""))
        if parent is None:
            raise CorrectiveFlowError("parent run is missing; refusing revision undo")
        if parent.get("selected_revision") != tx.get("after"):
            raise CorrectiveFlowError(
                "selected revision changed after this action; refusing stale undo"
            )
        if tx.get("before") is None:
            parent.pop("selected_revision", None)
        else:
            parent["selected_revision"] = deepcopy(tx["before"])
        if not replace_run(parent):
            raise CorrectiveFlowError("failed to restore prior selected revision")
        tx["undone_ts"] = clock
        result = doc["results"].get(str(tx.get("result_id") or ""))
        if isinstance(result, dict) and isinstance(result.get("transaction"), dict):
            result["transaction"]["undone_ts"] = clock
        _save(doc)
        return {
            "status": "undone",
            "undo_id": transaction_id,
            "scope": tx.get("scope"),
            "target": tx.get("target"),
        }


def compare_source_use(result_id: str, *, get_run: Callable[[str], dict | None]) -> dict:
    """Compare already-computed maps only; generation/scoring remains a separate expensive action."""
    with _LOCK:
        doc = _load(strict=True)
        result = doc["results"].get(str(result_id))
        if result is None:
            raise CorrectiveFlowError("unknown corrective action result")
        children = result.get("children") or {}
        baseline = get_run(str((children.get("baseline") or {}).get("run_id") or ""))
        corrected = get_run(str((children.get("corrected") or {}).get("run_id") or ""))
        if baseline is None or corrected is None:
            raise CorrectiveFlowError("both child runs are required for source-use comparison")
        left = baseline.get("influence_map")
        right = corrected.get("influence_map")
        for artifact in (left, right):
            if not (
                isinstance(artifact, dict)
                and artifact.get("schema") == "clozn.context_answer_influence.v1"
                and artifact.get("available") is True
                and artifact.get("status") == "ok"
            ):
                raise CorrectiveFlowError(
                    "compute an influence map for both child runs before comparing source use"
                )
        if left.get("method") != right.get("method") or left.get("thresholds") != right.get(
            "thresholds"
        ):
            raise CorrectiveFlowError(
                "source-use maps use different method/version/threshold contracts"
            )

        def observed(artifact: dict) -> dict:
            answer_ids = {
                str(item.get("id")) for item in artifact.get("answer_spans", [])
                if isinstance(item, dict) and item.get("id") is not None
            }
            clear_ids = {
                str(item.get("answer_span_id")) for item in artifact.get("links", [])
                if isinstance(item, dict) and item.get("clears_floor") is True
            } & answer_ids
            count = len(answer_ids)
            return {
                "answer_span_count": count,
                "answer_spans_with_clear_source": len(clear_ids),
                "observed_source_dependence_ratio": (
                    round(len(clear_ids) / count, 6) if count else 0.0
                ),
            }

        before, after = observed(left), observed(right)
        comparison = {
            "status": "available",
            "method": deepcopy(left["method"]),
            "thresholds": deepcopy(left["thresholds"]),
            "baseline": before,
            "corrected": after,
            "delta_observed_source_dependence_ratio": round(
                after["observed_source_dependence_ratio"]
                - before["observed_source_dependence_ratio"],
                6,
            ),
            "caveat": (
                "Higher observed source dependence under this intervention method does not "
                "establish that the sources or answer are factually correct."
            ),
        }
        result["source_use_comparison"] = comparison
        from clozn import schemas
        schemas.validate(result, RESULT_SCHEMA)
        doc["results"][result_id] = result
        _save(doc)
        return deepcopy(comparison)
