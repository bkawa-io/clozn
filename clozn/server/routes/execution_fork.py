"""Exact-only execution-fork planning and execution gateway.

This is deliberately separate from ``POST /runs/<id>/fork``.  That route remains the legacy
reconstructed-text path until FORK-02 can turn it into an explicit compatibility wrapper.
"""
from __future__ import annotations

from collections.abc import Mapping

CLOZN_ROUTE_AUTOLOAD = True
_PLAN_SUFFIX = "/execution-fork/plan"
_EXECUTE_SUFFIX = "/execution-fork"
_CHECKPOINT_SUFFIX = "/execution-fork/checkpoint"
_RESULT_PREFIX = "/execution-forks/"


def _sub_facts(sub) -> tuple[dict | None, dict | None, object | None]:
    engine = getattr(sub, "engine", None) if sub is not None else None
    if engine is None:
        return None, None, None
    health = {}
    try:
        value = engine.health()
        health = value if isinstance(value, dict) else {}
    except Exception:
        health = {}

    runtime = getattr(sub, "runtime_identity", None)
    try:
        runtime = runtime() if callable(runtime) else runtime
    except Exception:
        runtime = None
    if not isinstance(runtime, Mapping):
        runtime_key = getattr(sub, "runtime_key", None)
        runtime = runtime_key.as_dict() if hasattr(runtime_key, "as_dict") else runtime_key
    if not isinstance(runtime, Mapping):
        try:
            identity = sub.identity_meta() if callable(getattr(sub, "identity_meta", None)) else {}
        except Exception:
            identity = {}
        try:
            meta = sub.run_meta() if callable(getattr(sub, "run_meta", None)) else {}
        except Exception:
            meta = {}
        runtime = dict(identity or {})
        runtime["context_size"] = runtime.get(
            "context_size", (meta or {}).get("n_ctx", health.get("n_ctx")))
        runtime["backend"] = runtime.get(
            "backend", (meta or {}).get("device", health.get("device")))
        if "white_box_flags" not in runtime:
            flags = (meta or {}).get("white_box_flags")
            if not isinstance(flags, Mapping):
                capabilities = health.get("capabilities")
                capabilities = capabilities if isinstance(capabilities, Mapping) else {}
                flags = {
                    name: capabilities[name]
                    for name in ("sae", "jlens", "attn_knockout")
                    if isinstance(capabilities.get(name), bool)
                }
            runtime["white_box_flags"] = dict(flags)

    worker = getattr(sub, "worker_identity", None)
    try:
        worker = worker() if callable(worker) else worker
    except Exception:
        worker = None
    if not isinstance(worker, Mapping):
        generation = health.get("worker_generation_id")
        protocol = health.get("protocol_version")
        if isinstance(generation, str) and generation and protocol is not None:
            worker = {
                "worker_id": generation,
                "worker_generation_id": generation,
                "protocol_version": str(protocol),
            }
    return dict(runtime) if isinstance(runtime, Mapping) else None, (
        dict(worker) if isinstance(worker, Mapping) else None
    ), engine


def _identity_facts(selection) -> tuple[dict | None, dict | None, object | None]:
    """One resolved ``ModelSelection`` -> ``(runtime_identity, worker_identity, engine)``.

    Managed path: the router's own binding already carries exact runtime/worker identity (no
    extra probe). Legacy no-router path: ``select_control_model_for_run`` leaves those two ``None``
    (it is a zero-cost shim for callers that only need ``.sub``) -- derive them here, via
    ``_sub_facts``'s one ``engine.health()`` probe, for the heavier consumers that do: this
    function, ``clozn.server.routes.fork`` (the legacy fork wrapper), and
    ``clozn.server.routes.token_workbench_actions`` (the per-token fork/causal-trace actions).
    """
    if selection.runtime_key is not None:
        return dict(selection.runtime_key), dict(selection.worker_identity), selection.engine
    return _sub_facts(selection.sub)


def _parent_sub_facts(h, parent: Mapping, route_path: str):
    """Select the immutable parent's canonical model before reading worker facts.

    The process control/default worker is not a request-routing decision. In a multi-model gateway a
    historical run may belong to any configured worker, so every fork plan/capture/execute operation
    (and, per FORK-PIN-01, ``POST /runs/<id>/snapshot/pin`` -- see clozn/server/routes/snapshot.py,
    which reuses this exact function rather than re-deriving worker/runtime identity itself) resolves
    ``run.model`` through the same exact router as generation. Legacy one-worker serving has no router
    and retains its historical ``active_sub`` behavior. Thin wrapper over
    ``clozn.server.model_routing.select_control_model_for_run`` -- the shared resolver every other
    run-scoped route now goes through too -- plus ``_identity_facts`` to unpack the exact
    runtime/worker/engine triple this fork-shaped consumer needs (most other consumers only need
    ``.sub``).
    """
    from clozn.server.model_routing import select_control_model_for_run

    normalized_route = (
        "/runs/<id>/execution-fork/checkpoint"
        if route_path.endswith(_CHECKPOINT_SUFFIX)
        else "/runs/<id>/execution-fork/plan"
        if route_path.endswith(_PLAN_SUFFIX)
        else "/runs/<id>/snapshot/pin"
        if route_path.endswith("/snapshot/pin")
        else "/runs/<id>/time-machine/verify"
        if route_path.endswith("/time-machine/verify")
        else "/runs/<id>/time-machine/branch"
        if route_path.endswith("/time-machine/branch")
        else "/runs/<id>/execution-fork"
    )
    selection = select_control_model_for_run(h, parent.get("model"), route=normalized_route)
    if selection is None:
        return None
    return _identity_facts(selection)


def try_get(h, p):
    if not p.startswith(_RESULT_PREFIX):
        return False
    execution_id = p[len(_RESULT_PREFIX):]
    if not execution_id or "/" in execution_id:
        return False
    from clozn.replay import execution_fork_results
    result = execution_fork_results.get(execution_id)
    if result is None:
        h._json(404, {"error": "execution-fork result not found"})
    else:
        h._json(200, result)
    return True


def try_post(h, p, body):
    is_plan = p.startswith("/runs/") and p.endswith(_PLAN_SUFFIX)
    is_checkpoint = p.startswith("/runs/") and p.endswith(_CHECKPOINT_SUFFIX)
    is_execute = (
        p.startswith("/runs/") and p.endswith(_EXECUTE_SUFFIX)
        and not p.endswith(_PLAN_SUFFIX)
    )
    if not (is_plan or is_checkpoint or is_execute):
        return False
    suffix = (
        _PLAN_SUFFIX if is_plan
        else _CHECKPOINT_SUFFIX if is_checkpoint
        else _EXECUTE_SUFFIX
    )
    run_id = p[len("/runs/"):-len(suffix)]

    import clozn.runs.store as runlog
    parent = runlog.get_run(run_id)
    if parent is None:
        h._json(404, {"error": "run not found"})
        return True

    facts = _parent_sub_facts(h, parent, p)
    if facts is None:
        return True
    runtime, worker, engine = facts
    if engine is None or runtime is None or worker is None:
        h._json(503, {
            "error": "exact execution fork requires a ready identity-qualified product worker",
            "code": "execution_fork_worker_unavailable",
        })
        return True

    if is_checkpoint:
        use_pinned = False
        if isinstance(body, Mapping):
            unknown = set(body) - {"pinned"}
            if unknown:
                h._json(400, {
                    "error": "checkpoint capture accepts only an optional {\"pinned\": bool} body",
                    "code": "checkpoint_capture_options_unsupported",
                })
                return True
            raw_pinned = body.get("pinned", False)
            if not isinstance(raw_pinned, bool):
                h._json(400, {
                    "error": "pinned must be a boolean",
                    "code": "checkpoint_capture_invalid_pinned",
                })
                return True
            use_pinned = raw_pinned
        elif body:
            h._json(400, {
                "error": "checkpoint capture accepts only an optional {\"pinned\": bool} body",
                "code": "checkpoint_capture_options_unsupported",
            })
            return True
        pinned_envelope = None
        if use_pinned:
            from clozn.replay.checkpoint_pin_store import resolve_pin
            resolved = resolve_pin(run_id)
            if not isinstance(resolved, Mapping) or resolved.get("ok") is not True:
                reason = (
                    resolved.get("unavailable")
                    if isinstance(resolved, Mapping)
                    else "the pinned checkpoint could not be resolved")
                h._json(422, {
                    "error": str(reason),
                    "code": "checkpoint_capture_pinned_unavailable",
                    "run_id": run_id,
                })
                return True
            pinned_envelope = resolved.get("envelope")
            if not isinstance(pinned_envelope, Mapping):
                h._json(422, {
                    "error": "the pinned checkpoint did not contain a valid export envelope",
                    "code": "checkpoint_capture_pinned_unavailable",
                    "run_id": run_id,
                })
                return True
        try:
            from clozn.replay.checkpoint_capture import (
                CheckpointCaptureError,
                capture_parent_checkpoint,
            )
            artifact = capture_parent_checkpoint(
                parent,
                engine,
                runtime_identity=runtime,
                worker_identity=worker,
                checkpoint_envelope=pinned_envelope,
            )
        except CheckpointCaptureError as exc:
            h._json(400, {
                "error": str(exc),
                "code": "checkpoint_capture_request_invalid",
            })
            return True
        except Exception as exc:
            h._json(500, {
                "error": (
                    "checkpoint-reference receipt could not be built: "
                    f"{type(exc).__name__}: {exc}"),
                "code": "checkpoint_capture_receipt_error",
            })
            return True
        h._json(201 if artifact["status"] == "available" else 422, artifact)
        return True

    if is_plan:
        request = body.get("request")
        if not isinstance(request, Mapping):
            request = {
                "position": body.get("position"),
                "change": body.get("change"),
            }
        checkpoint = body.get("checkpoint_reference")
        if not isinstance(checkpoint, Mapping):
            h._json(400, {"error": "checkpoint_reference must be an object"})
            return True
        try:
            from clozn.replay.execution_fork import plan_execution_fork
            plan = plan_execution_fork(
                parent,
                request,
                checkpoint=checkpoint,
                runtime_identity=runtime,
                worker_identity=worker,
            )
        except ValueError as exc:
            h._json(400, {"error": str(exc)})
            return True
        h._json(200, plan)
        return True

    plan = body.get("plan")
    if not isinstance(plan, Mapping):
        h._json(400, {"error": "plan must be a clozn.execution-fork.v1 object"})
        return True
    cancel_check = getattr(h, "_execution_fork_cancelled", None)
    try:
        from clozn.replay.execution_fork_execute import (
            ExecutionForkExecutionError,
            execute_exact_fork,
        )
        result = execute_exact_fork(
            parent,
            plan,
            engine,
            runtime_identity=runtime,
            worker_identity=worker,
            reload_parent=runlog.get_run,
            cancel_check=cancel_check if callable(cancel_check) else None,
        )
    except ExecutionForkExecutionError as exc:
        h._json(409, {"error": str(exc), "code": "execution_fork_plan_invalid"})
        return True
    except Exception as exc:
        h._json(500, {
            "error": f"execution-fork receipt could not be persisted: {type(exc).__name__}: {exc}",
            "code": "execution_fork_persistence_error",
        })
        return True

    receipt = result["receipt"]
    status = 201 if receipt["phase"] == "completed" else (
        409 if receipt["phase"] == "cancelled" else 422)
    h._json(status, result)
    return True
