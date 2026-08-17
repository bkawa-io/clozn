"""V1 Run-scoped routes for the new StateRef/Generate time-travel kernel."""
from __future__ import annotations

from collections.abc import Mapping

from clozn.recipes.time_travel import (
    TimeTravelError,
    enumerate_answer_boundaries,
    resolve_time_travel,
    run_time_travel,
    time_travel_capabilities,
)

CLOZN_ROUTE_AUTOLOAD = True
_PREFIX = "/time-travel"


def _split(path: str):
    if not path.startswith("/runs/"):
        return None
    rest = path[len("/runs/"):]
    run_id, marker, tail = rest.partition(_PREFIX)
    if marker != _PREFIX or not run_id:
        return None
    return run_id, tail


def _run(h, run_id: str):
    import clozn.runs.store as runlog
    value = runlog.get_run(run_id)
    if value is None:
        h._json(404, {"error": "run not found", "code": "run_not_found"})
        return None
    return value


def _checkpoint(run_id: str, body: Mapping[str, object] | None):
    if isinstance(body, Mapping):
        for key in ("checkpoint_reference", "checkpoint"):
            value = body.get(key)
            if isinstance(value, Mapping):
                return value
    try:
        from clozn.replay.checkpoint_pin_store import resolve_pin
        resolved = resolve_pin(run_id)
        envelope = resolved.get("envelope") if isinstance(resolved, Mapping) and resolved.get("ok") is True else None
        if isinstance(envelope, Mapping):
            reference = envelope.get("checkpoint_reference")
            return reference if isinstance(reference, Mapping) else envelope
    except Exception:
        pass
    return None


def _identity_and_selection(h, run, *, route: str, required: bool):
    from clozn.server.model_routing import select_control_model_for_run
    selection = select_control_model_for_run(h, run.get("model"), route=route)
    if selection is None and required:
        return None, None, None
    if selection is None:
        return None, None, None
    if selection.runtime_key is not None:
        runtime = dict(selection.runtime_key)
        worker = dict(selection.worker_identity) if isinstance(selection.worker_identity, Mapping) else None
    else:
        # Reuse the existing run-scoped identity normalization seam. This is
        # planning/execution provenance, not the historical fork API.
        from clozn.server.routes.execution_fork import _identity_facts
        runtime, worker, _engine = _identity_facts(selection)
    return selection, runtime, worker


def _error(h, exc: TimeTravelError, *, default_status: int = 422):
    status = 404 if exc.code == "run_not_found" else (400 if exc.code in {"force_token_unsupported", "generation_unsupported"} else default_status)
    h._json(status, {"status": exc.status, "error": str(exc), "code": exc.code})


def try_get(h, path):
    parsed = _split(path)
    if parsed is None:
        return False
    run_id, tail = parsed
    run = _run(h, run_id)
    if run is None:
        return True
    if tail == "/boundaries":
        try:
            boundaries = enumerate_answer_boundaries(run)
            h._json(200, {"run_id": run_id, "boundaries": [item.to_dict() for item in boundaries]})
        except TimeTravelError as exc:
            _error(h, exc)
        return True
    if tail == "/capabilities":
        try:
            h._json(200, {"run_id": run_id, "capabilities": time_travel_capabilities(run)})
        except Exception as exc:
            h._json(422, {"status": "unavailable", "error": str(exc), "code": "capabilities_unavailable"})
        return True
    return False


def _body_object(body):
    return body if isinstance(body, Mapping) else {}


def _position(body):
    value = body.get("boundary", body.get("position"))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TimeTravelError("boundary must be a non-negative integer", code="token_boundary_out_of_range")
    return value


def try_post(h, path, body):
    parsed = _split(path)
    if parsed is None:
        return False
    run_id, tail = parsed
    run = _run(h, run_id)
    if run is None:
        return True
    body = _body_object(body)
    if tail == "/resolve":
        try:
            position = _position(body)
            policy = body.get("policy", "exact_preferred")
            if not isinstance(policy, str):
                raise TimeTravelError("policy must be a string", code="invalid_resolution_policy")
            checkpoint = _checkpoint(run_id, body)
            if checkpoint is None and policy != "exact_required":
                # Reconstructed planning is model-free and must remain
                # inspectable even when no live worker is currently ready.
                runtime = worker = None
            else:
                _selection, runtime, worker = _identity_and_selection(
                    h, run, route="/runs/<id>/time-travel/resolve", required=False,
                )
            result = resolve_time_travel(
                run, position=position, policy=policy, checkpoint=checkpoint,
                runtime_identity=runtime, worker_identity=worker,
                token_id=body.get("token_id"), token_piece=body.get("token_piece"),
            )
            h._json(200, result.to_dict())
        except TimeTravelError as exc:
            _error(h, exc)
        return True
    if tail not in {"/continue", "/force-token"}:
        if tail != "/materialize":
            return False
        try:
            experiment_id = body.get("experiment_id")
            arm_id = body.get("arm_id")
            observation_id = body.get("observation_id")
            if not all(isinstance(value, str) and value for value in (experiment_id, arm_id, observation_id)):
                raise TimeTravelError("experiment_id, arm_id, and observation_id are required", code="materialization_failed")
            import clozn.runs.store as runlog
            from clozn.experiments.persistence import ObservationStore
            from clozn.experiments.materialize import materialize_generated_observation
            result = materialize_generated_observation(
                run, str(experiment_id), str(arm_id), observation_id=str(observation_id),
                observation_store=ObservationStore(), reload_parent=runlog.get_run,
            )
            h._json(201 if result.get("state") == "completed" else 409, result)
        except TimeTravelError as exc:
            _error(h, exc, default_status=409)
        except Exception as exc:
            from clozn.experiments.materialize import MaterializationStaleError
            if isinstance(exc, MaterializationStaleError):
                h._json(409, {"status": "failed", "error": str(exc), "code": "observation_stale"})
                return True
            h._json(409, {"status": "failed", "error": str(exc), "code": "materialization_failed"})
        return True
    try:
        position = _position(body)
        policy = body.get("policy", "exact_preferred")
        if not isinstance(policy, str):
            raise TimeTravelError("policy must be a string", code="invalid_resolution_policy")
        if tail == "/force-token" and body.get("token_id") is None and body.get("token_piece") is None:
            raise TimeTravelError("force-token requires token_id or token_piece", code="force_token_unsupported")
        selection, runtime, worker = _identity_and_selection(
            h, run, route=f"/runs/<id>/time-travel{tail}", required=True,
        )
        if selection is None or selection.sub is None:
            h._json(503, {"status": "unavailable", "error": "time-travel requires a ready product worker", "code": "generation_unsupported"})
            return True
        result = run_time_travel(
            run, position=position, policy=policy, max_new=body.get("max_new", 32),
            token_id=body.get("token_id") if tail == "/force-token" else None,
            token_piece=body.get("token_piece") if tail == "/force-token" else None,
            checkpoint=_checkpoint(run_id, body), runtime_identity=runtime, worker_identity=worker,
            substrate=selection.sub, execution_adapter=None,
            observation_store=__import__("clozn.experiments.persistence", fromlist=["ObservationStore"]).ObservationStore(),
            decode_mode=body.get("decode_mode"), sampling=body.get("sampling"), stop=body.get("stop"),
        )
        h._json(200, result.to_dict())
    except TimeTravelError as exc:
        _error(h, exc)
    except Exception as exc:
        h._json(422, {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "code": "model_execution_failed"})
    return True


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_get", "try_post"]
