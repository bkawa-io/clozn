"""Thin Minimal Context winner materialization route over the generic kernel."""
from __future__ import annotations

from collections.abc import Mapping

CLOZN_ROUTE_AUTOLOAD = True
# The winner-bound result route in ``minimal_context.py`` is the product
# surface.  Keep this module importable as a historical low-level test seam,
# but do not expose the arbitrary client-selected branch endpoint.
CLOZN_ROUTE_ENABLED = False
_SUFFIX = "/minimal-context/branch"


def _required(body: Mapping[str, object], name: str):
    value = body.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} is required")
    return value


def try_post(h, path, body):
    if not (path.startswith("/runs/") and path.endswith(_SUFFIX)):
        return False
    import clozn.runs.store as runlog
    from clozn.experiments.materialize import MaterializationError, materialize_arm
    from clozn.experiments.persistence import ObservationStore
    from clozn.server.model_routing import select_control_model_for_run

    run_id = path[len("/runs/"):-len(_SUFFIX)]
    parent = runlog.get_run(run_id)
    if parent is None:
        h._json(404, {"error": "run not found", "code": "run_not_found"})
        return True
    if not isinstance(body, Mapping):
        h._json(400, {"error": "body must be an object", "code": "invalid_body"})
        return True
    try:
        experiment_id = _required(body, "experiment_id")
        arm_id = _required(body, "arm_id")
        observation_id = _required(body, "observation_id")
    except ValueError as exc:
        h._json(400, {"error": str(exc), "code": "invalid_materialization_reference"})
        return True
    selection = select_control_model_for_run(h, parent.get("model"), route="/runs/<id>/minimal-context/branch")
    if selection is None:
        return True
    try:
        result = materialize_arm(
            parent, experiment_id, arm_id, substrate=selection.sub,
            reload_parent=runlog.get_run, observation_id=observation_id,
            require_preserved=True, observation_store=ObservationStore(),
        )
    except MaterializationError as exc:
        h._json(409, {"error": str(exc), "code": "minimal_context_materialization_rejected"})
        return True
    h._json(201 if result.get("state") == "completed" else 409, result)
    return True


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_post"]
