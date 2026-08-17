"""Read-only generic Experiment and Observation endpoints."""
from __future__ import annotations

from clozn.experiments.persistence import (
    ExperimentPersistenceError,
    ObservationNotFound,
    ObservationStore,
)


CLOZN_ROUTE_AUTOLOAD = True


def try_get(h, path):
    clean = path.split("?", 1)[0]
    if clean.startswith("/experiments/"):
        experiment_id = clean[len("/experiments/"):]
        if not experiment_id or "/" in experiment_id or experiment_id == "types":
            return False
        try:
            payload = ObservationStore().get_experiment(experiment_id).to_dict()
        except ExperimentPersistenceError:
            h._json(404, {"error": "experiment not found", "code": "experiment_not_found"})
            return True
        h._json(200, payload)
        return True
    if clean.startswith("/observations/"):
        observation_id = clean[len("/observations/"):]
        if not observation_id or "/" in observation_id:
            return False
        try:
            payload = ObservationStore().get_observation(observation_id).to_dict()
        except (ObservationNotFound, ExperimentPersistenceError):
            h._json(404, {"error": "observation not found", "code": "observation_not_found"})
            return True
        h._json(200, payload)
        return True
    return False


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_get"]
