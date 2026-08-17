"""Durable experiment plans, arm associations, and standalone observations."""
from __future__ import annotations

from collections.abc import Mapping
from contextlib import closing
import json
import time
from copy import deepcopy
from typing import Any

from clozn.runs import store as run_store

from .evaluators import Evaluator
from .interventions import DeleteSource, ForceToken, Intervention, intervention_from_dict
from .kernel import Experiment
from .observations import (
    Observation, ObservationError, ObservationIntegrityError, TokenScoreObservation,
    GeneratedObservation, observation_from_dict, condition_for_intervention,
)
from .state import ExecutionState, canonical_json
from .state_ref import ResolvedState


EXPERIMENT_STORE_SCHEMA_VERSION = "clozn.experiment-store.v1"
EXPERIMENT_STATES = frozenset({"pending", "running", "completed", "cancelled", "failed", "blocked"})
ARM_STATES = frozenset({"pending", "running", "completed", "cancelled", "failed", "blocked", "not_executed"})
_UNSET = object()


class ExperimentPersistenceError(ValueError):
    """A durable experiment or association is malformed or inconsistent."""


class ObservationNotFound(ExperimentPersistenceError):
    """A requested immutable observation artifact does not exist."""


class ObservationPersistenceError(ExperimentPersistenceError):
    """The immutable evidence artifact could not be written."""


def _json(value: Any) -> str:
    return canonical_json(value)


def _loads(value: str | None, default: Any):
    if not isinstance(value, str):
        return deepcopy(default)
    try:
        return json.loads(value)
    except Exception as exc:
        raise ExperimentPersistenceError("durable experiment JSON is corrupt") from exc


class ExperimentArmView:
    """Read-side association: arm identity belongs here, never in Observation."""

    __slots__ = (
        "experiment_id", "arm_id", "ordinal", "intervention", "condition", "state",
        "observation_id", "observation", "error", "diagnostics",
    )

    def __init__(self, *, experiment_id: str, arm_id: str, ordinal: int,
                 intervention: Intervention | None, condition: Mapping[str, Any], state: str,
                 observation_id: str | None, observation: Observation | None,
                 error: Mapping[str, Any] | None = None, diagnostics: Mapping[str, Any] | None = None):
        self.experiment_id = experiment_id
        self.arm_id = arm_id
        self.ordinal = ordinal
        self.intervention = intervention
        self.condition = dict(condition)
        self.state = state
        self.observation_id = observation_id
        self.observation = observation
        self.error = dict(error or {})
        self.diagnostics = dict(diagnostics or {})

    @property
    def is_control(self) -> bool:
        return self.ordinal < 0

    @property
    def status(self) -> str:
        if self.observation is not None:
            return self.observation.status
        # ``state`` remains the authoritative arm execution state.  This
        # compatibility view preserves the old diagnostic status for a
        # blocked arm without fabricating an Observation row.
        return str(self.diagnostics.get("observation_status") or self.state)

    @property
    def completed(self) -> bool:
        return self.observation is not None and self.observation.completed

    def __getattr__(self, name: str) -> Any:
        observation = object.__getattribute__(self, "observation")
        if observation is not None:
            return getattr(observation, name)
        raise AttributeError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "arm_id": self.arm_id,
            "ordinal": self.ordinal,
            "intervention": self.intervention.to_dict() if self.intervention else None,
            "condition": dict(self.condition),
            "state": self.state,
            "observation_id": self.observation_id,
            "observation": self.observation.to_dict() if self.observation else None,
            "error": dict(self.error),
            "diagnostics": dict(self.diagnostics),
        }


class ExperimentView:
    """Assembled read model reconstructed from durable plan/arm/observation rows."""

    __slots__ = (
        "experiment_id", "base", "evaluator", "control", "arm_rows", "arm_observations",
        "arm_interventions", "state", "diagnostics", "timing", "execution_provenance",
        "created_ts", "updated_ts", "requested_by", "persisted",
    )

    def __init__(self, *, experiment_id: str, base: ExecutionState | ResolvedState, evaluator: Evaluator,
                 control: Observation | None, arm_rows: tuple[ExperimentArmView, ...],
                 state: str, diagnostics: Mapping[str, Any] | None = None,
                 timing: Mapping[str, Any] | None = None,
                 execution_provenance: Mapping[str, Any] | None = None,
                 created_ts: float | None = None, updated_ts: float | None = None,
                 requested_by: Mapping[str, Any] | None = None):
        self.experiment_id = experiment_id
        self.base = base
        self.evaluator = evaluator
        self.control = control
        self.arm_rows = tuple(arm_rows)
        self.arm_observations = tuple(row.observation for row in self.arm_rows if not row.is_control)
        self.arm_interventions = tuple(row.intervention for row in self.arm_rows if not row.is_control)
        self.state = state
        self.diagnostics = dict(diagnostics or {})
        self.timing = dict(timing or {})
        self.execution_provenance = dict(execution_provenance or {})
        self.created_ts = created_ts
        self.updated_ts = updated_ts
        self.requested_by = dict(requested_by or {})
        self.persisted = True

    @property
    def arms(self) -> tuple[ExperimentArmView, ...]:
        return tuple(row for row in self.arm_rows if not row.is_control)

    def observation_for(self, arm_id: str) -> Observation:
        for row in self.arms:
            if row.arm_id == arm_id and row.observation is not None:
                return row.observation
        raise KeyError(arm_id)

    def arm_for(self, arm_id: str) -> ExperimentArmView:
        for row in self.arms:
            if row.arm_id == arm_id:
                return row
        raise KeyError(arm_id)

    def score_delta_for(self, arm_id: str):
        from .observations import TokenScoreDelta
        arm = self.arm_for(arm_id)
        if not isinstance(self.control, TokenScoreObservation) or not isinstance(arm.observation, TokenScoreObservation):
            return TokenScoreDelta(observation_id=arm.observation_id or "obs_unavailable", status="unavailable",
                                   diagnostics={"reason": "baseline_score_unavailable"})
        return TokenScoreDelta.from_observations(self.control, arm.observation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EXPERIMENT_STORE_SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "base": self.base.to_dict(), "evaluator": self.evaluator.to_dict(),
            "state": self.state,
            "control": self.control.to_dict() if self.control else None,
            "arms": [row.to_dict() for row in self.arms],
            "diagnostics": dict(self.diagnostics), "timing": dict(self.timing),
            "execution_provenance": dict(self.execution_provenance),
            "created_ts": self.created_ts, "updated_ts": self.updated_ts,
            "requested_by": dict(self.requested_by),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())


class ObservationStore:
    """SQLite-backed durable store using the existing runs blob tree."""

    def __init__(self, *, runs_store: Any = run_store):
        self.runs_store = runs_store

    def _ensure(self) -> None:
        self.runs_store._ensure()

    def create_experiment(self, experiment: Experiment, *, requested_by: Mapping[str, Any] | None = None,
                          diagnostics: Mapping[str, Any] | None = None, now: float | None = None) -> str:
        if not isinstance(experiment, Experiment):
            raise TypeError("create_experiment requires a new-kernel Experiment")
        self._ensure()
        stamp = float(now if now is not None else time.time())
        stored_diagnostics = dict(diagnostics or {})
        if requested_by is not None:
            stored_diagnostics["requested_by"] = dict(requested_by)
        plan_json = _json(experiment.to_dict())
        with closing(self.runs_store._connect()) as db, db:
            existing = db.execute("SELECT plan_json, diagnostics_json FROM experiments WHERE id = ?",
                                  (experiment.experiment_id,)).fetchone()
            if existing is not None:
                if existing["plan_json"] != plan_json:
                    raise ExperimentPersistenceError("experiment ID already exists with a different plan")
                return experiment.experiment_id
            db.execute(
                "INSERT INTO experiments(id, run_id, base_execution_fingerprint, evaluator_kind, state, created_ts, updated_ts, plan_json, diagnostics_json) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (experiment.experiment_id, experiment.base.run_id, experiment.base.execution_fingerprint,
                 experiment.evaluator.to_dict()["kind"], "pending", stamp, stamp, plan_json, _json(stored_diagnostics)),
            )
            control_condition = condition_for_intervention(None)
            db.execute(
                "INSERT INTO experiment_arms(experiment_id, arm_id, ordinal, is_control, intervention_json, condition_json, state, observation_id, error_json, diagnostics_json) "
                "VALUES (?, 'control', -1, 1, NULL, ?, 'pending', NULL, NULL, ?)",
                (experiment.experiment_id, _json(control_condition), _json({})),
            )
            for ordinal, arm in enumerate(experiment.arms):
                condition = condition_for_intervention(arm.intervention)
                db.execute(
                    "INSERT INTO experiment_arms(experiment_id, arm_id, ordinal, is_control, intervention_json, condition_json, state, observation_id, error_json, diagnostics_json) "
                    "VALUES (?, ?, ?, 0, ?, ?, 'pending', NULL, NULL, ?)",
                    (experiment.experiment_id, arm.arm_id, ordinal,
                     _json(arm.intervention.to_dict() if arm.intervention is not None else None),
                     _json(condition), _json({})),
                )
        return experiment.experiment_id

    def _experiment_row(self, experiment_id: str):
        self._ensure()
        with closing(self.runs_store._connect()) as db:
            row = db.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,)).fetchone()
        if row is None:
            raise ExperimentPersistenceError(f"experiment {experiment_id!r} was not found")
        return row

    def _load_observation_row(self, row) -> Observation:
        if row is None:
            raise ObservationNotFound("observation row was not found")
        artifact_ref = _loads(row["artifact_ref_json"], {})
        artifact = self.runs_store._load_blob(artifact_ref, kind="experiment observation")
        if not isinstance(artifact, Mapping) or artifact.get("unavailable"):
            raise ObservationNotFound(f"observation artifact {row['id']!r} is unavailable")
        if artifact.get("observation_id") != row["id"] or artifact.get("observation_key_sha256") != row["observation_key_sha256"]:
            raise ObservationIntegrityError("observation artifact identity disagrees with its SQLite row")
        try:
            observation = observation_from_dict(artifact)
        except ObservationError as exc:
            raise ObservationIntegrityError("observation artifact failed validation") from exc
        if observation.observation_id != row["id"] or observation.observation_key_sha256 != row["observation_key_sha256"]:
            raise ObservationIntegrityError("observation artifact identity failed round-trip validation")
        return observation

    def get_observation(self, observation_id: str) -> Observation:
        self._ensure()
        with closing(self.runs_store._connect()) as db:
            row = db.execute("SELECT * FROM observations WHERE id = ?", (observation_id,)).fetchone()
        if row is None:
            raise ObservationNotFound(observation_id)
        return self._load_observation_row(row)

    def find_observation(self, observation_key_sha256: str) -> Observation | None:
        self._ensure()
        with closing(self.runs_store._connect()) as db:
            row = db.execute("SELECT * FROM observations WHERE observation_key_sha256 = ?",
                             (observation_key_sha256,)).fetchone()
        return self._load_observation_row(row) if row is not None else None

    def persist_observation(self, observation: Observation) -> str:
        if not isinstance(observation, Observation):
            raise TypeError("persist_observation requires an Observation")
        if not observation.completed:
            raise ObservationPersistenceError(
                "unavailable or failed evidence is transient and cannot occupy a reusable observation identity"
            )
        self._ensure()
        artifact_ref = self.runs_store._store_blob(observation.to_dict(), kind="experiment observation")
        if artifact_ref.get("write_failed"):
            raise ObservationPersistenceError("observation artifact could not be written")
        summary = {
            "observation_id": observation.observation_id,
            "status": observation.status,
            "evaluator_kind": observation.evaluator_kind,
            "condition_kind": observation.condition.get("kind"),
            "trusted": observation.trusted,
        }
        if isinstance(observation, TokenScoreObservation):
            summary.update({
                "token_count": len(observation.recorded_token_ids),
                "total_continuation_logprob": observation.total_continuation_logprob,
            })
        stamp = time.time()
        with closing(self.runs_store._connect()) as db, db:
            existing = db.execute("SELECT * FROM observations WHERE observation_key_sha256 = ?",
                                  (observation.observation_key_sha256,)).fetchone()
            if existing is not None:
                if existing["id"] != observation.observation_id:
                    raise ObservationIntegrityError("observation key maps to a different observation ID")
                prior = self._load_observation_row(existing)
                if prior.to_json() != observation.to_json():
                    raise ObservationIntegrityError("conflicting evidence under an existing observation key")
                return prior.observation_id
            try:
                db.execute(
                    "INSERT INTO observations(id, observation_key_sha256, run_id, base_execution_fingerprint, evaluator_kind, condition_kind, status, created_ts, artifact_ref_json, summary_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (observation.observation_id, observation.observation_key_sha256, observation.run_id,
                     observation.base_execution_fingerprint, observation.evaluator_kind,
                     str(observation.condition.get("kind") or ""), observation.status, stamp,
                     _json(artifact_ref), _json(summary)),
                )
            except Exception as exc:
                existing = db.execute("SELECT * FROM observations WHERE observation_key_sha256 = ?",
                                      (observation.observation_key_sha256,)).fetchone()
                if existing is None:
                    raise ObservationIntegrityError("observation insert failed") from exc
                prior = self._load_observation_row(existing)
                if prior.to_json() != observation.to_json():
                    raise ObservationIntegrityError("concurrent observation insert produced conflicting evidence") from exc
                return prior.observation_id
        return observation.observation_id

    def _arm_row(self, experiment_id: str, arm_id: str):
        self._ensure()
        with closing(self.runs_store._connect()) as db:
            row = db.execute("SELECT * FROM experiment_arms WHERE experiment_id = ? AND arm_id = ?",
                             (experiment_id, arm_id)).fetchone()
        if row is None:
            raise ExperimentPersistenceError(f"experiment arm {arm_id!r} was not found")
        return row

    def update_arm(self, experiment_id: str, arm_id: str, *, state: str,
                   observation_id: str | None | object = _UNSET, error: Mapping[str, Any] | None = None,
                   diagnostics: Mapping[str, Any] | None = None) -> None:
        if state not in ARM_STATES:
            raise ExperimentPersistenceError(f"unsupported arm state: {state!r}")
        row = self._arm_row(experiment_id, arm_id)
        if observation_id is not _UNSET and observation_id is not None:
            observation = self.get_observation(observation_id)
            condition = _loads(row["condition_json"], {})
            if observation.condition != condition:
                raise ObservationIntegrityError("associated observation condition does not match the experiment arm")
        stored_observation_id = row["observation_id"] if observation_id is _UNSET else observation_id
        with closing(self.runs_store._connect()) as db, db:
            db.execute(
                "UPDATE experiment_arms SET state = ?, observation_id = ?, error_json = ?, diagnostics_json = ? WHERE experiment_id = ? AND arm_id = ?",
                (state, stored_observation_id, _json(dict(error or {})) if error else None,
                 _json(dict(diagnostics or {})), experiment_id, arm_id),
            )
            db.execute("UPDATE experiments SET updated_ts = ? WHERE id = ?", (time.time(), experiment_id))

    def set_experiment_state(self, experiment_id: str, state: str, *, diagnostics: Mapping[str, Any] | None = None) -> None:
        if state not in EXPERIMENT_STATES:
            raise ExperimentPersistenceError(f"unsupported experiment state: {state!r}")
        self._experiment_row(experiment_id)
        with closing(self.runs_store._connect()) as db, db:
            if diagnostics is None:
                db.execute("UPDATE experiments SET state = ?, updated_ts = ? WHERE id = ?",
                           (state, time.time(), experiment_id))
            else:
                db.execute("UPDATE experiments SET state = ?, updated_ts = ?, diagnostics_json = ? WHERE id = ?",
                           (state, time.time(), _json(dict(diagnostics)), experiment_id))

    def associate_observation(self, experiment_id: str, arm_id: str, observation: Observation) -> str:
        observation_id = self.persist_observation(observation)
        self.update_arm(experiment_id, arm_id, state="completed", observation_id=observation_id)
        return observation_id

    def get_experiment(self, experiment_id: str) -> ExperimentView:
        row = self._experiment_row(experiment_id)
        plan = _loads(row["plan_json"], {})
        try:
            experiment = Experiment.from_dict(plan)
        except Exception as exc:
            raise ExperimentPersistenceError("stored experiment plan failed validation") from exc
        diagnostics = _loads(row["diagnostics_json"], {})
        with closing(self.runs_store._connect()) as db:
            arm_rows = db.execute("SELECT * FROM experiment_arms WHERE experiment_id = ? ORDER BY ordinal ASC",
                                  (experiment_id,)).fetchall()
        expected = {"control": (None, -1)} | {arm.arm_id: (arm.intervention, index) for index, arm in enumerate(experiment.arms)}
        views: list[ExperimentArmView] = []
        control = None
        control_seen = False
        for arm_row in arm_rows:
            arm_id = arm_row["arm_id"]
            if arm_id not in expected:
                raise ExperimentPersistenceError("stored experiment arm is not present in its plan")
            intervention, _ordinal = expected[arm_id]
            if int(arm_row["ordinal"]) != _ordinal:
                raise ExperimentPersistenceError("stored experiment arm ordinal disagrees with its plan")
            if bool(arm_row["is_control"]) != (arm_id == "control"):
                raise ExperimentPersistenceError("stored experiment control marker disagrees with its plan")
            stored_intervention = _loads(arm_row["intervention_json"], None)
            expected_intervention = intervention.to_dict() if intervention is not None else None
            if stored_intervention != expected_intervention:
                raise ExperimentPersistenceError("stored experiment arm intervention disagrees with its plan")
            expected_condition = condition_for_intervention(intervention)
            stored_condition = _loads(arm_row["condition_json"], {})
            if stored_condition != expected_condition:
                raise ObservationIntegrityError("stored experiment arm condition disagrees with its plan")
            observation = self.get_observation(arm_row["observation_id"]) if arm_row["observation_id"] else None
            if observation is not None:
                if observation.base_execution_fingerprint != experiment.base.execution_fingerprint:
                    raise ObservationIntegrityError("stored observation belongs to another base execution")
                if observation.evaluator_kind != experiment.evaluator.to_dict().get("kind"):
                    raise ObservationIntegrityError("stored observation belongs to another evaluator")
                if observation.condition != stored_condition:
                    raise ObservationIntegrityError("stored observation condition disagrees with its arm")
            view = ExperimentArmView(
                experiment_id=experiment_id, arm_id=arm_id, ordinal=int(arm_row["ordinal"]),
                intervention=intervention, condition=_loads(arm_row["condition_json"], {}),
                state=arm_row["state"], observation_id=arm_row["observation_id"], observation=observation,
                error=_loads(arm_row["error_json"], {}), diagnostics=_loads(arm_row["diagnostics_json"], {}),
            )
            views.append(view)
            if view.is_control:
                control_seen = True
                control = observation
        if not control_seen or len([view for view in views if not view.is_control]) != len(experiment.arms):
            raise ExperimentPersistenceError("stored experiment arm rows do not match its plan")
        return ExperimentView(
            experiment_id=experiment_id, base=experiment.base, evaluator=experiment.evaluator,
            control=control, arm_rows=tuple(view for view in views if not view.is_control),
            state=row["state"], diagnostics=diagnostics,
            execution_provenance={"store": "clozn.experiments.persistence", "durable": True},
            created_ts=float(row["created_ts"]), updated_ts=float(row["updated_ts"]),
            requested_by=diagnostics.get("requested_by") if isinstance(diagnostics, Mapping) else None,
        )

    def list_experiments(self, *, run_id: str | None = None, limit: int = 100) -> list[ExperimentView]:
        self._ensure()
        with closing(self.runs_store._connect()) as db:
            if run_id is None:
                rows = db.execute("SELECT id FROM experiments ORDER BY created_ts DESC, id DESC LIMIT ?", (max(0, int(limit)),)).fetchall()
            else:
                rows = db.execute("SELECT id FROM experiments WHERE run_id = ? ORDER BY created_ts DESC, id DESC LIMIT ?",
                                  (run_id, max(0, int(limit)))).fetchall()
        return [self.get_experiment(row["id"]) for row in rows]


__all__ = [
    "ARM_STATES", "EXPERIMENT_STATES", "EXPERIMENT_STORE_SCHEMA_VERSION",
    "ExperimentArmView", "ExperimentPersistenceError", "ExperimentView",
    "ObservationNotFound", "ObservationPersistenceError", "ObservationStore",
]
