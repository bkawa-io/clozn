"""Standalone direct evidence and deterministic counterfactual identities."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from typing import Any

from .interventions import DeleteSource
from .state import canonical_json, digest


SCHEMA_VERSION = "clozn.experiment-observation.v2"
TOKEN_SCORE_SCHEMA_VERSION = "clozn.experiment-token-score-observation.v2"
GENERATED_OBSERVATION_SCHEMA_VERSION = "clozn.experiment-generated-observation.v2"
OBSERVATION_STATUSES = frozenset({"completed", "exact_preserved", "diverged", "unavailable", "failed"})
TOKEN_SCORE_STATUSES = frozenset({"completed", "unavailable", "failed"})


class ObservationError(ValueError):
    """A malformed or internally inconsistent direct observation."""


class ObservationIntegrityError(ObservationError):
    """The same deterministic observation identity carries conflicting evidence."""


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return deepcopy(dict(value)) if isinstance(value, Mapping) else {}


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ObservationError(f"{name} must contain finite numbers")
    return float(value)


def _span_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        start, end = value.get("start"), value.get("end")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value
    else:
        raise ObservationError("token_spans must contain start/end pairs")
    if (
        isinstance(start, bool) or isinstance(end, bool)
        or not isinstance(start, int) or not isinstance(end, int)
        or start < 0 or end < start
    ):
        raise ObservationError("token_spans must contain non-negative half-open ranges")
    return int(start), int(end)


def condition_for_intervention(intervention: Any = None) -> dict[str, Any]:
    """Canonical condition identity; unchanged control is explicit, never a fake arm."""
    if intervention is None:
        return {"kind": "unchanged_condition"}
    to_dict = getattr(intervention, "to_dict", None)
    if not callable(to_dict):
        raise ObservationError("intervention must expose to_dict()")
    value = to_dict()
    if not isinstance(value, Mapping):
        raise ObservationError("intervention identity must be an object")
    return {"kind": "intervention", "intervention": dict(value)}


def observation_identity(*, base_execution_fingerprint: str, evaluator: Any,
                         condition: Mapping[str, Any], contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return the content identity shared by all experiments requesting one measurement."""
    if not isinstance(base_execution_fingerprint, str) or not base_execution_fingerprint:
        raise ObservationError("base execution fingerprint is required")
    evaluator_value = evaluator.to_dict() if hasattr(evaluator, "to_dict") else evaluator
    if not isinstance(evaluator_value, Mapping):
        raise ObservationError("evaluator identity must be an object")
    payload = {
        "base_execution_fingerprint": base_execution_fingerprint,
        "condition": deepcopy(dict(condition)),
        "evaluator": deepcopy(dict(evaluator_value)),
        "contract": deepcopy(dict(contract)),
    }
    key_sha256 = digest(payload)
    return {
        "observation_id": "obs_" + key_sha256[:24],
        "observation_key_sha256": key_sha256,
        "observation_key": payload,
    }


def execution_observation_identity(state: Any, evaluator: Any, intervention: Any = None) -> dict[str, Any]:
    """Build identity without experiment/arm/UI fields.

    A resolved state adds the logical boundary and realization regime to the
    condition identity.  This keeps two addresses or two fidelity regimes from
    sharing one reusable observation key.
    """
    state_dict = state.to_dict()
    execution = getattr(state, "execution", state)
    execution_dict = execution.to_dict()
    contract = {
        "execution_conditions_digest": execution.execution_conditions_digest,
        "generation_contract": execution_dict.get("generation_contract"),
        "output_contract": execution_dict.get("output_contract"),
        "runtime_identity": execution_dict.get("model_runtime_identity"),
        "recorded_answer_token_identity": execution_dict.get("recorded_answer_token_identity"),
    }
    base_execution_fingerprint = execution.execution_fingerprint
    if hasattr(state, "state_ref"):
        contract["state_ref"] = state.state_ref.identity_payload()
        contract["state_realization"] = {
            "classification": state.classification,
            "realization": deepcopy(state.realization),
            "realization_fingerprint": state.realization_fingerprint,
        }
    return observation_identity(
        base_execution_fingerprint=base_execution_fingerprint,
        evaluator=evaluator,
        condition=condition_for_intervention(intervention),
        contract=contract,
    )


class Observation:
    """One standalone exact-reference observation, independent of any arm row."""

    __slots__ = (
        "observation_id", "observation_key_sha256", "observation_key", "run_id",
        "base_execution_fingerprint", "evaluator", "condition", "contract",
        "status", "matched_token_count", "first_divergence_index", "divergence_kind",
        "execution_provenance", "proof_grade", "trusted", "diagnostics", "_sealed",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Observation is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, status: str, observation_id: str | None = None,
                 observation_key_sha256: str | None = None,
                 observation_key: Mapping[str, Any] | None = None,
                 run_id: str | None = None,
                 base_execution_fingerprint: str | None = None,
                 evaluator: Mapping[str, Any] | Any | None = None,
                 condition: Mapping[str, Any] | None = None,
                 contract: Mapping[str, Any] | None = None,
                 matched_token_count: int | None = None,
                 first_divergence_index: int | None = None,
                 divergence_kind: str | None = None,
                 execution_provenance: Mapping[str, Any] | None = None,
                 proof_grade: str = "unavailable", trusted: bool = False,
                 diagnostics: Mapping[str, Any] | None = None, _seal: bool = True):
        identity = deepcopy(dict(observation_key or {}))
        if base_execution_fingerprint is None:
            base_execution_fingerprint = identity.get("base_execution_fingerprint")
        if condition is None:
            condition = identity.get("condition")
        if evaluator is None:
            evaluator = identity.get("evaluator")
        if contract is None:
            contract = identity.get("contract")
        evaluator_value = evaluator.to_dict() if hasattr(evaluator, "to_dict") else evaluator
        if not isinstance(evaluator_value, Mapping):
            raise ObservationError("Observation.evaluator must be an identity object")
        if not isinstance(condition, Mapping) or not isinstance(contract, Mapping):
            raise ObservationError("Observation condition and contract identities are required")
        if not isinstance(run_id, str) or not run_id:
            raise ObservationError("Observation.run_id must be a non-empty string")
        if not isinstance(base_execution_fingerprint, str) or not base_execution_fingerprint:
            raise ObservationError("Observation.base_execution_fingerprint is required")
        calculated = observation_identity(
            base_execution_fingerprint=base_execution_fingerprint,
            evaluator=evaluator_value, condition=condition, contract=contract,
        )
        if observation_id is not None and observation_id != calculated["observation_id"]:
            raise ObservationIntegrityError("observation_id does not match its deterministic identity")
        if observation_key_sha256 is not None and observation_key_sha256 != calculated["observation_key_sha256"]:
            raise ObservationIntegrityError("observation key does not match its deterministic identity")
        if observation_key and dict(observation_key) != calculated["observation_key"]:
            raise ObservationIntegrityError("observation key payload does not match its identity")
        if status not in OBSERVATION_STATUSES:
            raise ObservationError(f"unsupported observation status: {status!r}")
        if matched_token_count is not None and (
            isinstance(matched_token_count, bool) or not isinstance(matched_token_count, int)
            or matched_token_count < 0
        ):
            raise ObservationError("matched_token_count must be a non-negative integer or None")
        if first_divergence_index is not None and (
            isinstance(first_divergence_index, bool) or not isinstance(first_divergence_index, int)
            or first_divergence_index < 0
        ):
            raise ObservationError("first_divergence_index must be a non-negative integer or None")
        if not isinstance(proof_grade, str) or not proof_grade:
            raise ObservationError("proof_grade must be a non-empty string")
        if not isinstance(trusted, bool):
            raise ObservationError("trusted must be a boolean")
        self.observation_id = observation_id or calculated["observation_id"]
        self.observation_key_sha256 = observation_key_sha256 or calculated["observation_key_sha256"]
        self.observation_key = deepcopy(calculated["observation_key"])
        self.run_id = run_id
        self.base_execution_fingerprint = base_execution_fingerprint
        self.evaluator = deepcopy(dict(evaluator_value))
        self.condition = deepcopy(dict(condition))
        self.contract = deepcopy(dict(contract))
        self.status = status
        self.matched_token_count = matched_token_count
        self.first_divergence_index = first_divergence_index
        self.divergence_kind = divergence_kind
        self.execution_provenance = _copy_mapping(execution_provenance)
        self.proof_grade = proof_grade
        self.trusted = trusted
        self.diagnostics = _copy_mapping(diagnostics)
        self._sealed = _seal

    @property
    def evaluator_kind(self) -> str:
        return str(self.evaluator.get("kind") or "")

    @property
    def preservation_status(self) -> str:
        return self.status

    @property
    def completed(self) -> bool:
        return self.status in {"completed", "exact_preserved", "diverged"}

    def _base_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "observation_id": self.observation_id,
            "observation_key_sha256": self.observation_key_sha256,
            "observation_key": deepcopy(self.observation_key),
            "run_id": self.run_id,
            "base_execution_fingerprint": self.base_execution_fingerprint,
            "evaluator": deepcopy(self.evaluator),
            "condition": deepcopy(self.condition),
            "contract": deepcopy(self.contract),
            "status": self.status,
            "execution_provenance": deepcopy(self.execution_provenance),
            "proof_grade": self.proof_grade,
            "trusted": self.trusted,
            "diagnostics": deepcopy(self.diagnostics),
        }

    def to_dict(self) -> dict[str, Any]:
        value = self._base_dict()
        value.update({
            "matched_token_count": self.matched_token_count,
            "first_divergence_index": self.first_divergence_index,
            "divergence_kind": self.divergence_kind,
        })
        return value

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def evidence_digest(self) -> str:
        return digest(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Observation":
        if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
            raise ObservationError(f"Observation must declare {SCHEMA_VERSION}")
        return cls(
            observation_id=value.get("observation_id"), observation_key_sha256=value.get("observation_key_sha256"),
            observation_key=value.get("observation_key"), run_id=value.get("run_id"),
            base_execution_fingerprint=value.get("base_execution_fingerprint"), evaluator=value.get("evaluator"),
            condition=value.get("condition"), contract=value.get("contract"), status=value.get("status"),
            matched_token_count=value.get("matched_token_count"), first_divergence_index=value.get("first_divergence_index"),
            divergence_kind=value.get("divergence_kind"), execution_provenance=value.get("execution_provenance"),
            proof_grade=value.get("proof_grade", "unavailable"), trusted=value.get("trusted", False),
            diagnostics=value.get("diagnostics"),
        )


class TokenScoreObservation(Observation):
    """One standalone full teacher-forced score vector."""

    __slots__ = (
        "recorded_token_ids", "token_pieces", "token_spans", "token_logprobs",
        "total_continuation_logprob", "evaluator_provenance", "score_basis",
    )

    def __init__(self, *, status: str, recorded_token_ids=(), token_pieces=(), token_spans=(),
                 token_logprobs=(), total_continuation_logprob: float | None = None,
                 evaluator_provenance: Mapping[str, Any] | None = None,
                 score_basis: Mapping[str, Any] | None = None, **kwargs):
        ids = tuple(recorded_token_ids)
        pieces = tuple(token_pieces)
        spans = tuple(_span_pair(item) for item in token_spans)
        logprobs = tuple(_finite_float(item, name="token_logprobs") for item in token_logprobs)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ids):
            raise ObservationError("recorded_token_ids must contain non-negative integers")
        if any(not isinstance(item, str) for item in pieces):
            raise ObservationError("token_pieces must contain strings")
        if not (len(ids) == len(pieces) == len(spans) == len(logprobs)):
            if status == "completed":
                raise ObservationError("completed token-score fields must have equal lengths")
            ids = pieces = spans = logprobs = ()
        if status == "completed" and total_continuation_logprob is None:
            raise ObservationError("completed token-score observations require a total logprob")
        total = None if total_continuation_logprob is None else _finite_float(
            total_continuation_logprob, name="total_continuation_logprob")
        super().__init__(status=status, _seal=False, **kwargs)
        self.recorded_token_ids = ids
        self.token_pieces = pieces
        self.token_spans = spans
        self.token_logprobs = logprobs
        self.total_continuation_logprob = total
        self.evaluator_provenance = _copy_mapping(evaluator_provenance)
        self.score_basis = _copy_mapping(score_basis)
        self._sealed = True

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def runtime_binding(self) -> Any:
        return self.score_basis.get("runtime_binding")

    def to_dict(self) -> dict[str, Any]:
        value = self._base_dict()
        value["schema_version"] = TOKEN_SCORE_SCHEMA_VERSION
        value.update({
            "recorded_token_ids": list(self.recorded_token_ids), "token_pieces": list(self.token_pieces),
            "token_spans": [{"start": start, "end": end} for start, end in self.token_spans],
            "token_logprobs": list(self.token_logprobs),
            "total_continuation_logprob": self.total_continuation_logprob,
            "evaluator_provenance": dict(self.evaluator_provenance), "score_basis": dict(self.score_basis),
        })
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TokenScoreObservation":
        if not isinstance(value, Mapping) or value.get("schema_version") != TOKEN_SCORE_SCHEMA_VERSION:
            raise ObservationError(f"TokenScoreObservation must declare {TOKEN_SCORE_SCHEMA_VERSION}")
        return cls(
            status=value.get("status"), recorded_token_ids=value.get("recorded_token_ids") or (),
            token_pieces=value.get("token_pieces") or (), token_spans=value.get("token_spans") or (),
            token_logprobs=value.get("token_logprobs") or (), total_continuation_logprob=value.get("total_continuation_logprob"),
            evaluator_provenance=value.get("evaluator_provenance"), score_basis=value.get("score_basis"),
            observation_id=value.get("observation_id"), observation_key_sha256=value.get("observation_key_sha256"),
            observation_key=value.get("observation_key"), run_id=value.get("run_id"),
            base_execution_fingerprint=value.get("base_execution_fingerprint"), evaluator=value.get("evaluator"),
            condition=value.get("condition"), contract=value.get("contract"),
            execution_provenance=value.get("execution_provenance"), proof_grade=value.get("proof_grade", "unavailable"),
            trusted=value.get("trusted", False), diagnostics=value.get("diagnostics"),
        )


class GeneratedObservation(Observation):
    """Standalone evidence for a continuation already generated by an arm."""

    __slots__ = (
        "state_ref", "realization", "fidelity", "intervention", "generated_suffix_text",
        "generated_token_ids", "generated_steps", "finish_reason", "generation_contract",
        "runtime_provenance", "exact_control_proof", "input_snapshot",
    )

    def __init__(self, *, state_ref: Any = None, realization: Mapping[str, Any], fidelity: Mapping[str, Any],
                 intervention: Any, generated_suffix_text: str, generated_token_ids=None,
                 generated_steps=None, finish_reason: str | None = None,
                 generation_contract: Mapping[str, Any] | None = None,
                 runtime_provenance: Mapping[str, Any] | None = None,
                 exact_control_proof: Mapping[str, Any] | None = None,
                 input_snapshot: Mapping[str, Any] | None = None, **kwargs):
        from .state_ref import StateRef
        if state_ref is not None and not isinstance(state_ref, StateRef):
            raise ObservationError("GeneratedObservation.state_ref must be a StateRef or None")
        if state_ref is None:
            evaluator_value = kwargs.get("evaluator")
            if not isinstance(evaluator_value, Mapping) or evaluator_value.get("kind") != "generate":
                raise ObservationError("StateRef-free GeneratedObservation must be a Generate observation")
            if not isinstance(intervention, DeleteSource):
                raise ObservationError("StateRef-free GeneratedObservation requires DeleteSource")
        if not isinstance(realization, Mapping) or not isinstance(fidelity, Mapping):
            raise ObservationError("GeneratedObservation realization and fidelity are required")
        if input_snapshot is not None and not isinstance(input_snapshot, Mapping):
            raise ObservationError("GeneratedObservation.input_snapshot must be an object when supplied")
        if not isinstance(generated_suffix_text, str):
            raise ObservationError("generated_suffix_text must be a string")
        ids = tuple(generated_token_ids or ())
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ids):
            raise ObservationError("generated_token_ids must contain non-negative integers")
        if generated_steps is not None and not isinstance(generated_steps, (list, tuple)):
            raise ObservationError("generated_steps must be a list when supplied")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise ObservationError("finish_reason must be a string when supplied")
        supplied_key = kwargs.get("observation_key")
        if isinstance(supplied_key, Mapping):
            supplied_contract = supplied_key.get("contract")
            if isinstance(supplied_contract, Mapping):
                if state_ref is not None and supplied_contract.get("state_ref") != state_ref.identity_payload():
                    raise ObservationIntegrityError("GeneratedObservation StateRef disagrees with its identity")
                realization_identity = supplied_contract.get("state_realization")
                if isinstance(realization_identity, Mapping) and realization_identity.get("realization") != dict(realization):
                    raise ObservationIntegrityError("GeneratedObservation realization disagrees with its identity")
        supplied_base = kwargs.get("base_execution_fingerprint")
        if state_ref is not None and supplied_base is not None and supplied_base != state_ref.execution_fingerprint:
            raise ObservationIntegrityError("GeneratedObservation base execution disagrees with its StateRef")
        supplied_condition = kwargs.get("condition")
        expected_condition = condition_for_intervention(intervention)
        if isinstance(supplied_condition, Mapping) and dict(supplied_condition) != expected_condition:
            raise ObservationIntegrityError("GeneratedObservation intervention disagrees with its identity")
        super().__init__(**kwargs, _seal=False)
        self.state_ref = state_ref
        self.realization = _copy_mapping(realization)
        self.fidelity = _copy_mapping(fidelity)
        self.intervention = deepcopy(intervention.to_dict() if hasattr(intervention, "to_dict") else intervention)
        self.generated_suffix_text = generated_suffix_text
        self.generated_token_ids = ids
        self.generated_steps = deepcopy(list(generated_steps)) if generated_steps is not None else None
        self.finish_reason = finish_reason
        self.generation_contract = _copy_mapping(generation_contract)
        self.runtime_provenance = _copy_mapping(runtime_provenance)
        self.exact_control_proof = _copy_mapping(exact_control_proof)
        self.input_snapshot = _copy_mapping(input_snapshot)
        self._sealed = True

    @property
    def fidelity_classification(self) -> str:
        return str(self.fidelity.get("classification") or self.realization.get("classification") or "unavailable")

    def to_dict(self) -> dict[str, Any]:
        value = self._base_dict()
        value["schema_version"] = GENERATED_OBSERVATION_SCHEMA_VERSION
        value.update({
            "state_ref": self.state_ref.to_dict() if self.state_ref is not None else None,
            "realization": deepcopy(self.realization), "fidelity": deepcopy(self.fidelity),
            "intervention": deepcopy(self.intervention), "generated_suffix_text": self.generated_suffix_text,
            "generated_token_ids": list(self.generated_token_ids),
            "generated_steps": deepcopy(self.generated_steps), "finish_reason": self.finish_reason,
            "generation_contract": deepcopy(self.generation_contract),
            "runtime_provenance": deepcopy(self.runtime_provenance),
            "exact_control_proof": deepcopy(self.exact_control_proof),
            "input_snapshot": deepcopy(self.input_snapshot),
        })
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "GeneratedObservation":
        if not isinstance(value, Mapping) or value.get("schema_version") != GENERATED_OBSERVATION_SCHEMA_VERSION:
            raise ObservationError(f"GeneratedObservation must declare {GENERATED_OBSERVATION_SCHEMA_VERSION}")
        from .interventions import intervention_from_dict
        from .state_ref import StateRef
        try:
            intervention = (intervention_from_dict(value.get("intervention"))
                            if value.get("intervention") is not None else None)
            state_ref = StateRef.from_dict(value.get("state_ref")) if value.get("state_ref") is not None else None
        except Exception as exc:
            raise ObservationError("GeneratedObservation contains malformed state or intervention") from exc
        return cls(
            state_ref=state_ref, realization=value.get("realization") or {}, fidelity=value.get("fidelity") or {},
            intervention=intervention, generated_suffix_text=value.get("generated_suffix_text", ""),
            generated_token_ids=value.get("generated_token_ids") or (), generated_steps=value.get("generated_steps"),
            finish_reason=value.get("finish_reason"), generation_contract=value.get("generation_contract"),
            runtime_provenance=value.get("runtime_provenance"), exact_control_proof=value.get("exact_control_proof"),
            input_snapshot=value.get("input_snapshot"),
            observation_id=value.get("observation_id"), observation_key_sha256=value.get("observation_key_sha256"),
            observation_key=value.get("observation_key"), run_id=value.get("run_id"),
            base_execution_fingerprint=value.get("base_execution_fingerprint"), evaluator=value.get("evaluator"),
            condition=value.get("condition"), contract=value.get("contract"), status=value.get("status"),
            execution_provenance=value.get("execution_provenance"), proof_grade=value.get("proof_grade", "unavailable"),
            trusted=value.get("trusted", False), diagnostics=value.get("diagnostics"),
        )


def observation_from_dict(value: Mapping[str, Any]) -> Observation:
    """Central immutable-evidence deserializer used by runner and persistence."""
    if not isinstance(value, Mapping):
        raise ObservationError("observation must be an object")
    schema = value.get("schema_version")
    if schema == TOKEN_SCORE_SCHEMA_VERSION:
        return TokenScoreObservation.from_dict(value)
    if schema == GENERATED_OBSERVATION_SCHEMA_VERSION:
        return GeneratedObservation.from_dict(value)
    if schema == SCHEMA_VERSION:
        return Observation.from_dict(value)
    raise ObservationError("unsupported observation schema")


class TokenScoreDelta:
    """Pure signed baseline-minus-intervention evidence derived from two observations."""

    __slots__ = (
        "observation_id", "status", "recorded_token_ids", "token_pieces", "token_spans",
        "baseline_logprobs", "intervened_logprobs", "deltas", "total_delta_nats",
        "baseline_total_logprob", "intervened_total_logprob", "provenance", "diagnostics",
    )

    def __init__(self, *, observation_id: str, status: str, recorded_token_ids=(), token_pieces=(), token_spans=(),
                 baseline_logprobs=(), intervened_logprobs=(), deltas=(), total_delta_nats=None,
                 baseline_total_logprob=None, intervened_total_logprob=None, provenance=None, diagnostics=None):
        if status not in TOKEN_SCORE_STATUSES:
            raise ObservationError(f"unsupported token-score delta status: {status!r}")
        self.observation_id = observation_id
        self.status = status
        self.recorded_token_ids = tuple(recorded_token_ids)
        self.token_pieces = tuple(token_pieces)
        self.token_spans = tuple(tuple(item) for item in token_spans)
        self.baseline_logprobs = tuple(baseline_logprobs)
        self.intervened_logprobs = tuple(intervened_logprobs)
        self.deltas = tuple(deltas)
        self.total_delta_nats = total_delta_nats
        self.baseline_total_logprob = baseline_total_logprob
        self.intervened_total_logprob = intervened_total_logprob
        self.provenance = _copy_mapping(provenance)
        self.diagnostics = _copy_mapping(diagnostics)

    @classmethod
    def from_observations(cls, baseline: TokenScoreObservation, intervention: TokenScoreObservation) -> "TokenScoreDelta":
        if not isinstance(baseline, TokenScoreObservation) or not isinstance(intervention, TokenScoreObservation):
            raise TypeError("TokenScoreDelta requires TokenScoreObservation objects")
        if not baseline.completed or not intervention.completed:
            status = "failed" if "failed" in {baseline.status, intervention.status} else "unavailable"
            return cls(
                observation_id=intervention.observation_id, status=status,
                provenance={"basis": "persisted_token_score_observations"},
                diagnostics={"baseline_status": baseline.status, "intervention_status": intervention.status},
            )
        aligned = (
            baseline.recorded_token_ids == intervention.recorded_token_ids
            and baseline.token_pieces == intervention.token_pieces
            and baseline.token_spans == intervention.token_spans
            and len(baseline.token_logprobs) == len(intervention.token_logprobs)
        )
        if not aligned:
            return cls(
                observation_id=intervention.observation_id, status="unavailable",
                provenance={"basis": "persisted_token_score_observations"},
                diagnostics={"reason": "token_alignment_mismatch"},
            )
        deltas = tuple(base - altered for base, altered in zip(baseline.token_logprobs, intervention.token_logprobs))
        return cls(
            observation_id=intervention.observation_id, status="completed",
            recorded_token_ids=baseline.recorded_token_ids, token_pieces=baseline.token_pieces,
            token_spans=baseline.token_spans, baseline_logprobs=baseline.token_logprobs,
            intervened_logprobs=intervention.token_logprobs, deltas=deltas,
            total_delta_nats=baseline.total_continuation_logprob - intervention.total_continuation_logprob,
            baseline_total_logprob=baseline.total_continuation_logprob,
            intervened_total_logprob=intervention.total_continuation_logprob,
            provenance={
                "basis": "persisted_token_score_observations", "sign": "baseline_minus_intervention",
                "baseline_observation_id": baseline.observation_id,
                "intervention_observation_id": intervention.observation_id,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "token_score_delta", "observation_id": self.observation_id, "status": self.status,
            "recorded_token_ids": list(self.recorded_token_ids), "token_pieces": list(self.token_pieces),
            "token_spans": [{"start": start, "end": end} for start, end in self.token_spans],
            "baseline_logprobs": list(self.baseline_logprobs), "intervened_logprobs": list(self.intervened_logprobs),
            "deltas": list(self.deltas), "total_delta_nats": self.total_delta_nats,
            "baseline_total_logprob": self.baseline_total_logprob,
            "intervened_total_logprob": self.intervened_total_logprob,
            "provenance": dict(self.provenance), "diagnostics": dict(self.diagnostics),
        }


__all__ = [
    "Observation", "ObservationError", "ObservationIntegrityError", "OBSERVATION_STATUSES",
    "GeneratedObservation", "GENERATED_OBSERVATION_SCHEMA_VERSION", "SCHEMA_VERSION", "TokenScoreDelta", "TokenScoreObservation", "TOKEN_SCORE_SCHEMA_VERSION",
    "TOKEN_SCORE_STATUSES", "condition_for_intervention", "execution_observation_identity",
    "observation_from_dict", "observation_identity",
]
