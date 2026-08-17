"""Immutable, model-free bindings for the execution an experiment observes.

The experimental kernel keeps references and digests here rather than copying a
whole run into every arm.  A runner/adapter reloads the run by ``run_id`` and
must prove that the reloaded execution still has this binding before doing
work.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from types import MappingProxyType
from typing import Any

from clozn.receipts.rederive import with_arm_conditions
from clozn.replay.execution_fork import parent_execution_fingerprint
from clozn.runs.answer_preservation import (
    generation_contract_from_run,
    _trace_token_pieces,
)
from clozn.runs.context_receipt import read_receipt


SCHEMA_VERSION = "clozn.experiment-execution-state.v1"


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON-shaped values for dataclass fields."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, frozenset)):
        return [_thaw(item) for item in value]
    return value


def canonical_json(value: Any) -> str:
    """Canonical JSON used by all new-kernel identities."""
    return json.dumps(
        _thaw(value), ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _runtime_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    identity = run.get("identity")
    return {
        "model": run.get("model"),
        "substrate": run.get("substrate"),
        "identity": deepcopy(dict(identity)) if isinstance(identity, Mapping) else {},
    }


def _receipt_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    raw = run.get("context_receipt")
    if not isinstance(raw, Mapping):
        return {"state": "unavailable", "digest": None, "source_ids": []}
    source_ids: set[str] = set()
    delivered = raw.get("delivered")
    if isinstance(delivered, list):
        for segment in delivered:
            if not isinstance(segment, Mapping):
                continue
            segment_id = segment.get("segment_id")
            if isinstance(segment_id, str) and segment_id:
                source_ids.add(segment_id)
            sources = segment.get("sources")
            if isinstance(sources, list):
                for source in sources:
                    source_id = source.get("source_id") if isinstance(source, Mapping) else None
                    if isinstance(source_id, str) and source_id:
                        source_ids.add(source_id)
    try:
        view = read_receipt(dict(run))
        state = "current" if view.get("shape") == "new" else "legacy"
    except Exception:
        state = "malformed"
    return {
        "state": state,
        "schema": raw.get("schema_version", raw.get("schema")),
        "run_id": raw.get("run_id"),
        "digest": digest(raw),
        "source_ids": sorted(source_ids),
    }


def _answer_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    ids, reason = _trace_token_pieces(run)
    response = run.get("response")
    result: dict[str, Any] = {
        "response_sha256": digest(response if isinstance(response, str) else ""),
        "token_count": len(ids) if ids is not None else None,
        "token_ids_sha256": digest(ids) if ids is not None else None,
    }
    if reason:
        result["unavailable_reason"] = reason
    return result


class ExecutionState:
    """Frozen binding to one recorded execution and its exact-answer evidence."""

    __slots__ = (
        "run_id", "model_runtime_identity", "generation_contract", "output_contract",
        "context_receipt_identity", "recorded_answer_token_identity",
        "execution_conditions_digest", "execution_fingerprint",
        "generation_contract_reason", "_sealed",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("ExecutionState is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, run_id: str, model_runtime_identity: Mapping[str, Any],
                 generation_contract: Mapping[str, Any] | None,
                 output_contract: Mapping[str, Any] | None,
                 context_receipt_identity: Mapping[str, Any],
                 recorded_answer_token_identity: Mapping[str, Any],
                 execution_conditions_digest: str,
                 execution_fingerprint: str,
                 generation_contract_reason: str | None = None):
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("ExecutionState.run_id must be a non-empty string")
        for name, value in (("execution_conditions_digest", execution_conditions_digest),
                            ("execution_fingerprint", execution_fingerprint)):
            if not isinstance(value, str) or not value:
                raise ValueError(f"ExecutionState.{name} must be a non-empty digest")
        self.run_id = run_id
        self.model_runtime_identity = _freeze(dict(model_runtime_identity))
        self.generation_contract = _freeze(dict(generation_contract)) if generation_contract else None
        self.output_contract = _freeze(dict(output_contract)) if output_contract else None
        self.context_receipt_identity = _freeze(dict(context_receipt_identity))
        self.recorded_answer_token_identity = _freeze(dict(recorded_answer_token_identity))
        self.execution_conditions_digest = execution_conditions_digest
        self.execution_fingerprint = execution_fingerprint
        self.generation_contract_reason = generation_contract_reason
        self._sealed = True

    @classmethod
    def from_run(cls, run: Mapping[str, Any]) -> "ExecutionState":
        if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
            raise ValueError("ExecutionState.from_run requires a stored run with a non-empty id")
        contract, contract_reason = generation_contract_from_run(run)
        conditions = with_arm_conditions(dict(run))
        condition_projection = {
            "messages": conditions.get("messages"),
            "block": conditions.get("block"),
            "steer_strengths": conditions.get("steer_strengths"),
            "continuation_ids": conditions.get("continuation_ids"),
        }
        receipt = _receipt_identity(run)
        answer = _answer_identity(run)
        try:
            legacy_fingerprint = parent_execution_fingerprint(run)
        except Exception as exc:
            raise ValueError(f"run execution fingerprint is unavailable: {exc}") from exc
        execution_fingerprint = digest({
            "legacy_execution_fingerprint": legacy_fingerprint,
            "run_id": run["id"],
            "model_runtime_identity": _runtime_identity(run),
            "generation_contract": contract,
            "generation_contract_reason": contract_reason,
            "output_contract": run.get("output_contract") if isinstance(run.get("output_contract"), Mapping) else {},
            "context_receipt_digest": receipt.get("digest"),
            "conditions_digest": digest(condition_projection),
            "answer_identity": answer,
        })
        return cls(
            run_id=run["id"],
            model_runtime_identity=_runtime_identity(run),
            generation_contract=contract,
            output_contract=run.get("output_contract") if isinstance(run.get("output_contract"), Mapping) else None,
            context_receipt_identity=receipt,
            recorded_answer_token_identity=answer,
            execution_conditions_digest=digest(condition_projection),
            execution_fingerprint=execution_fingerprint,
            generation_contract_reason=contract_reason,
        )

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "model_runtime_identity": _thaw(self.model_runtime_identity),
            "generation_contract": _thaw(self.generation_contract) if self.generation_contract else None,
            "output_contract": _thaw(self.output_contract) if self.output_contract else None,
            "context_receipt_identity": _thaw(self.context_receipt_identity),
            "recorded_answer_token_identity": _thaw(self.recorded_answer_token_identity),
            "execution_conditions_digest": self.execution_conditions_digest,
            "execution_fingerprint": self.execution_fingerprint,
        }
        if self.generation_contract_reason:
            result["generation_contract_reason"] = self.generation_contract_reason
        return result

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionState":
        if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"ExecutionState must declare {SCHEMA_VERSION}")
        return cls(
            run_id=value.get("run_id"),
            model_runtime_identity=value.get("model_runtime_identity") or {},
            generation_contract=value.get("generation_contract"),
            output_contract=value.get("output_contract"),
            context_receipt_identity=value.get("context_receipt_identity") or {},
            recorded_answer_token_identity=value.get("recorded_answer_token_identity") or {},
            execution_conditions_digest=value.get("execution_conditions_digest"),
            execution_fingerprint=value.get("execution_fingerprint"),
            generation_contract_reason=value.get("generation_contract_reason"),
        )

    @property
    def runtime_identity(self):
        return self.model_runtime_identity

    @property
    def recorded_answer_identity(self):
        return self.recorded_answer_token_identity

    @property
    def context_receipt_digest(self):
        return self.context_receipt_identity.get("digest")


__all__ = ["ExecutionState", "SCHEMA_VERSION", "canonical_json", "digest"]
