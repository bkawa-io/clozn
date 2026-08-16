"""Model-free orchestration for the experimental persistent exact-reference session.

The worker owns tokenization and KV.  This module owns only lifecycle binding, candidate identity,
scalar-confirmation gates, and deterministic telemetry.  It intentionally does not select a reducer
candidate: the reducer supplies the candidate rank/order and calls ``promote`` only after its own
trusted scalar evidence accepts the candidate.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any


class PersistentParentSessionError(ValueError):
    """Typed lifecycle, stale-state, cancellation, or parity failure."""

    def __init__(self, message: str, *, code: str = "persistent_parent_session_invalid", status: int = 409):
        super().__init__(message)
        self.code = code
        self.status = status


class PersistentParentParityError(PersistentParentSessionError):
    """Native and trusted scalar exact classifications disagree."""

    def __init__(self, mismatches: Sequence[Mapping[str, Any]]):
        self.mismatches = [deepcopy(dict(item)) for item in mismatches]
        super().__init__(
            "persistent native/scalar parity failure",
            code="persistent_parent_parity_failure",
            status=409,
        )


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def candidate_id(retained_ids: Sequence[Any]) -> str:
    """Stable caller-side identity; the worker never derives ranking from it."""
    return "ppc_" + _digest(list(retained_ids))[:24]


def evidence_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in (
        "status", "matched_token_count", "first_divergence_index", "expected_token_id",
        "actual_token_id", "divergence_kind", "termination_match",
    )}


def assert_scalar_parity(native_rows: Sequence[Mapping[str, Any]],
                         scalar_rows: Sequence[Mapping[str, Any]]) -> None:
    if len(native_rows) != len(scalar_rows):
        raise PersistentParentParityError([{
            "kind": "row_count", "native_count": len(native_rows), "scalar_count": len(scalar_rows),
        }])
    mismatches = []
    for index, (native, scalar) in enumerate(zip(native_rows, scalar_rows)):
        native_projection = evidence_projection(native)
        scalar_projection = evidence_projection(scalar)
        if native_projection != scalar_projection:
            mismatches.append({
                "arm_index": index,
                "native": native_projection,
                "scalar": scalar_projection,
            })
    if mismatches:
        raise PersistentParentParityError(mismatches)


@dataclass
class PersistentParentSessionClient:
    """Small typed client/state machine for one worker-local persistent session."""

    engine: Any
    reference_token_ids: tuple[int, ...]
    generation_contract: Mapping[str, Any]
    session_id: str | None = field(default=None, init=False)
    parent_version: int | None = field(default=None, init=False)
    parent_prompt_digest: str | None = field(default=None, init=False)
    runtime_identity: dict[str, Any] = field(default_factory=dict, init=False)
    telemetry: dict[str, Any] = field(default_factory=dict, init=False)
    closed: bool = field(default=False, init=False)
    _last_round: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def _require_open(self) -> None:
        if self.closed:
            raise PersistentParentSessionError(
                "persistent parent session is closed", code="session_closed", status=409,
            )

    def _require_created(self) -> None:
        self._require_open()
        if self.session_id is None or self.parent_version is None:
            raise PersistentParentSessionError(
                "persistent parent session has not been created", code="session_not_created", status=409,
            )

    def create(self, prompt: str) -> dict[str, Any]:
        self._require_open()
        if self.session_id is not None:
            raise PersistentParentSessionError(
                "persistent parent session was already created", code="session_already_created", status=409,
            )
        response = dict(self.engine.reference_match_persistent_create(
            prompt,
            reference_token_ids=self.reference_token_ids,
            generation_contract=self.generation_contract,
        ))
        session_id = response.get("session_id")
        version = response.get("parent_version")
        digest = response.get("parent_prompt_digest")
        if (not isinstance(session_id, str) or not session_id or
                isinstance(version, bool) or not isinstance(version, int) or version < 0 or
                not isinstance(digest, str) or not digest):
            raise PersistentParentSessionError(
                "persistent session create returned incomplete identity", code="malformed_session_create", status=502,
            )
        identity = response.get("runtime_identity")
        self.session_id = session_id
        self.parent_version = version
        self.parent_prompt_digest = digest
        self.runtime_identity = deepcopy(dict(identity)) if isinstance(identity, Mapping) else {}
        self.telemetry = deepcopy(dict(response.get("telemetry") or {}))
        return deepcopy(response)

    def probe_round(self, children: Sequence[Mapping[str, Any]], *, expected_parent_version: int | None = None) -> dict[str, Any]:
        self._require_created()
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes, bytearray)) or not children:
            raise PersistentParentSessionError("children must be a non-empty sequence", code="invalid_children", status=400)
        expected = self.parent_version if expected_parent_version is None else expected_parent_version
        if isinstance(expected, bool) or not isinstance(expected, int):
            raise PersistentParentSessionError("expected_parent_version must be an integer",
                                               code="invalid_parent_version", status=400)
        if expected != self.parent_version:
            raise PersistentParentSessionError("probe round is bound to an older parent version",
                                               code="stale_parent_state", status=409)
        ids: set[str] = set()
        ranks: set[int] = set()
        normalized = []
        for child in children:
            if not isinstance(child, Mapping):
                raise PersistentParentSessionError("each child must be an object", code="invalid_child", status=400)
            cid = child.get("candidate_id")
            rank = child.get("candidate_rank")
            prompt = child.get("prompt")
            if (not isinstance(cid, str) or not cid or isinstance(rank, bool) or not isinstance(rank, int)
                    or rank < 0 or not isinstance(prompt, str) or not prompt):
                raise PersistentParentSessionError(
                    "each child requires candidate_id, prompt, and non-negative candidate_rank",
                    code="invalid_child", status=400,
                )
            if cid in ids or rank in ranks:
                raise PersistentParentSessionError(
                    "candidate_id and candidate_rank must be unique", code="invalid_child", status=400,
                )
            ids.add(cid)
            ranks.add(rank)
            normalized.append({"candidate_id": cid, "candidate_rank": rank, "prompt": prompt})
        response = dict(self.engine.reference_match_persistent_probe(
            self.session_id, expected_parent_version=expected, children=normalized,
        ))
        rows = response.get("results")
        if not isinstance(rows, list) or len(rows) != len(normalized):
            raise PersistentParentSessionError(
                "persistent session probe returned the wrong child count", code="malformed_probe", status=502,
            )
        by_id = {}
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("candidate_id"), str):
                raise PersistentParentSessionError("persistent session probe returned malformed child identity",
                                                   code="malformed_probe", status=502)
            if row["candidate_id"] in by_id:
                raise PersistentParentSessionError("persistent session probe returned duplicate child identity",
                                                   code="malformed_probe", status=502)
            by_id[row["candidate_id"]] = deepcopy(dict(row))
        if set(by_id) != ids:
            raise PersistentParentSessionError("persistent session probe returned the wrong candidate IDs",
                                               code="malformed_probe", status=502)
        self._last_round = {
            cid: {"parent_version": expected, "row": row}
            for cid, row in by_id.items()
        }
        self.telemetry = deepcopy(dict(response.get("telemetry") or self.telemetry))
        return deepcopy(response)

    def promote(self, candidate: Mapping[str, Any] | str, *, scalar_preserves: bool,
                native_preserves: bool = True) -> dict[str, Any]:
        self._require_created()
        if not scalar_preserves:
            raise PersistentParentSessionError(
                "trusted scalar rejection prevents persistent promotion",
                code="scalar_rejection_prevents_promotion", status=409,
            )
        if not native_preserves:
            raise PersistentParentSessionError(
                "native session did not nominate a preserving candidate",
                code="native_rejection_prevents_promotion", status=409,
            )
        cid = candidate if isinstance(candidate, str) else candidate.get("candidate_id")
        if not isinstance(cid, str) or not cid:
            raise PersistentParentSessionError("candidate_id is required for promotion",
                                               code="invalid_candidate", status=400)
        record = self._last_round.get(cid)
        if record is None or record.get("parent_version") != self.parent_version:
            raise PersistentParentSessionError(
                "candidate was not produced from the current parent version",
                code="stale_candidate", status=409,
            )
        row = record["row"]
        if row.get("native_preserves") is False:
            raise PersistentParentSessionError(
                "native session did not nominate this candidate", code="native_rejection_prevents_promotion", status=409,
            )
        response = dict(self.engine.reference_match_persistent_promote(
            self.session_id, expected_parent_version=self.parent_version, candidate_id=cid,
        ))
        version = response.get("parent_version")
        if isinstance(version, bool) or not isinstance(version, int) or version <= self.parent_version:
            raise PersistentParentSessionError("persistent promotion returned an invalid parent version",
                                               code="malformed_promotion", status=502)
        self.parent_version = version
        self.parent_prompt_digest = response.get("parent_prompt_digest", self.parent_prompt_digest)
        self._last_round = {}
        self.telemetry = deepcopy(dict(response.get("telemetry") or self.telemetry))
        return deepcopy(response)

    def cancel_round(self) -> None:
        """Discard local nominations; no promotion is implied by cancellation."""
        self._require_created()
        self._last_round = {}

    def close(self) -> dict[str, Any] | None:
        if self.closed:
            return None
        if self.session_id is None:
            self.closed = True
            self._last_round = {}
            return None
        response = dict(self.engine.reference_match_persistent_close(self.session_id))
        if response.get("closed") is not True:
            raise PersistentParentSessionError("persistent session close was not acknowledged",
                                               code="malformed_close", status=502)
        self.closed = True
        self._last_round = {}
        return deepcopy(response)

    def report(self) -> dict[str, Any]:
        """Return a deterministic detached state/telemetry document for benchmark output."""
        return {
            "session_id": self.session_id,
            "parent_version": self.parent_version,
            "parent_prompt_digest": self.parent_prompt_digest,
            "runtime_identity": deepcopy(self.runtime_identity),
            "telemetry": deepcopy(self.telemetry),
            "closed": self.closed,
        }


__all__ = [
    "PersistentParentParityError",
    "PersistentParentSessionClient",
    "PersistentParentSessionError",
    "assert_scalar_parity",
    "candidate_id",
    "evidence_projection",
]
