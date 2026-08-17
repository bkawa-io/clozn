"""Optional shared accepted-parent execution for exact-reference batches.

The client is an execution optimization only.  It never decides whether an
observation preserves the recorded answer and it never re-probes a cached
condition just to populate a worker session.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any


class SharedParentSessionError(ValueError):
    def __init__(self, message: str, *, code: str = "shared_parent_session_invalid", status: int = 409):
        super().__init__(message)
        self.code = code
        self.status = status


def condition_candidate_id(condition: Any) -> str:
    """Make an ephemeral worker ID from the full condition identity."""
    payload = json.dumps(condition, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return "spc_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def evidence_projection(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in (
        "status", "matched_token_count", "first_divergence_index", "divergence_kind",
        "termination_match", "finish_reason",
    )}


@dataclass
class SharedParentSessionClient:
    """Typed lifecycle client for one worker-local accepted-parent session."""

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
            raise SharedParentSessionError("shared parent session is closed", code="session_closed")

    def _require_created(self) -> None:
        self._require_open()
        if self.session_id is None or self.parent_version is None:
            raise SharedParentSessionError("shared parent session has not been created", code="session_not_created")

    def create(self, prompt: str) -> dict[str, Any]:
        self._require_open()
        if self.session_id is not None:
            raise SharedParentSessionError("shared parent session was already created", code="session_already_created")
        response = dict(self.engine.reference_match_persistent_create(
            prompt, reference_token_ids=self.reference_token_ids,
            generation_contract=dict(self.generation_contract),
        ))
        session_id = response.get("session_id")
        version = response.get("parent_version")
        digest = response.get("parent_prompt_digest")
        if (not isinstance(session_id, str) or not session_id or isinstance(version, bool)
                or not isinstance(version, int) or version < 0 or not isinstance(digest, str) or not digest):
            raise SharedParentSessionError("persistent session create returned incomplete identity", code="malformed_session_create", status=502)
        self.session_id = session_id
        self.parent_version = version
        self.parent_prompt_digest = digest
        self.runtime_identity = deepcopy(dict(response.get("runtime_identity") or {}))
        self.telemetry = deepcopy(dict(response.get("telemetry") or {}))
        return deepcopy(response)

    def probe_round(self, children: Sequence[Mapping[str, Any]], *, expected_parent_version: int | None = None) -> dict[str, Any]:
        self._require_created()
        if not isinstance(children, Sequence) or isinstance(children, (str, bytes, bytearray)) or not children:
            raise SharedParentSessionError("children must be a non-empty sequence", code="invalid_children", status=400)
        expected = self.parent_version if expected_parent_version is None else expected_parent_version
        if expected != self.parent_version:
            raise SharedParentSessionError("probe round is bound to an older parent version", code="stale_parent_state")
        normalized: list[dict[str, Any]] = []
        ids: set[str] = set()
        ranks: set[int] = set()
        for child in children:
            if not isinstance(child, Mapping):
                raise SharedParentSessionError("each child must be an object", code="invalid_child", status=400)
            candidate = child.get("candidate_id")
            rank = child.get("candidate_rank")
            prompt = child.get("prompt")
            if (not isinstance(candidate, str) or not candidate or isinstance(rank, bool)
                    or not isinstance(rank, int) or rank < 0 or not isinstance(prompt, str) or not prompt):
                raise SharedParentSessionError("each child requires candidate_id, candidate_rank, and prompt", code="invalid_child", status=400)
            if candidate in ids or rank in ranks:
                raise SharedParentSessionError("candidate IDs and ranks must be unique", code="invalid_child", status=400)
            ids.add(candidate); ranks.add(rank)
            normalized.append({"candidate_id": candidate, "candidate_rank": rank, "prompt": prompt})
        response = dict(self.engine.reference_match_persistent_probe(
            self.session_id, expected_parent_version=expected, children=normalized,
        ))
        rows = response.get("results")
        if not isinstance(rows, list) or len(rows) != len(normalized):
            raise SharedParentSessionError("persistent session probe returned the wrong child count", code="malformed_probe", status=502)
        by_id: dict[str, dict[str, Any]] = {}
        for row in rows:
            if not isinstance(row, Mapping) or not isinstance(row.get("candidate_id"), str):
                raise SharedParentSessionError("persistent session probe returned malformed child identity", code="malformed_probe", status=502)
            if row["candidate_id"] in by_id:
                raise SharedParentSessionError("persistent session probe returned duplicate child identity", code="malformed_probe", status=502)
            by_id[row["candidate_id"]] = deepcopy(dict(row))
        if set(by_id) != ids:
            raise SharedParentSessionError("persistent session probe returned the wrong candidate IDs", code="malformed_probe", status=502)
        self._last_round = {candidate: {"parent_version": expected, "row": row} for candidate, row in by_id.items()}
        self.telemetry = deepcopy(dict(response.get("telemetry") or self.telemetry))
        return deepcopy(response)

    def promote(self, candidate: Mapping[str, Any] | str, *, exact_preserved: bool) -> dict[str, Any]:
        self._require_created()
        if not exact_preserved:
            raise SharedParentSessionError("only directly preserved evidence may promote a parent", code="promotion_requires_exact_preservation")
        candidate_id = candidate if isinstance(candidate, str) else candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise SharedParentSessionError("candidate_id is required for promotion", code="invalid_candidate", status=400)
        record = self._last_round.get(candidate_id)
        if record is None or record.get("parent_version") != self.parent_version:
            raise SharedParentSessionError("candidate was not produced from the current parent version", code="stale_candidate")
        response = dict(self.engine.reference_match_persistent_promote(
            self.session_id, expected_parent_version=self.parent_version, candidate_id=candidate_id,
        ))
        version = response.get("parent_version")
        if isinstance(version, bool) or not isinstance(version, int) or version <= self.parent_version:
            raise SharedParentSessionError("persistent promotion returned an invalid parent version", code="malformed_promotion", status=502)
        self.parent_version = version
        self.parent_prompt_digest = response.get("parent_prompt_digest", self.parent_prompt_digest)
        self._last_round = {}
        self.telemetry = deepcopy(dict(response.get("telemetry") or self.telemetry))
        return deepcopy(response)

    def close(self) -> dict[str, Any] | None:
        if self.closed:
            return None
        if self.session_id is None:
            self.closed = True
            self._last_round = {}
            return None
        response = dict(self.engine.reference_match_persistent_close(self.session_id))
        if response.get("closed") is not True:
            raise SharedParentSessionError("persistent session close was not acknowledged", code="malformed_close", status=502)
        self.closed = True
        self._last_round = {}
        return deepcopy(response)

    def report(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id, "parent_version": self.parent_version,
            "parent_prompt_digest": self.parent_prompt_digest,
            "runtime_identity": deepcopy(self.runtime_identity),
            "telemetry": deepcopy(self.telemetry), "closed": self.closed,
        }


__all__ = ["SharedParentSessionClient", "SharedParentSessionError", "condition_candidate_id", "evidence_projection"]
