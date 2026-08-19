"""Canonical historical exact-resume evidence over persisted observations.

One projection, one definition.  A boundary is *historically verified exact* when a persisted,
completed :class:`GeneratedObservation` is bound to this run and its current execution fingerprint,
addresses a StateRef on that run, was realized as an exact execution state, and carries confirmed
matched unchanged-control evidence.

Two things this deliberately does NOT require:

* **Materialization.**  A GeneratedObservation is already historical execution evidence.  Requiring
  a child Run would mean generation only counts once it branches, which is exactly the coupling the
  kernel removed.  A materialized child may enrich a display, but it never establishes exactness.
* **Live availability.**  Historically verified exact says a rewind *was* proven exact, not that one
  is runnable now.  Live readiness needs current checkpoint, runtime, and worker resolution and is
  the canonical resolver's answer, never this projection's.

Every check is explicit and fails closed.  Nothing is inferred from one field implying another.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .observations import GeneratedObservation
from .state import ExecutionState

PROOF = {"proof_status": "confirmed", "unchanged_control_status": "matched", "exact_match": True}
BOUNDARY_STATE = "historically_verified_exact"
_KNOWN_REGIMES = ("generated_token_live_kv", "prompt_boundary_reprefill")


def load_exact_evidence(run_id: str, *, observation_store: Any = None) -> list[GeneratedObservation]:
    """Read persisted generated observations for one run, newest first.

    This is the only I/O in this module, and it is a pure read: no worker, resolver, checkpoint
    creation, or experiment execution.  A missing or unreadable store yields no evidence rather than
    an error, because a read-only product surface must stay useful without one.
    """
    if not isinstance(run_id, str) or not run_id:
        return []
    store = observation_store
    if store is None:
        try:
            from .persistence import ObservationStore

            store = ObservationStore()
        except Exception:
            return []
    try:
        loaded = store.list_observations(run_id=run_id, evaluator_kind="generate", status="completed")
    except Exception:
        return []
    return [item for item in loaded if isinstance(item, GeneratedObservation)]


def _mapping(value: Any) -> dict:
    return dict(value) if isinstance(value, Mapping) else {}


def _control_proof_confirms(observation: GeneratedObservation) -> bool:
    """The attached control proof must itself say the unchanged control matched.

    The fidelity block is a claim; this is the receipt behind it.  A claim without a consistent
    receipt is not proof, so the two are checked separately rather than one trusted to imply the
    other.
    """
    proof = _mapping(observation.exact_control_proof)
    if proof.get("status") != "matched":
        return False
    result = _mapping(proof.get("result"))
    return result.get("status") == "matched" and result.get("exact_match") is True


def is_verified_exact(observation: Any, *, run: Mapping[str, Any], run_id: str,
                      fingerprint: str) -> bool:
    """Every condition for historically verified exact evidence, checked independently.

    ``fingerprint`` is the run's CURRENT execution fingerprint -- the same digest an observation is
    keyed on -- so evidence recorded against a run whose identity, trace, or response has since
    changed is stale and never counts.
    """
    if not isinstance(observation, GeneratedObservation):
        return False
    if observation.status != "completed":
        return False
    if observation.run_id != run_id:
        return False
    if observation.base_execution_fingerprint != fingerprint:
        return False
    state_ref = observation.state_ref
    if state_ref is None:
        return False
    try:
        if state_ref.run_id != run_id or state_ref.execution_fingerprint != fingerprint:
            return False
        # The canonical staleness primitive: the StateRef must still address THIS run's current
        # recorded execution, not merely carry a matching digest string.
        state_ref.assert_current(dict(run))
        position = state_ref.position.index
    except Exception:
        return False
    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        return False
    fidelity = _mapping(observation.fidelity)
    if fidelity.get("classification") != "exact_execution_fork":
        return False
    if fidelity.get("proof_status") != "confirmed":
        return False
    if fidelity.get("exact_match") is not True:
        return False
    if fidelity.get("unchanged_control") != "matched":
        return False
    return _control_proof_confirms(observation)


def _position_of(observation: GeneratedObservation) -> int:
    return observation.state_ref.position.index


def _regimes(group: Sequence[GeneratedObservation]) -> list[str]:
    found = set()
    for observation in group:
        regime = _mapping(observation.realization).get("regime")
        if isinstance(regime, str) and regime in _KNOWN_REGIMES:
            found.add(regime)
    return sorted(found)


def _runtime_key(observation: GeneratedObservation) -> str | None:
    runtime = _mapping(_mapping(observation.realization).get("runtime_identity"))
    value = runtime.get("runtime_key_sha256")
    return value if isinstance(value, str) and len(value) == 64 else None


def verified_exact_boundaries(run: Mapping[str, Any],
                              observations: Sequence[Any] = ()) -> dict[str, Any]:
    """Group verified exact evidence by the answer-token boundary it addresses.

    Pure: reads only its arguments and mutates neither.  Evidence that fails any condition is
    excluded and counted, never best-effort reinterpreted into a weaker claim.
    """
    run = run if isinstance(run, Mapping) else {}
    run_id = str(run.get("id") or "")
    if not run_id:
        return {"state": "available", "verified_boundaries": []}
    try:
        fingerprint = ExecutionState.from_run(dict(run)).execution_fingerprint
    except Exception:
        # A run with no resolvable execution state can carry no historical exact proof at all.
        return {"state": "available", "verified_boundaries": []}

    verified_by_position: dict[int, list[GeneratedObservation]] = {}
    rejected = 0
    for observation in list(observations or ()):
        if not isinstance(observation, GeneratedObservation):
            rejected += 1
            continue
        if not is_verified_exact(observation, run=run, run_id=run_id, fingerprint=fingerprint):
            continue
        verified_by_position.setdefault(_position_of(observation), []).append(observation)

    boundaries = []
    for position in sorted(verified_by_position):
        # The loader already returns newest-first, so the first entry is the latest evidence.
        group = verified_by_position[position]
        latest = group[0]
        entry = {
            "position": position,
            "state": BOUNDARY_STATE,
            "verified_observation_count": len(group),
            "latest_observation_id": latest.observation_id,
            "parent_fingerprint_sha256": fingerprint,
            "regimes": _regimes(group),
            "proof": dict(PROOF),
        }
        runtime_key = _runtime_key(latest)
        if runtime_key is not None:
            entry["runtime_key_sha256"] = runtime_key
        boundaries.append(entry)

    out: dict[str, Any] = {
        "state": "partially_unavailable" if rejected else "available",
        "verified_boundaries": boundaries,
    }
    if rejected:
        out["rejected_evidence_count"] = rejected
    return out


__all__ = [
    "BOUNDARY_STATE", "PROOF", "is_verified_exact", "load_exact_evidence",
    "verified_exact_boundaries",
]
