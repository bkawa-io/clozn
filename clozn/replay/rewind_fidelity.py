"""Rewind Fidelity -- a read-only, offline-safe projection answering "if I rewind this recorded run,
what fidelity can Clozn truthfully promise?" (E10).

THREE CONCEPTS, NEVER COLLAPSED
------------------------------------
1. RECONSTRUCTED replay. Clozn can rebuild the prefix and replay from recorded text/tokens. This does
   NOT restore the same internal execution state -- KV state is not restored, sampler state is
   reinitialized, the prompt prefix is retokenized, batch shape is not preserved (see the neutral
   execution-facts contract, reused verbatim, never copied). Labeled
   `reconstructed`, NEVER `exact`.
2. EXACT rewind ELIGIBILITY. The recorded run carries enough evidence that an exact execution fork MAY
   be possible, but exact execution still needs LIVE checks this module cannot and does not perform:
   matching runtime identity, matching worker generation, a compatible checkpoint, valid checkpoint
   token history, and the mandatory unchanged control. Labeled `requires_live_plan` -- never
   `exact_available`. The canonical Time Travel resolver owns those live checks; this module
   remains a read-only projection.
3. VERIFIED exact rewind. A prior exact execution fork actually ran from this boundary and its
   MANDATORY unchanged control reproduced the parent's continuation under Clozn's strict token/text
   comparison through the exact-resume proof seam. This is HISTORICAL
   evidence -- `historically_verified_exact` -- and even then this module reports
   `exact_rewind.state == "requires_live_plan"` and `live_execution.state == "not_checked"` right next
   to it. "Was exact then" is never conflated with "is executable exactly now": the checkpoint or
   worker generation that made the proof possible may no longer exist. See
   `tests/test_rewind_fidelity.py::test_historical_verification_never_upgrades_live_state` for the
   regression this module exists to prevent.

EXACT IS A PROOF STATE, NOT A PLANNING CLASSIFICATION
-----------------------------------------------------------
`plan_execution_fork` can return `classification == "exact_execution_fork"` with
`exactness.proof_status == "planned"` -- that means preflight eligibility passed, nothing more. Only a
TERMINAL receipt (`phase == "completed"`) whose `exactness.proof_status == "confirmed"` AND
`unchanged_control.status == "matched"` AND `unchanged_control.result.exact_match is True` counts as
proof (`_is_verified_exact` below checks all four explicitly, never inferring one from another). A
`planned` plan, a `diverged`/`failed`/`cancelled` execution, or `unchanged_control.status ==
"required_not_run"` never produces a verified boundary -- see the failed-attempt language rule below.

REUSE, NEVER A SECOND DEFINITION
-------------------------------------
Static prerequisites and supported change vocabularies come from the neutral execution-facts
contract. This module does NOT replicate a live runtime-identity match (there is no selected runtime
to match against without a live caller); it only asks whether the PARENT's own recorded runtime
identity resolves at all.

FAILED EXACTNESS ATTEMPTS ARE NOT "IMPOSSIBLE FOREVER"
------------------------------------------------------------
A `diverged`/`failed`/`cancelled` terminal receipt means only "this exact attempt did not establish
exactness" -- the worker, checkpoint, or runtime may differ on a future attempt. V1 simply omits these
from `historical_proof.verified_boundaries` (never labels the boundary `exact_unavailable_forever`, and
never surfaces a separate "failed attempt" record in this first version -- see the module's own test
suite for the `diverged`/`failed`/`cancelled` cases, all asserting "no verified boundary", not a
negative claim).

NO WORKER, NO CHECKPOINT, EVER
-----------------------------------
This module imports NOTHING that can reach a worker, an engine, model routing, or checkpoint capture --
its only execution dependency is the neutral model-free facts module. The receipt store is READ ONLY
consulted by the caller (route), never imported here.
`build_rewind_fidelity` takes `historical_receipts` as a plain argument specifically so this module never
needs to know how they were loaded (filesystem access stays in the route/results-store layer).

DETERMINISM
-----------
Pure function of `(run, historical_receipts)`: no randomness, no wall-clock, no model/engine access.
Neither `run` nor any receipt in `historical_receipts` is ever mutated.
"""
from __future__ import annotations

from clozn.experiments.execution_facts import (
    KNOWN_CHANGES, RECONSTRUCTED_CHANGES, RECONSTRUCTION_DIFFERENCES,
    parent_execution_fingerprint,
    recorded_execution_prerequisites,
)

SCHEMA_VERSION = "clozn.rewind-fidelity.v1"

RECORDED_CAPABILITY_STATES = frozenset({"available", "limited", "unavailable"})
RECONSTRUCTED_REPLAY_STATES = frozenset({"available", "unavailable"})
EXACT_REWIND_STATES = frozenset({"requires_live_plan", "static_prerequisites_unavailable"})

_LIVE_REQUIREMENTS = [
    "matching_runtime_identity",
    "matching_worker_generation",
    "compatible_checkpoint",
    "valid_checkpoint_token_history",
    "unchanged_control",
]

_PROOF = {"proof_status": "confirmed", "unchanged_control_status": "matched", "exact_match": True}


def _state(available: bool) -> str:
    return "available" if available else "unavailable"


def _identity_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _is_verified_exact(receipt: dict) -> bool:
    """All FOUR conditions, explicitly -- never inferred from just one. See the module docstring's
    "EXACT IS A PROOF STATE" section for why each is checked independently rather than trusted to imply
    the others."""
    if receipt.get("phase") != "completed":
        return False
    if receipt.get("classification") != "exact_execution_fork":
        return False
    exactness = _identity_dict(receipt.get("exactness"))
    if exactness.get("proof_status") != "confirmed":
        return False
    control = _identity_dict(receipt.get("unchanged_control"))
    if control.get("status") != "matched":
        return False
    result = _identity_dict(control.get("result"))
    if result.get("exact_match") is not True:
        return False
    return True


def _boundary_sort_key(receipt: dict) -> tuple:
    execution = _identity_dict(receipt.get("execution"))
    ended = execution.get("ended_ts")
    ended = ended if isinstance(ended, (int, float)) and not isinstance(ended, bool) else 0.0
    execution_id = receipt.get("execution_id")
    return (-ended, execution_id if isinstance(execution_id, str) else "")


def _historical_proof(run_id: str, run: dict, historical_receipts) -> dict:
    from clozn import schemas

    fingerprint = parent_execution_fingerprint(run)
    malformed = 0
    verified_by_position: dict[int, list[dict]] = {}

    for receipt in list(historical_receipts or ()):
        if not isinstance(receipt, dict):
            malformed += 1
            continue
        try:
            schemas.validate(receipt, "clozn.execution-fork.v1")
        except (schemas.ValidationError, schemas.SchemaError):
            malformed += 1
            continue
        if receipt.get("parent_run_id") != run_id:
            continue
        if receipt.get("parent_fingerprint_sha256") != fingerprint:
            # Stale relative to the CURRENT immutable parent state -- never attached as proof of
            # rewind fidelity for a run that has since changed identity/trace/response.
            continue
        if not _is_verified_exact(receipt):
            continue
        position = _identity_dict(receipt.get("request")).get("position")
        if not isinstance(position, int) or isinstance(position, bool):
            continue
        verified_by_position.setdefault(position, []).append(receipt)

    boundaries = []
    for position in sorted(verified_by_position):
        group = sorted(verified_by_position[position], key=_boundary_sort_key)
        latest = group[0]
        regimes = sorted({
            _identity_dict(receipt.get("exactness")).get("regime") for receipt in group
        } - {None})
        entry = {
            "position": position,
            "state": "historically_verified_exact",
            "verified_execution_count": len(group),
            "latest_execution_id": latest.get("execution_id"),
            "parent_fingerprint_sha256": fingerprint,
            "regimes": regimes,
            "proof": dict(_PROOF),
        }
        runtime_key = _identity_dict(
            _identity_dict(latest.get("identity")).get("parent_runtime")
        ).get("runtime_key_sha256")
        if isinstance(runtime_key, str) and runtime_key:
            entry["runtime_key_sha256"] = runtime_key
        boundaries.append(entry)

    out = {
        "state": "partially_unavailable" if malformed else "available",
        "verified_boundaries": boundaries,
    }
    if malformed:
        out["malformed_receipt_count"] = malformed
    return out


def build_rewind_fidelity(run: dict, *, historical_receipts: list = ()) -> dict:
    """Build and validate one derived `clozn.rewind-fidelity.v1` document.

    Pure function of its arguments: reads only `run` and `historical_receipts` (neither mutated).
    Never imports an engine client, worker, model-routing module, or checkpoint primitive -- works
    identically with no active worker attached (see tests/test_rewind_fidelity.py's
    `test_no_engine_or_worker_access`). `historical_receipts` should be every terminal execution-fork
    receipt already known for this run's id (typically `execution_fork_results.list_for_parent(run_id)`,
    loaded by the caller -- this function performs no I/O itself).

    Raises `ValueError` only for a structurally invalid `run` (missing/empty id) -- there is no other
    caller-supplied input to validate; malformed entries in `historical_receipts` degrade
    `historical_proof.state` to `"partially_unavailable"` rather than raising (see `_historical_proof`).
    """
    run = run if isinstance(run, dict) else {}
    run_id = str(run.get("id") or "")
    if not run_id:
        raise ValueError("run.id must be a non-empty string")

    prerequisites = recorded_execution_prerequisites(run)
    token_ready = prerequisites["token_alignment_available"]
    reconstructed_available = token_ready and prerequisites["final_prompt_available"]
    exact_static_available = token_ready and prerequisites["parent_runtime_identity_available"]

    if reconstructed_available:
        reconstructed_replay = {
            "state": "available",
            "supported_change_types": sorted(RECONSTRUCTED_CHANGES),
            "unavoidable_differences": list(RECONSTRUCTION_DIFFERENCES),
        }
    else:
        reasons = []
        if not token_ready:
            reasons.append("recorded_response_token_trace_unavailable")
        if not prerequisites["final_prompt_available"]:
            reasons.append("recorded_final_prompt_unavailable")
        reconstructed_replay = {"state": "unavailable", "reasons": reasons}

    static_prerequisites = {
        "recorded_token_pieces": _state(prerequisites["token_pieces_available"]),
        "recorded_token_ids": _state(prerequisites["token_ids_available"]),
        "token_alignment": _state(prerequisites["token_alignment_available"]),
        "runtime_identity": _state(prerequisites["parent_runtime_identity_available"]),
    }
    if exact_static_available:
        exact_rewind = {
            "state": "requires_live_plan",
            "static_prerequisites": static_prerequisites,
            "supported_change_types_if_live_plan_succeeds": sorted(KNOWN_CHANGES),
            "live_requirements": list(_LIVE_REQUIREMENTS),
            "authority": "execution_fork_plan",
        }
    else:
        reasons = []
        if not token_ready:
            reasons.append("recorded_response_token_trace_unavailable")
        if not prerequisites["parent_runtime_identity_available"]:
            reasons.append("parent_runtime_identity_unavailable")
        exact_rewind = {
            "state": "static_prerequisites_unavailable",
            "static_prerequisites": static_prerequisites,
            "reasons": reasons,
        }

    if reconstructed_available:
        recorded_state = "available" if exact_static_available else "limited"
    else:
        recorded_state = "unavailable"

    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "recorded_capability": {
            "state": recorded_state,
            "reconstructed_replay": reconstructed_replay,
            "exact_rewind": exact_rewind,
        },
        "historical_proof": _historical_proof(run_id, run, historical_receipts),
        "live_execution": {
            "state": "not_checked",
            "reason": "read_only_projection",
            "authority": "execution_fork_plan",
        },
        "privacy": "metadata_only",
    }
    token_count = prerequisites["recorded_token_count"]
    if token_count is not None:
        document["coordinates"] = {
            "kind": "recorded_response_token_boundary",
            "index_base": 0,
            "start": 0,
            "end_exclusive": token_count,
            "recorded_token_count": token_count,
        }

    from clozn import schemas
    schemas.validate(document)
    return document


__all__ = [
    "EXACT_REWIND_STATES",
    "RECONSTRUCTED_REPLAY_STATES",
    "RECORDED_CAPABILITY_STATES",
    "SCHEMA_VERSION",
    "build_rewind_fidelity",
]
