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
`resolve_state` can return `classification == "exact_execution_fork"` with
`proof_status == "planned"` -- that means preflight eligibility passed, nothing more. Only a COMPLETED
`GeneratedObservation` whose `fidelity.proof_status == "confirmed"` AND
`fidelity.unchanged_control == "matched"` AND `fidelity.exact_match is True`, backed by a consistent
`exact_control_proof`, counts as proof (`clozn.experiments.historical_evidence.is_verified_exact`
checks each explicitly, never inferring one from another). A merely planned resolution, an
unavailable or failed observation, or a reconstructed realization never produces a verified boundary
-- see the failed-attempt language rule below.

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
its only execution dependencies are the neutral model-free facts module and the canonical historical
evidence projection. The observation store is READ ONLY consulted by the caller (route), never here.
`build_rewind_fidelity` takes `historical_observations` as a plain argument specifically so this module
never needs to know how they were loaded (I/O stays in the route layer).

DETERMINISM
-----------
Pure function of `(run, historical_observations)`: no randomness, no wall-clock, no model/engine access.
Neither `run` nor any observation is ever mutated.
"""
from __future__ import annotations

from collections.abc import Sequence

from clozn.experiments.execution_facts import (
    KNOWN_CHANGES, RECONSTRUCTED_CHANGES, RECONSTRUCTION_DIFFERENCES,
    recorded_execution_prerequisites,
)
from clozn.experiments.historical_evidence import verified_exact_boundaries

SCHEMA_VERSION = "clozn.rewind-fidelity.v2"

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

def _state(available: bool) -> str:
    return "available" if available else "unavailable"


def _identity_dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def build_rewind_fidelity(run: dict, *, historical_observations: Sequence = ()) -> dict:
    """Build and validate one derived `clozn.rewind-fidelity.v1` document.

    Pure function of its arguments: reads only `run` and `historical_observations` (neither mutated).
    Never imports an engine client, worker, model-routing module, or checkpoint primitive -- works
    identically with no active worker attached (see tests/test_rewind_fidelity.py's
    `test_no_engine_or_worker_access`). `historical_observations` should be the persisted canonical
    GeneratedObservation evidence for this run's id (typically
    `clozn.experiments.historical_evidence.load_exact_evidence(run_id)`, loaded by the caller -- this
    function performs no I/O itself).

    Historical proof does not require materialization: a completed, exact, control-confirmed
    observation IS the historical execution evidence. It also never implies live availability.

    Raises `ValueError` only for a structurally invalid `run` (missing/empty id) -- there is no other
    caller-supplied input to validate; non-canonical entries in `historical_observations` degrade
    `historical_proof.state` to `"partially_unavailable"` rather than raising.
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
            "authority": "exact_state_resolution",
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
        "historical_proof": verified_exact_boundaries(run, historical_observations),
        "live_execution": {
            "state": "not_checked",
            "reason": "read_only_projection",
            "authority": "exact_state_resolution",
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
