"""Explicit conversion of one completed ephemeral arm into one child Run."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from clozn.analysis.model_diff import diff_runs
from clozn.runs import store as run_store
from clozn.receipts.rederive import with_arm_conditions
from clozn.replay.replay import replay as replay_run
from clozn.replay.span_bridge import ContextReceiptSourceResolutionError

from .execution import (
    DeleteSourceExactReferenceAdapter,
    ExecutionAdapterError,
    ExecutionStateStaleError,
    resolve_delete_source,
)
from .effective_prompt import resolve_effective_prompt
from .interventions import DeleteSource, ForceToken
from .observations import GeneratedObservation, execution_observation_identity
from .persistence import ExperimentArmView, ExperimentView, ObservationStore
from .runner import ExperimentResult
from .state import ExecutionState
from .state_ref import ResolvedState


SCHEMA_VERSION = "clozn.experiment-materialization.v1"


class MaterializationError(ValueError):
    """The requested arm cannot become a faithful child run."""


class MaterializationStaleError(MaterializationError):
    """The parent or source binding changed after the experiment was run."""


def _arm(result: ExperimentResult | ExperimentView, arm_id: str) -> DeleteSource:
    try:
        arm = result.arm_for(arm_id)
    except KeyError as exc:
        raise MaterializationError(f"experiment result has no arm {arm_id!r}") from exc
    if not isinstance(arm, ExperimentArmView):
        raise MaterializationError("experiment result returned an invalid arm association")
    if arm.state != "completed" or arm.observation is None:
        raise MaterializationError(
            f"arm {arm_id!r} has no completed exact-reference observation"
        )
    if arm.observation.status not in {"exact_preserved", "diverged"}:
        raise MaterializationError(
            f"arm {arm_id!r} has no completed exact-reference observation"
        )
    if not isinstance(arm.intervention, DeleteSource):
        raise MaterializationError(f"arm {arm_id!r} has no materializable delete-source intervention")
    return arm.intervention


def _resolve_result(result: ExperimentResult | ExperimentView | str | None, *,
                    experiment_id: str | None, observation_store: ObservationStore | None) -> ExperimentResult | ExperimentView:
    if isinstance(result, (ExperimentResult, ExperimentView)):
        return result
    resolved_id = result if isinstance(result, str) else experiment_id
    if resolved_id and observation_store is not None:
        return observation_store.get_experiment(resolved_id)
    raise TypeError("materialize_arm requires an ExperimentResult or a persisted experiment ID/store")


def _materialize_context_generated_observation(
    base_run: Mapping[str, Any], resolved: ExperimentResult | ExperimentView,
    arm_id: str, *, observation_id: str | None,
    reload_parent: Callable[[str], Mapping[str, Any] | None] | None,
) -> dict[str, Any]:
    """Promote a prompt-boundary Generate observation without re-rendering or running a model."""
    from .evaluators import Generate

    if not isinstance(resolved.evaluator, Generate):
        raise MaterializationError("context generated materialization requires a Generate evaluator")
    try:
        arm = resolved.arm_for(arm_id)
    except KeyError as exc:
        raise MaterializationError(f"experiment result has no arm {arm_id!r}") from exc
    if arm.state != "completed" or not isinstance(arm.observation, GeneratedObservation):
        raise MaterializationError("arm has no completed GeneratedObservation")
    observation = arm.observation
    if observation.status != "completed":
        raise MaterializationError("unavailable or failed generation cannot be materialized")
    if observation.state_ref is not None:
        raise MaterializationError("context GeneratedObservation must not carry a StateRef")
    if not isinstance(arm.intervention, DeleteSource):
        raise MaterializationError("context generated materialization requires DeleteSource")
    if observation_id is not None and observation.observation_id != observation_id:
        raise MaterializationStaleError("the requested GeneratedObservation does not match the persisted arm")
    if resolved.base.run_id != base_run.get("id"):
        raise MaterializationStaleError("experiment result is bound to another parent run")

    current_parent = reload_parent(base_run["id"]) if callable(reload_parent) else base_run
    if not isinstance(current_parent, Mapping):
        raise MaterializationStaleError("the parent could not be reloaded")
    current_parent = dict(current_parent)
    try:
        current_state = ExecutionState.from_run(current_parent)
    except Exception as exc:
        raise MaterializationStaleError(f"the current parent has no valid execution state: {exc}") from exc
    if current_state.execution_fingerprint != resolved.base.execution_fingerprint:
        raise MaterializationStaleError("the parent execution fingerprint changed after the experiment")
    if current_state.context_receipt_identity.get("digest") != resolved.base.context_receipt_identity.get("digest"):
        raise MaterializationStaleError("the parent Context Receipt changed after the experiment")

    expected = execution_observation_identity(resolved.base, resolved.evaluator, arm.intervention)
    if (observation.observation_id != expected["observation_id"]
            or observation.observation_key_sha256 != expected["observation_key_sha256"]):
        raise MaterializationStaleError("GeneratedObservation identity does not match the persisted arm")
    if observation.intervention != arm.intervention.to_dict():
        raise MaterializationStaleError("GeneratedObservation intervention does not match the persisted arm")

    snapshot = observation.input_snapshot
    if not isinstance(snapshot, Mapping):
        raise MaterializationError("GeneratedObservation has no immutable context input snapshot")
    if snapshot.get("context_receipt_digest") != current_state.context_receipt_digest:
        raise MaterializationStaleError("GeneratedObservation Context Receipt binding is stale")
    if list(snapshot.get("source_ids") or []) != list(arm.intervention.source_ids):
        raise MaterializationStaleError("GeneratedObservation source binding does not match the arm")

    try:
        current_resolved = resolve_delete_source(current_parent, arm.intervention)
        current_prompt = resolve_effective_prompt(current_parent, arm.intervention)
    except Exception as exc:
        raise MaterializationStaleError(f"current source deletion cannot be revalidated: {exc}") from exc
    for key in ("exact_removed_ranges", "source_basis", "basis_digest", "intervened_context_digest"):
        snapshot_key = key
        resolved_key = "basis" if key == "source_basis" else key
        if snapshot.get(snapshot_key) != current_resolved.get(resolved_key):
            raise MaterializationStaleError(f"GeneratedObservation {key} no longer matches the current Context Receipt")
    if snapshot.get("messages") != current_prompt.worker_messages():
        raise MaterializationStaleError("GeneratedObservation deleted messages no longer match the current prompt")
    if snapshot.get("assembled_messages") != current_prompt.rendered_messages():
        raise MaterializationStaleError("GeneratedObservation assembled prompt no longer matches the current prompt")
    final_prompt = snapshot.get("final_prompt")
    if not isinstance(final_prompt, str) or not final_prompt:
        raise MaterializationError("GeneratedObservation is missing the exact final prompt")

    response = observation.generated_suffix_text
    child_trace = None
    trace_state = "unavailable"
    if isinstance(observation.generated_steps, (list, tuple)) and observation.generated_steps:
        steps = [deepcopy(step) for step in observation.generated_steps]
        valid = all(
            isinstance(step, Mapping)
            and isinstance(step.get("piece"), str)
            and isinstance(step.get("token_id"), int)
            and not isinstance(step.get("token_id"), bool)
            and step.get("token_id") >= 0
            for step in steps
        )
        if valid and "".join(step["piece"] for step in steps) == response:
            from clozn.runs.trace import steps_to_trace
            child_trace = steps_to_trace(steps)
            trace_state = "available" if child_trace else "unavailable"

    changes = {
        "experiment": {
            "experiment_id": resolved.experiment_id,
            "arm_id": arm_id,
            "observation_id": observation.observation_id,
            "base_state": {
                "run_id": resolved.base.run_id,
                "origin": "recorded_prompt_boundary",
                "realized_fidelity": observation.fidelity_classification,
            },
            "operation": "delete_source_generate",
            "origin": {"kind": "recorded_prompt_boundary"},
            "intervention": arm.intervention.to_dict(),
        },
    }
    captured_runtime = snapshot.get("runtime_identity")
    child_identity = (
        deepcopy(dict(captured_runtime))
        if isinstance(captured_runtime, Mapping) else
        deepcopy(current_parent.get("identity") or {})
    )
    child_id = run_store.record(
        source="experiment", client="experimental_kernel",
        model=current_parent.get("model", ""), substrate=current_parent.get("substrate", ""),
        messages=deepcopy(snapshot.get("messages") or []),
        assembled_messages=deepcopy(snapshot.get("assembled_messages") or []),
        final_prompt=final_prompt, response=response,
        trace=child_trace, finish_reason=observation.finish_reason,
        parent_run_id=current_parent["id"], changes_applied=changes,
        meta=deepcopy(current_parent.get("meta") or {}),
        identity=child_identity,
        output_contract=deepcopy(current_parent.get("output_contract") or {}),
    )
    if not child_id:
        return {
            "schema_version": SCHEMA_VERSION, "state": "failed",
            "parent_run_id": current_parent["id"], "experiment_id": resolved.experiment_id,
            "arm_id": arm_id, "observation_id": observation.observation_id,
            "reason": "run_persistence_failed",
        }
    child = run_store.get_run(child_id)
    comparison = diff_runs(current_parent, child) if isinstance(child, Mapping) else None
    return {
        "schema_version": SCHEMA_VERSION, "state": "completed",
        "parent_run_id": current_parent["id"], "child_run_id": child_id,
        "experiment_id": resolved.experiment_id, "arm_id": arm_id,
        "observation_id": observation.observation_id,
        "realized_fidelity": observation.fidelity_classification,
        "trace_state": trace_state, "comparison": comparison,
        "compare_path": f"#/compare/{current_parent['id']}/{child_id}",
    }


def materialize_arm(
    base_run: Mapping[str, Any],
    result: ExperimentResult | ExperimentView | str | None,
    arm_id: str,
    *,
    substrate: Any | None = None,
    execution_adapter: DeleteSourceExactReferenceAdapter | None = None,
    reload_parent: Callable[[str], Mapping[str, Any] | None] | None = None,
    max_new: int | None = None,
    replay_fn: Callable[..., Mapping[str, Any] | None] = replay_run,
    experiment_id: str | None = None,
    observation_id: str | None = None,
    require_preserved: bool = False,
    observation_store: ObservationStore | None = None,
    store: ObservationStore | None = None,
    materialization_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Revalidate one arm and persist exactly one ordinary replay child.

    Search/proof arms remain outside the run store.  Only this explicit call
    invokes the existing replay persistence seam.
    """
    if not isinstance(base_run, Mapping) or not isinstance(base_run.get("id"), str) or not base_run["id"]:
        raise MaterializationError("base_run must carry a non-empty id")
    if observation_store is not None and store is not None and observation_store is not store:
        raise ValueError("pass only one observation store")
    result = _resolve_result(
        result, experiment_id=experiment_id,
        observation_store=observation_store or store,
    )
    if result.state not in {"completed", "cancelled"}:
        raise MaterializationError("only an experiment with a completed arm can be materialized")
    if result.base.run_id != base_run["id"]:
        raise MaterializationStaleError("experiment result is bound to another parent run")
    try:
        selected_arm = result.arm_for(arm_id)
    except KeyError as exc:
        raise MaterializationError(f"experiment result has no arm {arm_id!r}") from exc
    if observation_id is not None and selected_arm.observation_id != observation_id:
        raise MaterializationStaleError("the requested observation does not match the persisted arm")
    if require_preserved and (
        selected_arm.observation is None or selected_arm.observation.status != "exact_preserved"
    ):
        raise MaterializationError("materialization requires a directly observed exact_preserved arm")

    if callable(reload_parent):
        current_parent = reload_parent(base_run["id"])
    elif execution_adapter is not None and callable(getattr(execution_adapter, "load_run", None)):
        current_parent = execution_adapter.load_run(result.base)
    else:
        current_parent = base_run
    if not isinstance(current_parent, Mapping):
        raise MaterializationStaleError("the parent could not be reloaded")
    current_parent = dict(current_parent)
    try:
        current_state = ExecutionState.from_run(current_parent)
    except Exception as exc:
        raise MaterializationStaleError(f"the current parent has no valid execution state: {exc}") from exc
    if current_state.execution_fingerprint != result.base.execution_fingerprint:
        raise MaterializationStaleError("the parent execution fingerprint changed after the experiment")
    if current_state.context_receipt_identity.get("digest") != result.base.context_receipt_identity.get("digest"):
        raise MaterializationStaleError("the parent Context Receipt changed after the experiment")

    intervention = _arm(result, arm_id)
    try:
        resolved = resolve_delete_source(current_parent, intervention)
    except (ContextReceiptSourceResolutionError, ExecutionAdapterError) as exc:
        raise MaterializationStaleError(f"canonical source binding is stale or unavailable: {exc}") from exc

    conditions = with_arm_conditions(current_parent)
    if resolved.get("basis") != "assembled_messages" and conditions.get("block") not in (None, ""):
        raise MaterializationStaleError(
            "the recorded prompt block cannot be faithfully reconstructed through replay"
        )

    adapter = execution_adapter
    if adapter is None:
        if substrate is None:
            raise MaterializationError("materialization requires a substrate or execution_adapter")
        adapter = DeleteSourceExactReferenceAdapter(substrate, run=current_parent)
    if substrate is None:
        substrate = getattr(adapter, "substrate", None)
    if substrate is None:
        raise MaterializationError("execution_adapter has no substrate for ordinary generation")

    changes: dict[str, Any] = {
        "experiment": {
            "experiment_id": result.experiment_id,
            "arm_id": arm_id,
            "observation_id": selected_arm.observation_id,
            "intervention": {
                "kind": "delete_source",
                "source_ids": list(intervention.source_ids),
            },
        },
    }
    # Opaque derived provenance is supplied by the caller-facing recipe.  The
    # generic materializer records it without interpreting Minimal Context or
    # introducing a recipe-specific execution path.
    if materialization_context is not None:
        if not isinstance(materialization_context, Mapping):
            raise MaterializationError("materialization_context must be an object")
        changes["experiment"]["derived_provenance"] = deepcopy(dict(materialization_context))
    contract = result.base.generation_contract or {}
    decode_mode = contract.get("decode_mode") if isinstance(contract, Mapping) else None
    if decode_mode == "greedy":
        changes["greedy"] = True
        sampling_override: bool | dict[str, Any] = False
    elif decode_mode == "sample" and isinstance(contract.get("sampling"), Mapping):
        sampling_override = deepcopy(dict(contract["sampling"]))
    else:
        raise MaterializationError("the parent generation contract cannot be faithfully replayed")

    kwargs: dict[str, Any] = {
        "messages_override": deepcopy(resolved.get("messages") or []),
        "sampling_override": sampling_override,
    }
    if isinstance(max_new, int) and not isinstance(max_new, bool) and max_new > 0:
        kwargs["max_new"] = max_new
    child = replay_fn(current_parent, changes, substrate, **kwargs)
    if not isinstance(child, Mapping) or not child.get("id"):
        return {
            "schema_version": SCHEMA_VERSION,
            "state": "failed",
            "parent_run_id": current_parent["id"],
            "experiment_id": result.experiment_id,
            "arm_id": arm_id,
            "intervention": deepcopy(changes["experiment"]["intervention"]),
            "reason": "generation_failed",
        }

    child_copy = deepcopy(dict(child))
    comparison = diff_runs(current_parent, child_copy)
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "completed",
        "parent_run_id": current_parent["id"],
        "child_run_id": child_copy["id"],
        "experiment_id": result.experiment_id,
        "arm_id": arm_id,
        "intervention": deepcopy(changes["experiment"]["intervention"]),
        "comparison": comparison,
        "compare_path": f"#/compare/{current_parent['id']}/{child_copy['id']}",
    }


def materialize_generated_observation(
    base_run: Mapping[str, Any],
    result: ExperimentResult | ExperimentView | str | None,
    arm_id: str,
    *,
    experiment_id: str | None = None,
    observation_store: ObservationStore | None = None,
    store: ObservationStore | None = None,
    reload_parent: Callable[[str], Mapping[str, Any] | None] | None = None,
    observation_id: str | None = None,
) -> dict[str, Any]:
    """Promote one persisted GeneratedObservation to one child Run.

    This function is intentionally a pure persistence promotion: it performs
    no substrate/model work.  Repeated calls explicitly create another child
    with the same immutable evidence and provenance.
    """
    if not isinstance(base_run, Mapping) or not isinstance(base_run.get("id"), str) or not base_run["id"]:
        raise MaterializationError("base_run must carry a non-empty id")
    if observation_store is not None and store is not None and observation_store is not store:
        raise ValueError("pass only one observation store")
    durable = observation_store or store
    resolved = _resolve_result(result, experiment_id=experiment_id, observation_store=durable)
    if isinstance(resolved.base, ExecutionState):
        return _materialize_context_generated_observation(
            base_run, resolved, arm_id, observation_id=observation_id,
            reload_parent=reload_parent,
        )
    if not isinstance(resolved.base, ResolvedState):
        raise MaterializationError("generated materialization requires an ExecutionState or ResolvedState base")
    try:
        arm = resolved.arm_for(arm_id)
    except KeyError as exc:
        raise MaterializationError(f"experiment result has no arm {arm_id!r}") from exc
    if arm.state != "completed" or not isinstance(arm.observation, GeneratedObservation):
        raise MaterializationError("arm has no completed GeneratedObservation")
    observation = arm.observation
    if observation.status != "completed":
        raise MaterializationError("unavailable or failed generation cannot be materialized")
    if observation_id is not None and observation.observation_id != observation_id:
        raise MaterializationStaleError("the requested GeneratedObservation does not match the persisted arm")
    if arm.intervention is not None and not isinstance(arm.intervention, ForceToken):
        raise MaterializationError("generated materialization requires ForceToken or an unchanged condition")
    if resolved.base.run_id != base_run["id"]:
        raise MaterializationStaleError("experiment result is bound to another parent run")
    current_parent = reload_parent(base_run["id"]) if callable(reload_parent) else base_run
    if not isinstance(current_parent, Mapping):
        raise MaterializationStaleError("the parent could not be reloaded")
    current_parent = dict(current_parent)
    try:
        current_state = observation.state_ref.assert_current(current_parent)
    except Exception as exc:
        raise MaterializationStaleError(f"StateRef is stale relative to the current parent: {exc}") from exc
    if current_state.execution_fingerprint != resolved.base.execution_fingerprint:
        raise MaterializationStaleError("the parent execution fingerprint changed after the experiment")
    if observation.state_ref != resolved.base.state_ref:
        raise MaterializationStaleError("GeneratedObservation StateRef does not match the experiment base")
    expected = execution_observation_identity(resolved.base, resolved.evaluator, arm.intervention)
    if observation.observation_id != expected["observation_id"] \
            or observation.observation_key_sha256 != expected["observation_key_sha256"]:
        raise MaterializationStaleError("GeneratedObservation identity does not match the persisted arm")
    expected_intervention = arm.intervention.to_dict() if arm.intervention is not None else None
    if observation.intervention != expected_intervention:
        raise MaterializationStaleError("GeneratedObservation intervention does not match the persisted arm")

    trace = current_parent.get("trace") if isinstance(current_parent.get("trace"), Mapping) else {}
    pieces = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
    position = resolved.base.position.index
    if len(pieces) < position:
        raise MaterializationStaleError("the parent token boundary is no longer available")
    prefix = "".join(str(piece) for piece in pieces[:position])
    response = prefix + observation.generated_suffix_text
    child_trace = None
    if observation.fidelity_classification == "exact_execution_fork" and isinstance(observation.generated_steps, list):
        parent_steps = trace.get("steps")
        generated_steps = observation.generated_steps
        if isinstance(parent_steps, list) and all(isinstance(step, Mapping) for step in generated_steps):
            full_steps = [deepcopy(step) for step in parent_steps[:position]] + [deepcopy(step) for step in generated_steps]
            valid_steps = all(
                isinstance(step.get("piece"), str)
                and isinstance(step.get("token_id"), int)
                and not isinstance(step.get("token_id"), bool)
                and step.get("token_id") >= 0
                for step in full_steps
            )
            if valid_steps and "".join(step["piece"] for step in full_steps) == response:
                from clozn.runs.trace import steps_to_trace
                child_trace = steps_to_trace(full_steps)

    intervention = arm.intervention.to_dict() if arm.intervention is not None else None
    changes = {
        "experiment": {
            "experiment_id": resolved.experiment_id,
            "arm_id": arm_id,
            "observation_id": observation.observation_id,
            "base_state": {
                "run_id": resolved.base.run_id,
                "execution_fingerprint": resolved.base.execution_fingerprint,
                "state_ref": resolved.base.state_ref.to_dict(),
                "position": resolved.base.position.to_dict(),
                "resolved_classification": resolved.base.classification,
                "resolved_proof_status": resolved.base.proof_status,
                "realization_fingerprint": resolved.base.realization_fingerprint,
                "realized_fidelity": observation.fidelity_classification,
                "fidelity": deepcopy(observation.fidelity),
            },
            "operation": "force_token" if arm.intervention is not None else "continue",
            "intervention": intervention,
            "exact_control_proof": deepcopy(observation.exact_control_proof),
        },
    }
    child_id = run_store.record(
        source="experiment", client="experimental_kernel",
        model=current_parent.get("model", ""), substrate=current_parent.get("substrate", ""),
        messages=deepcopy(current_parent.get("messages") or []), response=response,
        trace=child_trace,
        final_prompt=current_parent.get("final_prompt"), finish_reason=observation.finish_reason,
        parent_run_id=current_parent["id"], changes_applied=changes,
        meta=deepcopy(current_parent.get("meta") or {}),
        assembled_messages=deepcopy(current_parent.get("assembled_messages")),
        identity=deepcopy(current_parent.get("identity") or {}),
        output_contract=deepcopy(current_parent.get("output_contract") or {}),
    )
    if not child_id:
        return {
            "schema_version": SCHEMA_VERSION, "state": "failed",
            "parent_run_id": current_parent["id"], "experiment_id": resolved.experiment_id,
            "arm_id": arm_id, "observation_id": observation.observation_id,
            "reason": "run_persistence_failed",
        }
    child = run_store.get_run(child_id)
    comparison = diff_runs(current_parent, child) if isinstance(child, Mapping) else None
    return {
        "schema_version": SCHEMA_VERSION, "state": "completed", "child_run_id": child_id,
        "parent_run_id": current_parent["id"], "experiment_id": resolved.experiment_id,
        "arm_id": arm_id, "observation_id": observation.observation_id,
        "realized_fidelity": observation.fidelity_classification,
        "trace_state": "available" if child_trace is not None else "unavailable",
        "comparison": comparison,
        "compare_path": f"#/compare/{current_parent['id']}/{child_id}",
    }


MaterializeBranch = materialize_arm


__all__ = [
    "MaterializationError", "MaterializationStaleError", "MaterializeBranch",
    "SCHEMA_VERSION", "materialize_arm", "materialize_generated_observation",
]
