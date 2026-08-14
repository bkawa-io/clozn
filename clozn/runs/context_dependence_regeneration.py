"""One optional free-generation observation for a measured Context Dependence set.

Teacher-forced Context Dependence is the primary measurement.  This module
does not score tokens, estimate a source effect, or call a generated child a
causal confirmation.  It only binds an *already measured* ``delete_source``
experiment to the current parent, reconstructs its exact canonical receipt
deletion through :mod:`clozn.replay.span_bridge`, and requests exactly one
``replay(..., messages_override=...)`` child.

The returned object intentionally keeps the teacher-forced reference and the
free-generation observation in separate branches.  ``diff_runs`` is used once
after a successful child exists so its first-divergence view remains the
repository's canonical comparison, rather than a second implementation here.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math
from typing import Any

from clozn import schemas
from clozn.analysis.model_diff import diff_runs
from clozn.receipts import rederive
from clozn.replay.span_bridge import (
    ContextReceiptSourceResolutionError,
    resolve_context_receipt_source_set,
)
from clozn.replay.replay import replay as replay_run


# v1 remains readable for historical target-scoped studies.  v2 is the
# canonical run-level artifact and has no target in its identity or records.
STUDY_SCHEMA_VERSIONS = frozenset({
    "clozn.context-dependence-study.v1", "clozn.context-dependence-study.v2",
})
RESULT_SCHEMA_VERSION = "clozn.context-dependence-regeneration.v1"
INTERVENTION_OPERATOR = "delete_source"


class ContextDependenceRegenerationError(ValueError):
    """The supplied study/experiment cannot be bound to this parent faithfully."""


class ContextDependenceRegenerationStaleError(ContextDependenceRegenerationError):
    """The parent changed after a model-free regeneration plan was made."""


class ContextDependenceRegenerationExperimentNotFoundError(ContextDependenceRegenerationError):
    """The requested experiment is absent from the recorded study."""


def _digest(value: Any) -> str:
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContextDependenceRegenerationError(
            f"Context Dependence regeneration binding is not JSON-serializable: {exc}") from exc
    return hashlib.sha256(encoded).hexdigest()


def _study_copy(study: Any) -> dict:
    if not isinstance(study, Mapping):
        raise ContextDependenceRegenerationError("study must be a Context Dependence study object")
    document = deepcopy(dict(study))
    schema_version = document.get("schema_version")
    if schema_version not in STUDY_SCHEMA_VERSIONS:
        raise ContextDependenceRegenerationError(
            "study must declare a supported Context Dependence schema version")
    try:
        schemas.validate(document, schema_version)
    except Exception as exc:  # schema failure is input evidence failure, never something to repair
        raise ContextDependenceRegenerationError(
            f"study is not a valid {schema_version}: {type(exc).__name__}: {exc}") from exc
    return document


def _parent_binding(parent: Mapping) -> str:
    """The parent facts that affect this exact deletion/replay request."""
    return _digest({
        "id": parent.get("id"),
        "messages": parent.get("messages"),
        "assembled_messages": parent.get("assembled_messages"),
        "context_receipt": parent.get("context_receipt"),
        "identity": parent.get("identity"),
        "behavior": parent.get("behavior"),
        "meta": parent.get("meta"),
    })


def _exact_ranges(value: Any) -> list[dict]:
    if not isinstance(value, list) or not value or not all(isinstance(item, Mapping) for item in value):
        raise ContextDependenceRegenerationError("experiment.exact_removed_ranges is malformed")
    return [deepcopy(dict(item)) for item in value]


def _ranges_still_bind(expected: list[dict], resolved: list[dict]) -> bool:
    """Compare legacy range evidence without pretending old records had new hashes.

    v1 experiments recorded the four coordinate fields.  Span-aware studies
    additionally carry the source-content hash and source kind.  The old
    record remains valid when those coordinates re-resolve exactly; whenever a
    newer field was recorded it must agree too.
    """
    if len(expected) != len(resolved):
        return False
    required = ("source_id", "message_index", "unicode_range", "byte_range")
    for old, current in zip(expected, resolved):
        if any(old.get(key) != current.get(key) for key in required):
            return False
        for key in ("content_sha256", "source_kind"):
            if key in old and old.get(key) != current.get(key):
                return False
    return True


def _find_experiment(study: Mapping, experiment_id: Any) -> dict:
    if not isinstance(experiment_id, str) or not experiment_id:
        raise ContextDependenceRegenerationError("experiment_id must be a non-empty string")
    matches = [item for item in study.get("experiments", [])
               if isinstance(item, Mapping) and item.get("experiment_id") == experiment_id]
    if not matches:
        raise ContextDependenceRegenerationExperimentNotFoundError(
            "experiment_id does not identify an experiment in the supplied study")
    if len(matches) != 1:
        raise ContextDependenceRegenerationError(
            "experiment_id does not identify exactly one experiment in the supplied study")
    experiment = deepcopy(dict(matches[0]))
    if experiment.get("intervention_operator") != INTERVENTION_OPERATOR:
        raise ContextDependenceRegenerationError("experiment is not a delete_source intervention")
    if experiment.get("provenance") != "measured":
        raise ContextDependenceRegenerationError("experiment is not a directly measured intervention")
    removed = experiment.get("removed_source_ids")
    if (
        not isinstance(removed, list) or not removed
        or any(not isinstance(value, str) or not value for value in removed)
        or len(set(removed)) != len(removed) or removed != sorted(removed)
    ):
        raise ContextDependenceRegenerationError(
            "experiment.removed_source_ids must be one sorted, unique canonical source-ID set")
    _exact_ranges(experiment.get("exact_removed_ranges"))
    if not isinstance(experiment.get("context_hash"), str) or len(experiment["context_hash"]) != 64:
        raise ContextDependenceRegenerationError("experiment.context_hash is malformed")
    return experiment


def _study_source_binding(study: Mapping, resolved: Mapping) -> None:
    identity = study.get("source_identity")
    if not isinstance(identity, Mapping) or identity.get("kind") not in {
        "context_receipt_segment_id", "context_receipt_source_span",
    }:
        raise ContextDependenceRegenerationError("study is not bound to canonical Context Receipt sources")
    # v1 wrote assembled/delivered while v2 preserves the resolver's actual
    # scoring basis spelling.  Both describe the same verified basis.
    expected_view = "assembled" if resolved.get("basis") == "assembled_messages" else "delivered"
    actual_view = identity.get("view")
    if actual_view == resolved.get("basis"):
        actual_view = expected_view
    if actual_view != expected_view:
        raise ContextDependenceRegenerationError("study source basis does not match the exact replay basis")
    sources = identity.get("sources")
    if not isinstance(sources, list) or not all(isinstance(item, Mapping) for item in sources):
        raise ContextDependenceRegenerationError("study source identity is malformed")
    study_ids = [item.get("source_id") for item in sources]
    if study_ids != resolved.get("available_source_ids"):
        raise ContextDependenceRegenerationError(
            "study source identities no longer exactly match the current Context Receipt basis")


def _score_context_hash(parent: Mapping, resolved: Mapping) -> str:
    """Rebuild Task 1's score-context digest and reject un-replayable blocks.

    Current receipts use an assembled, block-free basis.  A legacy prompt block
    has no representation in replay's ``messages_override`` seam, so allowing
    it through would bind the generation to a different condition than the
    measured experiment.  Refuse it rather than silently omitting it.
    """
    conditions = rederive.with_arm_conditions(dict(parent))
    expected_basis = "assembled_messages" if conditions.get("block_source") == "assembled_messages" else "messages"
    if resolved.get("basis") != expected_basis:
        raise ContextDependenceRegenerationError(
            "teacher-forced and replay message bases do not exactly agree")
    block = conditions.get("block")
    if block not in (None, ""):
        raise ContextDependenceRegenerationError(
            "the measured score condition includes a prompt block replay cannot faithfully override")
    return _digest({"messages": resolved.get("messages"), "block": block})


def _recorded_active_dials(parent: Mapping) -> dict[str, float]:
    """Return the exact recorded steering state for the replay child."""
    behavior = parent.get("behavior")
    value = behavior.get("active_dials") if isinstance(behavior, Mapping) else {}
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContextDependenceRegenerationError("recorded active dials are malformed")
    result: dict[str, float] = {}
    for name, amount in value.items():
        if (
            not isinstance(name, str) or not name
            or not isinstance(amount, (int, float)) or isinstance(amount, bool)
            or not math.isfinite(float(amount))
        ):
            raise ContextDependenceRegenerationError("recorded active dials are malformed")
        result[name] = float(amount)
    return result


def _reference(experiment: Mapping) -> dict:
    """Small, structural teacher-forced reference; it is not a generated result."""
    result = {
        "measurement_mode": "teacher_forced",
        "provenance": "measured",
        "experiment_id": experiment["experiment_id"],
        "intervention_operator": experiment["intervention_operator"],
        "removed_source_ids": deepcopy(experiment["removed_source_ids"]),
        "delta_nats": deepcopy(experiment.get("delta_nats")),
    }
    if "intervened_logp" in experiment or "baseline_logp" in experiment:
        result["continuation_logp"] = deepcopy(experiment.get("intervened_logp"))
        result["baseline_continuation_logp"] = deepcopy(experiment.get("baseline_logp"))
    else:
        # v1's target-scoped public shape remains exactly readable.
        result["target_logp"] = deepcopy(experiment.get("target_logp"))
        result["baseline_target_logp"] = deepcopy(experiment.get("baseline_target_logp"))
    return result


def plan_context_dependence_regeneration(parent_run: Mapping, study: Mapping, experiment_id: str) -> dict:
    """Bind an existing measured deletion experiment without generating or scoring.

    The plan is deliberately inexpensive and reproducible.  Execution rebuilds
    it against the current parent again, so an earlier plan is never authority
    to delete a stale source position.
    """
    if not isinstance(parent_run, Mapping) or not isinstance(parent_run.get("id"), str) or not parent_run["id"]:
        raise ContextDependenceRegenerationError("parent_run must be a stored run with a non-empty id")
    document = _study_copy(study)
    if document.get("run_id") != parent_run["id"]:
        raise ContextDependenceRegenerationError("study.run_id does not match the requested parent run")
    experiment = _find_experiment(document, experiment_id)
    try:
        resolved = resolve_context_receipt_source_set(dict(parent_run), experiment["removed_source_ids"])
    except ContextReceiptSourceResolutionError as exc:
        raise ContextDependenceRegenerationStaleError(
            f"canonical receipt source deletion is unavailable: {exc}") from exc
    try:
        _study_source_binding(document, resolved)
    except ContextDependenceRegenerationError as exc:
        raise ContextDependenceRegenerationStaleError(str(exc)) from exc
    if resolved["canonical_source_ids"] != experiment["removed_source_ids"]:
        raise ContextDependenceRegenerationStaleError("experiment source set did not re-resolve exactly")
    expected_ranges = _exact_ranges(experiment["exact_removed_ranges"])
    if not _ranges_still_bind(expected_ranges, resolved["exact_removed_ranges"]):
        raise ContextDependenceRegenerationStaleError(
            "experiment exact_removed_ranges no longer match the current Context Receipt basis")
    if _score_context_hash(parent_run, resolved) != experiment["context_hash"]:
        raise ContextDependenceRegenerationStaleError(
            "experiment context_hash no longer matches the exact current score/replay basis")
    recorded_dials = _recorded_active_dials(parent_run)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "state": "ready",
        "parent_run_id": str(parent_run["id"]),
        "parent_binding_sha256": _parent_binding(parent_run),
        "teacher_forced_reference": _reference(experiment),
        "intervention": {
            "operator": INTERVENTION_OPERATOR,
            "removed_source_ids": deepcopy(resolved["canonical_source_ids"]),
            # Retain v1's old coordinate-only public shape verbatim; v2
            # retains its richer span evidence because it was recorded.
            "exact_removed_ranges": deepcopy(expected_ranges),
            "source_basis": resolved["basis"],
            "basis_digest": resolved["basis_digest"],
            "intervened_context_digest": resolved["intervened_context_digest"],
            "teacher_forced_context_hash": experiment["context_hash"],
        },
        "execution": {
            "requires_generation": True,
            "generation_calls": 1,
            "decode_regime": "greedy",
            "recorded_active_dials": recorded_dials,
            "messages_override": deepcopy(resolved["messages"]),
        },
    }


def _changes_applied(plan: Mapping) -> dict:
    intervention = plan["intervention"]
    # ``greedy`` gives replay's generation seam an explicit deterministic
    # decode request. Clear live controls, then restore the recorded controls
    # so canonical source deletion remains the only planned prompt difference.
    changes = {
        "greedy": True,
        "behavior_off": True,
        "context_dependence_regeneration": {
            "parent_run_id": plan["parent_run_id"],
            "experiment_id": plan["teacher_forced_reference"]["experiment_id"],
            "intervention_operator": intervention["operator"],
            "intervention": {
                "operator": intervention["operator"],
                "scope": "canonical_context_receipt_span_aware_deletion",
            },
            # Keep the compact conventional spelling alongside the more
            # explicit name below; both are sorted canonical segment IDs.
            "removed_source_ids": deepcopy(intervention["removed_source_ids"]),
            "removed_canonical_source_ids": deepcopy(intervention["removed_source_ids"]),
            "exact_removed_ranges": deepcopy(intervention["exact_removed_ranges"]),
            "source_basis": intervention["source_basis"],
            "basis_digest": intervention["basis_digest"],
            "intervened_context_digest": intervention["intervened_context_digest"],
            "teacher_forced_context_hash": intervention["teacher_forced_context_hash"],
        }
    }
    dials = plan["execution"].get("recorded_active_dials")
    if isinstance(dials, Mapping) and dials:
        changes["behavior_overrides"] = deepcopy(dict(dials))
    return changes


def execute_context_dependence_regeneration(
    parent_run: Mapping,
    study: Mapping,
    experiment_id: str,
    sub,
    *,
    plan: Mapping | None = None,
    reload_parent=None,
    max_new: int | None = None,
) -> dict:
    """Create exactly one canonical-source-deletion replay child, if still bound.

    No teacher-forced scorer is called here.  A successful generated child is
    compared to its parent through :func:`diff_runs`; the result is explicitly
    an observed regeneration, not causal confirmation of the likelihood arm.
    """
    current_parent = reload_parent(parent_run.get("id")) if callable(reload_parent) else parent_run
    if not isinstance(current_parent, Mapping):
        raise ContextDependenceRegenerationStaleError("the parent could not be reloaded for execution")
    current = plan_context_dependence_regeneration(current_parent, study, experiment_id)
    if plan is not None:
        if not isinstance(plan, Mapping) or plan.get("parent_binding_sha256") != current["parent_binding_sha256"]:
            raise ContextDependenceRegenerationStaleError(
                "the parent or canonical receipt changed after regeneration planning")
        if plan.get("intervention") != current["intervention"]:
            raise ContextDependenceRegenerationStaleError(
                "the planned deletion no longer matches the current canonical receipt")

    kwargs: dict[str, Any] = {"messages_override": deepcopy(current["execution"]["messages_override"])}
    if isinstance(max_new, int) and not isinstance(max_new, bool) and max_new > 0:
        kwargs["max_new"] = max_new
    # Exactly one generation seam invocation.  Do not add a no-op control, do
    # not use reference_tokens, and do not retry a failure: this is an optional
    # observational regeneration, not a causal confirmation harness.
    child = replay_run(dict(current_parent), _changes_applied(current), sub, **kwargs)
    result = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "parent_run_id": current["parent_run_id"],
        "teacher_forced_reference": deepcopy(current["teacher_forced_reference"]),
        "intervention": deepcopy(current["intervention"]),
        "regeneration": {
            "measurement_mode": "free_generation",
            "provenance": "observed_generated",
            "generation_calls": 1,
            "causal_confirmation": False,
            "note": (
                "This is one observed generated child under the already measured deletion; it is not "
                "causal confirmation of the teacher-forced Context Dependence experiment."
            ),
        },
    }
    if not isinstance(child, Mapping) or not child.get("id"):
        result["regeneration"].update({"state": "failed", "reason": "replay_generation_failed"})
        return result
    child_copy = deepcopy(dict(child))
    comparison = diff_runs(dict(current_parent), child_copy)
    result["regeneration"].update({
        "state": "completed",
        "child_run_id": child_copy["id"],
        # This is the canonical model_diff output, including its
        # first_divergence_view; no second comparison algorithm is introduced.
        "comparison": comparison,
    })
    return result


__all__ = [
    "ContextDependenceRegenerationError",
    "ContextDependenceRegenerationExperimentNotFoundError",
    "ContextDependenceRegenerationStaleError",
    "INTERVENTION_OPERATOR",
    "RESULT_SCHEMA_VERSION",
    "execute_context_dependence_regeneration",
    "plan_context_dependence_regeneration",
]
