"""Bounded, run-scoped assembly of a Context Dependence study.

The individual Context Dependence modules deliberately keep their experiment
selection policies separate.  This module is the small execution seam used by
the HTTP job: it gives Quick and Standard one Task 1 measurement study, hence
one cached full-context baseline and one content-addressed experiment cache,
then attaches the available direct/search layers to one schema-valid artifact.

It never reads or writes ``influence_map``.  The caller owns persistence.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Callable, Iterable, Mapping

from clozn import schemas
from clozn.receipts.context_dependence import ContextDependenceStudy
from clozn.runs.context_dependence_preserving import (
    ComputeLevelPolicy,
    get_compute_level_policy,
    run_preserving_subset_search,
)
from clozn.runs.context_dependence_search import run_subset_screen


SCHEMA = "clozn.context-dependence-study.v2"
EXECUTION_METHOD = "context_dependence_execution.v2"
INTERVENTION_OPERATOR = "delete_source"


class ContextDependenceExecutionError(ValueError):
    """The requested run-scoped study cannot be executed faithfully."""


class ContextDependenceExecutionCancelled(RuntimeError):
    """The job was cancelled between scoreable experiments or layers."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite(value: Any, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContextDependenceExecutionError(f"{name} must be a finite number")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ContextDependenceExecutionError(f"{name} must be >= {minimum}")
    return number


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContextDependenceExecutionError("sampling_seed must be an integer")
    return value


def _source_set(value: Any, *, name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise ContextDependenceExecutionError(f"{name} must be an array of canonical source IDs")
    if not isinstance(value, (list, tuple)):
        raise ContextDependenceExecutionError(f"{name} must be an array of canonical source IDs")
    values = list(value)
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise ContextDependenceExecutionError(f"{name} must contain non-empty canonical source IDs")
    if len(values) != len(set(values)):
        raise ContextDependenceExecutionError(f"{name} must not contain duplicate source IDs")
    # Do not sort here: source IDs are later normalized into Context Receipt
    # order.  Keeping input order in the cache request also never creates a
    # false hit between two requests whose validation may differ in a future
    # receipt revision.
    return values


def _source_sets(value: Any, *, name: str) -> list[list[str]]:
    """Validate requested control sets without inventing source identity.

    Receipt-order normalization and unknown-ID rejection happen only after the
    same strict source resolver used for measurement has supplied the current
    source catalogue.  This parser deliberately preserves caller order so it
    can never silently turn a malformed set into a different one.
    """
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ContextDependenceExecutionError(f"{name} must be an array of canonical source-ID arrays")
    sets: list[list[str]] = []
    for index, source_ids in enumerate(value):
        parsed = _source_set(source_ids, name=f"{name}[{index}]")
        # ``_source_set`` returns None only for a direct None value, which is
        # never a valid member of the outer control-set list.
        if parsed is None:
            raise ContextDependenceExecutionError(f"{name}[{index}] must be a canonical source-ID array")
        sets.append(parsed)
    return sets


def _canonical_control_sets(
    source_order: Iterable[str], requested_sets: Iterable[Iterable[str]], *, name: str,
) -> tuple[tuple[str, ...], ...]:
    """Canonicalize explicit control sets in the public source-set ID order.

    This is intentionally stricter than a generic set conversion: requested
    unknown IDs are typed failures, while duplicate *sets* are deduplicated in
    their first requested order.  ``resolve_context_receipt_source_set`` has
    always made set identity lexical by canonical ID, so controls use that
    same order while their range evidence remains in prompt-basis order.  A caller consequently receives every
    distinct control it asked for or a clear refusal before any score pass.
    """
    order = tuple(source_order)
    available = set(order)
    seen: set[tuple[str, ...]] = set()
    result: list[tuple[str, ...]] = []
    for index, raw_set in enumerate(requested_sets):
        raw = tuple(raw_set)
        unknown = set(raw).difference(available)
        if unknown:
            raise ContextDependenceExecutionError(
                f"{name}[{index}] includes unknown canonical source IDs: "
                + ", ".join(sorted(unknown))
            )
        canonical = tuple(sorted(raw))
        if not canonical:
            # The parser already rejects empty lists.  Keep this fail-closed
            # guard adjacent to the receipt mapping in case it evolves.
            raise ContextDependenceExecutionError(f"{name}[{index}] resolves to no current sources")
        if canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return tuple(result)


def _cache_control_sets(run: Mapping[str, Any], request: Mapping[str, Any]) -> list[list[str]]:
    """Return strict canonical control identity without model scoring."""
    requested = request.get("neutralization_source_sets")
    if not requested:
        return []
    # Study construction only reads/strictly resolves the receipt; it does not
    # invoke score_tokens until ``document``/an arm is requested.
    source_order = ContextDependenceStudy(dict(run), None).source_ids
    return [list(item) for item in _canonical_control_sets(
        source_order, requested, name="neutralization_source_sets",
    )]


def normalize_request(body: Mapping[str, Any] | None) -> dict[str, Any]:
    """Strictly normalize the tiny public execution request.

    Every value affecting an experiment selection is retained in the returned
    object and therefore included in the cache key.  ``refresh`` is deliberately
    not an identity component; it only asks the route to bypass a valid cache.
    """
    if body is None:
        body = {}
    if not isinstance(body, Mapping):
        raise ContextDependenceExecutionError("request body must be an object")
    allowed = {
        "compute_level", "target", "root_source_ids", "method",
        "sampling_seed", "tolerance_nats", "neutralization_source_sets", "refresh",
    }
    unknown = set(body).difference(allowed)
    if unknown:
        raise ContextDependenceExecutionError(
            "unknown Context Dependence request field(s): " + ", ".join(sorted(map(str, unknown)))
        )
    level = body.get("compute_level", "Quick")
    try:
        policy = get_compute_level_policy(level)
    except ValueError as exc:
        raise ContextDependenceExecutionError(str(exc)) from None
    if "target" in body:
        raise ContextDependenceExecutionError(
            "target is not accepted by clozn.context-dependence-study.v2; "
            "the study always measures the full recorded continuation and selections are projections"
        )
    method = body.get("method", EXECUTION_METHOD)
    if not isinstance(method, str) or not method.strip():
        raise ContextDependenceExecutionError("method must be a non-empty string")
    refresh = body.get("refresh", False)
    if not isinstance(refresh, bool):
        raise ContextDependenceExecutionError("refresh must be a boolean")
    return {
        "compute_level": policy.name,
        "root_source_ids": _source_set(body.get("root_source_ids"), name="root_source_ids"),
        "method": method.strip(),
        "sampling_seed": _seed(body.get("sampling_seed", 0)),
        "tolerance_nats": _finite(body.get("tolerance_nats", 0.3), name="tolerance_nats", minimum=0.0),
        "neutralization_source_sets": _source_sets(
            body.get("neutralization_source_sets"), name="neutralization_source_sets",
        ),
        "refresh": refresh,
    }


def _identity_run_view(run: Mapping[str, Any]) -> dict[str, Any]:
    """Only immutable scoring inputs, not other derived attachments, bind reuse.

    This intentionally includes whole receipt/template/runtime records instead
    of guessing the one field a backend calls its fingerprint.  It therefore
    fails closed when a newer runtime adds another meaningful identity facet.
    """
    identity = deepcopy(run.get("identity")) if isinstance(run.get("identity"), Mapping) else {}
    if isinstance(identity, dict):
        identity.pop("captured_at", None)
    return {
        "run_id": run.get("id"),
        "model": deepcopy(run.get("model")),
        "substrate": deepcopy(run.get("substrate")),
        "runtime_identity": identity,
        "messages": deepcopy(run.get("messages")),
        "assembled_messages": deepcopy(run.get("assembled_messages")),
        "memory": deepcopy(run.get("memory")),
        "behavior": deepcopy(run.get("behavior")),
        "context_receipt": deepcopy(run.get("context_receipt")),
        "final_prompt": deepcopy(run.get("final_prompt")),
        "response": deepcopy(run.get("response")),
        "trace": deepcopy(run.get("trace")),
    }


def cache_binding(run: Mapping[str, Any], request: Mapping[str, Any]) -> dict[str, Any]:
    """The complete study-cache identity material, exposed for audit/tests."""
    return {
        "schema_version": SCHEMA,
        "execution_method": EXECUTION_METHOD,
        "run_scoring_identity": _identity_run_view(run),
        "requested_root_source_ids": deepcopy(request.get("root_source_ids")),
        "intervention_operator": INTERVENTION_OPERATOR,
        "compute_policy": request.get("compute_level"),
        "method": request.get("method"),
        "sampling_seed": request.get("sampling_seed"),
        "tolerance_nats": request.get("tolerance_nats"),
        # Controls are a separate optional robustness collection, but their
        # exact requested source sets still bind cache reuse.  Normalize with
        # the strict resolver's canonical source-set ID order so caller set ordering cannot mint
        # duplicate artifacts for the same control request.
        "neutralization_source_sets": _cache_control_sets(run, request),
    }


def cache_identity(run: Mapping[str, Any], request: Mapping[str, Any]) -> str:
    return f"cdcache_{_digest(cache_binding(run, request))[:32]}"


def cache_matches(run: Mapping[str, Any], artifact: Any, request: Mapping[str, Any]) -> bool:
    if not isinstance(artifact, Mapping) or artifact.get("schema_version") != SCHEMA:
        return False
    try:
        return artifact.get("cache_identity") == cache_identity(run, request)
    except ContextDependenceExecutionError:
        # A malformed/unknown control set must never be treated as a cache
        # hit.  Execution will surface the typed request/receipt error before
        # any score pass instead of leaking it from a read-only cache probe.
        return False


def _canonical_root(source_order: Iterable[str], requested: list[str] | None) -> tuple[str, ...]:
    source_order = tuple(source_order)
    if not source_order:
        raise ContextDependenceExecutionError("the Context Receipt has no canonical deletable sources")
    if requested is None:
        return source_order
    unknown = set(requested).difference(source_order)
    if unknown:
        raise ContextDependenceExecutionError(
            "root_source_ids includes unknown canonical source IDs: " + ", ".join(sorted(unknown))
        )
    requested_set = set(requested)
    return tuple(source_id for source_id in source_order if source_id in requested_set)


def _effect(experiment: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": experiment["experiment_id"],
        "delta_nats": experiment["delta_nats"],
        "provenance": "measured",
    }


def _hierarchy(root_source_ids: tuple[str, ...], experiments: list[Mapping[str, Any]], *,
               source_order: tuple[str, ...]) -> dict[str, Any]:
    """A compact Task 3-compatible measured direct hierarchy.

    The execution layer uses the shared Task 1 study directly so later screen,
    coalition, and preserving calls reuse its experiment cache.  This small
    projection retains the Task 3 invariant: every visible number cites a real
    experiment; it never creates an inferred effect.
    """
    by_set = {tuple(item["removed_source_ids"]): item for item in experiments}
    root_id = f"cdh_{_digest({'root': root_source_ids})[:24]}"
    root_exp = by_set.get(tuple(sorted(root_source_ids)))
    nodes: list[dict[str, Any]] = [{
        "node_id": root_id,
        "node_kind": "requested_root_set",
        "source_ids": list(root_source_ids),
        "measurement_state": "measured" if root_exp else "unmeasured_budget_exhausted",
        **({"direct_effect": _effect(root_exp)} if root_exp else {}),
    }]
    children: list[dict[str, Any]] = []
    for source_id in root_source_ids:
        exp = by_set.get((source_id,))
        node = {
            "node_id": f"cdh_{_digest({'root': root_id, 'source': source_id})[:24]}",
            "node_kind": "source_unit",
            "source_ids": [source_id],
            "parent_node_id": root_id,
            "measurement_state": "measured" if exp else "unmeasured_budget_exhausted",
        }
        if exp:
            node["direct_effect"] = _effect(exp)
        nodes.append(node)
        children.append(node)
    nonadditivity = []
    if root_exp and children and all("direct_effect" in item for item in children):
        nonadditivity.append({
            "parent_node_id": root_id,
            "parent_experiment_id": root_exp["experiment_id"],
            "child_experiment_ids": [item["direct_effect"]["experiment_id"] for item in children],
            "derived_value_nats": root_exp["delta_nats"] - sum(
                item["direct_effect"]["delta_nats"] for item in children
            ),
            "provenance": "derived_search_metadata",
            "not_a_measured_effect": True,
        })
    return {
        "provenance": "measured_direct_experiments",
        "root_node_id": root_id,
        "nodes": nodes,
        "nonadditivity": nonadditivity,
        "unmeasured_node_ids": [item["node_id"] for item in nodes
                                 if item["measurement_state"] != "measured"],
        "source_order": list(source_order),
    }


def _checkpoint(callback: Callable[..., Any] | None, *, phase: str, completed: int, total: int) -> None:
    if callback is None:
        return
    callback(phase=phase, completed=completed, total=total)


def run_context_dependence_execution(
    run: Mapping[str, Any], sub: Any, request: Mapping[str, Any] | None = None, *,
    checkpoint: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Execute Quick or Standard against exactly one Task 1 study/cache.

    ``checkpoint`` has the ``JobControl.checkpoint`` shape.  It is invoked
    before every direct experiment requested by this orchestration and before
    each larger layer.  The job registry turns its cancellation signal into a
    terminal cancelled job; no attachment is attempted after that signal.
    """
    normalized = normalize_request(request)
    policy: ComputeLevelPolicy = get_compute_level_policy(normalized["compute_level"])
    identity = cache_identity(run, normalized)
    measurement = ContextDependenceStudy(dict(run), sub)
    source_order = tuple(measurement.source_ids)
    root_source_ids = _canonical_root(source_order, normalized["root_source_ids"])
    neutralization_source_sets = _canonical_control_sets(
        source_order, normalized["neutralization_source_sets"],
        name="neutralization_source_sets",
    )
    # Baseline + the required canonical deletion root + every explicitly
    # requested robustness control must fit before the first score.  Later
    # optional layers merely consume the remaining shared policy budget.
    minimum_required_passes = 2 + len(neutralization_source_sets)
    if minimum_required_passes > policy.pass_budget:
        raise ContextDependenceExecutionError(
            "Context Dependence compute policy cannot score every requested neutralization control "
            f"within its {policy.pass_budget}-pass budget"
        )

    # Force/account for the shared baseline once, before reserving every
    # deletion arm.  The Task 1 document is the authority for consumed passes.
    _checkpoint(checkpoint, phase="baseline", completed=0, total=policy.pass_budget)
    measurement.document()

    def consumed() -> int:
        return int(measurement.document()["budget"]["passes_consumed"])

    def measure(source_ids: Iterable[str], *, phase: str) -> dict[str, Any]:
        _checkpoint(checkpoint, phase=phase, completed=consumed(), total=policy.pass_budget)
        if consumed() + 1 > policy.pass_budget:
            raise ContextDependenceExecutionError("Context Dependence compute policy score budget is exhausted")
        result = measurement.measure_removal_effect(source_ids)
        _checkpoint(checkpoint, phase=phase, completed=consumed(), total=policy.pass_budget)
        return result

    def neutralize(source_ids: Iterable[str]) -> dict[str, Any]:
        _checkpoint(checkpoint, phase="robustness_controls", completed=consumed(), total=policy.pass_budget)
        if consumed() + 1 > policy.pass_budget:
            # This should be impossible after ``minimum_required_passes``.
            # Keep the invariant explicit rather than silently omitting a
            # requested control if future orchestration changes consume early.
            raise ContextDependenceExecutionError(
                "Context Dependence compute policy exhausted before every requested neutralization control"
            )
        result = measurement.measure_neutralization_control(source_ids)
        _checkpoint(checkpoint, phase="robustness_controls", completed=consumed(), total=policy.pass_budget)
        return result

    # Layer 1/2. Quick measures the requested whole set; Standard additionally
    # measures receipt-native singleton units. Both share the exact baseline.
    measured_sets: set[tuple[str, ...]] = set()
    if consumed() + 1 <= policy.pass_budget:
        measured_sets.add(tuple(sorted(root_source_ids)))
        measure(root_source_ids, phase="direct_root")
    for source_ids in neutralization_source_sets:
        neutralize(source_ids)
    if policy.direct_source_work:
        for source_id in root_source_ids:
            source_set = (source_id,)
            if source_set in measured_sets or consumed() + 1 > policy.pass_budget:
                continue
            measured_sets.add(source_set)
            measure(source_set, phase="direct_sources")

    document = measurement.document()
    document["hierarchy"] = _hierarchy(
        root_source_ids, document["experiments"], source_order=source_order,
    )

    # Layer 3. ``run_subset_screen`` uses the same measurement adapter; Task
    # 1 content-addressing prevents duplicate source sets from being rescored.
    if policy.subset_screen and consumed() < policy.pass_budget:
        _checkpoint(checkpoint, phase="subset_screen", completed=consumed(), total=policy.pass_budget)
        # Preset mask counts are upper bounds.  A small receipt has only
        # ``2**n - 1`` unique non-empty deletion masks, so cap the request
        # instead of failing or manufacturing duplicate observations.
        available_mask_count = (1 << len(root_source_ids)) - 1
        document["screen"] = run_subset_screen(
            root_source_ids,
            lambda source_ids: measure(source_ids, phase="subset_screen"),
            sampling_seed=normalized["sampling_seed"],
            passes_requested=policy.pass_budget,
            initial_passes_consumed=consumed(),
            mask_count=min(policy.subset_mask_count, available_mask_count),
            existing_measurements=measurement.document().get("experiments", ()),
            min_holdout_observations=2,
        )

    # Layer 4. Available Task 5 verification is deliberately optional when a
    # screen did not qualify candidates; it still carries only real Task 1 IDs.
    if policy.subset_screen and consumed() < policy.pass_budget:
        _checkpoint(checkpoint, phase="coalition_verification", completed=consumed(), total=policy.pass_budget)
        from clozn.runs.context_dependence_coalitions import verify_coalitions
        coalitions = verify_coalitions(
            lambda source_ids: measure(source_ids, phase="coalition_verification"),
            source_ids=root_source_ids,
            passes_requested=policy.pass_budget,
            passes_consumed=consumed(),
            existing_experiments=measurement.document().get("experiments", ()),
            hierarchy=document,
            screen=document.get("screen"),
            max_candidates=policy.coalition_candidate_limit,
        )
        document["coalition_verification"] = coalitions
        document["verified_sets"] = coalitions["verified_sets"]

    # Layer 5 is included in Standard/Deep where budget remains.  It only
    # accepts experiments from the same Task 1 object and never promotes a
    # screen estimate to a preserving result.
    if policy.name in {"Standard", "Deep"} and consumed() < policy.pass_budget:
        _checkpoint(checkpoint, phase="preserving_subset_search", completed=consumed(), total=policy.pass_budget)
        document["preserving_subsets"] = run_preserving_subset_search(
            root_source_ids,
            lambda source_ids: measure(source_ids, phase="preserving_subset_search"),
            tolerance_nats=normalized["tolerance_nats"],
            passes_requested=policy.pass_budget,
            compute_level=policy,
            initial_passes_consumed=consumed(),
            existing_experiments=measurement.document().get("experiments", ()),
        )

    # Pick up all direct experiments added by later layers.  The one document
    # is the completed artifact emitted and persisted by the route.
    final = measurement.document()
    for key in ("hierarchy", "screen", "coalition_verification", "verified_sets", "preserving_subsets"):
        if key in document:
            final[key] = deepcopy(document[key])
    final["hierarchy"] = _hierarchy(root_source_ids, final["experiments"], source_order=source_order)
    final["budget"] = {
        "passes_requested": policy.pass_budget,
        "passes_consumed": int(measurement.document()["budget"]["passes_consumed"]),
        "passes_remaining": policy.pass_budget - int(measurement.document()["budget"]["passes_consumed"]),
        "state": "exhausted" if consumed() >= policy.pass_budget else "complete",
    }
    final["cache_identity"] = identity
    final["execution"] = {
        "method": normalized["method"],
        "execution_method": EXECUTION_METHOD,
        "intervention_operator": INTERVENTION_OPERATOR,
        "compute_policy": policy.name,
        "sampling_seed": normalized["sampling_seed"],
        "tolerance_nats": normalized["tolerance_nats"],
        "requested_root_source_ids": list(root_source_ids),
        "requested_neutralization_source_sets": [list(item) for item in neutralization_source_sets],
        "cache_binding": cache_binding(run, normalized),
    }
    schemas.validate(final, SCHEMA)
    return final


__all__ = [
    "ContextDependenceExecutionCancelled",
    "ContextDependenceExecutionError",
    "EXECUTION_METHOD",
    "SCHEMA",
    "cache_binding",
    "cache_identity",
    "cache_matches",
    "normalize_request",
    "run_context_dependence_execution",
]
